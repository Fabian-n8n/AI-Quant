"""
Structured logging.

Structured, not free text, because these logs get queried later. "Why did the
system not take that trade on the 14th" is only answerable if rejections were
logged as records with fields, not as sentences.

Three things must always be logged, whatever else is:
- every signal generated, including the ones the risk manager rejected, and why
- every order submitted, filled, partially filled, or rejected by the broker
- every breaker trigger, with the regime in force at the time

Two sinks, one call. JSON lines to disk so the dashboard and any later analysis
can read it; a readable line to console so you can watch it run. Writing both
from one `log_event` is what keeps them from drifting: a console message with no
matching record on disk is the thing that makes an incident unreconstructable.

Implemented in Phase 7. Phase 1 shipped this as a stub; the main loop is the
first caller, so it lands with the loop rather than with the dashboard.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class EventType(str, Enum):
    REGIME_CHANGE = "regime_change"
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_REJECTED = "signal_rejected"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    BREAKER_TRIGGERED = "breaker_triggered"
    RECONCILE_MISMATCH = "reconcile_mismatch"
    STOP_UPDATED = "stop_updated"
    BAR_PROCESSED = "bar_processed"
    SYSTEM_START = "system_start"
    SYSTEM_HALT = "system_halt"
    SYSTEM_SHUTDOWN = "system_shutdown"
    DATA_FEED_DOWN = "data_feed_down"
    DATA_FEED_UP = "data_feed_up"
    ERROR = "error"


# Events that should reach the console at WARNING or above. Everything else is
# INFO on console and always present on disk. Chosen so a quiet console means
# nothing needing attention happened, not that logging broke.
_LOUD = {
    EventType.BREAKER_TRIGGERED,
    EventType.RECONCILE_MISMATCH,
    EventType.SYSTEM_HALT,
    EventType.DATA_FEED_DOWN,
    EventType.ORDER_REJECTED,
    EventType.ERROR,
}


def _jsonable(value: Any) -> Any:
    """Coerce anything the trading path produces into something json can write.

    Enums become their value, datetimes and Timestamps their ISO string, numpy
    scalars their Python equivalent. Falls back to repr rather than raising: a
    log write must never be the thing that kills the trading loop.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


class TradingLogger:
    """Structured event log with a JSONL file sink and a console sink."""

    def __init__(self, name: str = "regime-trader", log_dir: Path = LOG_DIR) -> None:
        self.name = name
        self.log_dir = Path(log_dir)
        self._logger: Optional[logging.Logger] = None
        self._lock = threading.Lock()
        self._event_path: Optional[Path] = None
        self._counts: dict[str, int] = {}

    # -- setup --------------------------------------------------------------

    def setup(self, level: str = "INFO", session_id: Optional[str] = None) -> "TradingLogger":
        """Configure handlers. Idempotent: calling twice does not double every line."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = session_id or datetime.now(timezone.utc).strftime("%Y%m%d")
        self._event_path = self.log_dir / f"events-{stamp}.jsonl"

        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False

        if not logger.handlers:
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                                   datefmt="%H:%M:%S"))
            logger.addHandler(console)

            text_file = logging.FileHandler(self.log_dir / f"trader-{stamp}.log")
            text_file.setFormatter(
                logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
            )
            logger.addHandler(text_file)

        self._logger = logger
        return self

    @property
    def event_path(self) -> Path:
        if self._event_path is None:
            self.setup()
        return self._event_path  # type: ignore[return-value]

    def _log(self, level: int, message: str) -> None:
        if self._logger is None:
            self.setup()
        self._logger.log(level, message)  # type: ignore[union-attr]

    # -- the one write path -------------------------------------------------

    def log_event(self, event_type: EventType, message: str = "", **fields: Any) -> dict[str, Any]:
        """Write one structured event to disk and one readable line to console.

        Returns the record, which makes it testable without reading the file
        back. Never raises: a disk that filled up must not stop the loop, so a
        failed write degrades to a console warning.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type.value,
            "message": message,
            **{k: _jsonable(v) for k, v in fields.items()},
        }

        with self._lock:
            self._counts[event_type.value] = self._counts.get(event_type.value, 0) + 1
            try:
                self.event_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.event_path, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
            except Exception as exc:  # pragma: no cover - disk failure path
                self._log(logging.WARNING, f"event log write failed: {exc}")

        level = logging.WARNING if event_type in _LOUD else logging.INFO
        if message:
            self._log(level, message)
        return record

    # -- typed helpers ------------------------------------------------------

    def log_regime_change(self, previous: str, current: str, confidence: float,
                          **fields: Any) -> dict[str, Any]:
        return self.log_event(
            EventType.REGIME_CHANGE,
            f"Regime {previous} -> {current} (p={confidence:.2f})",
            previous=previous, current=current, confidence=confidence, **fields,
        )

    def log_signal(self, signal, decision) -> dict[str, Any]:
        """Log a signal and its risk verdict together.

        Rejections are logged as prominently as approvals. A log of only the
        trades that happened cannot tell you the system stopped trading three
        weeks ago because a breaker never reset.
        """
        approved = getattr(decision, "approved", False)
        shares = getattr(decision, "modified_signal", {}).get("shares", 0)
        notional = getattr(decision, "approved_notional", 0.0)

        fields = {
            "symbol": signal.symbol,
            "direction": signal.direction,
            "strategy": signal.strategy_name,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "requested_allocation": signal.position_size_pct,
            "leverage": signal.leverage,
            "regime": getattr(signal, "regime", None),
            "confidence": getattr(signal, "confidence", None),
            "approved": approved,
            "action": getattr(decision, "action", None),
            "shares": shares,
            "notional": notional,
            "modifications": list(getattr(decision, "modifications", [])),
            "rejection_reason": getattr(decision, "rejection_reason", None),
            "reason": getattr(decision, "reason", ""),
        }

        if approved:
            return self.log_event(
                EventType.SIGNAL_GENERATED,
                f"{signal.symbol}: approved {shares:g} shares (${notional:,.2f})"
                + (f" [{'; '.join(decision.modifications)}]" if decision.modifications else ""),
                **fields,
            )
        return self.log_event(
            EventType.SIGNAL_REJECTED,
            f"{signal.symbol}: rejected, {decision.reason}",
            **fields,
        )

    def log_order(self, order, event_type: EventType = EventType.ORDER_SUBMITTED,
                  **fields: Any) -> dict[str, Any]:
        symbol = getattr(order, "symbol", "?")
        # Order calls it `quantity`, TradeRecord calls it `approved_qty`. Both
        # reach this method, so try each rather than silently logging a zero.
        quantity = next(
            (getattr(order, name) for name in ("quantity", "approved_qty", "requested_qty")
             if getattr(order, name, None) is not None),
            0,
        )
        return self.log_event(
            event_type,
            f"{symbol}: {event_type.value.replace('_', ' ')} x{quantity:g}",
            symbol=symbol,
            quantity=quantity,
            order_id=getattr(order, "order_id", None),
            side=getattr(order, "side", None),
            order_type=getattr(order, "order_type", None),
            limit_price=getattr(order, "limit_price", None),
            filled_quantity=getattr(order, "filled_quantity", None),
            filled_avg_price=getattr(order, "filled_avg_price", None),
            status=getattr(order, "status", None),
            **fields,
        )

    def log_breaker(self, breaker: str, drawdown: float, equity: float,
                    regime: str, **fields: Any) -> dict[str, Any]:
        return self.log_event(
            EventType.BREAKER_TRIGGERED,
            f"BREAKER {breaker}: drawdown {drawdown:.2%}, equity ${equity:,.2f}, "
            f"regime at the time: {regime}",
            breaker=breaker, drawdown=drawdown, equity=equity, regime=regime, **fields,
        )

    def log_error(self, where: str, exc: BaseException, traceback_text: str = "") -> dict[str, Any]:
        return self.log_event(
            EventType.ERROR,
            f"{where}: {type(exc).__name__}: {exc}",
            where=where, error_type=type(exc).__name__, error=str(exc),
            traceback=traceback_text,
        )

    # -- reading back -------------------------------------------------------

    def read_events(self, path: Optional[Path] = None) -> Iterator[dict[str, Any]]:
        """Stream events back off disk, skipping any line that failed to write cleanly."""
        target = Path(path) if path else self.event_path
        if not target.exists():
            return
        with open(target) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def get_trade_log(self, limit: int = 200, path: Optional[Path] = None) -> list[dict[str, Any]]:
        """Signal and order history, newest last. The dashboard's signal feed."""
        wanted = {
            EventType.SIGNAL_GENERATED.value,
            EventType.SIGNAL_REJECTED.value,
            EventType.ORDER_SUBMITTED.value,
            EventType.ORDER_FILLED.value,
            EventType.ORDER_REJECTED.value,
        }
        events = [e for e in self.read_events(path) if e.get("event") in wanted]
        return events[-limit:]

    def counts(self) -> dict[str, int]:
        """Per-event-type totals for this session. Used by the session summary."""
        with self._lock:
            return dict(self._counts)


# The Phase 1 stub named this StructuredLogger. Kept as an alias so nothing that
# imported the old name breaks.
StructuredLogger = TradingLogger
