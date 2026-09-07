"""
Structured logging.

Structured, not free text, because these logs get queried later. "Why did the
system not take that trade on the 14th" is only answerable if rejections were
logged as records with fields, not as sentences.

Three things must always be logged, whatever else is:
- every signal generated, including the ones the risk manager rejected, and why
- every order submitted, filled, partially filled, or rejected by the broker
- every breaker trigger, with the regime in force at the time

## Four streams

Per the Phase 8 spec, events fan out to four rotating files by subject:

| File | What lands in it |
|---|---|
| `main.log` | everything, the full timeline |
| `trades.log` | signals, orders, fills, positions |
| `alerts.log` | breakers, halts, reconciliation, errors |
| `regime.log` | regime changes and per-bar classifications |

`main.log` is deliberately a superset rather than a leftovers bucket. When you
are reconstructing an incident you want one file with the true ordering; the
other three exist so that reading about trades does not mean scrolling past
every bar classification.

## Rotation

10MB per file, 30 days of history, whichever bound is hit first. Python ships a
size-based handler and a time-based handler and no handler that does both, so
`SizedTimedRotatingHandler` below is the small amount of glue required. Getting
this wrong is not cosmetic: a single unbounded `.log` on a machine that runs
this daily is how you find out about disk exhaustion from a broker error.

## Context on every record

The spec requires every entry to carry timestamp, regime, probability, equity,
positions and daily_pnl. Those are not properties of the event, they are
properties of the moment, so the engine pushes them once per bar with
`set_context()` and every record written afterwards carries them. Passing them
at each call site would mean six arguments repeated forty times, and the first
one anyone forgot would be the one that mattered.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

MAX_BYTES = 10 * 1024 * 1024      # 10MB per file, per spec
BACKUP_DAYS = 30                  # 30 days of history, per spec


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
    MODEL_RETRAINED = "model_retrained"
    FLICKER_EXCEEDED = "flicker_exceeded"
    LARGE_PNL = "large_pnl"
    ALERT_SENT = "alert_sent"
    ERROR = "error"


#: Which stream each event is filed under, on top of `main.log`.
STREAM_BY_EVENT: dict[EventType, str] = {
    EventType.SIGNAL_GENERATED: "trades",
    EventType.SIGNAL_REJECTED: "trades",
    EventType.ORDER_SUBMITTED: "trades",
    EventType.ORDER_FILLED: "trades",
    EventType.ORDER_REJECTED: "trades",
    EventType.POSITION_OPENED: "trades",
    EventType.POSITION_CLOSED: "trades",
    EventType.STOP_UPDATED: "trades",
    EventType.REGIME_CHANGE: "regime",
    EventType.BAR_PROCESSED: "regime",
    EventType.FLICKER_EXCEEDED: "regime",
    EventType.MODEL_RETRAINED: "regime",
    EventType.BREAKER_TRIGGERED: "alerts",
    EventType.SYSTEM_HALT: "alerts",
    EventType.RECONCILE_MISMATCH: "alerts",
    EventType.DATA_FEED_DOWN: "alerts",
    EventType.DATA_FEED_UP: "alerts",
    EventType.LARGE_PNL: "alerts",
    EventType.ALERT_SENT: "alerts",
    EventType.ERROR: "alerts",
}

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


class SizedTimedRotatingHandler(logging.handlers.TimedRotatingFileHandler):
    """Rotates on a daily boundary **or** at `maxBytes`, whichever comes first.

    The stdlib gives you one or the other. A purely time-based handler lets one
    bad day write an unbounded file; a purely size-based one silently discards
    history whenever a burst of errors churns through the backup count. The spec
    asks for both bounds, and both bounds are the right answer.

    `doRollover` is reimplemented rather than inherited because the stdlib's
    version begins:

        if os.path.exists(dfn):
            # Already rolled over.
            return

    The dated backup name is the same for every rotation on a given day, so once
    a daily rotation has happened, **every later size-triggered rotation that day
    becomes a no-op and the file grows without bound**. That is invisible in a
    short test and shows up as a full disk on a busy day in a long-running
    process, which is precisely what the 10MB bound exists to prevent. Same-day
    rotations get a `.1`, `.2` suffix here so each one lands somewhere new.
    """

    def __init__(self, filename, maxBytes: int = MAX_BYTES,
                 backupCount: int = BACKUP_DAYS, **kwargs: Any) -> None:
        super().__init__(filename, when="midnight", backupCount=backupCount, **kwargs)
        self.maxBytes = maxBytes

    def shouldRollover(self, record: logging.LogRecord) -> int:
        if super().shouldRollover(record):
            return 1
        if self.maxBytes <= 0:
            return 0
        if self.stream is None:
            self.stream = self._open()
        self.stream.seek(0, os.SEEK_END)
        # `+ 1` accounts for the newline the formatter will append.
        return 1 if self.stream.tell() + len(self.format(record)) + 1 >= self.maxBytes else 0

    def _target_name(self) -> str:
        """The dated backup name, made unique for repeat rotations in one day."""
        import time as _time

        stamp = _time.strftime(self.suffix, _time.localtime(self.rolloverAt - self.interval))
        base = self.rotation_filename(f"{self.baseFilename}.{stamp}")
        if not os.path.exists(base):
            return base
        index = 1
        while os.path.exists(f"{base}.{index}"):
            index += 1
        return f"{base}.{index}"

    def _prune(self) -> None:
        """Keep the newest `backupCount` backups.

        The parent's `getFilesToDelete` matches names against its date-suffix
        regex, which the `.1`/`.2` names above do not satisfy, so they would
        never be pruned. Sorting by mtime is both simpler and correct for a
        mixture of the two naming schemes.
        """
        if self.backupCount <= 0:
            return
        directory, prefix = os.path.split(self.baseFilename)
        try:
            candidates = [
                os.path.join(directory or ".", name)
                for name in os.listdir(directory or ".")
                if name.startswith(prefix + ".")
            ]
        except OSError:      # pragma: no cover - unreadable log directory
            return
        candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        for stale in candidates[self.backupCount:]:
            try:
                os.remove(stale)
            except OSError:  # pragma: no cover
                pass

    def doRollover(self) -> None:
        import time as _time

        if self.stream:
            self.stream.close()
            self.stream = None

        if os.path.exists(self.baseFilename):
            self.rotate(self.baseFilename, self._target_name())

        self._prune()
        if not self.delay:
            self.stream = self._open()

        now = int(_time.time())
        next_at = self.computeRollover(now)
        while next_at <= now:
            next_at += self.interval
        self.rolloverAt = next_at


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
    """Structured event log: four rotating JSONL streams plus a console sink."""

    STREAMS = ("main", "trades", "alerts", "regime")

    def __init__(self, name: str = "regime-trader", log_dir: Path = LOG_DIR) -> None:
        self.name = name
        self.log_dir = Path(log_dir)
        self._logger: logging.Logger | None = None
        self._lock = threading.Lock()
        self._handlers: dict[str, logging.Handler] = {}
        self._counts: dict[str, int] = {}
        self._context: dict[str, Any] = {}
        self._recent: list[dict[str, Any]] = []
        self._event_path: Path | None = None

    # -- setup --------------------------------------------------------------

    def setup(self, level: str = "INFO", session_id: str | None = None) -> TradingLogger:
        """Configure the four streams and the console. Idempotent."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False
        if not logger.handlers:
            console = logging.StreamHandler()
            console.setFormatter(
                logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
            )
            logger.addHandler(console)
        self._logger = logger

        if not self._handlers:
            for stream in self.STREAMS:
                handler = SizedTimedRotatingHandler(self.log_dir / f"{stream}.log")
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._handlers[stream] = handler

        # The JSONL event file the dashboard and the publisher read.
        stamp = session_id or datetime.now(UTC).strftime("%Y%m%d")
        self._event_path = self.log_dir / f"events-{stamp}.jsonl"
        return self

    @property
    def event_path(self) -> Path:
        if self._event_path is None:
            self.setup()
        return self._event_path  # type: ignore[return-value]

    def close(self) -> None:
        for handler in self._handlers.values():
            try:
                handler.close()
            except Exception:
                pass
        self._handlers.clear()

    # -- per-bar context ----------------------------------------------------

    def set_context(self, **fields: Any) -> None:
        """Fields stamped onto every subsequent record.

        The spec's required set is regime, probability, equity, positions and
        daily_pnl. They describe the moment, not the event, so they are pushed
        once per bar rather than passed at every call site.
        """
        with self._lock:
            self._context.update({k: _jsonable(v) for k, v in fields.items()})

    @property
    def context(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._context)

    def _log(self, level: int, message: str) -> None:
        if self._logger is None:
            self.setup()
        self._logger.log(level, message)  # type: ignore[union-attr]

    # -- the one write path -------------------------------------------------

    def log_event(self, event_type: EventType, message: str = "", **fields: Any) -> dict[str, Any]:
        """Write one structured event to its streams and one line to console.

        Returns the record, which makes it testable without reading the file
        back. Never raises: a disk that filled up must not stop the loop, so a
        failed write degrades to a console warning.
        """
        if self._logger is None:
            self.setup()

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event_type.value,
            "message": message,
            **self.context,
            **{k: _jsonable(v) for k, v in fields.items()},
        }
        line = json.dumps(record)

        with self._lock:
            self._counts[event_type.value] = self._counts.get(event_type.value, 0) + 1
            self._recent.append(record)
            del self._recent[:-500]

            targets = ["main"]
            stream = STREAM_BY_EVENT.get(event_type)
            if stream:
                targets.append(stream)

            for target in targets:
                handler = self._handlers.get(target)
                if handler is None:
                    continue
                try:
                    handler.emit(logging.LogRecord(
                        name=self.name, level=logging.INFO, pathname="", lineno=0,
                        msg=line, args=(), exc_info=None,
                    ))
                except Exception as exc:      # pragma: no cover - disk failure
                    self._log(logging.WARNING, f"log write to {target} failed: {exc}")

            try:
                with open(self.event_path, "a") as fh:
                    fh.write(line + "\n")
            except Exception as exc:          # pragma: no cover - disk failure
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
            "signal_regime": signal.regime_name,
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
            filled_quantity=getattr(order, "filled_quantity", getattr(order, "filled_qty", None)),
            status=getattr(order, "status", None),
            **fields,
        )

    def log_breaker(self, breaker: str, drawdown: float, equity: float,
                    regime: str, **fields: Any) -> dict[str, Any]:
        return self.log_event(
            EventType.BREAKER_TRIGGERED,
            f"BREAKER {breaker}: drawdown {drawdown:.2%}, equity ${equity:,.2f}, "
            f"regime at the time: {regime}",
            breaker=breaker, drawdown=drawdown, equity=equity,
            breaker_regime=regime, **fields,
        )

    def log_error(self, where: str, exc: BaseException, traceback_text: str = "") -> dict[str, Any]:
        return self.log_event(
            EventType.ERROR,
            f"{where}: {type(exc).__name__}: {exc}",
            where=where, error_type=type(exc).__name__, error=str(exc),
            traceback=traceback_text,
        )

    # -- reading back -------------------------------------------------------

    def read_events(self, path: Path | None = None) -> Iterator[dict[str, Any]]:
        """Stream events back off disk, skipping any line that failed to write."""
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

    def recent(self, limit: int = 50, events: set[str] | None = None) -> list[dict[str, Any]]:
        """In-memory tail. Used by the live dashboard, which refreshes every 5
        seconds and should not re-read a 10MB file to do it."""
        with self._lock:
            records = list(self._recent)
        if events:
            records = [r for r in records if r.get("event") in events]
        return records[-limit:]

    def get_trade_log(self, limit: int = 200, path: Path | None = None) -> list[dict[str, Any]]:
        """Signal and order history, newest last. The dashboard's signal feed."""
        wanted = {
            EventType.SIGNAL_GENERATED.value,
            EventType.SIGNAL_REJECTED.value,
            EventType.ORDER_SUBMITTED.value,
            EventType.ORDER_FILLED.value,
            EventType.ORDER_REJECTED.value,
        }
        in_memory = self.recent(limit=limit, events=wanted)
        if in_memory:
            return in_memory
        return [e for e in self.read_events(path) if e.get("event") in wanted][-limit:]

    def counts(self) -> dict[str, int]:
        """Per-event-type totals for this session. Used by the session summary."""
        with self._lock:
            return dict(self._counts)

    def stream_paths(self) -> dict[str, Path]:
        return {name: self.log_dir / f"{name}.log" for name in self.STREAMS}


# The Phase 1 stub named this StructuredLogger. Kept as an alias so nothing that
# imported the old name breaks.
StructuredLogger = TradingLogger
