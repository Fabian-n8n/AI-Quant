"""
Track open positions and P&L, reconcile against Alpaca, and consume fills.

Phase 6.

RECONCILIATION MATTERS MORE THAN IT SOUNDS
------------------------------------------
The system's idea of what it owns and Alpaca's will drift: partial fills, manual
intervention in the web dashboard, rejected orders, corporate actions. When they
disagree the broker is right, and `sync()` adopts what the broker reports rather
than arguing with it.

Two drift directions, handled differently:

- **Untracked**: a position Alpaca has and we do not. Adopted, but flagged,
  because its stop and regime context are unknown and the risk layer needs both.
- **Stale**: a position we have and Alpaca does not. Removed. It was closed by a
  stop, a manual action, or a liquidation, and continuing to size against it
  would be sizing against a fiction.

THREAD SAFETY
-------------
Fills arrive on a WebSocket thread while the main loop reads positions. Every
mutation takes `threading.Lock`. Without it, a fill landing mid-iteration
produces a `RuntimeError: dictionary changed size during iteration` at the worst
possible time, or worse, a torn read of a position being updated.

THE BRIDGE TO PHASE 5
---------------------
`to_portfolio_state()` converts tracked positions into the `PortfolioState` that
`RiskManager.validate_signal` expects. That is the join between the broker layer
and the risk layer, and it is the only place the conversion happens.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from broker.alpaca_client import AlpacaClient, Position

logger = logging.getLogger(__name__)


@dataclass
class TrackedPosition:
    """Local view of a position, with the context Alpaca does not store.

    Alpaca knows the shares and the average price. It does not know why the
    position was opened, where the stop sits, or what regime was in force at
    entry. That context is what makes the trade log worth reading later, and
    `regime_at_entry` versus `regime_current` is how you find out whether the
    regime layer is adding anything.
    """
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    regime_at_entry: str = "unknown"
    confidence_at_entry: float = 0.0
    regime_current: str = "unknown"
    holding_periods: int = 0
    trade_id: str | None = None
    rationale: str = ""
    strategy_name: str = ""
    adopted: bool = False        # found on the broker, not opened by us

    @property
    def market_value(self) -> float:
        return self.quantity * (self.current_price or self.entry_price)

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price

    @property
    def unrealised_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealised_pnl_pct(self) -> float:
        return (self.unrealised_pnl / self.cost_basis) if self.cost_basis else 0.0

    @property
    def has_stop(self) -> bool:
        return self.stop_loss is not None

    @property
    def regime_changed(self) -> bool:
        """True when the regime moved since entry.

        Not an exit signal on its own. It is the flag that makes "this position
        was opened for a reason that no longer holds" visible on the dashboard.
        """
        return (
            self.regime_current != "unknown"
            and self.regime_at_entry != "unknown"
            and self.regime_current != self.regime_at_entry
        )

    @property
    def distance_to_stop_pct(self) -> float:
        if not self.stop_loss or not self.current_price:
            return 0.0
        return (self.current_price - self.stop_loss) / self.current_price


class PositionTracker:
    """Positions, fills and reconciliation. Thread-safe."""

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client
        self.positions: dict[str, TrackedPosition] = {}
        self._lock = threading.Lock()
        self._fill_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._stream = None
        self._stream_thread: threading.Thread | None = None
        self._stop_stream = threading.Event()

    # -- reconciliation -----------------------------------------------------

    def sync(self) -> dict[str, list[str]]:
        """Reconcile tracked positions against the broker.

        Returns `{"adopted": [...], "removed": [...], "adjusted": [...]}`.

        A non-empty result at any time other than startup is a stop-and-alert
        condition: it means something moved the account without going through
        this system, and continuing to trade on a stale view is how a small
        problem becomes an expensive one.
        """
        actual = {p.symbol: p for p in self.client.get_positions()}
        report: dict[str, list[str]] = {"adopted": [], "removed": [], "adjusted": []}

        with self._lock:
            for symbol, broker_position in actual.items():
                tracked = self.positions.get(symbol)
                if tracked is None:
                    self.positions[symbol] = self._adopt(broker_position)
                    report["adopted"].append(symbol)
                elif abs(tracked.quantity - broker_position.quantity) > 1e-9:
                    logger.warning(
                        "%s quantity drift: tracked %g, broker %g. Broker wins.",
                        symbol, tracked.quantity, broker_position.quantity,
                    )
                    tracked.quantity = broker_position.quantity
                    tracked.entry_price = broker_position.average_entry_price
                    report["adjusted"].append(symbol)
                    tracked.current_price = broker_position.current_price
                else:
                    tracked.current_price = broker_position.current_price

            for symbol in [s for s in self.positions if s not in actual]:
                logger.warning("%s tracked but not held at broker. Removing.", symbol)
                del self.positions[symbol]
                report["removed"].append(symbol)

        if any(report.values()):
            logger.warning("Reconciliation drift: %s", {k: v for k, v in report.items() if v})
        return report

    def _adopt(self, position: Position) -> TrackedPosition:
        """Adopt a broker position we were not tracking.

        Flagged `adopted=True` with no stop, because we do not know where its
        stop is or why it exists. `positions_without_stops()` will surface it,
        which is the correct outcome: an unprotected position should be loud.
        """
        logger.warning(
            "Adopting untracked position %s x%g. No stop or regime context known.",
            position.symbol, position.quantity,
        )
        return TrackedPosition(
            symbol=position.symbol,
            quantity=position.quantity,
            entry_price=position.average_entry_price,
            entry_time=datetime.now(UTC),
            current_price=position.current_price,
            adopted=True,
            rationale="adopted during reconciliation; origin unknown",
        )

    # -- fills --------------------------------------------------------------

    def register_fill(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        trade_id: str | None = None,
        stop_loss: float | None = None,
        regime: str = "unknown",
        confidence: float = 0.0,
        rationale: str = "",
        strategy_name: str = "",
    ) -> TrackedPosition | None:
        """Apply a fill: open, average up, partially close, or close out.

        Averaging up recomputes the weighted entry price. Keeping the original
        entry would make every subsequent P&L figure and stop distance wrong.

        Returns the resulting position, or None when the fill closed it.
        """
        with self._lock:
            existing = self.positions.get(symbol)
            direction = 1.0 if side.lower() == "buy" else -1.0
            delta = direction * abs(quantity)

            if existing is None:
                if delta <= 0:
                    logger.warning("%s: sell fill with no tracked position, ignoring", symbol)
                    return None
                position = TrackedPosition(
                    symbol=symbol, quantity=delta, entry_price=price,
                    entry_time=datetime.now(UTC), current_price=price,
                    stop_loss=stop_loss, regime_at_entry=regime, regime_current=regime,
                    confidence_at_entry=confidence, trade_id=trade_id,
                    rationale=rationale, strategy_name=strategy_name,
                )
                self.positions[symbol] = position
                logger.info("Opened %s x%g @ %.2f", symbol, delta, price)
                result = position
            else:
                new_quantity = existing.quantity + delta
                if abs(new_quantity) < 1e-9:
                    logger.info("Closed %s (was x%g)", symbol, existing.quantity)
                    del self.positions[symbol]
                    result = None
                else:
                    if delta > 0:
                        # Weighted average. Keeping the original entry would make
                        # every later P&L and stop-distance figure wrong.
                        total_cost = existing.cost_basis + delta * price
                        existing.entry_price = total_cost / new_quantity
                        logger.info("Averaged up %s to x%g @ %.2f", symbol,
                                    new_quantity, existing.entry_price)
                    else:
                        logger.info("Reduced %s to x%g", symbol, new_quantity)
                    existing.quantity = new_quantity
                    existing.current_price = price
                    if stop_loss is not None:
                        existing.stop_loss = stop_loss
                    result = existing

        for callback in list(self._fill_callbacks):
            try:
                callback({"symbol": symbol, "quantity": quantity, "price": price,
                          "side": side, "trade_id": trade_id})
            except Exception as exc:
                logger.error("Fill callback failed: %s", exc)
        return result

    def on_fill(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a fill listener.

        The orchestrator uses this to refresh `PortfolioState` and advance the
        circuit breakers the moment a fill lands, rather than waiting for the
        next bar. A breaker that only updates on bar close is blind to whatever
        happens between bars.
        """
        self._fill_callbacks.append(callback)

    # -- websocket ----------------------------------------------------------

    def start_websocket(self, daemon: bool = True) -> threading.Thread:
        """Subscribe to Alpaca's trade updates on a background thread.

        Fills arrive within milliseconds rather than at the next poll. The
        thread reconnects on failure; if it dies permanently the orchestrator
        still has `sync()` as a slower fallback, which is why the loop must call
        `sync()` regardless of whether the socket is up.
        """
        from alpaca.trading.stream import TradingStream

        def run() -> None:
            backoff = 1.0
            while not self._stop_stream.is_set():
                try:
                    stream = TradingStream(
                        self.client._api_key, self.client._secret_key, paper=self.client.paper
                    )
                    self._stream = stream
                    stream.subscribe_trade_updates(self._handle_trade_update)
                    logger.info("Trade update stream connected")
                    backoff = 1.0
                    stream.run()
                except Exception as exc:
                    if self._stop_stream.is_set():
                        break
                    logger.warning("Trade stream dropped (%s), reconnecting in %.0fs", exc, backoff)
                    self._stop_stream.wait(backoff)
                    backoff = min(backoff * 2, 60.0)

        thread = threading.Thread(target=run, name="alpaca-trade-stream", daemon=daemon)
        thread.start()
        self._stream_thread = thread
        return thread

    def stop_websocket(self) -> None:
        self._stop_stream.set()
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass

    async def _handle_trade_update(self, data) -> None:
        """Alpaca trade update callback. Only fills mutate state."""
        try:
            event = str(getattr(data, "event", ""))
            order = getattr(data, "order", None)
            if order is None or event not in ("fill", "partial_fill"):
                return
            self.register_fill(
                symbol=order.symbol,
                quantity=float(getattr(data, "qty", 0) or 0),
                price=float(getattr(data, "price", 0) or 0),
                side=str(getattr(order.side, "value", order.side)),
                trade_id=getattr(order, "client_order_id", None),
            )
            if event == "partial_fill":
                logger.warning("%s partial fill: stop must cover the filled quantity only",
                               order.symbol)
        except Exception as exc:
            logger.error("Failed to handle trade update: %s", exc)

    # -- state --------------------------------------------------------------

    def update_prices(self, prices: dict[str, float]) -> None:
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self.positions:
                    self.positions[symbol].current_price = price

    def update_stop(self, symbol: str, stop_loss: float) -> None:
        with self._lock:
            if symbol in self.positions:
                self.positions[symbol].stop_loss = stop_loss

    def update_regime(self, regime: str) -> None:
        """Stamp the current regime onto every position.

        `regime_at_entry` is left alone. The pair is what makes
        `regime_changed` meaningful.
        """
        with self._lock:
            for position in self.positions.values():
                position.regime_current = regime

    def increment_holding_periods(self) -> None:
        """Advance the holding-period counter. Called once per bar close."""
        with self._lock:
            for position in self.positions.values():
                position.holding_periods += 1

    def get_open_positions(self) -> dict[str, TrackedPosition]:
        with self._lock:
            return dict(self.positions)

    def positions_without_stops(self) -> list[str]:
        """Positions missing a protective stop.

        Should always be empty. A non-empty list is a stop-and-alert condition,
        not a warning to log and move past: every one of them is an unbounded
        loss.
        """
        with self._lock:
            return [s for s, p in self.positions.items() if not p.has_stop]

    def total_exposure(self, equity: float) -> float:
        if equity <= 0:
            return 0.0
        with self._lock:
            return sum(abs(p.market_value) for p in self.positions.values()) / equity

    def unrealised_pnl(self) -> float:
        with self._lock:
            return sum(p.unrealised_pnl for p in self.positions.values())

    # -- bridge to the risk layer -------------------------------------------

    def to_portfolio_state(
        self,
        account=None,
        peak_equity: float = 0.0,
        day_start_equity: float = 0.0,
        week_start_equity: float = 0.0,
        daily_trades: int = 0,
        regime: str = "unknown",
        regime_confirmed: bool = True,
        regime_confidence: float = 1.0,
        flicker_rate: int = 0,
    ):
        """Build the `PortfolioState` that `RiskManager.validate_signal` expects.

        The single join between the broker layer and the risk layer. Positions
        are handed over as plain dicts so the risk manager never imports a
        broker type, which is what keeps it broker-agnostic.
        """
        from core.risk_manager import PortfolioState

        account = account or self.client.get_account()
        with self._lock:
            positions = {
                symbol: {
                    "market_value": position.market_value,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "stop_loss": position.stop_loss,
                    "unrealised_pnl": position.unrealised_pnl,
                }
                for symbol, position in self.positions.items()
            }

        return PortfolioState(
            equity=account.equity,
            cash=account.cash,
            buying_power=account.buying_power,
            positions=positions,
            peak_equity=peak_equity or account.equity,
            day_start_equity=day_start_equity or account.last_equity or account.equity,
            week_start_equity=week_start_equity or account.equity,
            daily_trades=daily_trades,
            timestamp=datetime.now(UTC),
            regime=regime,
            regime_confirmed=regime_confirmed,
            regime_confidence=regime_confidence,
            flicker_rate=flicker_rate,
        )

    def to_frame(self) -> pd.DataFrame:
        """Positions as a table, for the dashboard and the logs."""
        with self._lock:
            if not self.positions:
                return pd.DataFrame()
            return pd.DataFrame([
                {
                    "symbol": p.symbol, "quantity": p.quantity,
                    "entry_price": p.entry_price, "current_price": p.current_price,
                    "market_value": p.market_value, "unrealised_pnl": p.unrealised_pnl,
                    "unrealised_pnl_pct": p.unrealised_pnl_pct, "stop_loss": p.stop_loss,
                    "distance_to_stop_pct": p.distance_to_stop_pct,
                    "holding_periods": p.holding_periods,
                    "regime_at_entry": p.regime_at_entry, "regime_current": p.regime_current,
                    "regime_changed": p.regime_changed, "adopted": p.adopted,
                }
                for p in self.positions.values()
            ])
