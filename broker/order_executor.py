"""
Order placement, modification and cancellation.

Phase 6.

Sits between the risk manager and the Alpaca client. Nothing else in the system
calls `submit_order` directly, which makes this the single choke point where the
"no order without a stop" rule is enforced a second time, independently of the
risk manager. One rule, two places it is checked, because it is the rule that
cannot be allowed to fail quietly.

LIMIT ORDERS BY DEFAULT
-----------------------
Market orders on a daily-bar swing system are unnecessary risk. The system has
until the next bar to get filled, so it posts a limit 0.1% through the current
price, waits 30 seconds, and cancels if unfilled. Retrying at market is opt-in
per call rather than automatic: a limit that will not fill is usually telling you
something about liquidity, and converting it to a market order discards that
information at exactly the wrong moment.

TRADE IDS
---------
Every order carries a `trade_id` that links signal to risk decision to order to
fill. Without it, reconstructing why a position exists means correlating three
logs by timestamp and hoping. Alpaca's `client_order_id` carries it, so the link
survives on the broker's side too.

WHAT IS NEVER RETRIED
---------------------
Order submission. A network timeout does not tell you whether the order reached
the exchange, and a blind retry is how you end up holding two copies of the same
position. Reads retry with backoff; writes fail loudly and leave the decision to
a human or to the orchestrator's error path.

IDEMPOTENCY
-----------
Two independent mechanisms, because they fail in different places.

`_equivalent_open_order` asks the broker whether an order for this symbol, side
and roughly this price is already resting, and skips if so. This catches the
common case: a loop that comes round again while the previous order is still
working. Every open order also consumes buying power, so resubmitting does not
merely duplicate the intent, it starves the rest of the universe.

The deterministic `client_order_id` (`rt-SYMBOL-SIDE-YYYYMMDD`) catches what the
broker query cannot. A restarted process has no memory, and an order that
already filled is no longer open, so nothing local or in the open-order list
would stop a second entry on the same signal. Alpaca rejects a repeated
client_order_id itself, which makes the broker the arbiter rather than our
in-memory state. One signal on one bar date can produce exactly one order,
across restarts, forever.

`risk_manager.is_duplicate` stays as the cheap in-process check before either.

THE PRICE SANITY GUARD
----------------------
Every limit price is checked against the live quote before submission and
refused above `max_price_deviation`. A limit far below the market does not
error, it just never fills, so the failure mode is a silent one: orders that
look placed, an account that never moves, and nothing in the logs to say why.
The guard converts that into a loud rejection naming both prices.

It is deliberately overridable per call. The integration suite prices 20% below
the touch so its order rests instead of filling, which is the guard's exact
trigger condition. A guard that the test disables globally to keep working
protects nothing, so the opt-out is one argument on one call site.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from broker.alpaca_client import AlpacaClient, Order, OrderSide, OrderStatus, OrderType
from core.regime_strategies import Direction, Signal
from core.risk_manager import RiskDecision

logger = logging.getLogger(__name__)

DEFAULT_LIMIT_OFFSET = 0.001      # 0.1% through the touch
DEFAULT_FILL_TIMEOUT = 30.0       # seconds before an unfilled limit is cancelled
DEFAULT_MAX_DEVIATION = 0.05      # reject a limit more than 5% off the market
DEFAULT_ORDER_PREFIX = "rt-"      # every real order. Tests override it, so the
                                  # broker's order book distinguishes the two.
DEFAULT_PRICE_TOLERANCE = 0.005   # "roughly the same price" for dedupe: 0.5%


@dataclass
class TradeRecord:
    """One trade's full provenance: signal -> risk decision -> order -> fill.

    The object that makes "why do I own this" answerable six months later
    without correlating three logs by timestamp.
    """
    trade_id: str
    symbol: str
    side: OrderSide
    requested_qty: float
    approved_qty: float
    signal_reasoning: str
    risk_modifications: list[str]
    regime: str
    regime_confidence: float
    stop_loss: float | None
    take_profit: float | None
    order_id: str | None = None
    client_order_id: str | None = None   # the idempotency key, as sent
    stop_order_id: str | None = None
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    fill_price: float | None = None
    filled_qty: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    notes: list[str] = field(default_factory=list)
    # Set when the idempotency layer declined to submit. A skip is not an
    # error, so it comes back as a record rather than an exception, and callers
    # check this instead of counting it as a placed order.
    skipped_reason: str | None = None

    @property
    def was_submitted(self) -> bool:
        return self.skipped_reason is None


class OrderExecutionError(RuntimeError):
    """Raised when an order cannot be placed. Never retried automatically."""


def _is_duplicate_id_error(exc: Exception) -> bool:
    """Is this the broker rejecting a repeated client_order_id?

    Matched on the message because alpaca-py raises a generic APIError for
    this, with no distinguishing type or code we can rely on. Narrow patterns
    only: mistaking a real submission failure for a duplicate would mean
    silently not trading and reporting it as fine.
    """
    text = str(exc).lower()
    return "client_order_id" in text and (
        "already" in text or "duplicate" in text or "exists" in text
    )




class OrderExecutor:
    """Places orders that the risk manager has already approved."""

    def __init__(
        self,
        client: AlpacaClient,
        limit_offset: float = DEFAULT_LIMIT_OFFSET,
        fill_timeout: float = DEFAULT_FILL_TIMEOUT,
        max_price_deviation: float = DEFAULT_MAX_DEVIATION,
        order_id_prefix: str = DEFAULT_ORDER_PREFIX,
        deterministic_ids: bool = True,
        price_tolerance: float = DEFAULT_PRICE_TOLERANCE,
    ) -> None:
        self.client = client
        self.limit_offset = limit_offset
        self.fill_timeout = fill_timeout
        self.max_price_deviation = max_price_deviation
        self.order_id_prefix = order_id_prefix
        self.deterministic_ids = deterministic_ids
        self.price_tolerance = price_tolerance
        self.trades: dict[str, TradeRecord] = {}
        self.skipped: list[str] = []

    # -- submission ---------------------------------------------------------

    def submit_order(
        self,
        signal: Signal,
        decision: RiskDecision,
        order_type: OrderType = OrderType.LIMIT,
        reference_price: float | None = None,
        wait_for_fill: bool = False,
        retry_at_market: bool = False,
        allow_price_deviation: bool = False,
    ) -> TradeRecord:
        """Place an order the risk manager approved.

        Refuses if the decision was not approved, or if the signal has no stop,
        even though the risk manager checks both. Two independent checks on the
        one rule that cannot be allowed to fail.

        `reference_price` overrides the live quote, which matters when the market
        is closed: Alpaca's IEX feed returns a zero ask outside hours, and
        pricing a limit off zero would submit a nonsense order.

        `allow_price_deviation` skips the sanity guard. Only the integration
        suite passes it, because it prices deliberately far from the market so
        the order rests. Nothing in the trading path should ever set it.
        """
        if not decision.approved:
            raise OrderExecutionError(
                f"{signal.symbol}: risk manager rejected this signal "
                f"({decision.rejection_reason}). It may not be submitted."
            )
        if signal.stop_loss is None:
            raise OrderExecutionError(
                f"{signal.symbol}: no stop loss. The system does not place "
                f"unprotected orders."
            )

        quantity = float(decision.modified_signal.get("shares", 0))
        if quantity <= 0:
            raise OrderExecutionError(f"{signal.symbol}: approved quantity is zero")

        side = OrderSide.BUY if signal.direction is Direction.LONG else OrderSide.SELL
        trade = self._new_trade_record(signal, decision, side, quantity)

        price = reference_price if reference_price is not None else self._reference_price(signal)
        limit = round(self._limit_price(price, side), 2)
        if order_type is OrderType.LIMIT and not allow_price_deviation:
            self._assert_price_sane(signal.symbol, limit, side)

        existing = self._equivalent_open_order(signal.symbol, side, limit)
        if existing is not None:
            return self._skip(
                trade,
                f"an equivalent {side.value} order is already open "
                f"({existing.order_id[:8]}, {existing.quantity:g} @ "
                f"{existing.limit_price if existing.limit_price else 'market'})",
                existing,
            )

        client_order_id = self._client_order_id(trade, signal)
        trade.client_order_id = client_order_id
        request = self._build_request(
            signal.symbol, quantity, side, order_type, price,
            client_order_id=client_order_id,
        )

        try:
            raw = self.client.trading_client.submit_order(request)
        except Exception as exc:
            if _is_duplicate_id_error(exc):
                # The broker refusing our client_order_id is the idempotency
                # key doing its job, not a fault. This is the path a restarted
                # process takes when the original order already filled and so
                # no longer appears in the open-order list.
                return self._skip(
                    trade,
                    f"broker already has an order with id {client_order_id}",
                )
            trade.status = OrderStatus.REJECTED
            trade.notes.append(f"submission failed: {exc}")
            self.trades[trade.trade_id] = trade
            raise OrderExecutionError(f"{signal.symbol}: {exc}") from exc

        order = self.client.to_order(raw)
        trade.order_id = order.order_id
        trade.submitted_at = order.submitted_at or datetime.now(UTC)
        trade.status = order.status
        self.trades[trade.trade_id] = trade

        logger.info(
            "Submitted %s %s x%g @ %s (trade %s, order %s)",
            side.value, signal.symbol, quantity,
            f"{price:.2f} limit" if order_type is OrderType.LIMIT else "market",
            trade.trade_id[:8], order.order_id[:8],
        )

        if wait_for_fill:
            self._await_fill(trade, retry_at_market, signal, side, quantity)
        return trade

    def submit_bracket_order(
        self,
        signal: Signal,
        decision: RiskDecision,
        reference_price: float | None = None,
    ) -> TradeRecord:
        """Entry, stop and take-profit as one Alpaca bracket.

        Preferred over submitting the entry and then attaching a stop, because
        the broker links the legs: the stop exists the instant the entry fills,
        with no window in which the position is naked. That window is small but
        it is exactly when a gap would hurt.

        Requires a take_profit. Alpaca's bracket class demands both legs; use
        `submit_order` plus `place_stop` when there is no profit target.
        """
        if not decision.approved:
            raise OrderExecutionError(f"{signal.symbol}: not approved by risk manager")
        if signal.stop_loss is None:
            raise OrderExecutionError(f"{signal.symbol}: no stop loss")
        if signal.take_profit is None:
            raise OrderExecutionError(
                f"{signal.symbol}: bracket orders need a take_profit. Use "
                f"submit_order() then place_stop() for a stop-only position."
            )

        from alpaca.trading.enums import OrderClass, TimeInForce
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.requests import (
            LimitOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        quantity = float(decision.modified_signal.get("shares", 0))
        side = OrderSide.BUY if signal.direction is Direction.LONG else OrderSide.SELL
        trade = self._new_trade_record(signal, decision, side, quantity)
        trade.client_order_id = self._client_order_id(trade, signal)
        price = reference_price if reference_price is not None else self._reference_price(signal)
        limit = round(self._limit_price(price, side), 2)
        self._assert_price_sane(signal.symbol, limit, side)

        existing = self._equivalent_open_order(signal.symbol, side, limit)
        if existing is not None:
            return self._skip(
                trade,
                f"an equivalent {side.value} order is already open "
                f"({existing.order_id[:8]})",
                existing,
            )

        request = LimitOrderRequest(
            symbol=signal.symbol,
            qty=quantity,
            side=AlpacaSide.BUY if side is OrderSide.BUY else AlpacaSide.SELL,
            time_in_force=TimeInForce.GTC,      # bracket legs must outlive the session
            order_class=OrderClass.BRACKET,
            limit_price=limit,
            client_order_id=trade.client_order_id,
            take_profit=TakeProfitRequest(limit_price=round(signal.take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(signal.stop_loss, 2)),
        )

        try:
            raw = self.client.trading_client.submit_order(request)
        except Exception as exc:
            trade.status = OrderStatus.REJECTED
            trade.notes.append(f"bracket submission failed: {exc}")
            self.trades[trade.trade_id] = trade
            raise OrderExecutionError(f"{signal.symbol}: {exc}") from exc

        order = self.client.to_order(raw)
        trade.order_id = order.order_id
        trade.submitted_at = order.submitted_at or datetime.now(UTC)
        trade.status = order.status
        trade.notes.append(f"bracket: stop {signal.stop_loss:.2f}, target {signal.take_profit:.2f}")
        self.trades[trade.trade_id] = trade
        logger.info("Submitted bracket %s x%g (trade %s)", signal.symbol, quantity, trade.trade_id[:8])
        return trade

    def place_stop(self, symbol: str, quantity: float, stop_price: float,
                   trade_id: str | None = None) -> Order:
        """Attach a protective stop to a filled position.

        Placed immediately after the entry fills. A position that exists without
        its stop, even for one cycle, is an unbounded loss.

        Sized to the quantity actually filled, not the quantity requested. A
        partial fill with a full-size stop leaves the excess as a naked short
        the moment it triggers.
        """
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import StopOrderRequest

        request = StopOrderRequest(
            symbol=symbol, qty=quantity, side=AlpacaSide.SELL,
            time_in_force=TimeInForce.GTC, stop_price=round(stop_price, 2),
        )
        raw = self.client.trading_client.submit_order(request)
        order = self.client.to_order(raw)
        if trade_id and trade_id in self.trades:
            self.trades[trade_id].stop_order_id = order.order_id
        logger.info("Placed stop for %s x%g @ %.2f", symbol, quantity, stop_price)
        return order

    # -- modification -------------------------------------------------------

    def modify_stop(self, symbol: str, new_stop: float) -> Order | None:
        """Move a stop, tightening only.

        A stop that can move away from price is not a stop, it is a hope. This
        refuses to widen and says so rather than failing silently, because the
        caller asking to widen has a bug worth surfacing.

        Implemented as cancel-then-replace: Alpaca cannot change the stop price
        of a live stop order in place.
        """
        existing = [
            o for o in self.client.get_open_orders()
            if o.symbol == symbol and o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
        ]
        if not existing:
            logger.warning("%s: no open stop order to modify", symbol)
            return None

        current = existing[0]
        if current.stop_price is not None and new_stop <= current.stop_price:
            logger.info(
                "%s: refusing to widen stop from %.2f to %.2f (tighten only)",
                symbol, current.stop_price, new_stop,
            )
            return None

        self.cancel_order(current.order_id)
        return self.place_stop(symbol, current.quantity, new_stop)

    # -- cancellation and closing -------------------------------------------

    def cancel_order(self, order_id: str) -> None:
        self.client.trading_client.cancel_order_by_id(order_id)
        logger.info("Cancelled order %s", order_id[:8])

    def cancel_all_orders(self) -> None:
        self.client.trading_client.cancel_orders()
        logger.info("Cancelled all open orders")

    def close_position(self, symbol: str) -> Order | None:
        try:
            raw = self.client.trading_client.close_position(symbol)
        except Exception as exc:
            logger.warning("Could not close %s: %s", symbol, exc)
            return None
        logger.info("Closing position %s", symbol)
        return self.client.to_order(raw)

    def close_all_positions(self, reason: str = "") -> list[Order]:
        """Flatten everything. Called by the risk manager on a halt breach.

        Cancels resting orders first. Closing a position while a stop for it is
        still live would leave the stop as a naked short once the position is
        gone.
        """
        logger.warning("Closing ALL positions%s", f": {reason}" if reason else "")
        try:
            self.cancel_all_orders()
        except Exception as exc:
            logger.warning("Could not cancel open orders before flattening: %s", exc)

        raw_responses = self.client.trading_client.close_all_positions(cancel_orders=True)
        orders = []
        for response in raw_responses or []:
            body = getattr(response, "body", None)
            if body is not None:
                try:
                    orders.append(self.client.to_order(body))
                except Exception:
                    continue
        return orders

    # -- fill handling ------------------------------------------------------

    def _await_fill(
        self, trade: TradeRecord, retry_at_market: bool,
        signal: Signal, side: OrderSide, quantity: float,
    ) -> None:
        """Poll until filled or the timeout expires, then cancel.

        Cancelling an unfilled limit is the default. Retrying at market is
        opt-in: a limit that will not fill is usually saying something about
        liquidity, and converting it to a market order discards that information
        at the worst possible moment.
        """
        deadline = time.monotonic() + self.fill_timeout
        while time.monotonic() < deadline:
            order = self.client.get_order(trade.order_id)
            trade.status = order.status
            if order.status is OrderStatus.FILLED:
                trade.filled_at = order.filled_at
                trade.fill_price = order.average_fill_price
                trade.filled_qty = order.filled_quantity
                logger.info("Filled %s x%g @ %.2f", trade.symbol, trade.filled_qty,
                            trade.fill_price or 0)
                return
            if order.is_done:
                trade.notes.append(f"terminal without fill: {order.status.value}")
                return
            time.sleep(1.0)

        logger.info("%s: unfilled after %.0fs, cancelling", trade.symbol, self.fill_timeout)
        try:
            self.cancel_order(trade.order_id)
            trade.notes.append(f"limit unfilled after {self.fill_timeout:.0f}s, cancelled")
            trade.status = OrderStatus.CANCELLED
        except Exception as exc:
            trade.notes.append(f"cancel failed: {exc}")

        if retry_at_market:
            trade.notes.append("retried at market")
            logger.warning("%s: retrying at market", trade.symbol)
            self._submit_market_retry(trade, signal, side, quantity)

    def _submit_market_retry(self, trade: TradeRecord, signal: Signal,
                             side: OrderSide, quantity: float) -> None:
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=signal.symbol, qty=quantity,
            side=AlpacaSide.BUY if side is OrderSide.BUY else AlpacaSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"{trade.trade_id}-mkt",
        )
        order = self.client.to_order(self.client.trading_client.submit_order(request))
        trade.order_id = order.order_id
        trade.status = order.status

    def handle_partial_fill(self, trade: TradeRecord, order: Order) -> None:
        """Reconcile a partial fill.

        The stop must cover the quantity actually filled, not the quantity
        requested. Getting this wrong leaves part of the position naked, or
        turns the excess into a short when the stop triggers.
        """
        trade.filled_qty = order.filled_quantity
        trade.fill_price = order.average_fill_price
        trade.status = OrderStatus.PARTIALLY_FILLED
        trade.notes.append(f"partial fill {order.filled_quantity}/{order.quantity}")
        logger.warning(
            "%s partially filled %g of %g: stop must cover the filled quantity only",
            trade.symbol, order.filled_quantity, order.quantity,
        )

    # -- helpers ------------------------------------------------------------

    def _new_trade_record(
        self, signal: Signal, decision: RiskDecision, side: OrderSide, quantity: float
    ) -> TradeRecord:
        return TradeRecord(
            trade_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=side,
            requested_qty=signal.position_size_pct,
            approved_qty=quantity,
            signal_reasoning=signal.reasoning,
            risk_modifications=list(decision.modifications),
            regime=signal.regime_name,
            regime_confidence=signal.regime_probability,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    def _limit_price(self, price: float, side: OrderSide) -> float:
        """Price 0.1% through the touch, so a resting limit can actually fill."""
        offset = 1 + self.limit_offset if side is OrderSide.BUY else 1 - self.limit_offset
        return price * offset

    def _assert_price_sane(self, symbol: str, limit: float, side: OrderSide) -> None:
        """Refuse a limit price too far from the market to ever fill.

        The whole point is that the bad case is silent. A buy limit 19% below
        the market is accepted by the broker, rests, expires, and leaves an
        order log full of `canceled` with `filled_qty` 0. Nothing raises, so the
        only symptom is an account that never moves.

        Skipped when the quote is unusable rather than treated as a failure.
        Alpaca's IEX feed returns a zero bid and ask outside regular hours, and
        blocking every after-hours order because the reference is missing would
        trade one silent failure for a louder one.
        """
        if self.max_price_deviation <= 0:
            return
        try:
            quote = self.client.get_latest_quote(symbol)
        except Exception as exc:
            logger.debug("%s: no quote for the price guard (%s), allowing", symbol, exc)
            return

        bid, ask = quote.get("bid", 0.0) or 0.0, quote.get("ask", 0.0) or 0.0
        market = (bid + ask) / 2 if bid > 0 and ask > 0 else (ask or bid)
        if not market or market <= 0:
            logger.debug("%s: quote has no usable price, skipping the guard", symbol)
            return

        deviation = (limit - market) / market
        if abs(deviation) <= self.max_price_deviation:
            return

        logger.error(
            "%s: REFUSING %s limit %.2f against market %.2f (%.2f%% off, cap %.2f%%). "
            "Not submitted.",
            symbol, side.value, limit, market, deviation * 100,
            self.max_price_deviation * 100,
        )
        raise OrderExecutionError(
            f"{symbol}: limit {limit:.2f} deviates {deviation:+.2%} from the market "
            f"price {market:.2f}, beyond the {self.max_price_deviation:.0%} cap. "
            f"An order this far off would rest unfilled rather than fail, so it is "
            f"refused instead of submitted."
        )

    def _client_order_id(self, trade: TradeRecord, signal: Signal | None = None) -> str:
        """The id the broker stores alongside the order.

        Deterministic by default: `rt-SPY-buy-20260904`. Alpaca refuses a
        repeated client_order_id, which turns "one signal on one bar date
        produces one order" into a rule the broker enforces rather than one our
        process memory hopes to. That distinction is the whole point, because
        the case it protects against is a restart, when there is no memory.

        Falls back to the random trade_id when there is no signal to key on, or
        when `deterministic_ids` is off. The integration suite turns it off: it
        submits the same synthetic probe repeatedly by design, and re-running
        the test an hour later must not be rejected as a duplicate.

        Carries `order_id_prefix` either way, so the order book says which
        process placed a given order.
        """
        if not self.deterministic_ids or signal is None:
            return f"{self.order_id_prefix}{trade.trade_id}"
        side = "buy" if signal.direction is Direction.LONG else "sell"
        bar_date = pd.Timestamp(signal.timestamp).strftime("%Y%m%d")
        return f"{self.order_id_prefix}{signal.symbol}-{side}-{bar_date}"

    def _equivalent_open_order(
        self, symbol: str, side: OrderSide, price: float
    ) -> Order | None:
        """A resting order for the same symbol, side and roughly this price.

        Resubmitting is worse than merely redundant. Every open order holds
        buying power until it cancels, so a loop that reposts the same intent
        each cycle starves the rest of the universe of capital while filling
        nothing.

        Price is compared within `price_tolerance` rather than exactly, because
        the limit is derived from a live quote that moves between cycles. Two
        orders a cent apart are the same intent, not two intents.
        """
        try:
            resting = self.client.get_open_orders()
        except Exception as exc:
            # Failing open. A broker we cannot query is a broker we cannot
            # trade against either, and the submission below will surface it
            # with a better message than a dedupe check can.
            logger.warning("could not read open orders for the dedupe check: %s", exc)
            return None

        for order in resting:
            if order.symbol != symbol or order.side != side or not order.is_open:
                continue
            if order.limit_price is None or price <= 0:
                return order
            if abs(order.limit_price - price) / price <= self.price_tolerance:
                return order
        return None

    def _skip(self, trade: TradeRecord, reason: str, existing: Order | None = None
              ) -> TradeRecord:
        """Mark a submission that did not happen, and say why.

        Returned rather than raised. A skip is the idempotency layer working,
        not a failure, and routing it through the exception path would file it
        alongside genuine broker errors in the orchestrator's error count.

        Logged at INFO, because a system that silently declines to trade looks
        exactly like a system with no signals, and the difference is the first
        thing you want to know when nothing happened.
        """
        trade.skipped_reason = reason
        trade.status = OrderStatus.CANCELLED
        trade.notes.append(f"not submitted: {reason}")
        if existing is not None:
            trade.order_id = existing.order_id
        self.trades[trade.trade_id] = trade
        self.skipped.append(reason)
        logger.info("%s: not submitting, %s", trade.symbol, reason)
        return trade

    def _build_request(self, symbol: str, quantity: float, side: OrderSide,
                       order_type: OrderType, price: float,
                       client_order_id: str | None = None):
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        alpaca_side = AlpacaSide.BUY if side is OrderSide.BUY else AlpacaSide.SELL
        if order_type is OrderType.MARKET:
            return MarketOrderRequest(
                symbol=symbol, qty=quantity, side=alpaca_side,
                time_in_force=TimeInForce.DAY, client_order_id=client_order_id,
            )
        return LimitOrderRequest(
            symbol=symbol, qty=quantity, side=alpaca_side, time_in_force=TimeInForce.DAY,
            limit_price=round(self._limit_price(price, side), 2),
            client_order_id=client_order_id,
        )

    def _reference_price(self, signal: Signal) -> float:
        """Price to build the limit around.

        Falls back to the signal's entry price when the quote is unusable.
        Outside market hours Alpaca's IEX feed returns a zero ask, and pricing a
        limit off zero would submit a nonsense order at a fraction of a cent.
        """
        try:
            quote = self.client.get_latest_quote(signal.symbol)
            bid, ask = quote.get("bid", 0.0), quote.get("ask", 0.0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            if ask > 0:
                return ask
            if bid > 0:
                return bid
        except Exception as exc:
            logger.debug("%s: quote unavailable (%s), using signal entry price", signal.symbol, exc)
        return signal.entry_price

    def get_trade(self, trade_id: str) -> TradeRecord | None:
        return self.trades.get(trade_id)
