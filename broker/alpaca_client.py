"""
Alpaca API wrapper. The connection to the brokerage.

Phase 6.

Credentials come from `.env` and are never hardcoded, never logged, and never
pasted into a chat window. `.env` is gitignored and `tests/test_orders.py` scans
the whole repo for anything shaped like an Alpaca key on every run.

PAPER IS THE DEFAULT AND LIVE IS DELIBERATELY AWKWARD
-----------------------------------------------------
`paper=True` is the default everywhere. Going live requires both
`ALPACA_PAPER=false` in the environment and typing a confirmation phrase at an
interactive prompt. A non-interactive process (cron, CI, a scheduled job) cannot
satisfy the second condition, so it cannot ever reach the live endpoint by
accident. That asymmetry is the point.

The client also cross-checks the key prefix: Alpaca issues paper keys beginning
`PK` and live keys beginning `AK`, so a live key with `ALPACA_PAPER=true`
mismatches and is refused rather than quietly connecting somewhere unexpected.

THIN ON PURPOSE
---------------
This wrapper translates Alpaca's types into the dataclasses below and does
nothing else. Keeping broker-specific objects from escaping this file is what
makes a later move to IBKR a contained job rather than a rewrite of the strategy
layer.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
LIVE_CONFIRMATION = "YES I UNDERSTAND THE RISKS"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


#: Alpaca's status vocabulary is wider than ours. Anything unmapped becomes
#: PENDING rather than raising: an unknown status must not stop the loop, and
#: treating it as "not done yet" is the safe reading.
_ALPACA_STATUS = {
    "new": OrderStatus.OPEN, "accepted": OrderStatus.OPEN,
    "pending_new": OrderStatus.PENDING, "accepted_for_bidding": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIALLY_FILLED, "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.CANCELLED, "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED, "expired": OrderStatus.EXPIRED,
    "replaced": OrderStatus.CANCELLED, "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.PENDING, "stopped": OrderStatus.CANCELLED,
    "calculated": OrderStatus.PENDING, "held": OrderStatus.PENDING,
    "pending_cancel": OrderStatus.OPEN, "pending_replace": OrderStatus.OPEN,
    "pending_review": OrderStatus.PENDING,
}


@dataclass(frozen=True)
class Account:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    is_paper: bool
    currency: str = "USD"
    multiplier: float = 1.0
    daytrade_count: int = 0
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    account_blocked: bool = False
    status: str = "unknown"
    last_equity: float = 0.0

    @property
    def available_margin(self) -> float:
        """Buying power beyond the cash on hand. Borrowable, not owned."""
        return max(0.0, self.buying_power - self.cash)

    @property
    def is_tradeable(self) -> bool:
        return not (self.trading_blocked or self.account_blocked) and self.status.upper() == "ACTIVE"


@dataclass(frozen=True)
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    filled_quantity: float
    status: OrderStatus
    order_type: OrderType
    limit_price: float | None
    stop_price: float | None
    average_fill_price: float | None
    submitted_at: datetime | None
    filled_at: datetime | None
    client_order_id: str | None = None
    legs: tuple = ()

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_done(self) -> bool:
        return self.status in (
            OrderStatus.FILLED, OrderStatus.CANCELLED,
            OrderStatus.REJECTED, OrderStatus.EXPIRED,
        )


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    average_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealised_pnl: float
    unrealised_pnl_pct: float
    side: str = "long"
    asset_id: str | None = None


class BrokerConnectionError(RuntimeError):
    """Raised when the broker is unreachable after the retry budget is spent."""


class LiveTradingNotConfirmed(RuntimeError):
    """Raised when live mode is requested without the typed confirmation."""


def _retry(attempts: int = 4, base_delay: float = 0.5):
    """Retry a broker call with exponential backoff.

    Wraps reads only. Order submission is deliberately never retried
    automatically: a network timeout does not tell you whether the order reached
    the exchange, and a blind retry is how you end up holding two copies of the
    same position.
    """
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last: Exception | None = None
            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last = exc
                    if attempt == attempts - 1:
                        break
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "%s failed (%s), retrying in %.1fs [%d/%d]",
                        fn.__name__, exc, delay, attempt + 1, attempts,
                    )
                    time.sleep(delay)
            raise BrokerConnectionError(f"{fn.__name__} failed after {attempts} attempts: {last}")
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


class AlpacaClient:
    """Thin wrapper over alpaca-py.

    Construct, then `connect()`. `connect()` runs a health check and refuses to
    proceed if the account is blocked or the paper/live mode does not match what
    was requested.
    """

    def __init__(
        self,
        paper: bool | None = None,
        api_key: str | None = None,
        secret_key: str | None = None,
        confirm_live: Callable[[str], str] | None = None,
        load_env: bool = True,
    ) -> None:
        if load_env:
            self._load_env()

        env_paper = os.getenv("ALPACA_PAPER", "true").strip().lower() not in ("false", "0", "no")
        self.paper = env_paper if paper is None else paper
        self.base_url = PAPER_URL if self.paper else LIVE_URL

        # `None` means "read the environment"; an explicit empty string means
        # "no credentials", which is how tests exercise the missing-key path
        # without having to unset a global.
        self._api_key = os.getenv("ALPACA_API_KEY") if api_key is None else api_key
        self._secret_key = os.getenv("ALPACA_SECRET_KEY") if secret_key is None else secret_key
        self._confirm_live = confirm_live or input

        self._trading_client = None
        self._data_client = None
        self._clock_cache: tuple[float, bool] | None = None
        self._connected = False

    @staticmethod
    def _load_env() -> None:
        from pathlib import Path

        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    # -- connection ---------------------------------------------------------

    def connect(self) -> Account:
        """Open the session, verify credentials, and health check the account.

        Returns the account so a caller cannot proceed without having seen its
        state. Raises rather than returning a degraded client: a half-connected
        broker that fails on the first order is worse than one that never
        started.
        """
        if not self._api_key or not self._secret_key:
            raise BrokerConnectionError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set. Copy .env.example "
                "to .env and fill them in. Never paste keys into a chat window."
            )

        self._check_key_prefix()
        if not self.paper:
            self._require_live_confirmation()

        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self._trading_client = TradingClient(self._api_key, self._secret_key, paper=self.paper)
        self._data_client = StockHistoricalDataClient(self._api_key, self._secret_key)
        self._connected = True

        account = self.health_check()
        logger.info(
            "Connected to Alpaca %s: equity $%s, buying power $%s",
            "PAPER" if self.paper else "LIVE",
            f"{account.equity:,.2f}", f"{account.buying_power:,.2f}",
        )
        return account

    def _check_key_prefix(self) -> None:
        """Alpaca issues paper keys as PK... and live keys as AK...

        A live key used with `ALPACA_PAPER=true` would otherwise connect to the
        paper endpoint with live credentials, or worse. Refuse the mismatch
        rather than guess which the operator meant.
        """
        prefix = (self._api_key or "")[:2].upper()
        if prefix == "AK" and self.paper:
            raise BrokerConnectionError(
                "ALPACA_API_KEY looks like a LIVE key (AK...) but ALPACA_PAPER=true. "
                "Refusing to connect. Use a paper key, or set ALPACA_PAPER=false "
                "deliberately."
            )
        if prefix == "PK" and not self.paper:
            raise BrokerConnectionError(
                "ALPACA_API_KEY looks like a PAPER key (PK...) but live mode was "
                "requested. Refusing to connect."
            )

    def _require_live_confirmation(self) -> None:
        """Typed confirmation before any live connection.

        Deliberately an interactive prompt. A scheduled job or CI run has no
        stdin, so it cannot satisfy this and cannot reach the live endpoint by
        accident, which is the failure mode worth engineering against.
        """
        banner = (
            "\n"
            "==============================================================\n"
            "  LIVE TRADING MODE. Real money is at risk.\n"
            f"  Endpoint: {LIVE_URL}\n"
            f"  Type '{LIVE_CONFIRMATION}' to confirm: "
        )
        try:
            answer = self._confirm_live(banner)
        except (EOFError, OSError) as exc:
            raise LiveTradingNotConfirmed(
                "Live trading requires an interactive confirmation and there is no "
                "stdin available. A non-interactive process may not trade live."
            ) from exc

        if (answer or "").strip() != LIVE_CONFIRMATION:
            raise LiveTradingNotConfirmed(
                f"Live trading not confirmed. Expected exactly '{LIVE_CONFIRMATION}'."
            )
        logger.warning("LIVE TRADING CONFIRMED by operator")

    def health_check(self) -> Account:
        """Verify the account is active and matches the requested mode."""
        account = self.get_account()
        if account.is_paper != self.paper:
            raise BrokerConnectionError(
                f"Mode mismatch: requested {'paper' if self.paper else 'live'} but the "
                f"account reports {'paper' if account.is_paper else 'live'}."
            )
        if not account.is_tradeable:
            raise BrokerConnectionError(
                f"Account not tradeable: status={account.status}, "
                f"trading_blocked={account.trading_blocked}, "
                f"account_blocked={account.account_blocked}"
            )
        return account

    def is_connected(self) -> bool:
        return self._connected and self._trading_client is not None

    def disconnect(self) -> None:
        self._trading_client = None
        self._data_client = None
        self._clock_cache: tuple[float, bool] | None = None
        self._connected = False

    def _require_connection(self):
        if not self.is_connected():
            raise BrokerConnectionError("Not connected. Call connect() first.")
        return self._trading_client

    # -- account ------------------------------------------------------------

    @_retry()
    def get_account(self) -> Account:
        raw = self._require_connection().get_account()
        number = str(getattr(raw, "account_number", "") or "")
        account = Account(
            equity=float(raw.equity or 0),
            cash=float(raw.cash or 0),
            buying_power=float(raw.buying_power or 0),
            portfolio_value=float(raw.portfolio_value or raw.equity or 0),
            # Alpaca prefixes paper account numbers with PA. Read from the
            # account itself rather than trusting the flag we asked for.
            is_paper=number.upper().startswith("PA"),
            multiplier=float(raw.multiplier or 1),
            daytrade_count=int(raw.daytrade_count or 0),
            pattern_day_trader=bool(raw.pattern_day_trader),
            trading_blocked=bool(raw.trading_blocked),
            account_blocked=bool(raw.account_blocked),
            status=str(getattr(raw.status, "value", raw.status)),
            last_equity=float(raw.last_equity or 0),
        )
        return self._remark_account_if_closed(account)

    def _remark_account_if_closed(self, account: Account) -> Account:
        """Keep equity consistent with the re-marked positions.

        `_remark_if_closed` corrects what each position is worth. This corrects
        the total, which is otherwise Alpaca's own figure carrying exactly the
        same stale off-hours marks. Fixing one without the other would be worse
        than fixing neither: the dashboard would print position P&L that does
        not add up to the equity above it, and there would be no way to tell
        which of the two to believe.

        This is not only cosmetic. Every circuit breaker is a ratio against
        equity, so a phantom mark is a phantom drawdown. At the 22% exposure
        this account runs, the gap was 0.32% and nowhere near the 5% reduce
        level; the same per-position mismarks of up to 3.9% at full exposure
        would be within reach of it, and would halt trading over nothing.

        Costs one clock call and one latest-trade call per account refresh,
        which at a fifteen-minute cadence is not worth optimising away.
        """
        try:
            if self.is_market_open():
                return account
            raw_positions = self._require_connection().get_all_positions()
            positions = self._remark_if_closed(
                [self._to_position(p) for p in raw_positions], market_open=False,
            )
        except Exception as exc:
            logger.warning("Could not re-mark equity off the last trade: %s", exc)
            return account

        if not positions:
            return account
        equity = account.cash + sum(p.market_value for p in positions)
        if equity <= 0:
            return account
        return replace(account, equity=equity, portfolio_value=equity)

    def get_available_margin(self) -> float:
        """Buying power beyond cash. Borrowable, not owned.

        Alpaca offers 2x overnight (Reg T) and 4x intraday (PDT). The system's
        configured 1.25x ceiling is far below both and is deliberately
        conservative.
        """
        return self.get_account().available_margin

    # -- market hours -------------------------------------------------------

    @_retry()
    def get_clock(self) -> dict[str, Any]:
        clock = self._require_connection().get_clock()
        return {
            "is_open": bool(clock.is_open),
            "timestamp": clock.timestamp,
            "next_open": clock.next_open,
            "next_close": clock.next_close,
        }

    def is_market_open(self, max_age_seconds: float = 30.0) -> bool:
        """Whether a session is running, cached briefly.

        Every quote now carries a `usable` flag derived from this, so a scan of
        fourteen symbols would otherwise ask the broker what time it is fourteen
        times. The open and close are fixed points in the day, so a reading a
        few seconds old cannot be wrong about anything except the single tick
        either side of the bell, and nothing here trades on that tick.
        """
        now = time.monotonic()
        cached = self._clock_cache
        if cached is not None and now - cached[0] < max_age_seconds:
            return cached[1]
        is_open = bool(self.get_clock()["is_open"])
        self._clock_cache = (now, is_open)
        return is_open

    # -- positions ----------------------------------------------------------

    @_retry()
    def get_positions(self) -> list[Position]:
        return self._remark_if_closed(
            [self._to_position(p) for p in self._require_connection().get_all_positions()]
        )

    def _remark_if_closed(self, positions: list[Position],
                          *, market_open: bool | None = None) -> list[Position]:
        """Re-price positions off the last real trade while the market is shut.

        `position.current_price` from Alpaca is not a traded price outside
        regular hours, and it drifts. Measured on this account at 07:14 UTC on a
        Monday, with no share having changed hands since Friday's close:

            symbol   broker mark   last trade   official close
            AVGO          352.47       361.90           361.99
            SMCI           38.54        40.06            40.10
            QQQ           705.05       714.99           714.88

        All eight positions were marked below their own close, by up to 3.9%,
        and the marks kept moving across a weekend. That turned a real -$93 of
        unrealised P&L into a displayed -$413, and took $319 off published
        equity. Three of the eight quotes were one-sided with a zero ask, which
        is the likeliest source of the bad mark.

        This is the normal case rather than an edge case: a dashboard read from
        Singapore is read outside US market hours nearly every time. During the
        session Alpaca's mark is live and correct, so it is left alone.
        """
        if not positions:
            return positions
        try:
            if market_open is None:
                market_open = self.is_market_open()
            if market_open:
                return positions
        except Exception:
            return positions        # cannot tell what the session is: do not guess

        try:
            from alpaca.data.requests import StockLatestTradeRequest

            trades = self.data_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=[p.symbol for p in positions])
            )
        except Exception as exc:
            logger.warning("Could not re-mark positions off the last trade: %s", exc)
            return positions

        remarked = []
        for position in positions:
            price = float(getattr(trades.get(position.symbol), "price", 0) or 0)
            if price <= 0 or position.quantity == 0:
                remarked.append(position)
                continue
            value = price * position.quantity
            pnl = value - position.cost_basis
            remarked.append(replace(
                position,
                current_price=price,
                market_value=value,
                unrealised_pnl=pnl,
                unrealised_pnl_pct=pnl / position.cost_basis if position.cost_basis else 0.0,
            ))
        return remarked

    def get_position(self, symbol: str) -> Position | None:
        try:
            return self._to_position(self._require_connection().get_open_position(symbol))
        except Exception:
            return None      # no position is a normal answer, not an error

    @staticmethod
    def _to_position(raw) -> Position:
        return Position(
            symbol=raw.symbol,
            quantity=float(raw.qty or 0),
            average_entry_price=float(raw.avg_entry_price or 0),
            current_price=float(raw.current_price or 0),
            market_value=float(raw.market_value or 0),
            cost_basis=float(raw.cost_basis or 0),
            unrealised_pnl=float(raw.unrealized_pl or 0),
            unrealised_pnl_pct=float(raw.unrealized_plpc or 0),
            side=str(getattr(raw.side, "value", raw.side)),
            asset_id=str(getattr(raw, "asset_id", "") or ""),
        )

    # -- orders -------------------------------------------------------------

    @_retry()
    def get_order_history(
        self, status: str = "all", limit: int = 100, after: datetime | None = None
    ) -> list[Order]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        request = GetOrdersRequest(
            status=QueryOrderStatus(status), limit=limit,
            after=after or (datetime.now() - timedelta(days=30)),
        )
        return [self.to_order(o) for o in self._require_connection().get_orders(request)]

    @_retry()
    def get_order(self, order_id: str) -> Order:
        return self.to_order(self._require_connection().get_order_by_id(order_id))

    def get_open_orders(self) -> list[Order]:
        return self.get_order_history(status="open")

    @staticmethod
    def to_order(raw) -> Order:
        """Translate an Alpaca order into ours. The only place that mapping lives."""
        status = str(getattr(raw.status, "value", raw.status)).lower()
        order_type = str(getattr(raw.order_type, "value", getattr(raw, "type", "market"))).lower()
        return Order(
            order_id=str(raw.id),
            symbol=raw.symbol,
            side=OrderSide(str(getattr(raw.side, "value", raw.side)).lower()),
            quantity=float(raw.qty or 0),
            filled_quantity=float(raw.filled_qty or 0),
            status=_ALPACA_STATUS.get(status, OrderStatus.PENDING),
            order_type=OrderType(order_type) if order_type in {t.value for t in OrderType}
            else OrderType.MARKET,
            limit_price=float(raw.limit_price) if raw.limit_price else None,
            stop_price=float(raw.stop_price) if raw.stop_price else None,
            average_fill_price=float(raw.filled_avg_price) if raw.filled_avg_price else None,
            submitted_at=raw.submitted_at,
            filled_at=raw.filled_at,
            client_order_id=getattr(raw, "client_order_id", None),
            legs=tuple(str(leg.id) for leg in (raw.legs or [])) if getattr(raw, "legs", None) else (),
        )

    @_retry(attempts=2)
    def get_latest_quote(self, symbol: str) -> dict[str, float]:
        """Best bid and ask, with an explicit verdict on whether it is a market.

        `usable` is the important field. Outside regular hours the free IEX feed
        returns quotes that look like data and are not: one-sided books with a
        zero ask, and two-sided books 10% wide. Three separate guards used to
        each decide for themselves whether to believe a quote, and they decided
        differently, which is how a run reached the state where every one of
        fourteen symbols was refused:

            SPY QQQ NVDA META    wash-trade rejection at the broker
            AAPL AMD TSLA AVGO   "limit deviates +6% from the market", where
                                 the market was a lone bid with no ask
            MSFT AMZN GOOGL      "spread 10% exceeds 0.50%"
            PLTR COIN SMCI

        Every one of those was a closed-market quote being treated as a live
        one. The decision is taken after the close by design and the order
        queues for the next open, so the spread right now is not a fact about
        anything. One flag, computed once, and every guard honours it.
        """
        from alpaca.data.requests import StockLatestQuoteRequest

        quote = self.data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol)
        )[symbol]
        bid = float(quote.bid_price or 0)
        ask = float(quote.ask_price or 0)

        two_sided = bid > 0 and ask > 0
        try:
            session_open = self.is_market_open()
        except Exception:
            session_open = False        # cannot confirm a session: assume none

        return {
            "bid": bid,
            "ask": ask,
            "bid_size": float(quote.bid_size or 0),
            "ask_size": float(quote.ask_size or 0),
            "timestamp": quote.timestamp,
            "tradeable": True,
            # A quote is a market only when there is a session behind it and
            # both sides are there. Anything else is a number, not a price.
            "usable": bool(session_open and two_sided),
            "market_open": bool(session_open),
        }

    @property
    def trading_client(self):
        """Escape hatch for the order executor and the trading stream.

        Everything else should use the typed methods above. This exists because
        order submission needs Alpaca's request objects, and duplicating those
        builders here would put broker-specific types in two files instead of
        one.
        """
        return self._require_connection()

    @property
    def data_client(self):
        if self._data_client is None:
            raise BrokerConnectionError("Not connected. Call connect() first.")
        return self._data_client
