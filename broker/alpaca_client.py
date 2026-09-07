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
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd

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
    limit_price: Optional[float]
    stop_price: Optional[float]
    average_fill_price: Optional[float]
    submitted_at: Optional[datetime]
    filled_at: Optional[datetime]
    client_order_id: Optional[str] = None
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
    asset_id: Optional[str] = None


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
            last: Optional[Exception] = None
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
        paper: Optional[bool] = None,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        confirm_live: Optional[Callable[[str], str]] = None,
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
        return Account(
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

    def is_market_open(self) -> bool:
        return self.get_clock()["is_open"]

    # -- positions ----------------------------------------------------------

    @_retry()
    def get_positions(self) -> list[Position]:
        return [self._to_position(p) for p in self._require_connection().get_all_positions()]

    def get_position(self, symbol: str) -> Optional[Position]:
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
        self, status: str = "all", limit: int = 100, after: Optional[datetime] = None
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
        """Best bid and ask.

        Outside market hours the free IEX feed commonly returns a zero ask, so
        callers must treat zero as "no quote" rather than as a price. The risk
        manager's spread check and the executor's limit pricing both do.
        """
        from alpaca.data.requests import StockLatestQuoteRequest

        quote = self.data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol)
        )[symbol]
        return {
            "bid": float(quote.bid_price or 0),
            "ask": float(quote.ask_price or 0),
            "bid_size": float(quote.bid_size or 0),
            "ask_size": float(quote.ask_size or 0),
            "timestamp": quote.timestamp,
            "tradeable": True,
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
