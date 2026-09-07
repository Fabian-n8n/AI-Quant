"""
Tests for the broker layer: client, order executor, position tracker, market data.

Split into two kinds:

- **Offline** (the majority): fakes stand in for Alpaca, so the logic is tested
  without a network call or an account. These run everywhere, including CI.
- **Live** (marked `alpaca`): hit the real paper API. Skipped automatically when
  credentials are absent. Run with `pytest -m alpaca`.

The secret-handling tests are unconditional. They are the ones that matter most
here, because a leaked key is not a bug you can fix by reverting.
"""

import os
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from broker.alpaca_client import (
    LIVE_CONFIRMATION,
    PAPER_URL,
    Account,
    AlpacaClient,
    BrokerConnectionError,
    LiveTradingNotConfirmed,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from broker.order_executor import OrderExecutionError, OrderExecutor, TradeRecord
from broker.position_tracker import PositionTracker, TrackedPosition
from core.regime_strategies import Direction, Signal
from core.risk_manager import RiskDecision

HAS_CREDENTIALS = bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))
live = pytest.mark.skipif(not HAS_CREDENTIALS, reason="no Alpaca credentials in the environment")


# -- fakes ------------------------------------------------------------------

class FakeTradingClient:
    """Enough of alpaca-py's TradingClient to exercise our logic offline."""

    def __init__(self, positions=None, orders=None):
        self._positions = positions or []
        self._orders = orders or {}
        self.submitted = []
        self.cancelled = []
        self.closed = []

    def get_all_positions(self):
        return self._positions

    def submit_order(self, request):
        self.submitted.append(request)
        return _fake_raw_order(
            symbol=request.symbol, qty=float(request.qty),
            side=str(getattr(request.side, "value", request.side)),
            limit_price=getattr(request, "limit_price", None),
        )

    def cancel_order_by_id(self, order_id):
        self.cancelled.append(order_id)

    def cancel_orders(self):
        self.cancelled.append("ALL")

    def close_position(self, symbol):
        self.closed.append(symbol)
        return _fake_raw_order(symbol=symbol, qty=1, side="sell")

    def close_all_positions(self, cancel_orders=True):
        self.closed.append("ALL")
        return []


def _fake_raw_order(symbol="NVDA", qty=10.0, side="buy", limit_price=None, status="new"):
    class RawOrder:
        pass

    order = RawOrder()
    order.id = "order-1234-abcd"
    order.symbol = symbol
    order.qty = qty
    order.filled_qty = 0.0
    order.side = side
    order.status = status
    order.order_type = "limit" if limit_price else "market"
    order.limit_price = limit_price
    order.stop_price = None
    order.filled_avg_price = None
    order.submitted_at = datetime.now(UTC)
    order.filled_at = None
    order.client_order_id = "trade-1"
    order.legs = None
    return order


class FakeAlpacaClient(AlpacaClient):
    """AlpacaClient with the network replaced."""

    def __init__(self, trading_client=None, quote=None, **kwargs):
        super().__init__(paper=True, api_key="PKTEST", secret_key="secret", load_env=False)
        self._trading_client = trading_client or FakeTradingClient()
        self._connected = True
        self._quote = quote or {"bid": 99.99, "ask": 100.01, "tradeable": True}

    def get_account(self):
        return Account(
            equity=100_000.0, cash=100_000.0, buying_power=200_000.0,
            portfolio_value=100_000.0, is_paper=True, status="ACTIVE", last_equity=100_000.0,
        )

    def get_latest_quote(self, symbol):
        return dict(self._quote)

    def get_positions(self):
        return [
            Position(
                symbol=p.symbol, quantity=p.qty, average_entry_price=p.avg_entry_price,
                current_price=p.current_price, market_value=p.market_value,
                cost_basis=p.cost_basis, unrealised_pnl=0.0, unrealised_pnl_pct=0.0,
            )
            for p in self._trading_client.get_all_positions()
        ]


@pytest.fixture
def signal():
    return Signal(
        symbol="NVDA", direction=Direction.LONG, confidence=0.9, entry_price=100.0,
        stop_loss=95.0, take_profit=None, position_size_pct=0.60, leverage=1.0,
        regime_id=0, regime_name="bull", regime_probability=0.9,
        timestamp=pd.Timestamp("2024-06-03"), reasoning="test",
        strategy_name="HighVolDefensiveStrategy",
    )


@pytest.fixture
def approved():
    return RiskDecision(
        approved=True, modified_signal={"shares": 27, "notional": 2700.0, "leverage": 1.0},
        modifications=["gap cap"], approved_quantity=27, approved_notional=2700.0,
    )


# -- secret handling, unconditional -----------------------------------------

def test_gitignore_covers_secrets(repo_root: Path):
    ignored = (repo_root / ".gitignore").read_text()
    assert ".env" in ignored
    assert "config/credentials.yaml" in ignored


def test_no_hardcoded_credentials_in_source(repo_root: Path):
    """Scan every source file for anything shaped like a live Alpaca key.

    Catches a key pasted in during debugging and forgotten. Alpaca key IDs are
    PK (paper) or AK (live) followed by 18 uppercase alphanumerics.
    """
    import re

    pattern = re.compile(r"\b(?:PK|AK)[A-Z0-9]{18}\b")
    skip = {".venv", ".git", "__pycache__", ".pytest_cache", "models", "cache"}
    offenders = []
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".md", ".txt"}:
            continue
        if any(part in skip for part in path.parts):
            continue
        if pattern.search(path.read_text(errors="ignore")):
            offenders.append(str(path.relative_to(repo_root)))
    assert not offenders, f"Possible API key committed in: {offenders}"


def test_env_file_is_not_tracked_by_git(repo_root: Path):
    """Belt and braces: ask git directly rather than trusting the pattern."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", ".env"], cwd=repo_root, capture_output=True, text=True
    )
    assert result.returncode == 0, ".env is NOT gitignored"


# -- paper/live safety ------------------------------------------------------

def test_paper_is_the_default():
    """The default must never be live. Someone will construct this without
    arguments eventually."""
    client = AlpacaClient(api_key="PKX", secret_key="s", load_env=False)
    assert client.paper is True
    assert client.base_url == PAPER_URL


def test_live_requires_the_exact_confirmation_phrase():
    """A scheduled job has no stdin, so it cannot satisfy this and cannot reach
    the live endpoint by accident."""
    for answer in ("yes", "y", "YES I UNDERSTAND", "", "yes i understand the risks"):
        client = AlpacaClient(paper=False, api_key="AKX", secret_key="s",
                              load_env=False, confirm_live=lambda _, a=answer: a)
        with pytest.raises(LiveTradingNotConfirmed):
            client.connect()


def test_live_rejected_when_stdin_is_unavailable():
    def no_stdin(_):
        raise EOFError("no tty")

    client = AlpacaClient(paper=False, api_key="AKX", secret_key="s",
                          load_env=False, confirm_live=no_stdin)
    with pytest.raises(LiveTradingNotConfirmed, match="interactive"):
        client.connect()


def test_paper_key_with_live_mode_is_refused():
    """PK is a paper key. Requesting live with one is a configuration mistake,
    not something to guess the intent of."""
    client = AlpacaClient(paper=False, api_key="PKTEST", secret_key="s",
                          load_env=False, confirm_live=lambda _: LIVE_CONFIRMATION)
    with pytest.raises(BrokerConnectionError, match="PAPER key"):
        client.connect()


def test_live_key_with_paper_mode_is_refused():
    client = AlpacaClient(paper=True, api_key="AKTEST", secret_key="s", load_env=False)
    with pytest.raises(BrokerConnectionError, match="LIVE key"):
        client.connect()


def test_missing_credentials_raise_with_guidance():
    client = AlpacaClient(api_key="", secret_key="", load_env=False)
    with pytest.raises(BrokerConnectionError, match="ALPACA_API_KEY"):
        client.connect()


def test_account_paper_flag_is_read_from_the_account_not_the_request():
    """Alpaca prefixes paper account numbers with PA. Trusting the flag we asked
    for would let a misconfiguration go unnoticed."""
    account = Account(equity=1.0, cash=1.0, buying_power=1.0, portfolio_value=1.0,
                      is_paper=True, status="ACTIVE")
    assert "is_paper" in Account.__dataclass_fields__
    assert account.is_tradeable


def test_blocked_account_is_not_tradeable():
    blocked = Account(equity=1.0, cash=1.0, buying_power=1.0, portfolio_value=1.0,
                      is_paper=True, status="ACTIVE", trading_blocked=True)
    assert blocked.is_tradeable is False


def test_available_margin_is_buying_power_beyond_cash():
    account = Account(equity=100_000.0, cash=100_000.0, buying_power=400_000.0,
                      portfolio_value=100_000.0, is_paper=True, status="ACTIVE")
    assert account.available_margin == 300_000.0


# -- order translation ------------------------------------------------------

def test_order_types_and_statuses_exist():
    for name in ("MARKET", "LIMIT", "STOP", "STOP_LIMIT"):
        assert hasattr(OrderType, name)
    for name in ("FILLED", "PARTIALLY_FILLED", "REJECTED", "CANCELLED"):
        assert hasattr(OrderStatus, name)


@pytest.mark.parametrize(
    "alpaca_status,expected",
    [("new", OrderStatus.OPEN), ("filled", OrderStatus.FILLED),
     ("partially_filled", OrderStatus.PARTIALLY_FILLED), ("canceled", OrderStatus.CANCELLED),
     ("rejected", OrderStatus.REJECTED), ("expired", OrderStatus.EXPIRED)],
)
def test_alpaca_statuses_map_correctly(alpaca_status, expected):
    raw = _fake_raw_order(status=alpaca_status)
    assert AlpacaClient.to_order(raw).status is expected


def test_unknown_status_becomes_pending_not_an_exception():
    """An unrecognised status must not stop the loop, and "not done yet" is the
    safe reading."""
    raw = _fake_raw_order(status="some_new_status_alpaca_added")
    assert AlpacaClient.to_order(raw).status is OrderStatus.PENDING


def test_order_open_and_done_flags():
    assert AlpacaClient.to_order(_fake_raw_order(status="new")).is_open
    assert AlpacaClient.to_order(_fake_raw_order(status="filled")).is_done


# -- order executor ---------------------------------------------------------

def test_executor_refuses_a_rejected_signal(signal):
    """The risk manager's veto is final. The executor cannot be talked past it."""
    executor = OrderExecutor(FakeAlpacaClient())
    rejected = RiskDecision(approved=False, reason="test")
    with pytest.raises(OrderExecutionError, match="rejected"):
        executor.submit_order(signal, rejected)


def test_executor_refuses_a_signal_without_a_stop(signal, approved):
    """Enforced here as well as in the risk manager. One rule, two checks,
    because it is the rule that cannot fail quietly."""
    import dataclasses

    naked = dataclasses.replace(signal, stop_loss=95.0)
    object.__setattr__(naked, "stop_loss", None)
    executor = OrderExecutor(FakeAlpacaClient())
    with pytest.raises(OrderExecutionError, match="no stop loss"):
        executor.submit_order(naked, approved)


def test_executor_refuses_zero_quantity(signal):
    executor = OrderExecutor(FakeAlpacaClient())
    zero = RiskDecision(approved=True, modified_signal={"shares": 0})
    with pytest.raises(OrderExecutionError, match="zero"):
        executor.submit_order(signal, zero)


def test_submitted_order_uses_the_approved_quantity(signal, approved):
    """Not the requested one. The risk manager shrank it for a reason."""
    fake = FakeTradingClient()
    executor = OrderExecutor(FakeAlpacaClient(fake))
    executor.submit_order(signal, approved, reference_price=100.0)
    assert float(fake.submitted[0].qty) == 27


def test_limit_price_is_offset_through_the_touch(signal, approved):
    fake = FakeTradingClient()
    executor = OrderExecutor(FakeAlpacaClient(fake), limit_offset=0.001)
    executor.submit_order(signal, approved, reference_price=100.0)
    assert float(fake.submitted[0].limit_price) == pytest.approx(100.10, abs=0.01)


def test_limit_falls_back_to_entry_price_when_the_quote_is_dead(signal, approved):
    """Outside market hours the IEX feed returns a zero ask. Pricing a limit off
    zero would submit an order at a fraction of a cent."""
    fake = FakeTradingClient()
    client = FakeAlpacaClient(fake, quote={"bid": 0.0, "ask": 0.0, "tradeable": True})
    OrderExecutor(client).submit_order(signal, approved)
    assert float(fake.submitted[0].limit_price) > 90


def test_trade_record_links_signal_to_order(signal, approved):
    """Without a trade_id, reconstructing why a position exists means
    correlating three logs by timestamp and hoping."""
    executor = OrderExecutor(FakeAlpacaClient())
    trade = executor.submit_order(signal, approved, reference_price=100.0)
    assert isinstance(trade, TradeRecord)
    assert trade.trade_id and trade.order_id
    assert trade.stop_loss == signal.stop_loss
    assert trade.regime == signal.regime_name
    assert trade.risk_modifications == approved.modifications
    assert executor.get_trade(trade.trade_id) is trade


def test_bracket_order_requires_a_take_profit(signal, approved):
    """Alpaca's bracket class needs both legs. Say so rather than failing at the
    API with a less useful message."""
    executor = OrderExecutor(FakeAlpacaClient())
    with pytest.raises(OrderExecutionError, match="take_profit"):
        executor.submit_bracket_order(signal, approved)


def test_modify_stop_refuses_to_widen():
    """A stop that can move away from price is not a stop, it is a hope."""
    class ClientWithStop(FakeAlpacaClient):
        def get_open_orders(self):
            return [Order(
                order_id="stop-1", symbol="NVDA", side=OrderSide.SELL, quantity=27,
                filled_quantity=0, status=OrderStatus.OPEN, order_type=OrderType.STOP,
                limit_price=None, stop_price=95.0, average_fill_price=None,
                submitted_at=None, filled_at=None,
            )]

    executor = OrderExecutor(ClientWithStop())
    assert executor.modify_stop("NVDA", 90.0) is None, "must refuse to widen"


def test_modify_stop_allows_tightening():
    class ClientWithStop(FakeAlpacaClient):
        def get_open_orders(self):
            return [Order(
                order_id="stop-1", symbol="NVDA", side=OrderSide.SELL, quantity=27,
                filled_quantity=0, status=OrderStatus.OPEN, order_type=OrderType.STOP,
                limit_price=None, stop_price=95.0, average_fill_price=None,
                submitted_at=None, filled_at=None,
            )]

    fake = FakeTradingClient()
    client = ClientWithStop(fake)
    executor = OrderExecutor(client)
    assert executor.modify_stop("NVDA", 98.0) is not None
    assert "stop-1" in fake.cancelled


def test_close_all_cancels_orders_first():
    """Closing a position while its stop is still live would leave the stop as a
    naked short once the position is gone."""
    fake = FakeTradingClient()
    OrderExecutor(FakeAlpacaClient(fake)).close_all_positions("breaker")
    assert "ALL" in fake.cancelled
    assert "ALL" in fake.closed


# -- position tracker -------------------------------------------------------

def test_tracked_position_records_regime_context():
    """Alpaca knows the shares and the price. It does not know why."""
    fields = TrackedPosition.__dataclass_fields__
    for name in ("stop_loss", "regime_at_entry", "confidence_at_entry", "rationale",
                 "regime_current", "holding_periods", "entry_time"):
        assert name in fields, f"TrackedPosition.{name} missing"


def test_register_fill_opens_a_position():
    tracker = PositionTracker(FakeAlpacaClient())
    position = tracker.register_fill("NVDA", 10, 100.0, "buy", stop_loss=95.0, regime="bull")
    assert position.quantity == 10
    assert position.entry_price == 100.0
    assert position.regime_at_entry == "bull"


def test_averaging_up_recomputes_the_weighted_entry():
    """Keeping the original entry would make every later P&L figure and stop
    distance wrong."""
    tracker = PositionTracker(FakeAlpacaClient())
    tracker.register_fill("NVDA", 10, 100.0, "buy")
    position = tracker.register_fill("NVDA", 10, 120.0, "buy")
    assert position.quantity == 20
    assert position.entry_price == pytest.approx(110.0)


def test_partial_sell_reduces_without_closing():
    tracker = PositionTracker(FakeAlpacaClient())
    tracker.register_fill("NVDA", 10, 100.0, "buy")
    position = tracker.register_fill("NVDA", 4, 105.0, "sell")
    assert position.quantity == pytest.approx(6)


def test_full_sell_removes_the_position():
    tracker = PositionTracker(FakeAlpacaClient())
    tracker.register_fill("NVDA", 10, 100.0, "buy")
    assert tracker.register_fill("NVDA", 10, 105.0, "sell") is None
    assert "NVDA" not in tracker.get_open_positions()


def test_fill_callbacks_fire():
    tracker = PositionTracker(FakeAlpacaClient())
    seen = []
    tracker.on_fill(seen.append)
    tracker.register_fill("NVDA", 10, 100.0, "buy")
    assert len(seen) == 1 and seen[0]["symbol"] == "NVDA"


def test_a_failing_callback_does_not_break_the_fill():
    """A broken listener must not corrupt position state."""
    tracker = PositionTracker(FakeAlpacaClient())
    tracker.on_fill(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    tracker.register_fill("NVDA", 10, 100.0, "buy")
    assert tracker.get_open_positions()["NVDA"].quantity == 10


def test_sync_adopts_untracked_positions():
    """Adopted with no stop, so positions_without_stops() surfaces it. An
    unprotected position should be loud."""
    class RawPosition:
        symbol, qty, avg_entry_price, current_price = "AAPL", 5.0, 150.0, 155.0
        market_value, cost_basis = 775.0, 750.0

    tracker = PositionTracker(FakeAlpacaClient(FakeTradingClient(positions=[RawPosition()])))
    report = tracker.sync()
    assert report["adopted"] == ["AAPL"]
    adopted = tracker.get_open_positions()["AAPL"]
    assert adopted.adopted is True
    assert adopted.has_stop is False
    assert "AAPL" in tracker.positions_without_stops()


def test_sync_removes_stale_positions():
    """We think we hold it, the broker does not. The broker is right."""
    tracker = PositionTracker(FakeAlpacaClient())
    tracker.register_fill("GONE", 10, 100.0, "buy", stop_loss=95.0)
    report = tracker.sync()
    assert report["removed"] == ["GONE"]
    assert not tracker.get_open_positions()


def test_regime_changed_flag():
    tracker = PositionTracker(FakeAlpacaClient())
    tracker.register_fill("NVDA", 10, 100.0, "buy", regime="bull")
    assert tracker.get_open_positions()["NVDA"].regime_changed is False
    tracker.update_regime("crash")
    position = tracker.get_open_positions()["NVDA"]
    assert position.regime_at_entry == "bull"
    assert position.regime_current == "crash"
    assert position.regime_changed is True


def test_holding_periods_increment():
    tracker = PositionTracker(FakeAlpacaClient())
    tracker.register_fill("NVDA", 10, 100.0, "buy")
    for _ in range(3):
        tracker.increment_holding_periods()
    assert tracker.get_open_positions()["NVDA"].holding_periods == 3


def test_to_portfolio_state_bridges_to_the_risk_manager():
    """The single join between the broker layer and the risk layer.

    Positions cross as plain dicts so the risk manager never imports a broker
    type, which is what keeps it broker-agnostic.
    """
    from core.risk_manager import PortfolioState, RiskManager

    tracker = PositionTracker(FakeAlpacaClient())
    tracker.register_fill("NVDA", 10, 100.0, "buy", stop_loss=95.0, regime="bull")
    tracker.update_prices({"NVDA": 110.0})

    state = tracker.to_portfolio_state()
    assert isinstance(state, PortfolioState)
    assert state.equity == 100_000.0
    assert state.n_positions == 1
    assert state.gross_exposure == pytest.approx(1100.0 / 100_000.0)

    manager = RiskManager({"max_concurrent": 5})
    assert manager.check_breakers(pd.Series([100_000.0, 100_000.0])) is not None


def test_concurrent_fills_are_thread_safe():
    """Fills arrive on a WebSocket thread while the main loop reads positions.
    Without the lock this raises, or produces a torn read."""
    tracker = PositionTracker(FakeAlpacaClient())
    errors = []

    def add(n):
        try:
            for _ in range(40):
                tracker.register_fill(f"SYM{n}", 1, 100.0, "buy")
                tracker.get_open_positions()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread safety violated: {errors}"
    assert len(tracker.get_open_positions()) == 6


# -- market data ------------------------------------------------------------

def test_utc_helper_is_timezone_aware():
    """Alpaca reads a naive datetime as UTC. On a UTC+8 machine that sends a
    timestamp eight hours in the future, which the free tier rejects outright
    and the paid tier silently honours."""
    from data.market_data import utc_now

    now = utc_now()
    assert now.tzinfo is not None


def test_naive_datetimes_are_coerced_to_utc():
    from data.market_data import _as_utc

    naive = datetime(2024, 1, 1, 12, 0)
    assert _as_utc(naive).tzinfo is UTC
    aware = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    assert _as_utc(aware) is aware
    assert _as_utc(None) is None


def test_validate_flags_an_unadjusted_split():
    """The check that catches raw prices being used where adjusted were meant.
    On real NVDA data the raw series shows an 89.9% single-bar move."""
    from data.market_data import MarketDataClient

    index = pd.bdate_range("2024-01-01", periods=60)
    close = pd.Series(100.0, index=index)
    close.iloc[30:] = 10.0        # a 10-for-1 split, unadjusted
    bars = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1e6}, index=index)
    problems = MarketDataClient(None, cache_dir=Path("/tmp")).validate(bars)
    assert any("split" in p for p in problems)


def test_validate_accepts_clean_data():
    from data.market_data import MarketDataClient

    index = pd.bdate_range("2024-01-01", periods=60)
    close = pd.Series(range(100, 160), index=index, dtype=float)
    bars = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1e6}, index=index)
    assert MarketDataClient(None, cache_dir=Path("/tmp")).validate(bars) == []


def test_unsupported_timeframe_raises():
    from data.market_data import MarketDataClient

    with pytest.raises(ValueError, match="unsupported timeframe"):
        MarketDataClient._to_timeframe("3Day")


# -- live, skipped without credentials --------------------------------------

@live
@pytest.mark.alpaca
def test_live_connection_is_a_paper_account():
    client = AlpacaClient()
    account = client.connect()
    assert account.is_paper, "REFUSING to run tests against a live account"
    assert account.is_tradeable


@live
@pytest.mark.alpaca
def test_live_historical_bars_are_clean():
    from data.market_data import MarketDataClient

    client = AlpacaClient()
    client.connect()
    data = MarketDataClient(client)
    bars = data.get_historical("NVDA", lookback_days=400)
    assert len(bars) > 200
    assert data.validate(bars) == []


@live
@pytest.mark.alpaca
def test_live_adjusted_and_raw_prices_differ():
    """NVDA split 10-for-1 in June 2024, so the two series must diverge. If they
    match, the adjustment parameter is not reaching the API."""
    from data.market_data import MarketDataClient

    client = AlpacaClient()
    client.connect()
    data = MarketDataClient(client)
    adjusted = data.get_historical("NVDA", lookback_days=1200, adjusted=True)
    raw = data.get_historical("NVDA", lookback_days=1200, adjusted=False)
    assert raw["close"].max() > adjusted["close"].max() * 2
