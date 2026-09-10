"""
Phase 7: the main loop.

Everything here runs offline. The engine takes its broker, market data, logger
and alert manager as constructor arguments precisely so the loop can be driven
without a network, an account or a wall clock.

The tests that matter most are not the happy path. They are:

- `test_peak_equity_survives_restart`, because losing it silently disarms the
  one breaker that never resets
- `test_shutdown_leaves_positions_open_and_audits_stops`, because shutdown makes
  a promise about stops that has to be checked, not assumed
- `test_reduction_does_not_consult_the_risk_manager`, because routing exits
  through a veto is how a system gets trapped in a position it wanted out of
- the dry-run tests, because "no orders were placed" must be a property of the
  object graph, not of remembering to check a flag
"""

import copy
import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

import main as engine_module
from broker.alpaca_client import Account, Order, OrderSide, OrderStatus, OrderType, Position
from core.regime_strategies import Direction, Signal
from core.risk_manager import BreakerType, RejectionReason, RiskDecision
from main import (
    BarOutcome,
    BrokerUnavailable,
    EngineError,
    SessionState,
    TradingEngine,
    _is_daily,
    _RefusingExecutor,
    _timeframe_minutes,
    _week_start,
    build_parser,
)
from monitoring.alerts import AlertLevel, AlertManager
from monitoring.dashboard import DashboardState, TerminalDashboard
from monitoring.logger import EventType, TradingLogger

# ===========================================================================
# Fakes
# ===========================================================================

class FakeClock:
    def __init__(self, is_open=True):
        self.is_open = is_open


class FakeBroker:
    """The broker surface the engine actually touches."""

    def __init__(self, equity=100_000.0, positions=None, market_open=True,
                 open_orders=None, last_equity=None):
        self.equity = equity
        self.last_equity = last_equity if last_equity is not None else equity
        self._positions = list(positions or [])
        self.market_open = market_open
        self._open_orders = list(open_orders or [])
        self.submitted = []
        self.closed_all = 0
        self.connect_calls = 0
        self.fail_calls = 0

    def connect(self):
        self.connect_calls += 1
        return self.get_account()

    def get_account(self):
        if self.fail_calls > 0:
            self.fail_calls -= 1
            raise RuntimeError("alpaca unreachable")
        return Account(
            equity=self.equity, cash=self.equity, buying_power=self.equity * 2,
            portfolio_value=self.equity, is_paper=True, status="ACTIVE",
            last_equity=self.last_equity,
        )

    def get_clock(self):
        if self.fail_calls > 0:
            self.fail_calls -= 1
            raise RuntimeError("alpaca unreachable")
        return {"is_open": self.market_open, "timestamp": datetime.now(UTC),
                "next_open": datetime.now(UTC) + timedelta(hours=12),
                "next_close": datetime.now(UTC) + timedelta(hours=6)}

    def get_positions(self):
        return list(self._positions)

    def get_open_orders(self):
        return list(self._open_orders)

    @property
    def trading_client(self):
        return self

    def submit_order(self, request):
        self.submitted.append(request)
        return request

    def to_order(self, raw):
        return Order(
            order_id="ord-1", symbol=getattr(raw, "symbol", "SPY"),
            side=OrderSide.SELL, quantity=float(getattr(raw, "qty", 1)),
            filled_quantity=0.0, status=OrderStatus.OPEN, order_type=OrderType.MARKET,
            limit_price=None, stop_price=None, average_fill_price=None,
            submitted_at=datetime.now(UTC), filled_at=None,
        )


class FakeMarketData:
    def __init__(self, bars, quote=None, fail=False):
        self._bars = bars
        self._quote = quote or {"bid": 99.9, "ask": 100.1, "tradeable": True}
        self.fail = fail

    def get_training_window(self, symbol, n_bars=954):
        if self.fail:
            raise RuntimeError("data feed down")
        return self._bars

    def get_latest_quote(self, symbol):
        return dict(self._quote)

    def validate(self, bars):
        return []

    def stop_stream(self):
        pass


class RecordingExecutor:
    """Order executor that records rather than submitting."""

    def __init__(self, fail_trailing=False, fail_oco=False):
        self.orders = []
        self.stops = []
        self.trailing = []
        self.oco = []
        self.modified = []
        self.closed = []
        self.closed_all = []
        self.fail_trailing = fail_trailing
        self.fail_oco = fail_oco

    def submit_order(self, signal, decision, **kwargs):
        from broker.order_executor import TradeRecord

        trade = TradeRecord(
            trade_id=f"t{len(self.orders)}", symbol=signal.symbol, side=OrderSide.BUY,
            requested_qty=decision.approved_quantity, approved_qty=decision.approved_quantity,
            signal_reasoning=signal.reasoning, risk_modifications=list(decision.modifications),
            regime=signal.regime_name, regime_confidence=signal.regime_probability,
            stop_loss=signal.stop_loss, take_profit=None,
            filled_qty=decision.approved_quantity, status=OrderStatus.FILLED,
        )
        self.orders.append(trade)
        return trade

    def place_stop(self, symbol, quantity, stop_price, trade_id=None):
        self.stops.append((symbol, quantity, stop_price))
        return Order(
            order_id=f"stop-{symbol}", symbol=symbol, side=OrderSide.SELL,
            quantity=quantity, filled_quantity=0.0, status=OrderStatus.OPEN,
            order_type=OrderType.STOP, limit_price=None, stop_price=stop_price,
            average_fill_price=None, submitted_at=None, filled_at=None,
        )

    def place_trailing_stop(self, symbol, quantity, trail_percent, trade_id=None,
                            fallback_stop=None):
        if self.fail_trailing:
            raise RuntimeError("trailing stop rejected by the broker")
        self.trailing.append((symbol, quantity, trail_percent))
        return Order(
            order_id=f"trail-{symbol}", symbol=symbol, side=OrderSide.SELL,
            quantity=quantity, filled_quantity=0.0, status=OrderStatus.OPEN,
            order_type=OrderType.TRAILING_STOP, limit_price=None, stop_price=None,
            average_fill_price=None, submitted_at=None, filled_at=None,
        )

    def place_oco_exit(self, symbol, quantity, take_profit, stop_price, trade_id=None):
        if self.fail_oco:
            raise RuntimeError("OCO rejected by the broker")
        self.oco.append((symbol, quantity, take_profit, stop_price))
        return Order(
            order_id=f"oco-{symbol}", symbol=symbol, side=OrderSide.SELL,
            quantity=quantity, filled_quantity=0.0, status=OrderStatus.OPEN,
            order_type=OrderType.LIMIT, limit_price=take_profit,
            stop_price=stop_price, average_fill_price=None,
            submitted_at=None, filled_at=None,
        )

    def protect_position(self, symbol, quantity, *, trail_percent=None,
                         stop_price=None, take_profit=None, trade_id=None):
        """Mirrors the real fallback chain: OCO, then trailing, then a stop."""
        if take_profit is not None and stop_price is not None:
            try:
                return self.place_oco_exit(symbol, quantity, take_profit,
                                           stop_price, trade_id=trade_id)
            except Exception:
                pass
        if trail_percent is not None:
            try:
                return self.place_trailing_stop(symbol, quantity, trail_percent,
                                                trade_id=trade_id)
            except Exception:
                if stop_price is None:
                    raise
        return self.place_stop(symbol, quantity, stop_price, trade_id=trade_id)

    def modify_stop(self, symbol, new_stop):
        self.modified.append((symbol, new_stop))
        return self.place_stop(symbol, 1, new_stop)

    def close_position(self, symbol):
        self.closed.append(symbol)
        return None

    def close_all_positions(self, reason=""):
        self.closed_all.append(reason)
        return []


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def engine_settings(settings, tmp_path):
    config = json.loads(json.dumps(settings, default=str))
    config["broker"]["symbols"] = ["SPY"]
    config["orchestration"]["state_snapshot_path"] = str(tmp_path / "state_snapshot.json")
    config["orchestration"]["broker_retry_base_delay"] = 0.0
    config["orchestration"]["poll_seconds"] = 0.01
    # Tests drive equity through the fake broker, so the sizing override has to
    # be off by default or every drawdown assertion measures the override
    # instead. TestAccountSizeOverride turns it on explicitly.
    config["risk"]["account_size_override"] = None
    return config


@pytest.fixture
def model(fitted_engine):
    """A private copy of the fitted model.

    `fitted_engine` is session-scoped, and several tests here age the model
    or bump its bar counter to drive the retrain rules. Mutating the shared
    object would make every later test refit the HMM from scratch.
    """
    return copy.deepcopy(fitted_engine)


@pytest.fixture
def built_engine(engine_settings, tmp_path, synthetic_bars, model):
    """A started engine with every collaborator faked."""
    broker = FakeBroker()
    market_data = FakeMarketData(synthetic_bars)
    log = TradingLogger(log_dir=tmp_path / "logs").setup()
    alerts = AlertManager({}, rate_limit_minutes=0, sink=lambda *a: None)

    engine = TradingEngine(
        engine_settings, client=broker, market_data=market_data,
        trading_logger=log, alerts=alerts,
        snapshot_path=tmp_path / "state_snapshot.json",
        lock_file=tmp_path / "trading_halted.lock",
        model_path=tmp_path / "model.pkl",
    )
    engine.started_at = datetime.now(UTC)
    engine.account = broker.get_account()
    engine.hmm = model
    engine._init_risk_manager()
    engine._init_position_tracker()
    engine.order_executor = RecordingExecutor()
    engine._restore_session_state()
    engine._init_orchestrator()
    engine.dashboard_state = DashboardState(
        engine.position_tracker, engine.risk_manager, engine.hmm, log, engine
    )
    engine.refresh_portfolio()
    return engine


def make_position(symbol="SPY", qty=100.0, price=100.0):
    return Position(
        symbol=symbol, quantity=qty, average_entry_price=price, current_price=price,
        market_value=qty * price, cost_basis=qty * price,
        unrealised_pnl=0.0, unrealised_pnl_pct=0.0,
    )


# ===========================================================================
# SessionState
# ===========================================================================

def test_session_state_round_trip(tmp_path):
    path = tmp_path / "state.json"
    state = SessionState(
        session_id="s1", peak_equity=120_000.0, day_start_equity=100_000.0,
        daily_trades=4, bars_processed=17, stops={"SPY": 401.25},
        breaker_daily_tripped="daily_reduce",
    )
    state.save(path)
    restored = SessionState.load(path)

    assert restored.peak_equity == 120_000.0
    assert restored.stops == {"SPY": 401.25}
    assert restored.daily_trades == 4
    assert restored.breaker_daily_tripped == "daily_reduce"
    assert restored.saved_at


def test_session_state_write_is_atomic(tmp_path):
    """No .tmp file is left behind, and the target is only ever complete."""
    path = tmp_path / "state.json"
    SessionState(peak_equity=1.0).save(path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(path.read_text())["peak_equity"] == 1.0


def test_corrupt_snapshot_does_not_crash_startup(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ this is not json")
    assert SessionState.load(path) is None


def test_snapshot_ignores_unknown_fields(tmp_path):
    """A snapshot written by a later version must not break an older engine."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"peak_equity": 5.0, "a_field_from_the_future": 1}))
    assert SessionState.load(path).peak_equity == 5.0


def test_missing_snapshot_returns_none(tmp_path):
    assert SessionState.load(tmp_path / "nope.json") is None


# ===========================================================================
# Recovery: the part that silently disarms a breaker if it is wrong
# ===========================================================================

def test_peak_equity_survives_restart(engine_settings, tmp_path, model):
    """Restarting after a drawdown must not re-base the peak.

    The system peaked at 120k and is now at 90k, a 25% drawdown that should have
    the peak breaker halted. If the restart took the current equity as the new
    peak, drawdown_from_peak would read 0% and the breaker could never fire
    again. A crash-restart loop would then be a way to disable it.
    """
    snapshot = tmp_path / "state.json"
    SessionState(peak_equity=120_000.0, day_start_equity=120_000.0,
                 day_start_date=date.today().isoformat(),
                 week_start_date=_week_start(date.today()).isoformat()).save(snapshot)

    broker = FakeBroker(equity=90_000.0)
    engine = TradingEngine(engine_settings, client=broker,
                           market_data=FakeMarketData(pd.DataFrame()),
                           trading_logger=TradingLogger(log_dir=tmp_path / "l").setup(),
                           alerts=AlertManager({}, sink=lambda *a: None),
                           snapshot_path=snapshot, lock_file=tmp_path / "lock")
    engine.started_at = datetime.now(UTC)
    engine.account = broker.get_account()
    engine.hmm = model
    engine._init_risk_manager()
    engine._init_position_tracker()
    engine._restore_session_state()

    assert engine.session.peak_equity == 120_000.0

    portfolio = engine.refresh_portfolio()
    assert portfolio.drawdown_from_peak == pytest.approx(-0.25)
    assert engine.risk_manager.breaker.check(portfolio) is BreakerType.PEAK_HALT


def test_peak_equity_rises_to_current_after_a_deposit(engine_settings, tmp_path, model):
    """max(saved, current), so new money raises the peak rather than being
    treated as profit that can later be drawn down against."""
    snapshot = tmp_path / "state.json"
    SessionState(peak_equity=100_000.0, day_start_date=date.today().isoformat(),
                 week_start_date=_week_start(date.today()).isoformat()).save(snapshot)

    broker = FakeBroker(equity=150_000.0)
    engine = TradingEngine(engine_settings, client=broker,
                           market_data=FakeMarketData(pd.DataFrame()),
                           trading_logger=TradingLogger(log_dir=tmp_path / "l").setup(),
                           alerts=AlertManager({}, sink=lambda *a: None),
                           snapshot_path=snapshot, lock_file=tmp_path / "lock")
    engine.started_at = datetime.now(UTC)
    engine.account = broker.get_account()
    engine.hmm = model
    engine._init_risk_manager()
    engine._init_position_tracker()
    engine._restore_session_state()

    assert engine.session.peak_equity == 150_000.0


def test_latched_breakers_survive_restart(engine_settings, tmp_path, model):
    """A restart must not be a way to clear a latched breaker."""
    snapshot = tmp_path / "state.json"
    SessionState(peak_equity=100_000.0, breaker_daily_tripped="daily_reduce",
                 breaker_weekly_tripped="weekly_reduce", breaker_peak_tripped=False,
                 day_start_date=date.today().isoformat(),
                 week_start_date=_week_start(date.today()).isoformat()).save(snapshot)

    broker = FakeBroker()
    engine = TradingEngine(engine_settings, client=broker,
                           market_data=FakeMarketData(pd.DataFrame()),
                           trading_logger=TradingLogger(log_dir=tmp_path / "l").setup(),
                           alerts=AlertManager({}, sink=lambda *a: None),
                           snapshot_path=snapshot, lock_file=tmp_path / "lock")
    engine.started_at = datetime.now(UTC)
    engine.account = broker.get_account()
    engine.hmm = model
    engine._init_risk_manager()
    engine._init_position_tracker()
    engine._restore_session_state()

    assert engine.risk_manager.breaker.daily_tripped is BreakerType.DAILY_REDUCE
    assert engine.risk_manager.breaker.weekly_tripped is BreakerType.WEEKLY_REDUCE
    assert engine.risk_manager.breaker.size_multiplier == 0.5


def test_stops_are_restored_onto_adopted_positions(engine_settings, tmp_path, model):
    """A position adopted from the broker comes back with no stop attached.

    Without restoring the snapshot's stops, every restart would report its own
    positions as unprotected and, worse, would believe it.
    """
    snapshot = tmp_path / "state.json"
    SessionState(peak_equity=100_000.0, stops={"SPY": 95.5},
                 day_start_date=date.today().isoformat(),
                 week_start_date=_week_start(date.today()).isoformat()).save(snapshot)

    broker = FakeBroker(positions=[make_position("SPY", 100, 100.0)])
    engine = TradingEngine(engine_settings, client=broker,
                           market_data=FakeMarketData(pd.DataFrame()),
                           trading_logger=TradingLogger(log_dir=tmp_path / "l").setup(),
                           alerts=AlertManager({}, sink=lambda *a: None),
                           snapshot_path=snapshot, lock_file=tmp_path / "lock")
    engine.started_at = datetime.now(UTC)
    engine.account = broker.get_account()
    engine.hmm = model
    engine._init_risk_manager()
    engine._init_position_tracker()

    assert engine.position_tracker.positions["SPY"].stop_loss is None   # adopted, no stop
    engine._restore_session_state()
    assert engine.position_tracker.positions["SPY"].stop_loss == 95.5


def test_day_rollover_resets_the_daily_baseline(built_engine):
    engine = built_engine
    engine.session.day_start_date = (date.today() - timedelta(days=1)).isoformat()
    engine.session.day_start_equity = 80_000.0
    engine.session.daily_trades = 9
    engine.risk_manager.breaker.daily_tripped = BreakerType.DAILY_HALT

    engine._roll_periods(date.today(), 100_000.0)

    assert engine.session.day_start_equity == 100_000.0
    assert engine.session.daily_trades == 0
    assert engine.risk_manager.breaker.daily_tripped is BreakerType.NONE


def test_week_rollover_resets_the_weekly_baseline(built_engine):
    engine = built_engine
    engine.session.week_start_date = (date.today() - timedelta(days=14)).isoformat()
    engine.risk_manager.breaker.weekly_tripped = BreakerType.WEEKLY_REDUCE

    engine._roll_periods(date.today(), 100_000.0)

    assert engine.session.week_start_date == _week_start(date.today()).isoformat()
    assert engine.risk_manager.breaker.weekly_tripped is BreakerType.NONE


def test_week_start_is_monday():
    assert _week_start(date(2026, 9, 7)).weekday() == 0
    assert _week_start(date(2026, 9, 13)) == date(2026, 9, 7)   # Sunday belongs to its Monday


# ===========================================================================
# Retraining
# ===========================================================================

def test_retrain_when_no_model(built_engine):
    retrain, why = built_engine.needs_retrain(engine=None)
    assert retrain and "no fitted model" in why


def test_retrain_when_model_is_older_than_seven_days(built_engine, model):
    model.metadata.training_date = datetime.now(UTC) - timedelta(days=8)
    model._bars_since_fit = 0
    retrain, why = built_engine.needs_retrain(model)
    assert retrain and "days old" in why


def test_no_retrain_for_a_fresh_model(built_engine, model):
    model.metadata.training_date = datetime.now(UTC) - timedelta(days=1)
    model._bars_since_fit = 0
    retrain, _ = built_engine.needs_retrain(model)
    assert not retrain


def test_retrain_on_bar_count_even_when_calendar_is_fresh(built_engine, model):
    """Both rules apply. The calendar rule alone would skip a refit across a
    holiday break; the bar rule alone would let a model go stale while halted."""
    model.metadata.training_date = datetime.now(UTC)
    model._bars_since_fit = model.retrain_interval_bars + 1
    retrain, why = built_engine.needs_retrain(model)
    assert retrain and "bars since fit" in why


def test_fit_records_an_aware_utc_training_date(model):
    """The Phase 6 lesson: naive datetimes are how you get an eight-hour error
    on a UTC+8 machine. The fit path must not produce one."""
    assert model.metadata.training_date.tzinfo is not None


def test_naive_training_date_is_read_as_utc_not_local(built_engine, model):
    """Fallback for a model pickled by an older version.

    Read as UTC, so the age is right regardless of the machine's timezone.
    Reading it as local time would make the model look up to a day younger or
    older depending on where the process happens to run.
    """
    naive_utc = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=3)
    model.metadata.training_date = naive_utc
    assert built_engine.model_age_days(model) == pytest.approx(3.0, abs=0.01)


def test_retrain_rebuilds_the_strategy_map(built_engine, monkeypatch):
    """A refit renumbers the states. If the orchestrator keeps the old
    regime_info, it maps new state ids through old volatility ranks and every
    allocation is silently wrong."""
    engine = built_engine
    engine.hmm.metadata.training_date = datetime.now(UTC) - timedelta(days=30)

    rebuilt = {}
    monkeypatch.setattr(engine.orchestrator, "update_regime_infos",
                        lambda infos: rebuilt.update(infos))
    monkeypatch.setattr(engine, "train_model", lambda **kw: engine.hmm)

    assert engine._maybe_retrain() is True
    assert rebuilt == engine.hmm.regime_info
    assert engine.previous_regime is None


def test_failed_retrain_keeps_the_previous_model(built_engine, monkeypatch):
    engine = built_engine
    previous = engine.hmm
    engine.hmm.metadata.training_date = datetime.now(UTC) - timedelta(days=30)

    def boom(**kwargs):
        raise RuntimeError("no data")

    monkeypatch.setattr(engine, "train_model", boom)
    assert engine._maybe_retrain() is False
    assert engine.hmm is previous


# ===========================================================================
# Dry run: the capability is removed, not merely unused
# ===========================================================================

def test_refusing_executor_raises_on_every_method():
    executor = _RefusingExecutor()
    for method in ("submit_order", "place_stop", "close_all_positions", "modify_stop"):
        with pytest.raises(EngineError, match="dry run"):
            getattr(executor, method)()
    assert len(executor.attempts) == 4


def test_dry_run_engine_gets_a_refusing_executor(engine_settings, tmp_path, synthetic_bars):
    broker = FakeBroker()
    engine = TradingEngine(engine_settings, dry_run=True, client=broker,
                           market_data=FakeMarketData(synthetic_bars),
                           trading_logger=TradingLogger(log_dir=tmp_path / "l").setup(),
                           alerts=AlertManager({}, sink=lambda *a: None),
                           snapshot_path=tmp_path / "s.json", lock_file=tmp_path / "lock")
    engine.account = broker.get_account()
    engine._init_position_tracker()
    assert isinstance(engine.order_executor, _RefusingExecutor)


def test_dry_run_places_no_orders(built_engine, monkeypatch):
    engine = built_engine
    engine.dry_run = True
    engine.mode = "dry-run"
    engine.order_executor = _RefusingExecutor()

    outcome = engine.process_bar()

    assert engine.order_executor.attempts == []
    assert outcome.errors == []


# ===========================================================================
# The loop
# ===========================================================================

def test_process_bar_runs_the_whole_pipeline(built_engine):
    outcome = built_engine.process_bar()

    assert outcome.timestamp is not None
    assert outcome.regime != "unknown"
    assert 0.0 <= outcome.confidence <= 1.0
    assert outcome.target_allocation > 0
    assert outcome.errors == []
    assert built_engine.bars_processed == 1


def test_the_same_bar_is_not_processed_twice(built_engine):
    first = built_engine.process_bar()
    second = built_engine.process_bar()

    assert first.skipped != "bar already processed"
    assert second.skipped == "bar already processed"
    assert built_engine.bars_processed == 1


def test_state_is_saved_after_each_bar(built_engine):
    built_engine.process_bar()
    saved = SessionState.load(built_engine.snapshot_path)

    assert saved is not None
    assert saved.bars_processed == 1
    assert saved.last_regime == built_engine.regime_state.label.value
    assert saved.peak_equity >= 100_000.0


def test_no_rebalance_inside_the_threshold(built_engine, monkeypatch):
    """Without this gate the system re-submits the same order every bar and
    pays slippage for a drift that changes nothing."""
    engine = built_engine
    monkeypatch.setattr(engine.orchestrator, "needs_rebalance", lambda t, c: False)

    outcome = engine.process_bar()

    assert outcome.rebalanced is False
    assert outcome.submitted == 0
    assert "no rebalance" in outcome.skipped


def test_increase_routes_through_the_risk_manager(built_engine, monkeypatch):
    engine = built_engine
    seen = []
    original = engine.risk_manager.validate_signal

    def spy(signal, portfolio, **kwargs):
        seen.append(signal.symbol)
        return original(signal, portfolio, **kwargs)

    monkeypatch.setattr(engine.risk_manager, "validate_signal", spy)
    monkeypatch.setattr(engine.orchestrator, "needs_rebalance", lambda t, c: True)
    monkeypatch.setattr(engine.orchestrator, "target_allocation", lambda *a, **k: 0.95)

    engine.process_bar()
    assert seen, "an increase must be validated before it is submitted"


def test_rejected_signals_are_logged_and_not_submitted(built_engine, monkeypatch):
    engine = built_engine
    monkeypatch.setattr(engine.orchestrator, "needs_rebalance", lambda t, c: True)
    monkeypatch.setattr(engine.orchestrator, "target_allocation", lambda *a, **k: 0.95)
    monkeypatch.setattr(
        engine.risk_manager, "validate_signal",
        lambda *a, **k: RiskDecision.reject(RejectionReason.CIRCUIT_BREAKER, "nope"),
    )

    outcome = engine.process_bar()

    assert outcome.rejected > 0
    assert outcome.submitted == 0
    assert engine.order_executor.orders == []
    events = [e for e in engine.log.read_events() if e["event"] == "signal_rejected"]
    assert events and events[0]["rejection_reason"] == "circuit_breaker"


def test_approved_signal_is_submitted_with_a_stop(built_engine, monkeypatch):
    engine = built_engine
    monkeypatch.setattr(engine.orchestrator, "needs_rebalance", lambda t, c: True)
    monkeypatch.setattr(engine.orchestrator, "target_allocation", lambda *a, **k: 0.95)

    outcome = engine.process_bar()

    if outcome.submitted:
        assert engine.order_executor.orders
        # Every filled entry gets protection in the same cycle. Which kind is
        # config; that there is one is the rule.
        executor = engine.order_executor
        assert executor.stops or executor.trailing or executor.oco, \
            "a filled entry got no protective exit"
        if executor.stops:
            symbol, _qty, stop = executor.stops[0]
            assert stop < engine.bars[symbol]["close"].iloc[-1]


def test_a_filled_entry_gets_a_trailing_stop_when_enabled(built_engine, monkeypatch):
    engine = built_engine
    # No target, so the OCO route is not taken and the trailing stop is.
    engine.order_executor.fail_oco = True
    monkeypatch.setattr(engine.orchestrator, "needs_rebalance", lambda t, c: True)
    monkeypatch.setattr(engine.orchestrator, "target_allocation", lambda *a, **k: 0.95)

    outcome = engine.process_bar()
    if not outcome.submitted:
        pytest.skip("no order was submitted on this bar")

    assert engine.order_executor.trailing, "trailing stop is enabled but was not used"
    _symbol, _qty, trail = engine.order_executor.trailing[0]
    assert 1.5 <= trail <= 15.0, f"trail {trail}% is outside the configured band"


def test_a_rejected_trailing_stop_falls_back_to_a_fixed_one(built_engine, monkeypatch):
    """A filled position must never sit without a stop. "The trailing stop
    request errored" is not a reason to leave one naked."""
    engine = built_engine
    engine.order_executor.fail_trailing = True
    engine.order_executor.fail_oco = True
    monkeypatch.setattr(engine.orchestrator, "needs_rebalance", lambda t, c: True)
    monkeypatch.setattr(engine.orchestrator, "target_allocation", lambda *a, **k: 0.95)

    outcome = engine.process_bar()
    if not outcome.submitted:
        pytest.skip("no order was submitted on this bar")

    assert engine.order_executor.trailing == [], "the trailing stop should have failed"
    assert engine.order_executor.stops, "the fallback fixed stop was not placed"


def test_the_trail_percent_tracks_atr_not_a_fixed_number(built_engine):
    """5% is four ATR on SPY and half an ATR on COIN. One number is either too
    tight to hold a position or too loose to protect it."""
    from core.regime_strategies import Direction, Signal

    def signal_with(atr, price):
        return Signal(
            symbol="X", direction=Direction.LONG, confidence=0.9, entry_price=price,
            stop_loss=price * 0.9, take_profit=None, position_size_pct=0.1,
            leverage=1.0, regime_id=0, regime_name="strong_bull", regime_probability=0.9,
            timestamp=pd.Timestamp("2026-09-04"), reasoning="t",
            strategy_name="s", metadata={"atr": atr},
        )

    quiet = built_engine._trail_percent(signal_with(atr=2.0, price=770.0))
    wild = built_engine._trail_percent(signal_with(atr=12.0, price=185.0))
    assert wild > quiet, "a more volatile name should get a wider trail"

    # Both clamped into the configured band, whatever ATR says.
    assert built_engine._trail_percent(signal_with(atr=0.01, price=770.0)) == 1.5
    assert built_engine._trail_percent(signal_with(atr=500.0, price=100.0)) == 15.0


def test_reduction_does_not_consult_the_risk_manager(built_engine, monkeypatch):
    """Reducing risk never needs permission.

    A rejection on this path would leave the system holding an allocation the
    regime no longer supports, at exactly the moment it wanted out.
    """
    engine = built_engine
    engine.client._positions = [make_position("SPY", 100, 100.0)]
    engine.position_tracker.sync()
    engine.refresh_portfolio()

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("the reduce path must not consult validate_signal")

    monkeypatch.setattr(engine.risk_manager, "validate_signal", must_not_be_called)

    outcome = BarOutcome()
    engine._reduce_to_target(target=0.05, current=0.10, outcome=outcome)

    assert outcome.submitted == 1
    assert engine.client.submitted, "a sell order should have been sent"


def test_reduction_to_zero_closes_the_position(built_engine):
    engine = built_engine
    engine.client._positions = [make_position("SPY", 100, 100.0)]
    engine.position_tracker.sync()
    engine.session.stops["SPY"] = 95.0

    outcome = BarOutcome()
    engine._reduce_to_target(target=0.0, current=0.10, outcome=outcome)

    assert engine.order_executor.closed == ["SPY"]
    assert "SPY" not in engine.session.stops


def test_reduction_is_proportional_across_names(built_engine):
    engine = built_engine
    engine.client._positions = [
        make_position("SPY", 100, 100.0), make_position("QQQ", 50, 100.0)
    ]
    engine.position_tracker.sync()

    outcome = BarOutcome()
    engine._reduce_to_target(target=0.5, current=1.0, outcome=outcome)

    sold = {r.symbol: float(r.qty) for r in engine.client.submitted}
    assert sold == {"SPY": 50, "QQQ": 25}


# ===========================================================================
# Error handling
# ===========================================================================

def test_hmm_error_holds_the_current_regime(built_engine, monkeypatch):
    engine = built_engine
    engine.process_bar()
    held = engine.regime_state

    def boom(*args, **kwargs):
        raise RuntimeError("singular covariance")

    monkeypatch.setattr(engine.hmm, "classify", boom)
    engine.session.last_bar_timestamp = ""

    outcome = engine.process_bar()

    assert engine.regime_state is held
    assert outcome.regime == held.label.value
    assert any("hmm" in e for e in outcome.errors)


def test_broker_retry_backs_off_then_gives_up(built_engine):
    engine = built_engine
    calls = []

    def always_fails():
        calls.append(1)
        raise RuntimeError("timeout")

    with pytest.raises(BrokerUnavailable):
        engine._broker_retry(always_fails, what="test call")

    assert len(calls) == int(engine.orch["broker_retry_attempts"])
    assert engine.data_feed_healthy is False


def test_broker_retry_recovers_within_budget(built_engine):
    engine = built_engine
    engine.client.fail_calls = 2

    account = engine._broker_retry(engine.client.get_account, what="account")

    assert account.equity == 100_000.0
    assert engine.data_feed_healthy is True


def test_data_feed_drop_pauses_signals_but_leaves_stops(built_engine):
    """The spec's rule. Signals stop; nothing cancels a resting stop order."""
    engine = built_engine
    engine.client._positions = [make_position("SPY", 100, 100.0)]
    engine.position_tracker.sync()
    engine.position_tracker.update_stop("SPY", 95.0)
    engine.data_feed_healthy = False

    outcome = engine.process_bar()

    assert "data feed unhealthy" in outcome.skipped
    assert outcome.submitted == 0
    assert engine.order_executor.closed == []
    assert engine.order_executor.closed_all == []
    assert engine.position_tracker.positions["SPY"].stop_loss == 95.0
    # One cautious cycle, then trading resumes.
    assert engine.data_feed_healthy is True


def test_market_data_failure_is_caught_not_raised(built_engine):
    engine = built_engine
    engine.market_data.fail = True

    outcome = engine.process_bar()

    assert outcome.skipped == "broker unavailable"
    assert engine.consecutive_errors == 1


def test_error_budget_stops_the_loop(built_engine):
    engine = built_engine
    engine.market_data.fail = True
    limit = int(engine.orch["max_consecutive_errors"])

    for _ in range(limit):
        engine.process_bar()

    assert engine.consecutive_errors >= limit
    assert engine.running is False


def test_a_good_bar_resets_the_error_count(built_engine):
    engine = built_engine
    engine.market_data.fail = True
    engine.process_bar()
    assert engine.consecutive_errors == 1

    engine.market_data.fail = False
    engine.process_bar()
    assert engine.consecutive_errors == 0


def test_unhandled_error_is_logged_and_alerted(built_engine, monkeypatch):
    engine = built_engine
    sent = []
    engine.alerts._sink = lambda level, subject, body: sent.append(subject)

    monkeypatch.setattr(engine, "_fetch_bars", lambda: (_ for _ in ()).throw(ValueError("boom")))
    outcome = engine.process_bar()

    assert "boom" in outcome.errors[0]
    assert any("Unhandled error" in s for s in sent)
    errors = [e for e in engine.log.read_events() if e["event"] == "error"]
    assert errors and errors[-1]["traceback"]


# ===========================================================================
# Circuit breakers and halting
# ===========================================================================


def _below_peak(engine, peak: float) -> float:
    """Equity far enough below `peak` to trip the peak halt at any setting."""
    return peak * (1 - engine.risk_config["max_dd_from_peak"] - 0.05)


def _below_daily(engine, day_start: float) -> float:
    """Equity far enough below the day's open to trip the daily halt."""
    return day_start * (1 - engine.risk_config["daily_dd_halt"] - 0.01)


def test_halt_closes_positions(built_engine):
    """A halt is the one path that liquidates. Shutdown is not."""
    engine = built_engine
    # Comfortably past whatever peak threshold is configured. Hardcoding 20%
    # coupled this to a 10% setting, and re-tuning the policy broke four tests
    # that are about what a halt DOES, not about where it triggers.
    engine.client.equity = _below_peak(engine, 100_000.0)
    engine.refresh_portfolio()

    outcome = BarOutcome(regime="strong_bear")
    state = engine._check_breakers(outcome)

    assert state == "halted"
    assert engine.order_executor.closed_all, "a halt must close positions"
    assert engine.risk_manager.is_halted()
    assert engine.running is False


def test_halt_writes_the_lock_file(built_engine):
    engine = built_engine
    engine.client.equity = _below_peak(engine, 100_000.0)
    engine.refresh_portfolio()
    engine._check_breakers(BarOutcome())

    assert engine.risk_manager.breaker.lock_file.exists()


def test_breaker_state_is_persisted(built_engine):
    engine = built_engine
    engine.client.equity = _below_daily(engine, 100_000.0)
    engine.session.day_start_equity = 100_000.0
    engine.refresh_portfolio()

    engine._check_breakers(BarOutcome())

    assert engine.session.breaker_daily_tripped != "none"


def test_halted_engine_submits_nothing(built_engine):
    engine = built_engine
    engine.risk_manager.halt("test halt")

    outcome = engine.process_bar()

    assert outcome.skipped == "halted"
    assert outcome.submitted == 0
    assert engine.order_executor.orders == []


def test_breakers_ignore_the_regime(built_engine):
    """Phase 5's separation, re-checked at the loop level: the breaker reads
    realised P&L and nothing about what the model believes."""
    engine = built_engine
    engine.client.equity = _below_peak(engine, 100_000.0)
    engine.refresh_portfolio()

    for _regime in ("strong_bull", "strong_bear", "unknown"):
        engine.risk_manager.breaker.peak_tripped = False
        assert engine.risk_manager.breaker.check(engine.portfolio) is BreakerType.PEAK_HALT


# ===========================================================================
# Stops and shutdown
# ===========================================================================

def test_audit_stops_finds_naked_positions(built_engine):
    engine = built_engine
    engine.client._positions = [make_position("SPY", 100, 100.0)]
    engine.position_tracker.sync()

    unprotected = engine.audit_stops(alert=False)
    assert unprotected == ["SPY"]


def test_audit_stops_flags_a_local_stop_with_no_resting_order(built_engine):
    """A stop_loss float in memory protects nothing once the process exits."""
    engine = built_engine
    engine.client._positions = [make_position("SPY", 100, 100.0)]
    engine.position_tracker.sync()
    engine.position_tracker.update_stop("SPY", 95.0)
    engine.client._open_orders = []            # nothing resting at the broker

    unprotected = engine.audit_stops(alert=False)
    assert unprotected == ["SPY"]


def test_audit_stops_passes_when_a_resting_order_exists(built_engine):
    engine = built_engine
    engine.client._positions = [make_position("SPY", 100, 100.0)]
    engine.position_tracker.sync()
    engine.position_tracker.update_stop("SPY", 95.0)
    engine.client._open_orders = [
        Order(order_id="s1", symbol="SPY", side=OrderSide.SELL, quantity=100,
              filled_quantity=0, status=OrderStatus.OPEN, order_type=OrderType.STOP,
              limit_price=None, stop_price=95.0, average_fill_price=None,
              submitted_at=None, filled_at=None)
    ]
    assert engine.audit_stops(alert=False) == []


def test_missing_stop_raises_an_alert(built_engine):
    engine = built_engine
    sent = []
    engine.alerts._sink = lambda level, subject, body: sent.append(subject)
    engine.client._positions = [make_position("SPY", 100, 100.0)]
    engine.position_tracker.sync()

    engine.audit_stops(alert=True)
    assert any("no stop" in s.lower() for s in sent)


def test_stops_only_tighten(built_engine):
    engine = built_engine
    engine.client._positions = [make_position("SPY", 100, 100.0)]
    engine.position_tracker.sync()
    engine.bars = {"SPY": engine.market_data.get_training_window("SPY")}
    engine.position_tracker.update_stop("SPY", 1e9)      # absurdly tight already

    state = engine.hmm.classify(_features(engine))
    updated = engine._update_stops(state)

    assert updated == 0
    assert engine.order_executor.modified == []


def _features(engine):
    from data.feature_engineering import build_feature_matrix

    return build_feature_matrix(engine.market_data.get_training_window("SPY"))


def test_shutdown_leaves_positions_open_and_audits_stops(built_engine):
    """Shutdown must not liquidate, and must check the promise it relies on."""
    engine = built_engine
    engine.client._positions = [make_position("SPY", 100, 100.0)]
    engine.position_tracker.sync()
    sent = []
    engine.alerts._sink = lambda level, subject, body: sent.append(subject)

    engine.shutdown("test")

    assert engine.order_executor.closed_all == [], "shutdown must not close positions"
    assert engine.order_executor.closed == []
    assert engine.position_tracker.positions["SPY"].quantity == 100
    assert any("no stop" in s.lower() for s in sent), "an unprotected position must be loud"


def test_shutdown_saves_state(built_engine):
    engine = built_engine
    engine.process_bar()
    engine.shutdown("test")

    saved = SessionState.load(engine.snapshot_path)
    assert saved.bars_processed == 1
    assert saved.saved_at


def test_shutdown_is_idempotent(built_engine):
    engine = built_engine
    engine.shutdown("first")
    engine.shutdown("second")
    events = [e for e in engine.log.read_events() if e["event"] == "system_shutdown"]
    assert len(events) == 1


def test_shutdown_survives_a_failed_snapshot_write(built_engine, monkeypatch):
    engine = built_engine
    monkeypatch.setattr(engine.session, "save",
                        lambda path: (_ for _ in ()).throw(OSError("disk full")))
    engine.shutdown("test")     # must not raise


# ===========================================================================
# Market hours
# ===========================================================================

def test_run_exits_immediately_when_closed_and_configured_to_exit(built_engine):
    engine = built_engine
    engine.client.market_open = False
    engine.orch["market_closed_action"] = "exit"

    assert engine.run() == 0
    assert engine.bars_processed == 0


def test_market_closed_records_the_next_open(built_engine):
    engine = built_engine
    engine.client.market_open = False

    assert engine._market_is_open() is False
    assert engine.next_open is not None


def test_broker_down_reads_as_closed_rather_than_open(built_engine):
    """Fail safe. An unreachable clock must never be interpreted as "open"."""
    engine = built_engine
    engine.client.fail_calls = 99
    assert engine._market_is_open() is False


# ===========================================================================
# Timeframe helpers
# ===========================================================================

@pytest.mark.parametrize("value,expected", [
    ("1Day", True), ("day", True), ("1D", True), ("daily", True),
    ("5Min", False), ("1Hour", False), ("15Min", False),
])
def test_is_daily(value, expected):
    assert _is_daily(value) is expected


@pytest.mark.parametrize("value,minutes", [
    ("5Min", 5), ("15Min", 15), ("1Hour", 60), ("2H", 120), ("nonsense", 5),
])
def test_timeframe_minutes(value, minutes):
    assert _timeframe_minutes(value) == minutes


# ===========================================================================
# Logging and alerts
# ===========================================================================

def test_every_event_lands_on_disk_as_json(tmp_path):
    log = TradingLogger(log_dir=tmp_path).setup()
    log.log_event(EventType.SYSTEM_START, "up", equity=100.0)
    log.log_event(EventType.ORDER_FILLED, "fill", symbol="SPY")

    events = list(log.read_events())
    assert [e["event"] for e in events] == ["system_start", "order_filled"]
    assert events[0]["equity"] == 100.0


def test_log_survives_unserialisable_fields(tmp_path):
    log = TradingLogger(log_dir=tmp_path).setup()
    record = log.log_event(EventType.ERROR, "x", when=pd.Timestamp("2026-01-01"),
                           side=OrderSide.BUY, blob=object())
    assert record["when"].startswith("2026-01-01")
    assert record["side"] == "buy"
    assert isinstance(record["blob"], str)


def test_rejections_are_logged_as_prominently_as_approvals(tmp_path):
    log = TradingLogger(log_dir=tmp_path).setup()
    signal = Signal(symbol="SPY", direction=Direction.LONG, confidence=0.9,
                    entry_price=100.0, stop_loss=95.0, take_profit=None,
                    position_size_pct=0.6, leverage=1.0, regime_id=0,
                    regime_name="bull", regime_probability=0.9,
                    timestamp=pd.Timestamp("2026-01-01"), reasoning="r",
                    strategy_name="S")
    log.log_signal(signal, RiskDecision.reject(RejectionReason.NO_STOP_LOSS, "no stop"))

    feed = log.get_trade_log()
    assert feed[0]["event"] == "signal_rejected"
    assert feed[0]["rejection_reason"] == "no_stop_loss"


def test_alert_rate_limit_suppresses_repeats():
    sent = []
    alerts = AlertManager({}, rate_limit_minutes=15, sink=lambda level, s, b: sent.append(s))
    assert alerts.send(AlertLevel.WARNING, "same", "body") is True
    assert alerts.send(AlertLevel.WARNING, "same", "body") is False
    assert len(sent) == 1
    assert alerts.suppressed == 1


def test_critical_alerts_use_a_shorter_window():
    ticks = [0.0]
    alerts = AlertManager({}, rate_limit_minutes=15, sink=lambda *a: None,
                          clock=lambda: ticks[0])
    alerts.send(AlertLevel.CRITICAL, "halt", "b")
    ticks[0] = 6 * 60          # 6 minutes: past 15/3, not past 15
    assert alerts.send(AlertLevel.CRITICAL, "halt", "b") is True


def test_alert_delivery_failure_never_raises():
    def broken(level, subject, body):
        raise RuntimeError("smtp down")

    alerts = AlertManager({}, sink=broken)
    assert alerts.send(AlertLevel.CRITICAL, "s", "b") is True


# ===========================================================================
# Dashboard
# ===========================================================================

def test_dashboard_snapshot_has_every_panel(built_engine):
    built_engine.process_bar()
    snapshot = built_engine.dashboard_state.snapshot()

    assert set(snapshot) >= {"regime", "portfolio", "positions", "risk", "signals", "session"}
    assert snapshot["regime"]["regime"] != "unknown"
    assert snapshot["portfolio"]["equity"] == 100_000.0
    assert "from_peak" in snapshot["risk"]["drawdowns"]


def test_dashboard_cannot_place_orders(built_engine):
    """A display that can act is no longer a display."""
    state = built_engine.dashboard_state
    assert not hasattr(state, "order_executor")
    assert not any("submit" in name or "close" in name for name in dir(state))


def test_dashboard_renders_without_crashing(built_engine):
    built_engine.process_bar()
    TerminalDashboard().render(built_engine.dashboard_state.snapshot())


# ===========================================================================
# CLI
# ===========================================================================

def test_cli_defaults_to_trading():
    args = build_parser().parse_args([])
    assert args.mode == "trade" and not args.dry_run


@pytest.mark.parametrize("argv,attr", [
    (["--dry-run"], "dry_run"),
    (["--train-only"], "train_only"),
    (["--dashboard"], "dashboard"),
    (["--stress-test"], "stress_test"),
    (["--compare"], "compare"),
    (["--backtest"], "backtest"),
])
def test_cli_flags_from_the_spec(argv, attr):
    assert getattr(build_parser().parse_args(argv), attr) is True


def test_cli_backtest_invocation_from_the_tutorial():
    args = build_parser().parse_args(
        ["--mode", "backtest", "--symbols", "SPY", "--start", "2019-01-01",
         "--end", "2024-12-31"]
    )
    assert args.mode == "backtest"
    assert args.symbols == ["SPY"]
    assert args.start == "2019-01-01"


def test_status_mode_touches_no_network(capsys):
    assert engine_module.main(["--status"]) == 0
    assert "phase 7" in capsys.readouterr().out.lower()


def test_live_requires_an_explicit_flag():
    args = build_parser().parse_args([])
    assert args.i_understand_live is False


# ===========================================================================
# Account size override
# ===========================================================================

class TestAccountSizeOverride:
    """Sizing against a $100,000 paper balance produces orders nobody can place.

    The override makes the risk path behave as if the account held the real
    intended capital. Getting it half-right is worse than not having it: scale
    one field and leave another, and the breakers start measuring a drawdown
    that never happened.
    """

    def test_sizing_uses_the_override_not_the_broker_balance(self, built_engine):
        engine = built_engine
        engine.risk_config["account_size_override"] = 10_000
        assert engine.client.get_account().equity == 100_000.0
        assert engine.sizing_equity == 10_000.0

    def test_no_override_uses_the_real_balance(self, built_engine):
        built_engine.risk_config["account_size_override"] = None
        assert built_engine.sizing_equity == 100_000.0

    def test_every_monetary_field_scales_together(self, built_engine):
        """Scaling equity but not last_equity seeded a phantom -90% daily
        drawdown and halted the system on its first bar."""
        engine = built_engine
        engine.risk_config["account_size_override"] = 10_000
        scaled = engine._scaled_account()

        assert scaled.equity == 10_000.0
        assert scaled.last_equity == pytest.approx(10_000.0)
        assert scaled.cash == pytest.approx(10_000.0)
        assert scaled.buying_power == pytest.approx(20_000.0)

    def test_ratios_are_unchanged_by_the_override(self, built_engine):
        """A breaker must fire at the same percentage either way."""
        engine = built_engine
        engine.risk_config["account_size_override"] = 10_000
        engine.session.day_start_equity = 0.0
        engine.session.peak_equity = 0.0
        engine._restore_session_state()
        portfolio = engine.refresh_portfolio()

        assert portfolio.drawdown_daily == pytest.approx(0.0)
        assert portfolio.drawdown_from_peak == pytest.approx(0.0)

    def test_positions_are_affordable_at_the_override(self, built_engine):
        """The whole point: orders a real account could actually place."""
        engine = built_engine
        engine.risk_config["account_size_override"] = 10_000
        engine.risk_manager = None
        engine._init_risk_manager()
        engine._restore_session_state()
        engine.refresh_portfolio()

        signal = Signal(
            symbol="SPY", direction=Direction.LONG, confidence=0.9,
            entry_price=514.20, stop_loss=508.00, take_profit=None,
            position_size_pct=0.95, leverage=1.0, regime_id=0,
            regime_name="strong_bull", regime_probability=0.9,
            timestamp=pd.Timestamp("2026-09-04"), reasoning="sizing test",
            strategy_name="LowVolBullStrategy",
        )
        decision = engine.risk_manager.validate_signal(signal, engine.portfolio)

        if decision.approved:
            assert decision.approved_notional <= 10_000 * 0.15 + 1, \
                "position exceeds the single-position cap on the scaled account"

    def test_changing_the_override_rebases_rather_than_halting(self, built_engine, tmp_path):
        """A config edit must not read as a 90% loss.

        Restoring a $100,000 peak into a $10,000 account is a -90% drawdown,
        which trips the peak breaker and writes a lock file that has to be
        deleted by hand. For a settings change, not a loss.
        """
        snapshot = tmp_path / "state.json"
        SessionState(
            peak_equity=100_000.0, day_start_equity=100_000.0,
            week_start_equity=100_000.0, equity_at_save=100_000.0,
            sizing_basis=100_000.0,
            day_start_date=date.today().isoformat(),
            week_start_date=_week_start(date.today()).isoformat(),
            equity_history=[{"t": "2026-09-01", "equity": 100_000.0, "peak": 100_000.0}],
        ).save(snapshot)

        engine = built_engine
        engine.snapshot_path = snapshot
        engine.risk_config["account_size_override"] = 10_000
        engine._restore_session_state()

        assert engine.session.peak_equity == pytest.approx(10_000.0)
        assert engine.session.sizing_basis == pytest.approx(10_000.0)
        assert engine.session.equity_history[0]["equity"] == pytest.approx(10_000.0)

        portfolio = engine.refresh_portfolio()
        assert portfolio.drawdown_from_peak == pytest.approx(0.0)
        assert engine.risk_manager.breaker.check(portfolio) is BreakerType.NONE

    def test_a_genuine_drawdown_survives_the_rebase(self, built_engine, tmp_path):
        """Rescaling must preserve the ratio, not erase the loss."""
        snapshot = tmp_path / "state.json"
        SessionState(
            peak_equity=200_000.0,          # account was down 50% from peak
            day_start_equity=100_000.0, week_start_equity=100_000.0,
            equity_at_save=100_000.0, sizing_basis=100_000.0,
            day_start_date=date.today().isoformat(),
            week_start_date=_week_start(date.today()).isoformat(),
        ).save(snapshot)

        engine = built_engine
        engine.snapshot_path = snapshot
        engine.risk_config["account_size_override"] = 10_000
        engine._restore_session_state()

        assert engine.session.peak_equity == pytest.approx(20_000.0)
        portfolio = engine.refresh_portfolio()
        assert portfolio.drawdown_from_peak == pytest.approx(-0.5), \
            "the rebase erased a real 50% drawdown"


def test_adopted_position_with_a_broker_stop_is_protected(built_engine):
    """The regression that made the dashboard read "updated 37h ago".

    A position adopted from the broker comes back with no stop_loss attached,
    because Alpaca does not store one on the position. If the audit trusts only
    local memory, every restart reports every position as naked. The broker
    check used to be able to add to that list but never to clear it, so a real
    resting stop could not clear the flag, every scheduled run was stamped
    'failed', and the dashboard fell back to the last genuinely clean run.
    """
    engine = built_engine
    engine.client._positions = [make_position("SPY", 100, 100.0)]
    engine.position_tracker.sync()
    assert engine.position_tracker.positions["SPY"].stop_loss is None, "adopted flat"

    engine.client._open_orders = [
        Order(order_id="s1", symbol="SPY", side=OrderSide.SELL, quantity=100,
              filled_quantity=0, status=OrderStatus.OPEN, order_type=OrderType.STOP,
              limit_price=None, stop_price=95.0, average_fill_price=None,
              submitted_at=None, filled_at=None)
    ]

    assert engine.audit_stops(alert=False) == []
    # ...and the level is adopted too, not merely the fact that one exists.
    assert engine.position_tracker.positions["SPY"].stop_loss == 95.0


def test_unprotected_positions_do_not_fail_the_run(built_engine, tmp_path):
    """A naked position is a risk finding, not a broken job.

    Conflating the two is what made the dashboard treat a healthy run as stale.
    """
    from data.repository import open_repository

    engine = built_engine
    engine.repo = open_repository(tmp_path / "runs.db")
    engine.repo.migrate()
    engine.run_id = engine.repo.start_run(mode="paper", trigger="test")

    engine._close_run("refresh", unprotected=["SPY", "QQQ"])

    row = engine.repo.conn.execute(
        "SELECT status, error FROM runs WHERE id = ?", (engine.run_id,)).fetchone()
    assert row["status"] == "ok"
    # 'refresh' is how the loop exited, not a fault. It must not be logged as one.
    assert row["error"] is None


def test_a_real_error_still_fails_the_run(built_engine, tmp_path):
    from data.repository import open_repository

    engine = built_engine
    engine.repo = open_repository(tmp_path / "runs2.db")
    engine.repo.migrate()
    engine.run_id = engine.repo.start_run(mode="paper", trigger="test")
    engine.consecutive_errors = 3
    engine.last_error = "data feed unreachable"

    engine._close_run("shutdown", unprotected=[])

    row = engine.repo.conn.execute(
        "SELECT status, error FROM runs WHERE id = ?", (engine.run_id,)).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "data feed unreachable"
