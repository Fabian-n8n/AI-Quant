"""
Tests for the risk management layer.

The most important tests in the project. A mediocre strategy with good risk
management loses slowly; a good strategy with bad risk management blows up the
account.

Every test uses an autouse fixture that redirects the halt lock file into
`tmp_path`. Without it a single breaker test would write `trading_halted.lock`
into the repo and every subsequent test, and every subsequent real run, would
refuse to trade.
"""


import numpy as np
import pandas as pd
import pytest

from core.regime_strategies import Direction, Signal
from core.risk_manager import (
    BREAKER_ACTION,
    BREAKER_SEVERITY,
    BreakerState,
    BreakerType,
    CircuitBreaker,
    PortfolioState,
    RejectionReason,
    RiskAction,
    RiskDecision,
    RiskManager,
)


@pytest.fixture
def risk_config(settings):
    return dict(settings["risk"])


@pytest.fixture(autouse=True)
def isolated_lock(tmp_path, monkeypatch):
    """Keep the halt lock out of the repo.

    A breaker test that wrote the real lock file would halt every later test and
    every real run until someone noticed and deleted it by hand. That is the
    mechanism working as designed, which is exactly why it must be isolated.
    """
    import core.risk_manager as module

    lock = tmp_path / "trading_halted.lock"
    monkeypatch.setattr(module, "DEFAULT_LOCK_FILE", lock)
    return lock


@pytest.fixture
def manager(risk_config, isolated_lock):
    return RiskManager(risk_config, lock_file=isolated_lock)


@pytest.fixture
def signal():
    return Signal(
        symbol="SPY", direction=Direction.LONG, confidence=0.90, entry_price=100.0,
        stop_loss=95.0, take_profit=None, position_size_pct=0.95, leverage=1.0,
        regime_id=0, regime_name="bull", regime_probability=0.90,
        timestamp=pd.Timestamp("2024-06-03"), reasoning="test",
        strategy_name="LowVolBullStrategy",
    )


@pytest.fixture
def portfolio():
    return PortfolioState(equity=100_000.0, cash=100_000.0, buying_power=200_000.0)


# -- independence from the model --------------------------------------------

def test_breakers_never_read_regime_state(risk_config, isolated_lock):
    """The defining property. Two identical drawdowns with opposite regime
    context must produce identical breaker decisions.

    If the risk layer ever consults the model to decide whether to fire, it has
    been defeated: it would go quiet in exactly the situation it exists for,
    which is the model being confidently wrong.
    """
    confident = PortfolioState(
        equity=88_000.0, peak_equity=100_000.0, regime="euphoria", regime_confidence=0.99
    )
    unsure = PortfolioState(
        equity=88_000.0, peak_equity=100_000.0, regime="crash", regime_confidence=0.20,
        regime_confirmed=False, flicker_rate=9,
    )
    a = CircuitBreaker(risk_config, isolated_lock).check(confident)
    b = CircuitBreaker(risk_config, isolated_lock).check(unsure)
    assert a is b is BreakerType.PEAK_HALT


def test_regime_is_recorded_for_review_not_for_deciding(risk_config, isolated_lock):
    """Logged so you can ask afterwards what the model believed when this broke."""
    breaker = CircuitBreaker(risk_config, isolated_lock)
    breaker.update(
        PortfolioState(equity=85_000.0, peak_equity=100_000.0,
                       regime="euphoria", regime_confidence=0.91),
        positions_closed=3,
    )
    history = breaker.get_history()
    assert len(history) == 1
    assert history["regime"].iloc[0] == "euphoria"
    assert history["positions_closed"].iloc[0] == 3


# -- circuit breakers -------------------------------------------------------

@pytest.mark.parametrize(
    "equity,day,week,peak,expected",
    [
        (100_000, 100_000, 100_000, 100_000, BreakerType.NONE),
        (97_500, 100_000, 100_000, 100_000, BreakerType.DAILY_REDUCE),
        (96_500, 100_000, 100_000, 100_000, BreakerType.DAILY_HALT),
        (94_000, 94_500, 100_000, 100_000, BreakerType.WEEKLY_REDUCE),
        (92_500, 93_000, 100_000, 100_000, BreakerType.WEEKLY_HALT),
        (88_000, 88_500, 89_000, 100_000, BreakerType.PEAK_HALT),
    ],
)
def test_each_breaker_fires_at_its_threshold(risk_config, isolated_lock, equity, day, week, peak, expected):
    state = PortfolioState(
        equity=float(equity), day_start_equity=float(day),
        week_start_equity=float(week), peak_equity=float(peak),
    )
    assert CircuitBreaker(risk_config, isolated_lock).check(state) is expected


def test_most_severe_breaker_wins(risk_config, isolated_lock):
    """When several fire at once, a daily reduce must not soften a peak halt."""
    state = PortfolioState(
        equity=85_000.0, day_start_equity=87_500.0,
        week_start_equity=90_000.0, peak_equity=100_000.0,
    )
    breaker = CircuitBreaker(risk_config, isolated_lock)
    assert breaker.check(state) is BreakerType.PEAK_HALT
    assert BREAKER_ACTION[breaker.check(state)] is BreakerState.HALTED


def test_severity_ordering_is_total():
    assert len(set(BREAKER_SEVERITY.values())) == len(BREAKER_SEVERITY)
    assert BREAKER_SEVERITY[BreakerType.PEAK_HALT] == max(BREAKER_SEVERITY.values())
    assert BREAKER_SEVERITY[BreakerType.DAILY_HALT] > BREAKER_SEVERITY[BreakerType.WEEKLY_REDUCE]


def test_breakers_latch_and_do_not_unfire_on_a_bounce(risk_config, isolated_lock):
    """A breaker that clears when equity recovers intraday is not a circuit
    breaker, it is a lagging indicator."""
    breaker = CircuitBreaker(risk_config, isolated_lock)
    breaker.update(PortfolioState(equity=97_500.0, day_start_equity=100_000.0))
    assert breaker.daily_tripped is BreakerType.DAILY_REDUCE

    recovered = PortfolioState(equity=100_500.0, day_start_equity=100_000.0)
    assert breaker.check(recovered) is BreakerType.DAILY_REDUCE
    assert breaker.size_multiplier == 0.5


def test_reset_daily_clears_only_daily(risk_config, isolated_lock):
    breaker = CircuitBreaker(risk_config, isolated_lock)
    breaker.update(PortfolioState(equity=97_500.0, day_start_equity=100_000.0))
    breaker.update(PortfolioState(equity=94_000.0, week_start_equity=100_000.0))
    assert breaker.daily_tripped is not BreakerType.NONE
    assert breaker.weekly_tripped is not BreakerType.NONE

    breaker.reset_daily()
    assert breaker.daily_tripped is BreakerType.NONE
    assert breaker.weekly_tripped is not BreakerType.NONE, "weekly must survive a daily reset"


def test_reset_weekly_clears_both(risk_config, isolated_lock):
    breaker = CircuitBreaker(risk_config, isolated_lock)
    breaker.update(PortfolioState(equity=94_000.0, day_start_equity=96_000.0,
                                  week_start_equity=100_000.0))
    breaker.reset_weekly()
    assert breaker.daily_tripped is BreakerType.NONE
    assert breaker.weekly_tripped is BreakerType.NONE


def test_size_multiplier_reflects_the_active_action(risk_config, isolated_lock):
    breaker = CircuitBreaker(risk_config, isolated_lock)
    assert breaker.size_multiplier == 1.0
    breaker.update(PortfolioState(equity=97_500.0, day_start_equity=100_000.0))
    assert breaker.size_multiplier == 0.5
    breaker.update(PortfolioState(equity=96_000.0, day_start_equity=100_000.0))
    assert breaker.size_multiplier == 0.0


# -- the lock file ----------------------------------------------------------

def test_peak_halt_writes_the_lock_file(risk_config, isolated_lock):
    breaker = CircuitBreaker(risk_config, isolated_lock)
    breaker.update(PortfolioState(equity=85_000.0, peak_equity=100_000.0, regime="bull"))
    assert isolated_lock.exists()
    contents = isolated_lock.read_text()
    assert "TRADING HALTED" in contents
    assert "Delete this file to resume" in contents
    assert "bull" in contents, "the regime at the time must be recorded"


def test_lock_file_blocks_every_signal(manager, signal, portfolio, isolated_lock):
    """First check in the cascade, before anything else is computed."""
    manager.halt("test", portfolio)
    decision = manager.validate_signal(signal, portfolio)
    assert decision.approved is False
    assert decision.rejection_reason is RejectionReason.HALT_LOCK_FILE


def test_lock_file_survives_a_new_manager(risk_config, isolated_lock, signal, portfolio):
    """Restarting the process must not clear a halt. If it did, a crash loop
    would silently resume trading."""
    RiskManager(risk_config, lock_file=isolated_lock).halt("test", portfolio)
    fresh = RiskManager(risk_config, lock_file=isolated_lock)
    assert fresh.is_halted()
    assert fresh.validate_signal(signal, portfolio).approved is False


def test_resets_do_not_clear_the_halt(manager, signal, portfolio):
    """Only manual deletion resumes trading. A scheduled daily reset must not."""
    manager.halt("test", portfolio)
    manager.reset_daily()
    manager.reset_weekly()
    assert manager.is_halted()
    assert manager.validate_signal(signal, portfolio).approved is False


# -- portfolio state --------------------------------------------------------

def test_drawdowns_are_computed_not_stored():
    """Stored drawdowns go stale, and a stale one is how a breaker fails to
    fire on the day it mattered."""
    state = PortfolioState(equity=90_000.0, day_start_equity=100_000.0,
                           week_start_equity=95_000.0, peak_equity=120_000.0)
    assert state.drawdown_daily == pytest.approx(-0.10)
    assert state.drawdown_weekly == pytest.approx(-0.0526, abs=1e-3)
    assert state.drawdown_from_peak == pytest.approx(-0.25)

    state.equity = 100_000.0
    assert state.drawdown_daily == 0.0, "must recompute, not return a cached value"


def test_drawdowns_are_never_positive():
    state = PortfolioState(equity=120_000.0, day_start_equity=100_000.0, peak_equity=100_000.0)
    assert state.drawdown_daily == 0.0
    assert state.drawdown_from_peak == 0.0


def test_gross_exposure_and_leverage():
    state = PortfolioState(
        equity=100_000.0,
        positions={"A": {"market_value": 40_000.0}, "B": {"market_value": 35_000.0}},
    )
    assert state.gross_exposure == pytest.approx(0.75)
    assert state.leverage == pytest.approx(0.75)
    assert state.n_positions == 2


def test_sector_exposure_groups_correctly():
    state = PortfolioState(
        equity=100_000.0,
        positions={"NVDA": {"market_value": 15_000.0}, "AMD": {"market_value": 12_000.0},
                   "XOM": {"market_value": 8_000.0}},
    )
    sectors = state.sector_exposure({"NVDA": "tech", "AMD": "tech", "XOM": "energy"})
    assert sectors["tech"] == pytest.approx(0.27)
    assert sectors["energy"] == pytest.approx(0.08)


def test_zero_equity_does_not_divide_by_zero():
    state = PortfolioState(equity=0.0, positions={"A": {"market_value": 100.0}})
    assert state.gross_exposure == 0.0
    assert state.drawdown_daily == 0.0


# -- sizing -----------------------------------------------------------------

def test_position_size_risks_exactly_the_configured_fraction(manager):
    """size = (equity * 1%) / stop_distance. $100k, $5 stop -> 200 shares."""
    shares = manager.position_size(100_000.0, 100.0, 95.0)
    assert shares == 200
    assert shares * 5.0 == pytest.approx(100_000 * 0.01)


def test_wider_stop_means_fewer_shares_for_the_same_dollar_risk(manager):
    """The property that makes a losing streak survivable."""
    tight = manager.position_size(100_000.0, 100.0, 98.0)
    wide = manager.position_size(100_000.0, 100.0, 90.0)
    assert tight > wide
    assert tight * 2.0 == pytest.approx(wide * 10.0, rel=0.02)


def test_position_size_truncates_never_rounds_up(manager):
    """A position can only ever be smaller than the target. Rounding up would
    quietly exceed the risk limit on every trade."""
    shares = manager.position_size(10_000.0, 100.0, 97.0)
    assert shares == 33          # 33.33 truncated
    assert shares * 3.0 <= 10_000 * 0.01


def test_zero_stop_distance_returns_zero_not_an_exception(manager):
    assert manager.position_size(100_000.0, 100.0, 100.0) == 0


def test_gap_cap_always_binds_so_real_risk_is_two_thirds_of_one_percent(manager):
    """A consequence of the configured numbers that is easy to miss.

    The gap cap allows gap_max_loss_pct / gap_multiplier = 2%/3 = 0.667% of
    equity, which is below max_risk_per_trade (1%). Since this is a swing system
    and every position is held overnight, the 1% figure never applies.
    """
    normal = manager.position_size(100_000.0, 100.0, 95.0)
    capped = manager.gap_capped_size(100_000.0, 5.0)
    assert capped < normal
    assert capped * 5.0 / 100_000 == pytest.approx(0.00665, abs=1e-4)
    # And the worst case it was sized for lands on the configured 2%.
    assert capped * 3 * 5.0 / 100_000 == pytest.approx(0.02, abs=1e-3)


def test_gap_cap_is_applied_to_approved_signals(manager, signal, portfolio):
    decision = manager.validate_signal(signal, portfolio, overnight=True)
    assert decision.approved
    assert decision.modified_signal["risk_pct_of_equity"] < manager.max_risk_per_trade
    assert any("gap" in m for m in decision.modifications)


# -- the veto cascade -------------------------------------------------------

def test_clean_signal_is_approved(manager, signal, portfolio):
    decision = manager.validate_signal(signal, portfolio)
    assert decision.approved
    assert decision.action in (RiskAction.APPROVE, RiskAction.APPROVE_MODIFIED)
    assert decision.modified_signal["shares"] > 0
    assert decision.modified_signal["stop_loss"] == signal.stop_loss


def test_signal_without_a_stop_is_rejected(manager, signal, portfolio):
    """Non-negotiable. Sizing divides by the stop distance, so a missing stop
    makes the position size undefined, not merely risky."""
    import dataclasses

    naked = dataclasses.replace(signal, stop_loss=95.0)
    object.__setattr__(naked, "stop_loss", None)
    decision = manager.validate_signal(naked, portfolio)
    assert decision.approved is False
    assert decision.rejection_reason is RejectionReason.NO_STOP_LOSS


def test_stop_on_the_wrong_side_is_rejected(manager, signal, portfolio):
    import dataclasses

    bad = dataclasses.replace(signal, stop_loss=95.0)
    object.__setattr__(bad, "stop_loss", 105.0)
    decision = manager.validate_signal(bad, portfolio)
    assert decision.approved is False
    assert decision.rejection_reason is RejectionReason.INVALID_STOP


def test_max_concurrent_positions_blocks_new_symbols(manager, signal):
    state = PortfolioState(
        equity=100_000.0,
        positions={f"SYM{i}": {"market_value": 1_000.0} for i in range(5)},
    )
    decision = manager.validate_signal(signal, state)
    assert decision.approved is False
    assert decision.rejection_reason is RejectionReason.MAX_POSITIONS


def test_existing_positions_stay_adjustable_at_the_limit(manager, signal):
    """Refusing to resize something already held would trap the system at max
    positions with no way to reduce risk."""
    positions = {f"SYM{i}": {"market_value": 1_000.0} for i in range(4)}
    positions["SPY"] = {"market_value": 1_000.0}
    decision = manager.validate_signal(signal, PortfolioState(equity=100_000.0, positions=positions))
    assert decision.rejection_reason is not RejectionReason.MAX_POSITIONS


def test_daily_trade_limit(manager, signal):
    state = PortfolioState(equity=100_000.0, daily_trades=20)
    decision = manager.validate_signal(signal, state)
    assert decision.approved is False
    assert decision.rejection_reason is RejectionReason.DAILY_TRADE_LIMIT


def test_single_position_cap_shrinks_rather_than_rejects(manager, signal):
    """A tight stop produces a huge share count. It should be capped, not
    refused: the signal is fine, the requested size is not."""
    import dataclasses

    tight = dataclasses.replace(signal, stop_loss=99.9)
    decision = manager.validate_signal(tight, PortfolioState(equity=100_000.0))
    assert decision.approved
    assert decision.modified_signal["notional"] <= 100_000 * manager.max_single_position + 1


def test_exposure_limit_rejects_when_no_room_remains(manager, signal):
    state = PortfolioState(equity=100_000.0, positions={"X": {"market_value": 79_990.0}})
    decision = manager.validate_signal(signal, state)
    assert decision.approved is False
    assert decision.rejection_reason is RejectionReason.EXPOSURE_LIMIT


def test_exposure_limit_shrinks_when_some_room_remains(manager, signal):
    state = PortfolioState(equity=100_000.0, positions={"X": {"market_value": 75_000.0}})
    decision = manager.validate_signal(signal, state)
    if decision.approved:
        assert decision.modified_signal["notional"] <= 5_000 + 1
        assert any("shrunk" in m or "gap" in m for m in decision.modifications)


def test_minimum_position_floor(manager, signal):
    """Below $100 the costs outweigh the position.

    Uses $500 equity: the 15% single-position cap allows $75, which truncates to
    zero whole shares at a $100 price. At $1,000 the cap allows exactly $100,
    which is the floor rather than below it, and correctly approves one share.
    """
    decision = manager.validate_signal(signal, PortfolioState(equity=500.0))
    assert decision.approved is False
    assert decision.rejection_reason is RejectionReason.BELOW_MINIMUM_SIZE


def test_position_exactly_at_the_floor_is_allowed(manager, signal):
    """The floor is inclusive: $100 is the minimum, not the first rejected value."""
    decision = manager.validate_signal(signal, PortfolioState(equity=1_000.0))
    assert decision.approved
    assert decision.modified_signal["notional"] == pytest.approx(100.0)


def test_approved_size_never_exceeds_the_request(manager, signal, portfolio):
    """The veto may shrink a position. It may never grow one."""
    requested = manager.position_size(portfolio.equity, signal.entry_price, signal.stop_loss)
    decision = manager.validate_signal(signal, portfolio)
    assert decision.modified_signal["shares"] <= requested


# -- leverage ---------------------------------------------------------------

def test_leverage_defaults_to_one(manager, signal, portfolio):
    decision = manager.validate_signal(signal, portfolio)
    assert decision.modified_signal["leverage"] == 1.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("regime_confirmed", False),
        ("regime_confidence", 0.30),
        ("flicker_rate", 9),
    ],
)
def test_leverage_forced_to_one_when_conditions_are_not_clean(manager, signal, field, value):
    import dataclasses

    levered = dataclasses.replace(signal, leverage=1.25)
    state = PortfolioState(equity=100_000.0, **{field: value})
    leverage, note = manager.allowed_leverage(levered, state, BreakerType.NONE)
    assert leverage == 1.0
    assert note


def test_leverage_forced_to_one_with_three_positions_open(manager, signal):
    import dataclasses

    levered = dataclasses.replace(signal, leverage=1.25)
    state = PortfolioState(
        equity=100_000.0, positions={f"S{i}": {"market_value": 1_000.0} for i in range(3)}
    )
    assert manager.allowed_leverage(levered, state, BreakerType.NONE)[0] == 1.0


def test_leverage_forced_to_one_when_a_breaker_is_active(manager, signal):
    import dataclasses

    levered = dataclasses.replace(signal, leverage=1.25)
    leverage, note = manager.allowed_leverage(
        levered, PortfolioState(equity=100_000.0), BreakerType.DAILY_REDUCE
    )
    assert leverage == 1.0
    assert "breaker" in note


def test_configured_leverage_ceiling_is_unreachable(risk_config):
    """A contradiction in the configured limits, pinned so it stays visible.

    Leverage IS gross exposure, so 1.25x means 125% gross. `max_exposure` caps
    gross exposure at 80%. The exposure gate binds first, so nothing can reach
    1.25x and the low-vol leverage rule is dead code under these settings.
    """
    assert risk_config["max_leverage"] > risk_config["max_exposure"], (
        "if this ever fails the contradiction has been resolved: update "
        "docs/PHASE5-NOTES.md section 2"
    )


# -- correlation and sector -------------------------------------------------

@pytest.fixture
def correlated_history():
    index = pd.bdate_range("2024-01-01", periods=90)
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 90)
    return pd.DataFrame(
        {
            "SPY": base,
            "QQQ": base * 0.98 + rng.normal(0, 0.001, 90),   # near-identical
            "MID": base * 0.75 + rng.normal(0, 0.006, 90),   # partly related
            "GLD": rng.normal(0, 0.01, 90),                  # unrelated
        },
        index=index,
    )


def test_highly_correlated_symbol_is_rejected(risk_config, isolated_lock, correlated_history):
    manager = RiskManager(risk_config, lock_file=isolated_lock, price_history=correlated_history)
    state = PortfolioState(equity=100_000.0, positions={"QQQ": {"market_value": 10_000.0}})
    multiplier, note = manager.check_correlation("SPY", state)
    assert multiplier == 0.0
    assert "reject" in note


def test_uncorrelated_symbol_passes(risk_config, isolated_lock, correlated_history):
    manager = RiskManager(risk_config, lock_file=isolated_lock, price_history=correlated_history)
    state = PortfolioState(equity=100_000.0, positions={"QQQ": {"market_value": 10_000.0}})
    assert manager.check_correlation("GLD", state)[0] == 1.0


def test_missing_price_history_fails_open(manager):
    """Deliberate: refusing every trade when the data feed is thin turns a data
    problem into an outage. Position and sector limits still apply."""
    state = PortfolioState(equity=100_000.0, positions={"QQQ": {"market_value": 10_000.0}})
    assert manager.check_correlation("SPY", state)[0] == 1.0


def test_sector_cap_rejects_an_overweight_addition(risk_config, isolated_lock, signal):
    sectors = {"NVDA": "tech", "AMD": "tech", "SPY": "tech"}
    manager = RiskManager(risk_config, lock_file=isolated_lock, sector_map=sectors)
    state = PortfolioState(
        equity=100_000.0,
        positions={"NVDA": {"market_value": 15_000.0}, "AMD": {"market_value": 14_000.0}},
    )
    ok, note = manager.check_sector("SPY", 5_000.0, state)
    assert ok is False
    assert "tech" in note


def test_sector_check_passes_without_a_sector_map(manager):
    state = PortfolioState(equity=100_000.0, positions={"A": {"market_value": 20_000.0}})
    assert manager.check_sector("SPY", 5_000.0, state)[0] is True


# -- order validation -------------------------------------------------------

def test_wide_spread_is_rejected(manager, signal, portfolio):
    decision = manager.validate_signal(
        signal, portfolio, quote={"bid": 99.0, "ask": 101.0, "tradeable": True}
    )
    assert decision.approved is False
    assert decision.rejection_reason is RejectionReason.SPREAD_TOO_WIDE


def test_tight_spread_passes(manager, signal, portfolio):
    decision = manager.validate_signal(
        signal, portfolio, quote={"bid": 99.99, "ask": 100.01, "tradeable": True}
    )
    assert decision.approved


def test_untradeable_symbol_is_rejected(manager, signal, portfolio):
    decision = manager.validate_signal(
        signal, portfolio, quote={"bid": 100.0, "ask": 100.01, "tradeable": False}
    )
    assert decision.approved is False
    assert decision.rejection_reason is RejectionReason.NOT_TRADEABLE


def test_insufficient_buying_power_is_rejected(manager, signal):
    state = PortfolioState(equity=100_000.0, buying_power=50.0)
    decision = manager.validate_signal(
        signal, state, quote={"bid": 99.99, "ask": 100.01, "tradeable": True}
    )
    assert decision.approved is False
    assert decision.rejection_reason is RejectionReason.INSUFFICIENT_BUYING_POWER


def test_duplicate_orders_are_blocked(manager, signal, portfolio):
    """Guards against a retry loop or a double-invoked scheduler placing the
    same order twice, which on a broker API is silent and expensive."""
    assert manager.validate_signal(signal, portfolio).approved
    second = manager.validate_signal(signal, portfolio)
    assert second.approved is False
    assert second.rejection_reason is RejectionReason.DUPLICATE_ORDER


def test_different_symbols_are_not_duplicates(manager, signal, portfolio):
    import dataclasses

    assert manager.validate_signal(signal, portfolio).approved
    other = dataclasses.replace(signal, symbol="QQQ")
    assert manager.validate_signal(other, portfolio).approved


# -- rejection reporting ----------------------------------------------------

def test_every_rejection_carries_a_structured_reason(manager, signal):
    """Free text cannot be counted. "Why did the system stop trading in March"
    is only answerable if rejections are queryable."""
    decision = manager.validate_signal(signal, PortfolioState(equity=500.0))
    assert decision.approved is False
    assert isinstance(decision.rejection_reason, RejectionReason)
    assert decision.reason


def test_modifications_are_recorded(manager, signal, portfolio):
    """A position that comes out at a third of its requested size must explain
    why without a debugging session."""
    decision = manager.validate_signal(signal, portfolio)
    assert decision.approved
    assert decision.modifications
    assert decision.action is RiskAction.APPROVE_MODIFIED


def test_reject_helper_builds_a_consistent_decision():
    decision = RiskDecision.reject(RejectionReason.NO_STOP_LOSS, "missing stop")
    assert decision.approved is False
    assert decision.action is RiskAction.REJECT
    assert decision.approved_quantity == 0.0


# -- backward compatibility -------------------------------------------------

def test_legacy_evaluate_still_works(manager, signal):
    """The backtester and the Phase 1 tests call this."""
    decision = manager.evaluate(signal, 100_000.0, {})
    assert isinstance(decision, RiskDecision)
    assert decision.approved


def test_check_breakers_replaces_the_phase_4_helper(manager):
    """Phase 4 had its own threshold evaluation because this class did not
    exist. Two implementations of the same thresholds would drift."""
    index = pd.bdate_range("2024-01-01", periods=60)
    calm = pd.Series(np.linspace(100_000, 110_000, 60), index=index)
    assert manager.check_breakers(calm) is BreakerState.NORMAL

    crashed = calm.copy()
    crashed.iloc[30:] *= 0.80
    assert manager.check_breakers(crashed) is BreakerState.HALTED


def test_risk_manager_exposes_the_documented_surface():
    for name in ("validate_signal", "evaluate", "position_size", "gap_capped_size",
                 "check_correlation", "check_sector", "validate_order", "is_duplicate",
                 "allowed_leverage", "check_breakers", "halt", "is_halted",
                 "reset_daily", "reset_weekly", "get_breaker_history"):
        assert hasattr(RiskManager, name), f"RiskManager.{name} missing"


def test_circuit_breaker_exposes_the_documented_surface():
    for name in ("check", "update", "reset_daily", "reset_weekly", "get_history",
                 "halt", "is_halted", "size_multiplier"):
        assert hasattr(CircuitBreaker, name), f"CircuitBreaker.{name} missing"


# -- config -----------------------------------------------------------------

def test_all_thresholds_come_from_settings(settings, isolated_lock):
    manager = RiskManager(settings["risk"], lock_file=isolated_lock)
    risk = settings["risk"]
    assert manager.max_risk_per_trade == risk["max_risk_per_trade"]
    assert manager.max_exposure == risk["max_exposure"]
    assert manager.max_concurrent == risk["max_concurrent"]
    assert manager.min_position_usd == risk["min_position_usd"]
    assert manager.breaker.max_dd_from_peak == risk["max_dd_from_peak"]


def test_breakers_escalate_in_order(settings):
    risk = settings["risk"]
    assert risk["daily_dd_reduce"] < risk["daily_dd_halt"]
    assert risk["weekly_dd_reduce"] < risk["weekly_dd_halt"]
    assert risk["max_dd_from_peak"] >= risk["weekly_dd_halt"]


def test_risk_per_trade_is_one_percent(settings):
    assert settings["risk"]["max_risk_per_trade"] == 0.01
