"""
Tests for core/regime_strategies.py.

Three things here are load-bearing and everything else is detail:

1. Strategies map by measured volatility, never by regime label.
2. No signal is ever SHORT.
3. Every stop sits strictly below entry.

(3) is the one that would have shipped broken: the spec's stop formulas return
a stop at or above entry on a quarter to a third of all bars.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import (
    Regime,
    RegimeInfo,
    RegimeState,
    VolatilityRank,
    assign_volatility_ranks,
    volatility_rank_from_position,
)
from core.regime_strategies import (
    LABEL_TO_STRATEGY,
    STRATEGY_BY_VOL_RANK,
    UNCERTAINTY_TAG,
    BaseStrategy,
    BullTrendStrategy,
    CrashDefensiveStrategy,
    Direction,
    HighVolDefensiveStrategy,
    LowVolBullStrategy,
    MeanReversionStrategy,
    MidVolCautiousStrategy,
    RegimeStrategies,
    Signal,
    StrategyOrchestrator,
)


@pytest.fixture
def regime_state():
    return RegimeState(
        label=Regime.NEUTRAL, state_id=1, probability=0.95,
        state_probabilities={Regime.NEUTRAL: 0.95},
        timestamp=pd.Timestamp("2024-01-02"), is_confirmed=True, consecutive_bars=10,
    )


@pytest.fixture
def regime_infos():
    """Three regimes with deliberately misleading labels.

    "euphoria" is the most volatile and "crash" the calmest, which is backwards
    from intuition and exactly the case the volatility mapping must handle.
    """
    def info(i, name, vol):
        return RegimeInfo(
            regime_id=i, regime_name=name, expected_return=0.0, expected_volatility=vol,
            volatility_rank=VolatilityRank.MID, recommended_strategy_type="x",
            max_leverage_allowed=1.25, max_position_size_pct=0.15, min_confidence_to_act=0.55,
        )
    return {0: info(0, "crash", 0.08), 1: info(1, "neutral", 0.25), 2: info(2, "euphoria", 0.60)}


@pytest.fixture
def orchestrator(settings, regime_infos):
    config = dict(settings["strategy"])
    config["min_confidence"] = settings["hmm"]["min_confidence"]
    return StrategyOrchestrator(config=config, regime_infos=regime_infos)


# -- always long ------------------------------------------------------------

def test_direction_has_no_short_member():
    """Structural, not a convention. Shorting destroyed returns in walk-forward
    testing, so adding it has to be a code change rather than an argument."""
    assert {d.name for d in Direction} == {"LONG", "FLAT"}
    assert not hasattr(Direction, "SHORT")


def test_every_strategy_is_long(synthetic_bars, regime_state):
    for cls in (LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy):
        signal = cls().generate_signal("SPY", synthetic_bars, regime_state)
        assert signal.direction is Direction.LONG


def test_high_vol_stays_partially_invested(synthetic_bars, regime_state):
    """60%, not 0%. V-shaped recoveries are fast and the HMM is 2-3 days late,
    so sitting in cash misses the rebound that pays for the drawdown."""
    signal = HighVolDefensiveStrategy().generate_signal("SPY", synthetic_bars, regime_state)
    assert signal.position_size_pct == pytest.approx(0.60)
    assert signal.leverage == 1.0


# -- the stop clamp ---------------------------------------------------------

def test_stop_is_always_below_entry(synthetic_bars, regime_state):
    """The one that would have shipped broken.

    Swept across the whole series for all three strategies. The spec's raw
    formulas are anchored to the 50 EMA, and price sits below the EMA for most
    of any selloff, so they return a stop above entry a quarter to a third of
    the time.
    """
    for cls in (LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy):
        strategy = cls()
        for i in range(200, len(synthetic_bars), 23):
            signal = strategy.generate_signal("SPY", synthetic_bars.iloc[:i], regime_state)
            if signal is None:
                continue
            assert signal.stop_loss < signal.entry_price, (
                f"{cls.__name__} at bar {i}: stop {signal.stop_loss} >= entry "
                f"{signal.entry_price}"
            )


def test_clamp_actually_fires_and_rescues_fatal_stops(synthetic_bars, regime_state):
    """Confirms the clamp is doing real work, not sitting idle.

    If this ever reports zero rescues the raw formulas have changed, and the
    clamp's justification should be re-examined rather than assumed.
    """
    strategy = MidVolCautiousStrategy()
    fatal = total = 0
    for i in range(200, len(synthetic_bars), 11):
        signal = strategy.generate_signal("SPY", synthetic_bars.iloc[:i], regime_state)
        if signal is None:
            continue
        total += 1
        if signal.metadata["raw_stop"] >= signal.entry_price:
            fatal += 1
            assert signal.metadata["stop_clamped"] is True
    assert total > 100
    assert fatal > 0, "expected the raw formula to produce stops above entry"


def test_risk_per_share_is_never_zero(synthetic_bars, regime_state):
    """Phase 5 sizes as risk / abs(entry - stop). A zero denominator crashes."""
    for i in range(200, len(synthetic_bars), 31):
        signal = LowVolBullStrategy().generate_signal("SPY", synthetic_bars.iloc[:i], regime_state)
        if signal is not None:
            assert signal.risk_per_share > 0


def test_signal_rejects_a_stop_at_or_above_entry():
    """Defence in depth: even a hand-built Signal cannot carry a bad stop."""
    kwargs = dict(
        symbol="X", direction=Direction.LONG, confidence=1.0, entry_price=100.0,
        take_profit=None, position_size_pct=0.95, leverage=1.0, regime_id=0,
        regime_name="r", regime_probability=1.0, timestamp=pd.Timestamp("2024-01-02"),
        reasoning="", strategy_name="t",
    )
    with pytest.raises(ValueError, match="not below entry"):
        Signal(stop_loss=105.0, **kwargs)
    with pytest.raises(ValueError, match="not below entry"):
        Signal(stop_loss=100.0, **kwargs)
    assert Signal(stop_loss=95.0, **kwargs).risk_per_share == pytest.approx(5.0)


def test_percentage_floor_covers_collapsed_atr(regime_state):
    """When ATR collapses in a flat stretch, the ATR-based floor alone would put
    the stop a fraction of a cent below entry, and Phase 5 would size a position
    large enough to breach every limit it has."""
    n = 300
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.0001, "low": 99.9999, "close": 100.0, "volume": 1e6},
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    signal = LowVolBullStrategy(min_stop_pct=0.005).generate_signal("X", flat, regime_state)
    assert signal is not None
    assert signal.metadata["stop_distance_pct"] >= 0.005 - 1e-9


# -- allocation rules -------------------------------------------------------

def test_low_vol_is_the_only_levered_tier(synthetic_bars, regime_state):
    low = LowVolBullStrategy().generate_signal("SPY", synthetic_bars, regime_state)
    mid = MidVolCautiousStrategy().generate_signal("SPY", synthetic_bars, regime_state)
    high = HighVolDefensiveStrategy().generate_signal("SPY", synthetic_bars, regime_state)
    assert low.leverage == 1.25
    assert mid.leverage == 1.0 and high.leverage == 1.0


def test_mid_vol_trend_filter_switches_allocation(regime_state):
    """Above the 50 EMA stay at 95%, below it cut to 60%."""
    n = 300
    rising = pd.Series(np.linspace(100, 200, n), index=pd.bdate_range("2020-01-01", periods=n))
    falling = pd.Series(np.linspace(200, 100, n), index=pd.bdate_range("2020-01-01", periods=n))

    def frame(close):
        return pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": 1e6}, index=close.index)

    strategy = MidVolCautiousStrategy()
    assert strategy.has_trend(frame(rising))
    assert not strategy.has_trend(frame(falling))
    assert strategy.generate_signal("X", frame(rising), regime_state).position_size_pct == pytest.approx(0.95)
    assert strategy.generate_signal("X", frame(falling), regime_state).position_size_pct == pytest.approx(0.60)


def test_allocations_sit_in_the_specified_band(synthetic_bars, regime_state):
    """0.60 to 0.95 before uncertainty halving."""
    for cls in (LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy):
        signal = cls().generate_signal("SPY", synthetic_bars, regime_state)
        assert 0.60 <= signal.position_size_pct <= 0.95


def test_low_vol_stop_picks_the_tighter_of_the_two():
    """max(price - 3ATR, EMA50 - 0.5ATR) takes whichever is higher, so the stop
    rides up behind a trend rather than trailing three ATR below forever."""
    strategy = LowVolBullStrategy()
    assert strategy.compute_raw_stop(price=100.0, ema50=99.0, atr_value=1.0) == pytest.approx(98.5)
    assert strategy.compute_raw_stop(price=100.0, ema50=90.0, atr_value=1.0) == pytest.approx(97.0)


def test_high_vol_stop_is_wider_than_mid_vol():
    """1.0 ATR below the EMA versus 0.5. A normal-width stop in a turbulent
    regime is a slower way of selling the bottom."""
    args = dict(price=100.0, ema50=100.0, atr_value=2.0)
    assert HighVolDefensiveStrategy().compute_raw_stop(**args) < MidVolCautiousStrategy().compute_raw_stop(**args)


# -- volatility rank mapping ------------------------------------------------

def test_rank_boundaries_match_the_spec():
    assert volatility_rank_from_position(0.0) is VolatilityRank.LOW
    assert volatility_rank_from_position(0.33) is VolatilityRank.LOW
    assert volatility_rank_from_position(0.34) is VolatilityRank.MID
    assert volatility_rank_from_position(0.66) is VolatilityRank.MID
    assert volatility_rank_from_position(0.67) is VolatilityRank.HIGH
    assert volatility_rank_from_position(1.0) is VolatilityRank.HIGH


@pytest.mark.parametrize(
    "n_states,expected",
    [
        (3, ["low", "mid", "high"]),
        (4, ["low", "mid", "mid", "high"]),
        (5, ["low", "low", "mid", "high", "high"]),
        (6, ["low", "low", "mid", "mid", "high", "high"]),
        (7, ["low", "low", "mid", "mid", "mid", "high", "high"]),
    ],
)
def test_exact_partition_per_state_count(n_states, expected):
    """Pins the partition for every state count.

    These are not even thirds. The spec's literal 0.33 / 0.67 boundaries put
    rank 2 of 7 at 0.3333, just above 0.33, so it lands in MID rather than LOW.
    Pinned here so the behaviour is visible rather than a floating-point
    surprise discovered during a backtest.
    """
    ranks = assign_volatility_ranks({i: float(i) for i in range(n_states)})
    assert [ranks[i].value for i in range(n_states)] == expected


def test_orchestrator_maps_by_volatility_not_label(orchestrator, regime_infos):
    """The central architectural guarantee of this phase.

    The fixture labels the calmest regime "crash" and the most volatile
    "euphoria". A label-driven mapping would defend against the calm one and
    lever into the wild one, which is exactly backwards.
    """
    assert isinstance(orchestrator.get_strategy(0), LowVolBullStrategy)     # "crash", calmest
    assert isinstance(orchestrator.get_strategy(1), MidVolCautiousStrategy)
    assert isinstance(orchestrator.get_strategy(2), HighVolDefensiveStrategy)  # "euphoria", wildest


def test_label_to_strategy_is_a_fallback_not_the_path(orchestrator):
    """LABEL_TO_STRATEGY exists for callers holding a bare label, and it
    disagrees with the volatility mapping on the fixture. That disagreement is
    the point: it is why the orchestrator does not use it."""
    assert LABEL_TO_STRATEGY[Regime.CRASH] is HighVolDefensiveStrategy
    assert not isinstance(orchestrator.get_strategy(0), HighVolDefensiveStrategy)


def test_label_to_strategy_covers_every_regime():
    for regime in Regime:
        assert regime in LABEL_TO_STRATEGY


def test_unknown_regime_is_treated_as_dangerous():
    assert LABEL_TO_STRATEGY[Regime.UNKNOWN] is HighVolDefensiveStrategy


def test_backward_compatible_aliases_resolve():
    assert CrashDefensiveStrategy is HighVolDefensiveStrategy
    assert BullTrendStrategy is LowVolBullStrategy
    assert MeanReversionStrategy is MidVolCautiousStrategy


def test_strategy_by_vol_rank_covers_all_tiers():
    assert set(STRATEGY_BY_VOL_RANK) == set(VolatilityRank)


def test_update_regime_infos_rebuilds_after_retrain(orchestrator, regime_infos):
    """EM renumbers its states on every refit, so a mapping built against the
    previous fit points at the wrong regimes. Not calling this after a retrain
    is the subtle way this layer goes wrong."""
    flipped = {
        0: replace(regime_infos[0], expected_volatility=0.60),
        1: regime_infos[1],
        2: replace(regime_infos[2], expected_volatility=0.08),
    }
    orchestrator.update_regime_infos(flipped)
    assert isinstance(orchestrator.get_strategy(0), HighVolDefensiveStrategy)
    assert isinstance(orchestrator.get_strategy(2), LowVolBullStrategy)


def test_unmapped_regime_raises_with_guidance(orchestrator):
    with pytest.raises(KeyError, match="update_regime_infos"):
        orchestrator.get_strategy(99)


# -- uncertainty ------------------------------------------------------------

@pytest.mark.parametrize(
    "field,value,label",
    [
        ("probability", 0.40, "confidence"),
        ("is_flickering", True, "flickering"),
        ("is_confirmed", False, "unconfirmed"),
    ],
)
def test_all_three_uncertainty_triggers(orchestrator, synthetic_bars, regime_state, field, value, label):
    state = replace(regime_state, state_id=0, **{field: value})
    signal = orchestrator.generate_signals(["SPY"], {"SPY": synthetic_bars}, state)[0]
    assert UNCERTAINTY_TAG in signal.reasoning
    assert label in signal.metadata["uncertainty_reason"]
    assert signal.position_size_pct == pytest.approx(0.95 * 0.5)


def test_uncertainty_forces_leverage_to_one(orchestrator, synthetic_bars, regime_state):
    """Forced, not scaled. Leverage on a call the model admits is unreliable is
    what turns a bad week into an unrecoverable one."""
    state = replace(regime_state, state_id=0, probability=0.40)
    signal = orchestrator.generate_signals(["SPY"], {"SPY": synthetic_bars}, state)[0]
    assert signal.leverage == 1.0
    assert signal.metadata["pre_uncertainty_leverage"] == 1.25


def test_no_uncertainty_when_all_clear(orchestrator, synthetic_bars, regime_state):
    signal = orchestrator.generate_signals(
        ["SPY"], {"SPY": synthetic_bars}, replace(regime_state, state_id=0)
    )[0]
    assert UNCERTAINTY_TAG not in signal.reasoning
    assert signal.position_size_pct == pytest.approx(0.95)
    assert signal.leverage == 1.25


def test_uncertainty_records_pre_reduction_values(orchestrator, synthetic_bars, regime_state):
    """So a reviewer can see what was intended before the cut, not just what
    happened."""
    state = replace(regime_state, state_id=0, probability=0.40)
    signal = orchestrator.generate_signals(["SPY"], {"SPY": synthetic_bars}, state)[0]
    assert signal.metadata["pre_uncertainty_size"] == pytest.approx(0.95)
    assert signal.metadata["uncertainty"] is True


# -- rebalancing ------------------------------------------------------------

@pytest.mark.parametrize(
    "target,current,expected",
    [(0.95, 0.95, False), (0.95, 0.90, False), (0.95, 0.85, False),
     (0.95, 0.84, True), (0.95, 0.60, True), (0.60, 0.95, True)],
)
def test_rebalance_threshold(orchestrator, target, current, expected):
    assert orchestrator.needs_rebalance(target, current) is expected


def test_rebalance_is_symmetric(orchestrator):
    assert orchestrator.needs_rebalance(0.95, 0.60) == orchestrator.needs_rebalance(0.60, 0.95)


# -- signal generation ------------------------------------------------------

def test_generates_one_signal_per_symbol(orchestrator, synthetic_bars, regime_state):
    symbols = ["SPY", "QQQ", "AAPL"]
    bars = {s: synthetic_bars for s in symbols}
    signals = orchestrator.generate_signals(symbols, bars, replace(regime_state, state_id=0))
    assert len(signals) == 3
    assert {s.symbol for s in signals} == set(symbols)


def test_per_symbol_weight_divides_the_portfolio_allocation(orchestrator, synthetic_bars, regime_state):
    """position_size_pct is the portfolio-level target; the per-symbol share is
    that divided across the surviving signals. Carried explicitly so nothing
    downstream has to guess which one it is looking at."""
    symbols = ["SPY", "QQQ", "AAPL", "MSFT"]
    signals = orchestrator.generate_signals(
        symbols, {s: synthetic_bars for s in symbols}, replace(regime_state, state_id=0)
    )
    for signal in signals:
        assert signal.metadata["per_symbol_weight"] == pytest.approx(0.95 / 4)
        assert signal.metadata["n_symbols"] == 4


def test_symbols_without_bars_are_skipped(orchestrator, synthetic_bars, regime_state):
    signals = orchestrator.generate_signals(
        ["SPY", "MISSING"], {"SPY": synthetic_bars}, replace(regime_state, state_id=0)
    )
    assert [s.symbol for s in signals] == ["SPY"]


def test_insufficient_history_returns_none_not_an_error(regime_state):
    """A symbol with 30 bars is not an error, it is a symbol that cannot be
    traded yet."""
    short = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6},
        index=pd.bdate_range("2024-01-01", periods=30),
    )
    assert LowVolBullStrategy().generate_signal("X", short, regime_state) is None


def test_signal_carries_full_regime_context(orchestrator, synthetic_bars, regime_state):
    """So the trade log explains itself six months later."""
    signal = orchestrator.generate_signals(
        ["SPY"], {"SPY": synthetic_bars}, replace(regime_state, state_id=0)
    )[0]
    assert signal.regime_id == 0
    assert signal.regime_name
    assert 0.0 <= signal.regime_probability <= 1.0
    assert signal.strategy_name == "LowVolBullStrategy"
    assert signal.reasoning
    assert signal.metadata["volatility_rank"] == "low"
    assert signal.metadata["gross_exposure"] == pytest.approx(0.95 * 1.25)


def test_gross_exposure_is_reported(orchestrator, synthetic_bars, regime_state):
    """allocation x leverage. Phase 5 compares this against max_exposure, and
    the low-vol tier exceeds the configured limit, so it must be visible."""
    signal = orchestrator.generate_signals(
        ["SPY"], {"SPY": synthetic_bars}, replace(regime_state, state_id=0)
    )[0]
    assert signal.metadata["gross_exposure"] > 1.0


# -- config wiring ----------------------------------------------------------

def test_orchestrator_builds_from_settings_yaml(settings, regime_infos):
    config = dict(settings["strategy"])
    config["min_confidence"] = settings["hmm"]["min_confidence"]
    orchestrator = StrategyOrchestrator(config=config, regime_infos=regime_infos)
    assert orchestrator.rebalance_threshold == settings["strategy"]["rebalance_threshold"]
    assert orchestrator.uncertainty_size_mult == settings["strategy"]["uncertainty_size_mult"]
    low = orchestrator.get_strategy(0)
    assert low.allocation == settings["strategy"]["low_vol_allocation"]
    assert low.leverage == settings["strategy"]["low_vol_leverage"]


def test_leverage_is_capped_by_max_leverage(regime_infos):
    """The strategy may not ask for more than the risk layer's ceiling."""
    orchestrator = StrategyOrchestrator(
        config={"low_vol_leverage": 3.0, "max_leverage": 1.25}, regime_infos=regime_infos
    )
    assert orchestrator.get_strategy(0).leverage == 1.25


def test_empty_regime_infos_raises():
    with pytest.raises(ValueError, match="cannot map strategies"):
        StrategyOrchestrator(regime_infos={}).update_regime_infos({})


def test_base_strategy_is_abstract():
    with pytest.raises(TypeError):
        BaseStrategy(allocation=0.95)


# -- backward-compatible wrapper --------------------------------------------

def test_regime_strategies_wrapper_preserves_the_old_api(settings, regime_infos, synthetic_bars, regime_state):
    wrapper = RegimeStrategies(config=dict(settings["strategy"]), regime_infos=regime_infos)
    allocation, leverage = wrapper.compute_allocation(replace(regime_state, state_id=0), synthetic_bars)
    assert allocation == pytest.approx(0.95)
    assert leverage == 1.25
    assert wrapper.needs_rebalance(0.95, 0.60) is True


def test_wrapper_applies_uncertainty(settings, regime_infos, synthetic_bars, regime_state):
    wrapper = RegimeStrategies(config=dict(settings["strategy"]), regime_infos=regime_infos)
    allocation, leverage = wrapper.compute_allocation(
        replace(regime_state, state_id=0, probability=0.20), synthetic_bars
    )
    assert allocation == pytest.approx(0.475)
    assert leverage == 1.0
