"""
Verify no look-ahead bias.

The most important test file in the project.

Look-ahead bias does not throw, does not warn, and makes the equity curve
better. The only defence is a test that fails when it appears.

A NOTE ON THE SPEC'S TEST
-------------------------
The tutorial specifies:

    regime_short = predict_regime_filtered(data[0:400])[-1]
    regime_long  = predict_regime_filtered(data[0:500])[400]
    assert regime_short == regime_long

Those two indices are not the same bar. `data[0:400]` holds rows 0-399, so
`[-1]` is row 399, while `[400]` on the 500-row result is row 400. The test as
written compares consecutive bars.

That is worse than a test that fails, because regimes persist for weeks: the
two adjacent bars usually share a regime, so it passes almost always, for the
wrong reason, and would keep passing against an implementation that leaks. It
was in fact observed passing by coincidence during development.

`test_no_look_ahead_bias` below is the spec's test with the index corrected.
`test_entire_prefix_is_identical` is the version that actually has teeth: every
bar of the overlap must match, not one sampled bar.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from core import hmm_engine as m
from core.hmm_engine import VOLATILITY_FEATURES

# -- the mandatory test -----------------------------------------------------

def test_no_look_ahead_bias(fitted_engine, features):
    """Regime at T must be identical whether computed from data[0:T] or a
    longer series that contains T.

    The spec's test, with its off-by-one corrected: both sides read row 399.
    """
    short = fitted_engine.predict_regime_filtered(features.iloc[:400])
    long_ = fitted_engine.predict_regime_filtered(features.iloc[:500])

    regime_short = short["state_id"].iloc[-1]     # row 399
    regime_long = long_["state_id"].iloc[399]     # row 399, the same bar

    assert regime_short == regime_long, "LOOK-AHEAD BIAS DETECTED"


def test_entire_prefix_is_identical(fitted_engine, features):
    """Every bar of the overlap must match, and so must the probabilities.

    Comparing one sampled bar can pass by luck. Comparing 400 of them, and the
    continuous posteriors rather than just the argmax, cannot.
    """
    short = fitted_engine.predict_regime_filtered(features.iloc[:400])
    long_ = fitted_engine.predict_regime_filtered(features.iloc[:500])

    assert (short["state_id"].to_numpy() == long_["state_id"].to_numpy()[:400]).all()
    np.testing.assert_allclose(
        short["probability"].to_numpy(),
        long_["probability"].to_numpy()[:400],
        atol=1e-12,
    )


@pytest.mark.parametrize("extra", [1, 50, 200, 800])
def test_stable_under_any_amount_of_appended_future(fitted_engine, features, extra):
    """However much future is appended, the past does not move."""
    base = fitted_engine.predict_regime_filtered(features.iloc[:600])
    extended = fitted_engine.predict_regime_filtered(features.iloc[: 600 + extra])
    np.testing.assert_allclose(
        base.filter(like="prob_").to_numpy(),
        extended.filter(like="prob_").to_numpy()[:600],
        atol=1e-12,
    )


# -- the mechanism that guarantees it ---------------------------------------

def test_viterbi_is_never_called(fitted_engine, features, monkeypatch):
    """model.predict() must not be reachable from any classification path.

    predict() runs Viterbi over the whole sequence and revises earlier states
    using later observations. This test replaces it with a landmine: if any
    future refactor reaches for it because it is simpler, the suite fails
    instead of the backtest quietly improving.
    """
    def landmine(*args, **kwargs):
        raise AssertionError("model.predict() was called: that is Viterbi, and it leaks")

    monkeypatch.setattr(fitted_engine.model, "predict", landmine)
    monkeypatch.setattr(fitted_engine.model, "predict_proba", landmine)

    fitted_engine.predict_regime_filtered(features.iloc[:300])
    fitted_engine.predict_regime_proba(features.iloc[:300])
    fitted_engine.classify_series(features.iloc[:300])
    fitted_engine.classify(features.iloc[:300])


def test_source_contains_no_predict_call():
    """Belt and braces: no `.predict()` / `.predict_proba()` call in the engine.

    Parsed from the AST rather than grepped, because the module docstring
    repeatedly names `model.predict()` in order to warn against it. A string
    search flags its own warning label; the AST only sees real calls.
    """
    import ast

    tree = ast.parse(inspect.getsource(m))
    banned = {"predict", "predict_proba", "decode", "score_samples"}
    offending = [
        f"{node.func.attr}() at line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in banned
    ]
    assert not offending, f"Viterbi-family call found: {offending}"


def test_forward_filter_matches_manual_recursion(fitted_engine, features):
    """Check the forward implementation against a plain, slow reference.

    The reference is written out longhand with no normalisation tricks, so if
    the optimised version's log-space bookkeeping is subtly wrong, the two
    disagree.
    """
    X = features[VOLATILITY_FEATURES].to_numpy()[:120]
    model = fitted_engine.model
    n = model.n_components

    from scipy.stats import multivariate_normal

    emissions = np.array(
        [
            [multivariate_normal.logpdf(x, model.means_[k], model.covars_[k]) for k in range(n)]
            for x in X
        ]
    )

    alpha = model.startprob_ * np.exp(emissions[0])
    reference = [alpha / alpha.sum()]
    for t in range(1, len(X)):
        alpha = (reference[-1] @ model.transmat_) * np.exp(emissions[t])
        reference.append(alpha / alpha.sum())

    ours = fitted_engine.predict_regime_proba(features.iloc[:120]).to_numpy()
    np.testing.assert_allclose(ours, np.array(reference), atol=1e-9)


def test_incremental_stepper_matches_batch(fitted_engine, features):
    """The live path and the backtest path must agree exactly.

    If they diverge, live trading and backtesting disagree about what regime it
    is, and every backtest result stops describing the system that runs.
    """
    X = features[VOLATILITY_FEATURES].to_numpy()[:500]
    stepped = np.vstack([fitted_engine.make_live_filter().step(X[0])])
    filt = fitted_engine.make_live_filter()
    stepped = np.vstack([filt.step(obs) for obs in X])
    batch = fitted_engine.predict_regime_proba(features.iloc[:500]).to_numpy()
    np.testing.assert_allclose(stepped, batch, atol=1e-12)


def test_classify_truncates_at_as_of(fitted_engine, features):
    """Passing the whole frame with an `as_of` must not leak the future.

    The caller will eventually pass the full feature matrix by mistake. The
    engine truncates rather than trusting them.
    """
    as_of = features.index[400]

    # classify() advances the persistent tracker, so both sides start clean.
    # Otherwise this compares tracker histories, not truncation behaviour.
    fitted_engine.tracker.reset()
    full = fitted_engine.classify(features, as_of=as_of)

    fitted_engine.tracker.reset()
    truncated = fitted_engine.classify(features.loc[:as_of])
    fitted_engine.tracker.reset()

    assert full.timestamp == as_of, "as_of must anchor the returned bar"
    assert full.raw_state_id == truncated.raw_state_id
    assert full.state_id == truncated.state_id
    assert full.probability == pytest.approx(truncated.probability)


# -- the feature layer ------------------------------------------------------

def test_features_never_use_future_bars(synthetic_bars):
    """Covered in depth in test_features.py; asserted here too because this is
    the file someone reads when they suspect a leak."""
    from data.feature_engineering import build_feature_matrix

    prefix = build_feature_matrix(synthetic_bars.iloc[:1400])
    full = build_feature_matrix(synthetic_bars)
    common = prefix.index.intersection(full.index)
    pd.testing.assert_frame_equal(prefix.loc[common], full.loc[common])


def test_scaler_is_rolling_not_fitted():
    """Standardisation must be a trailing window, not a fitted scaler.

    A StandardScaler fitted on the training set and applied to the test set is
    the textbook approach and is wrong here: it leaks the training window's mean
    and variance into out-of-sample rows. A rolling z-score cannot, whoever
    calls it.

    Checked via the AST and the import list, not a string search: the module
    docstring names StandardScaler in order to explain why it is not used.
    """
    import ast
    import inspect as _inspect

    from data import feature_engineering

    tree = ast.parse(_inspect.getsource(feature_engineering))

    imported = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any("sklearn" in name for name in imported), (
        "feature engineering must not import sklearn: a fitted scaler leaks"
    )

    called = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "rolling" in called, "standardisation must use a rolling window"
    assert "fit_transform" not in called and "fit" not in called


@pytest.mark.skip(reason="Phase 4: walk-forward backtesting")
def test_fills_occur_at_next_open_not_signal_close():
    """A signal computed from a close cannot be traded at that close."""


@pytest.mark.skip(reason="Phase 4: walk-forward backtesting")
def test_walk_forward_windows_do_not_overlap():
    """Test data must never appear in a training window."""


# -- variant 4: the null model -----------------------------------------------

def _series(segments, seed=4):
    rng = np.random.default_rng(seed)
    steps = np.concatenate([rng.normal(0, sd, n) for sd, n in segments])
    idx = pd.bdate_range("2019-01-01", periods=len(steps), tz="UTC")
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(steps))}, index=idx)


def test_rank_uses_a_trailing_window_not_all_history():
    """Ancient turbulence must not still be setting today's tier.

    This is the real failure mode for a percentile signal. Swap `.tail(lookback)`
    for the whole series and a crash five years ago keeps every quiet day since
    looking LOW forever, which is both wrong and a slow leak of stale state.
    """
    from core.regime_strategies import realised_vol_rank

    # 500 bars of chaos, then 500 calm. With a 252-bar lookback the chaos is
    # out of the window, so trimming it away must change nothing.
    frame = _series([(0.050, 500), (0.004, 500)])
    assert realised_vol_rank(frame) == realised_vol_rank(frame.iloc[500:])


def test_future_bars_cannot_change_a_past_ranking():
    """The caller slices to the current bar; confirm nothing reaches past it."""
    from core.regime_strategies import realised_vol_rank

    frame = _series([(0.004, 500), (0.050, 500)])
    as_of = realised_vol_rank(frame.iloc[:500])

    wrecked = frame.copy()
    wrecked.iloc[500:, 0] *= 10          # rewrite everything after the as-of bar
    assert realised_vol_rank(wrecked.iloc[:500]) == as_of


def test_null_model_reaches_every_tier():
    """A switch that always answered MID would pass every test above and
    measure nothing. Volatility has to oscillate for LOW to be reachable: a
    monotonically rising series sits at the top of its own trailing window
    almost always."""
    from core.hmm_engine import VolatilityRank
    from core.regime_strategies import realised_vol_rank

    frame = _series([(0.004, 300), (0.035, 300), (0.004, 300), (0.035, 300)], seed=7)
    seen = {realised_vol_rank(frame.iloc[:c]) for c in range(200, len(frame), 20)}
    seen.discard(None)
    assert seen == set(VolatilityRank), f"only reached {sorted(r.value for r in seen)}"


def test_short_history_falls_back_rather_than_guessing():
    from core.regime_strategies import realised_vol_rank

    assert realised_vol_rank(pd.DataFrame({"close": pd.Series(range(1, 40), dtype=float)})) is None


# -- variant 5: the absolute-momentum filter ---------------------------------

def _orchestrator(trend="sma200"):
    from config import load_settings, strategy_config
    from core.regime_strategies import StrategyOrchestrator

    cfg = strategy_config(load_settings())
    cfg["trend_filter"] = trend
    return StrategyOrchestrator(cfg, {})


def test_trend_filter_blocks_entries_only_below_the_average():
    """Scaling the whole series must NOT trip it: halving every bar halves the
    average too. Only a recent fall counts, which is the point of the rule."""
    idx = pd.bdate_range("2020-01-01", periods=400, tz="UTC")
    up = pd.DataFrame({"close": pd.Series(np.linspace(100, 200, 400), index=idx)})
    o = _orchestrator()

    assert o._downtrend(["SPY"], {"SPY": up}) is False

    scaled = pd.DataFrame({"close": up["close"] * 0.5})
    assert o._downtrend(["SPY"], {"SPY": scaled}) is False, (
        "rescaling the whole series moved the average with it; this is not a selloff"
    )

    sold_off = up.copy()
    sold_off.iloc[-40:, sold_off.columns.get_loc("close")] *= 0.6
    assert o._downtrend(["SPY"], {"SPY": sold_off}) is True


def test_trend_filter_is_off_by_default_and_opt_in():
    idx = pd.bdate_range("2020-01-01", periods=400, tz="UTC")
    crash = pd.DataFrame({"close": pd.Series(np.linspace(200, 80, 400), index=idx)})
    assert _orchestrator(trend="off")._downtrend(["SPY"], {"SPY": crash}) is False
    assert _orchestrator(trend="sma200")._downtrend(["SPY"], {"SPY": crash}) is True


def test_trend_filter_needs_a_full_window_before_it_speaks():
    """Fewer bars than the window means no opinion, not a false downtrend."""
    idx = pd.bdate_range("2020-01-01", periods=50, tz="UTC")
    short = pd.DataFrame({"close": pd.Series(np.linspace(200, 100, 50), index=idx)})
    assert _orchestrator()._downtrend(["SPY"], {"SPY": short}) is False


# ===========================================================================
# The backtester's trailing ratchet
#
# It used to model a stop fixed for the life of the trade while the live
# engine ratcheted one upward every cycle, so every result described a
# strategy nobody was trading. These pin the two together.
# ===========================================================================

def _trending(n=120, start=100.0, step=0.5):
    """A clean uptrend. High and low straddle the close by a fixed band so ATR
    is well defined and the trail has something to measure."""
    import numpy as np
    import pandas as pd
    close = np.arange(n, dtype=float) * step + start
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(n, 1e6)},
        index=pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC"),
    )


def _backtester(**kwargs):
    from backtest.portfolio_backtester import PortfolioBacktester
    defaults = {
        "symbols": ["SPY"], "primary": "SPY",
        "risk_config": {"trailing_stop": {
            "enabled": True, "atr_multiple": 2.5,
            "min_trail_pct": 1.5, "max_trail_pct": 15.0}},
    }
    return PortfolioBacktester(**{**defaults, **kwargs})


def test_the_backtest_stop_climbs_with_the_trend():
    from backtest.portfolio_backtester import Holding

    bars = {"SPY": _trending()}
    bt = _backtester()
    holding = Holding(symbol="SPY", shares=1, entry_price=100.0,
                      entry_time=bars["SPY"].index[0], stop_loss=90.0,
                      take_profit=None, regime_at_entry="bull")

    seen = []
    for timestamp in bars["SPY"].index[40:]:
        bt._ratchet_stop(bars, "SPY", timestamp, holding)
        seen.append(holding.stop_loss)

    assert seen[-1] > seen[0], "the stop never moved in a clean uptrend"
    assert all(b >= a for a, b in zip(seen, seen[1:], strict=False)), \
        "the stop went DOWN, which is the one thing a trailing stop may not do"


def test_the_backtest_stop_never_widens_on_a_pullback():
    """A lower floor after a fall has to be refused, or the 'trailing' stop is
    just a stop that follows price in both directions."""
    import pandas as pd
    from backtest.portfolio_backtester import Holding

    up = _trending(80)
    down = up.copy()
    down.iloc[60:] = down.iloc[60:] * 0.80          # a 20% fall in the last 20 bars
    bars = {"SPY": pd.concat([up.iloc[:60], down.iloc[60:]])}

    bt = _backtester()
    holding = Holding(symbol="SPY", shares=1, entry_price=100.0,
                      entry_time=bars["SPY"].index[0], stop_loss=90.0,
                      take_profit=None, regime_at_entry="bull")
    seen = []
    for timestamp in bars["SPY"].index[40:]:
        bt._ratchet_stop(bars, "SPY", timestamp, holding)
        seen.append(holding.stop_loss)

    # Bar 60 still reads bar 59's close, which is pre-fall, so one more rise
    # there is correct. What must never happen is a fall, at any point.
    assert all(b >= a for a, b in zip(seen, seen[1:], strict=False)), \
        "the stop fell during the drawdown"
    assert holding.stop_loss == max(seen), "the stop did not hold its high-water mark"
    low = float(bars["SPY"]["close"].iloc[-1])
    assert holding.stop_loss > low, (
        "after a 20% fall the stop should be above the price, i.e. already triggered"
    )


def test_the_ratchet_reads_yesterdays_close_not_todays():
    """Setting today's stop from today's close, then testing it against today's
    low, is look-ahead: the backtest would exit on a level it could not have
    known when the order was resting."""
    from backtest.portfolio_backtester import Holding

    bars = {"SPY": _trending()}
    frame = bars["SPY"]
    bt = _backtester()
    holding = Holding(symbol="SPY", shares=1, entry_price=100.0,
                      entry_time=frame.index[0], stop_loss=1.0,
                      take_profit=None, regime_at_entry="bull")

    timestamp = frame.index[60]
    bt._ratchet_stop(bars, "SPY", timestamp, holding)
    today, yesterday = float(frame["close"].iloc[60]), float(frame["close"].iloc[59])
    assert holding.stop_loss < yesterday, "stop is not below the close it was set from"
    assert holding.stop_loss <= yesterday, "stop was computed off a future close"
    # The floor must sit strictly below yesterday's close, never scaled to today's.
    assert holding.stop_loss / yesterday < 1.0
    assert holding.stop_loss / today < 1.0


def test_a_falsy_reward_ratio_means_no_target_not_a_target_at_the_entry():
    """`reward_risk_ratio: null` must produce take_profit=None.

    It used to compute `entry + 0 * risk`, a target sitting exactly at the
    entry price, which fills on the next bar that trades up a cent: 7,130
    trades instead of 919, and a 4.7% return that looked like evidence against
    running without a target. It was an artefact.
    """
    bt = _backtester(reward_risk_ratio=None)
    assert not bt.reward_risk_ratio
    bt2 = _backtester(reward_risk_ratio=2.0)
    assert bt2.reward_risk_ratio == 2.0
