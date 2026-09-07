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
from core.hmm_engine import HMMEngine, VOLATILITY_FEATURES


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
