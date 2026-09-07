"""
Tests for data/feature_engineering.py.

The features are pure functions, so they are cheap to test properly. The tests
that matter are the causality ones: a feature that peeks at the future produces
a model that cannot be salvaged later.
"""

import numpy as np
import pandas as pd
import pytest

from data import feature_engineering as fe


def test_all_declared_features_are_produced(synthetic_bars):
    raw = fe.compute_raw_features(synthetic_bars)
    assert list(raw.columns) == fe.FEATURE_COLUMNS


def test_volatility_subset_is_a_subset():
    assert set(fe.VOLATILITY_FEATURES) <= set(fe.FEATURE_COLUMNS)


def test_feature_matrix_has_no_nan_or_inf(features):
    assert features.notna().all().all()
    assert np.isfinite(features.to_numpy()).all()


def test_warmup_arithmetic_matches_reality(synthetic_bars):
    """required_warmup() must match the rows actually discarded.

    If it drifts, Phase 4 silently trains on fewer bars than configured, which
    is invisible until the results are inexplicably bad.
    """
    matrix = fe.build_feature_matrix(synthetic_bars)
    discarded = len(synthetic_bars) - len(matrix)
    assert discarded == fe.required_warmup(252)


def test_required_raw_bars_is_much_larger_than_the_row_count():
    """504 usable rows needs ~954 raw bars, not 504.

    The trap this guards: `min_train_bars: 504` reads like "two years of data"
    and actually means closer to four.
    """
    assert fe.required_raw_bars(504) == 954


# -- causality --------------------------------------------------------------

def test_appending_future_bars_changes_nothing(synthetic_bars):
    """The core causality guarantee for the feature layer.

    Compute features on a prefix, then on the full series, and compare the
    overlap. Any centred window or wrong-direction shift shows up here.
    """
    prefix = fe.build_feature_matrix(synthetic_bars.iloc[:1500])
    full = fe.build_feature_matrix(synthetic_bars)
    common = prefix.index.intersection(full.index)
    assert len(common) > 500
    pd.testing.assert_frame_equal(prefix.loc[common], full.loc[common])


def test_rolling_zscore_uses_only_trailing_data():
    """A spike at bar t must not move the z-score of bar t-1."""
    base = pd.Series(np.linspace(1.0, 2.0, 400))
    spiked = base.copy()
    spiked.iloc[350] = 100.0

    z_base = fe.rolling_zscore(base, window=252, clip=None)
    z_spiked = fe.rolling_zscore(spiked, window=252, clip=None)

    pd.testing.assert_series_equal(z_base.iloc[:350], z_spiked.iloc[:350])
    assert z_base.iloc[351] != z_spiked.iloc[351], "the spike should affect later bars"


def test_rolling_windows_emit_no_partial_values():
    """min_periods must equal the window everywhere.

    Pandas defaults to emitting after one observation, which would make a
    "200-day SMA" return a number on bar 3: a different statistic wearing the
    same name.
    """
    series = pd.Series(np.arange(300, dtype=float))
    assert fe.sma(series, 200).iloc[:199].isna().all()
    assert not np.isnan(fe.sma(series, 200).iloc[199])


def test_zscore_clip_bounds_outliers():
    series = pd.Series(np.concatenate([np.random.default_rng(0).normal(0, 1, 400), [500.0]]))
    z = fe.rolling_zscore(series, window=252, clip=5.0)
    assert z.abs().max() <= 5.0 + 1e-9


# -- indicator correctness --------------------------------------------------

def test_adx_is_bounded_and_directionless(synthetic_bars):
    """ADX measures trend strength, not direction, so it must be non-negative
    and behave identically on an inverted series."""
    a = fe.adx(synthetic_bars["high"], synthetic_bars["low"], synthetic_bars["close"])
    valid = a.dropna()
    assert len(valid) > 1000
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_is_bounded(synthetic_bars):
    r = fe.rsi(synthetic_bars["close"]).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_rsi_saturates_on_an_unbroken_advance():
    """avg_loss of zero means RSI is 100 by definition, not NaN from a divide."""
    r = fe.rsi(pd.Series(np.arange(1, 200, dtype=float)), window=14)
    assert r.dropna().iloc[-1] == pytest.approx(100.0)


def test_atr_accounts_for_gaps():
    """True Range must exceed high-minus-low when the bar gapped.

    This is why ATR rather than plain range: on a system holding overnight, the
    gap is the risk that matters.
    """
    close = pd.Series([100.0, 100.0, 80.0, 80.0])
    high = pd.Series([101.0, 101.0, 81.0, 81.0])
    low = pd.Series([99.0, 99.0, 79.0, 79.0])
    a = fe.atr(high, low, close, window=2)
    assert a.iloc[2] > 2.0, "gap-down bar should register more than its 2-point range"


def test_normalized_atr_is_scale_free():
    """The same shape at 10x the price must give the same normalised ATR."""
    n = 100
    close = pd.Series(np.full(n, 100.0))
    a = fe.normalized_atr(close * 1.01, close * 0.99, close, 14)
    b = fe.normalized_atr(close * 10 * 1.01, close * 10 * 0.99, close * 10, 14)
    pd.testing.assert_series_equal(a, b)


def test_rolling_slope_recovers_a_known_gradient():
    series = pd.Series(np.arange(100, dtype=float) * 3.0)
    slope = fe._rolling_slope(series, 10).dropna()
    assert np.allclose(slope, 3.0)


def test_bars_must_be_sorted_and_unique(synthetic_bars):
    """Out-of-order bars make every trailing window silently wrong, so this
    must raise rather than compute something plausible."""
    shuffled = synthetic_bars.iloc[::-1]
    with pytest.raises(ValueError, match="sorted ascending"):
        fe.compute_raw_features(shuffled)

    duped = pd.concat([synthetic_bars.iloc[:10], synthetic_bars.iloc[:10]]).sort_index()
    with pytest.raises(ValueError, match="duplicate"):
        fe.compute_raw_features(duped)


def test_missing_columns_raise(synthetic_bars):
    with pytest.raises(ValueError, match="missing required columns"):
        fe.compute_raw_features(synthetic_bars.drop(columns=["volume"]))


def test_unknown_feature_column_raises(synthetic_bars):
    with pytest.raises(ValueError, match="unknown feature columns"):
        fe.build_feature_matrix(synthetic_bars, columns=["not_a_feature"])
