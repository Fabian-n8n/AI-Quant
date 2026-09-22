"""
Verify the signal-quality metrics measure what they claim.

These metrics exist to answer one question: does the regime call predict
anything? That answer is only worth having if the instrument is calibrated, so
each test here feeds in a signal whose true information content is known by
construction and asserts the metric recovers it.

The ordering that matters: a perfect signal must score near +1, an inverted one
near -1, and noise near 0. A metric that cannot separate those three cannot be
trusted to report ~0 on a real strategy and have that mean anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.performance import (
    MIN_IC_OBSERVATIONS,
    consistency,
    forward_return,
    ic_information_ratio,
    information_coefficient,
    rolling_ic,
    signal_decay,
)

BARS = 500


@pytest.fixture
def prices() -> pd.Series:
    rng = np.random.default_rng(42)
    steps = rng.normal(0.0004, 0.01, BARS)
    return pd.Series(
        100 * np.exp(np.cumsum(steps)),
        index=pd.date_range("2020-01-01", periods=BARS, freq="B"),
    )


# ---------------------------------------------------------------------------
# forward_return: alignment is the whole contract
# ---------------------------------------------------------------------------

def test_forward_return_is_indexed_at_the_signal_bar(prices):
    fwd = forward_return(prices, horizon=1)
    expected = prices.iloc[1] / prices.iloc[0] - 1
    assert fwd.iloc[0] == pytest.approx(expected)


def test_forward_return_leaves_the_tail_unmeasurable(prices):
    """The last `horizon` rows must be NaN, never 0.0.

    Filling them would score a stretch with no future against it as a correct
    flat call, which flatters exactly the end of the sample a reader trusts most.
    """
    for h in (1, 5, 20):
        fwd = forward_return(prices, horizon=h)
        assert fwd.iloc[-h:].isna().all()
        assert fwd.iloc[: -h].notna().all()


def test_forward_return_rejects_impossible_horizons(prices):
    assert forward_return(prices, horizon=0).empty
    assert forward_return(prices.iloc[:5], horizon=10).empty


# ---------------------------------------------------------------------------
# IC: calibration against signals of known information content
# ---------------------------------------------------------------------------

def test_perfect_foresight_scores_near_one(prices):
    """A signal that IS the forward return must score ~+1."""
    perfect = forward_return(prices, 1)
    assert information_coefficient(perfect, prices, 1) == pytest.approx(1.0, abs=1e-9)


def test_inverted_foresight_scores_near_minus_one(prices):
    """Informative but backwards is a different failure from noise."""
    assert information_coefficient(-forward_return(prices, 1), prices, 1) == pytest.approx(
        -1.0, abs=1e-9
    )


def test_pure_noise_scores_near_zero(prices):
    rng = np.random.default_rng(7)
    noise = pd.Series(rng.normal(size=len(prices)), index=prices.index)
    assert abs(information_coefficient(noise, prices, 1)) < 0.15


def test_constant_signal_is_a_real_zero_not_a_missing_measurement(prices):
    flat = pd.Series(0.6, index=prices.index)
    assert information_coefficient(flat, prices, 1) == 0.0


def test_too_little_overlap_returns_nan_not_zero(prices):
    """Unmeasured must never be reportable as measured-and-worthless."""
    short = prices.iloc[: MIN_IC_OBSERVATIONS - 5]
    sig = pd.Series(np.arange(len(short)), index=short.index)
    assert np.isnan(information_coefficient(sig, short, 1))


# ---------------------------------------------------------------------------
# ICIR and decay
# ---------------------------------------------------------------------------

def test_rolling_ic_of_perfect_signal_is_pinned_high(prices):
    series = rolling_ic(forward_return(prices, 1), prices, 1, window=63).dropna()
    assert len(series) > 0
    assert series.min() > 0.99


def test_icir_ranks_consistent_above_erratic(prices):
    """The ordering the metric exists to produce.

    `steady` is a weak signal that is always mildly right. `erratic` carries the
    same average information but delivers it in bursts. Total return cannot
    separate them; ICIR must.
    """
    rng = np.random.default_rng(11)
    fwd = forward_return(prices, 1).fillna(0.0)
    noise = pd.Series(rng.normal(size=len(prices)), index=prices.index)

    steady = 0.35 * fwd / fwd.std() + 0.65 * noise
    burst = noise.copy()
    half = len(prices) // 2
    burst.iloc[:half] = (fwd / fwd.std()).iloc[:half]

    assert ic_information_ratio(steady, prices, 1) > ic_information_ratio(burst, prices, 1)


def test_signal_decay_covers_every_horizon_and_counts_shrink(prices):
    table = signal_decay(forward_return(prices, 1), prices, horizons=(1, 5, 20))
    assert list(table.index) == [1, 5, 20]
    assert {"ic", "icir", "n_obs"} <= set(table.columns)
    # Longer horizons consume more of the tail, so fewer usable observations.
    assert table.loc[1, "n_obs"] > table.loc[20, "n_obs"]


def test_decay_shows_a_short_lived_signal_dying(prices):
    """A signal informative only about the very next bar must fade with horizon."""
    table = signal_decay(forward_return(prices, 1), prices, horizons=(1, 20))
    assert table.loc[1, "ic"] > 0.9
    assert abs(table.loc[20, "ic"]) < table.loc[1, "ic"]


# ---------------------------------------------------------------------------
# consistency
# ---------------------------------------------------------------------------

def test_consistency_rewards_the_steady_curve():
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    steady = pd.Series(100 * (1.0008 ** np.arange(252)), index=idx)

    lumpy = np.full(252, 100.0)
    lumpy[100:] = 122.0          # one jump, flat either side
    lumpy = pd.Series(lumpy, index=idx)

    steady_c, lumpy_c = consistency(steady), consistency(lumpy)
    assert steady_c["hit_rate"] == 1.0
    assert steady_c["ratio"] > lumpy_c["ratio"]


def test_consistency_reports_nothing_rather_than_guessing_on_short_curves():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    out = consistency(pd.Series(np.linspace(100, 110, 10), index=idx), window=21)
    assert out["n_windows"] == 0
    assert np.isnan(out["hit_rate"])
