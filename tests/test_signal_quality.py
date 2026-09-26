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
    calibration_z,
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


# ===========================================================================
# calibration_z — are we as right as we said we would be?
#
# Adopted in idea from bennyjo/phil, whose own scorer reported z = -3.98 two
# months in. The metric is only worth carrying if it actually catches that,
# so these check it fires on a known-overconfident forecaster and stays quiet
# on an honest one.
# ===========================================================================

def test_an_honest_forecaster_scores_near_zero_sigma():
    rng = np.random.default_rng(0)
    stated = rng.uniform(0.3, 0.9, 600)
    wins = (rng.random(600) < stated).astype(int)      # reality matches the claim

    result = calibration_z(pd.Series(stated), pd.Series(wins))

    assert abs(result["z"]) < 2, f"honest forecaster flagged at z={result['z']:.2f}"
    assert result["verdict"] == "consistent with the stated confidence"
    assert result["n_trades"] == 600


def test_an_overconfident_forecaster_is_caught_and_named():
    """Claims p, wins at p - 0.25. This is the failure mode that matters:
    position sizes keyed to confidence are largest when least deserved."""
    rng = np.random.default_rng(1)
    stated = rng.uniform(0.3, 0.9, 600)
    wins = (rng.random(600) < np.clip(stated - 0.25, 0, 1)).astype(int)

    result = calibration_z(pd.Series(stated), pd.Series(wins))

    assert result["z"] < -3, f"missed a 25-point gap, z={result['z']:.2f}"
    assert "OVERCONFIDENT" in result["verdict"]
    assert result["actual_wins"] < result["expected_wins"]


def test_underconfidence_is_reported_separately_not_as_a_pass():
    rng = np.random.default_rng(2)
    stated = rng.uniform(0.2, 0.6, 600)
    wins = (rng.random(600) < np.clip(stated + 0.25, 0, 1)).astype(int)

    result = calibration_z(pd.Series(stated), pd.Series(wins))
    assert result["z"] > 3
    assert "underconfident" in result["verdict"]


def test_it_refuses_to_speak_on_too_few_trades():
    """Six trades cannot tell you about calibration, and a confident-looking
    sigma computed from six would be worse than no number."""
    result = calibration_z(pd.Series([0.6] * 6), pd.Series([1, 0, 1, 0, 1, 1]))
    assert np.isnan(result["z"])
    assert result["verdict"] == "not enough trades"
    # The Brier score is still returned: it needs no sample-size excuse.
    assert np.isfinite(result["brier"])


def test_claims_pinned_at_zero_or_one_have_no_variance_to_test():
    result = calibration_z(pd.Series([1.0] * 50), pd.Series([1] * 50))
    assert np.isnan(result["z"])
    assert "no spread" in result["verdict"]


def test_a_probability_outside_zero_to_one_is_dropped_not_clipped():
    """Clipping would hide a broken upstream scorer behind a plausible z."""
    stated = pd.Series([1.4, -0.2, 3.0] + [0.5] * 40)
    wins = pd.Series([1] * 43)

    result = calibration_z(stated, wins)
    assert result["dropped_out_of_range"] == 3
    assert result["n_trades"] == 40


def test_brier_matches_the_hand_computed_value():
    """One arithmetic anchor so a refactor cannot silently change the scale."""
    result = calibration_z(pd.Series([0.0, 1.0]), pd.Series([1, 1]), min_trades=1)
    # (0 - 1)^2 and (1 - 1)^2, averaged.
    assert result["brier"] == pytest.approx(0.5)
