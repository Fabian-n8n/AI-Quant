"""
Tests for core/hmm_engine.py.

Run against synthetic data with three constructed volatility regimes. That is
deliberate: real market data cannot tell you whether the model found the right
answer, because nobody knows the right answer. Here the regimes are built in, so
a detector that cannot separate 0.6% daily vol from 3.2% is broken rather than
unlucky.
"""

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import (
    REGIME_LABEL_SETS,
    HMMEngine,
    InsufficientDataError,
    Regime,
    RegimeInfo,
    RegimeState,
    RegimeTracker,
    VolatilityRank,
    count_parameters,
    check_fittability,
    resolve_feature_columns,
)
from data.feature_engineering import FEATURE_COLUMNS, VOLATILITY_FEATURES


# -- labelling --------------------------------------------------------------

def test_label_sets_cover_every_candidate():
    """Every count in the default n_candidates needs a label set, or fit()
    raises a KeyError after the expensive part has already run."""
    for n in [3, 4, 5, 6, 7]:
        assert n in REGIME_LABEL_SETS
        assert len(REGIME_LABEL_SETS[n]) == n
        assert len(set(REGIME_LABEL_SETS[n])) == n, "labels must be distinct"


def test_label_sets_match_the_spec():
    assert REGIME_LABEL_SETS[3] == [Regime.BEAR, Regime.NEUTRAL, Regime.BULL]
    assert REGIME_LABEL_SETS[5] == [
        Regime.CRASH, Regime.BEAR, Regime.NEUTRAL, Regime.BULL, Regime.EUPHORIA,
    ]
    assert REGIME_LABEL_SETS[7][0] == Regime.CRASH
    assert REGIME_LABEL_SETS[7][-1] == Regime.EUPHORIA


def test_labels_are_ordered_by_mean_return(fitted_engine):
    """The label order must track the return order.

    This is what makes labels stable across refits. EM numbers its states
    arbitrarily and renumbers them every time, so without sorting, "state 3"
    would mean something different after each retrain.
    """
    order = REGIME_LABEL_SETS[fitted_engine.n_states]
    by_return = sorted(fitted_engine.regime_info.values(), key=lambda i: i.expected_return)
    assert [Regime(i.regime_name) for i in by_return] == order


def test_expected_return_is_in_real_units_not_z_scores(fitted_engine):
    """Regime statistics must come from raw returns, not the model's means_.

    means_ lives in standardised feature space, so a mean read off it is a
    unitless number that looks like a percentage and is not one. Annualised
    returns from real data land in a plausible band; z-scores would cluster
    tightly around zero.
    """
    returns = [i.expected_return for i in fitted_engine.regime_info.values()]
    assert max(returns) - min(returns) > 0.05, "regimes should differ in real return"
    assert all(-5.0 < r < 5.0 for r in returns)
    vols = [i.expected_volatility for i in fitted_engine.regime_info.values()]
    assert all(v > 0 for v in vols), "annualised volatility is positive by construction"


# -- model selection --------------------------------------------------------

def test_parameter_counts_match_the_closed_form():
    """n-1 start, n(n-1) transition, n*d means, n*d(d+1)/2 full covariance."""
    assert count_parameters(3, 14, "full") == 2 + 6 + 42 + 315
    assert count_parameters(5, 14, "full") == 4 + 20 + 70 + 525
    assert count_parameters(5, 14, "diag") == 4 + 20 + 70 + 70
    assert count_parameters(7, 6, "full") == 6 + 42 + 42 + 147


def test_fittability_check_rejects_the_spec_defaults_at_minimum_data():
    """14 features, full covariance, 504 rows: n=5/6/7 need more parameters
    than there are samples.

    The failure mode this prevents is not a crash. It is a fit that appears to
    work while BIC's penalty silently collapses selection onto n=3 regardless
    of what the data says, so "automatic model selection" stops selecting.
    """
    with pytest.raises(InsufficientDataError, match="Model too large"):
        check_fittability(504, 14, [3, 4, 5, 6, 7], "full")


def test_fittability_check_accepts_the_volatility_subset():
    counts = check_fittability(504, 6, [3, 4, 5, 6, 7], "full")
    assert max(counts.values()) < 504


def test_fittability_check_accepts_full_features_with_enough_data():
    """The constraint is the ratio, not the feature count. With 2000 rows the
    full set fits."""
    counts = check_fittability(2000, 14, [3, 4, 5, 6, 7], "full")
    assert max(counts.values()) < 2000


def test_bic_scores_recorded_for_every_candidate(fitted_engine):
    """All candidate scores are kept, not just the winner, so the selection can
    be second-guessed later."""
    scores = fitted_engine.metadata.all_bic_scores
    assert len(scores) >= 3
    assert fitted_engine.metadata.bic == pytest.approx(min(scores.values()))
    assert scores[fitted_engine.n_states] == pytest.approx(fitted_engine.metadata.bic)


def test_selected_state_count_is_in_the_candidate_range(fitted_engine):
    assert fitted_engine.n_states in fitted_engine.n_candidates


def test_fit_is_deterministic(features, returns):
    """Same data must give the same model. Without a fixed seed the backtest is
    not reproducible, and CLAUDE.md requires the core to be deterministic."""
    a = HMMEngine(feature_columns=VOLATILITY_FEATURES, n_init=3, random_state=42).fit(features, returns)
    b = HMMEngine(feature_columns=VOLATILITY_FEATURES, n_init=3, random_state=42).fit(features, returns)
    assert a.n_states == b.n_states
    assert a.metadata.bic == pytest.approx(b.metadata.bic)
    np.testing.assert_allclose(a.model.transmat_, b.model.transmat_)


def test_refuses_to_fit_below_min_train_bars(features, returns):
    with pytest.raises(InsufficientDataError, match="at least"):
        HMMEngine(feature_columns=VOLATILITY_FEATURES, min_train_bars=504).fit(
            features.iloc[:100], returns
        )


# -- volatility ranking -----------------------------------------------------

def test_volatility_rank_is_independent_of_the_label(fitted_engine):
    """The spec's key architectural point: labels sort by return, the strategy
    sorts by volatility, and the two orderings genuinely disagree.

    Crash and euphoria sit at opposite ends of the return sort and next to each
    other on the volatility sort. If ranks tracked labels, the strategy would be
    sizing off direction rather than turbulence.
    """
    infos = list(fitted_engine.regime_info.values())
    by_return = [i.regime_id for i in sorted(infos, key=lambda i: i.expected_return)]
    by_vol = [i.regime_id for i in sorted(infos, key=lambda i: i.expected_volatility)]
    assert by_return != by_vol, "return and volatility orderings should differ"


def test_every_state_has_a_volatility_rank(fitted_engine):
    for state in range(fitted_engine.n_states):
        assert isinstance(fitted_engine.get_volatility_rank(state), VolatilityRank)


def test_high_volatility_states_get_no_leverage(fitted_engine):
    """Leverage is only ever offered in low-volatility regimes."""
    for info in fitted_engine.regime_info.values():
        if info.volatility_rank == VolatilityRank.HIGH:
            assert info.max_leverage_allowed == 1.0


# -- stability filter -------------------------------------------------------

def test_regime_change_needs_consecutive_bars():
    """A one-bar blip must not change the acted-on regime."""
    tracker = RegimeTracker(stability_bars=3)
    for _ in range(5):
        tracker.update(0)
    assert tracker.update(1)["state_id"] == 0, "one bar is not a regime change"
    assert tracker.update(1)["state_id"] == 0, "two bars is not either"
    assert tracker.update(1)["state_id"] == 1, "three consecutive bars confirms"


def test_transition_reduces_size_by_25_percent():
    tracker = RegimeTracker(stability_bars=3, transition_size_mult=0.75)
    for _ in range(5):
        tracker.update(0)
    assert tracker.update(1)["size_multiplier"] == 0.75
    assert tracker.update(0)["size_multiplier"] == 1.0, "back to the confirmed regime"


def test_alternating_states_never_confirm():
    """A model flipping every bar must never move the acted-on regime.

    This is the case the filter exists for: without it, the system rebalances
    on every flip and pays slippage to chase noise.
    """
    tracker = RegimeTracker(stability_bars=3)
    tracker.update(0)
    for i in range(1, 40):
        result = tracker.update(i % 2)
    assert result["state_id"] == 0


def test_flicker_detection_forces_uncertainty_mode():
    """More than flicker_threshold raw changes in the window triggers the
    deeper size cut."""
    tracker = RegimeTracker(stability_bars=3, flicker_window=20,
                            flicker_threshold=4, uncertainty_size_mult=0.5)
    result = None
    for i in range(20):
        result = tracker.update(i % 2)
    assert tracker.get_flicker_rate() > 4
    assert tracker.is_flickering()
    assert result["size_multiplier"] == 0.5, "uncertainty cut wins over transition cut"


def test_stable_sequence_does_not_flicker():
    tracker = RegimeTracker(flicker_threshold=4)
    for _ in range(30):
        result = tracker.update(2)
    assert tracker.get_flicker_rate() == 0
    assert not tracker.is_flickering()
    assert result["size_multiplier"] == 1.0
    assert result["is_confirmed"]


def test_flicker_counts_raw_changes_not_confirmed_ones():
    """Counting confirmed changes would be circular: the transition damper
    suppresses exactly the flips the flicker detector needs to see, so the
    detector would almost never fire."""
    tracker = RegimeTracker(stability_bars=3, flicker_window=20, flicker_threshold=4)
    for i in range(20):
        tracker.update(i % 2)
    assert tracker.confirmed == 0, "no change was ever confirmed"
    assert tracker.get_flicker_rate() > 4, "but raw flicker is still detected"


def test_stability_filter_suppresses_changes_in_practice(fitted_engine, features):
    classified = fitted_engine.classify_series(features)
    raw_changes = (classified["raw_state_id"].diff() != 0).sum()
    confirmed_changes = (classified["state_id"].diff() != 0).sum()
    assert confirmed_changes <= raw_changes


# -- introspection ----------------------------------------------------------

def test_transition_matrix_rows_sum_to_one(fitted_engine):
    matrix = fitted_engine.get_transition_matrix()
    np.testing.assert_allclose(matrix.sum(axis=1).to_numpy(), 1.0, atol=1e-8)
    assert matrix.shape == (fitted_engine.n_states, fitted_engine.n_states)


def test_regimes_are_persistent_not_noise(fitted_engine):
    """The transition matrix diagonal is the probability a regime survives to
    the next bar, so 1/(1-diagonal) is its expected duration.

    A diagonal near 0.5 would mean the "regimes" last two days, which is a
    model describing noise rather than market states.
    """
    durations = fitted_engine.get_expected_durations()
    assert durations.median() > 5, f"regimes too short-lived: {durations.to_dict()}"


def test_summary_has_a_row_per_state(fitted_engine):
    summary = fitted_engine.summary()
    assert len(summary) == fitted_engine.n_states
    assert summary["ann_return"].is_monotonic_increasing


# -- classification output --------------------------------------------------

def test_classify_returns_a_populated_regime_state(fitted_engine, features):
    fitted_engine.tracker.reset()
    state = fitted_engine.classify(features.iloc[:600])
    assert isinstance(state, RegimeState)
    assert isinstance(state.label, Regime)
    assert 0.0 <= state.probability <= 1.0
    assert pytest.approx(sum(state.state_probabilities.values()), abs=1e-8) == 1.0
    assert state.timestamp == features.index[599]


def test_classify_series_covers_every_bar(fitted_engine, features):
    classified = fitted_engine.classify_series(features)
    assert len(classified) == len(features)
    assert classified.index.equals(features.index)
    for column in ("state_id", "label", "probability", "is_confirmed",
                   "size_multiplier", "volatility_rank", "meets_confidence"):
        assert column in classified.columns


def test_probabilities_sum_to_one_on_every_bar(fitted_engine, features):
    proba = fitted_engine.predict_regime_proba(features.iloc[:400])
    np.testing.assert_allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_min_confidence_is_reported_not_silently_applied(fitted_engine, features):
    """The engine reports whether confidence was met and leaves the decision to
    the strategy layer. Swallowing it here would hide the reason a signal never
    appeared."""
    classified = fitted_engine.classify_series(features.iloc[:500])
    expected = classified["probability"] >= fitted_engine.min_confidence
    assert (classified["meets_confidence"] == expected).all()


# -- configuration ----------------------------------------------------------

def test_feature_column_aliases_resolve():
    assert resolve_feature_columns("all") == FEATURE_COLUMNS
    assert resolve_feature_columns(None) == FEATURE_COLUMNS
    assert resolve_feature_columns("volatility") == VOLATILITY_FEATURES
    assert resolve_feature_columns(["adx_14"]) == ["adx_14"]


def test_bad_feature_alias_is_rejected():
    with pytest.raises(ValueError, match="unknown feature_columns alias"):
        resolve_feature_columns("typo")
    with pytest.raises(ValueError, match="unknown feature columns"):
        resolve_feature_columns(["not_a_feature"])


def test_engine_constructs_from_settings_yaml(settings):
    """The hmm block must map onto the constructor with no translation layer.

    Catches the common drift where a parameter is added to the YAML and never
    wired in, so it silently does nothing.
    """
    engine = HMMEngine(**settings["hmm"])
    assert engine.n_candidates == [3, 4, 5, 6, 7]
    assert engine.covariance_type == settings["hmm"]["covariance_type"]
    assert engine.min_train_bars == settings["hmm"]["min_train_bars"]
    assert engine.feature_columns


def test_column_order_is_enforced(fitted_engine, features):
    """A reordered column would rotate every covariance matrix and produce
    confident nonsense rather than an error, so the order is pinned."""
    shuffled = features[list(reversed(features.columns))]
    ordered = fitted_engine._prepare_matrix(shuffled)
    expected = features[fitted_engine.feature_columns].to_numpy()
    np.testing.assert_allclose(ordered, expected)


def test_nan_features_are_rejected(fitted_engine, features):
    """NaN in a column the engine reads must raise, not propagate into the fit.

    Poisons a column from the engine's own configured set. Corrupting an unused
    column is correctly ignored, since _prepare_matrix selects before checking.
    """
    dirty = features.copy()
    dirty.loc[dirty.index[5], fitted_engine.feature_columns[0]] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        fitted_engine._prepare_matrix(dirty)


def test_unused_columns_may_be_dirty(fitted_engine, features):
    """A NaN outside the configured feature set is not the engine's problem.

    Worth pinning: it means callers can pass the full 14-column matrix while the
    engine reads only its 6, which is exactly how classify_series is used.
    """
    unused = [c for c in features.columns if c not in fitted_engine.feature_columns]
    dirty = features.copy()
    dirty.loc[dirty.index[5], unused[0]] = np.nan
    fitted_engine._prepare_matrix(dirty)


def test_unfitted_engine_raises_rather_than_guessing():
    engine = HMMEngine(feature_columns=VOLATILITY_FEATURES)
    assert not engine.is_fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        engine.get_transition_matrix()


# -- persistence ------------------------------------------------------------

def test_save_and_load_round_trip(fitted_engine, features, tmp_path):
    path = fitted_engine.save(tmp_path / "hmm.pkl")
    restored = HMMEngine.load(path)

    assert restored.n_states == fitted_engine.n_states
    assert restored.state_labels == fitted_engine.state_labels
    assert restored.feature_columns == fitted_engine.feature_columns
    np.testing.assert_allclose(
        restored.predict_regime_proba(features.iloc[:300]).to_numpy(),
        fitted_engine.predict_regime_proba(features.iloc[:300]).to_numpy(),
    )


def test_metadata_records_what_the_model_is(fitted_engine):
    meta = fitted_engine.metadata
    assert meta.n_regimes == fitted_engine.n_states
    assert meta.training_date is not None
    assert meta.n_train_samples > 0
    assert meta.feature_columns == fitted_engine.feature_columns
    assert len(meta.labels) == fitted_engine.n_states
    assert meta.n_parameters > 0
    assert np.isfinite(meta.log_likelihood)


def test_should_retrain_respects_the_interval(features, returns):
    engine = HMMEngine(feature_columns=VOLATILITY_FEATURES, n_init=2,
                       retrain_interval_bars=5)
    assert engine.should_retrain(), "an unfitted engine always needs training"
    engine.fit(features, returns)
    assert not engine.should_retrain()
    for i in range(5):
        engine.classify(features.iloc[: 600 + i])
    assert engine.should_retrain()


# -- the model actually works ------------------------------------------------

def test_detects_the_constructed_volatility_regimes(fitted_engine, features, synthetic_bars):
    """The end-to-end check: bars from the high-volatility block must be
    classified into higher-volatility states than bars from the calm block.

    Everything else in this file tests mechanics. This tests whether the brain
    works at all.
    """
    classified = fitted_engine.classify_series(features)
    block = pd.Series(
        (np.arange(len(synthetic_bars)) // 180) % 3, index=synthetic_bars.index
    ).reindex(classified.index)

    state_vol = {i.regime_id: i.expected_volatility for i in fitted_engine.regime_info.values()}
    assigned_vol = classified["state_id"].map(state_vol)

    calm = assigned_vol[block == 0].mean()
    turbulent = assigned_vol[block == 2].mean()
    assert turbulent > calm, (
        f"turbulent blocks got mean vol {turbulent:.3f}, calm got {calm:.3f}: "
        "the classifier is not separating the regimes it was built to find"
    )
