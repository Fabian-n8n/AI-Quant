"""The multiple-testing correction has to be right, or it is worse than nothing.

A miscalibrated correction that passes everything gives false confidence with
a statistical veneer, which is more dangerous than no correction at all. These
check calibration against known answers rather than only checking that the
functions run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.trials import (
    Trial,
    TrialRegistry,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


class TestProbabilisticSharpe:
    def test_calibrated_on_strategies_with_no_edge(self):
        """PSR must be uniform on [0,1] when the true Sharpe is zero.

        If it skews high, every worthless strategy looks promising.
        """
        rng = np.random.default_rng(42)
        values = [probabilistic_sharpe_ratio(pd.Series(rng.normal(0, 0.01, 1500)))
                  for _ in range(300)]
        assert 0.42 < float(np.mean(values)) < 0.58
        # A 0.95 threshold should admit about 5% of pure noise, not 30%.
        assert float(np.mean(np.array(values) > 0.95)) < 0.10

    def test_a_real_edge_is_recognised(self):
        rng = np.random.default_rng(7)
        good = pd.Series(rng.normal(0.0008, 0.01, 2000))
        assert probabilistic_sharpe_ratio(good) > 0.95

    def test_a_short_sample_is_punished(self):
        """The same Sharpe over fewer bars must be less convincing."""
        rng = np.random.default_rng(3)
        long_run = pd.Series(rng.normal(0.0006, 0.01, 2000))
        short_run = long_run.iloc[:60]
        assert probabilistic_sharpe_ratio(long_run) > probabilistic_sharpe_ratio(short_run)

    def test_degenerate_inputs_do_not_raise(self):
        assert np.isnan(probabilistic_sharpe_ratio(pd.Series([0.01, 0.01, 0.01])))
        assert np.isnan(probabilistic_sharpe_ratio(pd.Series([0.01])))


class TestHurdle:
    def test_one_trial_has_no_hurdle(self):
        """Testing a single idea is not a search, so nothing to correct."""
        assert expected_max_sharpe(1, 0.02) == 0.0

    def test_more_trials_raise_the_bar(self):
        assert expected_max_sharpe(100, 0.02) > expected_max_sharpe(10, 0.02) > 0

    def test_agreeing_trials_raise_a_lower_bar_than_scattered_ones(self):
        """Variance across trials is half the correction.

        If every configuration scored the same, the best is not lucky. If they
        scattered, the search had room to find a fluke.
        """
        assert expected_max_sharpe(20, 0.001) < expected_max_sharpe(20, 0.05)

    def test_no_variance_means_no_hurdle(self):
        assert expected_max_sharpe(50, 0.0) == 0.0


class TestDeflatedSharpe:
    def test_rejects_the_winner_of_a_search_over_pure_noise(self):
        """The whole point. Best-of-20 on noise must not pass.

        An uncorrected PSR passes most of these, which is exactly how a swept
        parameter grid produces a strategy that fails in production.
        """
        rng = np.random.default_rng(11)
        passed = 0
        for _ in range(40):
            arms = [pd.Series(rng.normal(0, 0.01, 1500)) for _ in range(20)]
            sharpes = [float(a.mean() / a.std() * np.sqrt(252)) for a in arms]
            best = arms[int(np.argmax(sharpes))]
            if deflated_sharpe_ratio(best, sharpes)["dsr"] > 0.95:
                passed += 1
        assert passed / 40 < 0.15, f"{passed}/40 noise searches wrongly passed"

    def test_is_never_more_generous_than_the_uncorrected_number(self):
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0.0006, 0.01, 1500))
        sharpes = [float(rng.normal(0.5, 0.5)) for _ in range(25)]
        verdict = deflated_sharpe_ratio(returns, sharpes)
        assert verdict["dsr"] <= verdict["psr"] + 1e-9

    def test_the_hurdle_is_reported_alongside_the_verdict(self):
        """A bare probability invites 'compared to what'."""
        rng = np.random.default_rng(9)
        verdict = deflated_sharpe_ratio(pd.Series(rng.normal(0.0005, 0.01, 800)),
                                        [0.8, 0.3, -0.2, 0.5, 1.1])
        assert set(verdict) == {"dsr", "psr", "sr_annual", "hurdle_annual", "n_trials"}
        assert verdict["n_trials"] == 5

    def test_non_finite_trial_sharpes_are_ignored(self):
        rng = np.random.default_rng(13)
        verdict = deflated_sharpe_ratio(pd.Series(rng.normal(0.0005, 0.01, 800)),
                                        [0.8, float("nan"), None, 0.3])
        assert verdict["n_trials"] == 2


class TestRegistry:
    def test_round_trips_a_trial(self, tmp_path):
        registry = TrialRegistry(tmp_path / "t.db")
        registry.record(Trial(study="risk", label="3% cap", config={"cap": 0.03},
                              total_return=0.13, sharpe=0.71, max_drawdown=-0.11,
                              avg_exposure=0.30, n_trades=834, n_bars=504))
        rows = registry.all()
        assert len(rows) == 1
        assert rows[0]["label"] == "3% cap"
        assert registry.sharpes() == [0.71]

    def test_counts_every_arm_including_the_bad_ones(self, tmp_path):
        """Counting only the survivors understates the search.

        That is the same mistake the correction exists to prevent, so the
        registry has no delete method and the count is unconditional.
        """
        registry = TrialRegistry(tmp_path / "t.db")
        for i, sharpe in enumerate([1.2, -0.4, 0.1, -1.1]):
            registry.record(Trial(study="risk", label=f"arm{i}", sharpe=sharpe))
        assert registry.count() == 4
        assert len(registry.sharpes()) == 4
        assert not hasattr(registry, "delete")

    def test_studies_are_separable_but_the_global_count_is_the_honest_one(self, tmp_path):
        registry = TrialRegistry(tmp_path / "t.db")
        registry.record(Trial(study="a", label="x", sharpe=1.0))
        registry.record(Trial(study="b", label="y", sharpe=0.2))
        assert registry.count("a") == 1
        assert registry.count() == 2
        assert len(registry.sharpes()) == 2


class TestAblationSwitch:
    def test_rejects_an_unknown_regime_mode(self):
        from backtest.portfolio_backtester import PortfolioBacktester

        with pytest.raises(ValueError, match="hmm|fixed|shuffled"):
            PortfolioBacktester(symbols=["SPY"], regime_mode="magic")
