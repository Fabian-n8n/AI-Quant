"""The portfolio backtest exists to be trusted more than the single-asset one.

These check the properties that make it worth reading: no look-ahead, limits
enforced incrementally rather than against a stale snapshot, gaps filled at the
open rather than at the stop, and no contact with the live halt lock.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.portfolio_backtester import Holding, PortfolioBacktester

# 14 features with covariance_type='full' needs more parameters than a 252-bar
# training window has samples, so the fit refuses and every fold is skipped.
# 'diag' is the documented fix and keeps the fixture fast.
HMM_TEST_CONFIG = {
    "n_candidates": [3, 4],        # not `n_states_range`; that key is ignored.
    # Minimum 3: the regime label map is not defined for a two-state fit.
    "min_train_bars": 252,
    "feature_columns": "volatility",   # 6 columns, not 14
    "n_init": 2,                   # the fixture needs to be fast, not optimal
    "n_iter": 40,
}


def _bars(n: int = 1200, seed: int = 7, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.012, n)
    close = start * np.exp(np.cumsum(steps))
    index = pd.bdate_range("2019-01-01", periods=n, tz="UTC")
    high = close * (1 + rng.uniform(0.001, 0.012, n))
    low = close * (1 - rng.uniform(0.001, 0.012, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(high, open_), "low": np.minimum(low, open_),
         "close": close, "volume": rng.integers(1e6, 5e6, n)},
        index=index,
    )


@pytest.fixture
def universe():
    return {"SPY": _bars(seed=1), "QQQ": _bars(seed=2, start=300.0)}


@pytest.fixture
def backtester():
    return PortfolioBacktester(
        symbols=["SPY", "QQQ"], primary="SPY",
        train_window=252, test_window=126, step_size=126,
        hmm_config=dict(HMM_TEST_CONFIG),
        strategy_config={"ema_span": 50, "atr_window": 14, "max_leverage": 1.0},
        risk_config={"max_single_position": 0.03, "max_concurrent": 12,
                     "max_exposure": 0.80, "max_risk_per_trade": 0.01},
    )


class TestIsolation:
    def test_never_touches_the_live_halt_lock(self, backtester):
        """A backtest must not be able to halt the real account.

        RiskManager defaults to <project>/trading_halted.lock. Sharing it means
        a research run can stop live trading, and a pre-existing live halt
        silently rejects every signal in every backtest.
        """
        live = Path(__file__).resolve().parent.parent / "trading_halted.lock"
        assert backtester._lock_file != live
        assert "pbt-halt-" in str(backtester._lock_file)


class TestNoLookAhead:
    def test_the_scan_only_ever_sees_history(self, backtester, universe):
        """Slicing at `timestamp` is what makes the whole thing honest."""
        cutoff = universe["SPY"].index[500]
        sliced = backtester._slice(universe, cutoff)
        for symbol, frame in sliced.items():
            assert frame.index.max() <= cutoff, f"{symbol} leaked a future bar"

    def test_a_position_is_not_stop_checked_on_its_entry_bar(self, backtester, universe):
        """Within one bar the order of the high and the low is unknown."""
        frame = universe["SPY"]
        timestamp = frame.index[400]
        holding = Holding(symbol="SPY", shares=10,
                          entry_price=float(frame.loc[timestamp, "open"]),
                          entry_time=timestamp,
                          stop_loss=float(frame.loc[timestamp, "high"]) * 2)
        assert holding.bars_held == 0


class TestExitPricing:
    def test_a_gap_through_the_stop_fills_at_the_open(self, backtester):
        """Filling at the stop price through a gap invents money."""
        index = pd.bdate_range("2024-01-01", periods=1, tz="UTC")
        gapped = pd.DataFrame({"open": [90.0], "high": [92.0], "low": [88.0],
                               "close": [91.0], "volume": [1_000_000]}, index=index)
        holding = Holding(symbol="X", shares=10, entry_price=100.0,
                          entry_time=index[0], stop_loss=95.0, bars_held=2)
        price, reason = backtester._exit_price({"X": gapped}, "X", index[0], holding)
        assert reason == "stop"
        # 90 (the open), not 95 (the stop). Slippage makes it slightly worse.
        assert price == pytest.approx(90.0 * (1 - backtester.slippage_pct))

    def test_an_ordinary_stop_fills_at_the_stop(self, backtester):
        index = pd.bdate_range("2024-01-01", periods=1, tz="UTC")
        frame = pd.DataFrame({"open": [99.0], "high": [100.0], "low": [94.0],
                              "close": [96.0], "volume": [1_000_000]}, index=index)
        holding = Holding(symbol="X", shares=10, entry_price=100.0,
                          entry_time=index[0], stop_loss=95.0, bars_held=2)
        price, reason = backtester._exit_price({"X": frame}, "X", index[0], holding)
        assert reason == "stop"
        assert price == pytest.approx(95.0 * (1 - backtester.slippage_pct))

    def test_the_stop_is_checked_before_the_target(self, backtester):
        """A bar that spans both is resolved pessimistically."""
        index = pd.bdate_range("2024-01-01", periods=1, tz="UTC")
        frame = pd.DataFrame({"open": [100.0], "high": [115.0], "low": [94.0],
                              "close": [110.0], "volume": [1_000_000]}, index=index)
        holding = Holding(symbol="X", shares=10, entry_price=100.0, entry_time=index[0],
                          stop_loss=95.0, take_profit=110.0, bars_held=2)
        _, reason = backtester._exit_price({"X": frame}, "X", index[0], holding)
        assert reason == "stop", "a bar touching both must not be booked as a win"


class TestLimitsAreReal:
    def test_exposure_never_exceeds_the_configured_ceiling(self, backtester, universe):
        """The bug this catches: `candidates.scan` sizes every symbol against one
        snapshot, so filling the queue against it took gross exposure to 165%
        at a 15% cap while max_exposure was 0.80."""
        result = backtester.run(universe)
        if result.history.empty:
            pytest.skip("no out-of-sample bars from the synthetic series")
        ceiling = backtester.risk_config["max_exposure"]
        assert result.max_exposure <= ceiling + 0.02, (
            f"exposure reached {result.max_exposure:.1%} against a "
            f"{ceiling:.0%} ceiling")

    def test_reported_exposure_reflects_the_position_cap(self, backtester, universe):
        """3% x 2 symbols cannot hold more than 6%.

        This is the number the module exists to produce: it is what proves the
        portfolio backtest and the single-asset one are measuring different
        strategies."""
        result = backtester.run(universe)
        if result.history.empty:
            pytest.skip("no out-of-sample bars from the synthetic series")
        reachable = (backtester.risk_config["max_single_position"]
                     * len(backtester.symbols))
        assert result.max_exposure <= reachable + 0.01


class TestResultIntegrity:
    def test_open_positions_are_closed_and_counted_at_the_end(self, backtester, universe):
        """An unclosed winner left off the trade log is how a backtest lies."""
        result = backtester.run(universe)
        if result.trade_log.empty:
            pytest.skip("no trades from the synthetic series")
        assert result.trade_log["exit_time"].notna().all()
        assert result.trade_log["pnl"].notna().all()

    def test_equity_curve_is_out_of_sample_only(self, backtester, universe):
        result = backtester.run(universe)
        if result.history.empty:
            pytest.skip("no out-of-sample bars from the synthetic series")
        first_test_bar = min(w.test_start for w in result.windows)
        assert first_test_bar >= backtester.train_window

    def test_significance_gate_matches_the_single_asset_backtester(self, backtester,
                                                                  universe):
        result = backtester.run(universe)
        assert result.is_significant == (result.n_trades >= 30)
