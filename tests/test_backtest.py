"""
Tests for the walk-forward backtester, metrics and stress testing.

The allocation math tests are the ones that matter most. A backtester with a
subtly wrong equity calculation does not fail loudly: it produces a plausible
number that gets believed, and every decision after it is built on fiction.
"""

import numpy as np
import pandas as pd
import pytest

from backtest import performance
from backtest.backtester import WalkForwardBacktester, _PortfolioState
from backtest.stress_test import (
    CRASH_SCENARIOS,
    StressTester,
    evaluate_breakers,
    inject_crashes,
    inject_gaps,
)


@pytest.fixture(scope="module")
def backtester(settings_module):
    strategy_config = dict(settings_module["strategy"])
    strategy_config["min_confidence"] = settings_module["hmm"]["min_confidence"]
    hmm_config = dict(settings_module["hmm"])
    hmm_config["n_init"] = 2          # keep the suite quick
    hmm_config["n_candidates"] = [3, 4]
    return WalkForwardBacktester(
        **settings_module["backtest"],
        hmm_config=hmm_config,
        strategy_config=strategy_config,
    )


@pytest.fixture(scope="module")
def settings_module():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parent.parent
    with open(root / "config" / "settings.yaml") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def backtest_result(backtester, synthetic_bars_module):
    return backtester.run(synthetic_bars_module, "TEST")


@pytest.fixture(scope="module")
def synthetic_bars_module():
    from data.market_data import synthetic_bars

    return synthetic_bars()


# -- allocation math --------------------------------------------------------

def test_allocation_math_matches_the_spec():
    """equity = cash + shares*price;  target_shares = int(equity*alloc/price)."""
    state = _PortfolioState(cash=100_000.0)
    state.rebalance(0.95, 100.0, slippage_pct=0.0, commission=0.0)
    assert state.shares == 950
    assert state.cash == pytest.approx(5_000.0)
    assert state.equity_at(100.0) == pytest.approx(100_000.0)


def test_leverage_produces_negative_cash_not_an_error():
    """target_allocation above 1.0 means the position costs more than equity.

    Cash goes negative and that is margin, not a bug. Clamping it to zero would
    silently cap leverage at 1.0 and the backtest would stop measuring the
    strategy that was configured.
    """
    state = _PortfolioState(cash=100_000.0)
    state.rebalance(1.1875, 100.0, slippage_pct=0.0, commission=0.0)
    assert state.shares == 1187
    assert state.cash < 0
    assert state.equity_at(100.0) == pytest.approx(100_000.0, abs=1.0)


def test_equity_is_correct_after_a_price_move():
    state = _PortfolioState(cash=100_000.0)
    state.rebalance(1.0, 100.0, 0.0, 0.0)
    assert state.equity_at(110.0) == pytest.approx(state.cash + 1000 * 110.0)
    assert state.equity_at(110.0) > 100_000


def test_shares_are_truncated_never_rounded_up():
    """int() truncates, so the position can only ever be under-allocated.

    Rounding up would let a rebalance exceed the target allocation, which at
    1.25x leverage means quietly exceeding the leverage limit too.
    """
    state = _PortfolioState(cash=1000.0)
    state.rebalance(1.0, 300.0, 0.0, 0.0)
    assert state.shares == 3          # 3.33 truncated, not 4


def test_slippage_always_works_against_the_trade():
    buy = _PortfolioState(cash=100_000.0)
    buy.rebalance(0.95, 100.0, slippage_pct=0.001, commission=0.0)
    clean = _PortfolioState(cash=100_000.0)
    clean.rebalance(0.95, 100.0, slippage_pct=0.0, commission=0.0)
    assert buy.cash < clean.cash, "buying should cost more with slippage"

    seller = _PortfolioState(cash=0.0)
    seller.shares = 1000
    before = seller.equity_at(100.0)
    seller.rebalance(0.0, 100.0, slippage_pct=0.001, commission=0.0)
    assert seller.cash < before, "selling should realise less with slippage"


def test_no_rebalance_when_target_already_held():
    state = _PortfolioState(cash=100_000.0)
    state.rebalance(0.95, 100.0, 0.0, 0.0)
    assert state.rebalance(0.95, 100.0, 0.0, 0.0) is None


# -- windows ----------------------------------------------------------------

def test_train_and_test_windows_never_overlap(backtester):
    """The defining property of walk-forward. Overlap means the model is being
    evaluated on data it trained on, which is just an in-sample result."""
    for window in backtester.build_windows(2000):
        assert window.test_start == window.train_end
        assert window.train_start < window.train_end < window.test_end


def test_window_sizes_match_config(backtester):
    windows = backtester.build_windows(2000)
    assert windows
    for window in windows[:-1]:
        assert window.train_size == backtester.train_window
        assert window.test_size == backtester.test_window


def test_windows_step_forward_contiguously(backtester):
    """With step == test_window the OOS periods join up, so the stitched equity
    curve is continuous rather than a set of disconnected fragments."""
    windows = backtester.build_windows(2000)
    for earlier, later in zip(windows, windows[1:], strict=False):
        assert later.train_start == earlier.train_start + backtester.step_size
        assert later.test_start == earlier.test_end


def test_too_little_data_yields_no_windows(backtester):
    assert backtester.build_windows(100) == []


def test_run_rejects_insufficient_history(backtester, synthetic_bars_module):
    """The error must name the warmup, since 'not enough bars' is confusing when
    you supplied more bars than the window size."""
    with pytest.raises(ValueError, match="warmup"):
        backtester.run(synthetic_bars_module.iloc[:600], "SHORT")


# -- the run ----------------------------------------------------------------

def test_produces_a_complete_result(backtest_result):
    assert backtest_result.n_folds > 0
    assert not backtest_result.equity_curve.empty
    assert not backtest_result.regime_history.empty
    assert not backtest_result.trade_log.empty


def test_equity_curve_has_no_gaps_or_nans(backtest_result):
    equity = backtest_result.equity_curve
    assert equity.notna().all()
    assert (equity > 0).all(), "equity going non-positive means a margin bug"
    assert equity.index.is_monotonic_increasing
    assert not equity.index.has_duplicates


def test_every_trade_is_closed_and_priced(backtest_result):
    """An open final trade would silently drop its P&L from every metric."""
    trades = backtest_result.trade_log
    assert trades["pnl"].notna().all()
    assert trades["exit_time"].notna().all()


def test_regime_history_covers_only_out_of_sample_bars(backtest_result, backtester):
    """No in-sample bar may appear in the results. This is the whole point."""
    expected = sum(w.test_size for w in backtest_result.windows)
    assert len(backtest_result.regime_history) == expected


def test_result_reports_significance(backtest_result):
    assert backtest_result.is_significant == (backtest_result.n_trades >= 30)


def test_rebalance_threshold_limits_churn(backtest_result):
    """Rebalancing every bar would mean thousands of trades and slippage eating
    the return. The 10% threshold should keep it to a small fraction."""
    bars = len(backtest_result.equity_curve)
    assert backtest_result.n_trades < bars * 0.25


# -- metrics ----------------------------------------------------------------

def test_total_return_and_cagr_agree_in_sign():
    equity = pd.Series([100.0, 110.0, 120.0], index=pd.bdate_range("2024-01-01", periods=3))
    assert performance.total_return(equity) == pytest.approx(0.20)
    assert performance.cagr(equity) > 0


def test_max_drawdown_finds_the_worst_trough():
    equity = pd.Series(
        [100, 120, 60, 80, 130], index=pd.bdate_range("2024-01-01", periods=5), dtype=float
    )
    depth, duration, trough = performance.max_drawdown(equity)
    assert depth == pytest.approx(-0.5)      # 120 -> 60
    assert duration > 0


def test_sharpe_is_zero_for_a_flat_curve():
    flat = pd.Series([100.0] * 50, index=pd.bdate_range("2024-01-01", periods=50))
    assert performance.sharpe_ratio(flat.pct_change()) == 0.0


def test_sortino_ignores_upside_volatility():
    """Two series, same mean, one with violent upside. Sortino should not
    punish the second; Sharpe does."""
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2020-01-01", periods=500)
    steady = pd.Series(rng.normal(0.0005, 0.005, 500), index=index)
    spiky = steady.copy()
    spiky.iloc[::50] += 0.05
    assert performance.sortino_ratio(spiky) > performance.sortino_ratio(steady)


def test_trade_stats_flags_insufficient_samples():
    """Under 30 trades a win rate is a coin landing heads. It must be flagged,
    not printed as a number."""
    trades = pd.DataFrame({
        "pnl": [100.0, -50.0, 200.0], "return_pct": [0.01, -0.005, 0.02],
        "bars_held": [5, 3, 8], "slippage_cost": [1.0, 1.0, 1.0],
    })
    stats = performance.trade_stats(trades)
    assert stats["n_trades"] == 3
    assert stats["significant"] is False
    assert stats["win_rate"] == pytest.approx(2 / 3)


def test_profit_factor_and_expectancy():
    trades = pd.DataFrame({
        "pnl": [300.0, -100.0, 200.0, -100.0],
        "return_pct": [0.03, -0.01, 0.02, -0.01],
        "bars_held": [5, 5, 5, 5], "slippage_cost": [0.0] * 4,
    })
    stats = performance.trade_stats(trades)
    assert stats["profit_factor"] == pytest.approx(500 / 200)
    assert stats["expectancy"] == pytest.approx(75.0)


def test_max_consecutive_losses():
    trades = pd.DataFrame({
        "pnl": [100.0, -10.0, -10.0, -10.0, 50.0, -10.0],
        "return_pct": [0.0] * 6, "bars_held": [1] * 6, "slippage_cost": [0.0] * 6,
    })
    assert performance.trade_stats(trades)["max_consecutive_losses"] == 3


def test_alpha_beta_recovers_a_known_relationship():
    """Construct a series that is exactly 0.5x the benchmark. Beta must be 0.5."""
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2020-01-01", periods=400)
    benchmark_returns = pd.Series(rng.normal(0.0004, 0.01, 400), index=index)
    benchmark = 100 * (1 + benchmark_returns).cumprod()
    strategy = 100 * (1 + benchmark_returns * 0.5).cumprod()
    result = performance.alpha_beta(strategy, benchmark)
    assert result["beta"] == pytest.approx(0.5, abs=0.05)
    assert result["r_squared"] > 0.95


def test_regime_breakdown_covers_every_regime(backtest_result):
    table = performance.regime_breakdown(backtest_result.regime_history, backtest_result.trade_log)
    assert not table.empty
    assert set(table["regime"]) == set(backtest_result.regime_history["regime"].unique())
    assert table["pct_time_in"].sum() == pytest.approx(1.0)


def test_confidence_buckets_use_the_spec_edges(backtest_result):
    table = performance.confidence_buckets(backtest_result.regime_history, backtest_result.trade_log)
    assert not table.empty
    assert set(table["confidence"]) <= {"<50%", "50-60%", "60-70%", "70%+"}


def test_analyse_produces_a_full_report(backtest_result, synthetic_bars_module):
    report = performance.analyse(backtest_result, synthetic_bars_module, compare=False)
    for key in ("total_return", "cagr", "sharpe", "sortino", "calmar",
                "max_drawdown", "max_drawdown_duration"):
        assert key in report.core
    assert report.beats_all_benchmarks is None, "no benchmarks run means no verdict"


def test_benchmarks_run_and_produce_a_verdict(backtest_result, synthetic_bars_module):
    report = performance.analyse(
        backtest_result, synthetic_bars_module, compare=True, n_random_seeds=5
    )
    assert set(report.benchmarks) >= {"buy_and_hold", "sma_200", "random", "alpha_beta"}
    assert isinstance(report.beats_all_benchmarks, bool)


def test_random_benchmark_is_not_clairvoyant(backtest_result, synthetic_bars_module):
    """The backtester's own sanity check.

    Random allocation should land near the asset's return scaled by average
    exposure. A Sharpe of 2 from random decisions means the engine is broken,
    not that randomness works.
    """
    report = performance.analyse(
        backtest_result, synthetic_bars_module, compare=True, n_random_seeds=10
    )
    assert abs(report.benchmarks["random"]["mean_sharpe"]) < 2.0


def test_export_writes_the_four_csvs(backtest_result, synthetic_bars_module, tmp_path):
    report = performance.analyse(
        backtest_result, synthetic_bars_module, compare=True, n_random_seeds=3
    )
    written = performance.export(report, tmp_path)
    assert set(written) == {"equity_curve", "trade_log", "regime_history", "benchmark_comparison"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0


# -- stress testing ---------------------------------------------------------

def test_seven_crash_scenarios_defined():
    assert len(CRASH_SCENARIOS) == 7
    magnitudes = [s["magnitude"] for s in CRASH_SCENARIOS]
    assert min(magnitudes) == -0.15 and max(magnitudes) == -0.05


def test_crash_injection_lowers_prices_permanently(synthetic_bars_module):
    """A shock that heals by the next bar is not a crash. The level must shift."""
    rng = np.random.default_rng(0)
    shocked = inject_crashes(synthetic_bars_module, -0.10, 10, rng)
    assert shocked["close"].iloc[-1] < synthetic_bars_module["close"].iloc[-1]
    assert len(shocked) == len(synthetic_bars_module)


def test_gap_injection_scales_with_atr(synthetic_bars_module):
    small = inject_gaps(synthetic_bars_module, 2.0, 10, np.random.default_rng(0))
    large = inject_gaps(synthetic_bars_module, 5.0, 10, np.random.default_rng(0))
    assert large["close"].iloc[-1] < small["close"].iloc[-1]


def test_breaker_evaluation_reads_only_pnl():
    """Breakers must fire on realised P&L, never on model state. That is what
    makes them work on the day the model is confidently wrong."""
    index = pd.bdate_range("2024-01-01", periods=100)
    calm = pd.Series(np.linspace(100_000, 110_000, 100), index=index)
    assert evaluate_breakers(calm, {"max_dd_from_peak": 0.10})["halted"] is False

    crashed = calm.copy()
    crashed.iloc[50:] *= 0.80
    assert evaluate_breakers(crashed, {"max_dd_from_peak": 0.10})["halted"] is True


def test_breaker_fires_on_the_configured_threshold():
    index = pd.bdate_range("2024-01-01", periods=60)
    equity = pd.Series(100_000.0, index=index)
    equity.iloc[30:] = 88_000.0        # -12% from peak
    assert evaluate_breakers(equity, {"max_dd_from_peak": 0.10})["peak_halt"] is True
    assert evaluate_breakers(equity, {"max_dd_from_peak": 0.20})["peak_halt"] is False


def test_stress_tester_restores_config_after_misclassification(backtester, synthetic_bars_module):
    """The misclassification test mutates strategy config. If it leaks, every
    later run silently uses shuffled allocations."""
    before = dict(backtester.strategy_config)
    tester = StressTester(backtester, {"max_dd_from_peak": 0.10})
    tester.regime_misclassification(synthetic_bars_module, n_simulations=1, symbol="T")
    assert backtester.strategy_config == before


def test_monte_carlo_summary_reports_worst_not_just_mean(backtester, synthetic_bars_module):
    """You do not get a typical run, you get one run. The worst case is the
    number to size against."""
    tester = StressTester(backtester, {"max_dd_from_peak": 0.10})
    summary = tester.crash_injection(
        synthetic_bars_module, CRASH_SCENARIOS[0], n_simulations=2, symbol="T"
    )
    assert summary.n_simulations > 0
    assert summary.worst_max_drawdown <= summary.mean_max_drawdown
    assert 0.0 <= summary.breaker_fire_rate <= 1.0
    assert 0.0 <= summary.survival_rate <= 1.0


def test_zero_volatility_does_not_explode_the_ratios():
    """Regression: a flat equity curve produced a Sharpe of -1.04e17.

    std() of a constant series is ~1e-20 rather than exactly zero, so a
    `sd > 0` guard passes and the division explodes. Flat equity is normal here
    (system halted, or fully in cash through a turbulent stretch), so this is a
    path that gets hit in ordinary use, and the bogus value looks like a triumph
    rather than an error.
    """
    index = pd.bdate_range("2024-01-01", periods=60)
    flat = pd.Series(100_000.0, index=index)
    assert performance.sharpe_ratio(flat.pct_change()) == 0.0
    assert performance.sortino_ratio(flat.pct_change()) == 0.0
    assert performance.calmar_ratio(flat) == 0.0

    # A curve that rises then goes flat must also stay finite.
    partly_flat = flat.copy()
    partly_flat.iloc[:30] = np.linspace(90_000, 100_000, 30)
    for value in (performance.sharpe_ratio(partly_flat.pct_change()),
                  performance.sortino_ratio(partly_flat.pct_change()),
                  performance.calmar_ratio(partly_flat)):
        assert np.isfinite(value) and abs(value) < 1e6


def test_no_metric_returns_a_non_finite_value(backtest_result, synthetic_bars_module):
    """Sweep the whole report. An inf or nan anywhere gets printed as a number
    and believed."""
    report = performance.analyse(backtest_result, synthetic_bars_module, compare=False)
    for key, value in report.core.items():
        if isinstance(value, float):
            assert np.isfinite(value), f"core[{key}] is {value}"
    for key, value in report.trades.items():
        if isinstance(value, float) and key != "profit_factor":
            assert np.isfinite(value), f"trades[{key}] is {value}"


def test_load_bars_reports_synthetic_honestly(monkeypatch):
    """The flag callers rely on must be true exactly when the data is fake.

    load_bars used to raise NotImplementedError whenever credentials were
    present and silently return a random walk whenever they were not. Nothing
    called it with credentials loaded, so every stored backtest in this project
    was measured on data that never existed, exported without a marking, and
    read back by preflight as a verdict on the strategy.
    """
    from data import market_data

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(market_data, "load_dotenv", lambda *a, **k: None, raising=False)

    bars, synthetic = market_data.load_bars("SPY", allow_synthetic=True)
    assert synthetic is True, "synthetic data must never be reported as real"
    assert not bars.empty


def test_load_bars_refuses_to_invent_data_when_told_not_to(monkeypatch):
    from data import market_data

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(market_data, "load_dotenv", lambda *a, **k: None, raising=False)

    with pytest.raises(RuntimeError, match="No real bars"):
        market_data.load_bars("SPY", allow_synthetic=False)
