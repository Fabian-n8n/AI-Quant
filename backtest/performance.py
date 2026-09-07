"""
Performance metrics, regime attribution, benchmarks and reporting.

Phase 4.

Every number here is **out-of-sample** by construction: the backtester only ever
records OOS bars. That is stated explicitly on every report, because a metric
quoted without its in-sample/out-of-sample label should be assumed wrong.

Two attribution tables carry more weight than the headline numbers:

- **Regime breakdown** answers whether the HMM is earning its keep. If returns
  are indistinguishable across volatility regimes, the regime layer is
  complexity and nothing else, and the correct response is to delete it rather
  than add a sixth state.
- **Confidence buckets** answer whether the confidence score means anything.
  High-confidence bars should outperform low-confidence ones. If they do not,
  `uncertainty_size_mult` is scaling positions on noise.

Trade definition: a "trade" is a rebalance-to-rebalance holding period, not an
entry/exit round trip. Win rate and holding period read against that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
MIN_TRADES_FOR_SIGNIFICANCE = 30

#: Volatility below this counts as zero.
#:
#: `std()` of a constant series is not exactly 0.0 in floating point, it is
#: around 1e-20. Testing `sd > 0` therefore passes and the division explodes: a
#: flat equity curve produced a Sharpe of -1.04e17 before this guard existed.
#: That is not a rounding nuisance, it is a number that looks like a triumph.
#: Flat equity is common and expected here (system halted, or fully in cash
#: through a high-volatility stretch), so this path gets hit in normal use.
ZERO_VOL_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    years = len(equity) / periods_per_year
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.045) -> float:
    """Annualised excess return per unit of volatility.

    Reported alongside total return, never instead of it. This system sits in
    cash during high-volatility regimes, so a strategy 60% invested that matches
    buy-and-hold is genuinely winning while one fully invested that matches it
    has added nothing. Raw return cannot tell those apart.
    """
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - risk_free_rate / TRADING_DAYS
    sd = float(excess.std(ddof=1))
    if not np.isfinite(sd) or sd < ZERO_VOL_TOLERANCE:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.045) -> float:
    """Sharpe penalising downside deviation only.

    Upside volatility is not risk. On a long-only allocation strategy the two
    ratios diverge meaningfully, and the gap is itself informative.
    """
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0]
    dd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    if not np.isfinite(dd) or dd < ZERO_VOL_TOLERANCE:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> tuple[float, int, pd.Timestamp | None]:
    """Return (depth as a negative fraction, duration in bars, trough date).

    Duration is measured peak to recovery, not peak to trough. A 12% drawdown
    that recovers in a fortnight and one that takes three years are different
    experiences, and only the second one makes people abandon a system.
    """
    if equity.empty:
        return 0.0, 0, None
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    trough = drawdown.idxmin()
    depth = float(drawdown.min())

    peak_value = running_max.loc[trough]
    peak_pos = int(np.argmax(equity.index == equity.loc[:trough][equity.loc[:trough] >= peak_value].index[-1])) \
        if (equity.loc[:trough] >= peak_value).any() else 0
    after = equity.loc[trough:]
    recovered = after[after >= peak_value]
    end_pos = equity.index.get_loc(recovered.index[0]) if len(recovered) else len(equity) - 1
    return depth, int(end_pos - peak_pos), trough


def calmar_ratio(equity: pd.Series) -> float:
    """CAGR divided by max drawdown. Return per unit of worst-case pain."""
    depth, _, _ = max_drawdown(equity)
    if depth >= -ZERO_VOL_TOLERANCE:
        return 0.0
    return float(cagr(equity) / abs(depth))


def time_underwater(equity: pd.Series) -> int:
    """Longest run of bars below a prior peak, in bars."""
    if equity.empty:
        return 0
    below = equity < equity.cummax()
    longest = current = 0
    for flag in below:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------

def trade_stats(trades: pd.DataFrame) -> dict[str, Any]:
    """Win rate, profit factor, holding period, consecutive losses.

    Returns `significant: False` under 30 closed trades. Below that a win rate
    is a coin landing heads, and reporting it as a number invites belief it has
    not earned.
    """
    if trades.empty or "pnl" not in trades:
        return {"n_trades": 0, "significant": False}

    closed = trades.dropna(subset=["pnl"])
    if closed.empty:
        return {"n_trades": 0, "significant": False}

    pnl = closed["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit, gross_loss = float(wins.sum()), float(abs(losses.sum()))

    streak = worst_streak = 0
    for value in pnl:
        streak = streak + 1 if value < 0 else 0
        worst_streak = max(worst_streak, streak)

    return {
        "n_trades": len(closed),
        "significant": len(closed) >= MIN_TRADES_FOR_SIGNIFICANCE,
        "win_rate": float(len(wins) / len(closed)),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "avg_pnl": float(pnl.mean()),
        "expectancy": float(pnl.mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        "avg_return_pct": float(closed["return_pct"].mean()) if "return_pct" in closed else 0.0,
        "avg_holding_bars": float(closed["bars_held"].mean()) if "bars_held" in closed else 0.0,
        "max_consecutive_losses": worst_streak,
        "total_slippage": float(closed["slippage_cost"].sum()) if "slippage_cost" in closed else 0.0,
    }


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def regime_breakdown(history: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Per-regime: time in, return contribution, avg trade P&L, win rate, Sharpe.

    The table that tells you whether the regime layer is real. If every row
    looks the same, the HMM is not separating anything the allocation can act on.
    """
    if history.empty or "regime" not in history:
        return pd.DataFrame()

    history = history.copy()
    history["bar_return"] = history["equity"].pct_change().fillna(0.0)
    closed = trades.dropna(subset=["pnl"]) if not trades.empty and "pnl" in trades else pd.DataFrame()

    rows = []
    for regime, group in history.groupby("regime"):
        regime_trades = closed[closed["regime"] == regime] if "regime" in closed else pd.DataFrame()
        pnl = regime_trades["pnl"] if not regime_trades.empty else pd.Series(dtype=float)
        rows.append(
            {
                "regime": regime,
                "pct_time_in": len(group) / len(history),
                "bars": len(group),
                # Sum of bar returns while in this regime: an additive share of
                # the total, not a compounded sub-period return.
                "return_contribution": float(group["bar_return"].sum()),
                "avg_trade_pnl": float(pnl.mean()) if len(pnl) else 0.0,
                "n_trades": len(pnl),
                "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
                "sharpe": sharpe_ratio(group["bar_return"]),
                "avg_allocation": float(group["target_allocation"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_time_in", ascending=False).reset_index(drop=True)


def confidence_buckets(history: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Performance split by the HMM's confidence, in the spec's four buckets.

    The falsification test for the confidence score. If the 70%+ bucket does not
    beat the sub-50% bucket, confidence is noise and every mechanism keyed to it
    (uncertainty mode, min_confidence) is scaling positions on nothing.
    """
    if history.empty or "confidence" not in history:
        return pd.DataFrame()

    history = history.copy()
    history["bar_return"] = history["equity"].pct_change().fillna(0.0)
    edges = [0.0, 0.50, 0.60, 0.70, 1.01]
    labels = ["<50%", "50-60%", "60-70%", "70%+"]
    history["bucket"] = pd.cut(history["confidence"], bins=edges, labels=labels, right=False)

    closed = trades.dropna(subset=["pnl"]) if not trades.empty and "pnl" in trades else pd.DataFrame()
    if not closed.empty and "confidence" in closed:
        closed = closed.copy()
        closed["bucket"] = pd.cut(closed["confidence"], bins=edges, labels=labels, right=False)

    rows = []
    for label in labels:
        group = history[history["bucket"] == label]
        if group.empty:
            continue
        pnl = closed[closed["bucket"] == label]["pnl"] if not closed.empty and "bucket" in closed else pd.Series(dtype=float)
        rows.append(
            {
                "confidence": label,
                "bars": len(group),
                "n_trades": len(pnl),
                "sharpe": sharpe_ratio(group["bar_return"]),
                "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
                "avg_pnl": float(pnl.mean()) if len(pnl) else 0.0,
                "mean_bar_return": float(group["bar_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Worst case
# ---------------------------------------------------------------------------

def worst_case(equity: pd.Series) -> dict[str, Any]:
    """Worst day, week and month, plus the longest stretch underwater."""
    if equity.empty:
        return {}
    daily = equity.pct_change().dropna()
    depth, duration, trough = max_drawdown(equity)
    return {
        "worst_day": float(daily.min()) if len(daily) else 0.0,
        "worst_week": float(equity.resample("W").last().pct_change().min()) if len(equity) > 5 else 0.0,
        "worst_month": float(equity.resample("ME").last().pct_change().min()) if len(equity) > 21 else 0.0,
        "max_drawdown": depth,
        "max_drawdown_duration_bars": duration,
        "max_drawdown_date": trough,
        "longest_underwater_bars": time_underwater(equity),
    }


def alpha_beta(strategy_equity: pd.Series, benchmark_equity: pd.Series) -> dict[str, float]:
    """Alpha and beta against buy-and-hold by OLS on daily returns.

    Beta is the point. This strategy deliberately sits in cash during turbulent
    regimes, so a beta well below 1 with alpha near zero is the expected shape:
    it is taking less market risk, not finding independent return. Reading only
    total return would miss that entirely.
    """
    joined = pd.concat(
        [strategy_equity.pct_change(), benchmark_equity.pct_change()], axis=1
    ).dropna()
    if len(joined) < 3:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}

    y = joined.iloc[:, 0].to_numpy()
    x = joined.iloc[:, 1].to_numpy()
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept, beta = float(coefficients[0]), float(coefficients[1])

    predicted = design @ coefficients
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())

    return {
        "alpha": intercept * TRADING_DAYS,   # annualised
        "beta": beta,
        "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def benchmark_buy_and_hold(bars: pd.DataFrame, index: pd.Index, capital: float) -> pd.Series:
    """The opportunity cost. If the system cannot beat doing nothing, doing
    nothing is better, and free."""
    prices = bars.loc[index, "close"]
    return capital * prices / prices.iloc[0]


def benchmark_sma_trend(
    bars: pd.DataFrame, index: pd.Index, capital: float, window: int = 200
) -> pd.Series:
    """Long above the 200 SMA, cash below.

    The "is the HMM's complexity earning its keep" test: trend following in one
    line of logic against a fitted multi-state model. The SMA is computed on the
    full series and then sliced, which is causal because a trailing rolling mean
    at t depends only on bars up to t.
    """
    sma = bars["close"].rolling(window, min_periods=window).mean()
    invested = (bars["close"] > sma).reindex(index).fillna(False)
    returns = bars["close"].pct_change().reindex(index).fillna(0.0)
    # shift(1): act on yesterday's signal, so the decision cannot use today's
    # close that it is being paid on.
    strategy_returns = returns * invested.shift(1).fillna(False).astype(float)
    return capital * (1 + strategy_returns).cumprod()


def benchmark_random_allocation(
    bars: pd.DataFrame,
    index: pd.Index,
    capital: float,
    n_rebalances: int,
    allocation_choices: list[float],
    n_seeds: int = 100,
    slippage_pct: float = 0.0005,
) -> dict[str, Any]:
    """Random allocation changes at the same frequency, same sizing rules.

    Isolates the regime signal. If shuffling *when* and *what* to allocate does
    as well as the HMM's decisions, then the edge is in the position sizing
    envelope, not in regime detection, and the HMM should be deleted.

    Also the backtester's own sanity check: this has a known expected shape,
    roughly the asset's return scaled by average exposure, minus costs. A random
    benchmark with a Sharpe of 2 means the engine is broken, not clairvoyant.
    """
    from backtest.backtester import _PortfolioState

    prices = bars.loc[index, ["open", "close"]]
    finals, sharpes, drawdowns = [], [], []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        state = _PortfolioState(cash=capital)
        rebalance_bars = set(
            rng.choice(len(index), size=min(n_rebalances, len(index)), replace=False)
        )
        curve = []
        for i, timestamp in enumerate(index):
            if i in rebalance_bars:
                state.rebalance(
                    float(rng.choice(allocation_choices)),
                    float(prices.loc[timestamp, "open"]),
                    slippage_pct,
                    0.0,
                )
            curve.append(state.equity_at(float(prices.loc[timestamp, "close"])))
        equity = pd.Series(curve, index=index)
        finals.append(float(equity.iloc[-1]))
        sharpes.append(sharpe_ratio(equity.pct_change()))
        drawdowns.append(max_drawdown(equity)[0])

    return {
        "mean_final_equity": float(np.mean(finals)),
        "std_final_equity": float(np.std(finals)),
        "mean_total_return": float(np.mean(finals) / capital - 1),
        "std_total_return": float(np.std(finals) / capital),
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "mean_max_drawdown": float(np.mean(drawdowns)),
        "n_seeds": n_seeds,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class PerformanceReport:
    """Everything computed for one backtest. Out-of-sample throughout."""
    symbol: str
    core: dict[str, Any] = field(default_factory=dict)
    trades: dict[str, Any] = field(default_factory=dict)
    regimes: pd.DataFrame = field(default_factory=pd.DataFrame)
    confidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    worst: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def beats_all_benchmarks(self) -> bool | None:
        """True only if the strategy beats all three on total return.

        None when the benchmarks were not run, which is distinct from False and
        must not be reported as a pass.
        """
        if not self.benchmarks:
            return None
        own = self.core.get("total_return", 0.0)
        beaten = [
            own > self.benchmarks.get("buy_and_hold", {}).get("total_return", np.inf),
            own > self.benchmarks.get("sma_200", {}).get("total_return", np.inf),
            own > self.benchmarks.get("random", {}).get("mean_total_return", np.inf),
        ]
        return all(beaten)


def analyse(
    result,
    bars: pd.DataFrame,
    risk_free_rate: float = 0.045,
    compare: bool = False,
    n_random_seeds: int = 100,
) -> PerformanceReport:
    """Compute the full metric set for a BacktestResult."""
    equity = result.equity_curve
    if equity.empty:
        return PerformanceReport(symbol=result.symbol)

    returns = equity.pct_change()
    depth, duration, trough = max_drawdown(equity)

    core = {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "sharpe": sharpe_ratio(returns, risk_free_rate),
        "sortino": sortino_ratio(returns, risk_free_rate),
        "calmar": calmar_ratio(equity),
        "max_drawdown": depth,
        "max_drawdown_duration": duration,
        "max_drawdown_date": trough,
        "final_equity": float(equity.iloc[-1]),
        "initial_capital": result.initial_capital,
        "n_bars": len(equity),
        "n_folds": result.n_folds,
        "time_in_market": float((result.regime_history["allocation"].abs() > 0.01).mean())
        if "allocation" in result.regime_history else 0.0,
        "avg_allocation": float(result.regime_history["target_allocation"].mean())
        if "target_allocation" in result.regime_history else 0.0,
    }

    report = PerformanceReport(
        symbol=result.symbol,
        core=core,
        trades=trade_stats(result.trade_log),
        regimes=regime_breakdown(result.regime_history, result.trade_log),
        confidence=confidence_buckets(result.regime_history, result.trade_log),
        worst=worst_case(equity),
        equity_curve=equity,
        trade_log=result.trade_log,
        regime_history=result.regime_history,
    )

    if compare:
        report.benchmarks = _run_benchmarks(result, bars, equity, risk_free_rate, n_random_seeds)
    return report


def _run_benchmarks(result, bars, equity, risk_free_rate, n_seeds) -> dict[str, Any]:
    index = equity.index
    capital = result.initial_capital

    hold = benchmark_buy_and_hold(bars, index, capital)
    sma = benchmark_sma_trend(bars, index, capital)

    allocations = sorted(result.regime_history["target_allocation"].unique().tolist()) \
        if "target_allocation" in result.regime_history else [0.0, 0.6, 0.95]

    random_result = benchmark_random_allocation(
        bars, index, capital,
        n_rebalances=max(result.n_trades, 1),
        allocation_choices=allocations,
        n_seeds=n_seeds,
        slippage_pct=result.config.get("slippage_pct", 0.0005),
    )

    return {
        "buy_and_hold": {
            "total_return": total_return(hold),
            "cagr": cagr(hold),
            "sharpe": sharpe_ratio(hold.pct_change(), risk_free_rate),
            "max_drawdown": max_drawdown(hold)[0],
            "final_equity": float(hold.iloc[-1]),
            "equity": hold,
        },
        "sma_200": {
            "total_return": total_return(sma),
            "cagr": cagr(sma),
            "sharpe": sharpe_ratio(sma.pct_change(), risk_free_rate),
            "max_drawdown": max_drawdown(sma)[0],
            "final_equity": float(sma.iloc[-1]),
            "equity": sma,
        },
        "random": random_result,
        "alpha_beta": alpha_beta(equity, hold),
    }


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def render(report: PerformanceReport, console=None) -> None:
    """Rich tables to the terminal.

    Every panel is labelled out-of-sample. The trade count sits next to the win
    rate rather than below it, so a 70% win rate over 8 trades cannot be read
    without its context.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = console or Console()
    core, trades = report.core, report.trades

    console.print()
    console.print(Panel(
        f"[bold]{report.symbol}[/bold]  walk-forward, "
        f"{core.get('n_folds', 0)} folds, {core.get('n_bars', 0)} bars  "
        f"[dim]ALL FIGURES OUT-OF-SAMPLE[/dim]",
        style="cyan",
    ))

    table = Table(title="Core metrics", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for label, value, fmt in [
        ("Total return", core.get("total_return", 0), "pct"),
        ("CAGR", core.get("cagr", 0), "pct"),
        ("Sharpe", core.get("sharpe", 0), "num"),
        ("Sortino", core.get("sortino", 0), "num"),
        ("Calmar", core.get("calmar", 0), "num"),
        ("Max drawdown", core.get("max_drawdown", 0), "pct"),
        ("Max DD duration (bars)", core.get("max_drawdown_duration", 0), "int"),
        ("Avg allocation", core.get("avg_allocation", 0), "pct"),
        ("Final equity", core.get("final_equity", 0), "money"),
    ]:
        table.add_row(label, _fmt(value, fmt))
    console.print(table)

    table = Table(title="Trade statistics", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    n = trades.get("n_trades", 0)
    table.add_row("Total trades (rebalances)", str(n))
    if not trades.get("significant", False):
        table.add_row(
            "[yellow]Statistical significance[/yellow]",
            f"[yellow]NOT ENOUGH DATA ({n} < {MIN_TRADES_FOR_SIGNIFICANCE})[/yellow]",
        )
    for label, key, fmt in [
        ("Win rate", "win_rate", "pct"), ("Avg win", "avg_win", "money"),
        ("Avg loss", "avg_loss", "money"), ("Expectancy per trade", "expectancy", "money"),
        ("Profit factor", "profit_factor", "num"),
        ("Avg holding (bars)", "avg_holding_bars", "num"),
        ("Max consecutive losses", "max_consecutive_losses", "int"),
        ("Total slippage paid", "total_slippage", "money"),
    ]:
        if key in trades:
            table.add_row(label, _fmt(trades[key], fmt))
    console.print(table)

    if not report.regimes.empty:
        table = Table(
            title="Regime breakdown  [dim](flat rows mean the HMM adds nothing)[/dim]",
            header_style="bold",
        )
        for col in ("Regime", "% Time", "Contribution", "Avg Trade P&L", "Trades", "Win Rate", "Sharpe", "Avg Alloc"):
            table.add_column(col, justify="right" if col != "Regime" else "left")
        for _, row in report.regimes.iterrows():
            table.add_row(
                row["regime"], _fmt(row["pct_time_in"], "pct"),
                _fmt(row["return_contribution"], "pct"), _fmt(row["avg_trade_pnl"], "money"),
                str(int(row["n_trades"])), _fmt(row["win_rate"], "pct"),
                _fmt(row["sharpe"], "num"), _fmt(row["avg_allocation"], "pct"),
            )
        console.print(table)

    if not report.confidence.empty:
        table = Table(
            title="Confidence buckets  [dim](70%+ should beat <50%, or confidence is noise)[/dim]",
            header_style="bold",
        )
        for col in ("Confidence", "Bars", "Trades", "Sharpe", "Win Rate", "Avg P&L"):
            table.add_column(col, justify="right" if col != "Confidence" else "left")
        for _, row in report.confidence.iterrows():
            table.add_row(
                row["confidence"], str(int(row["bars"])), str(int(row["n_trades"])),
                _fmt(row["sharpe"], "num"), _fmt(row["win_rate"], "pct"),
                _fmt(row["avg_pnl"], "money"),
            )
        console.print(table)

    if report.worst:
        table = Table(title="Worst case", header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for label, key, fmt in [
            ("Worst day", "worst_day", "pct"), ("Worst week", "worst_week", "pct"),
            ("Worst month", "worst_month", "pct"),
            ("Longest underwater (bars)", "longest_underwater_bars", "int"),
        ]:
            table.add_row(label, _fmt(report.worst.get(key, 0), fmt))
        console.print(table)

    if report.benchmarks:
        _render_benchmarks(report, console)


def _render_benchmarks(report: PerformanceReport, console) -> None:
    from rich.table import Table

    b = report.benchmarks
    table = Table(
        title="Benchmarks  [dim](must beat ALL THREE out-of-sample)[/dim]",
        header_style="bold",
    )
    for col in ("Strategy", "Total Return", "CAGR", "Sharpe", "Max DD"):
        table.add_column(col, justify="right" if col != "Strategy" else "left")

    table.add_row(
        "[bold]regime-trader[/bold]", _fmt(report.core["total_return"], "pct"),
        _fmt(report.core["cagr"], "pct"), _fmt(report.core["sharpe"], "num"),
        _fmt(report.core["max_drawdown"], "pct"), style="cyan",
    )
    for key, label in [("buy_and_hold", "Buy and hold"), ("sma_200", "200 SMA trend")]:
        row = b[key]
        table.add_row(label, _fmt(row["total_return"], "pct"), _fmt(row["cagr"], "pct"),
                      _fmt(row["sharpe"], "num"), _fmt(row["max_drawdown"], "pct"))
    r = b["random"]
    table.add_row(
        f"Random alloc ({r['n_seeds']} seeds)",
        f"{r['mean_total_return']:+.2%} ± {r['std_total_return']:.2%}",
        "-", f"{r['mean_sharpe']:.2f} ± {r['std_sharpe']:.2f}",
        _fmt(r["mean_max_drawdown"], "pct"),
    )
    console.print(table)

    ab = b["alpha_beta"]
    console.print(
        f"  Alpha (annualised): [bold]{ab['alpha']:+.2%}[/bold]   "
        f"Beta: [bold]{ab['beta']:.2f}[/bold]   R²: {ab['r_squared']:.2f}"
    )
    console.print(
        "  [dim]Beta well below 1 with alpha near zero means less market risk "
        "taken, not independent return found.[/dim]"
    )

    verdict = report.beats_all_benchmarks
    if verdict:
        console.print("\n  [bold green]Beats all three benchmarks on total return.[/bold green]")
    else:
        console.print("\n  [bold red]Does NOT beat all three benchmarks. No demonstrated edge.[/bold red]")
    if not report.trades.get("significant", False):
        console.print(
            f"  [yellow]Only {report.trades.get('n_trades', 0)} trades: below the "
            f"{MIN_TRADES_FOR_SIGNIFICANCE}-trade floor, so treat the verdict as noise.[/yellow]"
        )


def _fmt(value: Any, kind: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if kind == "pct":
        return f"{value:+.2%}"
    if kind == "money":
        return f"${value:,.0f}"
    if kind == "int":
        return f"{int(value):,}"
    return f"{value:.2f}"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(report: PerformanceReport, output_dir: Path) -> dict[str, Path]:
    """Write equity_curve, trade_log, regime_history and benchmark_comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if not report.equity_curve.empty:
        path = output_dir / "equity_curve.csv"
        report.equity_curve.rename("equity").to_frame().to_csv(path)
        written["equity_curve"] = path

    if not report.trade_log.empty:
        path = output_dir / "trade_log.csv"
        report.trade_log.to_csv(path, index=False)
        written["trade_log"] = path

    if not report.regime_history.empty:
        path = output_dir / "regime_history.csv"
        report.regime_history.to_csv(path)
        written["regime_history"] = path

    if report.benchmarks:
        rows = [{
            "strategy": "regime-trader",
            "total_return": report.core["total_return"],
            "cagr": report.core["cagr"],
            "sharpe": report.core["sharpe"],
            "max_drawdown": report.core["max_drawdown"],
        }]
        for key, label in [("buy_and_hold", "buy_and_hold"), ("sma_200", "sma_200_trend")]:
            row = report.benchmarks[key]
            rows.append({
                "strategy": label, "total_return": row["total_return"], "cagr": row["cagr"],
                "sharpe": row["sharpe"], "max_drawdown": row["max_drawdown"],
            })
        r = report.benchmarks["random"]
        rows.append({
            "strategy": f"random_allocation_{r['n_seeds']}_seeds",
            "total_return": r["mean_total_return"], "cagr": np.nan,
            "sharpe": r["mean_sharpe"], "max_drawdown": r["mean_max_drawdown"],
        })
        path = output_dir / "benchmark_comparison.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        written["benchmark_comparison"] = path

    logger.info("Exported %d files to %s", len(written), output_dir)
    return written
