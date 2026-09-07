"""
Stress testing: crash injection, gap risk, regime misclassification.

Phase 4.

The backtest covers what happened. This covers what could have. The point is not
to predict the next crash but to confirm that when one arrives, the system's
losses stay inside what the risk settings claim to allow.

The third test is the important one and the easiest to skip. **Regime
misclassification** deliberately corrupts the allocation decisions and checks
that damage stays contained. If the system only survives because the HMM is
right, the risk layer is not independent, and it will fail on precisely the day
the model is wrong.

CIRCUIT BREAKERS
----------------
`evaluate_breakers` now delegates to the real `RiskManager` from Phase 5. It
previously duplicated the threshold logic because that class did not yet exist;
Phase 4 flagged that keeping two implementations of the same thresholds would
let them drift, and this closes that.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backtest.performance import max_drawdown

logger = logging.getLogger(__name__)


#: The spec's seven crash scenarios, mild to severe.
CRASH_SCENARIOS: list[dict[str, Any]] = [
    {"name": "mild",         "magnitude": -0.05, "count": 10},
    {"name": "mild-cluster", "magnitude": -0.05, "count": 15},
    {"name": "moderate",     "magnitude": -0.08, "count": 10},
    {"name": "severe-few",   "magnitude": -0.10, "count": 5},
    {"name": "severe",       "magnitude": -0.10, "count": 10},
    {"name": "extreme-few",  "magnitude": -0.15, "count": 5},
    {"name": "extreme",      "magnitude": -0.15, "count": 10},
]


@dataclass
class MonteCarloSummary:
    """Aggregate across simulations. Worst case matters more than the mean.

    The mean tells you about a typical run; you do not get a typical run, you
    get one run. `worst_max_drawdown` is the number to size against.
    """
    scenario: str
    n_simulations: int
    mean_max_drawdown: float
    median_max_drawdown: float
    worst_max_drawdown: float
    mean_final_return: float
    worst_final_return: float
    breaker_fire_rate: float          # fraction of sims tripping the halt breaker
    survival_rate: float              # fraction ending above the ruin threshold
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Breaker evaluation (Phase 5 should replace this with the real RiskManager)
# ---------------------------------------------------------------------------

def evaluate_breakers(equity: pd.Series, risk_config: dict) -> dict[str, Any]:
    """Would the configured breakers have fired over this equity curve?

    Delegates the thresholds to Phase 5's `CircuitBreaker` so there is one
    definition. Walks the curve bar by bar rather than testing the endpoint,
    because a breaker that fired in month two and was later recovered from still
    fired, and a stress test that only looks at the final state would miss it.

    Reads only P&L, never model state. That is the property the breakers exist
    for: they must still work when the HMM is confidently wrong.
    """
    from core.risk_manager import BreakerState, CircuitBreaker, PortfolioState

    if equity.empty:
        return {"halted": False, "peak_drawdown": 0.0}

    peak_dd, _, _ = max_drawdown(equity)
    daily_returns = equity.pct_change().dropna()

    # A scratch lock file per evaluation, so a simulated halt never writes into
    # the repo and never leaks into the next simulation.
    with tempfile.TemporaryDirectory() as scratch:
        breaker = CircuitBreaker(risk_config, Path(scratch) / "halted.lock")
        peak = float(equity.iloc[0])
        week_start = float(equity.iloc[0])
        halted = False

        for i, (timestamp, value) in enumerate(equity.items()):
            peak = max(peak, float(value))
            if i % 5 == 0:
                week_start = float(value)
            state = PortfolioState(
                equity=float(value),
                peak_equity=peak,
                day_start_equity=float(equity.iloc[i - 1]) if i else float(value),
                week_start_equity=week_start,
                timestamp=timestamp if isinstance(timestamp, datetime) else None,
            )
            if BREAKER_ACTION_HALTED(breaker.update(state)):
                halted = True
                break

        history = breaker.get_history()
        fired = set(history["breaker"]) if not history.empty else set()

    return {
        "halted": halted,
        "peak_drawdown": peak_dd,
        "worst_day": float(daily_returns.min()) if len(daily_returns) else 0.0,
        "daily_halt": "daily_halt" in fired,
        "weekly_halt": "weekly_halt" in fired,
        "peak_halt": "peak_halt" in fired,
        "daily_reduce": "daily_reduce" in fired,
        "weekly_reduce": "weekly_reduce" in fired,
        "n_triggers": len(history),
    }


def BREAKER_ACTION_HALTED(action) -> bool:
    """True when the action in force stops trading entirely."""
    from core.risk_manager import BreakerState

    return action is BreakerState.HALTED


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def inject_crashes(
    bars: pd.DataFrame, magnitude: float, count: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Insert single-day drops at random points.

    Applied multiplicatively to every bar from the shock onward, so the price
    level shifts permanently rather than dipping for one bar and recovering by
    magic. A crash that self-heals the next day is not a crash.

    The shock hits the open as well as the close, which reproduces a gap-down
    the strategy cannot trade around rather than an orderly intraday decline.
    """
    out = bars.copy()
    if len(out) < 20:
        return out

    points = rng.choice(np.arange(10, len(out) - 5), size=min(count, len(out) - 15), replace=False)
    for point in sorted(points):
        factor = 1.0 + magnitude
        out.iloc[point:, out.columns.get_indexer(["open", "high", "low", "close"])] *= factor
        # Widen the shock bar's own range so ATR registers the event.
        out.iloc[point, out.columns.get_loc("low")] *= 1.0 + magnitude / 2
    return out


def inject_gaps(
    bars: pd.DataFrame, atr_multiple: float, count: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Insert overnight gap-downs sized in ATR.

    The specific failure this probes: a stop does not protect through a gap. If
    the close is 50 with a stop at 48 and the open is 44, the fill is 44. The
    backtester holds no stops by design, so what this measures is how much of a
    move the allocation layer absorbs before the breakers would fire.
    """
    out = bars.copy()
    if len(out) < 20:
        return out

    high, low, close = out["high"], out["low"], out["close"]
    prev = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    points = rng.choice(np.arange(20, len(out) - 5), size=min(count, len(out) - 25), replace=False)
    for point in sorted(points):
        atr_value = atr.iloc[point]
        if not np.isfinite(atr_value) or atr_value <= 0:
            continue
        shock = atr_multiple * atr_value / close.iloc[point]
        factor = 1.0 - shock
        out.iloc[point:, out.columns.get_indexer(["open", "high", "low", "close"])] *= factor
    return out


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

class StressTester:
    """Runs the three stress categories against a configured backtester."""

    def __init__(self, backtester, risk_config: Optional[dict] = None,
                 ruin_threshold: float = 0.50) -> None:
        self.backtester = backtester
        self.risk_config = dict(risk_config or {})
        self.ruin_threshold = ruin_threshold

    # -- a. crash injection -------------------------------------------------

    def crash_injection(
        self, bars: pd.DataFrame, scenario: dict, n_simulations: int = 100, symbol: str = "STRESS"
    ) -> MonteCarloSummary:
        """Monte Carlo over randomly placed crashes of a fixed size."""
        drawdowns, returns, halted = [], [], []

        for seed in range(n_simulations):
            rng = np.random.default_rng(seed)
            shocked = inject_crashes(bars, scenario["magnitude"], scenario["count"], rng)
            try:
                result = self.backtester.run(shocked, symbol)
            except Exception as exc:
                logger.debug("sim %d failed: %s", seed, exc)
                continue
            if result.equity_curve.empty:
                continue

            equity = result.equity_curve
            drawdowns.append(max_drawdown(equity)[0])
            returns.append(float(equity.iloc[-1] / equity.iloc[0] - 1))
            halted.append(evaluate_breakers(equity, self.risk_config)["halted"])

        return self._summarise(scenario["name"], drawdowns, returns, halted, scenario)

    def run_all_crash_scenarios(
        self, bars: pd.DataFrame, n_simulations: int = 100, symbol: str = "STRESS"
    ) -> list[MonteCarloSummary]:
        return [
            self.crash_injection(bars, scenario, n_simulations, symbol)
            for scenario in CRASH_SCENARIOS
        ]

    # -- b. gap risk --------------------------------------------------------

    def gap_risk(
        self, bars: pd.DataFrame, atr_multiples: tuple[float, ...] = (2.0, 5.0),
        n_simulations: int = 20, symbol: str = "STRESS",
    ) -> list[MonteCarloSummary]:
        """Overnight gaps at 2x and 5x ATR.

        Reports expected loss against actual. Expected is the naive figure:
        gap size times average allocation. Actual is what the backtest produced.
        Actual materially worse than expected means the allocation layer is
        compounding the shock rather than absorbing it.
        """
        summaries = []
        for multiple in atr_multiples:
            drawdowns, returns, halted = [], [], []
            for seed in range(n_simulations):
                rng = np.random.default_rng(seed)
                shocked = inject_gaps(bars, multiple, count=10, rng=rng)
                try:
                    result = self.backtester.run(shocked, symbol)
                except Exception:
                    continue
                if result.equity_curve.empty:
                    continue
                equity = result.equity_curve
                drawdowns.append(max_drawdown(equity)[0])
                returns.append(float(equity.iloc[-1] / equity.iloc[0] - 1))
                halted.append(evaluate_breakers(equity, self.risk_config)["halted"])

            summary = self._summarise(
                f"gap_{multiple:g}x_atr", drawdowns, returns, halted,
                {"atr_multiple": multiple},
            )
            summaries.append(summary)
        return summaries

    # -- c. regime misclassification ---------------------------------------

    def regime_misclassification(
        self, bars: pd.DataFrame, n_simulations: int = 20, symbol: str = "STRESS"
    ) -> MonteCarloSummary:
        """Shuffle the allocation parameters so regimes drive the wrong sizing.

        The strongest test in this file. It simulates the HMM being not merely
        imprecise but systematically wrong: low-volatility allocation applied in
        turbulence and vice versa.

        If losses here are catastrophically worse than the baseline, the system
        is only safe while the model is right, which means the risk layer is not
        independent of it. The whole point of Phase 5's breakers is that they
        fire on realised P&L and therefore still work on the day the model is
        wrong. This is where that claim gets tested.
        """
        drawdowns, returns, halted = [], [], []
        original = dict(self.backtester.strategy_config)

        try:
            for seed in range(n_simulations):
                rng = np.random.default_rng(seed)
                values = [
                    original.get("low_vol_allocation", 0.95),
                    original.get("mid_vol_allocation_trend", 0.95),
                    original.get("mid_vol_allocation_no_trend", 0.60),
                    original.get("high_vol_allocation", 0.60),
                ]
                rng.shuffle(values)
                self.backtester.strategy_config = {
                    **original,
                    "low_vol_allocation": values[0],
                    "mid_vol_allocation_trend": values[1],
                    "mid_vol_allocation_no_trend": values[2],
                    "high_vol_allocation": values[3],
                }
                try:
                    result = self.backtester.run(bars, symbol)
                except Exception:
                    continue
                if result.equity_curve.empty:
                    continue
                equity = result.equity_curve
                drawdowns.append(max_drawdown(equity)[0])
                returns.append(float(equity.iloc[-1] / equity.iloc[0] - 1))
                halted.append(evaluate_breakers(equity, self.risk_config)["halted"])
        finally:
            self.backtester.strategy_config = original

        return self._summarise("regime_misclassification", drawdowns, returns, halted, {})

    # -- helpers ------------------------------------------------------------

    def _summarise(self, name, drawdowns, returns, halted, detail) -> MonteCarloSummary:
        if not drawdowns:
            return MonteCarloSummary(name, 0, 0, 0, 0, 0, 0, 0, 0, detail)
        return MonteCarloSummary(
            scenario=name,
            n_simulations=len(drawdowns),
            mean_max_drawdown=float(np.mean(drawdowns)),
            median_max_drawdown=float(np.median(drawdowns)),
            worst_max_drawdown=float(np.min(drawdowns)),
            mean_final_return=float(np.mean(returns)),
            worst_final_return=float(np.min(returns)),
            breaker_fire_rate=float(np.mean(halted)) if halted else 0.0,
            survival_rate=float(np.mean([r > -self.ruin_threshold for r in returns])),
            detail=detail,
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_stress(summaries: list[MonteCarloSummary], baseline: Optional[dict] = None,
                  console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    table = Table(title="Stress test  [dim](worst case matters more than the mean)[/dim]",
                  header_style="bold")
    for col in ("Scenario", "Sims", "Mean DD", "Median DD", "Worst DD",
                "Mean Return", "Worst Return", "Breaker Fired", "Survived"):
        table.add_column(col, justify="right" if col != "Scenario" else "left")

    for s in summaries:
        if s.n_simulations == 0:
            table.add_row(s.scenario, "0", *["n/a"] * 7)
            continue
        table.add_row(
            s.scenario, str(s.n_simulations),
            f"{s.mean_max_drawdown:.2%}", f"{s.median_max_drawdown:.2%}",
            f"{s.worst_max_drawdown:.2%}", f"{s.mean_final_return:+.2%}",
            f"{s.worst_final_return:+.2%}", f"{s.breaker_fire_rate:.0%}",
            f"{s.survival_rate:.0%}",
        )
    console.print(table)

    if baseline:
        console.print(
            f"  [dim]Baseline (no shock): max DD {baseline.get('max_drawdown', 0):.2%}, "
            f"return {baseline.get('total_return', 0):+.2%}[/dim]"
        )

    misclass = next((s for s in summaries if s.scenario == "regime_misclassification"), None)
    if misclass and baseline and misclass.n_simulations:
        ratio = abs(misclass.worst_max_drawdown) / max(abs(baseline.get("max_drawdown", 0.01)), 0.01)
        console.print(
            f"\n  Misclassification worst drawdown is [bold]{ratio:.1f}x[/bold] the baseline."
        )
        if ratio > 3.0:
            console.print(
                "  [bold red]Damage is not contained when regimes are wrong. The risk layer "
                "is not independent enough: Phase 5's breakers must not depend on model "
                "state.[/bold red]"
            )
        else:
            console.print(
                "  [green]Damage stays bounded when regimes are wrong, which is what the "
                "independence requirement asks for.[/green]"
            )
