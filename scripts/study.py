#!/usr/bin/env python
"""Run a configuration study with the multiple-testing correction attached.

Sweeping settings and reporting the winner is how a backtest manufactures an
edge that does not exist. This runs the sweep, records every arm to
`data/trials.db` including the failures, and scores the best one against the
Sharpe that the best of N skill-free arms would have reached anyway.

    python scripts/study.py ablation     does the HMM earn its complexity
    python scripts/study.py risk         position cap and circuit-breaker levels
    python scripts/study.py report       what the registry already knows

Nothing here changes a setting. It produces evidence; deciding is separate and
belongs to a person.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.portfolio_backtester import PortfolioBacktester  # noqa: E402
from backtest.trials import Trial, TrialRegistry, deflated_sharpe_ratio  # noqa: E402
from config import load_settings, strategy_config  # noqa: E402
from data.market_data import load_bars  # noqa: E402

# Ten years rather than the five the default range returns. More folds is the
# cheapest available defence against fitting one market.
HISTORY_START = "2015-01-01"


def quiet() -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)


def load_universe(settings) -> tuple[dict[str, pd.DataFrame], bool]:
    bars, synthetic = {}, False
    for symbol in settings["broker"]["symbols"]:
        frame, is_synthetic = load_bars(symbol, HISTORY_START, None)
        synthetic |= is_synthetic
        bars[symbol] = frame
    return bars, synthetic


def score(result, initial_capital: float) -> dict:
    equity = result.equity_curve
    if equity.empty:
        return {}
    daily = equity.pct_change().dropna()
    return {
        "total_return": float(equity.iloc[-1] / initial_capital - 1),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "avg_exposure": result.avg_exposure,
        "n_trades": result.n_trades,
        "n_bars": len(equity),
        "returns": daily,
    }


def build(settings, bars, risk_overrides=None, regime_mode="hmm", regime_seed=0):
    risk = dict(settings["risk"])
    risk.update(risk_overrides or {})
    return PortfolioBacktester(
        symbols=list(bars), primary=settings["broker"]["symbols"][0],
        train_window=settings["backtest"]["train_window"],
        test_window=settings["backtest"]["test_window"],
        step_size=settings["backtest"]["step_size"],
        initial_capital=settings["backtest"]["initial_capital"],
        slippage_pct=settings["backtest"]["slippage_pct"],
        hmm_config=dict(settings["hmm"]),
        strategy_config=strategy_config(settings),
        risk_config=risk,
        reward_risk_ratio=settings["risk"].get("reward_risk_ratio", 2.0),
        regime_mode=regime_mode, regime_seed=regime_seed,
    )


def run_arm(registry, study, label, settings, bars, risk_overrides=None,
            regime_mode="hmm", regime_seed=0):
    backtester = build(settings, bars, risk_overrides, regime_mode, regime_seed)
    result = backtester.run(bars)
    metrics = score(result, settings["backtest"]["initial_capital"])
    if not metrics:
        return None
    registry.record(Trial(
        study=study, label=label,
        config={"risk": risk_overrides or {}, "regime_mode": regime_mode,
                "regime_seed": regime_seed},
        total_return=metrics["total_return"], sharpe=metrics["sharpe"],
        max_drawdown=metrics["max_drawdown"], avg_exposure=metrics["avg_exposure"],
        n_trades=metrics["n_trades"], n_bars=metrics["n_bars"],
    ))
    return metrics


def header(columns: tuple[str, ...], widths: tuple[int, ...]) -> None:
    print("  " + "".join(c.rjust(w) for c, w in zip(columns, widths, strict=True)))
    print("  " + "-" * sum(widths))


def row(label, m, width=26):
    print(f"  {label:<{width}}{m['total_return']:>9.2%}{m['max_drawdown']:>9.2%}"
          f"{m['sharpe']:>8.2f}{m['avg_exposure']:>8.1%}{m['n_trades']:>8}")


# --------------------------------------------------------------------------


def study_ablation(registry, settings, bars) -> None:
    """Does the HMM regime layer earn its complexity?

    Everything else is held constant. If `hmm` cannot beat `shuffled`, the
    classifier is contributing a distribution of labels and no timing, and the
    honest response is to delete several hundred lines of it.
    """
    print("\nABLATION: is the regime classifier doing anything?\n")
    header(("", "return", "maxDD", "Sharpe", "expo", "trades"), (26, 9, 9, 8, 8, 8))

    results = {}
    results["hmm"] = run_arm(registry, "ablation", "hmm", settings, bars)
    if results["hmm"]:
        row("HMM regime (shipped)", results["hmm"])

    results["fixed"] = run_arm(registry, "ablation", "fixed", settings, bars,
                               regime_mode="fixed")
    if results["fixed"]:
        row("fixed regime (no HMM)", results["fixed"])

    shuffled = []
    for seed in range(5):
        m = run_arm(registry, "ablation", f"shuffled-{seed}", settings, bars,
                    regime_mode="shuffled", regime_seed=seed)
        if m:
            shuffled.append(m)
    if shuffled:
        mean = {k: float(np.mean([s[k] for s in shuffled]))
                for k in ("total_return", "max_drawdown", "sharpe", "avg_exposure")}
        mean["n_trades"] = int(np.mean([s["n_trades"] for s in shuffled]))
        row("shuffled regime (5 seeds)", mean)

    print()
    if results["hmm"] and shuffled:
        edge = results["hmm"]["sharpe"] - float(np.mean([s["sharpe"] for s in shuffled]))
        spread = float(np.std([s["sharpe"] for s in shuffled], ddof=1)) if len(shuffled) > 1 else 0.0
        print(f"  HMM Sharpe minus shuffled mean: {edge:+.2f} "
              f"(shuffled spread {spread:.2f})")
        if spread > 0 and abs(edge) < spread:
            print("  Inside the noise of shuffling its own labels. On this evidence "
                  "the\n  classifier is not timing anything.")
        elif edge > 0:
            print("  Outside the shuffle noise, so the timing is contributing "
                  "something.")
        else:
            print("  Worse than its own shuffled labels. The timing is actively "
                  "costing.")


def study_risk(registry, settings, bars) -> None:
    """Position cap against circuit-breaker level, corrected for the search."""
    print("\nRISK SETTINGS: position cap x circuit breaker\n")
    header(("", "return", "maxDD", "Sharpe", "expo", "trades"), (26, 9, 9, 8, 8, 8))

    best = None
    for peak_dd, breaker_label in ((0.10, "shipped"), (0.25, "swing")):
        for cap in (0.03, 0.05, 0.08, 0.12):
            overrides = {
                "max_single_position": cap, "max_dd_from_peak": peak_dd,
                "daily_dd_halt": 0.03 if peak_dd <= 0.10 else 0.08,
                "weekly_dd_halt": 0.07 if peak_dd <= 0.10 else 0.15,
                "daily_dd_reduce": 0.02 if peak_dd <= 0.10 else 0.05,
                "weekly_dd_reduce": 0.05 if peak_dd <= 0.10 else 0.10,
            }
            label = f"{breaker_label} breakers, {cap:.0%} cap"
            m = run_arm(registry, "risk", label, settings, bars, overrides)
            if not m:
                continue
            row(label, m)
            if best is None or m["sharpe"] > best[1]["sharpe"]:
                best = (label, m)

    if best is None:
        print("\n  No arm produced an equity curve.")
        return

    label, metrics = best
    trial_sharpes = registry.sharpes()          # every trial ever, not just this study
    verdict = deflated_sharpe_ratio(metrics["returns"], trial_sharpes)

    print(f"\n  Best arm: {label}")
    print(f"    Sharpe                       {verdict['sr_annual']:.2f}")
    print(f"    Hurdle from {verdict['n_trials']:>3} trials       "
          f"{verdict['hurdle_annual']:.2f}   "
          f"(what the best of {verdict['n_trials']} skill-free arms reaches)")
    print(f"    Probabilistic Sharpe         {verdict['psr']:.3f}   "
          f"(ignores the search)")
    print(f"    Deflated Sharpe              {verdict['dsr']:.3f}   "
          f"(accounts for it)")
    print()
    if verdict["dsr"] >= 0.95:
        print("  Above 0.95. This survives the correction and is worth acting on.")
    else:
        print("  Below 0.95. After accounting for how many configurations were")
        print("  tried, this is not distinguishable from a lucky search. Do not")
        print("  change a setting on the strength of it.")


def study_report(registry) -> None:
    rows = registry.all()
    if not rows:
        print("  No trials recorded yet.")
        return
    print(f"\n  {registry.count()} trials recorded across "
          f"{len({r['study'] for r in rows})} studies\n")
    header(("study / label", "return", "maxDD", "Sharpe", "expo", "trades"),
           (34, 9, 9, 8, 8, 8))
    for r in rows:
        if r["sharpe"] is None:
            continue
        name = f"{r['study']}/{r['label']}"[:33]
        print(f"  {name:<34}{r['total_return']:>9.2%}{r['max_drawdown']:>9.2%}"
              f"{r['sharpe']:>8.2f}{r['avg_exposure']:>8.1%}{r['n_trades']:>8}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("ablation", "risk", "report", "all"))
    parser.add_argument("--db", default=None, help="trial registry path")
    args = parser.parse_args()

    quiet()
    registry = TrialRegistry(args.db) if args.db else TrialRegistry()

    if args.study == "report":
        study_report(registry)
        return 0

    settings = load_settings()
    bars, synthetic = load_universe(settings)
    if synthetic:
        print("  SYNTHETIC data. Refusing: a study on a random walk is worse "
              "than no study.")
        return 1

    primary = settings["broker"]["symbols"][0]
    print(f"  {len(bars)} symbols, {len(bars[primary])} bars, "
          f"{bars[primary].index[0].date()} -> {bars[primary].index[-1].date()}")

    if args.study in ("ablation", "all"):
        study_ablation(registry, settings, bars)
    if args.study in ("risk", "all"):
        study_risk(registry, settings, bars)

    print(f"\n  Registry: {registry.path} ({registry.count()} trials total)")
    registry.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
