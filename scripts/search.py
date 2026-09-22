#!/usr/bin/env python
"""The iteration loop: generate variants, score every one, correct for the search.

    python scripts/search.py            search, then one holdout run
    python scripts/search.py --report   what the registry already knows

WHAT MAKES THIS DIFFERENT FROM A SWEEP
--------------------------------------
Sweeping configurations and reporting the winner is not research, it is a way
of manufacturing an edge that is not there. A viral post doing the rounds
reports 100,384 backtests and a best Sharpe of 2.18; at that trial count the
best of *skill-free* strategies reaches about 4.39, so their winner is half the
noise floor. The sweep was the problem, not the evidence.

Three things stop that happening here, and none are optional:

1. Every arm is written to `data/trials.db`, including the failures, so the
   trial count cannot be quietly understated later.
2. The winner is scored with a deflated Sharpe against that full count.
3. A holdout slice is never touched during the search and is run exactly once.

WHAT IT SEARCHES OVER
---------------------
Signals, never HMM hyperparameters. Variant 3 closed that question: the
classifier carries no forward-return information at any state count, so another
state count is not a variant worth a run.

The dimension that matters is the **universe**. A long-only book of 14 US tech
names cannot profit when US tech falls; the best it can do is lose less, which
is what every backtest here has shown. Holding assets that rise in different
conditions is the only structural answer to "works in any market", and the
candidate ranker already does the work: it scores price above its own EMA50 and
20-day return, so given defensive sleeves it rotates into whatever is trending
without a line of new allocation logic. That is dual momentum, reusing what is
already here.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.performance import ic_information_ratio  # noqa: E402
from backtest.portfolio_backtester import PortfolioBacktester  # noqa: E402
from backtest.trials import Trial, TrialRegistry, deflated_sharpe_ratio  # noqa: E402
from config import load_settings, strategy_config  # noqa: E402
from data.market_data import load_bars  # noqa: E402

EQUITY = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "AMD", "TSLA", "AVGO", "PLTR", "COIN", "SMCI"]
#: Equities plus sleeves that rise when equities do not: long and intermediate
#: Treasuries, gold, broad commodities, and T-bills as a cash proxy.
DEFENSIVE = ["TLT", "IEF", "GLD", "DBC", "BIL"]

UNIVERSES = {"equity": EQUITY, "diversified": EQUITY + DEFENSIVE}
HOLDOUT_START = "2023-07-01"        # never seen during the search
HISTORY_START = "2015-01-01"


def quiet() -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)


def load(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for symbol in symbols:
        frame, synthetic = load_bars(symbol, HISTORY_START, None)
        if synthetic:
            raise SystemExit(f"{symbol}: synthetic data. A study on a random walk is worse than none.")
        out[symbol] = frame
    return out


def score(result, initial: float, primary: pd.Series) -> dict | None:
    """Return, risk, and whether the exposure signal predicts anything."""
    equity = result.equity_curve
    if equity.empty or len(equity) < 60:
        return None
    daily = equity.pct_change().dropna()
    exposure = result.history.set_index("timestamp")["exposure"] \
        if "timestamp" in result.history else result.history["exposure"]
    try:
        icir = float(ic_information_ratio(exposure, primary.reindex(exposure.index).dropna()))
    except Exception:
        icir = float("nan")
    return {
        "total_return": float(equity.iloc[-1] / initial - 1),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "avg_exposure": result.avg_exposure,
        "n_trades": result.n_trades,
        "n_bars": len(equity),
        "icir": icir,
        "returns": daily,
        "equity": equity,
    }


def build(settings, bars, universe, regime_source, trend="off"):
    cfg = strategy_config(settings)
    cfg["regime_source"] = regime_source
    cfg["trend_filter"] = trend
    return PortfolioBacktester(
        symbols=list(bars), primary="SPY",
        train_window=settings["backtest"]["train_window"],
        test_window=settings["backtest"]["test_window"],
        step_size=settings["backtest"]["step_size"],
        initial_capital=settings["backtest"]["initial_capital"],
        slippage_pct=settings["backtest"]["slippage_pct"],
        hmm_config=dict(settings["hmm"]), strategy_config=cfg,
        risk_config=dict(settings["risk"]),
        reward_risk_ratio=settings["risk"].get("reward_risk_ratio", 2.0),
        entry_fill="limit",
    )


def run_arm(settings, all_bars, universe, regime_source, upto=None, trend="off"):
    bars = {s: all_bars[s] for s in UNIVERSES[universe] if s in all_bars}
    if upto:
        bars = {s: f.loc[:upto] for s, f in bars.items()}
    result = build(settings, bars, universe, regime_source, trend).run(bars)
    return score(result, settings["backtest"]["initial_capital"], bars["SPY"]["close"])


def by_year(equity: pd.Series) -> pd.Series:
    return equity.resample("YE").last().pct_change().dropna()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="print the registry and exit")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    quiet()
    registry = TrialRegistry(args.db) if args.db else TrialRegistry()

    if args.report:
        rows = registry.all()
        print(f"\n  {registry.count()} trials recorded\n")
        print(f"  {'study/label':<40}{'return':>9}{'maxDD':>9}{'Sharpe':>8}{'trades':>8}")
        for r in rows:
            if r["sharpe"] is None:
                continue
            print(f"  {(r['study'] + '/' + r['label'])[:39]:<40}{r['total_return']:>8.1%}"
                  f"{r['max_drawdown']:>9.1%}{r['sharpe']:>8.2f}{r['n_trades']:>8}")
        registry.close()
        return 0

    settings = load_settings()
    all_bars = load(sorted(set(EQUITY + DEFENSIVE)))
    print(f"  {len(all_bars)} symbols loaded, holdout begins {HOLDOUT_START}\n")

    # ---- search, on data before the holdout only --------------------------
    print("SEARCH (holdout withheld)\n")
    print(f"  {'universe':<14}{'regime':<12}{'trend':<9}{'return':>9}{'maxDD':>9}{'Sharpe':>8}"
          f"{'ICIR':>8}{'expo':>7}{'trades':>8}")
    arms = {}
    for universe, source, trend in product(UNIVERSES, ("hmm", "volatility"), ("off", "sma200")):
        metrics = run_arm(settings, all_bars, universe, source, upto=HOLDOUT_START, trend=trend)
        if not metrics:
            continue
        label = f"{universe}-{source}-{trend}"
        arms[label] = metrics
        registry.record(Trial(
            study="search", label=label,
            config={"universe": universe, "regime_source": source,
                    "trend_filter": trend, "span": "pre-holdout"},
            total_return=metrics["total_return"], sharpe=metrics["sharpe"],
            max_drawdown=metrics["max_drawdown"], avg_exposure=metrics["avg_exposure"],
            n_trades=metrics["n_trades"], n_bars=metrics["n_bars"],
        ))
        print(f"  {universe:<14}{source:<12}{trend:<9}{metrics['total_return']:>8.1%}"
              f"{metrics['max_drawdown']:>9.1%}{metrics['sharpe']:>8.2f}"
              f"{metrics['icir']:>8.2f}{metrics['avg_exposure']:>7.1%}{metrics['n_trades']:>8}")

    if not arms:
        print("\n  No arm produced an equity curve.")
        registry.close()
        return 1

    # Ranked on Sharpe, not total return. A curve that gains steadily survives
    # contact with a live account; one that gained it all in 2020 does not.
    best_label = max(arms, key=lambda k: arms[k]["sharpe"])
    best = arms[best_label]
    verdict = deflated_sharpe_ratio(best["returns"], registry.sharpes())
    print(f"\n  Best in search: {best_label}")
    print(f"    Sharpe                {verdict['sr_annual']:.2f}")
    print(f"    Hurdle, {verdict['n_trials']:>3} trials    {verdict['hurdle_annual']:.2f}"
          f"   (what the best of {verdict['n_trials']} skill-free arms reaches)")
    print(f"    Deflated Sharpe       {verdict['dsr']:.3f}   "
          f"{'survives the correction' if verdict['dsr'] >= 0.95 else 'NOT distinguishable from a lucky search'}")

    # ---- holdout: one run, no tuning afterwards ---------------------------
    print(f"\nHOLDOUT — {best_label}, full span including {HOLDOUT_START} onward")
    print("One run. Changing anything after reading this makes it no longer a holdout.\n")
    universe, source, trend = best_label.split("-")
    full = run_arm(settings, all_bars, universe, source, trend=trend)
    if full:
        print(f"  {'':<14}{'return':>9}{'maxDD':>9}{'Sharpe':>8}{'expo':>7}{'trades':>8}")
        print(f"  {'full span':<14}{full['total_return']:>8.1%}{full['max_drawdown']:>9.1%}"
              f"{full['sharpe']:>8.2f}{full['avg_exposure']:>7.1%}{full['n_trades']:>8}")
        registry.record(Trial(
            study="holdout", label=best_label,
            config={"universe": universe, "regime_source": source,
                    "trend_filter": trend, "span": "full"},
            total_return=full["total_return"], sharpe=full["sharpe"],
            max_drawdown=full["max_drawdown"], avg_exposure=full["avg_exposure"],
            n_trades=full["n_trades"], n_bars=full["n_bars"],
        ))

        # The question that actually matters: does it make money when the
        # market does not? Per calendar year against SPY.
        spy = all_bars["SPY"]["close"].reindex(full["equity"].index).ffill()
        print(f"\n  {'year':<8}{'strategy':>11}{'SPY':>10}")
        strat_y, spy_y = by_year(full["equity"]), by_year(spy)
        for year in strat_y.index:
            s_val, b_val = strat_y.loc[year], spy_y.get(year, float("nan"))
            flag = "  <- SPY down" if b_val < 0 else ""
            print(f"  {year.year:<8}{s_val:>10.1%}{b_val:>10.1%}{flag}")
        down = [y for y in strat_y.index if spy_y.get(y, 0) < 0]
        if down:
            print(f"\n  In the {len(down)} year(s) SPY fell, the strategy averaged "
                  f"{np.mean([strat_y.loc[y] for y in down]):.1%}.")

    print(f"\n  Registry: {registry.path} ({registry.count()} trials total)")
    registry.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
