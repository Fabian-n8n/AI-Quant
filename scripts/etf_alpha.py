#!/usr/bin/env python
"""Does a strategy have alpha, or just more beta? Ask the ETF that already runs it.

    python scripts/etf_alpha.py NANC KRUZ            # congress-trade trackers
    python scripts/etf_alpha.py MTUM --since 2014-01-01

WHY THIS EXISTS
---------------
Before paying for a data feed or building a signal, check whether someone has
already packaged the idea as a fund. If they have, their track record is a
better test than any backtest run here: real money, real fees, real execution,
real reporting lag, and no way for me to peek at the future while building it.

The number that matters is not total return. A fund can beat SPY purely by
holding more equity risk, and that is not a reason to change anything here --
the same exposure is available for free. So this regresses the fund's daily
returns on the benchmark's and reports the intercept:

    fund_return = alpha + beta * benchmark_return + noise

`alpha` is the part the benchmark does not explain, annualised. `t(alpha)` says
whether it is distinguishable from zero. This project uses |t| >= 2 everywhere
else (variant 3 rejected the HMM on exactly that bar) so it uses it here too.

READ THE STANDARD ERROR, NOT JUST THE VERDICT. A short track record cannot
detect a small alpha. "Not distinguishable from zero" means the data cannot
tell, not that the alpha is provably zero -- the printed detection floor says
how large an alpha would have had to be before this sample could see it.
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

from data.market_data import load_bars  # noqa: E402

TBAR = 2.0          # |t| for "distinguishable from zero", as used across this repo


def closes(symbol: str, since: str) -> pd.Series | None:
    frame, synthetic = load_bars(symbol, since, None)
    if synthetic:
        print(f"  {symbol}: synthetic data, skipped. A regression on a random walk says nothing.")
        return None
    return frame["close"]


def regress(fund: np.ndarray, bench: np.ndarray) -> dict:
    """OLS of fund on benchmark. Returns annualised alpha, its t-stat, beta, R^2."""
    x = np.column_stack([np.ones(len(bench)), bench])
    coef, *_ = np.linalg.lstsq(x, fund, rcond=None)
    resid = fund - x @ coef
    # ddof=2: two parameters estimated, so the residual variance loses two df.
    se = np.sqrt(np.diag(resid.var(ddof=2) * np.linalg.inv(x.T @ x)))
    return {
        "alpha": coef[0] * 252,
        "t": coef[0] / se[0],
        "beta": coef[1],
        "r2": 1 - resid.var() / fund.var(),
        "floor": TBAR * se[0] * 252,      # smallest alpha this sample could detect
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("funds", nargs="+", help="fund tickers to test")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--since", default="2023-03-01")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    bench = closes(args.benchmark, args.since)
    if bench is None:
        return 1
    series = {args.benchmark: bench}
    for f in args.funds:
        got = closes(f, args.since)
        if got is not None:
            series[f] = got

    # One shared calendar. A fund listed later than --since would otherwise be
    # compared against a benchmark window it never traded in.
    rets = pd.DataFrame({s: v.reindex(bench.index).ffill().pct_change()
                         for s, v in series.items()}).dropna()
    if len(rets) < 60:
        print(f"  Only {len(rets)} overlapping bars. Too short to regress.")
        return 1

    print(f"\n  {len(rets)} daily bars, {rets.index[0].date()} -> {rets.index[-1].date()}, "
          f"benchmark {args.benchmark}\n")
    print(f"  {'fund':<7}{'total':>9}{'Sharpe':>8}{'alpha/yr':>10}{'t':>7}"
          f"{'beta':>7}{'R2':>7}   verdict")
    b_ret = rets[args.benchmark].values
    for s in [args.benchmark, *args.funds]:
        if s not in rets:
            continue
        r = rets[s].values
        total = float(np.prod(1 + r) - 1)
        sharpe = float(r.mean() / r.std() * np.sqrt(252))
        if s == args.benchmark:
            print(f"  {s:<7}{total:>8.1%}{sharpe:>8.2f}{'--':>10}{'--':>7}{1.0:>7.2f}{1.0:>7.2f}"
                  f"   (benchmark)")
            continue
        m = regress(r, b_ret)
        verdict = ("alpha survives" if abs(m["t"]) >= TBAR
                   else f"not distinguishable from 0 (needs >{m['floor']:.1%}/yr to detect)")
        print(f"  {s:<7}{total:>8.1%}{sharpe:>8.2f}{m['alpha']:>9.2%}{m['t']:>7.2f}"
              f"{m['beta']:>7.2f}{m['r2']:>7.2f}   {verdict}")

    print(f"\n  A fund beating {args.benchmark} on total return while its alpha t-stat sits near")
    print(f"  zero is holding more {args.benchmark}, not finding something. Beta is free here.\n")
    return 0


def _self_check() -> None:
    """One runnable check: a fund that IS the benchmark plus a known drift."""
    rng = np.random.default_rng(0)
    bench = rng.normal(0.0004, 0.01, 4000)
    drift = 0.0004                                      # ~10%/yr of pure alpha
    m = regress(1.2 * bench + drift, bench)
    assert abs(m["beta"] - 1.2) < 0.01, m               # recovers beta
    assert abs(m["alpha"] - drift * 252) < 0.01, m      # recovers alpha
    assert m["t"] > TBAR, m                             # noiseless alpha is detectable
    m0 = regress(1.2 * bench + rng.normal(0, 0.01, 4000), bench)
    assert abs(m0["t"]) < TBAR, m0                      # pure noise is not
    print("self-check ok")


if __name__ == "__main__":
    raise SystemExit(_self_check() if "--self-check" in sys.argv else main())
