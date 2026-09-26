#!/usr/bin/env python
"""What did the risk gates cost, or save? Measured, not argued.

    python scripts/counterfactual.py
    python scripts/counterfactual.py --hold 20 --json

THE QUESTION
------------
`data/state.db` records every signal the strategy produced, including the ones
the risk layer refused, with the reason. Nothing in that record says what the
refusals cost or saved. A gate that blocks trades which would have lost is
earning its place; one that blocks trades which would have won is a tax being
paid in silence, and it looks identical in the log.

Adopted in idea from `bennyjo/phil`'s `core/counterfactual.py`, which does this
for declined prediction-market bets. The instrument is different, the question
is the same, and the code here is ours: it replays equity bars against the stop
and target that were actually recorded on the signal.

THE CONTROL, WHICH PHIL'S VERSION DOES NOT HAVE
-----------------------------------------------
Knowing refused trades lost money is not enough. The gate is only doing work if
the trades it ALLOWED did better than the ones it refused. So approved signals
are replayed through the identical simulation and reported beside them. If both
sides land in the same place, the gate is not discriminating, it is only
reducing turnover.

WHY THE COUNTERFACTUAL FLATTERS ITSELF, AND BY HOW MUCH
-------------------------------------------------------
Read the `spread_too_wide` rows with this in mind. Those signals were refused
*because* the quoted spread was wide, and this simulation fills them at the
recorded `entry_price` as though it were not. The real fill would have been
worse by roughly half the spread on entry and again on exit. So the refused-row
P&L printed here is an UPPER BOUND on what taking them would have made, and the
`spread_too_wide` bound is the loosest of the three.

Two more limits worth holding: the sample is small, and signals cluster on the
same few days, so the rows are not independent draws.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB = Path(__file__).resolve().parent.parent / "data" / "state.db"


def replay(bars, entry_price, stop, target, hold):
    """Walk bars forward from the signal until stop, target, or the horizon.

    Same ordering rule as `PortfolioBacktester._exit_price`: the stop is
    checked before the target, because within one daily bar the sequence of the
    high and the low is unknown and assuming the good one first is how a
    backtest invents money. A gap through the level fills at the open.
    """
    for i in range(min(hold, len(bars))):
        row = bars.iloc[i]
        low, high, open_ = float(row["low"]), float(row["high"]), float(row["open"])
        if stop and low <= stop:
            return (min(open_, stop) / entry_price - 1), "stop", i + 1
        if target and high >= target:
            return (max(open_, target) / entry_price - 1), "target", i + 1
    if len(bars) == 0:
        return None, "no bars", 0
    last = float(bars.iloc[min(hold, len(bars)) - 1]["close"])
    return (last / entry_price - 1), "horizon", min(hold, len(bars))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hold", type=int, default=20, help="max bars to hold (default 20)")
    parser.add_argument("--db", default=str(DB))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    from data.market_data import load_bars

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        "SELECT symbol, bar_date, entry_price, stop_loss, take_profit, approved, "
        "rejection_reason FROM signals WHERE entry_price > 0 ORDER BY bar_date"))
    if not rows:
        print("  No signals recorded yet.")
        return 1

    cache: dict[str, object] = {}
    buckets: dict[str, list] = defaultdict(list)
    skipped = incomplete = 0
    for row in rows:
        symbol = row["symbol"]
        if symbol not in cache:
            try:
                frame, synthetic = load_bars(symbol, "2026-01-01", None)
                cache[symbol] = None if synthetic else frame
            except Exception:
                cache[symbol] = None
        frame = cache[symbol]
        if frame is None:
            skipped += 1
            continue
        # Strictly AFTER the signal's bar: acting on the bar that produced the
        # signal would be the look-ahead this repo tests for everywhere else.
        forward = frame.loc[frame.index > str(row["bar_date"])[:10]]
        # A signal whose forward window has not finished yet CANNOT be compared
        # with one whose has. Three bars of drift is not a 20-bar outcome: the
        # trade has had almost no chance to reach its target, and its "horizon"
        # exit is noise. Including them silently is what made the first run of
        # this script report that refused trades beat approved ones -- every
        # refusal sat in the first week with a full window, while a third of
        # the approvals were from the last few days with three bars each.
        if len(forward) < args.hold:
            incomplete += 1
            continue
        pnl, reason, held = replay(forward, float(row["entry_price"]),
                                   row["stop_loss"], row["take_profit"], args.hold)
        if pnl is None:
            skipped += 1
            continue
        key = "APPROVED" if row["approved"] else (row["rejection_reason"] or "rejected")
        buckets[key].append({"symbol": symbol, "pnl_pct": pnl, "date": str(row["bar_date"])[:10],
                             "exit": reason, "bars_held": held})

    def summarise(name, entries):
        n = len(entries)
        wins = sum(1 for e in entries if e["pnl_pct"] > 0)
        mean = sum(e["pnl_pct"] for e in entries) / n
        total = sum(e["pnl_pct"] for e in entries)
        return {"bucket": name, "n": n, "win_rate": wins / n,
                "mean_pnl_pct": mean, "total_pnl_pct": total,
                "stopped": sum(1 for e in entries if e["exit"] == "stop"),
                "targeted": sum(1 for e in entries if e["exit"] == "target")}

    report = [summarise(k, v) for k, v in buckets.items() if v]
    report.sort(key=lambda r: (r["bucket"] != "APPROVED", -r["n"]))

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"\n  {len(rows)} signals: {incomplete} still inside their {args.hold}-bar "
          f"window, {skipped} unusable\n")
    print(f"  {'bucket':<22}{'n':>5}{'win rate':>10}{'mean':>9}{'total':>9}"
          f"{'stops':>7}{'targets':>9}")
    approved = None
    for r in report:
        if r["bucket"] == "APPROVED":
            approved = r
        print(f"  {r['bucket']:<22}{r['n']:>5}{r['win_rate']:>9.0%}"
              f"{r['mean_pnl_pct']:>8.2%}{r['total_pnl_pct']:>8.2%}"
              f"{r['stopped']:>7}{r['targeted']:>9}")

    # The verdict the table exists to produce.
    refused = [r for r in report if r["bucket"] != "APPROVED"]
    if approved and refused:
        pooled_n = sum(r["n"] for r in refused)
        pooled = sum(r["mean_pnl_pct"] * r["n"] for r in refused) / pooled_n
        gap = approved["mean_pnl_pct"] - pooled
        print(f"\n  approved mean {approved['mean_pnl_pct']:+.2%} vs refused mean {pooled:+.2%}"
              f"   gap {gap:+.2%}")
        if gap > 0:
            print("  The gates let through the better trades. They are doing work.")
        else:
            print("  The refused trades did BETTER. On this sample the gates cost money,")
            print("  and the spread rows are an upper bound that flatters them further.")
    # The buckets do not cover the same calendar days, and a week of market
    # direction dwarfs any gate effect. Restrict both sides to days where both
    # actually occur, which is the only comparison that controls for it.
    approved_days = {e["date"] for e in buckets.get("APPROVED", [])}
    refused_days = {e["date"] for r in buckets for e in buckets[r] if r != "APPROVED"}
    shared = approved_days & refused_days
    if shared:
        a = [e for e in buckets.get("APPROVED", []) if e["date"] in shared]
        f = [e for r in buckets for e in buckets[r] if r != "APPROVED" and e["date"] in shared]
        if a and f:
            am = sum(e["pnl_pct"] for e in a) / len(a)
            fm = sum(e["pnl_pct"] for e in f) / len(f)
            print(f"\n  ON THE {len(shared)} DAYS BOTH OCCUR (the controlled comparison):")
            print(f"    approved  n={len(a):<4} mean {am:+.2%}   win rate "
                  f"{sum(1 for e in a if e['pnl_pct'] > 0) / len(a):.0%}")
            print(f"    refused   n={len(f):<4} mean {fm:+.2%}   win rate "
                  f"{sum(1 for e in f if e['pnl_pct'] > 0) / len(f):.0%}")
            print(f"    gap {am - fm:+.2%}")

    print("\n  Small sample, clustered dates, and refused fills assume a spread")
    print("  that was the reason for the refusal. Treat as a direction, not a P&L.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
