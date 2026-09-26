#!/usr/bin/env python
"""Re-price closed positions from the broker's actual fills.

    python scripts/repair_exits.py            show what would change
    python scripts/repair_exits.py --apply    write it

WHY THIS EXISTS
---------------
`main.py` used to record an exit at `current_price`, the last mark taken
before the position vanished, and infer the reason from where that mark sat
relative to the stop. The fill price was never asked for, although the filled
order carries it.

The error was one-directional. The last mark predates the fall through the
stop, so every long exit was booked ABOVE what it really made. Across the
first ten closed trades on the paper account the table said -$348.92 and the
broker's fills said -$818.74: overstated by $469.82, with a reported win rate
of 20% against a true rate of zero.

`scripts/preflight.py` gates live trading on closed-trade count and positive
expectancy read from this table, so the bias pointed at opening that gate.

The live path is fixed. This repairs rows written before the fix. It is
idempotent: rows already matching their fill are left alone.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB = Path(__file__).resolve().parent.parent / "data" / "state.db"
REASONS = {"stop": "stop", "stop_limit": "stop", "trailing_stop": "trailing stop",
           "limit": "target", "market": "closed"}


def broker_sells() -> dict[str, list[tuple]]:
    """Every filled sell, newest last, grouped by symbol."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    client = TradingClient(os.environ["ALPACA_API_KEY"],
                           os.environ["ALPACA_SECRET_KEY"], paper=True)
    out: dict[str, list[tuple]] = {}
    for order in client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500)):
        if not order.filled_at or not order.filled_avg_price:
            continue
        if not str(order.side).lower().endswith("sell"):
            continue
        out.setdefault(order.symbol, []).append((
            str(order.filled_at)[:10], float(order.filled_avg_price),
            REASONS.get(str(order.order_type).split(".")[-1].lower(), "closed"),
        ))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the corrections")
    parser.add_argument("--db", default=str(DB))
    args = parser.parse_args()

    sells = broker_sells()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        "SELECT * FROM positions WHERE exit_at IS NOT NULL ORDER BY exit_at"))

    print(f"\n  {len(rows)} closed positions in {args.db}\n")
    print(f"  {'sym':<7}{'entry':>9}{'was':>9}{'is':>9}{'old pnl':>10}{'new pnl':>10}  reason")
    fixes, old_total, new_total = [], 0.0, 0.0
    for row in rows:
        # Match on the exit DATE: a symbol can be traded more than once, and
        # the id is not carried on the position row.
        same_day = [s for s in sells.get(row["symbol"], []) if s[0] == str(row["exit_at"])[:10]]
        old_pnl = row["realised_pnl"] or 0.0
        old_total += old_pnl
        if not same_day:
            new_total += old_pnl
            print(f"  {row['symbol']:<7}{row['entry_price']:>9.2f}"
                  f"{(row['exit_price'] or 0):>9.2f}{'-':>9}{old_pnl:>10.2f}{'-':>10}  no fill found, left alone")
            continue
        price, reason = same_day[-1][1], same_day[-1][2]
        pnl = (price - row["entry_price"]) * row["quantity"]
        new_total += pnl
        print(f"  {row['symbol']:<7}{row['entry_price']:>9.2f}{(row['exit_price'] or 0):>9.2f}"
              f"{price:>9.2f}{old_pnl:>10.2f}{pnl:>10.2f}  {reason}")
        if abs(price - (row["exit_price"] or 0)) > 0.005 or reason != row["exit_reason"]:
            fixes.append((price, reason, pnl, row["id"]))

    print(f"\n  reported net  ${old_total:>10.2f}")
    print(f"  actual net    ${new_total:>10.2f}")
    print(f"  difference    ${new_total - old_total:>10.2f}")
    wins = sum(1 for r in rows if (r['realised_pnl'] or 0) > 0)
    print(f"\n  reported win rate {wins}/{len(rows)}")

    if not fixes:
        print("\n  Nothing to correct.\n")
        return 0
    if not args.apply:
        print(f"\n  {len(fixes)} row(s) would change. Re-run with --apply to write.\n")
        return 0

    conn.executemany(
        "UPDATE positions SET exit_price = ?, exit_reason = ?, realised_pnl = ? WHERE id = ?",
        fixes)
    conn.commit()
    print(f"\n  Corrected {len(fixes)} row(s).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
