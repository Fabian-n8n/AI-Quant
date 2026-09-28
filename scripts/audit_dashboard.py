#!/usr/bin/env python
"""Does the dashboard show what the account actually holds?

    python scripts/audit_dashboard.py

Compares every field in `dashboard/public/data/state.json` against Alpaca and
exits non-zero on any mismatch, so it can gate a deploy.

USE THE LAST TRADE, NOT `position.current_price`
------------------------------------------------
The obvious check -- compare against `position.current_price` -- is wrong
outside regular hours, and confidently wrong. Alpaca derives that field from
the quote, and off-hours quotes are garbage: measured 2026-09-28 at 07:30 ET,
GLD quoted bid 380.38 against an ask of **0.00**, and Alpaca reported
`current_price` 380.62 while the last actual trade was 393.02. On a position
with a stop at 390.19 that is the difference between "fine" and "about to be
sold", and the naive version of this script raised exactly that false alarm.

`_remark_if_closed` already re-marks positions off the last trade for this
reason. This script has to use the same source or it will keep reporting the
engine as broken every night and be ignored by morning.

What it checks per position: quantity, entry, current price, stop price, and
the two derived fields `locked_pnl` and `trail_pct` recomputed from scratch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PAYLOAD = Path(__file__).resolve().parent.parent / "dashboard" / "public" / "data" / "state.json"
TOL = 0.02


def main() -> int:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    trading = TradingClient(key, secret, paper=True)
    data = StockHistoricalDataClient(key, secret)

    positions = {p.symbol: p for p in trading.get_all_positions()}
    stops = {o.symbol: float(o.stop_price) for o in
             trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=300))
             if o.stop_price}
    if not positions:
        print("  No open positions to check.")
        return 0
    trades = data.get_stock_latest_trade(
        StockLatestTradeRequest(symbol_or_symbols=list(positions)))

    published = {r["symbol"]: r for r in
                 json.loads(PAYLOAD.read_text())["activity"]["open_positions"]}

    problems: list[str] = []
    print(f"\n  {'sym':<7}{'last trade':>11}{'published':>11}{'stop':>9}"
          f"{'locked':>11}{'trail':>9}")
    for symbol, position in sorted(positions.items()):
        row = published.get(symbol)
        if row is None:
            problems.append(f"{symbol}: held at the broker, missing from the payload")
            continue

        entry, qty = float(position.avg_entry_price), float(position.qty)
        last = float(trades[symbol].price)
        stop = stops.get(symbol)

        if abs(row["quantity"] - qty) > 1e-6:
            problems.append(f"{symbol}.quantity: broker {qty}, payload {row['quantity']}")
        if abs(row["entry_price"] - entry) > TOL:
            problems.append(f"{symbol}.entry: broker {entry}, payload {row['entry_price']}")
        if abs((row["current_price"] or 0) - last) > TOL:
            problems.append(f"{symbol}.current: last trade {last}, payload {row['current_price']}")
        if stop is None and row["stop_price"] is not None:
            problems.append(f"{symbol}: payload shows a stop, the broker has none")
        elif stop is not None and abs(stop - (row["stop_price"] or 0)) > TOL:
            problems.append(f"{symbol}.stop: broker {stop}, payload {row['stop_price']}")

        want_locked = round((stop - entry) * qty, 2) if stop and stop > entry else None
        if abs((row["locked_pnl"] or 0) - (want_locked or 0)) > TOL:
            problems.append(
                f"{symbol}.locked_pnl: expected {want_locked}, payload {row['locked_pnl']}")

        want_trail = round((last - stop) / last, 6) if stop and last else None
        if want_trail is not None and abs((row["trail_pct"] or 0) - want_trail) > 5e-4:
            problems.append(
                f"{symbol}.trail_pct: expected {want_trail:.4f}, payload {row['trail_pct']}")

        trail = row["trail_pct"]
        print(f"  {symbol:<7}{last:>11.2f}{(row['current_price'] or 0):>11.2f}{(stop or 0):>9.2f}"
              f"{(f'+{want_locked:.2f}' if want_locked else '--'):>11}"
              f"{(f'{trail * 100:.2f}%' if trail is not None else '--'):>9}")

    missing = set(published) - set(positions)
    if missing:
        problems.append(f"payload shows positions the account does not hold: {sorted(missing)}")

    print(f"\n  {len(positions)} positions checked against the broker")
    if problems:
        print(f"  {len(problems)} MISMATCH(ES):")
        for p in problems:
            print(f"    {p}")
        return 1
    print("  every published field matches.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
