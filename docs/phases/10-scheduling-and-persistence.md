# Phase 10: scheduling, persistence, and the live gate

Eight changes that move the system from "runs a minute-cadence loop against
demo data" to "runs once a day on a schedule, records what it did, and can
prove it before going live".

---

## 0. The bug that was not a bug

The work started from a report that Alpaca paper orders never filled: repeated
SPY buy limits at $616.50, status `canceled`, `filled_qty` 0, roughly every
60-80 seconds, against a market around $765.

Every detail was accurate. The conclusion was not. Tracing the price path:

```
tests/test_integration.py   entry = round(reference * 0.80, 2)

  SPY bid (live)      769.85
  x 0.80              615.88     <- the test prices 20% below market on purpose
  _limit_price x1.001 616.4968
  round(, 2)          616.50     <- exact match, all 30 orders
```

`TestAlpacaPaperRoundTrip` buys 1 share far below the touch so the order
*rests* instead of filling, asserts it is open, cancels it, and asserts clean
state. `qty=1`, `stop=None`, `canceled`, `filled_qty=0` are all it working
correctly. The 60-80 second cadence was repeated pytest runs, not a loop.

The production path was healthy throughout: `_reference_price` reads the live
quote and only falls back to the signal's entry price when the quote is
unusable.

**What was actually wrong** was that there was no way to tell a test order from
a real one while looking at the Alpaca dashboard. Orders now carry an
`order_id_prefix` in `client_order_id`: `rt-` for real, `itest-` for the
integration suite. That is the fix. The price guard below is defence in depth
for a bug that did not exist yet.

Worth keeping in mind generally: a report can be entirely correct about the
symptoms and still point at the wrong cause, and the cheapest way to find out
is to trace the actual values rather than reason about the code.

---

## 1. Price sanity guard

`_assert_price_sane` refuses any limit more than `max_price_deviation_pct`
(5%) from the live quote, naming both prices.

The failure it exists for is silent. A limit far from the market is accepted by
the broker, rests, and expires. Nothing raises, so the only symptom is an order
log full of `canceled` and an account that never moves.

A dead quote skips the check rather than blocking. Outside regular hours the
IEX feed returns a zero bid and ask, and refusing every after-hours order
because the reference is missing trades one silent failure for a louder one.

The integration suite opts out per call, on the one call site that needs it,
rather than lowering the cap. A guard the test disables globally protects
nothing.

## 2. Idempotency

Two mechanisms, because they fail in different places.

`_equivalent_open_order` asks the broker whether an order for this symbol, side
and roughly this price is already resting. Price matches within 0.5% rather
than exactly, since the limit derives from a quote that moves between cycles.
This matters beyond duplication: every open order holds buying power, so a loop
that reposts starves the rest of the universe.

The deterministic `client_order_id` (`rt-SPY-buy-20260904`) catches what the
broker query cannot. A restarted process has no memory, and a filled order is
no longer open. Alpaca rejects the repeated id itself, making the broker the
arbiter rather than our process. Keyed on `Signal.timestamp`, which was already
the bar timestamp.

A skip returns a `TradeRecord` with `skipped_reason` set rather than raising.
Routing it through the exception path would file the idempotency layer working
correctly alongside genuine broker errors.

## 3. Daily cadence

`core/calendar.py` wraps Alpaca's calendar endpoint. `get_clock().is_open` was
the wrong question for a scheduled job: it is false both on Thanksgiving and at
09:29 on a Tuesday without distinguishing them.

Verified against live data: 2026-09-07 correctly not a trading day (Labor Day),
2026-11-27 correctly closing at 13:00.

**Daylight saving is the reason the check lives inside the job.** 21:15 UTC is
17:15 ET in winter and 16:15 ET in summer. No cron expression means "after the
close" all year, so cron fires on the wider of the two and the calendar decides
whether the session actually finished. Both boundaries are pinned by tests.

`_bar_has_closed` refuses to act on a session still in progress. A forming
bar's close, high and low all still change, so trading one is look-ahead bias
arriving through the data feed rather than through the feature code.

Two cadences: entries wait for the daily decision, because the signal comes
from a bar that will not change. Stops do not. `monitor_positions` runs every
`monitor_seconds` and structurally cannot open a position.

## 4. Persistence

**Bars** are partitioned by symbol at `data/cache/bars/{SYMBOL}_{tf}_{adj}.parquet`,
append-only. The old scheme hashed the whole request into one filename, so a
window one day longer shared nothing and refetched four years. Nineteen files
had accumulated, none reused. Cold 1.13s, warm 0.29s.

Gaps shorter than one bar are ignored. `start` carries a time of day and the
first stored bar sits at the session open, so a naive comparison finds a
thirteen-hour gap at the head of every request and refetches it forever.

**State** is `data/state.db`: stdlib `sqlite3`, plain SQL, numbered migrations,
WAL. Six tables: `signals`, `orders`, `positions`, `equity_snapshots`,
`breaker_events`, `runs`.

The run row is opened at startup, not written at the end, so a process that
dies mid-bar leaves evidence. "It failed" and "it never ran" need different
responses.

`active_breakers` orders by id, not `occurred_at`. Timestamps store to the
second, so a breaker tripped and cleared within one second matched both rows
and reported as still tripped.

## 5. Scheduling

`.github/workflows/trade.yml`, weekdays 21:15 UTC plus `workflow_dispatch`.
Separate from `ci.yml`, because CI runs on every push and must never reach a
broker.

`DRY_RUN` is a repository variable defaulting to `true`. A week of intended
orders appears in the logs, then flipping the variable goes live. No code
change, so nothing to accidentally revert.

`state_snapshot.json` is committed. The runner is a fresh machine each time, so
without it the peak-equity baseline resets to whatever the account is worth
today and the drawdown breaker can never fire. `trading_halted.lock` stays
ignored: committing a halt would propagate it to every future run.

## 6. Trailing stops

Alpaca does not allow a trailing stop as a bracket or OCO leg, so it is a
standalone order placed after the entry fills. `protect_position` tries the
trailing stop, falls back to a plain stop, and raises only if both fail.

`trail_percent` is derived from ATR. On real values: 2.5 ATR is 2.05% on SPY
and 11.70% on COIN. A fixed 5% would be six ATR of slack on one and one ATR on
the other.

Three documented limitations, repeated in code and config because each has cost
someone money by being forgotten:

1. Trailing stops do not trigger outside regular hours. Overnight gaps are
   unprotected until 09:30.
2. When triggered they become market orders, so the fill can be worse than the
   trail.
3. `time_in_force` must be `day` or `gtc`.

`audit_stops` now counts `TRAILING_STOP` as a resting stop. Without that every
trailing-stopped position reads as naked, and an alert that cries wolf gets
muted. It also gained `repair=True`, used at startup: positions adopted at
startup were opened by a process that no longer exists.

## 7. Dashboard

Stayed on Next.js rather than converting to Streamlit as the brief said.
Streamlit needs a persistent Python server, so it cannot deploy to Vercel, and
the Vercel link was the requirement. Sidebar plus an `/activity` route.

Activity's job is keeping orders and trades apart. A cancelled order with
`filled_qty` 0 moved no money. Unfilled rows are dimmed and only fills show a
fill price, because that exact confusion is what started this phase.

Two bugs found while wiring it: the fetch used a relative path, which on
`/activity/` resolves to `/activity/data/state.json` and 404s; and the contract
test fixture had drifted from `publish_demo`, building a payload with no
activity block while the shipped file had one. Both now call `demo_payload()`.

## 8. Preflight

`scripts/preflight.py`. Six checks, all must pass before `main.py` runs live.

It fails today, which is correct. Against the real exported backtest:

| | Total return |
|---|---|
| regime-trader | **+55.0%** |
| buy-and-hold | +118.3% |
| random allocation | +84.7% |
| SMA-200 trend | -0.1% |

It beats only the benchmark that lost money. Losing to random entry means the
regime signal is adding nothing.

Every item here was already in `docs/`, unticked, while the system happily
accepted `--i-understand-live`. The difference between a document and a gate is
that the gate is checked by the machine at the moment it matters.

An unmeasured benchmark counts as not beaten, and a check that crashes counts
as failed. A preflight that passes because it could not run converts "we did
not check" into "we checked and it was fine".

---

## Verification

```bash
make lint && pytest -q              # 607 passing, 3 skipped
pytest -m alpaca -q                 # live paper round trip
python main.py --dry-run --once     # intended orders, nothing submitted
python scripts/preflight.py         # expected: NOT READY, exit 1
cd dashboard && npx tsc --noEmit && npm run build
```
