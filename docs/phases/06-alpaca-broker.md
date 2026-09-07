# Phase 6 notes: Alpaca broker integration

Live connection, real data, and one bug that would have been silently wrong on a
paid plan.

---

## THE TEST TRADE

Placed through the full pipeline, not a raw API call. Every layer ran:

```
1. BROKER   paper=True  equity=$100,000.00  buying_power=$400,000.00
2. DATA     1029 real bars -> 2026-09-04, last close $230.36, validation clean
3. REGIME   7 states selected. Current: strong_bear (p=1.00) -> volatility rank HIGH
4. SIGNAL   HighVolDefensiveStrategy -> LONG, 60% unlevered
            entry $230.36, stop $206.43 (10.39% away)
5. RISK     approved, shrunk to 27 shares = $6,219.72 (6.22% of equity)
            risk $646.09 = 0.646% of equity
            modification: gap cap, 3x stop gap-through kept under 2% of portfolio
6. ORDER    NVDA buy x27 limit $230.59
            order  5620725c-a8a5-4e4f-8c70-75dc9b7b66a1
            trade  0ce0c86e-0e74-42c5-bca0-788a5313a5f0
7. CONFIRM  status open, resting until Monday's open
```

Two things worth reading off that.

**The risk layer did real work.** It shrank 60% of portfolio down to 6.22%, and
the resulting risk of **0.646% of equity** is the gap cap from Phase 5 binding
exactly as predicted (0.667%, slightly under because shares truncate). The
Phase 5 arithmetic is confirmed against a real order, not just a unit test.

**The strategy behaved defensively and was right to.** NVDA's current regime
classified as high volatility, so the defensive tier took 60% unlevered rather
than the low-vol tier's 95% at 1.25x. That is the volatility mapping working on
live data.

---

## 1. Naive datetimes against Alpaca: a real bug, found the hard way

The first historical fetch failed with:

```
{"message":"subscription does not permit querying recent SIP data"}
```

The obvious reading is a free-tier limitation, and it partly is: the free plan
rejects any request whose `end` touches the last 15 minutes. But moving `end`
back 16 minutes did not fix it.

The actual cause: **Alpaca interprets a naive datetime as UTC.** This machine
runs UTC+8 (Singapore), so `datetime.now()` returns 13:12 local while real UTC
is 05:12. Sent naive, that arrives at the API as 13:12 UTC, eight hours in the
future. `now - 16 minutes` was still hours inside the forbidden window.

```
local now (naive): 2026-09-07 13:12:38
utc now (aware)  : 2026-09-07 05:12:38+00:00

naive now-16min   FAIL  subscription does not permit querying recent SIP data
aware now-16min   OK    1029 bars
```

**Why this matters beyond the error.** On the free tier it fails loudly, which
is lucky. On a paid plan the same code would not error at all: it would return a
window shifted eight hours and nobody would notice. And the same class of bug
applies to `start`, where every backtest window would have been silently off.

Fixed with `utc_now()` and `_as_utc()` in `data/market_data.py`, used
everywhere. `SIP_DELAY` handles the free-tier window on top.

Chose the delay over hardcoding `feed="iex"`, which also works but pins every
request to the thin IEX feed even after a paid upgrade.

**Carry this into Phase 7.** The orchestrator schedules on daily bar close,
which is 4pm New York, 4am or 5am Singapore depending on daylight saving. Every
timestamp in that scheduling must be timezone-aware for the same reason.

---

## 2. Adjusted versus raw prices, demonstrated on real data

Phase 1 flagged this as a silent-failure risk. Here it is on NVDA, which split
10-for-1 in June 2024:

| | pre-split close | max single-bar move |
|---|---|---|
| adjusted | $120.79 | 24.3% |
| raw | $1,209.98 | **89.9%** |

The raw series contains an 89.9% single-bar "crash" that is really a corporate
action. Feeding that to the HMM would teach it a crash regime that does not
exist.

`validate()` catches it: on the raw series it reports *"1 single-bar moves above
35%, possibly an unadjusted split"*, and on the adjusted series it reports
clean. The rule stands: **signals on adjusted prices, orders and stops on raw**.

---

## 3. Free tier: the zero ask

Outside market hours the IEX feed returns `bid $219.49, ask $0.00`. A zero is
"no quote", not a price.

Two places handle it:

- `reference_price()` falls back to the bid, then the last daily close. Without
  it, every limit order placed on a weekend would be priced off zero.
- The risk manager's spread check already skipped zero-sided quotes (`if bid > 0
  and ask > 0`), so it was correct by accident. It is now correct on purpose.

---

## Design decisions

**Live trading is deliberately awkward.** It needs `ALPACA_PAPER=false` *and* a
typed confirmation at an interactive prompt. A cron job or CI run has no stdin,
so it cannot reach the live endpoint by accident. That asymmetry is the point.

**Key prefixes are cross-checked.** Alpaca issues paper keys as `PK...` and live
keys as `AK...`. A mismatch between the key and the requested mode is refused
rather than guessed at.

**The paper flag is read from the account, not the request.** Alpaca prefixes
paper account numbers with `PA`, so `health_check()` compares what we asked for
against what the broker reports and refuses a mismatch.

**Reads retry with exponential backoff. Order submission never does.** A network
timeout does not tell you whether the order reached the exchange, and a blind
retry is how you end up holding two copies of the same position.

**Limit orders by default**, 0.1% through the touch, cancelled after 30s.
Retrying at market is opt-in per call: a limit that will not fill is usually
telling you something about liquidity, and converting it to market discards that
information at the worst moment.

**`modify_stop` tightens only.** A stop that can move away from price is not a
stop.

**`close_all_positions` cancels resting orders first.** Closing a position while
its stop is still live would leave the stop as a naked short.

**Reconciliation adopts, and adopting is loud.** A broker position we were not
tracking gets adopted with `adopted=True` and no stop, so
`positions_without_stops()` surfaces it immediately. An unprotected position
should be noisy.

**Averaging up recomputes the weighted entry.** Keeping the original entry would
make every later P&L figure and stop distance wrong.

**Every position mutation takes a lock.** Fills arrive on the WebSocket thread
while the main loop reads positions. `test_concurrent_fills_are_thread_safe`
runs six threads against it.

**`to_portfolio_state()` is the only bridge to the risk layer**, and positions
cross as plain dicts so the risk manager never imports a broker type. That is
what keeps it broker-agnostic and the eventual IBKR port contained.

---

## Testing

53 tests in `tests/test_orders.py`, split three ways:

- **Offline** (47): fakes stand in for Alpaca, so logic is tested without a
  network call. These run in CI.
- **Live** (3, marked `alpaca`): hit the real paper API. `pytest -m alpaca`.
  They assert the account is paper before doing anything.
- **Secret handling** (3): unconditional. `.gitignore` coverage, a `git
  check-ignore` on `.env`, and a regex sweep of every source file for anything
  shaped like an Alpaca key. A leaked key is not a bug you fix by reverting.

`conftest.py` now loads `.env`, because without it the live tests skipped
silently, and a silent skip looks identical to a pass.

---

## Still open

**Paper account is $100,000.** CLAUDE.md says to set it to the real intended
starting capital, because commission drag and minimum viable position size
behave completely differently at $5k. Still unset.

**The strategy has no demonstrated edge.** Phase 4 showed it losing to
buy-and-hold and to random allocation, and Phase 5 showed its 48% drawdown trips
any sane circuit breaker within 5% of the backtest. Phase 6 connects a broker to
a strategy that has not earned one yet. That is fine for paper, and it is the
whole reason to paper trade, but nothing here changes the verdict.

**Nothing runs on a schedule yet.** The orchestrator is Phase 7.
