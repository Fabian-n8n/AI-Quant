# Phase 7 notes: main loop and orchestration

Built 2026-09-07. 374 tests pass (89 new), 2 skipped, 3 live against the Alpaca
paper API.

The spec for this phase is short and the implementation is not, because three of
its lines are load-bearing in ways that are not visible from the line itself.
Those three are sections 2, 4 and 5 below. Everything else is plumbing.

---

## 1. What was built

| File | What it does |
|---|---|
| `main.py` | `TradingEngine` (startup, loop, shutdown), `SessionState`, the CLI |
| `monitoring/logger.py` | `TradingLogger`: JSONL to disk, readable line to console |
| `monitoring/alerts.py` | `AlertManager`: rate-limited console/email/webhook |
| `monitoring/dashboard.py` | `DashboardState` + `TerminalDashboard` |
| `config/__init__.py` | `load_settings`, `load_credentials`, `assert_paper_mode` |
| `tests/test_orchestration.py` | 89 tests, all offline |

Phase 1 shipped the three `monitoring/` modules as stubs that raised
`NotImplementedError`, deferred to Phase 8. The main loop calls all three, so
they had to land here. What is genuinely still Phase 8 is the **Streamlit web
dashboard**; `run_streamlit_app()` remains a stub and says so.

### CLI

```
python main.py                                          # live/paper trading
python main.py --dry-run                                # full pipeline, no orders
python main.py --dry-run --once                         # one bar, then exit
python main.py --mode backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
python main.py --mode backtest --compare --export
python main.py --mode backtest --stress-test --mc-sims 100
python main.py --train-only --symbols SPY
python main.py --dashboard
python main.py --status                                 # build state, no network
```

`--once` is an addition. The spec assumes a long-running process; on a daily-bar
system a cron entry that processes one bar and exits is the more sensible
deployment, and it is the only way to run this without a machine that stays up.

---

## 2. The risk manager vetoes increases, not decreases

**The spec's step 7 routes every signal through `risk_manager.validate_signal()`.
Taken literally, that includes the orders that reduce a position. It must not.**

`validate_signal` can reject. A rejection on the path that *sells* would leave
the system holding an allocation the regime no longer supports, at exactly the
moment it wanted out. Worse, the most likely reason for a rejection there is a
tripped circuit breaker or an exposure limit, so the check would fail hardest
precisely when reducing matters most.

So `process_bar` splits at step 7:

```
target < current  ->  _reduce_to_target()     no veto, proportional scale-down
target > current  ->  _increase_to_target()   full validate_signal cascade
```

Reducing risk never needs permission. Increasing it always does.

The reduction is proportional across every open name rather than name-by-name,
so the relative weights of the book are preserved and no separate decision about
which name to cut is smuggled into the risk layer. Exits go out as **market**
orders, not limits: an unfilled limit that leaves the system over-allocated
through a regime it wanted out of is worse than paying the spread.

Guarded by `test_reduction_does_not_consult_the_risk_manager`, which replaces
`validate_signal` with a function that raises if called.

---

## 3. `needs_rebalance` is what stops the system trading itself to death

Step 6 produces a target allocation every single bar. Without a gate, step 7
submits orders every single bar, and on a 10-symbol universe that is 10 orders
per bar forever, each paying slippage to correct a drift of a fraction of a
percent.

`StrategyOrchestrator.needs_rebalance` (Phase 3) requires the gap between target
and current to exceed `rebalance_threshold`, 10%. A move from 95% to 60% acts; a
move from 95% to 90% does not. The live loop's `outcome.skipped` records which
bars did nothing and why, so a quiet week is distinguishable from a broken one.

---

## 4. "Do NOT close positions on shutdown" is a promise about stops

The spec's shutdown section says: close the WebSockets, do **not** close
positions, because the stops are in place.

That is only true if the stops exist **at the broker as resting orders**. A stop
held as a float on `TrackedPosition.stop_loss` is a number in this process's
memory. The moment the process exits, it protects nothing. Following the spec
literally with in-memory stops means every shutdown leaves the account holding
unhedged positions with no protection at all, and the system would report this
as a clean exit.

`audit_stops()` is the check that makes the promise true. It runs at startup and
again at shutdown, and it checks two things:

1. does every open position have a `stop_loss` recorded, and
2. does a **stop order actually rest at the broker** for it
   (`require_broker_stops: true`)

Anything failing either check is logged as an error, raises a CRITICAL alert,
and is printed in red in the session summary. Shutdown still does not liquidate
— that remains the spec's call and it is the right one — but it can no longer
be silent about a position it is leaving naked.

A related consequence: `_attach_stop` places the resting stop against the
**filled** quantity, and when an entry has not filled yet it records the stop
locally and leaves placement to a later cycle. `audit_stops` is what guarantees
that later cycle is not simply forgotten.

The one path that *does* close positions is a circuit-breaker halt. That
distinction is deliberate: a halt means the system lost more than it is allowed
to, and holding through that is the thing the breaker exists to prevent.

---

## 5. `peak_equity` has to survive a restart

`max_dd_from_peak` is the only breaker that never resets on its own. It compares
current equity against the highest equity the system has ever seen.

Alpaca does not store that number. It knows today's equity and yesterday's
close. The peak is this system's own state, and if it is not persisted, then on
every restart the peak re-bases to whatever equity happens to be right now.

The failure mode is quiet and total:

```
peak 120k, equity 90k   ->  drawdown -25%, PEAK_HALT fires
restart without the snapshot
peak 90k,  equity 90k   ->  drawdown   0%, breaker cannot fire
```

A crash-restart loop would then be a way to disarm the one breaker with no
automatic reset. `state_snapshot.json` carries it, and recovery takes
`max(saved, current)` — never the saved value alone, because a deposit should
raise the peak, and never the current value alone, for the reason above.

The same snapshot carries the **latched breaker states**, so a daily reduce that
tripped at 10am is still latched after a 10:05 restart. Without that, restarting
clears a breaker, which is the same bug wearing a different hat.

It also carries the **stops**, because `PositionTracker.sync()` adopts positions
from the broker with `stop_loss=None` (correctly: it genuinely does not know
where their stops are). Without restoring them, every restart would report its
own positions as unprotected, and, worse, would believe it.

Snapshot writes are atomic (temp file plus `replace`). A snapshot truncated by a
crash mid-write would restore `peak_equity: 0.0`, which is worse than no
snapshot at all.

---

## 6. `--dry-run` removes the capability rather than checking a flag

In dry-run mode `order_executor` is a `_RefusingExecutor`: every attribute
access returns a function that records the attempt and raises.

A dry run that places an order because one code path forgot `if self.dry_run` is
the exact failure worth engineering against, and a flag checked in six places is
five opportunities to miss one. Here there is nothing to miss — the object
cannot submit.

Verified end-to-end: after a full `--dry-run --once` cycle that decided to buy
27 NVDA, the paper account showed no new orders.

---

## 7. Error handling

| Spec rule | Implementation |
|---|---|
| Alpaca API: 3 retries, exponential backoff | `_broker_retry`, 3 attempts, 2s doubling. `AlpacaClient` also retries its own reads, so a composite refresh retries as a unit rather than one call at a time. |
| HMM error: hold current regime | `_classify` catches, logs, returns the previous `RegimeState`. Allocation stays where it is rather than defaulting to a regime nobody computed. |
| Data feed drop: pause signals, keep stops active | `BrokerUnavailable` returns early from `process_bar`: nothing submitted, nothing cancelled, resting stops untouched. |
| Unhandled: log traceback, save state, alert | `process_bar` catches everything, writes the traceback to the JSONL log, alerts CRITICAL, and `max_consecutive_errors` (5) stops the loop with positions and stops left in place. |

One design note on the feed flag. `data_feed_healthy` means "the **last complete
cycle** was healthy", not "the last call succeeded". It is cleared only at the
end of a fully clean bar. Clearing it on the first successful call inside a bar
made the pause-signals branch unreachable — the very bar that had just failed
would go on to place orders. The cost of the stricter reading is one cautious
cycle after any feed problem, which is the right side to err on.

---

## 8. Startup and loop, mapped to the spec

Startup, in the spec's order:

1. `_connect_broker` — `assert_paper_mode` first, then connect, then verify the
   account is tradeable. Three sources must agree that this is paper:
   `settings.yaml`, `ALPACA_PAPER`, and the key prefix (`PK` paper, `AK` live).
2. `_check_market_hours` — logs the next open; the loop does the waiting.
   `market_closed_action: wait | exit` picks between a long-running process and
   a cron job.
3. `_load_or_train_model` — retrains if missing, if older than
   `model_max_age_days` (7, per spec), **or** if `should_retrain()` says so on
   bar count. Both rules apply: calendar alone skips a refit across a long
   holiday break, bars alone lets a model go stale while the system is halted.
4. `_init_risk_manager`
5. `_init_position_tracker` + `sync()`
6. `_restore_session_state` — section 5 above
7. `_start_data_feeds` — section 9 below
8. `_print_system_state` — renders the dashboard, logs "System online", and runs
   the first stop audit

Loop, per bar: fetch bars → features → filtered classify → (stability and
flicker are inside `classify`, by construction) → target allocation → validate
and act → tighten stops → breakers → dashboard → maybe retrain → save state.

Steps 3, 4 and 5 of the spec are one call. `RegimeTracker` inside
`HMMEngine.classify` applies the 3-bar persistence rule and the flicker counter
already. Splitting them at the loop level would create a second place that
decides what the regime is, and two such places eventually disagree.

Step 11's retrain rebuilds the orchestrator's strategy map via
`update_regime_infos`. A refit renumbers the states, so keeping the old
`regime_info` would map new state ids through old volatility ranks and every
allocation would be silently wrong. `previous_regime` is also cleared, because a
regime name from before the refit means something different after it.

---

## 9. There is no daily-bar WebSocket

The spec's loop begins "new bar from WebSocket". Alpaca's `subscribe_bars`
streams **minute** bars regardless of what `timeframe` says. There is no
daily-bar socket to subscribe to.

So two feeds, and only one of them is a clock:

- **Trade updates** — always started. This is how the system learns a stop was
  hit without polling for it.
- **Bar updates** — started only for intraday timeframes, and used purely to
  wake the loop early.

The loop polls on its own schedule regardless. The socket is an optimisation,
never the thing the loop depends on. That is not incidental: "data feed drop:
pause signals, keep stops active" is only implementable if a dropped socket does
not also stop the loop that would notice the drop.

---

## 10. Verified end to end

Against the live Alpaca paper account, `--dry-run --once --symbols NVDA`:

```
BROKER   PAPER, equity $100,000.00, buying power $393,774.06
MARKET   closed, next open 2026-09-08 09:30 ET
MODEL    no fitted model -> trained. BIC picked 7 states from 564 usable rows
REGIME   crash (p=1.00, confirmed), volatility rank HIGH
STRATEGY HighVolDefensiveStrategy, target 60.0% vs current 0.0% -> rebalance
RISK     approved 27 shares ($6,219.72), risk 0.646% of equity
         modification: gap cap, 3x stop gap-through kept under 2% of portfolio
ORDER    dry run: refused. Paper account confirmed unchanged afterwards.
STATE    state_snapshot.json written, 1 bar processed
```

**27 shares and 0.646% risk are the same numbers Phase 6's manual NVDA trade
produced.** The orchestrated path and the hand-built path agree, which is the
cross-check worth having: the loop is not quietly sizing differently from the
code that was validated in Phase 6.

Restarting then confirmed recovery: loaded the saved model rather than
retraining, recovered the session at 1 bar with peak $100,000, and skipped the
bar it had already processed.

---

## 11. Open items

Nothing in Phase 7 changes the Phase 4 and 5 verdict: **the strategy still has
no demonstrated edge.** It loses to buy-and-hold and to random allocation
out-of-sample, and its 48% drawdown trips the peak breaker 5% of the way into
the backtest. Phase 7 makes it run unattended; it does not make it work.

Carried forward, unchanged:

- paper account is still $100,000, not a real starting figure (CLAUDE.md)
- 1.25x leverage is unreachable under an 80% exposure cap (conflict 1)
- the gap rule always binds, so real risk per trade is 0.667%, not 1%
- circuit breakers are day-trading thresholds on a swing system (conflict 2)

New from this phase:

- **Scheduling.** Daily bar close is 4pm New York, which is 4am or 5am in
  Singapore depending on US daylight saving. `--once` under cron is the
  intended deployment; the timezone still has to be decided deliberately, and
  the DST shift means a fixed local time is wrong for half the year.
- **Alpaca history depth.** `training_bars: 954` requested about 4 years and
  returned 1,014 raw NVDA bars, yielding 564 usable feature rows after the
  450-bar warmup. Above the 504 minimum, but not by much. Widening
  `n_candidates` (BIC picked the largest, 7) would need more history than the
  free tier readily returns.
