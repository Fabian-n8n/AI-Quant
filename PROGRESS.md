# Progress

Update when a phase completes, before starting the next one.

| Phase | What | Status |
|---|---|---|
| 1 | Project scaffolding and environment setup | **Done** (2026-09-07) |
| 2 | HMM regime detection engine | **Done** (2026-09-07) |
| 3 | Volatility-based allocation strategies | **Done** (2026-09-07) |
| 4 | Walk-forward backtesting and validation | **Done** (2026-09-07) |
| 5 | Risk management layer | **Done** (2026-09-07) |
| 6 | Alpaca broker integration | **Done** (2026-09-07) |
| 7 | Main loop and orchestration | **Done** (2026-09-07) |
| 8 | Monitoring, alerts and dashboard UI | **Next** |

Tests: 374 passing, 2 skipped (3 hit the live Alpaca paper API). Skips are phase-gated placeholders naming what
each later phase has to prove.

---

## Validation checklist

None of this is meaningful until the system has run. Repeated from CLAUDE.md so
there is one place to look.

- [ ] 30 to 50 closed trades minimum
- [ ] Walk-forward tested, never in-sample
- [ ] Beats buy-and-hold on total return **and** Sharpe, net of slippage
- [ ] Beats 200-day SMA trend following
- [ ] Beats random allocation under identical risk rules
- [ ] Expectancy per trade is positive
- [ ] High-confidence trades outperform low-confidence trades
- [ ] Returns differ meaningfully by regime, or the HMM is not earning its keep
- [ ] Variants tested are logged in `docs/EXPERIMENT-LOG.md`
- [ ] Paper traded for at least a month with every decision reviewed

---

## Log

### 2026-09-07 - Phase 1 complete

Scaffolding built to the tutorial's structure. 16 modules, all importing
cleanly. Every function is a stub raising `NotImplementedError` naming the phase
that fills it in.

**Environment verified.** Python 3.14.3 is the only interpreter on this machine
and new enough that missing wheels were a genuine risk. Everything installs:
hmmlearn 0.3.3 (built from source, no wheel needed), scikit-learn 1.9.0, scipy
1.18.1, pandas 3.0.5, numpy 2.5.3, alpaca-py 0.44.0, ta 0.11.0, rich 15.0.0,
pyyaml 6.0.3, schedule 1.2.2, websocket-client 1.9.2.

**One deliberate deviation from the tutorial's dependency list.**
`alpaca-trade-api` is excluded. It is Alpaca's deprecated SDK and installing it
alongside `alpaca-py` downgrades `websockets` 17.1 to 10.4 and `urllib3` 2.7.0
to 1.26.20. `alpaca-py` covers trading, market data and streaming on its own.
Recorded in `docs/TUTORIAL-CONFLICTS.md`.

**Two small additions to the tutorial's tree**, both flagged in place:
`config/__init__.py` for the settings and credential loaders, since the tutorial
shows `config/` holding only YAML and something has to read it; and `docs/` for
decision records, which nothing depends on.

**Four settings conflict with CLAUDE.md's hard constraints.** The tutorial's
values ship as specified. Each is annotated in `settings.yaml` at the line it
appears, with the full reasoning in `docs/TUTORIAL-CONFLICTS.md`: leverage 1.25x
against a stated "no leverage" constraint; day-trading circuit breakers on a
swing horizon; and `initial_capital: 100000` against "use the real number".
None of it matters on paper. All of it matters before funding.

**Phase 1 tests are real, not placeholders.** They check that settings.yaml is
internally consistent rather than merely parseable: breakers escalate before
they halt, daily thresholds are tighter than weekly, five positions at the
single-position cap fit inside the exposure ceiling, and the strategy's leverage
does not exceed the risk layer's cap. `test_orders.py` also scans every source
file for anything shaped like a live Alpaca key, which is the check that catches
a key pasted in during debugging and forgotten.

Superseded: an earlier draft of CLAUDE.md reordered the phases to put the
backtester before the strategy. Dropped in favour of the tutorial's order, on
the reasoning that a working reference implementation beats a better-sequenced
one that stalls. The tradeoff is recorded in CLAUDE.md.

### 2026-09-07 - Phase 2 complete

`core/hmm_engine.py` (1,250 lines) and `data/feature_engineering.py` implemented.
Tests went from 35 to 99 passing.

All 14 spec'd features built as pure functions, rolling-z-scored over 252 bars.
Gaussian HMM with BIC selection across 3-7 states, 10 manual restarts per
candidate, labels assigned by sorted mean return, forward-algorithm filtering,
stability filter, pickle persistence with metadata.

**Three things found while building that the spec did not account for.**

**1. The model as specified cannot fit at the data size it specifies.** With all
14 features and `covariance_type: full`, candidates 5/6/7 need 619/749/881
parameters against 504 training rows. Singular covariances, and BIC's penalty
(6.22 per parameter at 504 rows) would select 3 states every time regardless of
the data, so automatic model selection would keep running while having stopped
selecting. The tutorial's five-regime dashboard could never appear.
`check_fittability()` now raises with the fix spelled out, and
`hmm.feature_columns` defaults to a 6-column volatility subset (237 params at
n=7). All 14 are still computed; only the HMM's input set is narrowed, which
follows from the spec's own "this is a volatility classifier" framing. With
2,000+ rows the full set fits and the guard stays quiet: the constraint is the
ratio, not the feature count.

**2. `min_train_bars: 504` does not mean 504 bars of data.** The 200-bar SMA
plus the 252-bar z-score discard 450 rows, so 504 usable feature rows need 954
raw bars, roughly 3.8 years. `required_raw_bars()` computes it. Left unnoticed
this would surface in Phase 4 as walk-forward windows silently training on far
less data than configured.

**3. The spec's mandatory look-ahead test has an off-by-one.**
`data[0:400][-1]` is row 399; `data[0:500][400]` is row 400. Different bars.
Because regimes persist for weeks, adjacent bars usually share one, so the
assertion passes almost always for the wrong reason and would keep passing
against a leaking implementation. It was observed passing by coincidence during
development. The corrected test compares row 399 on both sides, and a stronger
version compares all 400 prefix rows on continuous probabilities.

**Look-ahead defences, since this is the failure mode that ruins the project:**
the forward algorithm is implemented explicitly in log space rather than
borrowed; `model.predict` and `predict_proba` are monkeypatched into landmines
during tests; an AST scan rejects any call to the Viterbi family; the forward
pass is cross-checked against a longhand reference implementation; and the
incremental live stepper is asserted numerically identical to the batch path at
1e-12, so live and backtest cannot disagree about the regime.

Verified working: BIC selection, deterministic refits, labels stable across
refits, regime persistence around 25 bars (not noise), stability filter
suppressing changes, save/load round trip, and the end-to-end check that
constructed high-volatility blocks are classified into higher-volatility states
than calm ones.

Not yet validated against real market data. `data/market_data.py` is still a
Phase 6 stub, so the first real fit waits on Alpaca credentials. Expect the
state count and regime statistics to shift, and expect to revisit
`feature_columns` when they do.

Full detail in `docs/PHASE2-NOTES.md`.

### 2026-09-07 - Phase 3 complete

`core/regime_strategies.py` (680 lines) implemented, `core/signal_generator.py`
rewritten to match. Tests 99 -> 144.

Three strategy tiers mapped by measured volatility, `StrategyOrchestrator`,
`Signal` dataclass, uncertainty mode, rebalance threshold, backward-compatible
aliases, `LABEL_TO_STRATEGY`, and the `RegimeStrategies` compatibility wrapper.

**The spec's stop formulas would have shipped broken.** Every stop is anchored to
the 50 EMA, and price sits below the EMA for most of any selloff, so the formulas
return a stop **above** entry on 25-36% of bars. On a long that is an instant
stop-out, and Phase 5 sizes as `risk / abs(entry - stop)`, so a stop equal to
entry is a division by zero. Stops are now clamped to at least 0.5 ATR below
entry with a 0.5% price floor for when ATR collapses. On the fixture the clamp
fires on 38-47% of bars and the majority of those were the fatal case. The raw
value is kept in `metadata["raw_stop"]` so the clamp is auditable.

**Labels and volatility disagree completely.** On the current fit, all 7 of 7
regimes sit in a different position under the return sort than the volatility
sort. `strong_bull` is the single most volatile regime (52.7% annualised) and is
correctly allocated 60% by the defensive tier; a label-driven mapping would have
levered into it at 95% and 1.25x. This is the spec's central architectural point
and it is now demonstrable rather than theoretical.

**Volatility rank was defined twice and the two disagreed.** Phase 2 used a
tercile split, Phase 3 specifies `rank / (n - 1)` against 0.33 / 0.67 boundaries.
They differ at n=5. Unified on the Phase 3 formula in `assign_volatility_ranks`,
used by both, with the exact partition per state count pinned by test since the
literal boundaries are not even thirds.

**`Signal` was about to be defined twice.** The Phase 1 skeleton declared its own
in `signal_generator.py` and `risk_manager.py` imported from there. Phase 5 would
have been type-checking against a class the strategy layer never produces. Now
single-defined in `regime_strategies.py` and re-exported.

**Open decision before Phase 4:** the strategy allocations exceed the risk
layer's ceiling. LowVol is 95% x 1.25 = 118.75% gross against
`risk.max_exposure: 80%`, and MidVol-with-trend is 95%. Phase 5 has absolute veto
so it will clamp both, meaning the strategy never runs at its stated allocation.
Compounding it, the universe is 10 symbols but `max_concurrent` is 5, so 95%
spread across 10 realises 47.5%. Settle this before Phase 4 measures anything,
or the backtest describes a different strategy than the documentation.
See `docs/PHASE3-NOTES.md` section 7 and `docs/TUTORIAL-CONFLICTS.md`.

Nothing in this phase is backtested. The backtester is Phase 4, so every
allocation number, the trend filter, the stop multiples and the rebalance
threshold are currently assertions rather than results.

### 2026-09-07 - Phase 4 complete

`backtest/backtester.py`, `performance.py` and `stress_test.py` implemented,
plus the `backtest` CLI and a synthetic data fallback. Tests 144 -> 184.

**First out-of-sample result: no demonstrated edge.** On synthetic data, 15
folds, 1,890 OOS bars, 168 rebalances:

| | Return | Sharpe | Max DD |
|---|---|---|---|
| regime-trader | +54.97% | 0.18 | -48.38% |
| Buy and hold | +118.31% | 0.34 | -46.61% |
| 200 SMA | -0.15% | -0.05 | -42.80% |
| Random (100 seeds) | +84.67% ± 72.12% | 0.24 | -40.73% |

It loses to buy-and-hold and to random allocation. The damaging detail is the
drawdown: 48.4% against buy-and-hold's 46.6% while averaging only 74%
allocation. The strategy's stated premise is that the edge comes from avoiding
big drawdowns through vol-based sizing, and it took less exposure and lost more.
Alpha -2.45%, beta 0.74. Logged as variant 1.

This is synthetic data with arbitrary drift, so it is not evidence about SPY.
It is evidence that the backtester, the benchmarks and the verdict logic work,
and that the framework will not rubber-stamp a strategy that fails.

**Three things found while building.**

**1. The 252-bar IS window is below `min_train_bars: 504`,** so every fold would
have raised. The backtester lowers the floor to the window size and logs it. A
252-bar fit is thin: BIC's penalty spread (492 at n=3 versus 1,310 at n=7) biases
selection toward fewer states, and across 15 folds it chose 4, 5 or 6, never 7.

**2. The naive loop was O(T^2) and timed out at two minutes.** Re-running the
forward pass and recomputing EMA/ATR over a growing slice per bar. Added
`HMMEngine.stream()` (cached forward alpha, one step per bar, warmed on the
training tail) and `StrategyOrchestrator.target_allocation()` (allocation
without building a Signal, since the backtester holds no stops by design).
11.5 seconds for 15 folds after.

**3. A flat equity curve produced a Sharpe of -1.04e17.** `std()` of a constant
series is ~1e-20, not zero, so the `sd > 0` guard passed and the division
exploded. Flat equity is normal here (fully in cash, or halted by Phase 5's
breakers), so this fires in ordinary use, and the value looks like a triumph
rather than an error. Guarded with a 1e-12 tolerance across Sharpe, Sortino and
Calmar, with a regression test and a sweep for non-finite values anywhere in the
report.

**Carried forward to Phase 5:** `stress_test.evaluate_breakers()` applies the
configured drawdown thresholds directly, because `RiskManager` does not exist
yet. Phase 5 must replace it with the real risk manager rather than leaving two
implementations of the same thresholds.

**Still unresolved from Phase 3, and now with a cost:** the backtest measures
the unclamped allocations, including 118.75% gross exposure. Phase 5 caps that
at 80%, so until the conflict is settled every backtest describes a
configuration that cannot actually be traded.

**Stress testing produced a fourth finding.** The circuit breakers fire 100% of
the time in every scenario, including the mildest, because the baseline max
drawdown (-44%) already exceeds `max_dd_from_peak: 0.10` with no shock applied.
A breaker with a 100% fire rate carries no information and, in live use, trains
you to delete the lock file without reading it. This is a quantified
confirmation of the argument in CLAUDE.md, and Phase 5 should retune against it.

Two other stress results: the regime misclassification test **passes** (worst
drawdown 1.0x baseline, so damage stays bounded when regimes are wrong, which is
what risk-layer independence requires), and overnight gaps are the dominant
risk: a 5x ATR gap sequence leaves only **33% of runs surviving** while every
crash scenario including -15% x10 survives 100%.

Full detail in `docs/PHASE4-NOTES.md`.

### 2026-09-07 - Phase 5 complete

`core/risk_manager.py` (970 lines): `PortfolioState`, `CircuitBreaker`,
`RiskDecision`, `RiskManager` with the 13-step veto cascade. Tests 184 -> 242,
69 of them in `tests/test_risk.py`.

**The finding that reframes Phase 4.** I previously concluded the breakers fired
too often and needed retuning. Running the real backtest equity curve through the
actual breakers is sharper: the strategy has a 48.4% peak drawdown, so the
configured 10% peak halt trips at **bar 97 of 1,890** and never resumes. The
looser swing-horizon alternative (18%) trips at bar 279. Both halt permanently.

Retuning the breakers is treating the symptom. No drawdown-based halt at any sane
level survives a 48% drawdown strategy. Combined with Phase 4's result that it
loses to buy-and-hold and random allocation, both point at the allocation layer,
starting with the 1.25x leverage. Fix the strategy until its drawdown is
survivable, then calibrate breakers against it.

**Three contradictions in the configured limits, all pinned by test.**

1. **1.25x leverage is unreachable.** Leverage IS gross exposure, so 1.25x means
   125% gross, and `max_exposure` caps it at 80%. The exposure gate binds first,
   so nothing exceeds 0.80x and the low-vol leverage rule is dead code.
2. **The gap rule always binds.** Overnight sizing allows 2%/3 = 0.667% of equity
   at risk, below `max_risk_per_trade` (1%). Every position in a swing system is
   overnight, so the 1% figure never applies to anything. Real risk per trade is
   0.667%.
3. **Breaker thresholds are day-trading numbers**, shipped as specified.

**Phase 4 loop closed.** `stress_test.evaluate_breakers` now delegates to the
real `CircuitBreaker` instead of duplicating the thresholds, and walks the curve
bar by bar rather than testing the endpoint, since a breaker that fired in month
two and was recovered from still fired.

**Not yet wired.** The risk manager validates signals but nothing calls it yet.
The backtester still runs unclamped Phase 3 allocations. Wiring is Phase 7's job,
and the useful order is: settle the leverage and exposure decisions, re-run the
backtest, then wire.

Full detail in `docs/PHASE5-NOTES.md`.

### 2026-09-07 - Phase 6 complete

`broker/alpaca_client.py`, `order_executor.py`, `position_tracker.py` and a real
`data/market_data.py`. Tests 242 -> 285, including 3 that hit the live paper API.

**Test trade placed end to end.** NVDA buy 27 shares, limit $230.59, order
`5620725c`, resting until Monday's open. It ran the whole pipeline: broker ->
1,029 real bars -> HMM (7 states, current regime strong_bear, volatility rank
high) -> HighVolDefensiveStrategy at 60% unlevered -> risk manager shrinking it
to 27 shares -> order at the broker.

The resulting risk was **0.646% of equity**, which is the Phase 5 gap cap
binding exactly as predicted (0.667%, slightly under because shares truncate).
That arithmetic is now confirmed against a real order rather than a unit test.

**One real bug, found the hard way.** The first data fetch failed with
"subscription does not permit querying recent SIP data". The obvious reading is
a free-tier limit, and partly it is, but moving `end` back 16 minutes did not fix
it. The actual cause: **Alpaca reads a naive datetime as UTC**, and this machine
runs UTC+8, so `datetime.now()` arrived eight hours in the future.

That matters more than the error. On the free tier it fails loudly, which is
lucky. On a paid plan the same code would not error at all, it would just return
a window shifted by eight hours, and every backtest `start` date would have been
silently wrong too. Fixed with timezone-aware helpers throughout. **Phase 7 must
carry this forward:** the daily-bar-close schedule is 4pm New York, which is 4am
or 5am Singapore depending on daylight saving.

**Adjusted vs raw confirmed on real data.** NVDA split 10-for-1 in June 2024:
pre-split close is $120.79 adjusted and $1,209.98 raw, and the raw series
contains an 89.9% single-bar "crash" that is really the split. `validate()`
flags it. Signals use adjusted, orders and stops use raw.

**Free tier quirk handled.** Outside market hours the IEX feed returns a zero
ask. `reference_price()` falls back to bid then last close; without it every
weekend limit order would be priced off zero.

Full detail in `docs/PHASE6-NOTES.md`.

### 2026-09-07 - Phase 7 complete

Main loop and orchestration. `main.py` now holds `TradingEngine`: the eight
startup steps, the eleven-step bar loop, and a shutdown that deliberately leaves
positions open. 374 tests pass (89 new, all offline), 2 skipped.

The three `monitoring/` modules moved from Phase 1 stubs to working code,
because the loop calls all of them. What is still Phase 8 is the Streamlit web
dashboard; the terminal one works now.

**Three parts of the spec needed more than a literal reading.**

*The risk manager vetoes increases, not decreases.* Step 7 routes every signal
through `validate_signal()`. Applied to the orders that reduce a position, a
rejection would trap the system in an allocation the regime no longer supports,
at the exact moment it wanted out, and the likeliest cause of that rejection is
a tripped breaker. Reductions bypass the veto and scale every position down
proportionally. Reducing risk never needs permission.

*"Do NOT close positions on shutdown" is a promise about stops.* It holds only
if the stops are resting orders at the broker. A `stop_loss` float in memory
protects nothing once the process exits, so following the spec literally with
in-memory stops leaves the account unprotected on every clean shutdown.
`audit_stops()` checks both the local record and the broker's open orders, at
startup and at shutdown, and raises a CRITICAL alert on anything naked.

*`peak_equity` has to survive a restart.* Alpaca does not store it, and
`max_dd_from_peak` is the only breaker that never resets on its own. Without
persistence, every restart re-bases the peak to the already-drawn-down equity
and the breaker can never fire again, making a crash-restart loop a way to
disarm it. `state_snapshot.json` carries the peak, the latched breaker states
and the stops; recovery takes `max(saved, current)`.

**Also worth recording.**

- `--dry-run` swaps in a `_RefusingExecutor` whose every method raises. The
  capability is removed rather than gated on a flag checked in six places.
- There is no daily-bar WebSocket at Alpaca; `subscribe_bars` streams minute
  bars whatever `timeframe` says. The loop polls on its own schedule and treats
  the socket as an optimisation, which is also what makes "pause signals, keep
  stops active" implementable.
- `data_feed_healthy` means "the last complete cycle was healthy". Clearing it
  on the first successful call inside a bar made the pause branch unreachable:
  the bar that had just failed would go on to place orders.
- Spec steps 3, 4 and 5 are one call. `RegimeTracker` inside `classify` already
  applies the 3-bar rule and the flicker counter, and two places deciding what
  the regime is would eventually disagree.
- Retraining rebuilds the orchestrator's strategy map. A refit renumbers states,
  so the old `regime_info` would map new ids through old volatility ranks and
  every allocation would be silently wrong.

**Verified against the live paper account.** `--dry-run --once --symbols NVDA`
ran the whole pipeline: trained a 7-state model, classified `crash` at p=1.00
with volatility rank HIGH, took `HighVolDefensiveStrategy` at a 60% target, and
the risk manager approved 27 shares at 0.646% of equity under the gap cap.
Those are the same numbers Phase 6's hand-built NVDA trade produced, so the
orchestrated path and the validated path agree. The paper account was confirmed
unchanged afterwards. A restart then loaded the saved model, recovered the
session, and skipped the bar it had already processed.

**New open items.** Daily bar close is 4pm New York, which is 4am or 5am in
Singapore depending on US daylight saving, so a fixed local cron time is wrong
for half the year. And `training_bars: 954` returned 1,014 raw NVDA bars for 564
usable feature rows: above the 504 minimum but not by much, and BIC selected the
largest candidate (7 states), which suggests the range wants widening and that
needs more history than the free tier readily returns.

Nothing here changes the Phase 4 and 5 verdict. The strategy still loses to
buy-and-hold and to random allocation out-of-sample. Phase 7 makes it run
unattended; it does not make it work.

Full detail in `docs/PHASE7-NOTES.md`.
