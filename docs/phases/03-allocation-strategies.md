# Phase 3 notes: volatility-based allocation

What was built, and the one thing in the spec that would have shipped broken.

---

## 1. The stop formulas produce invalid stops on a third of all bars

This is the most important finding in this phase.

Every stop in the spec is anchored to the 50 EMA:

| Strategy | Stop formula |
|---|---|
| LowVolBull | `max(price - 3*ATR, EMA50 - 0.5*ATR)` |
| MidVolCautious | `EMA50 - 0.5*ATR` |
| HighVolDefensive | `EMA50 - 1.0*ATR` |

Price sits **below** the 50 EMA for most of any selloff. When it does, these
formulas return a stop **above** the entry price. Measured on the synthetic
fixture, sweeping every bar:

| Strategy | Raw stop at or above entry |
|---|---|
| HighVolDefensive | 25.1% of bars (29.8% during turbulent blocks, when it runs) |
| MidVolCautious | 36.0% of bars |
| LowVolBull | 36.0% of bars |

For a long position that is an instant stop-out. And it is worse than a bad
trade, because Phase 5 sizes positions as:

```
position_size = (portfolio * 0.01) / abs(entry - stop_loss)
```

A stop **equal** to entry is a division by zero. A stop **above** entry passes
through `abs()` and silently sizes off a distance that means nothing.

**The fix.** Every stop is clamped to sit at least `min_stop_atr_mult` (0.5) ATR
below entry, with a secondary floor of `min_stop_pct` (0.5%) of price for when
ATR collapses in a very quiet stretch. Without that second floor a flat market
produces a stop a fraction of a cent below entry and a position size that
breaches every limit in Phase 5.

The spec's formula is preserved and recorded in `metadata["raw_stop"]` next to
`metadata["stop_clamped"]`, so the clamp is auditable rather than invisible.

On the fixture the clamp fires on 38-47% of bars, and of those, the majority
were the fatal case rather than merely a tight stop.

`test_clamp_actually_fires_and_rescues_fatal_stops` fails if the raw formulas
ever stop producing bad stops, which would mean the clamp's justification needs
re-examining rather than being assumed.

---

## 2. Labels and volatility genuinely disagree

The spec is emphatic that the orchestrator maps by `expected_volatility` and
ignores labels. It is worth seeing how far apart the two orderings actually are.
From the current fit on the synthetic fixture:

```
labels sorted by return:      crash < strong_bear < weak_bear < neutral < weak_bull < strong_bull < euphoria
the same regimes by volatility: weak_bull < euphoria < neutral < crash < strong_bear < weak_bear < strong_bull
```

**All 7 of 7 regimes sit in a different position under the two sorts.**

Concretely, `strong_bull` has the highest volatility of any regime (52.7%
annualised) and is correctly allocated 60% by `HighVolDefensiveStrategy`. A
label-driven mapping would have levered into it at 95% and 1.25x.

`LABEL_TO_STRATEGY` exists because the spec asks for it, but it is a fallback
for callers holding a bare label with no `RegimeInfo` to measure. The
orchestrator never consults it. `test_label_to_strategy_is_a_fallback_not_the_path`
asserts the two disagree, so the distinction cannot quietly collapse.

---

## 3. Volatility rank is now defined in exactly one place

Phase 2 assigned `RegimeInfo.volatility_rank` with a tercile split. Phase 3
specifies a different formula:

```
position = rank / (n_regimes - 1)
position <= 0.33 -> low,  position >= 0.67 -> high,  else mid
```

These disagree at n=5 (tercile gives low 1 / mid 3 / high 1; the spec's formula
gives 2 / 1 / 2). Left alone, the dashboard reading `RegimeInfo.volatility_rank`
and the allocator computing its own would have shown different answers for the
same regime.

`assign_volatility_ranks()` in `core/hmm_engine.py` is now the single definition,
used by both. The exact partition per state count is pinned by test, because the
literal 0.33 / 0.67 boundaries are **not** even thirds:

| states | low | mid | high |
|---|---|---|---|
| 3 | 1 | 1 | 1 |
| 4 | 1 | 2 | 1 |
| 5 | 2 | 1 | 2 |
| 6 | 2 | 2 | 2 |
| 7 | 2 | 3 | 2 |

At n=7, rank 2 gives 2/6 = 0.3333, which is above 0.33, so it lands in MID
rather than LOW. That is a floating-point boundary decision, and it is pinned by
test rather than left to be discovered mid-backtest.

---

## 4. `Signal` is defined once

The Phase 1 skeleton declared its own `Signal` and `AllocationTarget` in
`core/signal_generator.py`, and `core/risk_manager.py` imported from there.
Phase 3 defines `Signal` in `core/regime_strategies.py`.

Two dataclasses with the same name would have left Phase 5 type-checking against
a class the strategy layer never produces. `signal_generator.py` now re-exports
the single definition, and `risk_manager.py` imports from the same place.

---

## 5. Never short, structurally

`Direction` has `LONG` and `FLAT` and no `SHORT` member. Adding shorting is
therefore a code change to an enum, not a parameter someone can pass.

The spec's reasoning is worth keeping: shorting was tested in walk-forward
backtesting and destroyed returns, because markets drift upward, V-shaped
recoveries are fast, and the HMM is 2-3 days late detecting them. A short held
through the rebound gives back everything the crash earned.

---

## 6. Three uncertainty triggers, one response

Uncertainty mode fires when **any** of these holds:

- `probability < min_confidence` (0.55): unsure which regime
- `is_flickering`: the model keeps changing its mind
- `not is_confirmed`: a regime change has not yet held `stability_bars`

Response: halve `position_size_pct`, **force** leverage to 1.0x, append
`[UNCERTAINTY — size halved]` to the reasoning.

Leverage is forced rather than scaled. Leverage applied to a regime call the
model itself flags as unreliable is the specific mechanism that turns a bad week
into an unrecoverable one.

`metadata` records `pre_uncertainty_size` and `pre_uncertainty_leverage`, so a
reviewer can see what was intended before the cut.

---

## 7. Allocation conflicts with the risk limits, and Phase 5 will win

Not a bug, but it changes what this strategy actually does and should be settled
before Phase 4 measures anything.

| | value |
|---|---|
| LowVol gross exposure | 95% x 1.25 = **118.75%** |
| MidVol (trend intact) | 95% x 1.0 = **95%** |
| HighVol | 60% x 1.0 = 60% |
| `risk.max_exposure` | **80%** |

Two of the three tiers exceed the configured exposure ceiling. The risk manager
has absolute veto, so in Phase 5 it will clamp them to 80% and
`LowVolBullStrategy`'s headline 95%-at-1.25x will never happen as written.

A second interaction compounds it. The universe is 10 symbols but
`risk.max_concurrent` is 5. At 95% spread across 10 names each gets 9.5%, which
passes `max_single_position` (15%), but only 5 positions may be held, so realised
exposure is **47.5%**, not 95%.

So the strategy as specified would run at roughly half its stated allocation once
the risk layer is wired in. Three ways to resolve, and this is a decision rather
than a fix:

1. Raise `max_exposure` and accept the leverage (contradicts CLAUDE.md).
2. Lower the strategy allocations to fit inside 80%.
3. Raise `max_concurrent` toward the universe size so the spread is real.

`metadata["gross_exposure"]` is on every signal so the clamp is visible in Phase
5's logs rather than showing up as an unexplained shortfall in Phase 4.

---

## 8. `position_size_pct` is portfolio-level, not per-symbol

The spec's range (0.60 to 0.95) matches the allocation values exactly, so
`position_size_pct` is what fraction of the **portfolio** the regime calls for,
not what fraction one symbol should take.

`metadata["per_symbol_weight"]` carries `position_size_pct / n_signals` so
nothing downstream has to infer which of the two it is holding. Getting this
backwards would size every symbol at the full portfolio allocation, which at 10
symbols is 9.5x over-leveraged.

After uncertainty halving the range becomes 0.30 to 0.475, so the stated 0.60
floor applies before the cut, not after.

---

## 9. Not yet validated

Everything here is unit-tested. **None of it is backtested**, because the
backtester is Phase 4.

That ordering comes from the tutorial and it means the allocation numbers, the
trend filter, the stop multiples and the rebalance threshold are all currently
assertions rather than results. Nothing in this phase should be treated as tuned
until Phase 4 has run it out-of-sample against buy-and-hold, 200-day SMA, and
random allocation.
