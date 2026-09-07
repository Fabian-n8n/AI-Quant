# Experiment log

One line per distinct variant put through the backtester. No exceptions.

Walk-forward testing protects against tuning on the test set. It does not
protect against testing thirty variants and shipping the one that got lucky. If
variant twenty-three is the winner, the honest read is not "we found an edge",
it is "we bought twenty-three lottery tickets and one paid out".

This log exists so that when a good result appears, you know how many attempts
it took. That number changes what the result means.

| # | Date | What changed | OOS trades | Expectancy | Sharpe | vs B&H | vs SMA | vs Random | Kept? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-07 | Baseline: Phase 3 defaults, 95/95/60/60, 1.25x low-vol leverage. **Synthetic data** | 168 | +$327 | 0.18 | **LOSE** (+55% vs +118%) | win (+55% vs -0.2%) | **LOSE** (+55% vs +85%) | No |

**Variant 1 read:** loses to buy-and-hold and to random allocation on both
return and Sharpe. Max drawdown 48.4% against buy-and-hold's 46.6% while
averaging only 74% allocation, which directly contradicts the "edge comes from
avoiding big drawdowns" premise. Leverage in the low-vol tier is the first thing
to test removing.

Run on synthetic data, so this is not evidence about SPY. It is evidence the
backtester and the verdict logic work. Re-run as variant 2 once Phase 6 supplies
real data, before drawing any conclusion about the strategy.

Rules:

- **Log the failures too.** A log of only winners is worse than no log.
- A parameter change is a variant. Moving `high_vol_allocation` from 0.60 to
  0.50 counts. So does changing the HMM state count range.
- Under 30 closed out-of-sample trades, write "not enough data" rather than a
  number.
- Record all three benchmark comparisons, not just the flattering one.
