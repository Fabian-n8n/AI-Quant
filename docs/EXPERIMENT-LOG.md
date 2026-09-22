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
| 2 | 2026-09-22 | Unchanged config, **first real SPY data** (2024-07-12 → 2026-07-16, 504 OOS bars) | 54 | +$555 | 0.78 | **LOSE** (+29.9% vs +37.2%) | win (+29.9% vs +25.0%) | win, not significant (+29.9% vs +26.0% ± 9.6%) | No |

**Variant 1 read:** loses to buy-and-hold and to random allocation on both
return and Sharpe. Max drawdown 48.4% against buy-and-hold's 46.6% while
averaging only 74% allocation, which directly contradicts the "edge comes from
avoiding big drawdowns" premise. Leverage in the low-vol tier is the first thing
to test removing.

Run on synthetic data, so this is not evidence about SPY. It is evidence the
backtester and the verdict logic work. Re-run as variant 2 once Phase 6 supplies
real data, before drawing any conclusion about the strategy.

**Variant 2 read:** the first evidence about real SPY rather than the synthetic
fixture, so variant 1's numbers should not be quoted as a verdict on this
strategy. Better than variant 1 on every axis, and it clears both the SMA and
random benchmarks on total return — but it clears random by 0.41 standard
deviations of the random spread, which is not a win, it is the middle of the
distribution.

Alpha +2.34% on beta 0.66. The report's own line is the right read: less market
risk taken, not independent return found.

**New in this variant: signal quality measured directly** (`signal_decay` in
`backtest/performance.py`). Rank IC between the allocation the strategy wanted
and the return that followed, corrected for the overlap in forward returns:

| horizon | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| IC | 0.039 | 0.039 | 0.013 | 0.046 | -0.030 | -0.014 | 0.053 |
| t | 0.88 | 0.61 | 0.17 | 0.46 | -0.21 | -0.07 | 0.15 |

Not one horizon reaches \|t\| = 1. `confidence` and `regime_id` score the same
way. The sign flips across horizons rather than decaying, which is the shape of
noise, not of a signal with a shelf life.

The overlap correction is load-bearing: `confidence` scores IC 0.154 at h=20,
which reads as 3.4 sigma against the raw 484 observations and 0.74 sigma against
the ~24 independent ones. Uncorrected, that cell would have looked like the
edge.

This does not say the HMM cannot work. It says that on SPY, over these two
years, its output carried no measurable information about forward returns — and
that raising exposure or adding a short side would be scaling a signal that has
not yet been shown to exist. One symbol, one two-year window: widen both before
treating it as final.

---

## Variant 3 — universe-wide signal test, 2026-09-22

All 14 symbols, same config, 504 OOS bars each (2024-07-12 → 2026-07-16).

**Rank IC of `target_allocation` vs forward return, overlap-corrected:**

| | h=1 | h=5 | h=20 |
|---|---|---|---|
| mean IC across 14 symbols | +0.011 | -0.001 | -0.035 |

**42 tests. Zero reached \|t\| = 2. Max was 1.70 (MSFT, h=1).** At alpha 0.05,
chance alone produces about 2.1 such hits, so the signal scored *below* what
coin-flipping would have generated. No symbol, no horizon, no exception.

**What the return table shows instead.** Mean target allocation across the
universe is 77.8%. The strategy loses to buy-and-hold in every rising market and
beats everything in falling ones:

| | regime-trader | buy-and-hold |
|---|---|---|
| GOOGL (up) | +59.8% | +93.0% |
| NVDA (up) | +44.8% | +60.8% |
| MSFT (down) | +6.7% | -10.2% |
| SMCI (down) | -32.6% | -72.9% |

That is the signature of a beta reducer, not an alpha source: hold ~78% instead
of 100%, and cut further when measured volatility rises. Beta lands 0.66-0.76
across the universe and the report's own alpha/beta line reads it correctly.

**The conclusion the module docstring already anticipated:** "If returns are
indistinguishable across volatility regimes, the regime layer is complexity and
nothing else, and the correct response is to delete it rather than add a sixth
state."

Nothing here says volatility-responsive allocation is worthless — in the two
down markets it was worth 15-40 points. It says the *HMM* is not what produces
that, and a rolling-volatility threshold would deliver the same exposure curve
without BIC selection, ten random restarts, flicker detection or a confidence
score that measures nothing.

## Variant 4 — the null model, 2026-09-22

**Question:** does the HMM do anything the volatility percentile does not?

Swapped `HMM → VolatilityRank` for `60-day realised vol percentile →
VolatilityRank` behind `strategy.regime_source: hmm | volatility`. Nothing
downstream changed: same three strategy classes, same allocations, same risk
layer, same sizing, same stops, same universe, same dates, same limit-order
fill model. The HMM is the only variable.

Percentile is taken over a **trailing** 252-bar window ending at the current
bar, never full-sample. `tests/test_look_ahead.py` gained four tests, including
one that fails if the window is widened to all history.

**Portfolio backtest, 14 symbols, walk-forward, OOS 2018-10-16 → 2026-04-24:**

| arm | return | maxDD | Sharpe | avg exposure | trades |
|---|---:|---:|---:|---:|---:|
| HMM | +63.68% | -10.07% | 0.94 | 21.6% | 2505 |
| realised vol (null) | +60.37% | -11.74% | 0.91 | 21.1% | 2540 |
| SPY buy-and-hold | +185.72% | — | — | 100% | 0 |

- exposure curve correlation **0.870**
- equity curve correlation **0.997**
- 21-bar hit rate: HMM 0.652, null 0.663

**IC vs forward return, mean across all 14 symbols, 763 held-out bars:**

| horizon | HMM IC | HMM ICIR | null IC | null ICIR |
|---:|---:|---:|---:|---:|
| 1 | -0.0317 | -0.319 | -0.0230 | -0.294 |
| 5 | -0.0530 | -0.297 | -0.0423 | -0.317 |
| 10 | -0.0708 | -0.272 | -0.0780 | -0.448 |
| 20 | -0.0505 | -0.089 | -0.0685 | -0.338 |
| 50 | -0.1600 | -0.301 | -0.0895 | -0.222 |

Both are small and negative at every horizon. Neither carries signal. The null
model reproduces the HMM's absence of signal exactly as well as the HMM does.

**The finding that settles it.** The two arms agree on the volatility tier only
**67.1%** of bars, and their raw allocation series correlate just **0.264**.
They are making materially different calls a third of the time. The equity
curves still come out 0.997 correlated and the Sharpes differ by 0.03.

That is stronger evidence than agreement would have been. If they agreed on
every bar, "same result" would be trivial. They disagree constantly, and it
changes nothing, which means the specific thing the HMM decides is not the
thing that produces the return. The volatility overlay is.

**Verdict: the null model matches. `core/hmm_engine.py` is deletable.**

The +3.3 points of return and +0.03 Sharpe are inside the noise of a single
walk-forward run, and variant 3 already established there is no forward-return
information to lose. What earned the drawdown protection is cutting exposure
when measured volatility rises, and the null model does that in 25 lines.

**Not deleted yet**, deliberately — deletion is a separate commit from the
measurement that justifies it, and the decision is Fabian's. The switch
defaults to `hmm`, so nothing changed in behaviour until someone flips it.


**Next variant must be the null model**, not another HMM tweak: inverse-vol or
threshold allocation, same risk layer, same universe. If it matches, the HMM is
deletable and that is a genuine result.

Rules:

- **Log the failures too.** A log of only winners is worse than no log.
- A parameter change is a variant. Moving `high_vol_allocation` from 0.60 to
  0.50 counts. So does changing the HMM state count range.
- Under 30 closed out-of-sample trades, write "not enough data" rather than a
  number.
- Record all three benchmark comparisons, not just the flattering one.
