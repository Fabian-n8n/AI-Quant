# Phase 4 notes: walk-forward backtesting

What was built, what the spec did not account for, and the first honest
out-of-sample result.

---

## THE RESULT: no demonstrated edge

Run on synthetic data (no Alpaca credentials yet), 15 folds, 1,890
out-of-sample bars, 168 rebalances:

| Strategy | Total return | CAGR | Sharpe | Max DD |
|---|---|---|---|---|
| **regime-trader** | **+54.97%** | +6.01% | **0.18** | **-48.38%** |
| Buy and hold | +118.31% | +10.97% | 0.34 | -46.61% |
| 200 SMA trend | -0.15% | -0.02% | -0.05 | -42.80% |
| Random allocation (100 seeds) | +84.67% ± 72.12% | - | 0.24 ± 0.17 | -40.73% |

Alpha -2.45% annualised, beta 0.74, R² 0.87.

**It loses to buy-and-hold and to random allocation, on both return and Sharpe.**

The most damaging line is the drawdown. The strategy's entire premise is that
"the edge comes from avoiding big drawdowns through vol-based sizing". It drew
down **48.4% against buy-and-hold's 46.6%** while holding an average allocation
of only 74%. It took less exposure and lost more, which is the opposite of the
thesis. The 1.25x leverage in the low-volatility tier is the obvious suspect.

Beta 0.74 with alpha near zero says the same thing in different words: it took
26% less market risk and found no independent return to show for it.

**Caveat that matters more than the result.** This is synthetic data with
arbitrary drift terms. It is not evidence about SPY, and it must not be read as
"the strategy does not work". What it *is* evidence of: the backtester works,
the benchmark comparison is wired correctly, and the verdict logic correctly
refuses to pass a strategy that does not beat its benchmarks. The first real
verdict comes after Phase 6 supplies Alpaca data.

Logged as variant 1 in `docs/EXPERIMENT-LOG.md`.

---

## STRESS TEST RESULTS

Baseline (no shock): return +174.73%, max drawdown -44.20%.

| Scenario | Sims | Mean DD | Worst DD | Mean Return | Worst Return | Breaker Fired | Survived |
|---|---|---|---|---|---|---|---|
| Crash mild (-5% x10) | 3 | -34.1% | -35.8% | +89.0% | +72.1% | 100% | 100% |
| Crash severe (-10% x10) | 3 | -37.2% | -40.2% | +19.6% | -4.2% | 100% | 100% |
| Crash extreme (-15% x10) | 3 | -41.8% | -44.9% | -1.2% | -5.5% | 100% | 100% |
| Gap 2x ATR | 3 | -41.9% | -54.8% | +16.0% | -20.7% | 100% | 100% |
| **Gap 5x ATR** | 3 | **-63.6%** | **-70.6%** | -42.2% | -51.0% | 100% | **33%** |
| Regime misclassification | 4 | -37.2% | -44.2% | +118.5% | +78.0% | 100% | 100% |

Three things fall out of this.

**1. The misclassification test passes.** Worst drawdown with shuffled
allocations is 1.0x the baseline. Damage stays bounded when the regimes are
wrong, which is exactly what the independence requirement asks for. The system
is not merely surviving because the HMM happens to be right.

**2. Overnight gaps are the real risk, and it is not close.** A 5x ATR gap
sequence produces a 63.6% mean drawdown and **only 33% of runs survive**. Every
crash scenario, including -15% shocks ten times over, is survivable; the gaps
are not. That is the correct shape for a system holding positions overnight, and
it is the argument for the Phase 3 stop clamp and for Phase 5 sizing as though
the stop can gap through.

**3. The circuit breakers fire 100% of the time, in every scenario.** Including
the *mild* one. The baseline itself has a 44.2% max drawdown against
`max_dd_from_peak: 0.10`, so the halt breaker would trip with no shock at all.

That is a quantified confirmation of the argument CLAUDE.md makes: the
tutorial's breaker thresholds are day-trading numbers that misfire continuously
on a days-to-months horizon. A breaker with a 100% fire rate carries no
information. Worse, in live use it trains you to delete `trading_halted.lock`
without reading it, and at that point the safety net is gone while still
appearing to be in place.

Phase 5 should retune against these numbers rather than adopting the defaults.
The swing-horizon alternatives in `docs/TUTORIAL-CONFLICTS.md` (daily -6%,
weekly -10%, peak -18%) are a starting point, and this backtester is now the
tool to settle them.

**Caveat:** synthetic data, and 3-4 simulations per scenario rather than the
spec's 100. Directionally clear, not precise. Re-run at `--mc-sims 100` on real
data before treating any figure as a number.

---

## 1. The IS window is below the HMM's own minimum

Phase 4 specifies a 252-bar in-sample window. Phase 2 specifies
`min_train_bars: 504`. `HMMEngine.fit()` raises below its minimum, so **every
fold would have failed**.

The backtester lowers the floor to the IS window size and logs it once. That is
the spec-faithful choice, but a 252-bar fit is genuinely thin. BIC's penalty is
`n_params * log(n_samples)`, and at 252 samples with the 6-feature subset:

| states | params | BIC penalty |
|---|---|---|
| 3 | 89 | 492 |
| 7 | 237 | 1,310 |

An 818-point penalty spread that the likelihood has to overcome, so selection
biases toward fewer states than a longer window would choose. Observed across 15
folds: 4, 5 and 6 states selected, never 7. `backtest.hmm_min_train_bars`
controls the floor if you want to test 504 with a longer history.

---

## 2. The naive loop is O(T²) and does not finish

First implementation called `engine.classify(features.loc[:timestamp])` per bar,
which re-runs the forward pass over the entire prefix every time, and
`generate_signals` per bar, which recomputes EMA and ATR over a growing slice.
Both are O(T) per bar and O(T²) overall. It **timed out at two minutes** without
completing a single symbol.

Two changes brought it to 11.5 seconds for 15 folds:

- **`HMMEngine.stream()`**, a new `RegimeStream` that advances a cached forward
  alpha one step per bar and carries its own tracker. Phase 2's spec already
  called for caching alpha "for efficiency in live/backtest loop"; this is where
  that pays. It is warmed on the last 60 bars of the training window so the
  filtered distribution has settled before the first OOS bar rather than
  starting cold from `startprob_` exactly where results begin to count.
- **`StrategyOrchestrator.target_allocation()`**, which returns the gross
  allocation without constructing a `Signal`. The spec is explicit that the
  backtester carries no per-trade stops, so the ATR work behind stop placement
  was pure waste. It shares the allocation rules with `generate_signal` rather
  than restating them, because two copies would eventually disagree and the
  backtest would measure a strategy the live loop does not run.

Features are computed **once** on the full series and then sliced. That is safe
only because the Phase 2 feature layer is provably causal (trailing windows,
rolling z-score), so row t of a full-series computation is identical to row t of
any prefix computation. `test_appending_future_bars_changes_nothing` is what
licenses this.

---

## 3. A flat equity curve produced a Sharpe of -1.04e17

Found by a test, not by inspection.

`std()` of a constant series is not exactly zero in floating point, it is around
1e-20. The guard was `if sd > 0`, which passes, and the division explodes.

This is not a rounding nuisance. Flat equity is **normal** in this system:
whenever the allocator is fully in cash, or whenever Phase 5's breakers halt
trading, the curve goes flat. The path gets hit in ordinary use, and the bogus
value looks like a triumph rather than an error. A Sharpe of 1e17 in a summary
table is the kind of number that gets screenshotted.

Fixed with `ZERO_VOL_TOLERANCE = 1e-12` in `sharpe_ratio`, `sortino_ratio` and
`calmar_ratio`. `test_zero_volatility_does_not_explode_the_ratios` is the
regression, and `test_no_metric_returns_a_non_finite_value` sweeps the whole
report for infinities.

---

## 4. "Trade" means rebalance, not round trip

The spec says "record a trade whenever allocation changes". So a trade spans one
rebalance to the next, and its P&L is the equity change across that holding
period. There is no entry/exit pair to difference, because the position is
never fully closed, only resized.

Win rate, average holding period and consecutive losses all read against that
definition. A 57% win rate here means 57% of *holding periods between
rebalances* were profitable, which is a different statement from 57% of round
trips.

The final open trade is force-closed at the last mark, otherwise its P&L drops
silently out of every metric.

---

## 5. Circuit breakers are evaluated, not invoked

The spec asks stress testing to report "% where circuit breaker fired". The
`RiskManager` that owns breakers is **Phase 5** and does not exist yet.

Rather than skip the metric, `evaluate_breakers()` in `stress_test.py` applies
the configured thresholds from `settings.yaml` directly to an equity curve. It
reads only P&L, never model state, which is the property the real breakers must
also have.

**Phase 5 should replace this helper with the real risk manager** so the two
cannot drift apart. Two implementations of the same thresholds is exactly the
kind of duplication that ends with the stress test passing and live trading
failing.

---

## 6. The backtest measures the unclamped strategy

Per the spec there are no per-trade stops and no risk manager in the backtester.
The consequence is that these results reflect the **raw Phase 3 allocations**,
including the low-volatility tier's 118.75% gross exposure.

Phase 5 caps gross exposure at `risk.max_exposure: 80%`. Once that is wired,
live behaviour will differ from these numbers, and the backtest will stop
describing the system that runs.

This is the unresolved decision from Phase 3, and it now has a cost attached:
until it is settled, every backtest result is measuring a configuration that
cannot be traded. See `docs/PHASE3-NOTES.md` section 7.

---

## 7. Reading the two attribution tables

These matter more than the headline number.

**Regime breakdown** answers whether the HMM earns its keep. From the run above,
returns are not flat across regimes (the "neutral" regime contributed +24% of
return over 8% of the time, Sharpe 1.50; "euphoria" contributed -6.6% over 20%
of the time). So the regimes are separating *something*. Whether that survives
on real data is the open question.

**Confidence buckets** answer whether the confidence score means anything. The
70%+ bucket (1,747 bars, Sharpe 0.19) barely beats the sub-50% bucket (125 bars,
Sharpe -0.22) and the sample is lopsided, so this is weak evidence at best. If
that pattern holds on real data, `uncertainty_size_mult` is scaling positions on
a number that carries little information.

---

## 8. What the spec left open

| Ambiguity | Decision |
|---|---|
| Multiple symbols | One independent backtest per symbol. The spec's CLI is `--symbols SPY`, and the allocation math is single-asset |
| Which price fills a rebalance | Next bar's **open**, per the 1-bar fill delay. A target computed from a close cannot be filled at that close |
| Slippage direction | Always against the trade: pay up to buy, down to sell |
| Fractional shares | `int()` truncates, so a rebalance can only under-allocate. Rounding up would quietly exceed the leverage limit |
| Random benchmark allocations | Drawn from the allocation levels the strategy actually used, so it is the *timing and selection* being randomised, not the sizing envelope |
| Max drawdown duration | Peak to **recovery**, not peak to trough. A 12% drawdown that recovers in a fortnight and one that takes three years are different experiences |

---

## 9. Performance

| Operation | Time |
|---|---|
| Walk-forward, 15 folds, 2,600 bars | 11.5 s |
| Full report with 3 benchmarks (100 random seeds) | ~40 s |
| One crash scenario, 15 Monte Carlo sims | ~3 min |
| Full stress suite (7 scenarios + gaps + misclassification) | ~30 min at 15 sims |

The stress suite is the expensive one because each simulation is a complete
walk-forward. Use `--mc-sims 15` while iterating and the spec's 100 only for a
final run.
