# Current brief — read this first

Written 2026-09-22. Supersedes nothing; it is the *current task*, not the spec.
Spec is `docs/HANDOFF.md`. Evidence is `docs/EXPERIMENT-LOG.md`.

---

## The finding

The HMM regime signal carries **no measurable information about forward
returns.** Measured, not guessed:

- All 14 symbols, 504 OOS bars each, 2024-07-12 → 2026-07-16.
- Rank IC of `target_allocation` vs forward return, overlap-corrected:
  mean **+0.011** (h=1), **-0.001** (h=5), **-0.035** (h=20).
- **42 tests, zero reached |t| = 2.** Max 1.70. Chance alone at alpha 0.05
  gives ~2.1 such hits, so it scored *below* coin-flipping.
- `confidence` and `regime_id` score identically. Confidence buckets are
  inverted: <50% confidence → Sharpe 0.92, 70%+ → 0.74.

What it actually does: holds ~78% mean allocation, cuts on measured
volatility. Loses to buy-and-hold in every rising market, beats everything in
falling ones (SMCI -32.6% vs -72.9%). Beta 0.66-0.76, alpha ~0.

**It is a volatility-responsive beta reducer, not an alpha source.**

Full table and reasoning: `docs/EXPERIMENT-LOG.md` variant 3. Don't restate it,
read it if you need it.

---

## The task: variant 4, the null model

Prove the HMM is deletable, or prove it isn't. One question, falsifiable.

**The seam.** `core/regime_strategies.py:390` maps `VolatilityRank` →
strategy class. The HMM's only job is producing that rank. So:

> Replace "HMM → VolatilityRank" with "rolling realised vol percentile →
> VolatilityRank". Change nothing downstream.

Same three strategy classes, same allocations, same risk layer, same sizing,
same stops, same universe, same dates. The HMM becomes the *only* variable.

**Shape it like this** (lazy version, don't over-build):

- One function: 60-day realised vol of the reference symbol → terciles over a
  trailing window → LOW / MID / HIGH. Expanding or trailing window only —
  never full-sample terciles, that is look-ahead.
- One config switch: `strategy.regime_source: hmm | volatility`. Default `hmm`.
- No new abstraction layer. No factory. If it needs more than ~60 lines,
  you have misread the seam.

**Guardrails already in the repo — use them, don't rebuild them:**

- `tests/test_look_ahead.py` must still pass. The percentile window is exactly
  where leakage sneaks in.
- `backtest/performance.py` already has `signal_decay`, `ic_information_ratio`,
  `rolling_ic`, `consistency`. Score the null model with the same functions.
  `tests/test_signal_quality.py` (14 tests) calibrates them.
- `make backtest SYMBOLS="SPY QQQ AAPL MSFT NVDA META AMZN GOOGL AMD TSLA AVGO PLTR COIN SMCI"`

**Compare on:** exposure curve correlation, total return, Sharpe, max DD, and
IC per horizon. Log as variant 4 in `EXPERIMENT-LOG.md` — failures too, that is
the rule at the bottom of that file.

**The decision it produces:**

| Result | Action |
|---|---|
| Null model matches the HMM | Delete `core/hmm_engine.py` and its scaffolding. A few thousand lines that add nothing, gone. Keep the vol overlay — it earned 15-40 points in down markets. |
| HMM measurably beats it | First real evidence the regime layer does something. Keep it, and *now* the loop has a baseline to beat. |

Either outcome is a result. There is no wasted run here.

---

## After variant 4: the iteration loop

Only build this once variant 4 has answered. It is the Loop Engineering
pattern (generate → backtest → score → read why it failed → regenerate),
and it needs a baseline before it has anything to search against.

Four layers, and the last one is not automatable:

1. **Offline search** — generate variants, score every one on **ICIR**, not
   total return. Log all of them. `backtest/trials.py` already has
   `TrialRegistry` + deflated Sharpe for multiple-testing correction — use it,
   it is the thing that stops the loop finding prettier noise faster.
2. **OOS gate** — a held-out slice never touched during search. One shot per
   variant, enforced mechanically. `backtest/backtester.py:32` states the rule:
   the moment you change a parameter after seeing an OOS result, that result is
   no longer OOS.
3. **Paper forward-test** — 30-50 closed trades minimum before any verdict.
4. **Real money, small** — a human decision, never the loop's.

**Reject signals that decay in a few bars.** `signal_decay` already reports IC
across horizons 1-50. Sign flips across horizons = noise. Monotonic decay =
possible edge.

**What the loop must search over is signals, not HMM hyperparameters.** Another
state count is not a variant worth a run; variant 3 closed that question.

---

## Constraints — do not violate

- `broker.paper_trading: true` stays. No real money, ever, from a code change.
- **Do not touch** `max_single_position`, `max_concurrent`, or leverage yet.
  The 36% deployable cap is real (12 x 3%) and known — see
  `docs/TUTORIAL-CONFLICTS.md` — but raising exposure on a strategy that
  loses to random allocation amplifies the problem. Edge first, then sizing.
- **Do not add shorts.** Structurally excluded by design
  (`core/regime_strategies.py:31`), and `broker/order_executor.py` treats a
  naked short as a bug to prevent in four places. Adding a second direction to
  a strategy with no demonstrated edge doubles the ways to lose.
- Never paste API keys into a chat window. `tests/test_orders.py` scans the
  repo for live Alpaca keys on every run.
- `make test` and `make lint` clean before any variant is logged.

---

## Known open items, not this task

- Dashboard exposure diagnostic names `max_exposure` (80%) as the binding
  constraint. It isn't — `max_single_position x max_concurrent` (36%) is.
  Cosmetic, fix when convenient.
- The 2024-2026 window is mostly one regime, and 504 bars is a short test of a
  model needing ~954 raw bars to fit. **Running 2018-2022 would be a fairer
  trial** and is worth doing before deleting anything on variant 4's evidence
  alone.
