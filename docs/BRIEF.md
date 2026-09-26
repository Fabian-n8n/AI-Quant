# Current brief — read this first

Written 2026-09-22, updated 2026-09-26. The *current task*, not the spec.
Spec is `docs/HANDOFF.md`. Evidence is `docs/EXPERIMENT-LOG.md`.

**Status: everything this brief asked for is built.** Variant 4 answered,
variants 5 to 7 followed, and the iteration loop below is `scripts/search.py`.
What remains is the paper record, which no amount of code shortens. Read
"Where this actually stands" at the bottom before starting anything.

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

## The task: variant 4, the null model — **DONE 2026-09-22**

**Answer: the null model matches. The HMM is deletable.** Equity-curve
correlation 0.997, exposure correlation 0.870, and IC small and negative for
both arms at every horizon. The two arms disagree on the tier 33% of bars and
their raw allocation series correlate only 0.264, so they genuinely make
different calls a third of the time and it changes nothing.

`strategy.regime_source` still defaults to `hmm` because deleting
`core/hmm_engine.py` was not authorised, only shown to be justified. Full
tables in `EXPERIMENT-LOG.md` variant 4.

The original framing is kept below because the reasoning still applies to the
next null model somebody writes.

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

## After variant 4: the iteration loop — **BUILT, `scripts/search.py`**

8 arms, every one logged to `TrialRegistry`, winner scored with a deflated
Sharpe against the full trial count, holdout run exactly once. Best arm
`diversified-hmm-sma200` reached Sharpe 0.96 and **DSR 0.634 against a 0.95
bar, which does not clear.** What shipped from it shipped on risk structure,
not on that Sharpe. See variant 5.

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

---

## Reference implementation: Phil

`https://github.com/bennyjo/phil` — a self-improving Claude Code agent trading
Polymarket. **Read for ideas, not for code.** Different instrument: binary
prediction markets, not equities. Its own scorer reports it two months in,
behind its benchmark, **z = -3.98 overconfident, zero real bets placed.** That
is a system honest enough to measure its own failure, which is the part worth
taking.

Exactly three things, each with a gate. **Do not build one before its gate.**

| # | Item | Gate | Status |
|---|---|---|---|
| 1 | `.github/scripts/boundary.sh` — CI guard failing any non-`operator:` commit that touches protected paths | The day an agent first gets write permission on strategy files | **NOT BUILT.** No agent has that permission. Building it now guards nothing and adds a CI step that can only produce false confidence. |
| 2 | Luck-adjusted z-score next to the IC metrics | None, it is a measurement | **BUILT 2026-09-26.** `backtest.performance.calibration_z`, 7 tests. |
| 3 | Per-class promotion gating as config | Account funded with real money | **NOT BUILT.** Paper only. |

### What item 2 needs before it can report anything

`calibration_z` takes a stated **probability that the trade ends in profit**
and the realised outcome. This repo does not currently produce that number.

`regime_confidence` is **not** it, and passing it would be a category error:
it is the HMM's posterior over which volatility *state* the market is in.
"85% sure this is a high-volatility regime" makes no claim about whether the
next trade wins. Variant 3 measured that link directly and found the buckets
*inverted* — sub-50% confidence scored Sharpe 0.92, 70%+ scored 0.74.

So the metric is in place and deliberately unwired. Wiring it needs a real
per-trade win probability first, which is a modelling task nobody has started.
The function guards against the misuse in its own docstring.

---

## Where this actually stands

**The goal stated repeatedly is to beat the S&P 500. The evidence says this
configuration structurally cannot, and that conflict should be resolved
explicitly rather than absorbed quietly.**

Variant 4, same walk-forward window: strategy **+63.68%**, SPY buy-and-hold
**+185.72%**. Variant 5 holdout, per calendar year:

| year | strategy | SPY |
|---|---:|---:|
| 2019 | +8.2% | +31.1% |
| 2020 | +12.6% | +18.5% |
| 2021 | +6.7% | +28.6% |
| **2022** | **-7.2%** | **-18.2%** |
| 2023 | +22.7% | +26.2% |
| 2024 | +9.9% | +24.9% |
| 2025 | +3.7% | +17.7% |

It loses to SPY in every rising year and wins only in the falling one. Three
structural reasons, none of which a better signal fixes:

1. **Deployable capital is capped at ~36%** — `max_single_position` 0.03 x
   `max_concurrent` 12. Roughly a third invested cannot out-return something
   fully invested, whatever it picks. The constraints section forbids changing
   this, correctly, until there is an edge to size up.
2. **No shorts**, by design in four places. A long-only book cannot profit in
   a broad decline; losing less is the honest ceiling.
3. **No measured edge.** 42 IC tests at |t| < 2 in variant 3, and DSR 0.634 in
   variant 5. Nothing here has demonstrated alpha.

What it does deliver is a materially different risk shape: 2022 at -7.2%
against SPY's -18.2%, and max drawdown -8.4% against SPY's -24%. That is a
real product. It is just not "beats the S&P 500", and the two goals point in
opposite directions.

**The real gate is unchanged and is not a code problem: 10 closed paper
trades of the 30 preflight requires, win rate 0%, expectancy -$81.87.**
