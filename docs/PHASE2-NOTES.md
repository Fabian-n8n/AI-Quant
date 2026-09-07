# Phase 2 notes: the HMM regime engine

What was built, what the spec left ambiguous, and the two things that will bite
if they are not understood.

---

## 1. The capacity problem, and why `feature_columns` defaults to `volatility`

This is the most important thing on this page.

A full-covariance Gaussian HMM fits `n * d * (d+1) / 2` covariance terms alone.
With the spec's 14 features:

| states | parameters | vs 504 training rows |
|---|---|---|
| 3 | 365 | 0.72x |
| 4 | 491 | 0.97x |
| 5 | **619** | **1.23x** |
| 6 | **749** | **1.49x** |
| 7 | **881** | **1.75x** |

Five states and up have more parameters than data points. Those covariance
matrices are singular and the fit is meaningless.

The subtler half is worse. BIC's penalty is `n_params * log(n_samples)`, which
at 504 rows is 6.22 per parameter. Going from 3 states to 5 costs 254 extra
parameters, a penalty of about 1,580, which the likelihood cannot recover. **BIC
would select 3 states every single time, regardless of the data.** The automatic
model selection would still run, still log its candidate scores, and still
return an answer, while having stopped selecting anything. The tutorial's
five-regime dashboard could never appear.

`check_fittability()` raises `InsufficientDataError` at fit time rather than
letting this happen quietly. The message names the offending candidates, their
parameter counts, and the fixes.

`settings.yaml` therefore ships `hmm.feature_columns: volatility`, a six-column
subset costing 237 parameters at n=7. This is not a compromise on the spec, it
follows from its own design philosophy: the HMM is a **volatility classifier**,
and it does not need momentum and mean-reversion features to do that job.

```yaml
hmm:
  feature_columns: volatility   # 6 columns, fits at 504 rows
  # feature_columns: all        # all 14, needs ~1800+ rows
```

All 14 features are always computed. `feature_columns` only controls what the
HMM observes, so Phase 3 strategies and the Phase 9 dashboard can still use the
rest.

**The constraint is the ratio, not the feature count.** With 2,000+ usable rows
the full 14 fit without complaint, and the guard stays quiet. If you want all 14,
give it more history rather than overriding `strict_fittability`.

---

## 2. Warmup: `min_train_bars: 504` does not mean 504 bars of data

The longest base window is the 200-bar SMA, and the rolling z-score then needs
another 252 bars on top of that:

    (200 - 1) + (252 - 1) = 450 rows discarded

So 504 **usable feature rows** require **954 raw bars**, roughly 3.8 years, not
the two years the number suggests.

`required_raw_bars(504)` returns 954. Use it. Getting this wrong shows up in
Phase 4 as walk-forward windows that silently train on far less data than
configured, which is invisible until the results are inexplicably poor.

---

## 3. The spec's look-ahead test has an off-by-one

As written:

```python
regime_short = predict_regime_filtered(data[0:400])[-1]   # row 399
regime_long  = predict_regime_filtered(data[0:500])[400]  # row 400
assert regime_short == regime_long
```

Those are consecutive bars, not the same bar. `data[0:400]` holds rows 0-399, so
`[-1]` is row 399, while `[400]` is row 400.

This is worse than a test that fails. Regimes persist for weeks, so adjacent
bars usually share one, and the assertion passes almost always **for the wrong
reason** and would keep passing against an implementation that leaks. It was
observed passing by coincidence during development of this phase.

`tests/test_look_ahead.py` implements the corrected version (both sides read row
399) and adds the one with teeth: the entire 400-row prefix must match, on
continuous probabilities rather than just the argmax. Plus a parametrised
version appending 1, 50, 200 and 800 future bars.

Two structural defences sit alongside them:

- `test_viterbi_is_never_called` monkeypatches `model.predict` and
  `model.predict_proba` into landmines, so a future refactor that reaches for
  the simpler API fails the suite instead of quietly improving the backtest.
- `test_source_contains_no_predict_call` parses the AST for calls to
  `predict` / `predict_proba` / `decode` / `score_samples`. It parses rather than
  greps because the module docstring names `model.predict()` repeatedly in order
  to warn against it, and a string search flags its own warning label.

---

## 4. Two separate size reductions, often conflated

The spec describes both; they are independent and easy to merge by accident.

| Mechanism | Trigger | Multiplier |
|---|---|---|
| Transition damping | a regime change has not yet held `stability_bars` | `transition_size_mult: 0.75` |
| Uncertainty mode | more than `flicker_threshold` raw changes in `flicker_window` | `uncertainty_size_mult: 0.50` |

When both apply, the smaller wins.

**Flicker counts raw changes, not confirmed ones.** Counting confirmed changes
would be circular: the transition damper exists to suppress exactly the flips
the flicker detector needs to see, so the detector would almost never fire.
`test_flicker_counts_raw_changes_not_confirmed_ones` pins this.

---

## 5. Labels sort by return; the strategy sorts by volatility

The spec is explicit that these are separate, and it matters more than it
sounds. Crash and euphoria sit at opposite ends of the return ordering and
adjacent on the volatility ordering. A strategy keying off the label would size
off direction; keying off `volatility_rank` sizes off turbulence, which is what
the allocation layer is supposed to do.

`RegimeInfo` carries both. `get_volatility_rank()` is what Phase 3 should call.
`test_volatility_rank_is_independent_of_the_label` asserts the two orderings
actually differ on fitted data.

Related: `expected_return` and `expected_volatility` are computed from **raw
returns** of the bars assigned to each state, not from the model's `means_`.
Feature space is z-scored, so a mean read off `means_` would be a unitless
number that looks like a percentage and is not one.

---

## 6. Decisions the spec left open

| Ambiguity | Decision |
|---|---|
| "slope of N-period SMA" over what window | 10-bar least-squares slope, normalised by the SMA level so it is a fractional rate rather than dollars per bar, and comparable across symbols |
| "RSI(14) z-score" | RSI emitted raw; the global 252-bar z-score handles standardisation. Applying a second one would double-standardise it and shrink its influence against every other feature |
| z-score outliers | clipped at +/- 5 sigma by default. A single 40-sigma day will otherwise consume an entire Gaussian state, since the likelihood is quadratic in the residual |
| `n_init=10` | hmmlearn has no `n_init`, so restarts are looped manually with seeds derived from `random_state`, keeping the best log-likelihood. Deterministic, so the same data gives the same model |
| Wilder smoothing | `ewm(alpha=1/n, adjust=False)`. Wilder seeded from an SMA; the difference decays well inside the discarded warmup |
| First bar, no prior regime | adopted immediately rather than withholding classification for `stability_bars` |

---

## 7. Diagnostics worth reading after every refit

- `summary()` — one row per state with annualised return, volatility,
  volatility rank, frequency and expected duration.
- `get_expected_durations()` — `1 / (1 - p_stay)` in bars. **A median near 2-3
  bars means the model found noise, not regimes.** Currently around 25 bars on
  the synthetic fixture.
- `metadata.all_bic_scores` — every candidate's score, kept so the selection can
  be second-guessed.
- A **boundary warning** fires when BIC picks the largest or smallest candidate,
  because a selection landing on the edge of the search range means the range is
  the binding constraint rather than the data.

---

## 8. Performance, for Phase 4 planning

Measured on this machine, 2,600 raw bars, 7 states, 6 features:

| Operation | Time |
|---|---|
| Feature build (2,600 bars) | 13 ms |
| Forward filter (2,150 rows) | 156 ms (73 us/bar) |
| `classify_series` incl. tracker | 188 ms |
| One full fit (5 candidates x 10 restarts) | 4.1 s |
| Projected Phase 4 walk-forward, 18 folds | ~1.2 min |

Fast enough that Phase 4 needs no optimisation. The live loop uses
`make_live_filter()`, which advances one bar per step instead of replaying
history, and produces numerically identical output to the batch path
(asserted, at 1e-12).

---

## 9. Not yet validated against real data

Everything above is verified against synthetic data with three constructed
volatility regimes. That is the right test for correctness: real data cannot
tell you whether the model found the right answer, because nobody knows the
right answer.

It does mean the engine has not yet seen a real market. `data/market_data.py` is
still a Phase 6 stub, so the first real fit happens once Alpaca credentials are
in `.env`. Expect the state count and the regime statistics to look different,
and expect to revisit `feature_columns` when they do.
