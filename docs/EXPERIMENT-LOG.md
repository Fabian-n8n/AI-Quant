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


## Variant 5 — universe and absolute momentum, 2026-09-22

Built `scripts/search.py`, the iteration loop the brief describes. Eight arms:
universe x regime source x trend filter. Search ran on data before
**2023-07-01**; the holdout was run once, afterwards, and nothing was tuned
against it.

**Search (holdout withheld):**

| universe | regime | trend | return | maxDD | Sharpe | trades |
|---|---|---|---:|---:|---:|---:|
| equity | hmm | off | 25.1% | -10.1% | 0.77 | 1652 |
| equity | hmm | sma200 | 21.3% | -12.0% | 0.77 | 697 |
| equity | volatility | off | 19.1% | -11.7% | 0.60 | 1530 |
| equity | volatility | sma200 | 16.9% | -12.8% | 0.62 | 646 |
| diversified | hmm | off | 25.9% | -12.9% | 0.84 | 1387 |
| **diversified** | **hmm** | **sma200** | **23.8%** | **-8.4%** | **0.96** | **492** |
| diversified | volatility | off | 24.8% | -11.3% | 0.81 | 1258 |
| diversified | volatility | sma200 | 23.8% | -9.7% | 0.91 | 443 |

**Deflated Sharpe 0.634 at 41 trials against a 0.95 bar. It does not clear.**
The Sharpe improvement is not distinguishable from a lucky search and is not
the reason anything shipped.

**Holdout, `diversified-hmm-sma200`, full span:** +62.9%, maxDD -8.4%,
Sharpe 1.16, 919 trades.

| year | strategy | SPY |
|---|---:|---:|
| 2019 | +8.2% | +31.1% |
| 2020 | +12.6% | +18.5% |
| 2021 | +6.7% | +28.6% |
| **2022** | **-7.2%** | **-18.2%** |
| 2023 | +22.7% | +26.2% |
| 2024 | +9.9% | +24.9% |
| 2025 | +3.7% | +17.7% |

### The filter only works with the sleeves, and that matters

On the equity-only universe the trend filter made drawdown **worse**
(-10.1% to -12.0%). On the diversified universe it made it **better**
(-12.9% to -8.4%). Two of four arms moved each way, so the filter is not
independently robust; the combination is what changed the shape.

The reading: blocking entries while holding fourteen correlated tech names
just means riding them down with no alternative. Blocking entries while the
ranker can rotate into Treasuries, gold or T-bills is a different rule.

### What shipped, and on what grounds

Both, together, because splitting them gives the worst of the three:
diversified without the filter has the deepest drawdown measured.

**Justified by risk structure, not by the Sharpe.** Same reasoning as the
circuit-breaker levels in variant 2: drawdown cut by a third, the 2022 loss cut
38%, and 70% fewer trades, which is 70% less cost drag. Return gives up 1.3
points. The DSR failure is recorded above and is the reason no claim of edge is
being made.

**Still not an all-weather strategy.** 2022 is -7.2%. That is a smaller loss,
not a profit. A long-only book with no shorts cannot profit in a broad decline;
the honest ceiling is losing less, and this reaches it more efficiently.

Paper account at the time of the change: **6 closed trades of the 30 preflight
requires.** No verdict until that count is met.


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

---

## Variant 6 — congressional trading data, 2026-09-22

Tested whether QuiverQuant's alternative data (congress trades, insider Form 4,
13F, lobbying, dark pool) is worth adding as a second signal source.

**The MCP connector is free. The data behind it is not.** The token in
`~/.claude.json` authenticates fine and `tools/list` returns 20 tools, but every
tool that returns actual data answers:

> Your QuiverQuant account is authenticated but has no active subscription.

Ten data tools tested, ten paywalled. Only `search_datasets`, the catalogue,
responds. API access starts around $25-30/month on the annual Hobbyist tier;
insider and lobbying sit on a higher tier.

### The signal was tested anyway, for free

The congress-trading strategy already trades as two live ETFs: **NANC**
(Democrat disclosures) and **KRUZ** (Republican), both listed 2023-02. That is
a better test than any backtest run here — real money, real fees, real 45-day
STOCK Act disclosure lag, and no way to peek at the future while building it.

`scripts/etf_alpha.py`, 893 daily bars, 2023-03-02 to 2026-09-22:

| fund | total | Sharpe | alpha/yr | t | beta | R² |
|---|---:|---:|---:|---:|---:|---:|
| SPY | 105.0% | 1.43 | — | — | 1.00 | 1.00 |
| NANC | **122.8%** | 1.44 | 1.07% | **0.42** | 1.07 | **0.92** |
| KRUZ | 28.6% | 0.75 | -0.86% | -0.20 | 0.40 | 0.34 |

NANC beat SPY by 17.8 points and it means nothing. Its R² against SPY is 0.92
and its beta is 1.07: it is a slightly levered S&P fund. Sharpe 1.44 against
SPY's 1.43 is the same risk-adjusted return, and the alpha t-stat is 0.42
against this project's |t| >= 2 bar — the same bar that rejected the HMM in
variant 3.

KRUZ is worse outright: -10.4%, -19.0% and -14.1% against SPY in 2024, 2025 and
2026, mostly from sitting at beta 0.40 during a rising market.

**Honest limit on this finding.** 3.5 years cannot detect a small alpha. The
standard error puts the detection floor at **5.1%/yr for NANC** — an alpha
below that is invisible in this sample. The claim is "not demonstrated", not
"proven absent". Re-run the script later; the funds keep accumulating data.

### Three structural problems, independent of the evidence

1. **Disclosure lag.** The STOCK Act allows 45 days. Only `ReportDate` is
   actionable; a backtest keyed on `TransactionDate` is look-ahead, and it is
   the same trap as the `.loc[:timestamp]` slice fixed in variant 2.
2. **Coverage.** Congressional trades cluster on a handful of mega-caps. Across
   19 symbols on a daily cadence the series is almost entirely empty.
3. **Cadence.** A few disclosures a month, each a month stale, against a system
   that decides daily. It is a monthly-rebalance factor, not a daily one.

### MCP cannot be a cron job

MCP is a session-scoped protocol between a client and a server. GitHub Actions
runs `python main.py`; there is no Claude session, no MCP client, and no tool
call. Putting any MCP feed into the daily pipeline means going to the vendor's
REST API with a key in repo secrets — a separate integration, not a switch.

The MCP is a research instrument for use inside a session. It is not, and
cannot be, a data source for `trade.yml`.

### Verdict

**Not adopted. No subscription.** Revisit only if `scripts/etf_alpha.py` shows
NANC's alpha t-stat clearing 2, which needs either several more years or an
alpha above 5%/yr appearing. The underlying instinct — more independent
information sources — was right and was already acted on in variant 5, on the
asset side, by adding the five defensive sleeves.

---

## Variant 7 — exits: trailing stop vs fixed target, 2026-09-24

Prompted by a plain question: when does a position close, and is there a
target? Checking the account answered it badly. **Zero take-profit orders
existed at the broker**, and every one of nine open positions had only a stop.
The target `reward_risk_ratio: 2.0` computes was never placed, because
`protect_position`'s OCO route needs a filled quantity and the entries are
limit orders that fill hours later, so the stop always arrived via the repair
path instead, which places a plain stop and nothing else.

### The backtest was measuring a different system

`PortfolioBacktester` held each stop fixed for the life of the trade while the
live engine ratcheted stops upward every cycle. Every number in variants 1-6
described a strategy nobody was trading. Fixed: `_ratchet_stop` applies the
same floor as `main.py`, off the **previous** close so the stop cannot be set
from a bar it is then tested against.

A second bug surfaced while fixing it. A falsy `reward_risk_ratio` computed
`entry + 0 x risk`, a target sitting exactly at the entry price, filling on the
next bar that traded up a cent: **7,130 trades and a 4.7% return.** That was an
artefact, not evidence, and it had made the no-target arm untestable.

### Four combinations, diversified-hmm-sma200, 19 symbols, full span

| trail | target | return | maxDD | Sharpe | trades |
|---|---|---:|---:|---:|---:|
| off | 2.0 | 62.9% | -8.4% | 1.16 | 919 |
| off | none | **326.2%** | **-34.6%** | 0.81 | 69 |
| on | 2.0 | 52.6% | -8.7% | 1.05 | 2362 |
| **on** | **none** | **69.9%** | **-9.6%** | **1.18** | 1391 |

**Row three is dominated, and that is the finding.** A trailing stop and a
fixed target fight each other: the trail cuts the position before it reaches
2R, so the target is collected almost never while the churn is paid in full.
It loses to row four on return, on drawdown *and* on trade count. Being beaten
on every axis at once is a structural result, not a Sharpe claim.

Row two is the trap. 326% is the largest return measured in this project and
it comes with **-34.6%**, which spends the entire risk budget CLAUDE.md sets
for real capital on a single line.

Row four is what the live account has actually been running since the trailing
ratchet landed. `reward_risk_ratio` is now `null` so the config states it.

### What this is not

1.18 against 1.16 is noise, and these are four more trials on top of the 42
that already failed the deflated-Sharpe test at 0.634. **No edge is claimed.**
The change is justified by row three being dominated and row two breaching the
drawdown constraint, neither of which depends on a Sharpe comparison.

### Live state at the time

Nine positions, four with a stop above entry (META +5.5%, AMD +2.0%,
PLTR +1.3%, NVDA +0.2%). Three stopped out on 2026-09-23 on the newly
tightened stops: AVGO -$49, GOOGL -$44, AMZN -$1. That is the trail doing its
job, and it is also the cost of it: a tighter stop exits more often.

**9 closed trades of the 30 preflight requires.**

---

## Tooling — luck-adjusted calibration, 2026-09-26

Not a variant; no strategy behaviour changed. Adopted in idea from
`bennyjo/phil`, whose own scorer reported **z = -3.98** two months in: calling
its shots far better than it hit them.

`backtest.performance.calibration_z(confidence, outcomes)` treats the stated
win probabilities as Poisson-binomial and reports how many sigma the realised
hit rate sat from the claim. **Negative is overconfident**, the dangerous
direction, because position sizes keyed to confidence are largest exactly when
least deserved. Returns the Brier score alongside, which needs no sample-size
excuse. 7 tests: honest forecaster near zero, a 25-point overconfidence gap
caught at z < -3, underconfidence reported separately, too-few-trades and
degenerate-variance cases refusing to speak, out-of-range probabilities dropped
rather than clipped.

**Deliberately unwired.** It needs a per-trade probability that the trade ends
in profit, and this repo does not produce one. `regime_confidence` is not it:
that is the HMM's posterior over which volatility *state* the market is in, a
different sample space. Variant 3 measured the link and found the buckets
inverted, sub-50% confidence scoring Sharpe 0.92 against 70%+ scoring 0.74.
Feeding it would produce a confident-looking sigma about a quantity nobody
predicted. The docstring says so.

Variant 4 re-verified the same day: on 1,012 out-of-sample SPY bars the HMM and
the volatility percentile **disagree on the tier 39.6% of the time** and still
produce equity curves correlating 0.997. The null model result stands.

---

## Tooling — counterfactual gate ledger, 2026-09-26

Adopted in idea from `bennyjo/phil`'s `core/counterfactual.py`, which prices
the prediction-market bets its gates declined. Instrument differs, question
does not: **a gate that blocks losers earns its place, one that blocks winners
is a tax paid in silence, and the two look identical in the log.**

`scripts/counterfactual.py` replays every recorded signal, approved and
refused, against real forward bars using the stop and target that were on the
signal at the time. One addition Phil's version does not have: **approved
signals go through the identical simulation as a control.** Knowing refused
trades lost is not enough; the gate only works if what it let through did
better.

### Two bugs in my own first run, both found before believing the output

1. **Incomplete windows.** Signals from the last few days have three bars of
   forward data, not twenty, and were being averaged against signals with a
   full window. Every refusal sat in the first week with complete data while a
   third of approvals were recent stubs. Rows still inside their horizon are
   now excluded.
2. **Non-overlapping dates.** The buckets do not span the same days, and a
   week of market direction dwarfs any gate effect. A same-date comparison was
   added and is the only line worth reading.

### Result, 10-bar horizon, same-date control

| bucket | n | win rate | mean |
|---|---:|---:|---:|
| approved | 43 | 37% | +0.25% |
| refused | 43 | 60% | **+2.64%** |

The refused trades did better, by 2.39 points. **This is not a live finding.**
Every one of the 43 refusals is `spread_too_wide`, on MSFT, AMZN, GOOGL, AAPL,
META and TSLA — the most liquid instruments in the market, whose real spreads
are a basis point or two.

The last such rejection has `bar_date` **2026-09-11**. Commit `0053489`,
"Stop trusting quotes from a market that is shut", landed **2026-09-14**.
**Zero spread rejections in the fourteen signal-days since.**

So the ledger is measuring a bug that was already fixed, and it is the first
independent confirmation that the fix was right rather than merely plausible.
It also prices what the bug cost while it ran: 43 signals that would have
averaged +2.64% over ten bars at a 60% hit rate, refused on quotes taken from
a closed market.

The one live gate with data, `max_positions`, has n=6 at a 5-bar horizon.
Too few to say anything, and that cap was itself miscounting until `83de5e3`.

**Re-run this after thirty more signals.** It is the cheapest read available
on whether the risk layer is discriminating or just reducing turnover.

---

## Variant 8 — does trading MORE make more money? 2026-09-26

**Question:** the account closes roughly 3-4 trades a week and takes two months
to reach the 30 the preflight wants. Can turnover be raised, and does raising
it pay?

Tightening the trailing stop is the only lever that raises turnover without
changing instrument: a closer stop exits sooner and frees the slot. Swept
`risk.trailing_stop.atr_multiple` on the shipped arm, everything else fixed.

| atr_mult | min_trail | return | maxDD | Sharpe | trades | trades/wk |
|---:|---:|---:|---:|---:|---:|---:|
| **2.5** (shipped) | 1.5% | **69.9%** | -9.6% | **1.18** | 1391 | 3.7 |
| 1.5 | 1.0% | 51.7% | -8.2% | 0.97 | 2337 | 6.2 |
| 1.0 | 0.5% | 65.6% | -8.7% | **1.18** | 3278 | 8.7 |
| 0.5 | 0.3% | **37.6%** | -6.5% | 0.86 | 5297 | 14.0 |

**Trading 3.8x more than today nearly halves the return.** 14 trades a week
returns 37.6% against 69.9% at 3.7. Drawdown improves, because a tight stop is
a small stop, but the return it costs is not a trade worth making.

Non-monotonic in the middle: 1.5 is worse than 1.0 on both return and Sharpe,
which is a sign this is noise around a flat optimum rather than a curve with a
peak to find. Four arms on a study already at 40+ trials; **no arm here is
claimed as better than the shipped one.**

### The one real option

`atr_multiple: 1.0` reaches **8.7 trades a week at the same Sharpe (1.18)** and
gives up about four points of return. That is 30 closed trades in ~3.5 weeks
instead of ~8. **Not shipped**, because it is the user's call whether four
points of return is worth halving the wait, and because changing it resets the
count: the trades already closed measured a different exit rule.

### Thirty in one week is not reachable here

Thirty closed trades in five sessions needs ~8x current turnover, well past the
bottom row, where returns are already collapsing. The only route to that trade
count is intraday, and variant 8's companion measurement rules it out on cost:

**15-minute bars, our universe, 30 days.** Median absolute 15-minute move
**0.148%**. Round trip at the configured 5bp slippage, before any spread,
**0.100%**. Costs eat **67% of a typical move**. SPY is worst at 2.24x cost to
median move; only SMCI, the most volatile name held, has a median move three
times its cost.

And the scheduler cannot carry it regardless. Measured over the last twelve
scheduled refreshes: **median delay 1.3h, max 4.2h** behind the requested cron.
A scalp holds minutes.

### Why more trades would not fix the record anyway

Statistical power, two-sided, 95% confidence, 80% power:

| true win rate | trades needed to prove it beats a coin |
|---:|---:|
| 52% | 4,893 |
| 55% | 777 |
| 60% | 189 |
| 70% | 42 |

**30 trades distinguishes nothing.** The gate is a smoke test for the plumbing,
not a measurement of edge — and on that count it has already earned its keep,
catching four execution bugs in two weeks that no backtest could model. The
919-trade holdout is where statistical evidence comes from.
