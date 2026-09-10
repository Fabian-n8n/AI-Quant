# Phase 11: evidence, ablation, and the correction

## Why this phase exists

Three separate results were reported as fact and were not:

1. `load_bars` returned a random walk whenever it could not reach Alpaca, and
   no caller loaded `.env` first, so **every backtest before 2026-09-10 was
   measured on synthetic data**. "+55% against buy-and-hold's +118%" was a
   statement about a random number generator.
2. The walk-forward backtest rebalances one asset to a target allocation with
   no position caps, averaging ~80% invested. The live system averages 6-13%.
   The two are different strategies and only one of them runs.
3. A dozen configurations were swept and the best reported without correction.
   That is the procedure that manufactures edges.

## What was built

**`backtest/portfolio_backtester.py`** runs the universe through
`core.candidates.scan`, the same call the live engine makes, so the position
cap, correlation rejection and sector limits are the live implementations
rather than a copy. `--mode backtest --portfolio`.

**`backtest/trials.py`** implements the Deflated Sharpe Ratio. It answers "given
that I searched N configurations, how surprised should I be that the best
looked this good". Calibration is asserted in tests, not claimed: PSR on 300
zero-edge strategies has mean 0.484 and a 4.7% false-positive rate at the 0.95
threshold; best-of-20 searches over pure noise pass 0 times in 60.

**`scripts/study.py`** runs sweeps with the correction attached and records
every arm to `data/trials.db`, including failures. Counting only survivors is
the same mistake the correction prevents, so the registry is append-only.

## Results, SPY 2016-01 to 2026-09, 14 symbols, out of sample

### The circuit breakers are the binding constraint

| config | return | maxDD | Sharpe | exposure | trades |
|---|---:|---:|---:|---:|---:|
| shipped breakers, 3% cap | +2.73% | -10.75% | 0.12 | 5.9% | 565 |
| shipped breakers, 8% cap | -9.29% | -12.45% | -0.22 | 3.8% | 170 |
| swing breakers, 3% cap | +93.29% | -21.10% | 0.93 | 31.7% | 2859 |
| swing breakers, 12% cap | +66.43% | -29.13% | 0.51 | 35.2% | 856 |

`max_dd_from_peak: 0.10` with a `peak_equity` that never resets means that once
equity is 10% below its running high, every signal is rejected for the rest of
the run. SPY draws down more than 10% in ordinary years, so the system spends
most of a decade halted. `settings.yaml` already flagged these as day-trading
numbers on a days-to-months holding period.

`max_single_position: 0.03` is not protecting anything. It is the only size
small enough that equity never falls far enough to trip the halt.

### But it does not survive the correction

    Best arm: swing breakers, 3% cap
      Sharpe                  0.93
      Hurdle from 15 trials   0.63
      Probabilistic Sharpe    0.994   ignores the search
      Deflated Sharpe         0.789   accounts for it

Below 0.95. The verdict is robust to how the trial set is scoped: restricting
to the eight risk arms gives a hurdle of 0.64 rather than 0.63, because those
arms have a wider spread that offsets the smaller count. There is no scoping
choice that rescues it.

### The regime classifier is not timing anything

The first ablation ran at the shipped breakers and showed the HMM scoring
*below* a random permutation of its own labels. That was an artifact: at those
settings every arm sits near 6% exposure, so it measured a system that barely
trades rather than the classifier.

Retested at the swing breakers, where the system holds ~32% and takes ~2800
trades:

| arm | return | maxDD | Sharpe | exposure | trades |
|---|---:|---:|---:|---:|---:|
| HMM regime (shipped) | +93.29% | -21.10% | 0.93 | 31.7% | 2859 |
| fixed regime, no HMM at all | +87.33% | -22.05% | 0.89 | 31.4% | 2988 |
| shuffled regime, 5 seeds | +94.27% | -21.28% | 0.94 | 31.4% | 2704 |

HMM minus shuffled mean is **-0.014 Sharpe against a shuffle spread of 0.025**.
Seeds ran 0.91 to 0.98 and the HMM landed at 0.93, in the middle of them.

Shuffling destroys the timing and keeps the mix of labels. Scoring the same
either way means the classifier contributes the *distribution* of its labels
and no timing. Deleting it entirely (`fixed`) costs 0.04 Sharpe, which is
inside the same noise.

The HMM is the centrepiece of this project and, measured this way, it is not
earning its complexity. Whatever the strategy does earn comes from the entry
screen, the stop placement and the position sizing.

### Against the benchmark, honestly

Over the full out-of-sample span, 2018-10 to 2026-09:

| | return | maxDD | Sharpe |
|---|---:|---:|---:|
| strategy, swing breakers | +93.29% | -21.10% | **0.93** |
| SPY buy and hold | +203.74% | -33.79% | 0.83 |

Higher Sharpe, roughly a third less drawdown, and less than half the return,
because it holds about 32% and cash earns nothing here. Scaled to comparable
risk the two are close to a wash, and it still does not clear the correction.

## What changed, and what deliberately did not

**Changed:** the circuit breakers, to levels that fire on abnormal events. The
justification is frequency measured on SPY alone, not the return improvement,
because the sweep that found them scored 0.789 against a 0.95 bar. A breaker
that halts all trading three times a year is not protecting the strategy, it
is replacing it.

**Not changed:** `max_single_position`, the HMM, the universe, the entry
screen. The sweep favoured moving the position cap too and that part did not
clear the bar, so it stays.

**Kept under review:** the HMM. The ablation says it adds no timing at daily
frequency on this universe, but one test at one frequency is not grounds for
deleting the centrepiece. The next thing to try is a different feature set
before concluding the idea is wrong rather than the inputs.

## Standing rule

Nothing in `config/settings.yaml` changes on the strength of a sweep. A setting
moves when either the deflated number clears 0.95 on a holdout period the
search never touched, or the change corrects a defect that can be argued
independently of the backtest.
