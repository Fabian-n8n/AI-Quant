# Where the tutorial contradicts CLAUDE.md

Four settings in `config/settings.yaml` conflict with the hard constraints you
wrote in CLAUDE.md. The tutorial's values are what ship, as requested. This
file records what each one changes so the decision is deliberate rather than
inherited.

None of this matters on a paper account. All of it matters before real money.

---

## 1. Leverage: 1.25x versus none

| | |
|---|---|
| CLAUDE.md | `Leverage \| None. Cash equities only.` Listed under **hard constraints**. |
| settings.yaml | `strategy.low_vol_leverage: 1.25`, `risk.max_leverage: 1.25` |

At 1.25x, a 20% drawdown in the underlying becomes 25% in the account. Your
stated bar is that this capital can be down 30% without touching the HDB
purchase. Leverage moves you closer to that line, and it does so fastest in
exactly the situation the system is designed to be most invested in: a low
volatility regime, near fully allocated.

The specific risk is not steady decline, it is the gap. Low volatility regimes
are where leverage gets applied, and they are also where a shock is least
priced in. February 2018 and February 2020 both began from calm.

**To revert:** set both values to `1.0`. Nothing else changes. `LowVolStrategy`
takes leverage as a constructor argument and a value of 1.0 makes it a plain
long strategy.

---

## 2. Circuit breakers: day-trading thresholds on a swing system

| | |
|---|---|
| CLAUDE.md | An entire section arguing these will misfire. |
| settings.yaml | `daily_dd_reduce: 0.02`, `daily_dd_halt: 0.03`, `weekly_dd_reduce: 0.05`, `weekly_dd_halt: 0.07`, `max_dd_from_peak: 0.10` |

The argument in CLAUDE.md is correct and worth restating: on a position held
for weeks, a 3% down day is noise. SPY has 3% down days in ordinary years. A
halt-everything breaker at that level will close the book on normal volatility,
repeatedly.

The 10% peak drawdown is the sharper problem. SPY drew down more than 10% in
2018, 2020, 2022 and 2025. A breaker at that level writes `trading_halted.lock`
during a routine correction, and since the lock requires manual deletion, the
system stops until you notice.

The cost is not the halt itself. It is that after the third false halt you will
start deleting the lock file without reading it, and at that point the safety
net is gone while still appearing to be in place.

**Swing-horizon alternatives** to test against your own backtest:

```yaml
risk:
  daily_dd_reduce: 0.04       # was 0.02
  daily_dd_halt: 0.06         # was 0.03
  weekly_dd_reduce: 0.06      # was 0.05
  weekly_dd_halt: 0.10        # was 0.07
  max_dd_from_peak: 0.18      # was 0.10
```

Do not adopt these blind either. Phase 4 builds the backtester; run both sets
through it and pick based on how often each fires and what it costs.

---

## 3. Initial capital: 100,000 versus your real number

| | |
|---|---|
| CLAUDE.md | "Set the paper account to the real starting capital, NOT an inflated number." |
| settings.yaml | `backtest.initial_capital: 100000` |

The reasoning in CLAUDE.md is right. At $100k, 1% risk per trade is $1,000 and
commission is a rounding error. At $5k it is $50 per trade, and the minimum
viable position size starts to bind against the 15% single-position cap.

Backtesting at $100k and paper trading at $5k means the backtest is answering a
question about a different account than the one you have.

Worth noting that this cuts in one direction only: results at $100k will look
better than the same strategy at $5k, never worse.

**To fix:** set `initial_capital` to your real intended starting capital, and
set the Alpaca paper account to match. Still open, see
`docs/OPEN-QUESTIONS.md`.

---

## 3b. Strategy allocation exceeds the risk layer's exposure ceiling

Surfaced in Phase 3. Not a CLAUDE.md conflict, an internal one between two
blocks of `settings.yaml`.

| | value |
|---|---|
| `strategy.low_vol_allocation` x `low_vol_leverage` | 95% x 1.25 = **118.75%** |
| `strategy.mid_vol_allocation_trend` | **95%** |
| `risk.max_exposure` | **80%** |

Two of the three strategy tiers ask for more exposure than the risk layer
permits. Phase 5's risk manager has absolute veto, so it clamps them, and
`LowVolBullStrategy` never runs at its stated 95%-at-1.25x.

Compounding it: the universe is 10 symbols but `risk.max_concurrent` is 5, so a
95% allocation spread across 10 names realises 47.5% once only 5 can be held.

Decide before Phase 4 measures anything, otherwise the backtest describes a
strategy that differs from the one documented. Options: raise `max_exposure`
(contradicts the no-leverage constraint), lower the strategy allocations, or
raise `max_concurrent` toward the universe size.

Full detail in `docs/PHASE3-NOTES.md` section 7.

---

## 4. alpaca-trade-api excluded from requirements.txt

Not a CLAUDE.md conflict, a dependency conflict. The tutorial lists both
`alpaca-trade-api` and `alpaca-py`. Only `alpaca-py` is installed.

`alpaca-trade-api` is Alpaca's deprecated SDK, superseded by `alpaca-py`.
Installing both downgrades shared dependencies:

| Package | With alpaca-py alone | After adding alpaca-trade-api |
|---|---|---|
| websockets | 17.1 | 10.4 |
| urllib3 | 2.7.0 | 1.26.20 |

Verified on this machine, 2026-09-07. `alpaca-py` covers trading, market data
and streaming on its own, so nothing is lost. If a tutorial snippet uses
`alpaca_trade_api` imports, it needs translating to `alpaca-py`, which is a
different import path and a slightly different client API.

---

## 5. Bar interval: "default 5-min bars" versus a daily-bar system

**Tutorial:** the Phase 7 spec says "MAIN LOOP (each bar close, default 5-min
bars)".

**This system:** `broker.timeframe: 1Day`, and every feature in it is
calibrated for daily bars.

This is not a preference. Running the existing feature set on 5-minute bars
changes what every number means:

| Feature | On daily bars | On 5-minute bars |
|---|---|---|
| `zscore_lookback: 252` | one year | about 3.2 trading days |
| `dist_sma_200` | the 200-day average | about 2.6 days |
| `sma50_slope`, EMA stops | 50 days | about 4 hours |
| `realvol_20` | 20 days, annualised x252 | 100 minutes, annualised as if daily |
| feature warmup (450 bars) | about 21 months | about 6 days |
| `retrain_interval_bars: 21` | monthly | about twice a week |

The circuit breakers would also re-evaluate every five minutes against
thresholds (`daily_dd_halt: 0.03`) written for a full session, and CLAUDE.md's
holding period is days to months. Nothing in Phases 2 through 5 was validated
on intraday data.

**Resolution:** `--timeframe` exists and works, so the tutorial's setup is
reachable, but daily remains the default and passing anything else prints a
warning naming this section. Choosing 5-minute bars means recalibrating the
feature windows, the breakers and the retrain cadence together, and re-running
the Phase 4 walk-forward to see whether the result is a strategy at all.

The tutorial's own architecture diagram and its holding-period discussion both
describe a swing system, so the 5-minute default reads as a leftover rather than
a deliberate choice.
