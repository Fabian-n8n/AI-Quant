# regime-trader: handoff spec

Everything another tool needs to rebuild this, plus what the evidence says so
it does not get rebuilt blindly. Source: `~/Desktop/AI Quant Trading`,
`github.com/Fabian-n8n/AI-Quant`, dashboard `ai-quant-kohl.vercel.app`.

---

## 1. What this is, in one paragraph

A daily US equity swing system. After each close it classifies the market into
a "regime" with a Hidden Markov Model, maps that regime to a target exposure,
ranks 14 large-cap names, sizes positions by risk-per-trade, and submits limit
orders that fill at the next open. Every filled position immediately gets a
stop order resting at the broker. It runs unattended on GitHub Actions against
an Alpaca **paper** account. No real money has ever been deployed.

---

## 2. The strategy, complete

This section is the migratable part. Everything else in the repo is plumbing.

### Universe
```
SPY QQQ AAPL MSFT NVDA META AMZN GOOGL AMD TSLA AVGO PLTR COIN SMCI
```
Daily bars, split and dividend adjusted for indicators, raw prices for orders.

### Step 1: classify the regime
Fit a Gaussian HMM on 6 volatility features of SPY. Number of states chosen by
BIC from candidates 3 to 7. A regime change must persist **3 consecutive bars**
before it is acted on. Confidence below 0.55 is treated as unreliable and
halves position size.

States are then re-sorted by **measured volatility**, not by return, and
bucketed into low / mid / high volatility tiers. This is why a state labelled
"bull" can land in the high-volatility tier and be allocated defensively.

### Step 2: regime to target exposure

| tier | condition | target | stop rule |
|---|---|---|---|
| low vol | - | 95% | `max(price − 3·ATR, EMA50 − 0.5·ATR)` |
| mid vol | price > EMA50 | 95% | `EMA50 − 0.5·ATR` |
| mid vol | price ≤ EMA50 | 60% | `EMA50 − 0.5·ATR` |
| high vol | - | 60% | `EMA50 − 1.0·ATR` |

`EMA50` = 50-period exponential moving average. `ATR` = 14-period average true
range. Only rebalance when the gap to target exceeds **10%**.

Stop floors: at least `0.5·ATR` below entry **and** at least 0.5% of price.

### Step 3: rank candidates
Score 0 to 1, weights fixed and transparent:

```
0.35  price above its own 50 EMA          (1 or 0)
0.30  stop tightness in ATR terms          1/max(1, stop_atr_mult)
0.20  fraction of requested size that survived the risk layer
0.15  20-bar return, scaled (return + 0.10) / 0.20
```

### Step 4: size the position
Risk-based, not notional-based:

```
shares = (equity × risk_pct) / |entry − stop|
```

Then capped by, in order:
- `max_risk_per_trade` 1%
- **gap cap**: `(equity × 0.02) / (3 × stop_distance)` — assumes the stop gaps
  through by 3×. This binds first and makes the effective risk **0.667%**
- `max_single_position` **3% of equity**  ← in practice this dominates everything
- `max_concurrent` 12 positions
- `max_exposure` 80% gross
- correlation: halve above 0.70, reject above 0.85 (60-day rolling)
- sector cap 30%

### Step 5: exits
- **Stop loss**: resting stop order at the broker, placed immediately on fill.
  Tighten-only; a stop may never be moved away from price.
- **Take profit**: `entry + 2.0 × (entry − stop)`, i.e. 2:1 reward to risk.
- **Trailing stop**: Alpaca native `trailing_stop`, `trail_percent` derived from
  ATR. Must be standalone, Alpaca forbids it as a bracket leg.

### Step 6: circuit breakers
Levels set by how often they fire against SPY 2015-2026, **not** by returns:

| rule | level | fires |
|---|---|---|
| daily reduce | 5% | ~0.6×/year |
| daily halt | 8% | ~0.1×/year |
| weekly reduce | 10% | ~0.2×/year |
| weekly halt | 15% | never in 10.7 years |
| peak halt | 25% below running high | 1 year in 11 |

Peak halt writes `trading_halted.lock` and requires manual deletion.

### Execution
Decide after the close, submit **limit** orders, fill at the next open.
Limit offset 0.1% through the touch against a live quote, 0.5% when pricing off
a stale bar close. Reject the order if the quote spread exceeds 1%, or if the
limit deviates more than 5% from the reference, or if the stop would sit at or
above the limit.

### Schedule
- `15 21 * * 1-5` UTC — the daily decision, after the US close
- `*/15 13-21 * * 1-5` UTC — dashboard refresh during market hours

---

## 3. Current live state

```
equity      $99,597.92     from a $100,000 start
cash        $77,773.68
invested    22.3%
positions   8   AVGO COIN META NVDA PLTR QQQ SMCI SPY
closed trades: 0
```

---

## 4. What the evidence says. Read before rebuilding.

Tested on real Alpaca bars, 2016-01 to 2026-09, 14 symbols, walk-forward, out
of sample, with the real position limits applied.

### It does not beat buying the index

| | return | worst drawdown | Sharpe |
|---|---:|---:|---:|
| this strategy | +93% | −21% | **0.93** |
| SPY buy and hold | +204% | −34% | 0.83 |

Better risk-adjusted, less than half the money, because it holds ~30% and cash
earns nothing.

### The HMM is not doing anything

Shuffling the model's own regime labels into random order, keeping the mix and
destroying only the timing:

| arm | Sharpe |
|---|---:|
| HMM | 0.93 |
| no HMM at all | 0.89 |
| **shuffled labels** | **0.94** |

Gap is −0.014 against a shuffle spread of 0.025. **The regime model, which is
the centrepiece, adds no timing value.** Whatever the system earns comes from
the entry screen, the stop placement and the position sizing.

### Three traps that produced false results here

1. **Synthetic data.** `load_bars` silently returned a random walk when it
   could not reach the broker. Every backtest before 2026-09-10 was measured on
   noise. Verify any tool is actually loading real bars.
2. **Backtest ≠ live.** A single-asset backtest with no position caps averaged
   80% invested; the live system with caps averages 6-30%. They are different
   strategies. Any backtest must model the position cap and count limit.
3. **No multiple-testing correction.** A dozen configurations were swept and
   the best reported. Best-of-20 over pure noise passes an uncorrected Sharpe
   nearly always. The best setting found here scores **0.789** on a deflated
   Sharpe against a 0.95 bar, i.e. not distinguishable from a lucky search.

### The constraint no tool removes

Telling a good strategy from a lucky one, using live P&L alone, needs roughly
**7 years** at this Sharpe. Live results cannot be the feedback loop. Any
product promising a bot that learns from its own recent profits is selling
noise-fitting.

---

## 5. What migrates to Astral, and what does not

Astral takes a **description of rules** and generates strategy code. It cannot
import a Python repository.

**Migrates:** section 2 above. Paste it.

**Does not migrate, and would need rebuilding or abandoning:**

| capability | where it lives now |
|---|---|
| Next.js dashboard on Vercel | `dashboard/` |
| scheduled runs, state committed to git | `.github/workflows/` |
| broker reconciliation (fills, positions, stops) | `main.py`, `data/repository.py` |
| SQLite state, migrations, run history | `data/` |
| walk-forward portfolio backtest with the real risk layer | `backtest/portfolio_backtester.py` |
| deflated Sharpe + append-only trial registry | `backtest/trials.py` |
| preflight gate, 7 checks, blocks live trading | `scripts/preflight.py` |
| 659 tests | `tests/` |

Astral's backtester works on up to 40,000 bars and does not, as far as its
documentation states, model per-position caps or correct for multiple testing.
Those are the two things that changed every conclusion here.

---

## 6. Prompt to paste into Astral

> I want to build a daily US equity swing strategy on a paper account.
>
> Universe: SPY QQQ AAPL MSFT NVDA META AMZN GOOGL AMD TSLA AVGO PLTR COIN SMCI.
> Daily bars.
>
> Entry: only long. A symbol is a candidate when its price is above its own
> 50-period EMA. Rank candidates by a score of 0.35 × (price above EMA50) +
> 0.30 × stop tightness in ATR terms + 0.15 × 20-day return, and take the top
> names up to 12 concurrent positions.
>
> Stop loss: max(price − 3 × ATR(14), EMA50 − 0.5 × ATR(14)), and never less
> than 0.5 × ATR or 0.5% of price below entry. Stops may only tighten.
>
> Take profit: entry + 2 × (entry − stop).
>
> Position sizing: shares = (equity × 0.00667) / (entry − stop), then capped at
> 3% of equity per position, 80% gross exposure, and reject a new position if
> its 60-day correlation to an existing one exceeds 0.85.
>
> Circuit breakers: halve size after a 5% daily or 10% weekly drawdown; stop
> trading after 8% daily, 15% weekly, or 25% below the equity high.
>
> Orders: limit orders submitted after the close, priced 0.1% through the
> touch, filling at the next open.
>
> Backtest this walk-forward on 10 years, out of sample, with per-position caps
> applied, against three benchmarks: SPY buy-and-hold, a 200-day SMA trend
> system, and random entry with the same exposure. Report max drawdown and
> Sharpe, not just total return. Tell me how many configurations you tested.

The last sentence matters more than the rest. If the tool cannot say how many
configurations it tried, its best result cannot be interpreted.

---

## 7. Honest note on the migration

The infrastructure in this repo works and is finished. The strategy does not
have a demonstrated edge, and its central idea tested as no better than random
timing. Moving the same rules to a different tool changes the plumbing, not
the edge.

If the goal is a nicer way to iterate on strategy ideas, Astral is a reasonable
purchase. If the goal is for the strategy to start making money, the binding
problem is that the regime model does not work, and that travels with you.
