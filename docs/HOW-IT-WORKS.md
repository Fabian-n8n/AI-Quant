# How this thing actually works

Written for someone who is not a quant. No formulas. Every term defined the
first time it appears.

## The one-paragraph version

Every weekday after the US market closes, a computer on GitHub wakes up, looks
at how the market has been behaving lately, decides whether conditions look
calm or dangerous, picks a handful of stocks that fit those conditions, works
out how many shares it can buy without risking too much, and places the orders
with Alpaca. Alpaca is a broker with a practice mode, so no real money is
involved. A separate job wakes up every fifteen minutes during market hours to
check what happened and update the website.

## The pieces, in the order they run

### 1. Get the prices

Ask Alpaca for the daily open, high, low and close of fourteen stocks. Store
them so we do not re-download years of history every time.

**One trap worth knowing.** Prices have to be *adjusted* for stock splits. If a
company splits 4-for-1, the share price drops 75% overnight and nothing bad has
happened. Feed the unadjusted number to the model and it learns that a 75%
crash is a normal event.

### 2. Work out what kind of market this is (the HMM)

**HMM stands for Hidden Markov Model.** Ignore the name. Here is the whole idea:

The market is in some *mood* right now. Calm and rising. Choppy and going
nowhere. Panicking. You cannot observe the mood directly, which is the "hidden"
part. You can only observe prices. But different moods produce different
*looking* prices: a calm market makes small steady moves, a panicking one makes
big erratic ones.

So the model works backwards. It looks at recent price behaviour, and asks:
which mood would most likely have produced this? It also knows moods are
sticky. If yesterday was calm, today is probably calm too. Markets do not flip
from calm to panic and back every day, and a model that assumed they did would
be useless.

That is it. Recent behaviour goes in, "the market is probably in mood number 3"
comes out, along with how confident it is.

We make it wait three days before acting on a mood change, so one strange day
does not flip the whole system around.

### 3. Decide how invested to be

Each mood gets an instruction. Calm mood, be nearly fully invested. Turbulent
mood, hold more cash.

The moods are ranked by how *jumpy* prices were in each, not by whether prices
went up. That is deliberate and it looks wrong on a dashboard until you know
why: a mood labelled "bull" can still be a jumpy one, and jumpy means smaller
positions regardless of direction.

### 4. Pick stocks and size the positions

Score the fourteen stocks and take the best few. Then the important part, which
is how *much* to buy.

Not "spend a fixed amount". Instead: decide where you would admit you were
wrong and sell (**the stop**), then buy however many shares make that mistake
cost a fixed small percentage of the account. A stock whose stop is far away
gets fewer shares. A stock whose stop is close gets more. Every trade then
loses roughly the same amount when it is wrong, which is what makes a losing
streak survivable.

### 5. Place the orders

Send them to Alpaca, then immediately place a matching **stop order**, which is
a standing instruction to sell if the price falls to a set level. It sits at the
broker, so it protects the position even when our computer is switched off.

### 6. Circuit breakers

If the account loses more than certain amounts, the system stops trading.

**This is currently the biggest problem with the setup**, and it is covered
below.

## Words you will see

| Term | What it means |
|---|---|
| **Position** | Shares of one stock that you currently own |
| **Stop** | Standing order to sell if the price drops to a level, so a loss cannot run away |
| **Exposure** | How much of the account is in stocks rather than cash. 30% exposure means 70% cash |
| **Drawdown** | How far the account has fallen from its highest point. The pain measure |
| **Sharpe ratio** | Return per unit of bumpiness. Higher is better. Above 1.0 is good, 0.5 is mediocre, below 0 means you lost money |
| **Backtest** | Replaying the strategy over old prices to see what it would have done |
| **Out of sample** | Tested on data the model never saw while learning. The only kind that counts |
| **Walk-forward** | Learn on 2019, test on 2020. Learn on 2020, test on 2021. Repeat. Mimics how it would really have been used |
| **Buy and hold** | Just buying SPY and doing nothing. The thing to beat. Usually wins |
| **Slippage** | The gap between the price you wanted and the price you got |

## What the evidence says right now

Tested on real prices, 2016 to 2026, out of sample, fourteen stocks, with the
real position limits applied:

| setup | 10-year return | worst drawdown | Sharpe |
|---|---:|---:|---:|
| what is running today | +2.73% | -10.8% | 0.12 |
| same, with sane circuit breakers | +93.29% | -21.1% | 0.93 |

**Both lose to just buying SPY.** But look at the gap between those two rows.
That is the same strategy, same stocks, same everything, with one settings block
changed.

### Why the current setup barely trades

There is a rule that says: if the account ever falls 10% below its best-ever
value, stop trading entirely and wait for a human.

That number came from a day-trading tutorial. This system holds positions for
days to months, and the US market itself falls more than 10% in ordinary years.
So the rule fires early, and then everything stops for the rest of the decade.

The 3%-per-position limit is not protecting you either. It is the only size
small enough that the account never falls far enough to trip that halt. The
system has been quietly routing around its own emergency brake.

### Does the market-mood model help?

We tested this by taking the model's own mood labels and **shuffling them into a
random order**. Same moods, same proportions, timing destroyed. If the model has
skill, it must beat its own shuffled labels.

At the current settings it did not. That may be because the system barely trades
at those settings, so a retest at settings where it actually trades is in
progress.

## What has to be true before any of this touches real money

`python scripts/preflight.py` checks seven things and currently fails four. It
blocks live trading until they pass. The two that matter most:

- **30 completed trades.** There are zero. Nothing can be concluded from an
  account that has not closed a single position.
- **Beats buy-and-hold, trend-following, and random entry.** It does not.

There is also a hard limit nobody can engineer around. Telling a good strategy
apart from a lucky one, from live results alone, takes **years**. So live
profit and loss cannot be the feedback loop. Improvements have to come from
testing on history, carefully, with a correction for how many things you tried.

## Why "how many things you tried" matters

Test twenty settings on pure randomness and the best one will look good. Not
might, will. Reporting that one as though it were the only thing you tested is
the single most common way backtests promise returns that never arrive.

So every configuration ever tested gets written to `data/trials.db`, including
the failures, and results are scored against the bar that the best of N
worthless strategies would have cleared anyway.

The best setting found so far scores **0.789** on that adjusted measure. The
bar is 0.95. It does not pass, which is why nothing has been changed.
