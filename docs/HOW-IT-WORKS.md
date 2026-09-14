# How this thing actually works

Written for someone who is not a quant. No formulas. Every term defined the
first time it appears.

## The one-paragraph version

Every weekday after the US market closes, a computer on GitHub wakes up, looks
at how the market has been behaving lately, decides whether conditions look
calm or dangerous, picks a handful of stocks that fit those conditions, works
out how many shares it can buy without risking too much, and places the orders
with Alpaca. Alpaca is a broker with a practice mode, so no real money is
involved. A separate job refreshes the website every fifteen minutes while the US
market is open, so nothing changes overnight or at the weekend because nothing
is happening.

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

These levels used to be the biggest problem with the setup. They were set for a
day-trading system and this one holds positions for weeks, so they fired almost
immediately and then nothing traded for years. See below.

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
| the old circuit breakers | +2.73% | -10.8% | 0.12 |
| **what is running today** | **+98.15%** | **-18.8%** | **1.00** |
| just buying SPY | +203.74% | -33.8% | 0.83 |

**It still loses to just buying SPY on money made**, though it gets there with
about half the worst-case pain. Note the gap between the first two rows: that is
the same strategy, same stocks, same everything, with one settings block changed.

### Why the old setup barely traded

There was a rule that said: if the account ever falls 10% below its best-ever
value, stop trading entirely and wait for a human.

That number came from a day-trading tutorial. This system holds positions for
days to months, and the US market itself falls more than 10% in ordinary years.
So the rule fired early, and then everything stopped for the rest of the decade.

The 3%-per-position limit was not protecting you either. It was the only size
small enough that the account never fell far enough to trip that halt. The
system was quietly routing around its own emergency brake. The levels have since
been set by **how often they would fire** against ten years of real prices,
rather than copied from a tutorial for a different holding period.

### Does the market-mood model help?

We tested this by taking the model's own mood labels and **shuffling them into a
random order**. Same moods, same proportions, timing destroyed. If the model has
skill, it must beat its own shuffled labels.

It does not. Retested at settings where the system actually trades:

| setup | Sharpe |
|---|---:|
| the mood model | 0.93 |
| no mood model at all | 0.89 |
| **its own labels, shuffled** | **0.94** |

Shuffling scores the same, within the noise of shuffling. **The mood model is
the centrepiece of this system and it is not timing anything.** Whatever the
system earns comes from the entry screen, where the stop is placed, and how the
position is sized.

### Does it survive how orders actually fill?

The backtest used to assume every order fills at the next morning's open. Real
orders are limits that rest at a price, and fill only if the market comes to
them. Those are different strategies, so the honest version was built and run:

| fill model | return | worst drawdown | Sharpe |
|---|---:|---:|---:|
| everything fills at the open | +93% | -21.1% | 0.93 |
| **resting limit, what really happens** | **+98%** | **-18.8%** | **1.00** |

11.8% of orders never fill. The ones that do fill slightly cheaper, and that
more than pays for the misses. This was expected to be bad news and was not,
which is the only reason it is worth reporting.

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

## What was broken, and what fixing it changed

Found on 14 September 2026, all from live data rather than from reading code.

### The dashboard was showing a loss that had not happened

Alpaca reports a "current price" for each position. Outside US market hours that
number is not a price anything traded at, and it drifts. Checked on a Monday
morning, with no share having changed hands since Friday's close, **all eight
positions were marked below their own closing price**, by as much as 3.9%.

That turned a real **-$93** into a displayed **-$413**, and took $319 off the
equity on the dashboard. Since Singapore is thirteen hours ahead of New York,
almost every time you looked at it, you were looking at the broken version.

Positions and equity are now priced off the last trade that actually happened.
The first refresh after the fix moved equity from $99,588 back to $99,902.

This mattered beyond the display: every emergency brake is measured as a
percentage of equity, so a fake loss is a fake drawdown.

### Nothing could be bought at all

In the last full trading run, **all fourteen stocks were refused**, for three
different reasons that turned out to be one reason.

- Six were rejected for a bid-ask spread of about 10%. A wide spread normally
  means "this is hard to trade, stay away" — but the market was shut. There was
  no spread, only a stale quote.
- Four were refused because the order price looked 6% away from the market. The
  "market" was a bid with no ask on the other side.
- Four were rejected by the broker itself, for trying to buy more of something
  already held while a sell order sat protecting it.

The first two are now skipped when there is no live market behind the quote,
which is by design, since the decision is made after the close. Checked against
the real account: **14 of 14 stocks now get through, where 0 of 14 did.**

For the third, the system no longer adds to a position it already holds. That is
also what the backtest always did, so live and tested now match.

### The refresh was starting hours late

The schedule asked GitHub for a refresh every 15 minutes. GitHub treats frequent
schedules as low priority and quietly dropped most of them: on the days nobody
triggered it by hand, the first refresh of the day arrived at **16:57 and 17:07**
against a 13:30 open. Three and a half hours of each day had no dashboard.

It now asks four times a day instead of thirty-six, and each run keeps itself
alive for three hours. Asking for less is what makes it arrive.

### "Updated 2 days ago" was usually telling the truth

On a Monday morning in Singapore, the last US session really was Friday. The
badge was right; the sentence next to it was frozen at publish time and still
read "Opens in 2.6 days". The countdown is now worked out in your browser, so it
says how long until the next open as of the moment you look.
