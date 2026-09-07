# Regime Trader - Project Brief

Context carried over from a research session. This file is the source of truth for
project decisions. Read it fully before starting any phase.

Repo root lives at `~/Desktop/AI Quant Trading`. The project is called
**regime-trader** in code and docs.

---

## What this is

A rules-based swing trading system, built with Claude Code, that runs on paper first
and only touches real money after it has demonstrated an edge over a real sample.

**The build is the easy 10%. Strategy validation is the 90%.** Optimise the plan for
finding out fast whether the strategy works, not for having a complete system.

---

## Owner context

- Based in Singapore. Product designer and AI automation builder by trade, not a
  quant. Explain quantitative concepts in plain terms when they come up.
- Prior bad experience with crypto. **No crypto rails anywhere in this project.**
  No USDC, no MetaMask, no Arbitrum, no perpetuals, no DeFi brokers.
- Already has an Interactive Brokers account with SGD deposit and withdrawal.
- Has an HDB purchase on the horizon. Real capital deployed here must be money that
  can be down 30% without affecting that.
- Writing style for any docs produced: plain, direct, no em-dashes, no corporate
  filler.

---

## Hard constraints

| Constraint | Value |
|---|---|
| Holding period | Days to a few months (swing / breakout). NOT intraday. |
| Paper broker | Alpaca (free API paper account, no funding needed) |
| Live broker | Interactive Brokers (SGD in and out, MAS-regulated) |
| Leverage | Intended: none. **Currently 1.25x** per the tutorial, see `docs/TUTORIAL-CONFLICTS.md`. |
| Asset classes | US stocks and ETFs. No options, no futures, no crypto. |
| Max risk per trade | 1% of portfolio |
| Paper account size | Set to the real starting capital, NOT an inflated number. **Still unset**, see `docs/OPEN-QUESTIONS.md`. |

### Why two brokers

Alpaca's API is far less painful to develop against and its paper account is
instant and free. IBKR is where the real SGD lives. Build and iterate on Alpaca,
port to IBKR before going live.

**Porting caveat:** different commission models and fill behaviour. Run a few weeks
on IBKR paper to confirm nothing broke in the port. Do not go straight from Alpaca
paper to IBKR live.

Note: the IBKR *Claude connector* is read-and-suggest only and cannot place orders.
The IBKR *TWS API / Web API* is a separate thing and can. Use the API for automation.

---

## Build order

Technical spec for each phase is in `BUILD-REFERENCE.md`. Read the relevant phase
section there before starting that phase; do not load the whole file every session.

This follows the AI Pathways tutorial "How To Actually Build a Trading Bot With
Claude Code". Phases are numbered as the tutorial numbers them, so a screenshot
of a tutorial prompt maps to a phase here without translation.

| Phase | What | Status |
|---|---|---|
| 1 | Project scaffolding and environment setup | Done |
| 2 | HMM regime detection engine (the brain) | Next |
| 3 | Volatility-based allocation strategies | |
| 4 | Walk-forward backtesting and validation | |
| 5 | Risk management layer | |
| 6 | Alpaca broker integration | |
| 7 | Main loop and orchestration | |
| 8 | Monitoring, alerts and dashboard | |

The system in five parts, which is how the tutorial frames it:

1. **Brain.** The HMM classifies what kind of market this is, from price and
   volume. It does not predict prices.
2. **Allocation.** How much capital that kind of market deserves.
3. **Safety.** Circuit breakers that fire on actual P&L, independent of the model.
4. **Broker.** Alpaca places the orders.
5. **Dashboard.** Shows what all of the above is doing.

An earlier draft of this file reordered these to put the backtester before the
strategy. That reorder has been dropped in favour of following the tutorial, on
the reasoning that a working reference implementation beats a better-sequenced
one that stalls. The tradeoff is real and worth naming: allocation strategies
get written in Phase 3 and are not validated until Phase 4, so anything built in
Phase 3 is provisional until the backtester exists. Do not grow attached to it.

**Do not skip ahead to Phase 8.** The dashboard is the most enjoyable part to
build and contributes nothing to whether the system makes money.

---

## Risk layer

All thresholds live in `config/settings.yaml` under `risk:`. Nothing else in the
codebase may hardcode one.

The system ships with the tutorial's values. Several of them are day-trading
numbers that will misfire on a days-to-months holding period, and one of them
(leverage) contradicts a hard constraint above. That is a deliberate, recorded
decision, not an oversight: see `docs/TUTORIAL-CONFLICTS.md` for what each one
changes and how to revert it. Revisit after Phase 4, when the backtester exists
to settle it with evidence rather than argument.

The **structure** is what matters and does not change:

- The risk manager is independent of the regime engine and the strategy, and has
  absolute veto over any signal.
- Circuit breakers fire on actual P&L, never on model state, so they still work
  when the model is confidently wrong. That is what they are for.
- The halt breaker writes `trading_halted.lock`, which requires manual deletion
  to resume. The friction is the point.
- Every breaker trigger is logged with the regime in force at the time. That is
  how you find out whether the regime layer is adding anything or just noise.

A breaker that fires constantly trains you to override your own safety rules,
which is worse than having no breaker. If a threshold fires on ordinary noise,
that is a reason to change the threshold, not to start ignoring it.

Position sizing formula:

```
position_size = (portfolio_value * max_risk_per_trade) / abs(entry_price - stop_loss)
```

Every position must have a stop. The system refuses orders without one, and this
is checked twice: once in the risk manager, once in the order executor.

---

## Validation standard

The system is not validated until all of these hold:

- [ ] 30 to 50 closed trades minimum. Below this, results are noise. Breakout
      strategies typically win 35-45% of the time, so six or seven consecutive
      losses is statistically normal and means nothing.
- [ ] Walk-forward tested, never in-sample. Train on a window, test blind on the
      next, slide forward, repeat.
- [ ] Beats buy-and-hold on SPY over the same period, net of commissions and
      slippage, on **both total return and Sharpe**. A strategy that is only 40%
      invested on average and matches SPY's return is actually winning; one that is
      fully invested and matches it is not. Report time-in-market alongside returns.
- [ ] Also benchmarked against 200-day SMA trend following and random entry.
- [ ] Expectancy per trade is positive:
      `(win_rate x avg_win) - (loss_rate x avg_loss)`
- [ ] The number of strategy variants tested has been logged. See "Counting your
      attempts" below.

Realistic timeline at this holding period: 6 to 12 months of paper trading.

### Look-ahead bias

The single most likely way this project produces a beautiful, worthless result.

- HMM: use the forward algorithm only. Do NOT use the library's default `predict()`,
  which processes the entire sequence and leaks future information.
- Any external dataset with a filing or disclosure lag (congressional trades,
  insider filings): backtest on the **disclosure date**, never the transaction date.
- Never tune parameters while looking at the full history. That is overfitting and
  it is the main reason retail quant systems fail live.

### Survivorship bias

The second most likely way. Equal in damage to look-ahead bias and far easier to
commit by accident.

If the backtest universe is "today's S&P 500 constituents", every company that went
bankrupt, got delisted, or was acquired at a loss has already been deleted from the
sample. A breakout strategy backtested that way is buying breakouts in a universe
where failure was retroactively made impossible. Results will look excellent and
mean nothing.

- The universe must be **point-in-time**: which tickers were listed and eligible on
  the date being simulated, not which are listed today.
- Free data sources generally do not provide this. Either buy a survivorship-bias-free
  dataset, or restrict the universe to something stable and honest about its limits
  (for example, a fixed liquid ETF and large-cap list) and state the caveat loudly in
  the results.
- Whatever is chosen, write the decision and its limitation in `docs/DATA-SOURCES.md`.

### Corporate actions

Splits and dividends must be adjusted for, or the breakout detector will fire on
every 4-for-1 split in history. But orders and stops execute at real, unadjusted
prices.

Rule: compute **signals** on adjusted prices, compute **position size, stops and
commission** on unadjusted prices. Getting these backwards is a silent bug that only
shows up as unexplained backtest-to-live drift.

### Counting your attempts

Walk-forward testing protects against tuning on the test set. It does not protect
against testing thirty strategies and shipping the one that got lucky.

Every distinct strategy variant put through the backtester gets one line in
`docs/EXPERIMENT-LOG.md`: date, what changed, out-of-sample result. If variant
twenty-three is the winner, the honest read is not "we found an edge", it is "we
bought twenty-three lottery tickets and one paid out". Fewer, better-reasoned
hypotheses beat many cheap ones.

---

## Deferred decisions

**QuiverQuant alternative data** - not part of this build. The MCP server is
connected in Claude Code but the account has no active subscription, and it
comes from a different tutorial. Revisit as its own project once this system is
validated. Notes live in `~/Desktop/TradingQuiverQuant/`.

**Live deployment** - not until the validation checklist above is fully met, and
then at a fraction of the paper size.

---

## Anti-patterns

- **"All tests pass" is not validation.** Tests verify the plumbing works. They say
  nothing about whether the strategy is profitable. Do not conflate the two.
- **Do not write code the owner cannot explain.** Particularly sizing logic, stop
  logic, and anything touching order placement. If a function is opaque, explain it
  before moving on.
- **Do not add complexity to fix a strategy with no edge.** More regimes, more
  indicators, and more factors on a losing strategy produce an overfitted losing
  strategy.
- **Do not ask Claude to predict prices or read chart images for patterns.** Neither
  is reliable. Technical logic belongs in deterministic, backtestable code.
- **Never hardcode API keys.** They go in `.env`, which stays in `.gitignore`. Do not
  paste live keys into chat.

---

## Deterministic vs non-deterministic

The core system must be **deterministic**: same input, same output, every time.
That is what makes it backtestable.

Claude may be used **non-deterministically** as a supplementary reviewer, for example
scoring fundamentals on candidates the quant layer already surfaced, or poking holes
in a thesis. Those outputs vary between runs and can never be backtested, so they
must never drive the core strategy or sit on the critical path of an order.

Rule of thumb: Claude reviews, rules decide.

---

## Working agreements for Claude Code

- Every backtest result reported must state whether it is in-sample or
  out-of-sample. If that is not stated, assume it is wrong.
- Never report a strategy as working without the trade count. Under 30 closed
  trades, say "not enough data" rather than quoting a win rate.
- When a phase is finished, update `PROGRESS.md` before starting the next one.
- Stubs raise `NotImplementedError` with the phase number that fills them in. Do not
  quietly implement a stub from a later phase because it was convenient.
- Thresholds and parameters go in `config/settings.yaml` and nowhere else. A
  number hardcoded in a module is invisible to review and to the backtester.
- Tests passing is not validation. `tests/` verifies the plumbing. Whether the
  strategy makes money is a walk-forward question, answered in Phase 4.
