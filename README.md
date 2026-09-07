# regime-trader

A rules-based swing trading system. A hidden Markov model classifies the current
market regime from price and volume, portfolio allocation is set by the
**measured volatility** of that regime, and an independent risk layer holds an
absolute veto over every order. Orders go to Alpaca. Paper first.

**Status: all 9 phases complete.** 437 tests pass, 4 of them against the live
Alpaca paper API.

> ### Read this before anything else
>
> **This strategy has no demonstrated edge.** Out of sample it loses to
> buy-and-hold *and* to random allocation under identical risk rules, and its
> drawdown trips the peak circuit breaker about 5% of the way into the backtest.
>
> The system is correct about what it does. That is not the same as it working.
> It is a well-built implementation of a strategy that has not earned money in
> testing. Paper only. See [`docs/EXPERIMENT-LOG.md`](docs/EXPERIMENT-LOG.md).

---

## Philosophy: risk management > signal generation

Most trading projects spend their effort on the signal and treat risk as
plumbing. This one is built the other way round, for a simple reason: **a good
signal with bad risk management goes to zero, and a mediocre signal with good
risk management survives to be improved.**

Four consequences run through the whole codebase.

**The risk layer cannot be overruled.** `RiskManager.validate_signal` is a
13-step cascade with a veto. It can shrink a position, never grow one. The
strategy asks; it does not instruct.

**Circuit breakers fire on realised P&L, never on model state.** They do not
know what the HMM thinks and cannot be talked out of firing by a confident
classification. When the model is wrong is exactly when the breakers matter, so
they are wired to money lost rather than to anything the model believes.

**The future is never available.** Every feature is a rolling causal transform,
the model uses the forward algorithm rather than Viterbi, and `model.predict()`
is never called anywhere in the codebase. Look-ahead bias is the failure that
produces a beautiful backtest and a losing account, so it gets its own test file
and its own integration suite.

**Reducing risk never needs permission.** The veto sits on the path that buys.
Selling down bypasses it entirely, because a rejection while exiting would trap
the system in a position it wanted out of, at the worst possible moment.

---

## Architecture

```
                  ┌──────────────┐
   Alpaca ───────▶│     data     │  adjusted prices for signals,
   market data    │ market_data  │  raw prices for orders
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │   features   │  14 rolling, causal transforms
                  │  rolling z   │  450-bar warmup, no fitted scaler
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │     HMM      │  BIC picks 3-7 states
                  │ forward algo │  filtered, never smoothed
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │   vol rank   │  states sorted by MEASURED volatility,
                  │ low/mid/high │  not by their labels
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  allocation  │  low  95% @ 1.25x
                  │  3 strategies│  mid  95% / 60% on trend
                  └──────┬───────┘  high 60% @ 1.0x
                         │
                  ┌──────▼───────┐
                  │     RISK     │  ◀── ABSOLUTE VETO
                  │ 13-step veto │      breakers on realised P&L
                  └──────┬───────┘      shrink only, never grow
                         │
                  ┌──────▼───────┐
                  │    broker    │  limit entry + resting stop
                  │    Alpaca    │  paper unless three sources agree otherwise
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  monitoring  │  4 rotating logs, 7 alert triggers,
                  │ term + web   │  terminal and web dashboards
                  └──────────────┘
```

**Why volatility rank and not the regime label.** The states are labelled
`crash` through `euphoria` by sorting on expected return. Allocation sorts them
again, independently, on expected *volatility*. Those two orderings disagree: on
a recent NVDA fit, the state labelled `weak_bear` had the **lowest** measured
volatility of all seven and drew the most aggressive allocation. A regime
labelled "bull" is not necessarily calm, and it is calmness, not direction, that
should determine how much capital is at risk.

---

## Quick start

**1. Install.**

```bash
cd "~/Desktop/AI Quant Trading"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Add Alpaca keys.** Get them from alpaca.markets → API keys panel. The
secret is shown once. Paper keys begin `PK`; live keys begin `AK`.

```bash
cp .env.example .env    # then edit. .env is gitignored and must stay that way.
```

**3. Check the build.** Touches no network.

```bash
python main.py --status
pytest -q               # 437 passed, 3 skipped
```

**4. Train a model.**

```bash
python main.py --train-only --symbols SPY
```

**5. Run the pipeline without placing anything.** This is the step that proves
the whole chain works against your actual account.

```bash
python main.py --dry-run --once --symbols SPY
```

**6. Look at it.** Terminal, or the web dashboard.

```bash
python main.py --dashboard                        # terminal
python main.py --dry-run --once --publish         # then:
cd dashboard && npm install && npm run dev        # http://localhost:3000
```

When you are ready to let it place paper orders, drop `--dry-run`. Before ever
going near a live account, read [`docs/OPEN-QUESTIONS.md`](docs/OPEN-QUESTIONS.md)
and [`docs/TUTORIAL-CONFLICTS.md`](docs/TUTORIAL-CONFLICTS.md), and see the FAQ
entry on switching to live below.

---

## Deploying the dashboard

```bash
./scripts/deploy.sh
```

It refuses to run if anything key-shaped is committed, builds the dashboard, then
pushes and deploys. Both logins are browser flows, so the first run will stop and
tell you which one to do:

```bash
brew install gh && gh auth login    # GitHub
vercel login                        # Vercel
```

Import the **repository root** into Vercel, not `dashboard/` — the root
`vercel.json` points the build at the right place.

The deployed site is a static export, so its data is baked in at build time.
Refreshing it means committing a new snapshot:

```bash
python main.py --once --publish
git add dashboard/public/data/state.json && git commit -m "publish" && git push
```

Think before committing a snapshot of a real account to a public repository. The
publisher strips credentials, order ids and file paths, but equity and positions
are still in there. For a continuously updating view, use the terminal dashboard,
which reads local state directly.

---

## CLI reference

| Command | What it does |
|---|---|
| `python main.py` | live/paper trading, long-running |
| `python main.py --dry-run` | full pipeline, orders impossible |
| `python main.py --once` | process one bar and exit. The intended cron entry |
| `python main.py --status` | build state, no network |
| `python main.py --train-only --symbols SPY` | fit the HMM and exit |
| `python main.py --dashboard` | terminal dashboard from the saved snapshot |
| `python main.py --publish` | write `dashboard/public/data/state.json` each bar |
| `python main.py --publish-demo` | write a labelled demo snapshot and exit |
| `python main.py --mode backtest --symbols SPY --compare --export` | walk-forward with benchmarks |
| `python main.py --mode backtest --stress-test --mc-sims 100` | crash, gap and misclassification tests |

Useful flags: `--symbols`, `--timeframe`, `--start`, `--end`, `--log-level`,
`--i-understand-live`.

**Scheduling.** Daily bar close is 4pm New York, which is 4am or 5am in
Singapore depending on US daylight saving. A fixed local cron time is wrong for
half the year.

```cron
30 21 * * 1-5  cd ~/Desktop/AI\ Quant\ Trading && .venv/bin/python main.py --once --publish
```

---

## Configuration

Everything tunable lives in [`config/settings.yaml`](config/settings.yaml).
A number hardcoded in a module is invisible to review and to the backtester, so
there are none.

| Section | Controls | The one to watch |
|---|---|---|
| `broker` | universe, timeframe, paper flag | `paper_trading` — never flip casually |
| `hmm` | candidates, stability, flicker, warmup | `feature_columns: volatility` (see below) |
| `strategy` | allocation per volatility tier, stops | `low_vol_leverage: 1.25` |
| `risk` | the veto: exposure, breakers, gap sizing | `max_dd_from_peak: 0.10` |
| `backtest` | walk-forward windows, slippage | `initial_capital` |
| `monitoring` | alert rate limits, delivery | `alert_min_level` |
| `orchestration` | loop cadence, retraining, snapshots | `model_max_age_days: 7` |

**`feature_columns: volatility` is not a preference.** With all 14 features and
full covariance, a 7-state model needs 881 parameters against 504 training rows.
Covariances go singular, BIC's penalty collapses selection onto 3 states
regardless of the data, and the model looks like it worked. The 6-column
volatility subset needs 237 parameters and fits. The engine raises
`InsufficientDataError` rather than let the first case happen quietly.

**Four shipped defaults contradict the project's own constraints**, deliberately,
because they are the tutorial's values. Each is marked `CONFLICT` in the file
with the line to change. See
[`docs/TUTORIAL-CONFLICTS.md`](docs/TUTORIAL-CONFLICTS.md) before funding
anything.

---

## Monitoring

**Logs** land in `logs/`, rotating at 10MB with 30 days retained, across four
streams: `main.log` (everything, true ordering), `trades.log`, `alerts.log`,
`regime.log`. Every entry carries the regime, probability, equity, position
count and daily P&L in force at the time.

**Alerts** fire on seven triggers with the severity each deserves. Circuit
breaker, data feed down and API lost are CRITICAL. Large P&L and flicker are
WARNING. Regime change and retrain are INFO and never leave the console, because
an inbox that receives a routine notification every few days stops being read.
Rate limited to one per event type per 15 minutes.

**Dashboards**, terminal and web, render the same `DashboardState.snapshot()`.
Adding a surface means adding a renderer, never a second way of computing the
numbers, so the two cannot disagree about what the system thinks. Neither can
place an order: `DashboardState` takes no order executor, so the capability is
absent rather than merely unused.

---

## FAQ

**Why the forward algorithm instead of Viterbi?**
Viterbi finds the most likely *sequence* of states, and to do that it revises
its view of yesterday in light of today. That is the right tool for analysing
history and a catastrophic one for trading, because the revision is only
available after the fact. The forward algorithm gives the filtered probability
of today's state using only data up to today, which is the only thing you can
actually act on. `model.predict()` is Viterbi, and it is never called here — a
test parses the AST to make sure it stays that way.

**Why does BIC pick the number of regimes?**
Because otherwise you pick it, and you will pick the number that made the
backtest look best. BIC penalises parameters, so it will not hand you a 7-state
model unless the extra states earn their complexity. When it selects the largest
candidate, that is a signal the range wants widening, and the engine says so.

**Why did the system reject my trade?**
Rejections are structured and logged as prominently as approvals. Check
`logs/trades.log` for a `signal_rejected` entry; `rejection_reason` will be one
of fifteen values. The common ones are `below_minimum_size` (the gap cap shrank
it under $100), `correlation_too_high` (too close to something already held),
and `circuit_breaker`. If the system has stopped trading entirely, look for
`trading_halted.lock` in the project root.

**Why is my risk per trade 0.67% when I configured 1%?**
Because the overnight gap rule binds first. Positions are sized so a stop that
gaps through by 3x still costs under 2% of the portfolio, which works out to
0.667% — below the 1% limit, so the gap rule always wins. This is a swing system
where every position is held overnight, so it is not an edge case. It is the
real number.

**How do I switch to live trading?**
Deliberately, and not soon. Three independent sources must agree before the
system will touch a live account: `broker.paper_trading` in settings,
`ALPACA_PAPER` in the environment, and the key prefix Alpaca issues. You also
need `--i-understand-live` and a typed confirmation at an interactive prompt,
which a cron job cannot satisfy — that last part is the point. Before any of
that, the validation checklist below has to actually pass, which today it does
not.

**How do I clear a halt?**
Delete `trading_halted.lock` by hand. The friction is deliberate: it forces
someone to look at what broke before the system can lose more, and it cannot be
cleared by a restart or a scheduled job.

**Can the dashboard trade?**
No. It reads a published JSON file, holds no credentials, and has no route that
reaches a broker. A display that can act is no longer a display.

---

## Validation checklist

None of this is meaningful until the system has run. **Nothing here passes yet.**

- [ ] 30 to 50 closed trades minimum
- [ ] Walk-forward tested, never in-sample
- [ ] Beats buy-and-hold on total return **and** Sharpe, net of slippage
- [ ] Beats 200-day SMA trend following
- [ ] Beats random allocation under identical risk rules
- [ ] Expectancy per trade is positive
- [ ] High-confidence trades outperform low-confidence trades
- [ ] Returns differ meaningfully by regime, or the HMM is not earning its keep
- [ ] Variants tested are logged in `docs/EXPERIMENT-LOG.md`
- [ ] Paper traded for at least a month with every decision reviewed

---

## Layout

```
main.py       TradingEngine and the CLI
config/       settings.yaml, every tunable parameter and nothing else
core/         hmm_engine, regime_strategies, risk_manager, signal_generator
broker/       alpaca_client, order_executor, position_tracker
data/         market_data, feature_engineering, cache/
monitoring/   logger, dashboard, alerts, publish
backtest/     backtester, performance, stress_test
dashboard/    Next.js web UI, static export
tests/        one file per subsystem, plus look-ahead and integration
docs/         per-phase notes, conflicts, open questions, experiment log
```

Runtime files, all gitignored: `models/`, `logs/`, `state_snapshot.json`, and
`trading_halted.lock` when a breaker has halted the system.

Two rules that hold everywhere:

- **Every threshold lives in `config/settings.yaml`.**
- **Broker-specific types stay inside `broker/`.** That is what keeps a later
  move to another broker a contained job instead of a rewrite.

---

## Documentation

| File | What it is |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Source of truth for project decisions |
| [`PROGRESS.md`](PROGRESS.md) | Where the project stands, phase by phase |
| [`docs/TUTORIAL-CONFLICTS.md`](docs/TUTORIAL-CONFLICTS.md) | Where the tutorial's defaults contradict the constraints |
| [`docs/OPEN-QUESTIONS.md`](docs/OPEN-QUESTIONS.md) | What needs a decision from you |
| [`docs/EXPERIMENT-LOG.md`](docs/EXPERIMENT-LOG.md) | One line per strategy variant tested |
| `docs/PHASE2-NOTES.md` … `PHASE9-NOTES.md` | Per-phase design notes and the bugs found |

---

## Disclaimer

**This is educational software. It is not financial advice, and it does not
guarantee profits.**

It currently loses to buy-and-hold in testing. Trading involves substantial risk
of loss, and leveraged trading can lose more than the amount deposited. Past
performance, including any backtest in this repository, does not indicate future
results — and a backtest is a much weaker claim than live performance, because
it is fitted to data that already happened.

Paper trade for at least a month and review every decision the system makes
before considering real money. If you do deploy capital, deploy only money you
can lose entirely without it affecting anything you have planned.

The authors accept no liability for financial losses arising from use of this
software. You are responsible for your own trades.
