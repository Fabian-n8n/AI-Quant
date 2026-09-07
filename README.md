# regime-trader

A rules-based trading system that detects the current market regime with a
hidden Markov model, sizes portfolio allocation to match, and places orders
through Alpaca. Paper first.

**Status: Phase 7 of 8 complete.** Regime engine, allocator, backtester, risk
layer, Alpaca integration and the main loop all working. 374 tests passing, 3 of
them against the live paper API. Phase 8 is the dashboard UI.

**Current out-of-sample verdict: no demonstrated edge.** It loses to
buy-and-hold and to random allocation, and its drawdown trips the peak circuit
breaker 5% of the way into the backtest. The system runs unattended and is
correct about what it does; that is not the same as it working. See
`docs/EXPERIMENT-LOG.md`.

---

## The system in five parts

1. **Brain** (`core/hmm_engine.py`) - classifies the market into crash / bear /
   neutral / bull / euphoria from price and volume. It does not predict prices.
2. **Allocation** (`core/regime_strategies.py`) - decides how much capital that
   kind of market deserves. Calm markets get more, turbulent ones get less.
3. **Safety** (`core/risk_manager.py`) - circuit breakers with absolute veto,
   firing on actual P&L so they still work when the model is wrong.
4. **Broker** (`broker/`) - Alpaca places the orders.
5. **Orchestration** (`main.py`) - `TradingEngine`: startup, the bar loop,
   shutdown, session recovery.
6. **Monitoring** (`monitoring/`) - structured JSONL logs, rate-limited alerts,
   and a terminal dashboard. The web UI is Phase 8.

---

## Read these first

| File | What it is |
|---|---|
| `CLAUDE.md` | Source of truth for project decisions. Read fully before any phase. |
| `BUILD-REFERENCE.md` | Phase-by-phase spec. Read only the phase you are on. |
| `PROGRESS.md` | Where the project stands. |
| `docs/TUTORIAL-CONFLICTS.md` | Where the tutorial's defaults contradict your constraints. |
| `docs/OPEN-QUESTIONS.md` | What needs a decision from you. |
| `docs/EXPERIMENT-LOG.md` | One line per strategy variant tested. |
| `docs/PHASE2-NOTES.md` | HMM engine: capacity limits, warmup, look-ahead defences. |
| `docs/PHASE3-NOTES.md` | Allocation: the stop clamp, label vs volatility. |
| `docs/PHASE4-NOTES.md` | Backtester: the first out-of-sample result. |
| `docs/PHASE5-NOTES.md` | Risk layer: breakers, gap sizing, the 13-step cascade. |
| `docs/PHASE6-NOTES.md` | Alpaca: the timezone bug, adjusted vs raw prices. |
| `docs/PHASE7-NOTES.md` | Main loop: why exits skip the veto, why the peak must persist. |

---

## Layout

```
main.py      TradingEngine and the CLI
config/      settings.yaml, every tunable parameter and nothing else
core/        hmm_engine, regime_strategies, risk_manager, signal_generator
broker/      alpaca_client, order_executor, position_tracker
data/        market_data, feature_engineering, cache/
monitoring/  logger, dashboard, alerts
backtest/    backtester, performance, stress_test
tests/       one file per subsystem, plus test_look_ahead.py
```

Runtime files, all gitignored: `models/hmm_model.pkl`, `logs/events-*.jsonl`,
`state_snapshot.json`, and `trading_halted.lock` when a breaker has halted the
system.

Two rules that hold everywhere:

- **Every threshold lives in `config/settings.yaml`.** A number hardcoded in a
  module is invisible to review and to the backtester.
- **Broker-specific types stay inside `broker/`.** That is what keeps a later
  move to IBKR a contained job instead of a rewrite.

---

## Setup

```bash
cd "~/Desktop/AI Quant Trading"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your Alpaca keys; .env is gitignored

python main.py --status     # build state, touches nothing
python main.py --dry-run --once   # full pipeline against your account, no orders
pytest -q                   # 374 passing, 2 phase-gated skips
```

### Running it

```bash
python main.py                     # paper trading, long-running
python main.py --dry-run           # full pipeline, no orders can be placed
python main.py --dry-run --once    # one bar, then exit
python main.py --once              # one bar for real, the intended cron entry
python main.py --train-only --symbols SPY
python main.py --dashboard         # read the running instance's state
```

`--dry-run` replaces the order executor with one whose every method raises, so
"no orders were placed" is a property of the object graph rather than of
remembering to check a flag.

The system runs on **daily** bars. `--timeframe` accepts anything Alpaca does,
but every feature is calibrated for daily (252-day z-scores, 200-bar SMA, 50-bar
EMA stops) and passing something else warns and points at
`docs/TUTORIAL-CONFLICTS.md` section 5.

Daily bar close is 4pm New York, which is 4am or 5am in Singapore depending on
US daylight saving. Schedule `--once` accordingly.

### Backtesting

```bash
python main.py --mode backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
python main.py --mode backtest --symbols SPY --compare --export
python main.py --mode backtest --stress-test --mc-sims 100 --misclassification
```

`--compare` runs all three benchmarks and prints a pass/fail verdict.
`--export` writes equity_curve, trade_log, regime_history and
benchmark_comparison CSVs to `backtest/results/<symbol>/`.

Alpaca keys: log in at alpaca.markets, scroll to the API keys panel, generate a
key. The secret is shown once. Never paste keys into a chat window.

### If it halts

A circuit breaker writes `trading_halted.lock` and the system will reject every
signal until that file is deleted **by hand**. The friction is deliberate: it
forces someone to look at what broke before the system can lose more, and it
cannot be cleared by a restart or a scheduled job.

---

## The point of all this

The build is the easy part. Validation is the hard part.

A system is not validated because the tests pass. Tests verify the plumbing.
The bar is 30 to 50 closed trades, walk-forward tested, beating buy-and-hold,
200-day SMA trend following and random allocation, on both return and Sharpe,
net of slippage, with positive expectancy per trade.

Then paper trade for at least a month and review every decision it makes before
any real money is involved.
