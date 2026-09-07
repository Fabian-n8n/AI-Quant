# Build Reference

Phase-by-phase technical spec. Read the relevant phase before starting it; do not
load this whole file into context unnecessarily.

Follows the AI Pathways tutorial "How To Actually Build a Trading Bot With Claude
Code". Phase numbers match the tutorial's, so a screenshot of a tutorial prompt
maps here without translation.

Where this departs from the tutorial, it is noted and the reason is given.

---

## File structure (Phase 1, done)

```
AI Quant Trading/                  # repo root, project name "regime-trader"
├── config/
│   ├── settings.yaml              # ALL configurable parameters
│   ├── credentials.yaml.example
│   └── __init__.py                # settings and credential loaders
├── core/
│   ├── __init__.py
│   ├── hmm_engine.py              # HMM regime detection engine
│   ├── regime_strategies.py       # vol-based allocation strategies
│   ├── risk_manager.py            # position sizing, leverage, drawdown limits
│   └── signal_generator.py        # combines HMM + strategy into signals
├── broker/
│   ├── __init__.py
│   ├── alpaca_client.py           # Alpaca API wrapper
│   ├── order_executor.py          # order placement, modification, cancellation
│   └── position_tracker.py        # track open positions, P&L
├── data/
│   ├── __init__.py
│   ├── market_data.py             # real-time and historical data fetching
│   ├── feature_engineering.py     # technical indicators, feature computation
│   └── cache/
├── monitoring/
│   ├── __init__.py
│   ├── logger.py                  # structured logging
│   ├── dashboard.py               # terminal + Streamlit dashboard
│   └── alerts.py                  # email/webhook alerts for critical events
├── backtest/
│   ├── __init__.py
│   ├── backtester.py              # walk-forward allocation backtester
│   ├── performance.py             # Sharpe, drawdown, regime breakdown, benchmarks
│   ├── stress_test.py             # crash injection, gap simulation
│   └── results/
├── tests/
│   ├── conftest.py
│   ├── test_hmm.py
│   ├── test_look_ahead.py         # verify no look-ahead bias
│   ├── test_strategies.py
│   ├── test_risk.py
│   └── test_orders.py
├── docs/                          # decision records, not in the tutorial
├── main.py                        # entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── CLAUDE.md
├── BUILD-REFERENCE.md
├── PROGRESS.md
└── README.md
```

**Two small additions to the tutorial's tree**, both flagged where they appear:

- `config/__init__.py` holds the settings and credential loaders. The tutorial
  shows `config/` containing only the two YAML files, but something has to read
  them, and putting the loader here keeps file paths and yaml imports out of
  every other module.
- `docs/` holds decision records. Nothing depends on it.

Phase 1 creates structure only. No logic. Imports, class stubs, type hints and
docstrings, with each stub raising `NotImplementedError` naming the phase that
fills it in. A stub that returns `None` instead would let a later phase appear
to work while doing nothing.

---

## Phase 2 - HMM regime detection engine (DONE)

The brain. Files: `core/hmm_engine.py`, `data/feature_engineering.py`.

It does not predict prices. It classifies what kind of market this is, from
price action and volume, into calm / moderate / turbulent. Everything downstream
keys off that.

Implemented. Read `docs/phases/02-hmm-engine.md` before changing anything here; three
issues with the spec were found and worked around, and the workarounds are not
obvious from the code alone.

**Key API:**

```python
from core.hmm_engine import HMMEngine
from data.feature_engineering import build_feature_matrix, log_returns

features = build_feature_matrix(bars)            # 14 features, rolling z-scored
returns  = log_returns(bars["close"], 1)

engine = HMMEngine(**settings["hmm"])
engine.fit(features, returns)                    # BIC selection + labelling
engine.summary()                                 # read this after every refit

engine.classify_series(features)                 # backtest: causal, whole series
engine.classify(features, as_of=timestamp)       # live: one bar
engine.make_live_filter()                        # O(1) per-bar stepper
engine.get_volatility_rank(state_id)             # what Phase 3 keys off
engine.save() / HMMEngine.load()
```

**The three things to know:**

1. **Capacity.** All 14 features with full covariance needs ~1,800 rows. At the
   504-row minimum it is overparameterised and BIC collapses onto 3 states
   regardless of the data. `hmm.feature_columns` defaults to a 6-column
   volatility subset. `check_fittability()` raises rather than fitting garbage.
2. **Warmup.** 504 usable feature rows require 954 raw bars, not 504. Use
   `required_raw_bars()`. Phase 4 must budget for this when sizing windows.
3. **Look-ahead.** `model.predict()` is never called; the forward algorithm is
   implemented explicitly. The spec's mandatory test has an off-by-one that lets
   it pass for the wrong reason; the corrected and strengthened versions are in
   `tests/test_look_ahead.py`.

**Two size reductions, independent:** transition damping (0.75x while a change
is unconfirmed) and uncertainty mode (0.50x while flickering). Smaller wins.
Flicker counts raw changes, not confirmed ones.

**Labels sort by return; the strategy sorts by volatility.** Crash and euphoria
are opposite on the first ordering and adjacent on the second. Phase 3 must call
`get_volatility_rank()`, never switch on the label.

---

## Phase 3 - Volatility-based allocation strategies (DONE)

Files: `core/regime_strategies.py`, `core/signal_generator.py`.

The brain says what kind of market it is; this decides how much capital that
market deserves. Read `docs/phases/03-allocation-strategies.md` before changing anything here.

| Tier | Vol rank | Allocation | Leverage | Stop |
|---|---|---|---|---|
| `LowVolBullStrategy` | <= 0.33 | 95% | 1.25x | `max(price - 3ATR, EMA50 - 0.5ATR)` |
| `MidVolCautiousStrategy` | 0.34-0.66 | 95% above 50 EMA / 60% below | 1.0x | `EMA50 - 0.5ATR` |
| `HighVolDefensiveStrategy` | >= 0.67 | 60% | 1.0x | `EMA50 - 1.0ATR` |

**Always long, never short.** `Direction` has no SHORT member, so adding it is a
code change rather than a parameter. Shorting destroyed returns in walk-forward
testing: markets drift up, V-shaped recoveries are fast, and the HMM is 2-3 days
late detecting them.

**Mapped by measured volatility, never by label.** On the current fit all 7 of 7
regimes sit in a different position under the two sorts, and `strong_bull` is the
most volatile regime of all. `LABEL_TO_STRATEGY` exists as a fallback for callers
holding a bare label; the orchestrator never consults it.

**Stops are clamped.** The spec's raw formulas return a stop above entry on
25-36% of bars. Every stop is forced at least 0.5 ATR and 0.5% below entry. The
raw value is kept in `metadata["raw_stop"]`.

**Uncertainty mode** fires on low confidence, flickering, or an unconfirmed
regime. Halves size, forces leverage to 1.0x, appends the tag.

**Known conflict:** LowVol gross exposure is 118.75% against `max_exposure: 80%`.
Phase 5 will clamp it. Decide before Phase 4 measures anything.

```python
orchestrator = StrategyOrchestrator(settings["strategy"], engine.regime_info)
signals = orchestrator.generate_signals(symbols, bars, regime_state)
orchestrator.update_regime_infos(engine.regime_info)   # after EVERY HMM refit
```

---

## Phase 4 - Walk-forward backtesting and validation (DONE)

Files: `backtest/backtester.py`, `performance.py`, `stress_test.py`.

**Allocation-based**, not trade-based. It sets a target portfolio allocation each
bar and rebalances when the target drifts more than 10%. A "trade" is a
rebalance-to-rebalance holding period, not a round trip.

Read `docs/phases/04-walk-forward-backtest.md` before changing anything here.

```bash
python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
python main.py backtest --symbols SPY --compare --export
python main.py backtest --stress-test --mc-sims 100 --misclassification
```

Windows: 252 IS / 126 OOS / 126 step. Per fold: train the HMM on IS, walk OOS
bar by bar with forward-algorithm inference, rebalance at the **next bar's
open**.

Allocation math, exact:
```
equity        = cash + shares * price
target_shares = int(equity * target_allocation / price)
delta         = target_shares - shares
cash         -= delta * fill_price          # fill_price includes slippage
```
`target_allocation` above 1.0 drives cash negative. That is margin and it is
left alone; clamping it would cap leverage at 1.0 silently.

**Three things to know:**

1. **The IS window is below `min_train_bars`.** 252 < 504, so the backtester
   lowers the floor and logs it. A 252-bar fit biases BIC toward fewer states.
2. **Use `engine.stream()`, never `classify()` in a loop.** The latter is O(T)
   per bar and the backtest will not finish.
3. **Zero-volatility guard.** Flat equity is normal and used to produce a Sharpe
   of 1e17. `ZERO_VOL_TOLERANCE` in `performance.py`.

**Current verdict: no demonstrated edge on synthetic data.** Loses to
buy-and-hold and random allocation. Not evidence about SPY, but proof the
validation framework works. See `docs/EXPERIMENT-LOG.md` variant 1.

---

## Phase 5 - Risk management layer

File: `core/risk_manager.py`.

The most important file in the system, more important than the HMM. A mediocre
strategy with good risk management loses slowly. A good strategy with bad risk
management blows up the account.

**Independence.** Circuit breakers fire on actual P&L, never on model state, so
they still work when the HMM is confidently wrong, which is exactly what they
are for. The risk manager may log what the regime was; it may never ask the
regime engine for permission to fire.

**Absolute veto.** Every signal passes through `evaluate()`. Nothing routes
around it.

**Portfolio limits** (`settings.yaml`, `risk:`):
`max_exposure: 0.80`, `max_single_position: 0.15`, `max_concurrent: 5`,
`max_daily_trades: 20`, `max_leverage: 1.25`.

**Circuit breakers:** daily -2% reduce / -3% halt, weekly -5% reduce / -7% halt,
peak -10% halt and write `trading_halted.lock`. The lock requires manual
deletion to resume; the friction is the point. These are the tutorial's values
and several are day-trading numbers on a swing horizon. See
`docs/TUTORIAL-CONFLICTS.md` and settle it with Phase 4 output.

**Position sizing:**
```
position_size = (portfolio * max_risk_per_trade) / abs(entry - stop_loss)
```
Risk a fixed 1% per trade. A wide stop means fewer shares, a tight stop more, so
every trade loses roughly the same dollar amount when wrong. Then cap at
`max_single_position`, apply the active breaker's multiplier, and enforce a
minimum size floor so commission drag stays reasonable.

**Correlation check.** Five positions in NVDA, AMD, META, GOOGL and MSFT is not
five positions, it is one bet on large-cap tech wearing five hats. The
concurrent-position limit does nothing about this on its own.

**Every position must have a stop.** Orders without one are rejected, here and
again in the order executor.

**Log every breaker trigger** with type, drawdown, equity, positions closed, and
the regime at the time.

---

## Phase 6 - Alpaca broker integration

Files: `broker/alpaca_client.py`, `broker/order_executor.py`,
`broker/position_tracker.py`, `config/__init__.py`.

Deliverable: a test order visible in the Alpaca paper dashboard.

**Getting keys:** log in at alpaca.markets, scroll to the API keys panel,
generate a key. Copy the key, the secret and the base URL. The secret is shown
once. Put them in `.env`, which is gitignored. Never paste keys into a chat
window; the base URL is not confidential and can be typed anywhere.

`.env` shape is in `.env.example`:
```
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_PAPER=true
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**Use `alpaca-py`, not `alpaca-trade-api`.** The tutorial lists both. The latter
is deprecated and pins old `websockets` and `urllib3`, downgrading what
`alpaca-py` installs. See `docs/TUTORIAL-CONFLICTS.md`.

Implement: connect, account equity and buying power, market clock, submit /
cancel / modify order, fetch positions, close position, historical bars.

Keep the wrapper thin. It translates Alpaca's types into the dataclasses in
`alpaca_client.py` and does nothing else. Broker-specific objects escaping this
file is what would make a later IBKR port a rewrite instead of a contained job.

**Stops go on immediately after the entry fills.** A position that exists
without its stop, even for one cycle, is an unbounded loss.

Verify: account connects, equity reads correctly, a market buy queues, and it
appears in the Alpaca web dashboard. Paper orders queue outside market hours, so
this is testable on a weekend.

Set the paper account to the real intended starting capital, not the default
$100k. See `docs/OPEN-QUESTIONS.md`.

---

## Phase 7 - Main loop and orchestration

Startup: load config, connect Alpaca, verify the account, check the halt lock,
check market hours, fit the HMM, initialise the risk manager and position
tracker, start data feeds.

Main loop, on **daily bar close** (not 5-minute; this is a swing system):

1. check the halt lock, exit immediately if set
2. reconcile positions against the broker, stop on any discrepancy
3. classify the regime
4. check exits on open positions before considering entries
5. generate allocation signals
6. pass every signal through the risk manager
7. execute the survivors
8. log everything, including rejected signals and why

Exits before entries frees slots and capital in the same cycle.

**Reconciliation.** Local state and Alpaca will drift: partial fills, manual
intervention in the web dashboard, rejected orders, corporate actions. When they
disagree, the broker is right and the system stops and alerts. Never guess,
never auto-correct into the market.

**Error handling** for broker outages, data feed drops and partial fills. On any
unrecoverable error, stop and alert. Never retry blindly: a retry loop during an
outage is how you end up with four copies of the same position.

---

## Phase 8 - Monitoring, alerts and dashboard

Files: `monitoring/logger.py`, `monitoring/alerts.py`, `monitoring/dashboard.py`.

**Structured logging**, not free text, because these logs get queried later.
"Why did the system not take that trade on the 14th" is only answerable if
rejections were logged as records with fields. Always log: every signal
including rejections and why, every order and its broker outcome, every breaker
trigger with the regime at the time.

**Alerts** on things needing a human: a breaker halted the system, reconciliation
found a mismatch, the broker is unreachable, a position has no stop, the loop
died. Rate limited by `alert_rate_limit_minutes: 15`. An alert firing ten times
a day gets muted within a week, and the one that mattered gets muted with it.

**Dashboard**, terminal via `rich` and web via Streamlit. Panels:
regime and confidence, portfolio value and buying power, state count and active
positions, price chart with regime overlay, volume and confidence over time,
regime distribution, signal feed with allocation / entries / stops / live P&L,
and risk controls showing breaker states, drawdown levels and leverage.

Read-only by construction. The dashboard must never be able to place an order or
clear a breaker. A display that can act is no longer a display, and a stray
click should not move real money.

Streamlit is installed at this phase, not before. It pulls a large dependency
tree and contributes nothing to whether the system works.

Last for a reason. It is the most enjoyable part to build.

---

## Testing note

The tutorial celebrates "134 tests passing". Those tests verify the code runs:
the API connects, the sizing math computes, the breakers fire when called. They
say nothing about profitability.

Both kinds of verification are needed. Do not let green checkmarks stand in for
a walk-forward result.

Current: 35 passing, 28 skipped. The skips are phase-gated placeholders that
name what each later phase has to prove.
