# Open questions

Things needing an answer from you, in blocking order.

---

## 1. Alpaca API keys (blocks Phase 6)

Not a decision, just a step you have to do rather than me. Sign up at
alpaca.markets with the trading API, which lands you in a paper account by
default. Scroll to the API keys panel and generate a key. The secret is shown
once.

```bash
cp .env.example .env
# then fill in ALPACA_API_KEY and ALPACA_SECRET_KEY
```

`.env` is gitignored and `tests/test_orders.py` scans the repo for anything
shaped like a live Alpaca key on every run. Never paste keys into a chat window,
including to me. The base URL is not confidential.

Phases 2 through 5 do not need this. Phase 6 does.

---

## 2. Starting capital (blocks Phase 6, affects Phase 4)

`config/settings.yaml` has `backtest.initial_capital: 100000`, the tutorial's
value. CLAUDE.md says to use the real intended starting capital instead.

The reason it matters: at $100k, 1% risk per trade is $1,000 and costs are a
rounding error. At $5k it is $50 per trade, and the minimum viable position size
starts to bind against the 15% single-position cap. Backtesting at $100k and
paper trading at $5k means the backtest answers a question about a different
account than the one you have, and it errs in the flattering direction.

Given the HDB purchase, this should be money that can be down 30% without
mattering.

**Needed: a real number**, for `settings.yaml` and for the Alpaca paper account.

---

## 3. Leverage: keep 1.25x or revert to 1.0? (affects Phase 3 and 5)

The tutorial applies 1.25x in low-volatility regimes. Your CLAUDE.md lists
"Leverage: None. Cash equities only." under hard constraints. Both are currently
true in different files, which is why `settings.yaml` ships the tutorial value
with the conflict marked at the line.

Full reasoning in `docs/TUTORIAL-CONFLICTS.md`. Short version: leverage gets
applied in calm markets, which is also where a shock is least priced in, and it
moves a 20% drawdown to 25%.

Nothing needs deciding to run on paper. Worth deciding before Phase 4 so the
backtest measures the version you intend to run.

**Needed eventually: 1.25 or 1.0.** One line either way.

---

## 4. Circuit breaker thresholds (settle with Phase 4 output)

Shipped at the tutorial's values: daily -2%/-3%, weekly -5%/-7%, peak -10%.
CLAUDE.md argues these are day-trading numbers that will misfire on a
days-to-months hold, and it is right that SPY draws down 10% in ordinary years.

This one does not need an answer now. It needs the backtester. Phase 4 can run
both threshold sets and report how often each fires and what it costs, which
turns an argument into a measurement. Alternatives are in
`docs/TUTORIAL-CONFLICTS.md`.

---

## 5. Symbol universe (affects Phase 2 onward)

`settings.yaml` ships the tutorial's ten: SPY, QQQ, AAPL, MSFT, AMZN, GOOGL,
NVDA, META, TSLA, AMD.

Worth knowing what that list is: eight of the ten are large-cap US tech, and
SPY and QQQ both hold most of the other eight. It is close to one bet on tech
expressed ten ways, which is also what makes the Phase 5 correlation check load
bearing rather than theoretical.

That is fine for building and testing the machinery. It is worth revisiting
before the results mean anything, because a regime model tested only on
2015-2025 US large-cap tech has seen one very particular kind of market.

**Needed eventually: whether to broaden this.** Not blocking.

---

## 6. Where the daily run happens (blocks Phase 7)

A daily-bar system evaluates at 4pm New York, which is 4am or 5am Singapore
depending on daylight saving. The Mac will be asleep.

Options: a cheap always-on VPS, a scheduled cloud job, or evaluating on the
previous close each morning Singapore time. The last is nearly free and costs
very little on a days-to-months horizon, but it should be a decision rather than
something that happens by default.

**Needed before Phase 7.**

---

## Resolved

**Python 3.14 wheel availability.** Was a real risk; everything installs.
hmmlearn 0.3.3 builds from source cleanly, scikit-learn 1.9.0, scipy 1.18.1,
pandas 3.0.5, numpy 2.5.3, alpaca-py 0.44.0, ta 0.11.0 all fine. One thing to
remember rather than fix: pip resolves pandas to 3.x, where copy-on-write is the
default. Tutorial code relying on chained assignment to mutate a DataFrame will
silently no-op instead of working. If a transformation behaves oddly from Phase
2 onward, check this first.

**Market data source.** Alpaca, via `data/market_data.py`. Note the free plan
serves the IEX feed only, a fraction of consolidated volume, so free-tier bars
can differ from a charting site. Fine for paper and for regime detection on
liquid names. Worth revisiting before results are treated as evidence.

**QuiverQuant.** Out of scope for this build, it comes from a different
tutorial. The MCP server is connected but the account has no subscription. Notes
in `~/Desktop/TradingQuiverQuant/`.
