# Phase 9 notes: integration testing and documentation

Built 2026-09-07. **437 tests pass, 3 skipped**, 4 against the live Alpaca paper
API.

---

## 1. What integration tests are for

The unit suites prove each part behaves. These prove the parts behave
*together*, which is a different claim. Every look-ahead test in
`test_look_ahead.py` can pass while the live loop still leaks the future,
because the loop assembles the pieces differently from the way the unit tests
exercise them.

`tests/test_integration.py` has 19 tests in the spec's five groups.

### a. End-to-end dry run

`data → features → HMM → strategy → risk → simulated orders`, asserting at every
hand-off. A chain only ever tested link by link can be broken at every joint and
still pass.

The final assertion is the invariant the whole system exists to hold:

```python
risk_dollars = shares * abs(signal.entry_price - signal.stop_loss)
assert risk_dollars <= portfolio.equity * risk.max_risk_per_trade
```

A second test runs the actual CLI as a subprocess. `--dry-run` has to be a
property of the shipped entry point, not of a code path a test happened to take.

### b. Look-ahead bias

Three tests. The dedicated `test_look_ahead.py` suite is run as a subprocess
gate, per the spec naming it explicitly.

The sharper one is the spec's own rule: **features recomputed with 120 fewer
future bars must be identical over the overlap.** If any value moves, something
read a bar that had not happened yet.

The third asserts the forward algorithm's defining property: classifying a bar
with 200 later bars present gives the same answer as classifying it with them
absent. Viterbi would fail this by construction, because it revises the past in
light of the future.

### c. Risk stress

Extreme signals capped, rapid fire blocked, no-stop rejected — plus two the spec
does not name:

- **A one-cent stop cannot buy the universe.** Sizing divides by stop distance,
  so a stop a cent below entry produces a position larger than the account
  unless something catches it.
- **A sweep across six stop distances**, asserting no approval ever risks more
  than the configured fraction. One counterexample is a sizing bug.

Note that an absurd signal asking for 50x the account is *capped*, not rejected.
The risk layer's job is to shrink.

### d. Alpaca paper round trip

Places a real limit order 20% below the market so it rests rather than fills,
verifies it at the broker, cancels it, and asserts the account is left exactly as
it was found. Every test cleans up in a `finally` block — a test that leaves a
resting order behind changes the next run's behaviour, and on a broker that is a
test that lies.

Verified after the run: the only open order remaining was Phase 6's original
NVDA order.

### e. Recovery

The spec says "kill process, restart". So it does — `SIGKILL` to a real
subprocess, not a call to `shutdown()`. A graceful shutdown is not what recovery
has to survive.

`test_a_truncated_snapshot_does_not_restore_a_zero_peak` is the important one. A
snapshot cut short mid-write must be **rejected**, not partially read: restoring
`peak_equity: 0.0` would silently disarm the one breaker that never resets,
which is worse than having no snapshot at all.

---

## 2. Test inventory

| Suite | Tests | Covers |
|---|---|---|
| `test_hmm.py` | 40 | model selection, forward filter, labelling, persistence |
| `test_look_ahead.py` | 15 | the central failure mode, AST checks included |
| `test_features.py` | 18 | rolling transforms, warmup arithmetic |
| `test_strategies.py` | 52 | allocation, stop clamping, volatility mapping |
| `test_risk.py` | 69 | the 13-step cascade, breakers, gap sizing |
| `test_orders.py` | 53 | broker layer, offline fakes plus 3 live |
| `test_backtest.py` | 40 | walk-forward, metrics, benchmarks |
| `test_orchestration.py` | 89 | the main loop, recovery, error handling |
| `test_monitoring.py` | 45 | logging, alerts, dashboards, publishing |
| `test_integration.py` | 19 | the five groups above |
| **total** | **440** | 437 pass, 3 conditionally skipped |

The three skips are conditional: two require an open position to attach a stop
to, one requires credentials.

---

## 3. Documentation

`README.md` was rewritten to the spec's structure: philosophy, architecture
diagram, six-step quick start, CLI reference, configuration guide, FAQ,
disclaimer.

Two departures worth noting.

**The verdict is at the top, in a callout, before anything else.** The spec puts
the disclaimer at the bottom. A reader who stops after the first screen should
still know the strategy loses to buy-and-hold in testing — burying that under an
architecture diagram would be technically compliant and actually misleading.

**The FAQ answers the questions this system actually generates**, not generic
ones. "Why is my risk per trade 0.67% when I configured 1%" is the question
anyone running this will ask on day one, because the gap rule binds before the
risk limit does and the discrepancy looks like a bug. It is not, and the answer
is three sentences.

`settings.yaml` was already documented per-key from Phase 1 and gained an
`orchestration` block in Phase 7 and alert-delivery keys in Phase 8. The four
`CONFLICT` markers stay: they flag where the tutorial's shipped defaults
contradict the project's own constraints, and removing them would be quietly
losing an argument nobody had.

---

## 4. Where the project actually stands

**Built and tested:** all nine phases. 437 tests. The pipeline runs end to end
against a real paper account, recovers from a SIGKILL without double-entering,
places and cancels real orders, and reports itself honestly on two dashboards.

**Not demonstrated:** that any of it makes money.

The validation checklist in the README has ten items and **none of them pass**.
The system loses to buy-and-hold and to random allocation out of sample, its
drawdown trips the peak breaker about 5% into the backtest, and it has zero
closed trades against a bar of 30 to 50.

That is not a defect in the implementation. It is the honest state of a strategy
that has been built correctly and has not yet been shown to work. The next
useful work is not another phase — it is the allocation layer, starting with
whether 1.25x leverage belongs in a system whose own risk config caps gross
exposure at 80%.
