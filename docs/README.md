# Documentation

## Start here

| File | What it answers |
|---|---|
| [`../README.md`](../README.md) | What the system is, how to run it, what the numbers mean |
| [`../CLAUDE.md`](../CLAUDE.md) | The project's hard constraints and decisions |
| [`../PROGRESS.md`](../PROGRESS.md) | Where the build stands, phase by phase |

## Decisions and open items

| File | What it answers |
|---|---|
| [`TUTORIAL-CONFLICTS.md`](TUTORIAL-CONFLICTS.md) | Where the tutorial's shipped defaults contradict the project's own constraints. Read before funding anything. |
| [`OPEN-QUESTIONS.md`](OPEN-QUESTIONS.md) | Decisions still waiting on the operator |
| [`EXPERIMENT-LOG.md`](EXPERIMENT-LOG.md) | One line per strategy variant tested, and its result |

## Phase notes

Written as each phase landed. Each one records the design decisions, the bugs
found, and — more usefully — the bugs that only appeared by running the thing
rather than reasoning about it.

| Phase | Notes | The thing worth reading it for |
|---|---|---|
| 2 | [`phases/02-hmm-engine.md`](phases/02-hmm-engine.md) | Why 14 features overparameterise the model, and the warmup arithmetic |
| 3 | [`phases/03-allocation-strategies.md`](phases/03-allocation-strategies.md) | The stop clamp: raw formulas put the stop above entry on 25-36% of bars |
| 4 | [`phases/04-walk-forward-backtest.md`](phases/04-walk-forward-backtest.md) | The first out-of-sample result, and the verdict that still stands |
| 5 | [`phases/05-risk-layer.md`](phases/05-risk-layer.md) | Why the gap rule binds before the risk limit, making real risk 0.667% |
| 6 | [`phases/06-alpaca-broker.md`](phases/06-alpaca-broker.md) | A timezone bug that would have been silent on a paid data plan |
| 7 | [`phases/07-main-loop.md`](phases/07-main-loop.md) | Why exits skip the risk veto, and why `peak_equity` must survive a restart |
| 8 | [`phases/08-monitoring-dashboard.md`](phases/08-monitoring-dashboard.md) | A real bug in Python's own log rotation |
| 9 | [`phases/09-integration-and-docs.md`](phases/09-integration-and-docs.md) | What the integration tests actually prove, and what they cannot |
