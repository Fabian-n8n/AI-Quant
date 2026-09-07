# Phase 5 notes: the risk management layer

What was built, three contradictions in the configured limits, and the finding
that reframes Phase 4.

---

## THE FINDING: the breakers are not mis-set, the strategy is too volatile for them

Phase 4 measured the circuit breakers firing on 100% of stress simulations and I
concluded they needed retuning. Running the real backtest equity curve through
the actual Phase 5 breakers gives a sharper answer.

The strategy's out-of-sample equity curve has a **48.4% peak drawdown**. Against
that:

| Peak threshold | Trips at | Fraction of backtest survived |
|---|---|---|
| 10% (configured) | bar 97 of 1,890 | **5.1%** |
| 18% (the swing-horizon alternative in TUTORIAL-CONFLICTS) | bar 279 of 1,890 | 14.8% |

Both halt permanently, since the lock file requires manual deletion and no
automatic reset exists. Loosening the threshold buys 182 bars.

**Retuning the breakers is treating the symptom.** No drawdown-based halt at any
sane level survives a strategy that draws down 48%. This lines up with Phase 4's
other result, that the strategy loses to buy-and-hold and to random allocation,
and both point at the same place: the allocation layer, starting with the 1.25x
leverage that the low-volatility tier applies.

The right order of work is to fix the strategy until its drawdown is survivable,
then calibrate breakers against it. Doing it the other way round produces a
safety net tuned to accommodate a system that should not be running.

---

## 1. Leverage of 1.25x is unreachable

Leverage **is** gross exposure divided by equity. So:

- `max_leverage: 1.25` permits 125% gross exposure
- `max_exposure: 0.80` caps gross exposure at 80%

Both gates are checked, as the spec requires, and the exposure gate always binds
first. Nothing can exceed 0.80x. The entire rule "only low-vol regimes may use up
to 1.25x" is dead code, and `LowVolBullStrategy`'s leverage never reaches the
account.

The spec half-acknowledges this: "when using 1.25x leverage, the notional
exposure exceeds equity but the margin requirement stays within Alpaca's limits".
That sentence describes exposure above 100%, which the 80% cap forbids.

Resolving it is a decision, not a fix:

- Raise `max_exposure` above 1.25 to let leverage work, which contradicts
  CLAUDE.md's no-leverage constraint even more directly than it already does.
- Lower `max_leverage` to 0.80 so the two agree and the leverage rule is
  honestly absent.
- Leave it, and accept that leverage is documented but inoperative.

Given Phase 4 showed the strategy underperforming *with* the leverage in the
backtester (which has no risk manager and therefore applied it), the second
option looks best on current evidence. `test_configured_leverage_ceiling_is_unreachable`
fails if this is ever resolved, as a prompt to update this note.

---

## 2. The gap rule always binds, so risk per trade is 0.667%, not 1%

Overnight sizing caps the position so that a `gap_multiplier` (3x) gap-through
costs at most `gap_max_loss_pct` (2%) of equity:

```
gap_capped_shares = (equity * 0.02) / (3 * stop_distance)
                  = (equity * 0.00667) / stop_distance
normal_shares     = (equity * 0.01)  / stop_distance
```

0.667% is below 1%, so `min(normal, gap_capped)` always picks the gap-capped
figure. And this is a swing system, so **every** position is held overnight.

The consequence: `max_risk_per_trade: 0.01` never applies to anything. Real risk
per trade is 0.667% of equity. Worth knowing before someone raises the 1% to 1.5%
expecting a 50% increase in position size and gets nothing, because the binding
constraint is elsewhere.

Verified in `test_gap_cap_always_binds_so_real_risk_is_two_thirds_of_one_percent`,
which also asserts the worst case lands on exactly the configured 2%.

---

## 3. Breaker thresholds ship as specified, and they are day-trading numbers

Daily -2%/-3%, weekly -5%/-7%, peak -10%. CLAUDE.md argues these misfire on a
days-to-months horizon, and section above shows they do. They ship as the spec
gives them; `docs/TUTORIAL-CONFLICTS.md` has swing-horizon alternatives, and this
codebase now has the tooling to settle it with evidence.

---

## Design decisions worth knowing

**Breakers latch.** Once a daily breaker trips it stays tripped until
`reset_daily()`, even if equity recovers intraday. A breaker that un-fires on a
bounce is a lagging indicator, not a circuit breaker.

**Most severe wins.** `BREAKER_SEVERITY` gives a total ordering, so a daily
reduce firing on the same bar as a peak halt cannot soften it.

**The lock file survives a restart.** `is_halted()` reads the filesystem, not
memory, so a crash loop cannot silently resume trading. `reset_daily()` and
`reset_weekly()` deliberately do not clear it. `clear_halt()` exists only so
tests can clean up and is called nowhere in production paths.

**Drawdowns are computed properties, not stored fields.** A cached drawdown that
was not refreshed is how a breaker fails to fire on the one day it mattered.

**The regime is recorded but never read by a breaker.**
`test_breakers_never_read_regime_state` asserts that two identical drawdowns
with opposite regime context produce identical decisions. The regime is logged so
you can ask afterwards what the model believed when the loss happened.

**Correlation fails open.** With no price history, `check_correlation` returns
1.0 rather than rejecting. Refusing every trade when the data feed is thin turns
a data problem into an outage; the position, sector and exposure limits still
apply.

**Existing positions stay adjustable at the position limit.** Refusing to resize
something already held would trap the system at max positions with no way to
reduce risk.

**Exposure shrinks to fit rather than rejecting.** A position at 60% of the
requested size is better than none. It only rejects when the remaining room is
below the $100 minimum.

**The veto may shrink, never grow.** `test_approved_size_never_exceeds_the_request`
pins it.

---

## Phase 4 loop closed

`backtest/stress_test.evaluate_breakers` had its own copy of the threshold logic
because `RiskManager` did not exist yet, and Phase 4 flagged that two
implementations would drift. It now delegates to the real `CircuitBreaker`.

It also walks the curve bar by bar rather than testing the endpoint, because a
breaker that fired in month two and was later recovered from still fired, and an
endpoint test would miss it. Each evaluation gets a scratch lock file in a temp
directory so a simulated halt never writes into the repo.

---

## Test isolation

Every test in `tests/test_risk.py` uses an autouse fixture redirecting the lock
file into `tmp_path`. Without it, one breaker test would write
`trading_halted.lock` into the repo and every subsequent test and every real run
would refuse to trade until someone deleted it by hand.

That is the mechanism working exactly as designed, which is precisely why it has
to be isolated in tests.

---

## What is not built

The risk manager is not yet wired into anything. The backtester still runs the
unclamped Phase 3 allocations, so Phase 4's results describe a configuration the
risk layer would not permit.

Wiring it in is Phase 7's job (the orchestrator). Doing it now would change every
backtest number mid-stream, and the useful sequence is to settle the leverage and
exposure decisions first, then re-run, then wire.
