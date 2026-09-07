# Phase 8 notes: monitoring, alerts and dashboard

Built 2026-09-07. 437 tests pass (63 new), 3 skipped, 4 live against the Alpaca
paper API.

---

## 1. What was built

| File | What it does |
|---|---|
| `monitoring/logger.py` | four rotating JSONL streams, per-bar context |
| `monitoring/alerts.py` | the spec's seven triggers, rate limited per type |
| `monitoring/dashboard.py` | `DashboardState` + six-panel rich terminal view |
| `monitoring/publish.py` | snapshot → JSON for the web UI, plus demo data |
| `dashboard/` | Next.js static-export web dashboard |
| `tests/test_monitoring.py` | 45 tests |

---

## 2. A real bug in Python's own log rotation

The spec asks for "rotating files (10MB, 30 days)". Python ships
`RotatingFileHandler` (size) and `TimedRotatingFileHandler` (time) and nothing
that does both, so `SizedTimedRotatingHandler` subclasses the timed one and adds
a size check.

That worked once and then silently stopped. **Python 3.14's
`TimedRotatingFileHandler.doRollover()` begins:**

```python
dfn = self.rotation_filename(self.baseFilename + "." +
                             time.strftime(self.suffix, timeTuple))
if os.path.exists(dfn):
    # Already rolled over.
    return
```

The dated backup name is identical for every rotation on a given day. So after
the first rotation of the day, **every subsequent size-triggered rotation is a
no-op** and the file grows without bound until midnight.

Measured on a 400-byte cap: `main.log` reached **7,288 bytes**, and the handler
reported success the whole way. Scaled to the real 10MB bound on a day when the
broker is throwing errors, that is a full disk.

`doRollover` is now reimplemented. Same-day rotations get a `.1`, `.2` suffix so
each lands somewhere new, and pruning sorts by mtime rather than matching the
parent's date regex — which the suffixed names do not satisfy, so the parent
would never have pruned them either. After the fix, the same test ends at 384
bytes against the 400-byte cap.

Two regression tests: `test_rotation_actually_rotates` asserts the bound holds,
and `test_same_day_rotations_do_not_overwrite_each_other` asserts each rotation
lands on a distinct file. Both fail against the stdlib behaviour.

---

## 3. Context belongs to the moment, not the event

The spec requires every log entry to carry timestamp, regime, probability,
equity, positions and daily_pnl. None of those are properties of the event being
logged; they describe the moment it happened.

Passing six arguments at forty call sites would mean the first one anyone forgot
is the one that mattered. `logger.set_context(...)` is called once per bar and
every record written afterwards carries the fields. Explicit fields still win,
so a signal that knows a better regime value can say so.

---

## 4. Alert severity is not the same as the spec's trigger list

All seven triggers are implemented. Two of them should not be able to send you
email:

| Trigger | Level | Why |
|---|---|---|
| Circuit breaker | CRITICAL | needs manual intervention to clear |
| Data feed down | CRITICAL | trading blind |
| API lost | CRITICAL | orders may not be arriving |
| Large P&L | WARNING | worth knowing today |
| Flicker exceeded | WARNING | size is already cut automatically |
| **Regime change** | **INFO** | this is the system working |
| **HMM retrained** | **INFO** | routine, weekly |

INFO alerts reach console and log only. A regime change is a normal event on a
system whose entire job is detecting regime changes; routing it to email means
an inbox that receives a notification every few days, and an inbox like that
stops being read. When the breaker finally fires, it arrives in a thread nobody
opens. `alert_min_level` in settings controls the threshold.

One exception: a retrain that lands on a **different number of states** is
escalated to WARNING. The allocator's regime map has been redrawn, and that is
not routine.

**Rate limiting is keyed on the trigger, not the message.** "Flicker 5/20" and
"flicker 6/20" are the same condition. Keying on the message would let a value
that changes every bar defeat the limiter entirely.

---

## 5. The web dashboard holds no credentials

The Phase 8 spec and the tutorial describe a Streamlit dashboard. This project
ships a Next.js one, for two reasons.

**Streamlit cannot deploy to Vercel.** It needs a persistent Python server.
Vercel is where the user asked for this to live.

**More importantly, the deployment must not be able to reach the broker.** The
alternative design — serverless routes calling Alpaca with credentials in Vercel
env vars — would mean a public URL that can read a live account, and a dashboard
holding broker handles is one bad deploy away from being the reason the trading
process died.

So the split is:

```
your machine                          Vercel
─────────────                         ──────
engine + Alpaca keys
  │
  └─ python main.py --publish
       writes state.json  ──────────▶ static Next.js page reads it
                                      no keys, no broker, no order path
```

`monitoring/publish.py` strips a denylist of keys (`api_key`, `account_number`,
`order_id`, `trade_id`, `lock_file`, `traceback`) on the way out, and
`test_published_payload_carries_no_secrets` asserts none of them survive.
`lock_file` is on that list because it is an absolute path containing the
operator's home directory.

**Demo data is labelled.** A fresh clone shows a working interface rather than
six empty panels, but everything it renders is marked `source: "demo"` and sits
behind a banner saying so. Fabricated numbers presented as a real account would
be worse than an empty page.

---

## 6. Design decisions in the UI

Direction came from the `ui-ux-pro-max` skill's design-system query (dark OLED
style, Inter, dense 8-32px spacing, subtle 300-400ms motion), with the accent
swapped from its suggested green to purple as requested.

**Near-black with a violet cast (`#09070f`), not pure black.** The accent then
reads as belonging to the surface rather than floating on it.

**Semantic colours are deliberately not purple.** Accent means "this is the
system's state"; green and red mean money. Collapsing those into one hue is how
a dashboard stops being readable at a glance.

**Risk bars are graded on the fraction of the limit consumed, not on the
drawdown.** A 2% drawdown against a 3% halt is two thirds spent and reads
amber; the same 2% against a 10% peak limit has room and reads green. Grading
the raw number would colour both identically, which is backwards. Same rule in
the terminal view and the web view, tested in both.

**No charting library.** The equity chart is one SVG path plus a handful of
rects. Pulling in 40kB of recharts to draw that would have been a larger
dependency than the rest of the app. Total first-load JS is 93.5kB.

**Tabular figures everywhere.** Proportional digits make a column of currency
impossible to scan, and every number on this page exists to be compared with the
one above it.

**Inline SVG icons, never emoji.** Emoji render differently per platform and
cannot inherit colour. (The terminal dashboard is the exception — the spec draws
`✅` and rich renders it consistently.)

The `no demonstrated edge` verdict is on the page, not buried in a doc. A
monitoring surface that looks confident about a strategy that loses to
buy-and-hold is misleading by omission.

---

## 7. Both dashboards read one snapshot

`DashboardState.snapshot()` is the only place any of these numbers are computed.
The terminal renderer and the web renderer both consume it and nothing else.

Adding a third surface means adding a renderer, never a second way of computing
the numbers. Two independent computations of "current drawdown" would eventually
disagree, and the one on screen when it mattered would be the wrong one.

Neither surface can trade. `DashboardState` takes no order executor at all, so
the capability is absent rather than merely unused — asserted by
`test_dashboard_state_has_no_way_to_trade`.
