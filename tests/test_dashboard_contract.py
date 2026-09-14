"""
The contract between the Python publisher and the TypeScript dashboard.

Nothing else checks this. `monitoring/publish.py` writes JSON and
`dashboard/lib/types.ts` declares what the components read, and the two are
edited months apart in different languages. When they drift, the failure is not
an error — it is a panel quietly rendering `undefined`, which on a dashboard
looks like a zero, and a zero on a risk meter looks like safety.

So these tests parse the actual `.ts` file and assert every field it declares is
present in the actual published payload, for both the live and demo paths.
"""

import json
import re
from pathlib import Path

import pytest

from monitoring.publish import (
    demo_payload,
    publish_demo,
    regime_mix,
)

TYPES_TS = Path(__file__).resolve().parent.parent / "dashboard" / "lib" / "types.ts"
DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard"
STATE_JSON = DASHBOARD / "public" / "data" / "state.json"

#: Snapshot key -> the TypeScript interface describing one of its records.
PANELS = {
    "regime": "RegimePanel",
    "portfolio": "PortfolioPanel",
    "risk": "RiskPanel",
    "system": "SystemPanel",
    "freshness": "Freshness",
    "timing": "Timing",
}
ROW_PANELS = {"positions": "PositionRow", "signals": "SignalRow", "candidates": "Candidate"}

#: Activity key -> the interface for one of its rows. Nested under `activity`
#: rather than at the top level, so they need their own walk: a type that is
#: declared and never checked is worse than one that is not declared, because
#: it looks covered.
ACTIVITY_ROWS = {
    "orders": "OrderRow",
    "open_positions": "OpenPositionRow",
    "closed_positions": "ClosedPositionRow",
    "runs": "RunRow",
}


#: Strips the body of an inline object type, so `limits: { a: number }` yields
#: `limits` and not `a`. Applied repeatedly to handle nesting.
_INLINE_OBJECT = re.compile(r"\{[^{}]*\}")


def parse_interface(name: str) -> set[str]:
    """Field names declared by one interface in types.ts.

    A regex rather than a TypeScript parser: the file is a flat set of plain
    interfaces with no generics or inheritance, and a real parser would be a
    dependency doing far more than this needs.

    Note the absence of a `break` in the scan loop. An earlier version stopped
    after the first declaration on each line, and because this types file puts
    several fields per line, it silently checked about a third of them. The
    test passed while catching nothing, which is the worst state a test can be
    in. `test_the_parser_sees_every_field_on_a_line` pins the fix.
    """
    source = TYPES_TS.read_text()
    match = re.search(rf"export interface {name} \{{(.*?)\n\}}", source, re.S)
    assert match, f"interface {name} not found in types.ts"

    body = match.group(1)
    while _INLINE_OBJECT.search(body):
        body = _INLINE_OBJECT.sub(" ", body)

    fields: set[str] = set()
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "*", "/*")):
            continue
        fields.update(m.group(1) for m in re.finditer(r"(\w+)\??\s*:", stripped))
    return fields


@pytest.fixture(scope="module")
def payload():
    """The exact payload publish_demo writes.

    Built by the publisher rather than reassembled here. The two had already
    drifted once: this fixture was producing a payload with no activity block
    while the published file had one, so every activity assertion would have
    been checking something that never ships.
    """
    return demo_payload()


# ===========================================================================
# Every declared field is published
# ===========================================================================

def test_types_file_exists():
    assert TYPES_TS.exists(), "the dashboard's type declarations went missing"


def test_the_parser_sees_every_field_on_a_line():
    """The parser is the load-bearing part of every test below it.

    types.ts declares several fields per line. A parser that stops at the first
    one checks a third of the contract and reports success for the rest.
    """
    portfolio = parse_interface("PortfolioPanel")
    # All four of these share a single line in types.ts.
    assert {"equity", "cash", "buying_power", "daily_pnl"} <= portfolio
    assert len(portfolio) >= 13, f"only found {sorted(portfolio)}"


def test_the_parser_ignores_nested_object_keys():
    """`limits: { daily_halt: number }` declares `limits`, not `daily_halt`."""
    risk = parse_interface("RiskPanel")
    assert "limits" in risk
    assert "daily_halt" not in risk, "nested keys leaked into the outer field set"


@pytest.mark.parametrize("key,interface", sorted(PANELS.items()))
def test_panel_publishes_every_declared_field(payload, key, interface):
    declared = parse_interface(interface)
    published = set(payload[key])
    missing = declared - published
    assert not missing, (
        f"{interface} declares {sorted(missing)} but publish.py never writes them. "
        f"The panel will render undefined."
    )


@pytest.mark.parametrize("key,interface", sorted(ROW_PANELS.items()))
def test_row_publishes_every_declared_field(payload, key, interface):
    declared = parse_interface(interface)
    rows = payload[key]
    assert rows, f"{key} is empty, so the contract cannot be checked"
    for row in rows:
        missing = declared - set(row)
        # Signal rows legitimately vary: a rejection has no share count.
        if interface == "SignalRow":
            missing -= {"shares", "notional", "rejection_reason", "signal_regime", "message"}
        assert not missing, f"{interface} row {row.get('symbol')} is missing {sorted(missing)}"


def test_top_level_shape_matches_the_snapshot_interface(payload):
    declared = parse_interface("Snapshot")
    missing = declared - set(payload)
    assert not missing, f"Snapshot declares {sorted(missing)} which is never published"


def test_activity_declares_every_key_the_publisher_writes(payload):
    declared = parse_interface("Activity")
    published = set(payload["activity"])
    assert declared == published, (
        f"Activity declares {sorted(declared - published)} and the publisher "
        f"writes {sorted(published - declared)}"
    )


@pytest.mark.parametrize("key,interface", sorted(ACTIVITY_ROWS.items()))
def test_activity_rows_round_trip_from_sqlite(tmp_path, key, interface):
    """Build a database with one of everything and check the published shape.

    Against a real Repository rather than a hand-written dict, because the
    thing that breaks is a column rename in the schema, and a fixture that
    hardcodes the old name would keep passing through it.
    """
    from datetime import UTC, datetime

    from data.repository import open_repository
    from monitoring.publish import activity_from_repo

    repo = open_repository(tmp_path / "state.db")
    run_id = repo.start_run("paper", "schedule")

    class Trade:
        trade_id, order_id, symbol, side = "t1", "o1", "COIN", "buy"
        approved_qty, filled_qty, fill_price = 16, 16, 184.6
        status, stop_loss, take_profit, regime = "filled", 163.0, None, "strong_bull"
        submitted_at = filled_at = datetime.now(UTC)
        notes, skipped_reason = [], None

    repo.record_order(Trade(), run_id=run_id, client_order_id="rt-COIN-buy-20260904")
    repo.open_position("COIN", 16, 184.6, stop_price=163.0, regime="strong_bull")
    repo.update_open_position("COIN", current_price=190.0, unrealised_pnl=86.4)
    repo.open_position("SPY", 3, 770.0)
    repo.close_position("SPY", 800.0, "target")
    repo.finish_run(run_id, "ok")

    activity = activity_from_repo(repo)
    repo.close()

    declared = parse_interface(interface)
    rows = activity[key]
    assert rows, f"{key} came back empty, so the contract was not checked"
    for row in rows:
        missing = declared - set(row)
        assert not missing, f"{interface} declares {sorted(missing)}, not in the row"


@pytest.mark.parametrize("key,interface", sorted(ACTIVITY_ROWS.items()))
def test_demo_activity_matches_the_same_interfaces(payload, key, interface):
    """The demo data is what the deployed URL actually renders, so it has to
    satisfy the contract as strictly as real data does."""
    declared = parse_interface(interface)
    rows = payload["activity"][key]
    assert rows, f"demo {key} is empty, so the Activity page would look broken"
    for row in rows:
        missing = declared - set(row)
        assert not missing, f"demo {interface} row is missing {sorted(missing)}"


def test_the_demo_shows_orders_that_did_not_fill(payload):
    """A demo where everything filled hides the distinction the page exists to
    make. Cancelled and skipped orders are not trades."""
    orders = payload["activity"]["orders"]
    assert any(o["status"] == "filled" and o["fill_price"] for o in orders)
    assert any(o["filled_qty"] == 0 for o in orders), "no unfilled order in the demo"
    assert any(o["skipped_reason"] for o in orders), "no skipped order in the demo"


def test_the_demo_shows_a_failed_run(payload):
    """Otherwise the run-status treatment is never exercised by what ships."""
    assert any(r["status"] == "failed" for r in payload["activity"]["runs"])


def test_closed_position_pnl_matches_its_prices(payload):
    """Hand-written demo numbers drift from their own arithmetic."""
    for p in payload["activity"]["closed_positions"]:
        expected = (p["exit_price"] - p["entry_price"]) * p["quantity"]
        assert abs(p["realised_pnl"] - expected) < 0.02, f"{p['symbol']} P&L is inconsistent"


def test_the_publisher_redacts_broker_ids_from_activity(tmp_path):
    """Order ids and trade ids are broker-internal and the dashboard is public."""
    from datetime import UTC, datetime

    from data.repository import open_repository
    from monitoring.publish import _clean, activity_from_repo

    repo = open_repository(tmp_path / "state.db")

    class Trade:
        trade_id, order_id, symbol, side = "secret-trade", "secret-order", "COIN", "buy"
        approved_qty, filled_qty, fill_price = 16, 16, 184.6
        status, stop_loss, take_profit, regime = "filled", 163.0, None, "strong_bull"
        submitted_at = filled_at = datetime.now(UTC)
        notes, skipped_reason = [], None

    repo.record_order(Trade(), client_order_id="rt-COIN-buy-20260904")
    cleaned = _clean(activity_from_repo(repo))
    repo.close()

    blob = json.dumps(cleaned)
    assert "secret-trade" not in blob
    assert "secret-order" not in blob


# ===========================================================================
# Fields are populated, not merely present
# ===========================================================================

def test_no_panel_is_entirely_null(payload):
    """A field present but null renders as a blank cell, which reads as a
    working panel with nothing in it rather than as a bug."""
    for key in (*PANELS, "equity_history", "regime_history", "regime_mix"):
        value = payload[key]
        assert value, f"{key} published empty"
        if isinstance(value, dict):
            populated = [k for k, v in value.items() if v not in (None, "", [], {})]
            assert populated, f"every field in {key} is null or empty"


def test_numeric_fields_are_numbers_not_strings(payload):
    """`_jsonable` falls back to repr for anything it cannot serialise, so a
    stray object becomes a string and the UI's toFixed() throws."""
    for field in ("equity", "daily_pnl", "allocation", "leverage", "peak_equity"):
        assert isinstance(payload["portfolio"][field], (int, float)), \
            f"portfolio.{field} published as {type(payload['portfolio'][field]).__name__}"
    assert isinstance(payload["regime"]["confidence"], (int, float))


def test_risk_limits_are_all_present(payload):
    """The UI divides by these to size every meter. A missing one is a
    division by undefined, and the bar renders full."""
    required = {
        "daily_reduce", "daily_halt", "weekly_reduce", "weekly_halt",
        "max_from_peak", "max_exposure", "max_leverage", "max_risk_per_trade",
    }
    limits = payload["risk"]["limits"]
    assert required <= set(limits)
    assert all(isinstance(v, (int, float)) and v > 0 for v in limits.values())


def test_equity_history_points_have_the_charted_keys(payload):
    for point in payload["equity_history"]:
        assert {"t", "equity", "peak"} <= set(point)
        assert point["peak"] >= point["equity"] - 1e-6, "peak must be a high-water mark"


def test_regime_mix_shares_sum_to_one(payload):
    total = sum(entry["pct"] for entry in payload["regime_mix"])
    assert total == pytest.approx(1.0), f"regime shares sum to {total}, not 1"


def test_regime_mix_is_ordered_most_frequent_first(payload):
    shares = [entry["pct"] for entry in payload["regime_mix"]]
    assert shares == sorted(shares, reverse=True)


def test_regime_mix_ignores_unknown():
    """`unknown` is the absence of a classification, not a regime. Charting it
    would report a state the model never fitted."""
    mix = regime_mix([{"t": "1", "regime": "unknown"}, {"t": "2", "regime": "strong_bull"}])
    assert [entry["regime"] for entry in mix] == ["strong_bull"]


def test_regime_mix_of_nothing_is_empty_not_a_crash():
    assert regime_mix([]) == []


# ===========================================================================
# The committed snapshot is valid and safe
# ===========================================================================

def test_committed_snapshot_parses_and_is_labelled_honestly():
    """What ships in git is what the deployed URL renders on first load.

    This used to assert the committed snapshot was always demo, on the
    assumption that real data could only get here by someone running
    `--publish` locally and committing by accident. That stopped being true
    when the scheduled job started publishing real state and committing it,
    which is the intended pipeline rather than a mistake.

    So the invariant is no longer "it is demo". It is that the label matches
    the contents. Demo data rendered without its banner would be the actual
    problem, and that is what these two fields together prevent.
    """
    assert STATE_JSON.exists(), "no snapshot committed for the deployed dashboard"
    payload = json.loads(STATE_JSON.read_text())

    assert payload["source"] in ("demo", "live")
    assert payload["is_demo"] == (payload["source"] == "demo"), (
        f"source is {payload['source']!r} but is_demo is {payload['is_demo']}. "
        f"The banner is driven by is_demo, so a mismatch either hides the "
        f"warning over fabricated numbers or shows it over a real account."
    )


def test_committed_snapshot_carries_no_secrets():
    blob = STATE_JSON.read_text()
    for pattern in ("api_key", "secret", "account_number", "/Users/", "order_id"):
        assert pattern not in blob, f"{pattern!r} leaked into the committed snapshot"


def test_publish_demo_round_trips(tmp_path):
    payload = json.loads(publish_demo(tmp_path / "state.json").read_text())
    assert payload["source"] == "demo"
    assert parse_interface("Snapshot") - set(payload) == set()


# ===========================================================================
# Candidates: the dashboard's primary answer
# ===========================================================================

def test_candidates_are_ranked_contiguously(payload):
    ranks = [c["rank"] for c in payload["candidates"]]
    assert ranks == list(range(1, len(ranks) + 1)), f"ranks are {ranks}"


def test_approved_candidates_sort_above_blocked(payload):
    """A blocked name ranked above a tradable one would put a row you cannot act
    on at the top of a table whose whole purpose is telling you what to buy."""
    approvals = [c["approved"] for c in payload["candidates"]]
    assert approvals == sorted(approvals, reverse=True)


def test_approved_candidates_are_ordered_by_conviction(payload):
    """Ordering by notional alone let the 15% single-position cap flatten the
    ranking, putting a below-trend name first on a rounding difference."""
    convictions = [c["conviction"] for c in payload["candidates"] if c["approved"]]
    assert convictions == sorted(convictions, reverse=True)


def test_blocked_candidates_carry_a_reason(payload):
    """"Blocked" with no reason is the least useful row a dashboard can show."""
    for candidate in payload["candidates"]:
        if not candidate["approved"]:
            assert candidate["rejection_reason"], f"{candidate['symbol']} blocked with no reason"
            assert candidate["reason"], f"{candidate['symbol']} has no explanation"


def test_approved_candidates_have_a_size_and_a_stop(payload):
    for candidate in payload["candidates"]:
        if candidate["approved"]:
            assert candidate["shares"] > 0, f"{candidate['symbol']} approved for zero shares"
            assert candidate["stop_loss"] is not None, f"{candidate['symbol']} approved with no stop"
            assert candidate["stop_loss"] < candidate["entry_price"]


def test_conviction_is_a_fraction(payload):
    for candidate in payload["candidates"]:
        assert 0.0 <= candidate["conviction"] <= 1.0, candidate["symbol"]


def test_demo_shows_the_full_range_of_outcomes(payload):
    """A demo where everything is approved hides half the interface."""
    actions = {c["action"] for c in payload["candidates"]}
    reasons = {c["rejection_reason"] for c in payload["candidates"] if not c["approved"]}
    assert {"buy", "blocked"} <= actions
    assert len(reasons) >= 2, "every blocked demo row shares one rejection reason"


# ===========================================================================
# Freshness: the UI must not be able to imply real time
# ===========================================================================

def test_freshness_declares_it_is_not_realtime(payload):
    assert payload["freshness"]["realtime"] is False


def test_freshness_publishes_every_delay_in_the_chain(payload):
    """Three delays stack: bar interval, feed delay, publish cadence. A UI that
    polls every 5 seconds looks live, so all three have to be on the page."""
    freshness = payload["freshness"]
    assert freshness["timeframe"]
    assert freshness["sip_delay_minutes"] >= 15
    assert freshness["publish_cadence"]
    assert freshness["poll_seconds"] > 0


def test_freshness_survives_a_missing_bar_timestamp():
    """Before the first bar there is no timestamp, and the panel still renders."""
    from monitoring.publish import freshness

    result = freshness({"regime": {}, "system": {}})
    assert result["bar_age_hours"] is None
    assert result["realtime"] is False


def test_timing_answers_when_to_buy(payload):
    """"Buy this" is incomplete without when, at what price, and by what order
    type. On a daily-bar system the answer is never "right now"."""
    timing = payload["timing"]
    assert timing["acts_on"]
    assert timing["order_type"]
    assert timing["session_close_et"]
    assert 0 < timing["limit_offset_pct"] < 0.05


def test_sizing_is_affordable_for_the_configured_account(payload):
    """The complaint that started this: 62 shares of SPY is $31,880, which is
    not a position a real starter account can take."""
    equity = payload["portfolio"]["equity"]
    for candidate in payload["candidates"]:
        if candidate["approved"]:
            assert candidate["notional"] <= equity, \
                f"{candidate['symbol']} costs more than the whole account"


# ===========================================================================
# A blank snapshot must not overwrite a good one
# ===========================================================================

def test_a_skipped_bar_does_not_blank_the_dashboard(tmp_path):
    """The bug that emptied every panel.

    A scheduled run hit "bar already processed", returned an empty outcome,
    published it, and replaced a good snapshot with regime `unknown` and no
    candidates. The run succeeded and the dashboard went blank.
    """
    from monitoring.publish import _write, demo_payload, is_blank, publish

    path = tmp_path / "state.json"
    good = demo_payload()
    _write(good, path)
    assert not is_blank(good)

    blank_snapshot = {
        "regime": {"regime": "unknown", "confidence": 0.0, "timestamp": None},
        "portfolio": {}, "positions": [], "signals": [], "candidates": [],
        "risk": {}, "system": {}, "timestamp": None,
    }
    publish(blank_snapshot, path, source="live")

    after = json.loads(path.read_text())
    assert after["regime"]["regime"] == good["regime"]["regime"], \
        "an empty cycle overwrote a good snapshot"
    assert after["candidates"], "candidates were lost"


def test_activity_still_moves_forward_on_a_skipped_bar(tmp_path):
    """Run history is real even on a cycle that computed no regime, so it is
    merged forward rather than being frozen with the rest."""
    from monitoring.publish import _write, demo_payload, publish

    path = tmp_path / "state.json"
    _write(demo_payload(), path)

    blank = {"regime": {"regime": "unknown"}, "portfolio": {}, "positions": [],
             "signals": [], "candidates": [], "risk": {}, "system": {}}
    fresh_runs = {"orders": [], "open_positions": [], "closed_positions": [],
                  "runs": [{"id": 99, "status": "ok"}],
                  "expectancy": {"trades": 0, "expectancy": 0.0, "win_rate": 0.0,
                                 "avg_win": 0.0, "avg_loss": 0.0}}
    publish(blank, path, source="live", activity=fresh_runs)

    after = json.loads(path.read_text())
    assert after["activity"]["runs"][0]["id"] == 99, "run history did not move forward"
    assert after["regime"]["regime"] != "unknown", "regime was blanked anyway"


def test_a_good_snapshot_replaces_a_blank_one(tmp_path):
    """The guard is one-directional. Real data must always win."""
    from monitoring.publish import _write, demo_payload, publish

    path = tmp_path / "state.json"
    _write({"regime": {"regime": "unknown"}, "candidates": [], "positions": [],
            "activity": {}}, path)

    good = demo_payload()
    publish(good, path, source="demo",
            equity_history=[], regime_history=[])
    assert json.loads(path.read_text())["regime"]["regime"] != "unknown"


# -- bento grid alignment ----------------------------------------------------
#
# Reported as "the boxes don't fill the width" more than once, and fixed by eye
# more than once. The cause is arithmetic, not taste: the grid is 12 columns and
# every card declares a span, so any row whose spans sum to less than 12 leaves
# dead space on the right. Three rows were short (9, 9 and 6 of 12), which is
# exactly where the gaps appeared.

CELL_WIDTHS_LG = {"1": 3, "2": 6, "3": 9, "full": 12}


def _card_spans() -> dict[str, str]:
    """Map each panel component to the span its <Card> declares."""
    spans: dict[str, str] = {}
    for name in ("index.tsx", "candidates.tsx"):
        source = (DASHBOARD / "components" / "panels" / name).read_text()
        current = None
        for line in source.splitlines():
            declared = re.match(r"\s*(?:export\s+)?function\s+(\w+)", line)
            if declared:
                current = declared.group(1)
            found = re.search(r'<Card span="(\w+)"', line)
            if found and current and current not in spans:
                spans[current] = found.group(1)
    return spans


def _rendered_order() -> list[str]:
    """The cards as page.tsx actually lays them out, in order."""
    page = (DASHBOARD / "app" / "page.tsx").read_text()
    grid = page.split('<div className="grid grid-cols-12 gap-4">', 1)[1].split("</div>", 1)[0]
    return re.findall(r"<(\w+Card)\b", grid)


def test_every_bento_row_fills_the_full_width():
    spans, order = _card_spans(), _rendered_order()
    assert order, "no cards found in the bento grid"

    rows, row, used = [], [], 0
    for name in order:
        width = CELL_WIDTHS_LG[spans[name]]
        if used + width > 12:
            rows.append((row, used))
            row, used = [], 0
        row.append(name)
        used += width
    rows.append((row, used))

    short = [(names, total) for names, total in rows if total != 12]
    assert not short, (
        "these rows leave dead space on the right at lg:\n"
        + "\n".join(f"  {total}/12  {' + '.join(n)}" for n, total in short)
    )


def test_every_card_declares_a_span():
    """A card with no span inherits nothing and breaks the row arithmetic."""
    spans = _card_spans()
    missing = [name for name in _rendered_order() if name not in spans]
    assert not missing, f"cards rendered without a declared span: {missing}"


# -- signal outcomes ---------------------------------------------------------
#
# The feed carries three kinds of event and the panel recognised two, so an
# `order_rejected` fell through to the approved branch and rendered as a green
# "Approved · 0 shares · $0". Four real broker refusals were displayed as
# successes that day. The size was invented by a `?? 0` fallback on a field the
# event never carries.

SIGNAL_EVENTS = ("signal_generated", "signal_rejected", "order_rejected")


def _signals_panel_source(*, strip_comments: bool = False) -> str:
    """The SignalsCard body.

    `strip_comments` matters: the comment explaining the `?? 0` bug contains the
    string `?? 0`, so a naive search finds the prose that documents the fix and
    calls it the bug.
    """
    source = (DASHBOARD / "components" / "panels" / "index.tsx").read_text()
    start = source.index("export function SignalsCard")
    body = source[start:start + 6000]
    if strip_comments:
        body = re.sub(r"//.*?$", "", body, flags=re.MULTILINE)
    return body


@pytest.mark.parametrize("event", SIGNAL_EVENTS)
def test_the_signals_panel_handles_every_event_the_engine_emits(event):
    """An event the panel does not name gets whatever the else branch says."""
    emitted = (Path(__file__).resolve().parent.parent / "monitoring" / "logger.py").read_text()
    assert f'"{event}"' in emitted, f"{event} is no longer emitted; update this test"

    panel = _signals_panel_source()
    if event == "signal_generated":
        return          # the default branch, deliberately unnamed
    assert f'"{event}"' in panel, (
        f"{event} is not handled in SignalsCard, so it renders as an approval"
    )


def test_an_order_failure_never_reports_a_size_it_does_not_have():
    """`order_rejected` carries no shares. Defaulting it to 0 printed a
    failure as 'Approved, 0 shares, $0', which reads as a successful trade."""
    panel = _signals_panel_source(strip_comments=True)
    assert "s.shares ?? 0" not in panel, (
        "shares is defaulted to 0; an event with no size will render as '0 shares'"
    )
    assert "sized" in panel, "expected an explicit has-a-size check before printing one"


# -- freshness badge ---------------------------------------------------------
#
# The badge showed "69m ago" over a file written 8 minutes earlier. It read the
# newest run with status 'ok', which is by construction never the run that
# wrote the file: the publisher serialises from inside the run, so that row is
# still 'running' in its own snapshot and the badge always showed the previous
# one. The gap is one refresh interval during the session and grows to the
# whole weekend when the market is shut.


def test_the_freshness_badge_reads_the_file_timestamp():
    page = (DASHBOARD / "app" / "page.tsx").read_text()
    start = page.index("function LastRun")
    body = re.sub(r"//.*?$", "", page[start:start + 2000], flags=re.MULTILINE)

    assert "relativeTime(snap.published_at)" in body, (
        "the badge must age the snapshot itself, not a run row"
    )
    assert "lastSuccessfulRun" not in body, (
        "lastSuccessfulRun returns the run BEFORE the one that wrote the file"
    )


def test_a_run_in_flight_is_not_treated_as_a_failure():
    """The newest row is always 'running'. Counting it as unsettled is the
    whole point; counting it as broken lights the warning on every load."""
    page = (DASHBOARD / "app" / "page.tsx").read_text()
    start = page.index("function LastRun")
    body = re.sub(r"//.*?$", "", page[start:start + 2000], flags=re.MULTILINE)
    assert 'r.status !== "running"' in body, (
        "in-flight runs must be excluded before judging the last outcome"
    )
