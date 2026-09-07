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

from monitoring.publish import build_payload, demo_snapshot, publish_demo, regime_mix

TYPES_TS = Path(__file__).resolve().parent.parent / "dashboard" / "lib" / "types.ts"
STATE_JSON = Path(__file__).resolve().parent.parent / "dashboard" / "public" / "data" / "state.json"

#: Snapshot key -> the TypeScript interface describing one of its records.
PANELS = {
    "regime": "RegimePanel",
    "portfolio": "PortfolioPanel",
    "risk": "RiskPanel",
    "system": "SystemPanel",
}
ROW_PANELS = {"positions": "PositionRow", "signals": "SignalRow"}


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
    snapshot = demo_snapshot()
    return build_payload(
        snapshot, source="demo",
        equity_history=snapshot["equity_history"],
        regime_history=snapshot["regime_history"],
    )


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

def test_committed_snapshot_parses_and_is_demo():
    """What ships in git is what the deployed URL renders on first load."""
    assert STATE_JSON.exists(), "no snapshot committed for the deployed dashboard"
    payload = json.loads(STATE_JSON.read_text())
    assert payload["source"] == "demo", \
        "a real account snapshot is committed. Publish demo data before pushing."


def test_committed_snapshot_carries_no_secrets():
    blob = STATE_JSON.read_text()
    for pattern in ("api_key", "secret", "account_number", "/Users/", "order_id"):
        assert pattern not in blob, f"{pattern!r} leaked into the committed snapshot"


def test_publish_demo_round_trips(tmp_path):
    payload = json.loads(publish_demo(tmp_path / "state.json").read_text())
    assert payload["source"] == "demo"
    assert parse_interface("Snapshot") - set(payload) == set()
