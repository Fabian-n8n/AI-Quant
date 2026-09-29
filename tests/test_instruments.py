"""Every symbol the system trades must be describable on the dashboard.

The universe lives in `config/settings.yaml` and the descriptions live in
`core/instruments.py`. Nothing else connects them, so without this test adding
a symbol renders a blank name on the page and no build fails.
"""

from __future__ import annotations

import pytest

from config import load_settings
from core.instruments import INSTRUMENTS, describe


@pytest.fixture(scope="module")
def universe():
    return load_settings()["broker"]["symbols"]


def test_every_traded_symbol_has_a_description(universe):
    missing = [s for s in universe if s not in INSTRUMENTS]
    assert not missing, (
        f"{missing} are traded but not described. Add them to "
        f"core/instruments.py or the dashboard will show a bare ticker."
    )


def test_no_descriptions_for_symbols_that_are_not_traded(universe):
    """Stale entries are how a table slowly fills with names nobody recognises
    from a universe that changed two variants ago."""
    extra = [s for s in INSTRUMENTS if s not in universe]
    assert not extra, f"{extra} are described but no longer traded"


@pytest.mark.parametrize("symbol", sorted(INSTRUMENTS))
def test_each_entry_is_usable_on_a_dashboard_row(symbol):
    entry = INSTRUMENTS[symbol]
    assert set(entry) == {"name", "what"}, f"{symbol} has unexpected keys"

    # A name longer than the ticker, or it is not telling the reader anything.
    assert entry["name"] and entry["name"] != symbol, f"{symbol} has no real name"
    assert len(entry["name"]) <= 32, f"{symbol} name is too long for a table cell"

    # Two lines of prose. Long enough to say something, short enough to render.
    what = entry["what"]
    assert 40 <= len(what) <= 200, f"{symbol} description is {len(what)} chars"
    assert what[0].isupper(), f"{symbol} description should read as a sentence"
    assert what.rstrip().endswith("."), f"{symbol} description should end in a full stop"


def test_an_unknown_symbol_falls_back_instead_of_raising():
    """A missing description must leave a dull row, never break the page."""
    got = describe("ZZZZ")
    assert got == {"name": "ZZZZ", "what": ""}


def test_the_published_payload_carries_the_table():
    from monitoring.publish import demo_payload

    instruments = demo_payload()["instruments"]
    assert instruments, "the dashboard has no names to render"
    assert instruments["NVDA"]["name"] == "Nvidia"
