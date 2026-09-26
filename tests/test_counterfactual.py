"""The counterfactual gate ledger must not invent money.

`scripts/counterfactual.py` replays refused signals against real bars to ask
what the risk gates cost or saved. Every way that simulation can flatter itself
is a way to conclude a gate is bad when it is fine, so each one is pinned here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("cf", ROOT / "scripts" / "counterfactual.py")
cf = importlib.util.module_from_spec(spec)
sys.modules["cf"] = cf
spec.loader.exec_module(cf)


def _bars(rows):
    """rows = [(open, high, low, close), ...]"""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=pd.date_range("2026-01-01", periods=len(rows), freq="B"))


def test_the_stop_is_checked_before_the_target():
    """A bar that touches both levels must be booked as the stop.

    Within one daily bar the order of the high and the low is unknown, and
    assuming the good one first is exactly how a backtest invents money.
    """
    bars = _bars([(100, 130, 70, 100)])          # touches target 120 AND stop 80
    pnl, reason, _ = cf.replay(bars, entry_price=100, stop=80, target=120, hold=5)

    assert reason == "stop", "an ambiguous bar was resolved in our own favour"
    assert pnl < 0


def test_a_gap_through_the_stop_fills_at_the_open_not_the_stop():
    """Price gapping below the stop overnight does not fill at the stop. A
    simulation that pretends it does understates every bad day."""
    bars = _bars([(70, 75, 68, 72)])             # opens far below an 80 stop
    pnl, reason, _ = cf.replay(bars, entry_price=100, stop=80, target=120, hold=5)

    assert reason == "stop"
    assert pnl == pytest.approx(70 / 100 - 1), "filled better than the gap allowed"


def test_a_gap_through_the_target_does_not_fill_worse_than_the_target():
    bars = _bars([(130, 135, 128, 132)])
    pnl, reason, _ = cf.replay(bars, entry_price=100, stop=80, target=120, hold=5)

    assert reason == "target"
    assert pnl == pytest.approx(130 / 100 - 1)


def test_the_horizon_exit_uses_the_close_of_the_last_bar_held():
    bars = _bars([(100, 105, 98, 101), (101, 106, 99, 103), (103, 107, 100, 106)])
    pnl, reason, held = cf.replay(bars, entry_price=100, stop=50, target=200, hold=2)

    assert reason == "horizon"
    assert held == 2
    assert pnl == pytest.approx(103 / 100 - 1), "held past its own horizon"


def test_it_stops_at_the_horizon_even_when_more_bars_exist():
    """Holding longer than the stated horizon would make later signals look
    better than earlier ones purely by having more data."""
    rising = _bars([(100 + i, 105 + i, 99 + i, 101 + i) for i in range(30)])
    _, _, held = cf.replay(rising, entry_price=100, stop=50, target=1000, hold=10)
    assert held == 10


def test_no_forward_bars_returns_nothing_rather_than_zero():
    """Zero would be counted as a flat trade and dilute the averages. There is
    no outcome here, and the caller has to skip the row."""
    pnl, reason, _ = cf.replay(_bars([]), entry_price=100, stop=80, target=120, hold=5)
    assert pnl is None
    assert reason == "no bars"


def test_a_signal_with_no_stop_or_target_runs_to_the_horizon():
    bars = _bars([(100, 105, 95, 102), (102, 108, 100, 107)])
    pnl, reason, _ = cf.replay(bars, entry_price=100, stop=None, target=None, hold=2)
    assert reason == "horizon"
    assert pnl == pytest.approx(107 / 100 - 1)
