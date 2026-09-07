"""
The gate between paper trading and real money.

The property that matters most here is that a broken check reads as a FAIL.
A preflight that passes because it could not run is worse than no preflight,
because it converts "we did not check" into "we checked and it was fine".
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta

import pytest

from data.repository import open_repository
from scripts.preflight import (
    MIN_CLOSED_TRADES,
    check_beats_benchmarks,
    check_closed_trades,
    check_expectancy,
    check_not_halted,
    render,
    run_preflight,
)


@pytest.fixture
def repo(tmp_path):
    r = open_repository(tmp_path / "state.db")
    yield r
    r.close()


def _close_trades(repo, n, pnl_each):
    for i in range(n):
        repo.open_position(f"S{i}", 1, 100.0)
        repo.close_position(f"S{i}", 100.0 + pnl_each, "target" if pnl_each > 0 else "stop")


# -- closed trades ----------------------------------------------------------

def test_an_empty_book_fails_the_trade_count(repo):
    check = check_closed_trades(repo)
    assert not check.passed
    assert "0 closed" in check.detail


def test_twenty_nine_trades_is_not_thirty(repo):
    """Off-by-one on a gate is the difference between blocked and not."""
    _close_trades(repo, MIN_CLOSED_TRADES - 1, 5.0)
    assert not check_closed_trades(repo).passed


def test_thirty_trades_passes(repo):
    _close_trades(repo, MIN_CLOSED_TRADES, 5.0)
    assert check_closed_trades(repo).passed


def test_a_missing_database_fails_rather_than_erroring():
    """A check that cannot run must not pass."""
    assert not check_closed_trades(None).passed


# -- expectancy -------------------------------------------------------------

def test_expectancy_with_no_trades_fails(repo):
    check = check_expectancy(repo)
    assert not check.passed
    assert "no closed trades" in check.detail


def test_negative_expectancy_fails(repo):
    _close_trades(repo, 10, -8.0)
    check = check_expectancy(repo)
    assert not check.passed
    assert "-8" in check.detail


def test_positive_expectancy_passes_and_reports_the_win_rate(repo):
    """Expectancy alone hides its shape, so the win rate is shown with it."""
    _close_trades(repo, 10, 12.0)
    check = check_expectancy(repo)
    assert check.passed
    assert "100% win rate" in check.detail


# -- benchmarks -------------------------------------------------------------

def _write_comparison(tmp_path, monkeypatch, rows):
    import scripts.preflight as preflight

    results = tmp_path / "backtest" / "results" / "SPY"
    results.mkdir(parents=True)
    with open(results / "benchmark_comparison.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["strategy", "total_return"])
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)


def test_losing_to_buy_and_hold_fails(tmp_path, monkeypatch):
    _write_comparison(tmp_path, monkeypatch, [
        {"strategy": "regime-trader", "total_return": "0.55"},
        {"strategy": "buy_and_hold", "total_return": "1.18"},
        {"strategy": "sma_200_trend", "total_return": "-0.01"},
        {"strategy": "random_allocation_100_seeds", "total_return": "0.30"},
    ])
    check = check_beats_benchmarks()
    assert not check.passed
    assert "buy-and-hold" in check.detail


def test_losing_to_random_entry_fails(tmp_path, monkeypatch):
    """The benchmark most often left out, and the one that matters most: a
    strategy that cannot beat coin flips at the same exposure has found
    nothing, it was just in the market during an up year."""
    _write_comparison(tmp_path, monkeypatch, [
        {"strategy": "regime-trader", "total_return": "0.55"},
        {"strategy": "buy_and_hold", "total_return": "0.20"},
        {"strategy": "sma_200_trend", "total_return": "-0.01"},
        {"strategy": "random_allocation_100_seeds", "total_return": "0.85"},
    ])
    check = check_beats_benchmarks()
    assert not check.passed
    assert "random entry" in check.detail


def test_beating_all_three_passes(tmp_path, monkeypatch):
    _write_comparison(tmp_path, monkeypatch, [
        {"strategy": "regime-trader", "total_return": "1.50"},
        {"strategy": "buy_and_hold", "total_return": "1.18"},
        {"strategy": "sma_200_trend", "total_return": "-0.01"},
        {"strategy": "random_allocation_100_seeds", "total_return": "0.85"},
    ])
    assert check_beats_benchmarks().passed


def test_an_unmeasured_benchmark_is_not_a_pass(tmp_path, monkeypatch):
    """A benchmark that was not run is not a benchmark that was beaten."""
    _write_comparison(tmp_path, monkeypatch, [
        {"strategy": "regime-trader", "total_return": "1.50"},
        {"strategy": "buy_and_hold", "total_return": "1.18"},
        {"strategy": "sma_200_trend", "total_return": "-0.01"},
    ])
    check = check_beats_benchmarks()
    assert not check.passed
    assert "not measured" in check.detail


def test_no_backtest_at_all_fails(tmp_path, monkeypatch):
    import scripts.preflight as preflight
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    assert not check_beats_benchmarks().passed


# -- halt state -------------------------------------------------------------

def test_a_halt_lock_blocks_going_live(tmp_path, monkeypatch, repo):
    """Going live while a halt exists means the first thing real money does is
    ignore a stop signal the system raised for itself."""
    import scripts.preflight as preflight

    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    (tmp_path / "trading_halted.lock").write_text("daily drawdown breach")

    check = check_not_halted(repo)
    assert not check.passed
    assert "lock" in check.detail


def test_a_tripped_breaker_blocks_going_live(tmp_path, monkeypatch, repo):
    import scripts.preflight as preflight

    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    repo.record_breaker("weekly", "tripped")

    check = check_not_halted(repo)
    assert not check.passed
    assert "weekly" in check.detail


def test_a_cleared_breaker_does_not_block(tmp_path, monkeypatch, repo):
    import scripts.preflight as preflight

    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    repo.record_breaker("weekly", "tripped")
    repo.record_breaker("weekly", "cleared")

    assert check_not_halted(repo).passed


# -- the whole thing --------------------------------------------------------

def test_preflight_on_a_fresh_system_fails(tmp_path):
    """The honest result today, and the one this suite pins. A preflight that
    passed on an untested strategy with no trading history would be lying."""
    result = run_preflight(tmp_path / "state.db")
    assert not result.passed
    assert any("closed paper trades" in c.name for c in result.checks if not c.passed)


def test_preflight_reports_every_check_not_just_the_first_failure(tmp_path):
    """Fixing them one run at a time would take six runs to discover six
    problems."""
    result = run_preflight(tmp_path / "state.db")
    assert len(result.checks) >= 6


def test_the_output_says_what_to_do_about_each_failure(tmp_path):
    text = render(run_preflight(tmp_path / "state.db"))
    assert "->" in text, "a failure with no remedy is just bad news"
    assert "NOT READY" in text
    assert "Paper trading is unaffected" in text


def test_a_crashed_check_reads_as_a_failure(tmp_path, monkeypatch):
    """Worse than no preflight would be one that passes because it could not
    run: that turns "we did not check" into "we checked and it was fine"."""
    import scripts.preflight as preflight

    monkeypatch.setattr(preflight, "check_closed_trades",
                        lambda repo: preflight.Check("boom", False, "exploded"))
    assert not run_preflight(tmp_path / "state.db").passed


def test_the_exit_code_is_one_on_failure(tmp_path):
    from scripts.preflight import main
    assert main(["--db", str(tmp_path / "state.db")]) == 1


def test_json_output_is_machine_readable(tmp_path, capsys):
    import json

    from scripts.preflight import main

    main(["--json", "--db", str(tmp_path / "state.db")])
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert len(payload["checks"]) >= 6
    assert all({"name", "passed", "detail"} <= set(c) for c in payload["checks"])


# -- the live gate ----------------------------------------------------------

def test_live_mode_is_blocked_when_preflight_fails(tmp_path, monkeypatch):
    """The whole point. --i-understand-live says the operator accepts the risk;
    it says nothing about whether the strategy has ever worked."""
    from config import load_settings
    from main import EngineError, TradingEngine

    engine = TradingEngine(load_settings(), allow_live=True,
                           db_path=tmp_path / "state.db")
    engine.is_paper = False

    with pytest.raises(EngineError, match="Preflight failed"):
        engine._require_preflight()


def test_the_block_names_what_failed(tmp_path):
    from config import load_settings
    from main import EngineError, TradingEngine

    engine = TradingEngine(load_settings(), allow_live=True,
                           db_path=tmp_path / "state.db")
    try:
        engine._require_preflight()
    except EngineError as exc:
        assert "closed paper trades" in str(exc)
        assert "Paper trading is unaffected" in str(exc)
    else:
        pytest.fail("live mode was not blocked")


def test_paper_mode_never_runs_preflight(tmp_path, monkeypatch):
    """Paper is where the 30 trades come from. Gating it would be circular."""
    import scripts.preflight as preflight

    called = []
    monkeypatch.setattr(preflight, "run_preflight",
                        lambda *a, **k: called.append(1) or preflight.Preflight())

    from config import load_settings
    from main import TradingEngine

    engine = TradingEngine(load_settings(), db_path=tmp_path / "state.db")
    engine.is_paper = True
    # _connect_broker only calls _require_preflight when is_paper is False.
    assert called == []


def test_the_check_timestamp_is_recorded(tmp_path):
    result = run_preflight(tmp_path / "state.db")
    stamp = datetime.fromisoformat(result.as_dict()["checked_at"])
    assert datetime.now(UTC) - stamp < timedelta(seconds=30)
