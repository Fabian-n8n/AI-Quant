"""
Phase 8: the monitoring package.

Logging, alerting and the dashboard are the parts of a trading system nobody
tests, on the grounds that they are "just output". They are not: a log that
silently stopped rotating fills a disk, an alert that a rate limiter swallowed
is an incident nobody saw, and a dashboard that renders a stale snapshot is
worse than one that renders nothing, because it looks fine.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest

from monitoring.alerts import AlertLevel, AlertManager, AlertTrigger
from monitoring.dashboard import DashboardState, TerminalDashboard, _held_for, risk_bar
from monitoring.logger import (
    MAX_BYTES,
    STREAM_BY_EVENT,
    EventType,
    SizedTimedRotatingHandler,
    TradingLogger,
)
from monitoring.publish import SCHEMA_VERSION, build_payload, demo_snapshot, publish, publish_demo

# ===========================================================================
# Logger
# ===========================================================================

@pytest.fixture
def log(tmp_path):
    logger = TradingLogger(name=f"test-{tmp_path.name}", log_dir=tmp_path).setup()
    yield logger
    logger.close()


def test_four_streams_are_created(log, tmp_path):
    log.log_event(EventType.SYSTEM_START, "up")
    assert set(log.stream_paths()) == {"main", "trades", "alerts", "regime"}
    assert (tmp_path / "main.log").exists()


def test_events_are_filed_to_the_right_stream(log, tmp_path):
    log.log_event(EventType.ORDER_FILLED, "fill")
    log.log_event(EventType.REGIME_CHANGE, "change")
    log.log_event(EventType.BREAKER_TRIGGERED, "breaker")

    def lines(name):
        path = tmp_path / f"{name}.log"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    assert [e["event"] for e in lines("trades")] == ["order_filled"]
    assert [e["event"] for e in lines("regime")] == ["regime_change"]
    assert [e["event"] for e in lines("alerts")] == ["breaker_triggered"]
    # main is a superset: one file with the true ordering.
    assert [e["event"] for e in lines("main")] == [
        "order_filled", "regime_change", "breaker_triggered"
    ]


def test_every_event_type_is_filed_deliberately():
    """A new EventType that nobody routed lands in main only. That is a valid
    choice, but it should be one someone made rather than one that happened."""
    main_only = {e.value for e in EventType} - {e.value for e in STREAM_BY_EVENT}
    assert main_only == {"system_start", "system_shutdown"}


def test_context_is_stamped_on_every_record(log):
    """The spec requires timestamp, regime, probability, equity, positions and
    daily_pnl on every entry."""
    log.set_context(regime="strong_bull", probability=0.87, equity=105_230.0,
                    positions=2, daily_pnl=340.0)
    record = log.log_event(EventType.ORDER_FILLED, "fill")

    for field in ("timestamp", "regime", "probability", "equity", "positions", "daily_pnl"):
        assert field in record, f"{field} missing from the record"
    assert record["equity"] == 105_230.0


def test_explicit_fields_win_over_context(log):
    log.set_context(regime="neutral")
    record = log.log_event(EventType.ERROR, "x", regime="strong_bear")
    assert record["regime"] == "strong_bear"


def test_rotation_is_bounded_by_size_and_time(tmp_path):
    handler = SizedTimedRotatingHandler(tmp_path / "main.log", maxBytes=512, backupCount=30)
    assert handler.maxBytes == 512
    assert handler.when == "MIDNIGHT"
    assert handler.backupCount == 30
    handler.close()


def test_rotation_actually_rotates(tmp_path):
    """A handler configured to rotate but that never does is the failure mode
    worth catching: the config looks right and the disk still fills."""
    logger = TradingLogger(name="rot", log_dir=tmp_path).setup()
    for handler in logger._handlers.values():
        handler.maxBytes = 400

    for i in range(40):
        logger.log_event(EventType.BAR_PROCESSED, f"bar {i}", payload="x" * 80)
    logger.close()

    rotated = list(tmp_path.glob("main.log.*"))
    assert rotated, "main.log never rotated despite passing maxBytes many times over"
    # The bound is the whole point. Python 3.14's TimedRotatingFileHandler
    # returns early when the dated backup already exists, so before
    # SizedTimedRotatingHandler.doRollover was reimplemented this file reached
    # 7288 bytes against a 400-byte cap.
    assert (tmp_path / "main.log").stat().st_size <= 400 * 2


def test_same_day_rotations_do_not_overwrite_each_other(tmp_path):
    """The stdlib names every backup for the day identically. Without unique
    suffixes, rotation two silently discards rotation one's history."""
    logger = TradingLogger(name="samedaya", log_dir=tmp_path).setup()
    for handler in logger._handlers.values():
        handler.maxBytes = 400
    for i in range(30):
        logger.log_event(EventType.BAR_PROCESSED, f"bar {i}", payload="x" * 80)
    logger.close()

    backups = list(tmp_path.glob("main.log.*"))
    assert len(backups) > 1, "every same-day rotation landed on the same filename"
    assert all(b.stat().st_size > 0 for b in backups)


def test_backups_are_pruned_to_the_retention_limit(tmp_path):
    logger = TradingLogger(name="prune", log_dir=tmp_path).setup()
    for handler in logger._handlers.values():
        handler.maxBytes = 200
        handler.backupCount = 3
    for i in range(40):
        logger.log_event(EventType.BAR_PROCESSED, f"bar {i}", payload="x" * 80)
    logger.close()

    assert len(list(tmp_path.glob("main.log.*"))) <= 3


def test_default_rotation_bound_matches_the_spec(log):
    assert MAX_BYTES == 10 * 1024 * 1024
    for handler in log._handlers.values():
        assert handler.backupCount == 30


def test_log_never_raises_on_an_unserialisable_field(log):
    record = log.log_event(EventType.ERROR, "x", blob=object(), when=datetime.now(UTC))
    assert isinstance(record["blob"], str)


def test_recent_returns_the_in_memory_tail(log):
    for i in range(10):
        log.log_event(EventType.BAR_PROCESSED, f"bar {i}")
    assert len(log.recent(limit=3)) == 3


def test_recent_is_capped_so_a_long_session_cannot_grow_unbounded(log):
    for i in range(700):
        log.log_event(EventType.BAR_PROCESSED, f"bar {i}")
    assert len(log._recent) <= 500


# ===========================================================================
# Alerts
# ===========================================================================

def test_all_seven_spec_triggers_exist():
    """The spec names seven. Losing one to a refactor should fail here, not in
    production at 3am."""
    required = {
        "regime_change", "circuit_breaker", "large_pnl", "data_feed_down",
        "api_lost", "hmm_retrained", "flicker_exceeded",
    }
    assert required <= {t.value for t in AlertTrigger}


@pytest.mark.parametrize("method,args,level", [
    ("alert_breaker_triggered", ("daily_halt", -0.03, 97_000.0), AlertLevel.CRITICAL),
    ("alert_data_feed_down", ("timeout",), AlertLevel.CRITICAL),
    ("alert_broker_down", ("refused",), AlertLevel.CRITICAL),
    ("alert_large_pnl", (-2400.0, -0.024, 97_600.0), AlertLevel.WARNING),
    ("alert_flicker_exceeded", (6, 4, 20), AlertLevel.WARNING),
    ("alert_regime_change", ("neutral", "strong_bull", 0.87), AlertLevel.INFO),
])
def test_trigger_levels(method, args, level):
    sent = []
    alerts = AlertManager({}, sink=lambda level, s, b: sent.append(level))
    getattr(alerts, method)(*args)
    assert sent == [level]


def test_rate_limit_is_per_event_type_not_per_message():
    """Keying the limit on the message would let a value that changes every bar
    defeat it entirely."""
    sent = []
    alerts = AlertManager({}, rate_limit_minutes=15, sink=lambda level, s, b: sent.append(s))
    assert alerts.alert_flicker_exceeded(5, 4, 20) is True
    assert alerts.alert_flicker_exceeded(6, 4, 20) is False   # different wording, same type
    assert len(sent) == 1


def test_different_breakers_are_not_collapsed_together():
    sent = []
    alerts = AlertManager({}, sink=lambda level, s, b: sent.append(s))
    alerts.alert_breaker_triggered("daily_reduce", -0.02, 98_000)
    alerts.alert_breaker_triggered("peak_halt", -0.11, 89_000)
    assert len(sent) == 2


def test_info_alerts_do_not_reach_email_or_webhook(monkeypatch):
    """A regime change is the system working. An inbox that receives one every
    few days is an inbox that stops being read."""
    alerts = AlertManager({"alert_email": "x@example.com", "alert_webhook": "https://example.com"},
                          sink=lambda *a: None)
    emailed, hooked = [], []
    monkeypatch.setattr(alerts, "send_email", lambda s, b: emailed.append(s))
    monkeypatch.setattr(alerts, "send_webhook", lambda p: hooked.append(p))

    alerts.alert_regime_change("neutral", "strong_bull", 0.9)
    assert emailed == [] and hooked == []

    alerts.alert_breaker_triggered("peak_halt", -0.11, 89_000)
    assert emailed and hooked


def test_retrain_that_changes_state_count_is_escalated():
    """A refit landing on a different number of regimes has redrawn the map the
    allocator reads. That is worth more than the routine weekly note.

    One manager per call: they share a rate-limit key, and this is a test about
    severity rather than about the limiter.
    """
    def level_for(n_states, previous):
        sent = []
        AlertManager({}, sink=lambda level, s, b: sent.append(level)).alert_hmm_retrained(
            n_states, "weekly", previous_states=previous)
        return sent[0]

    assert level_for(7, 7) is AlertLevel.INFO
    assert level_for(5, 7) is AlertLevel.WARNING


def test_large_pnl_fires_on_gains_too():
    """Unless the system was meant to make 4% today, a large gain is as much a
    reason to look as a large loss."""
    sent = []
    alerts = AlertManager({}, sink=lambda level, s, b: sent.append(s))
    alerts.alert_large_pnl(4200.0, 0.042, 104_200.0)
    assert "gain" in sent[0]


def test_alerts_are_written_to_the_alerts_log(tmp_path):
    log = TradingLogger(name="alertlog", log_dir=tmp_path).setup()
    alerts = AlertManager({}, sink=lambda *a: None, trading_logger=log)
    alerts.alert_breaker_triggered("peak_halt", -0.11, 89_000)
    log.close()

    raw = (tmp_path / "alerts.log").read_text().splitlines()
    events = [json.loads(line) for line in raw if line.strip()]
    assert any(e["event"] == "alert_sent" and e["alert_level"] == "critical" for e in events)


def test_delivery_failure_never_raises():
    alerts = AlertManager({}, sink=lambda *a: (_ for _ in ()).throw(RuntimeError("smtp down")))
    assert alerts.send(AlertLevel.CRITICAL, "s", "b") is True


# ===========================================================================
# Dashboard
# ===========================================================================

def test_risk_bar_grades_on_the_limit_not_the_raw_drawdown():
    """The same 2% drawdown reads differently against different limits.

    2% of a 3% limit is two thirds spent; 2% of a 10% limit has plenty of room.
    Grading the raw drawdown would colour both identically, which is backwards.
    """
    _, tight = risk_bar(0.02, 0.03)     # 67% of the limit
    _, loose = risk_bar(0.02, 0.10)     # 20% of the limit
    assert tight == "yellow"
    assert loose == "green"
    assert tight != loose


@pytest.mark.parametrize("used,limit,expected", [
    (0.004, 0.03, "green"),    # 13%
    (0.018, 0.03, "yellow"),   # 60%
    (0.028, 0.03, "red"),      # 93%
])
def test_risk_bar_bands(used, limit, expected):
    assert risk_bar(used, limit)[1] == expected


def test_risk_bar_saturates_rather_than_overflowing():
    bar, colour = risk_bar(0.5, 0.03)
    assert colour == "red"
    assert bar.count("█") == len(bar)


def test_risk_bar_handles_a_zero_limit():
    bar, colour = risk_bar(0.02, 0.0)
    assert colour == "dim" and "█" not in bar


@pytest.mark.parametrize("delta,expected", [
    (timedelta(minutes=25), "25m"),
    (timedelta(hours=3), "3h"),
    (timedelta(days=2), "2d"),
])
def test_held_for(delta, expected):
    assert _held_for(datetime.now(UTC) - delta) == expected


def test_dashboard_renders_with_nothing_wired():
    """The empty state is the one that runs before anything else works."""
    TerminalDashboard().render(DashboardState().snapshot())


def test_snapshot_has_every_spec_panel():
    snapshot = DashboardState().snapshot()
    assert {"regime", "portfolio", "positions", "signals", "risk", "system"} <= set(snapshot)


def test_dashboard_state_has_no_way_to_trade():
    """A display that can act is no longer a display."""
    state = DashboardState()
    assert not hasattr(state, "order_executor")
    assert not any(n in dir(state) for n in ("submit_order", "close_position", "clear_halt"))


# ===========================================================================
# Publisher
# ===========================================================================

def test_publish_writes_valid_json(tmp_path):
    path = publish(demo_snapshot(), tmp_path / "state.json", source="demo")
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["source"] == "demo"


def test_publish_is_atomic(tmp_path):
    publish(demo_snapshot(), tmp_path / "state.json")
    assert not list(tmp_path.glob("*.tmp")), "a half-written file must never be readable"


def test_published_payload_carries_no_secrets():
    """The dashboard is deployed publicly. Anything in this file is public."""
    payload = build_payload({
        "system": {"api_key": "PKLEAKED", "account_number": "PA123", "paper": True},
        "positions": [{"symbol": "SPY", "order_id": "abc-123", "trade_id": "t-1"}],
        "risk": {"lock_file": "/Users/someone/trading_halted.lock", "halted": False},
    })
    blob = json.dumps(payload)
    for secret in ("PKLEAKED", "PA123", "abc-123", "t-1", "/Users/someone"):
        assert secret not in blob, f"{secret} leaked into the published payload"
    assert payload["system"]["paper"] is True     # non-secret fields survive


def test_demo_data_is_labelled_as_demo(tmp_path):
    """Fabricated numbers presented as a real account would be worse than an
    empty page."""
    payload = json.loads(publish_demo(tmp_path / "state.json").read_text())
    assert payload["source"] == "demo"
    assert "demo" in str(payload["notes"]).lower()


def test_demo_snapshot_is_shaped_like_the_real_one():
    demo = demo_snapshot()
    real_keys = set(DashboardState().snapshot())
    assert real_keys - {"session"} <= set(demo)


def test_demo_equity_history_is_plausible():
    history = demo_snapshot()["equity_history"]
    assert len(history) > 100
    assert all(p["peak"] >= p["equity"] - 1e-6 for p in history), "peak must be a high-water mark"
