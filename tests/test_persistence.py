"""
Durable state: migrations, the repository, and the partitioned bar cache.

The failures worth catching here are the ones that only show up on the second
run: a migration that is not idempotent, a cache that refetches everything
because a timestamp comparison went wrong, an order written twice because the
upsert key was not what anyone thought it was.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from data.repository import Repository, open_repository, utc_iso


@pytest.fixture
def repo(tmp_path):
    r = open_repository(tmp_path / "state.db")
    yield r
    r.close()


# -- migrations -------------------------------------------------------------

def test_migrations_apply_to_an_empty_database(tmp_path):
    r = Repository(tmp_path / "new.db")
    applied = r.migrate()
    assert "001_initial.sql" in applied
    assert r.counts() == {
        "runs": 0, "signals": 0, "orders": 0,
        "positions": 0, "equity_snapshots": 0, "breaker_events": 0,
    }
    r.close()


def test_migrations_are_idempotent(tmp_path):
    """Called unconditionally on every scheduled run, so applying twice must
    be a no-op rather than an error."""
    path = tmp_path / "twice.db"
    first = Repository(path)
    assert first.migrate() != []
    first.close()

    second = Repository(path)
    assert second.migrate() == []
    second.close()


def test_the_schema_version_is_recorded(repo):
    assert repo.schema_version() == "001_initial.sql"


def test_wal_is_enabled(repo):
    """Without it the publisher blocks behind the trading loop, or reads a
    partially applied transaction."""
    mode = repo.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


# -- runs -------------------------------------------------------------------

def test_a_run_is_recorded_before_it_can_fail(repo):
    """The row exists while status is still 'running'. A process that dies
    mid-bar leaves this behind, which is how "it failed" is told apart from
    "it never ran"."""
    run_id = repo.start_run("paper", "schedule")
    row = repo.recent_runs()[0]
    assert row["id"] == run_id
    assert row["status"] == "running"
    assert row["finished_at"] is None


def test_a_failed_run_keeps_its_error(repo):
    run_id = repo.start_run("paper")
    repo.finish_run(run_id, "failed", error="broker unreachable")
    row = repo.recent_runs()[0]
    assert row["status"] == "failed"
    assert "broker" in row["error"]


def test_last_successful_run_ignores_failures(repo):
    """What the dashboard's "updated X ago" badge reads. A failed run must not
    make stale data look fresh."""
    ok = repo.start_run("paper")
    repo.finish_run(ok, "ok")
    bad = repo.start_run("paper")
    repo.finish_run(bad, "failed", error="boom")

    assert repo.last_successful_run()["id"] == ok


def test_a_crashed_run_is_never_reported_as_successful(repo):
    repo.start_run("paper")            # left 'running', as a crash would
    assert repo.last_successful_run() is None


# -- orders -----------------------------------------------------------------

class FakeTrade:
    def __init__(self, **kwargs):
        self.trade_id = "t-1"
        self.order_id = "o-1"
        self.symbol = "COIN"
        self.side = "buy"
        self.approved_qty = 16
        self.fill_price = None
        self.filled_qty = 0.0
        self.status = "new"
        self.stop_loss = 163.0
        self.take_profit = None
        self.regime = "strong_bull"
        self.submitted_at = datetime.now(UTC)
        self.filled_at = None
        self.notes = []
        self.skipped_reason = None
        self.__dict__.update(kwargs)


def test_an_order_written_twice_updates_one_row(repo):
    """An order is written when submitted and again when it fills. The fill
    must land on the same row, not create a second one."""
    trade = FakeTrade()
    repo.record_order(trade, client_order_id="rt-COIN-buy-20260904")

    trade.status = "filled"
    trade.fill_price = 184.6
    trade.filled_qty = 16
    trade.filled_at = datetime.now(UTC)
    repo.record_order(trade, client_order_id="rt-COIN-buy-20260904")

    orders = repo.recent_orders()
    assert len(orders) == 1
    assert orders[0]["status"] == "filled"
    assert orders[0]["fill_price"] == 184.6


def test_two_different_signals_are_two_rows(repo):
    repo.record_order(FakeTrade(), client_order_id="rt-COIN-buy-20260904")
    repo.record_order(FakeTrade(trade_id="t-2"), client_order_id="rt-COIN-buy-20260905")
    assert len(repo.recent_orders()) == 2


def test_a_skipped_order_records_why(repo):
    """Skips are kept. A system that declined to trade looks identical to one
    with no signals unless the reason is stored."""
    repo.record_order(
        FakeTrade(skipped_reason="an equivalent buy order is already open"),
        client_order_id="rt-COIN-buy-20260904",
    )
    assert "already open" in repo.recent_orders()[0]["skipped_reason"]


def test_orders_can_be_filtered_by_status_and_date(repo):
    old = (datetime.now(UTC) - timedelta(days=10))
    repo.record_order(FakeTrade(submitted_at=old, status="canceled"),
                      client_order_id="a")
    repo.record_order(FakeTrade(trade_id="t-2", status="filled"), client_order_id="b")

    assert len(repo.recent_orders(status="filled")) == 1
    recent = repo.recent_orders(since=utc_iso(datetime.now(UTC) - timedelta(days=1)))
    assert len(recent) == 1


# -- positions --------------------------------------------------------------

def test_a_position_round_trips_from_open_to_closed(repo):
    repo.open_position("COIN", 16, 184.6, stop_price=163.0, regime="strong_bull")
    assert len(repo.open_positions()) == 1

    repo.close_position("COIN", 200.0, "target")
    assert repo.open_positions() == []

    closed = repo.closed_positions()[0]
    assert closed["exit_reason"] == "target"
    assert closed["realised_pnl"] == pytest.approx((200.0 - 184.6) * 16)


def test_realised_pnl_is_computed_in_one_place(repo):
    """Computed by close_position rather than by the caller, so it cannot mean
    something slightly different depending on which path closed the trade."""
    repo.open_position("SPY", 4, 100.0)
    repo.close_position("SPY", 90.0, "stop")
    assert repo.closed_positions()[0]["realised_pnl"] == pytest.approx(-40.0)


def test_holding_period_is_recorded(repo):
    entry = datetime.now(UTC) - timedelta(days=12)
    repo.open_position("SPY", 4, 100.0, entry_at=entry)
    repo.close_position("SPY", 110.0, "target")
    assert repo.closed_positions()[0]["holding_days"] == 12


def test_closing_a_position_that_is_not_open_is_survivable(repo):
    repo.close_position("NOPE", 100.0, "stop")     # logs, does not raise
    assert repo.closed_positions() == []


def test_expectancy_reports_its_own_shape(repo):
    """One outsized winner and nineteen losers is positive expectancy and not a
    strategy, so the win rate comes back with it."""
    for i, (entry, exit_price) in enumerate([(100, 90), (100, 90), (100, 140)]):
        repo.open_position(f"S{i}", 1, entry)
        repo.close_position(f"S{i}", exit_price, "stop")

    stats = repo.expectancy()
    assert stats["trades"] == 3
    assert stats["expectancy"] == pytest.approx(20 / 3)
    assert stats["win_rate"] == pytest.approx(1 / 3)
    assert stats["avg_loss"] == pytest.approx(-10.0)


def test_expectancy_on_an_empty_book_is_zero_not_an_error(repo):
    assert repo.expectancy()["trades"] == 0


# -- equity -----------------------------------------------------------------

def test_reprocessing_a_bar_overwrites_rather_than_duplicates(repo):
    repo.record_equity(100_000.0, bar_date="2026-09-04")
    repo.record_equity(101_000.0, bar_date="2026-09-04")

    curve = repo.equity_curve()
    assert len(curve) == 1
    assert curve[0]["equity"] == 101_000.0


def test_the_equity_curve_comes_back_oldest_first(repo):
    """Charts read left to right."""
    for day, equity in [("2026-09-02", 100.0), ("2026-09-03", 110.0),
                        ("2026-09-04", 105.0)]:
        repo.record_equity(equity, bar_date=day)

    curve = repo.equity_curve()
    assert [r["bar_date"] for r in curve] == ["2026-09-02", "2026-09-03", "2026-09-04"]


# -- breakers ---------------------------------------------------------------

def test_a_breaker_trip_is_recorded_and_stays_active(repo):
    repo.record_breaker("daily", "tripped", threshold=0.03, observed=0.04)
    assert repo.active_breakers() == ["daily"]


def test_clearing_a_breaker_makes_it_inactive(repo):
    repo.record_breaker("daily", "tripped")
    repo.record_breaker("daily", "cleared")
    assert repo.active_breakers() == []


# -- demo flag --------------------------------------------------------------

def test_a_run_alone_counts_as_real_data(repo):
    """A run that produced no signals is still real history, and showing
    invented numbers over it would be worse than an empty chart."""
    assert repo.has_data() is False
    repo.start_run("paper")
    assert repo.has_data() is True


# -- bar cache --------------------------------------------------------------

def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


class FakeDataClient:
    def __init__(self, frame):
        self.frame = frame
        self.requests = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        symbols = request.symbol_or_symbols
        symbols = [symbols] if isinstance(symbols, str) else list(symbols)

        start = _utc(request.start)
        end = _utc(request.end)
        window = self.frame[(self.frame.index >= start) & (self.frame.index <= end)]

        parts = {s: window for s in symbols}
        combined = pd.concat(parts, names=["symbol", "timestamp"])

        class Result:
            df = combined
        return Result()


class FakeAlpaca:
    def __init__(self, data_client):
        self.data_client = data_client


@pytest.fixture
def market_data(tmp_path):
    from data.market_data import MarketDataClient

    index = pd.date_range("2026-01-01", "2026-09-04", freq="B", tz="UTC")
    frame = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
         "volume": 1e6, "trade_count": 1000, "vwap": 100.2},
        index=index,
    )
    frame.index.name = "timestamp"
    client = FakeDataClient(frame)
    return MarketDataClient(FakeAlpaca(client), cache_dir=tmp_path / "cache"), client


def test_the_first_fetch_hits_the_api(market_data):
    md, client = market_data
    md.get_historical_bars(["SPY"], "1Day", start=datetime(2026, 6, 1, tzinfo=UTC))
    assert len(client.requests) == 1


def test_a_repeated_request_is_served_from_disk(market_data):
    """The previous scheme hashed the whole request into a filename, so a
    window one day longer shared nothing and refetched four years."""
    md, client = market_data
    start = datetime(2026, 6, 1, tzinfo=UTC)
    end = datetime(2026, 9, 4, tzinfo=UTC)

    md.get_historical_bars(["SPY"], "1Day", start=start, end=end)
    before = len(client.requests)
    md.get_historical_bars(["SPY"], "1Day", start=start, end=end)
    assert len(client.requests) == before, "a fully cached range still hit the API"


def test_only_the_missing_range_is_fetched(market_data):
    md, client = market_data
    end = datetime(2026, 9, 4, tzinfo=UTC)
    md.get_historical_bars(["SPY"], "1Day", start=datetime(2026, 6, 1, tzinfo=UTC), end=end)

    md.get_historical_bars(["SPY"], "1Day", start=datetime(2026, 5, 1, tzinfo=UTC), end=end)
    widened = client.requests[-1]
    fetched_days = (pd.Timestamp(widened.end) - pd.Timestamp(widened.start)).days
    assert fetched_days < 40, f"refetched {fetched_days} days for a one-month gap"


def test_a_sub_bar_gap_does_not_trigger_a_fetch(market_data):
    """`start` carries a time of day and the first stored bar sits at the
    session open, so a naive comparison finds a thirteen-hour gap at the head
    of every request and refetches it forever."""
    md, client = market_data
    start = datetime(2026, 6, 1, 15, 30, tzinfo=UTC)
    end = datetime(2026, 9, 4, 4, 0, tzinfo=UTC)

    md.get_historical_bars(["SPY"], "1Day", start=start, end=end)
    before = len(client.requests)
    md.get_historical_bars(["SPY"], "1Day", start=start, end=end)
    assert len(client.requests) == before


def test_each_symbol_gets_its_own_file(market_data, tmp_path):
    md, _ = market_data
    md.get_historical_bars(["SPY", "QQQ"], "1Day", start=datetime(2026, 6, 1, tzinfo=UTC))
    stored = sorted(p.name for p in (tmp_path / "cache" / "bars").glob("*.parquet"))
    assert stored == ["QQQ_1Day_adj.parquet", "SPY_1Day_adj.parquet"]


def test_adjusted_and_raw_are_stored_separately(market_data, tmp_path):
    """Features need adjusted prices, orders need raw ones. Mixing them would
    make a 4-for-1 split look like a 75% crash."""
    md, _ = market_data
    start = datetime(2026, 6, 1, tzinfo=UTC)
    md.get_historical_bars(["SPY"], "1Day", start=start, adjusted=True)
    md.get_historical_bars(["SPY"], "1Day", start=start, adjusted=False)

    stored = sorted(p.name for p in (tmp_path / "cache" / "bars").glob("*.parquet"))
    assert stored == ["SPY_1Day_adj.parquet", "SPY_1Day_raw.parquet"]


def test_the_returned_frame_keeps_the_timestamp_symbol_index(market_data):
    md, _ = market_data
    frame = md.get_historical_bars(["SPY", "QQQ"], "1Day",
                                   start=datetime(2026, 6, 1, tzinfo=UTC))
    assert list(frame.index.names) == ["timestamp", "symbol"]
    assert set(frame.index.get_level_values("symbol")) == {"SPY", "QQQ"}


def test_an_unwritable_cache_still_returns_the_fetched_bars(market_data, monkeypatch):
    """The cache is an optimisation and must never be load-bearing.

    This is the exact failure that broke the first scheduled run: pyarrow was
    absent on the runner, every parquet read and write raised, and because the
    return path read only from the cache, a successful API call came back as
    "No bars available". A cache failure should cost speed, not correctness.
    """
    md, client = market_data

    # Patched at the pandas level, which is where it actually failed, so the
    # module's own error handling is exercised rather than replaced.
    def no_engine(*args, **kwargs):
        raise ImportError("Unable to find a usable engine; tried using: 'pyarrow'")

    monkeypatch.setattr(pd, "read_parquet", no_engine)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_engine)

    frame = md.get_historical_bars(["SPY"], "1Day", start=datetime(2026, 6, 1, tzinfo=UTC))
    assert not frame.empty, "a broken cache swallowed a successful fetch"
    assert len(client.requests) == 1


def test_an_unreadable_cache_falls_back_for_every_symbol(market_data, monkeypatch):
    md, _ = market_data
    monkeypatch.setattr(pd, "read_parquet",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk gone")))

    frame = md.get_historical_bars(["SPY", "QQQ"], "1Day",
                                   start=datetime(2026, 6, 1, tzinfo=UTC))
    assert set(frame.index.get_level_values("symbol")) == {"SPY", "QQQ"}


def test_use_cache_false_always_fetches(market_data):
    md, client = market_data
    start = datetime(2026, 6, 1, tzinfo=UTC)
    md.get_historical_bars(["SPY"], "1Day", start=start)
    before = len(client.requests)
    md.get_historical_bars(["SPY"], "1Day", start=start, use_cache=False)
    assert len(client.requests) == before + 1


# ---------------------------------------------------------------------------
# Reconciliation: the dashboard showed five held positions as "not filled" and
# every scheduled run as failed. Four separate bugs, one regression test each.
# ---------------------------------------------------------------------------

class TestOrderSettlement:
    """Orders are recorded at submission and finish hours later, unattended."""

    def test_settle_order_writes_the_fill_back(self, tmp_path):
        from data.repository import open_repository

        repo = open_repository(tmp_path / "s.db")
        repo.migrate()
        with repo.tx() as conn:
            conn.execute(
                "INSERT INTO orders (order_id, client_order_id, symbol, side, "
                " order_type, quantity, status, submitted_at) "
                "VALUES ('oid-1','rt-X','X','buy','limit',10,'open','2026-01-01T00:00:00+00:00')")

        assert [r["symbol"] for r in repo.unsettled_orders()] == ["X"]

        repo.settle_order("oid-1", status="filled", fill_price=12.5,
                          filled_qty=10, filled_at="2026-01-02T14:30:00+00:00")

        row = repo.conn.execute("SELECT * FROM orders WHERE order_id='oid-1'").fetchone()
        assert row["status"] == "filled"
        assert row["fill_price"] == 12.5
        assert row["filled_qty"] == 10
        # ...and it drops out of the unsettled list, so it is not re-polled forever.
        assert repo.unsettled_orders() == []

    def test_terminal_orders_are_never_polled(self, tmp_path):
        from data.repository import open_repository

        repo = open_repository(tmp_path / "t.db")
        repo.migrate()
        with repo.tx() as conn:
            for i, status in enumerate(("filled", "cancelled", "rejected", "expired")):
                conn.execute(
                    "INSERT INTO orders (order_id, client_order_id, symbol, side, "
                    " order_type, quantity, status, submitted_at) "
                    f"VALUES ('o{i}','c{i}','X','buy','limit',1,'{status}','2026-01-01T00:00:00+00:00')")
        assert repo.unsettled_orders() == []
