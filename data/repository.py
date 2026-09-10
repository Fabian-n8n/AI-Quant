"""
Durable state: SQLite via stdlib sqlite3, plain SQL, thin repository.

Phase 10.

WHY SQLITE AND NOT POSTGRES
---------------------------
One writer, one reader, a few thousand rows a year. Postgres would add a
service to run, a connection string to keep secret, and a backup story, in
exchange for concurrency this system will never have. SQLite is a file, and a
file can be committed by the scheduled job and read by the publisher without
either knowing the other exists.

WHY NO ORM
----------
The queries are twenty lines total and each one is read more often than it is
written. An ORM would hide the only thing that actually matters here, which is
exactly what gets written and when.

MIGRATIONS
----------
Numbered .sql files in data/migrations/, applied in filename order, recorded in
schema_version. Idempotent: applying an already-applied migration is a no-op,
so the scheduled job can call `migrate()` unconditionally on every run rather
than needing to know whether the database is new.

WAL
---
Write-ahead logging, so the publisher can read a consistent snapshot while a
run is mid-write. Without it the dashboard export blocks behind the trading
loop, or worse, reads a partially applied transaction.

TIMES
-----
Every timestamp is stored as an ISO-8601 UTC string. SQLite has no date type.
Epoch integers would be smaller and completely unreadable from the sqlite3
CLI, which is the tool anyone reaches for when something has gone wrong.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "state.db"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def utc_iso(moment: datetime | None = None) -> str:
    """The one timestamp format this database stores."""
    moment = moment or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _iso(value: Any) -> str | None:
    """Coerce whatever a caller has (datetime, Timestamp, str, None) to storage."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return utc_iso(value)
    if isinstance(value, str):
        return value
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        return utc_iso(to_pydatetime())
    return str(value)


class Repository:
    """Every read and write of durable state goes through here.

    Not a general query interface on purpose. One method per question the
    system actually asks means the set of queries is enumerable, which is what
    lets the schema change without hunting for SQL scattered across modules.
    """

    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # -- connection ---------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, timeout=30.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Repository:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def tx(self):
        """One transaction. Commits on success, rolls back on anything else."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- migrations ---------------------------------------------------------

    def migrate(self, directory: Path = MIGRATIONS_DIR) -> list[str]:
        """Apply every migration not yet applied. Safe to call on every run."""
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            "  filename TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        self.conn.commit()

        applied = {
            row["filename"]
            for row in self.conn.execute("SELECT filename FROM schema_version")
        }
        pending = sorted(p for p in directory.glob("*.sql") if p.name not in applied)

        for migration in pending:
            logger.info("applying migration %s", migration.name)
            with self.tx() as conn:
                conn.executescript(migration.read_text())
                conn.execute(
                    "INSERT INTO schema_version (filename, applied_at) VALUES (?, ?)",
                    (migration.name, utc_iso()),
                )
        return [p.name for p in pending]

    def schema_version(self) -> str | None:
        row = self.conn.execute(
            "SELECT filename FROM schema_version ORDER BY filename DESC LIMIT 1"
        ).fetchone()
        return row["filename"] if row else None

    # -- runs ---------------------------------------------------------------

    def start_run(self, mode: str, trigger: str = "manual") -> int:
        """Open a run row. Returns its id, which every later write references.

        Written at the start rather than the end deliberately. A run that
        crashes mid-way still leaves a row saying it started and never
        finished, which is the difference between "it failed" and "it never
        ran" - and those need different responses.
        """
        with self.tx() as conn:
            cursor = conn.execute(
                "INSERT INTO runs (started_at, status, mode, trigger) VALUES (?, ?, ?, ?)",
                (utc_iso(), "running", mode, trigger),
            )
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str = "ok", *, error: str | None = None,
                   bars_processed: int = 0, orders_submitted: int = 0,
                   regime: str | None = None, equity: float | None = None) -> None:
        with self.tx() as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ?, status = ?, error = ?, "
                "bars_processed = ?, orders_submitted = ?, regime = ?, equity = ? "
                "WHERE id = ?",
                (utc_iso(), status, error, bars_processed, orders_submitted,
                 regime, equity, run_id),
            )

    def last_successful_run(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runs WHERE status = 'ok' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()

    def recent_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ))

    # -- signals ------------------------------------------------------------

    def record_signal(self, signal, decision, run_id: int | None = None) -> int:
        with self.tx() as conn:
            cursor = conn.execute(
                "INSERT INTO signals (run_id, created_at, bar_date, symbol, direction, "
                " regime, regime_confidence, entry_price, stop_loss, take_profit, "
                " approved, rejection_reason, approved_qty, approved_notional, reasoning) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, utc_iso(), _iso(signal.timestamp), signal.symbol,
                    getattr(signal.direction, "value", str(signal.direction)),
                    signal.regime_name, signal.regime_probability,
                    signal.entry_price, signal.stop_loss, signal.take_profit,
                    1 if decision.approved else 0,
                    getattr(decision, "rejection_reason", None),
                    getattr(decision, "approved_quantity", None),
                    getattr(decision, "approved_notional", None),
                    signal.reasoning,
                ),
            )
        return int(cursor.lastrowid)

    def recent_signals(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,)
        ))

    # -- orders -------------------------------------------------------------

    def record_order(self, trade, run_id: int | None = None,
                     client_order_id: str | None = None) -> int:
        """Insert or update by client_order_id.

        UPSERT rather than INSERT because an order is written once when
        submitted and again when it fills, and the fill must land on the same
        row. The unique constraint on client_order_id is what makes that
        possible, and it doubles as a check on the idempotency layer: two rows
        for one key would mean the guard in order_executor let a duplicate
        through.
        """
        coid = client_order_id or trade.trade_id
        with self.tx() as conn:
            cursor = conn.execute(
                "INSERT INTO orders (run_id, trade_id, order_id, client_order_id, symbol, "
                " side, order_type, quantity, submitted_price, fill_price, filled_qty, "
                " status, stop_loss, take_profit, regime, submitted_at, filled_at, "
                " skipped_reason, notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT (client_order_id) DO UPDATE SET "
                " order_id = excluded.order_id, fill_price = excluded.fill_price, "
                " filled_qty = excluded.filled_qty, status = excluded.status, "
                " filled_at = excluded.filled_at, notes = excluded.notes",
                (
                    run_id, trade.trade_id, trade.order_id, coid, trade.symbol,
                    getattr(trade.side, "value", str(trade.side)),
                    "limit", trade.approved_qty,
                    getattr(trade, "submitted_price", None), trade.fill_price,
                    trade.filled_qty,
                    getattr(trade.status, "value", str(trade.status)),
                    trade.stop_loss, trade.take_profit, trade.regime,
                    _iso(trade.submitted_at) or utc_iso(), _iso(trade.filled_at),
                    getattr(trade, "skipped_reason", None),
                    "; ".join(trade.notes) if trade.notes else None,
                ),
            )
        return int(cursor.lastrowid)

    def unsettled_orders(self) -> list[sqlite3.Row]:
        """Orders we recorded but never saw the end of.

        An order is written when it is submitted, and a limit order submitted
        after the close does not fill until the next session opens -- long
        after the process that placed it has exited. Nothing else ever revisits
        the row, so without this the table says 'not filled' forever while the
        broker holds a filled position. That is precisely what made the
        dashboard show eight resting orders and no trades.
        """
        return self.conn.execute(
            "SELECT id, order_id, client_order_id, symbol, status FROM orders "
            "WHERE order_id IS NOT NULL "
            "  AND status NOT IN ('filled','cancelled','rejected','expired') "
            "ORDER BY submitted_at DESC"
        ).fetchall()

    def settle_order(self, order_id: str, *, status: str,
                     fill_price: float | None = None, filled_qty: float | None = None,
                     filled_at: Any = None) -> None:
        """Write back what the broker says became of an order."""
        with self.tx() as conn:
            conn.execute(
                "UPDATE orders SET status = ?, "
                "  fill_price = COALESCE(?, fill_price), "
                "  filled_qty = COALESCE(?, filled_qty), "
                "  filled_at  = COALESCE(?, filled_at) "
                "WHERE order_id = ?",
                (status, fill_price, filled_qty, _iso(filled_at), order_id),
            )

    def recent_orders(self, limit: int = 100, status: str | None = None,
                      since: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM orders WHERE 1=1"
        params: list[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if since:
            sql += " AND submitted_at >= ?"
            params.append(since)
        sql += " ORDER BY submitted_at DESC LIMIT ?"
        params.append(limit)
        return list(self.conn.execute(sql, params))

    # -- positions ----------------------------------------------------------

    def open_position(self, symbol: str, quantity: float, entry_price: float, *,
                      stop_price: float | None = None, regime: str | None = None,
                      trade_id: str | None = None, entry_at: Any = None) -> int:
        with self.tx() as conn:
            cursor = conn.execute(
                "INSERT INTO positions (symbol, quantity, entry_price, entry_at, "
                " stop_price, regime_at_entry, trade_id) VALUES (?,?,?,?,?,?,?)",
                (symbol, quantity, entry_price, _iso(entry_at) or utc_iso(),
                 stop_price, regime, trade_id),
            )
        return int(cursor.lastrowid)

    def update_open_position(self, symbol: str, *, current_price: float | None = None,
                             stop_price: float | None = None,
                             unrealised_pnl: float | None = None,
                             holding_days: int | None = None) -> None:
        sets, params = [], []
        for column, value in (("current_price", current_price), ("stop_price", stop_price),
                              ("unrealised_pnl", unrealised_pnl),
                              ("holding_days", holding_days)):
            if value is not None:
                sets.append(f"{column} = ?")
                params.append(value)
        if not sets:
            return
        params.append(symbol)
        with self.tx() as conn:
            conn.execute(
                f"UPDATE positions SET {', '.join(sets)} "
                f"WHERE symbol = ? AND exit_at IS NULL", params
            )

    def close_position(self, symbol: str, exit_price: float, reason: str,
                       exit_at: Any = None) -> None:
        """Close the open row for `symbol`, computing realised P&L and duration.

        Both derived fields are computed here rather than by the caller so
        there is exactly one definition of each. A realised P&L that means
        something slightly different depending on which code path closed the
        position is worse than not recording it.
        """
        row = self.conn.execute(
            "SELECT id, quantity, entry_price, entry_at FROM positions "
            "WHERE symbol = ? AND exit_at IS NULL ORDER BY entry_at DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        if row is None:
            logger.warning("close_position: no open row for %s", symbol)
            return

        closed_at = _iso(exit_at) or utc_iso()
        realised = (exit_price - row["entry_price"]) * row["quantity"]
        try:
            held = (datetime.fromisoformat(closed_at)
                    - datetime.fromisoformat(row["entry_at"])).days
        except (TypeError, ValueError):
            held = None

        with self.tx() as conn:
            conn.execute(
                "UPDATE positions SET exit_price = ?, exit_at = ?, exit_reason = ?, "
                " realised_pnl = ?, holding_days = ?, unrealised_pnl = 0 WHERE id = ?",
                (exit_price, closed_at, reason, realised, held, row["id"]),
            )

    def open_positions(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM positions WHERE exit_at IS NULL ORDER BY entry_at DESC"
        ))

    def closed_positions(self, limit: int = 100) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM positions WHERE exit_at IS NOT NULL "
            "ORDER BY exit_at DESC LIMIT ?", (limit,)
        ))

    def closed_trade_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM positions WHERE exit_at IS NOT NULL"
        ).fetchone()
        return int(row["n"])

    def expectancy(self) -> dict[str, float]:
        """Average P&L per closed trade, plus the win rate behind it.

        Expectancy alone hides its own shape: one outsized winner and nineteen
        losers is positive and not a strategy. The win rate and the two average
        sizes are returned with it so the number can be read honestly.
        """
        rows = list(self.conn.execute(
            "SELECT realised_pnl FROM positions "
            "WHERE exit_at IS NOT NULL AND realised_pnl IS NOT NULL"
        ))
        if not rows:
            return {"trades": 0, "expectancy": 0.0, "win_rate": 0.0,
                    "avg_win": 0.0, "avg_loss": 0.0}

        pnls = [float(r["realised_pnl"]) for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return {
            "trades": len(pnls),
            "expectancy": sum(pnls) / len(pnls),
            "win_rate": len(wins) / len(pnls),
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        }

    # -- equity -------------------------------------------------------------

    def record_equity(self, equity: float, *, bar_date: Any = None,
                      cash: float | None = None, positions_value: float | None = None,
                      peak_equity: float | None = None, daily_pnl: float | None = None,
                      open_positions: int | None = None, regime: str | None = None,
                      run_id: int | None = None) -> None:
        """One row per bar. Re-processing a bar overwrites rather than duplicates."""
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO equity_snapshots (run_id, captured_at, bar_date, equity, "
                " cash, positions_value, peak_equity, daily_pnl, open_positions, regime) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT (bar_date) DO UPDATE SET "
                " equity = excluded.equity, cash = excluded.cash, "
                " positions_value = excluded.positions_value, "
                " peak_equity = excluded.peak_equity, daily_pnl = excluded.daily_pnl, "
                " open_positions = excluded.open_positions, regime = excluded.regime, "
                " captured_at = excluded.captured_at",
                (run_id, utc_iso(), _iso(bar_date), equity, cash, positions_value,
                 peak_equity, daily_pnl, open_positions, regime),
            )

    def equity_curve(self, limit: int = 400) -> list[sqlite3.Row]:
        rows = list(self.conn.execute(
            "SELECT * FROM equity_snapshots ORDER BY bar_date DESC LIMIT ?", (limit,)
        ))
        return list(reversed(rows))

    # -- breakers -----------------------------------------------------------

    def record_breaker(self, breaker: str, state: str, *, threshold: float | None = None,
                       observed: float | None = None, equity: float | None = None,
                       detail: str | None = None, run_id: int | None = None) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO breaker_events (run_id, occurred_at, breaker, state, "
                " threshold, observed, equity, detail) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, utc_iso(), breaker, state, threshold, observed, equity, detail),
            )

    def recent_breakers(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM breaker_events ORDER BY occurred_at DESC LIMIT ?", (limit,)
        ))

    def active_breakers(self) -> list[str]:
        """Breakers whose most recent event was a trip rather than a clear.

        Ordered by id, not by occurred_at. Timestamps are stored to the second,
        so a breaker tripped and cleared inside the same second would match
        both rows and report as still tripped. The id is monotonic and cannot
        tie, which for a "is trading currently blocked" query is the property
        that matters.
        """
        rows = self.conn.execute(
            "SELECT breaker, state FROM breaker_events e WHERE id = ("
            "  SELECT MAX(id) FROM breaker_events WHERE breaker = e.breaker)"
        )
        return [r["breaker"] for r in rows if r["state"] == "tripped"]

    # -- summary ------------------------------------------------------------

    def has_data(self) -> bool:
        """Is there any real activity here, or is this an empty database?

        What the dashboard's demo banner keys on. Runs alone are enough: a run
        that produced no signals is still real history, and showing invented
        data over it would be worse than showing an empty chart.
        """
        row = self.conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        return int(row["n"]) > 0

    def counts(self) -> dict[str, int]:
        out = {}
        for table in ("runs", "signals", "orders", "positions",
                      "equity_snapshots", "breaker_events"):
            row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            out[table] = int(row["n"])
        return out


def open_repository(path: Path | str = DEFAULT_DB, migrate: bool = True) -> Repository:
    """The normal entry point: open, migrate, return."""
    repo = Repository(path)
    if migrate:
        repo.migrate()
    return repo
