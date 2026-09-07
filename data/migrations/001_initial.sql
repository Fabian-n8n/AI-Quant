-- Initial schema.
--
-- Six tables, one job each. No ORM: the queries are short enough to read, and
-- an ORM would hide the one thing that matters here, which is exactly what
-- gets written when.
--
-- Times are ISO-8601 strings in UTC. SQLite has no date type, and storing
-- epoch integers would make the file unreadable with the sqlite3 CLI, which is
-- the tool most likely to be reached for when something has gone wrong.
--
-- Money is REAL. This is a record of what the broker did, not an accounting
-- ledger that has to balance to the cent.

-- Every scheduled execution. The table the dashboard's "last updated" reads,
-- and the only place a failed run leaves a trace: a crash writes status
-- 'failed' with the error, where a crashed process writes nothing at all.
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT    NOT NULL,
    finished_at   TEXT,
    status        TEXT    NOT NULL,     -- running | ok | failed | halted
    mode          TEXT    NOT NULL,     -- live | dry-run | backtest
    trigger       TEXT,                 -- schedule | manual | loop
    bars_processed INTEGER DEFAULT 0,
    orders_submitted INTEGER DEFAULT 0,
    regime        TEXT,
    equity        REAL,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status  ON runs (status, started_at DESC);

-- What the strategy wanted, including what the risk manager refused. The
-- rejections are the more interesting half: a system that never trades looks
-- identical to one with no signals unless you keep them.
CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER REFERENCES runs (id),
    created_at    TEXT    NOT NULL,
    bar_date      TEXT    NOT NULL,
    symbol        TEXT    NOT NULL,
    direction     TEXT    NOT NULL,
    regime        TEXT,
    regime_confidence REAL,
    entry_price   REAL,
    stop_loss     REAL,
    take_profit   REAL,
    approved      INTEGER NOT NULL DEFAULT 0,
    rejection_reason TEXT,
    approved_qty  REAL,
    approved_notional REAL,
    reasoning     TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_bar ON signals (bar_date DESC, symbol);

-- Orders as submitted, updated in place as the broker moves them along.
-- `client_order_id` is unique because it is the idempotency key: two rows with
-- the same one would mean the guard in order_executor failed.
CREATE TABLE IF NOT EXISTS orders (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER REFERENCES runs (id),
    trade_id      TEXT,
    order_id      TEXT,
    client_order_id TEXT UNIQUE,
    symbol        TEXT    NOT NULL,
    side          TEXT    NOT NULL,
    order_type    TEXT    NOT NULL,
    quantity      REAL    NOT NULL,
    submitted_price REAL,
    fill_price    REAL,
    filled_qty    REAL    DEFAULT 0,
    status        TEXT    NOT NULL,
    stop_loss     REAL,
    take_profit   REAL,
    regime        TEXT,
    submitted_at  TEXT    NOT NULL,
    filled_at     TEXT,
    skipped_reason TEXT,
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_submitted ON orders (submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_symbol    ON orders (symbol, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status    ON orders (status);

-- Open and closed positions in one table, separated by exit_at IS NULL.
-- Splitting them would mean moving a row between tables on exit, which is a
-- transaction that can half-happen.
CREATE TABLE IF NOT EXISTS positions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT    NOT NULL,
    quantity      REAL    NOT NULL,
    entry_price   REAL    NOT NULL,
    entry_at      TEXT    NOT NULL,
    exit_price    REAL,
    exit_at       TEXT,
    exit_reason   TEXT,                 -- target | stop | trailing_stop | breaker | manual
    stop_price    REAL,
    current_price REAL,
    unrealised_pnl REAL,
    realised_pnl  REAL,
    holding_days  INTEGER,
    regime_at_entry TEXT,
    trade_id      TEXT
);
CREATE INDEX IF NOT EXISTS idx_positions_open ON positions (exit_at, symbol);
CREATE INDEX IF NOT EXISTS idx_positions_exit ON positions (exit_at DESC);

-- One row per processed bar. The equity curve the dashboard draws.
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER REFERENCES runs (id),
    captured_at   TEXT    NOT NULL,
    bar_date      TEXT,
    equity        REAL    NOT NULL,
    cash          REAL,
    positions_value REAL,
    peak_equity   REAL,
    daily_pnl     REAL,
    open_positions INTEGER,
    regime        TEXT,
    UNIQUE (bar_date)
);
CREATE INDEX IF NOT EXISTS idx_equity_captured ON equity_snapshots (captured_at DESC);

-- Circuit breaker trips and clears. Kept separately from the log because
-- "has a breaker ever fired" must be answerable without parsing JSONL.
CREATE TABLE IF NOT EXISTS breaker_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER REFERENCES runs (id),
    occurred_at   TEXT    NOT NULL,
    breaker       TEXT    NOT NULL,     -- daily | weekly | peak | error_budget
    state         TEXT    NOT NULL,     -- tripped | cleared
    threshold     REAL,
    observed      REAL,
    equity        REAL,
    detail        TEXT
);
CREATE INDEX IF NOT EXISTS idx_breakers_occurred ON breaker_events (occurred_at DESC);
