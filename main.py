"""
regime-trader entry point. The complete trading engine.

    python main.py                                          # live/paper trading
    python main.py --dry-run                                # full pipeline, no orders
    python main.py --mode backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
    python main.py --mode backtest --compare --export
    python main.py --mode backtest --stress-test --mc-sims 100
    python main.py --train-only --symbols SPY
    python main.py --dashboard
    python main.py --status                                 # build state, no broker

Three things about this file that are not obvious from the spec:

**The risk manager vetoes increases, not decreases.** `validate_signal` is on
the path that buys. Selling down to a lower target allocation bypasses it
entirely. A rejection on the reduce path would trap the system in an oversized
position at exactly the moment the regime turned against it, which inverts what
the risk layer is for. See `_reduce_to_target`.

**Shutdown leaves positions open, so stops must be real orders.** The spec says
do not close positions on shutdown because the stops are in place. That is only
true if the stops exist at the broker as resting orders. A stop held as a float
in `TrackedPosition.stop_loss` protects nothing once the process exits. Startup
and shutdown both verify this and shout if it is not so.

**peak_equity has to survive a restart.** It lives in state_snapshot.json and is
restored as `max(saved, current)`. Without that, every restart re-bases the peak
to the already-drawn-down equity, and `max_dd_from_peak` can never fire. A
crash-restart loop would silently disarm the one breaker that never resets.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

PHASES = [
    ("1", "Project scaffolding and environment setup", True),
    ("2", "HMM regime detection engine", True),
    ("3", "Volatility-based allocation strategies", True),
    ("4", "Walk-forward backtesting and validation", True),
    ("5", "Risk management layer", True),
    ("6", "Alpaca broker integration", True),
    ("7", "Main loop and orchestration", True),
    ("8", "Monitoring, alerts and dashboard UI", True),
    ("9", "Integration testing and documentation", True),
]

logger = logging.getLogger("regime-trader.engine")

#: Distinguishes "argument not supplied" from an explicit None.
_UNSET = object()


# ===========================================================================
# Session state
# ===========================================================================

@dataclass
class SessionState:
    """What has to survive a restart.

    Everything here is either a high-water mark or a counter that a breaker
    reads. None of it can be recovered from the broker: Alpaca knows today's
    equity, it does not know what this system's peak equity was, how many trades
    it has placed today, or which breakers are latched.

    Restoring `peak_equity` is the load-bearing part. See the module docstring.
    """
    session_id: str = ""
    started_at: str = ""
    saved_at: str = ""

    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    week_start_equity: float = 0.0
    day_start_date: str = ""
    week_start_date: str = ""
    equity_at_save: float = 0.0
    #: The account size the baselines above were recorded against. Changing
    #: `risk.account_size_override` rescales every dollar figure, and a peak
    #: recorded in the old scale reads as a catastrophic drawdown in the new one.
    sizing_basis: float = 0.0

    daily_trades: int = 0
    bars_processed: int = 0
    last_bar_timestamp: str = ""
    last_regime: str = "unknown"
    last_regime_confidence: float = 0.0

    stops: dict[str, float] = field(default_factory=dict)
    #: One point per processed bar, capped. Enough for the dashboard's equity
    #: chart without turning the snapshot into an unbounded append-only log.
    equity_history: list = field(default_factory=list)
    regime_history: list = field(default_factory=list)
    breaker_daily_tripped: str = "none"
    breaker_weekly_tripped: str = "none"
    breaker_peak_tripped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SessionState:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def save(self, path: Path) -> Path:
        """Atomic write. A snapshot truncated by a crash mid-write would be
        worse than no snapshot: it would restore a peak_equity of zero."""
        self.saved_at = datetime.now(UTC).isoformat()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with open(temp, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        temp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> SessionState | None:
        path = Path(path)
        if not path.exists():
            return None
        try:
            with open(path) as fh:
                return cls.from_dict(json.load(fh))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("state_snapshot.json is unreadable (%s). Starting fresh.", exc)
            return None


@dataclass
class BarOutcome:
    """What one pass of the main loop did. Returned so it can be asserted on."""
    timestamp: Any | None = None
    regime: str = "unknown"
    confidence: float = 0.0
    confirmed: bool = False
    uncertain: bool = False
    target_allocation: float = 0.0
    current_allocation: float = 0.0
    rebalanced: bool = False
    signals: int = 0
    approved: int = 0
    rejected: int = 0
    submitted: int = 0
    deduped: int = 0          # approved, then skipped as an equivalent already open
    stops_updated: int = 0
    candidates: int = 0
    top_pick: str = ""
    breaker_state: str = "normal"
    halted: bool = False
    skipped: str = ""
    errors: list[str] = field(default_factory=list)


class EngineError(RuntimeError):
    """Startup failed in a way that makes trading unsafe."""


class BrokerUnavailable(RuntimeError):
    """The broker stayed unreachable across the whole retry budget."""


def _EventTypes():
    """Late import of the event enum. Keeps `import main` cheap for --status."""
    from monitoring.logger import EventType

    return EventType


class _RefusingExecutor:
    """Stands in for the order executor during a dry run.

    Not a flag checked before each submit: the capability is removed. A dry run
    that places an order because one code path forgot to check `self.dry_run` is
    the exact failure this prevents, so the object simply cannot submit.
    """

    def __init__(self) -> None:
        self.attempts: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str) -> Callable:
        def refuse(*args: Any, **kwargs: Any):
            self.attempts.append((name, args, kwargs))
            raise EngineError(
                f"dry run: refused to call order_executor.{name}(). "
                f"No order may leave the process in this mode."
            )
        return refuse


# ===========================================================================
# The engine
# ===========================================================================

class TradingEngine:
    """Startup, main loop, shutdown.

    Every collaborator is injectable so the loop can be tested without a broker,
    a network or a clock. Nothing here reaches for a global.
    """

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        dry_run: bool = False,
        symbols: list[str] | None = None,
        timeframe: str | None = None,
        client=None,
        market_data=None,
        trading_logger=None,
        alerts=None,
        snapshot_path: Path | None = None,
        lock_file: Path | None = None,
        model_path: Path | None = None,
        allow_live: bool = False,
        publish_path: Path | None = None,
        once: bool = False,
        db_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self.allow_live = allow_live
        self.once = once
        self.db_path = db_path

        self.orch = settings.get("orchestration", {})
        self.risk_config = settings["risk"]
        self.hmm_config = dict(settings["hmm"])
        self.monitoring_config = settings.get("monitoring", {})

        self.symbols = list(symbols or settings["broker"]["symbols"])
        self.timeframe = timeframe or settings["broker"]["timeframe"]
        self.mode = "dry-run" if dry_run else "live"

        self.snapshot_path = Path(snapshot_path or ROOT / self.orch.get(
            "state_snapshot_path", "state_snapshot.json"))
        self.lock_file = Path(lock_file) if lock_file else None
        self.model_path = Path(model_path) if model_path else None
        self.publish_path = Path(publish_path) if publish_path else None

        # Collaborators, wired in startup() unless injected.
        self.client = client
        self.market_data = market_data
        self.hmm = None
        self.orchestrator = None
        self.risk_manager = None
        self.position_tracker = None
        self.order_executor = None
        self.dashboard_state = None
        self.dashboard = None

        self.log = trading_logger
        self.alerts = alerts

        # Session.
        self.session = SessionState()
        self.account = None
        self.portfolio = None
        self.regime_state = None
        self.candidates: list = []
        self.previous_regime: str | None = None
        self.bars: dict[str, Any] = {}
        self.features = None

        # Cadence. A swing system decides once per session; re-reading the same
        # daily bar every minute produces the same signal sixty times an hour.
        from core.calendar import parse_run_after

        schedule = self.orch.get("schedule", {}) or {}
        self.schedule_mode = str(schedule.get("mode", "daily_close")).lower()
        self.run_after = parse_run_after(schedule.get("run_after", "16:15"))
        self.monitor_seconds = float(schedule.get("monitor_seconds", 300))
        self.calendar = None

        # Durable state. Opened in startup so a failure to migrate is a startup
        # failure rather than something discovered mid-bar.
        self.repo = None
        self.run_id: int | None = None

        self.started_at: datetime | None = None
        self.bars_processed = 0
        self.market_open: bool | None = None
        self.next_open: datetime | None = None
        self.data_feed_healthy = True
        self.api_latency_ms: float | None = None
        self.target_allocation: float | None = None
        self.consecutive_errors = 0
        self.is_paper = True
        self.orders_submitted = 0
        self.signals_rejected = 0
        self._bar_event = threading.Event()
        self._stop = threading.Event()
        self._shutdown_done = False
        self._streams: list[Any] = []

    # -- small helpers ------------------------------------------------------

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    def _emit(self, event, message: str = "", **fields: Any) -> dict[str, Any]:
        if self.log is not None:
            return self.log.log_event(event, message, **fields)
        logger.info(message or getattr(event, "value", event))
        return {}

    def _broker_retry(self, fn: Callable, *args: Any, what: str = "broker call", **kwargs: Any):
        """3 retries with exponential backoff, per the spec's error handling.

        `AlpacaClient` already retries its own reads. This wraps the composite
        operations the loop performs, where a partial failure halfway through a
        refresh needs the whole refresh redone rather than one call retried.
        """
        from monitoring.logger import EventType

        attempts = max(1, int(self.orch.get("broker_retry_attempts", 3)))
        delay = float(self.orch.get("broker_retry_base_delay", 2.0))
        last: BaseException | None = None

        for attempt in range(1, attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last = exc
                if attempt == attempts:
                    break
                wait = delay * (2 ** (attempt - 1))
                logger.warning("%s failed (%s), retry %d/%d in %.1fs",
                               what, exc, attempt, attempts, wait)
                self._stop.wait(wait)

        self.data_feed_healthy = False
        self._emit(EventType.DATA_FEED_DOWN,
                   f"{what} failed after {attempts} attempts: {last}", error=str(last))
        if self.alerts is not None:
            # Market data failing means trading blind; the trading API failing
            # means orders may not be arriving. Different alerts, different fixes.
            if "bars" in what or "data" in what:
                self.alerts.alert_data_feed_down(f"{what}: {last}")
            else:
                self.alerts.alert_broker_down(f"{what}: {last}")
        raise BrokerUnavailable(f"{what} failed after {attempts} attempts: {last}") from last

    # =======================================================================
    # STARTUP
    # =======================================================================

    def startup(self) -> TradingEngine:
        """The eight startup steps, in the spec's order.

        Any failure here raises. A half-started engine that trades with, say, an
        unsynced position tracker is more dangerous than one that never ran.
        """
        self.started_at = datetime.now(UTC)
        self._setup_logging()
        self._open_repository()         # 0

        self._connect_broker()          # 1
        self._check_market_hours()      # 2
        self._load_or_train_model()     # 3
        self._init_risk_manager()       # 4
        self._init_position_tracker()   # 5
        self._restore_session_state()   # 6
        self._start_data_feeds()        # 7
        self._print_system_state()      # 8
        return self

    # -- 1. broker ----------------------------------------------------------

    def _setup_logging(self) -> None:
        from monitoring.alerts import AlertManager
        from monitoring.logger import TradingLogger

        if self.log is None:
            self.log = TradingLogger().setup()
        if self.alerts is None:
            self.alerts = AlertManager(
                self.monitoring_config,
                rate_limit_minutes=int(self.monitoring_config.get("alert_rate_limit_minutes", 15)),
                trading_logger=self.log,
            )

    def _open_repository(self) -> None:
        """Open state.db, migrate, and open a run row.

        The run row is written now, before anything can fail, precisely so that
        a crash leaves evidence. A process that dies mid-bar writes nothing on
        the way out, and "the job never ran" and "the job ran and blew up" need
        different responses from whoever reads the dashboard next.
        """
        from data.repository import DEFAULT_DB, open_repository

        try:
            self.repo = open_repository(self.db_path or DEFAULT_DB)
            mode = "dry-run" if self.dry_run else ("paper" if self.is_paper else "live")
            trigger = "manual" if self.once else "loop"
            self.run_id = self.repo.start_run(mode, trigger)
            logger.debug("run %d recorded in %s", self.run_id, self.repo.path)
        except Exception as exc:
            # Persistence is for the record, not for trading. Losing it should
            # not stop the system from managing real positions.
            logger.warning("state database unavailable (%s), continuing without it", exc)
            self.repo = None

    def _record(self, method: str, *args, **kwargs) -> None:
        """Best-effort write to state.db. Never lets bookkeeping break a bar."""
        if self.repo is None:
            return
        try:
            getattr(self.repo, method)(*args, run_id=self.run_id, **kwargs)
        except Exception as exc:
            logger.debug("state write %s failed: %s", method, exc)

    def _connect_broker(self) -> None:
        """Load config, connect to Alpaca, verify account."""
        from broker.alpaca_client import AlpacaClient
        from config import assert_paper_mode

        if not self.allow_live:
            assert_paper_mode(self.settings)

        if self.client is None:
            self.client = AlpacaClient(paper=not self.allow_live or None)

        self.account = self._broker_retry(self.client.connect, what="broker connect")
        self.is_paper = self.account.is_paper

        if not self.account.is_tradeable:
            raise EngineError(
                f"Account is not tradeable: status {self.account.status}, "
                f"trading_blocked={self.account.trading_blocked}, "
                f"account_blocked={self.account.account_blocked}"
            )
        if not self.is_paper and not self.allow_live:
            raise EngineError(
                "Connected account is LIVE but --i-understand-live was not passed. "
                "Refusing to continue."
            )

        if self.market_data is None:
            from data.market_data import MarketDataClient

            self.market_data = MarketDataClient(self.client)

    # -- 2. market hours ----------------------------------------------------

    def _check_market_hours(self) -> None:
        """Check market hours. Logs the next open; the loop handles the waiting."""
        self._calendar()
        if self.schedule_mode == "daily_close":
            session = self._calendar().last_completed_session()
            due = self._calendar().next_run_time(self.run_after)
            logger.info(
                "Daily cadence: last completed session %s, next decision %s",
                session.day if session else "unknown",
                due.strftime("%Y-%m-%d %H:%M %Z"),
            )

        try:
            clock = self._broker_retry(self.client.get_clock, what="market clock")
        except BrokerUnavailable:
            self.market_open = None
            self.next_open = None
            logger.warning("Market clock unavailable. Assuming closed and polling.")
            return

        self.market_open = clock["is_open"]
        self.next_open = clock.get("next_open")
        if self.market_open:
            logger.info("Market is OPEN, next close %s", clock.get("next_close"))
        else:
            logger.info("Market is CLOSED, next open %s", self.next_open)

    # -- 3. model -----------------------------------------------------------

    def model_age_days(self, engine=_UNSET) -> float | None:
        """Age of the fitted model in days, or None if it never trained."""
        engine = self.hmm if engine is _UNSET else engine
        metadata = getattr(engine, "metadata", None)
        trained = getattr(metadata, "training_date", None)
        if trained is None:
            return None
        if trained.tzinfo is None:
            trained = trained.replace(tzinfo=UTC)
        return (datetime.now(UTC) - trained).total_seconds() / 86400.0

    def needs_retrain(self, engine=_UNSET) -> tuple[bool, str]:
        """Spec: retrain if the model is missing or more than 7 days old.

        Phase 2's own `should_retrain()` counts bars since the fit. Both apply:
        the calendar rule alone would skip a refit across a long holiday break,
        and the bar rule alone would let a model sit unrefreshed through a month
        in which the system was halted and processed no bars.
        """
        engine = self.hmm if engine is _UNSET else engine
        if engine is None or not getattr(engine, "is_fitted", False):
            return True, "no fitted model"

        max_age = float(self.orch.get("model_max_age_days", 7))
        age = self.model_age_days(engine)
        if age is None:
            return True, "model has no training date"
        if age > max_age:
            return True, f"model is {age:.1f} days old, limit {max_age:g}"
        if engine.should_retrain():
            return True, f"{engine._bars_since_fit} bars since fit"
        return False, f"model is {age:.1f} days old"

    def _load_or_train_model(self) -> None:
        """Load or train the HMM. Retrain if the model is missing or stale."""
        from core.hmm_engine import HMMEngine

        path = self.model_path or ROOT / self.hmm_config.get(
            "model_path", "models/hmm_model.pkl")

        engine = None
        if Path(path).exists():
            try:
                engine = HMMEngine.load(path)
            except Exception as exc:
                logger.warning("Could not load %s (%s). Retraining.", path, exc)

        retrain, why = self.needs_retrain(engine)
        if retrain:
            logger.info("Training HMM: %s", why)
            engine = self.train_model(save_to=path)
        else:
            logger.info("Loaded HMM: %s, %d states", why, engine.n_states)

        self.hmm = engine
        self._init_orchestrator()

    def train_model(self, save_to: Path | None = None, symbol: str | None = None):
        """Fit the HMM on a fresh training window and save it.

        Trains on `hmm.training_symbol` if set, else the first configured symbol.
        One model across the whole universe, not one per name: the regime is a
        property of the market, and ten models would disagree about what day it
        is.
        """
        from core.hmm_engine import HMMEngine
        from data.feature_engineering import build_feature_matrix, log_returns

        symbol = symbol or self.hmm_config.get("training_symbol") or self.symbols[0]
        n_bars = int(self.orch.get("training_bars", 954))

        bars = self._broker_retry(
            self.market_data.get_training_window, symbol, n_bars,
            what=f"training bars for {symbol}",
        )
        if bars is None or bars.empty:
            raise EngineError(f"No training data returned for {symbol}")

        problems = self.market_data.validate(bars)
        if problems:
            logger.warning("Training data quality: %s", "; ".join(problems))

        features = build_feature_matrix(bars)
        returns = log_returns(bars["close"], 1)

        config = {k: v for k, v in self.hmm_config.items()
                  if k not in ("model_path", "training_symbol")}
        engine = HMMEngine(**config).fit(features, returns)

        if save_to is not None:
            engine.save(Path(save_to))

        self._emit(
            _EventTypes().SYSTEM_START,
            f"Trained HMM on {symbol}: {engine.n_states} states, "
            f"{len(features)} usable rows from {len(bars)} raw bars",
            symbol=symbol, n_states=engine.n_states, bic=engine.metadata.bic,
            n_train_samples=engine.metadata.n_train_samples,
        )
        return engine

    def _init_orchestrator(self) -> None:
        from config import strategy_config
        from core.regime_strategies import StrategyOrchestrator

        self.orchestrator = StrategyOrchestrator(
            strategy_config(self.settings), self.hmm.regime_info
        )

    # -- 4. risk manager ----------------------------------------------------

    def _init_risk_manager(self) -> None:
        """Initialize the risk manager with the current portfolio from Alpaca.

        The manager is constructed here but the `PortfolioState` it judges
        against is built fresh every bar, and deliberately not before the
        session snapshot has been restored. A breaker evaluated against a
        peak_equity of "whatever equity is right now" is a breaker that cannot
        fire, so nothing is allowed to read one until step 6 has run.
        """
        from core.risk_manager import RiskManager

        # The sector map is what makes the concentration cap mean anything.
        # Without it, NVDA/AMD/AVGO/SMCI read as four independent positions
        # rather than one semiconductor bet wearing four hats.
        self.risk_manager = RiskManager(
            self.risk_config,
            lock_file=self.lock_file,
            sector_map=self.settings["broker"].get("sectors") or {},
        )

        if self.risk_manager.is_halted():
            lock = self.risk_manager.breaker.lock_file
            self._emit(_EventTypes().SYSTEM_HALT,
                       f"trading_halted.lock exists at {lock}. Signals will all be rejected.",
                       lock_file=str(lock))
            if self.alerts is not None:
                self.alerts.alert_halt(
                    f"Started with an existing halt lock at {lock}", self.account.equity
                )

    # -- 5. position tracker ------------------------------------------------

    def _init_position_tracker(self) -> None:
        """Initialize the position tracker and sync positions from Alpaca."""
        from broker.order_executor import OrderExecutor
        from broker.position_tracker import PositionTracker

        self.position_tracker = PositionTracker(self.client)
        self.order_executor = (
            _RefusingExecutor() if self.dry_run else OrderExecutor(
                self.client,
                max_price_deviation=float(
                    self.settings["broker"].get("max_price_deviation_pct", 0.05)
                ),
            )
        )

        report = self._broker_retry(self.position_tracker.sync, what="position sync")

        # At startup, adopting positions is expected: it is how a restart picks
        # up what it already held. Mid-session it is not, and process_bar treats
        # it as the alert-worthy event it is.
        if any(report.values()):
            detail = "; ".join(f"{k}: {', '.join(v)}" for k, v in report.items() if v)
            self._emit(_EventTypes().RECONCILE_MISMATCH,
                       f"Startup reconciliation: {detail}", **report)

        self.position_tracker.on_fill(self._on_fill)

    def _on_fill(self, event: dict[str, Any]) -> None:
        """WebSocket fill callback. Runs on the stream thread, so it does
        nothing but log: touching engine state from here would race the loop."""
        try:
            self._emit(_EventTypes().ORDER_FILLED,
                       f"Fill: {event.get('symbol')} {event.get('side')} "
                       f"x{event.get('quantity')} @ {event.get('price')}",
                       **event)
        except Exception as exc:  # pragma: no cover - callback must never raise
            logger.warning("fill callback failed: %s", exc)

    # -- 6. session recovery ------------------------------------------------

    def _restore_session_state(self) -> None:
        """Check for state_snapshot.json and recover the previous session.

        `peak_equity` is restored as max(saved, current). Taking the saved value
        blindly would be wrong after a deposit; taking the current value blindly
        re-bases the peak on every restart and permanently disarms the
        max-drawdown-from-peak breaker. The max of the two is the only choice
        that cannot silently weaken a breaker.
        """
        equity = self.sizing_equity
        today = date.today()
        restored = SessionState.load(self.snapshot_path)

        if restored is not None:
            restored = self._rebase_if_sizing_changed(restored, equity)

        if restored is None:
            self.session = SessionState(
                session_id=self.started_at.strftime("%Y%m%d-%H%M%S"),
                started_at=self.started_at.isoformat(),
                peak_equity=equity,
                # The SCALED previous close, not the raw one. Seeding this
                # from the broker's real last_equity while sizing against an
                # override produced a phantom -90% daily drawdown and halted
                # the system on its first bar.
                day_start_equity=(self._scaled_account().last_equity or equity),
                week_start_equity=equity,
                day_start_date=today.isoformat(),
                week_start_date=_week_start(today).isoformat(),
                equity_at_save=equity,
                sizing_basis=equity,
            )
            logger.info("No previous session snapshot. Starting fresh at $%s.",
                        f"{equity:,.2f}")
            return

        self.session = restored
        self.session.session_id = self.started_at.strftime("%Y%m%d-%H%M%S")
        self.session.started_at = self.started_at.isoformat()

        previous_peak = self.session.peak_equity
        self.session.peak_equity = max(previous_peak, equity)
        self.bars_processed = self.session.bars_processed
        self.previous_regime = self.session.last_regime

        # Roll the day and week windows forward if the calendar moved while the
        # process was down. Without this a restart on Monday would still be
        # measuring its daily drawdown against Friday's opening equity.
        self._roll_periods(today, equity, restoring=True)

        # Restore latched breakers. A daily reduce that latched at 10am must
        # still be latched after a 10:05 restart, or a crash-restart becomes a
        # way to clear a breaker.
        from core.risk_manager import BreakerType

        try:
            self.risk_manager.breaker.daily_tripped = BreakerType(
                self.session.breaker_daily_tripped)
            self.risk_manager.breaker.weekly_tripped = BreakerType(
                self.session.breaker_weekly_tripped)
        except ValueError:
            logger.warning("Unrecognised breaker state in snapshot, defaulting to none.")
        self.risk_manager.breaker.peak_tripped = bool(self.session.breaker_peak_tripped)

        # Restore stops onto positions the tracker adopted from the broker. An
        # adopted position has stop_loss=None, so without this the engine would
        # believe every position it already held is unprotected.
        restored_stops = 0
        for symbol, stop in (self.session.stops or {}).items():
            position = self.position_tracker.positions.get(symbol)
            if position is not None and position.stop_loss is None:
                position.stop_loss = float(stop)
                restored_stops += 1

        logger.info(
            "Recovered session %s: %d bars, peak $%s (was $%s), %d stops restored, "
            "breakers %s/%s/%s",
            restored.session_id or "?", self.bars_processed,
            f"{self.session.peak_equity:,.2f}", f"{previous_peak:,.2f}", restored_stops,
            self.session.breaker_daily_tripped, self.session.breaker_weekly_tripped,
            self.session.breaker_peak_tripped,
        )

    def _rebase_if_sizing_changed(self, restored: SessionState, equity: float
                                  ) -> SessionState:
        """Rescale the baselines when `account_size_override` has changed.

        Every dollar figure in the snapshot was recorded against the old account
        size. Restoring a $100,000 peak into a $10,000 account reads as a 90%
        drawdown and halts the system on startup, with a lock file that has to
        be deleted by hand — for a config edit, not a loss.

        Rescaling by the ratio preserves what the baselines actually mean. A
        peak 5% above current equity stays 5% above it, so the breakers keep
        measuring the same thing and a genuine drawdown is not erased.
        """
        previous = restored.sizing_basis
        if not previous or previous <= 0 or abs(previous - equity) < 1e-9:
            return restored

        factor = equity / previous
        for attribute in ("peak_equity", "day_start_equity", "week_start_equity",
                          "equity_at_save"):
            setattr(restored, attribute, getattr(restored, attribute) * factor)
        for point in restored.equity_history:
            point["equity"] = round(point.get("equity", 0.0) * factor, 2)
            point["peak"] = round(point.get("peak", 0.0) * factor, 2)
        restored.sizing_basis = equity

        logger.warning(
            "Account sizing basis changed from $%s to $%s. Rescaled the session "
            "baselines by %.4fx so the drawdown ratios are preserved. Stops are "
            "prices and were left alone.",
            f"{previous:,.2f}", f"{equity:,.2f}", factor,
        )
        return restored

    def _roll_periods(self, today: date, equity: float, restoring: bool = False) -> None:
        """Reset the daily and weekly baselines when the calendar rolls over."""
        day_start = self.session.day_start_date
        if day_start != today.isoformat():
            self.session.day_start_date = today.isoformat()
            self.session.day_start_equity = equity
            self.session.daily_trades = 0
            if self.risk_manager is not None:
                self.risk_manager.reset_daily()
            if not restoring:
                logger.info("New trading day. Daily baseline reset to $%s.", f"{equity:,.2f}")

        week_start = _week_start(today).isoformat()
        if self.session.week_start_date != week_start:
            self.session.week_start_date = week_start
            self.session.week_start_equity = equity
            if self.risk_manager is not None:
                self.risk_manager.reset_weekly()
            if not restoring:
                logger.info("New trading week. Weekly baseline reset to $%s.", f"{equity:,.2f}")

    # -- 7. data feeds ------------------------------------------------------

    def _start_data_feeds(self) -> None:
        """Start WebSocket feeds.

        Two of them, and only one is a clock:

        - **Trade updates** deliver fills. Always started, and the only way the
          system learns a stop was hit without polling for it.
        - **Bar updates** wake the loop early when a bar closes. Started only
          for intraday timeframes, because Alpaca has no daily-bar socket:
          `subscribe_bars` streams minute bars whatever `timeframe` says.

        The loop polls on its own schedule regardless. The socket is an
        optimisation, never the thing the loop depends on, because "data feed
        drop: pause signals, keep stops active" is only implementable if a
        dropped socket does not also stop the loop that would notice.
        """
        if self.dry_run:
            logger.info("Dry run: not starting live data feeds.")
            return

        try:
            self._streams.append(self.position_tracker.start_websocket())
        except Exception as exc:
            logger.warning("Trade update stream failed to start (%s). "
                           "Fills will be picked up by polling instead.", exc)

        if not _is_daily(self.timeframe):
            try:
                self._streams.append(
                    self.market_data.subscribe_bars(
                        self.symbols, self._on_bar_message, self.timeframe
                    )
                )
            except Exception as exc:
                logger.warning("Bar stream failed to start (%s). Polling instead.", exc)

    def _on_bar_message(self, bar: Any) -> None:  # pragma: no cover - network callback
        """Wake the loop. Deliberately does no work on the stream thread."""
        self._bar_event.set()

    # -- 8. system state ----------------------------------------------------

    def _print_system_state(self) -> None:
        """Print system state, log "System online"."""
        from monitoring.dashboard import DashboardState, TerminalDashboard

        self.dashboard_state = DashboardState(
            position_tracker=self.position_tracker,
            risk_manager=self.risk_manager,
            hmm_engine=self.hmm,
            logger=self.log,
            engine=self,
        )
        self.dashboard = TerminalDashboard(
            refresh_seconds=int(self.monitoring_config.get("dashboard_refresh_seconds", 5))
        )

        self.refresh_portfolio()

        summary = (
            f"System online. mode={self.mode} "
            f"account={'PAPER' if self.is_paper else 'LIVE'} "
            f"equity=${self.account.equity:,.2f} "
            f"symbols={len(self.symbols)} timeframe={self.timeframe} "
            f"states={self.hmm.n_states} "
            f"model_age={_fmt_age(self.model_age_days())} "
            f"positions={self.portfolio.n_positions} "
            f"peak=${self.session.peak_equity:,.2f}"
        )
        self._emit(
            _EventTypes().SYSTEM_START, summary,
            mode=self.mode, paper=self.is_paper, equity=self.account.equity,
            symbols=self.symbols, timeframe=self.timeframe,
            n_states=self.hmm.n_states, model_age_days=self.model_age_days(),
            peak_equity=self.session.peak_equity, halted=self.risk_manager.is_halted(),
            dry_run=self.dry_run,
        )
        self.audit_stops(alert=True)
        self.dashboard.render(self.dashboard_state.snapshot())

    # =======================================================================
    # PORTFOLIO AND STOPS
    # =======================================================================

    @property
    def sizing_equity(self) -> float:
        """The equity every position size is computed against.

        Defaults to the broker's real balance. When `risk.account_size_override`
        is set, that number is used instead, because an Alpaca paper account is
        funded with $100,000 of imaginary money and sizing against it produces
        orders nobody could place. Rehearsing at a size you will never trade
        teaches you nothing about the size you will.

        The override scales the whole risk path coherently: drawdowns are ratios
        and the session baselines are tracked in the same space, so a breaker
        fires at the same percentage either way.
        """
        real = self.account.equity if self.account else 0.0
        override = self.risk_config.get("account_size_override")
        return float(override) if override else real

    def refresh_portfolio(self):
        """Rebuild `PortfolioState` from the broker plus the session baselines."""
        started = time.perf_counter()
        self.account = self._broker_retry(self.client.get_account, what="account refresh")
        self.api_latency_ms = (time.perf_counter() - started) * 1000.0
        self._roll_periods(date.today(), self.sizing_equity)
        equity = self.sizing_equity
        self.session.peak_equity = max(self.session.peak_equity, equity)
        self.session.equity_at_save = equity
        self.session.sizing_basis = equity

        regime = self.regime_state
        scaled = self._scaled_account()
        self.portfolio = self.position_tracker.to_portfolio_state(
            account=scaled,
            peak_equity=self.session.peak_equity,
            day_start_equity=self.session.day_start_equity,
            week_start_equity=self.session.week_start_equity,
            daily_trades=self.session.daily_trades,
            regime=regime.label.value if regime else "unknown",
            regime_confirmed=regime.is_confirmed if regime else True,
            regime_confidence=regime.probability if regime else 1.0,
            flicker_rate=regime.flicker_rate if regime else 0,
        )
        return self.portfolio

    def _scaled_account(self):
        """The account as the risk layer should see it.

        Without an override this is the broker's account unchanged. With one,
        every monetary field is scaled by the same factor, so ratios (exposure,
        drawdown, leverage) are identical and only the absolute dollar sizes
        change. Scaling one field and not the others would silently corrupt
        every percentage the breakers read.
        """
        override = self.risk_config.get("account_size_override")
        if not override or not self.account or self.account.equity <= 0:
            return self.account

        from dataclasses import replace

        factor = float(override) / self.account.equity
        return replace(
            self.account,
            equity=float(override),
            cash=self.account.cash * factor,
            buying_power=self.account.buying_power * factor,
            portfolio_value=self.account.portfolio_value * factor,
            last_equity=self.account.last_equity * factor,
        )

    def audit_stops(self, alert: bool = True) -> list[str]:
        """Every open position must have a resting stop order at the broker.

        Not merely a `stop_loss` float on the tracked position. Shutdown leaves
        positions open on the promise that stops are in place, and a number held
        in this process's memory stops protecting anything the moment the
        process exits. This is the check that makes that promise true.
        """
        positions = self.position_tracker.get_open_positions()
        if not positions:
            return []

        unprotected = [s for s, p in positions.items() if not p.has_stop]

        if self.orch.get("require_broker_stops", True) and not self.dry_run:
            try:
                open_orders = self.client.get_open_orders()
            except Exception as exc:
                logger.warning("Could not read open orders to audit stops: %s", exc)
                open_orders = None

            if open_orders is not None:
                from broker.alpaca_client import OrderType

                resting = {
                    o.symbol for o in open_orders
                    if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
                }
                missing_at_broker = [
                    s for s in positions if s not in resting and s not in unprotected
                ]
                if missing_at_broker:
                    logger.warning(
                        "%s: stop is recorded locally but no resting stop order exists "
                        "at the broker.", ", ".join(missing_at_broker),
                    )
                    unprotected = sorted(set(unprotected) | set(missing_at_broker))

        if unprotected:
            self._emit(_EventTypes().ERROR,
                       f"Open positions with no protective stop: {', '.join(unprotected)}",
                       symbols=unprotected)
            if alert and self.alerts is not None:
                self.alerts.alert_missing_stop(unprotected)
        return unprotected

    # =======================================================================
    # MAIN LOOP
    # =======================================================================

    def run(self) -> int:
        """Wait for bars, process them, until stopped."""
        self._install_signal_handlers()
        if self.schedule_mode == "daily_close":
            return self._run_daily()
        action = str(self.orch.get("market_closed_action", "wait")).lower()
        poll = float(self.orch.get("poll_seconds", 60))
        max_wait = float(self.orch.get("max_wait_seconds", 86400))
        waited = 0.0

        try:
            while self.running:
                if not self._market_is_open():
                    if action == "exit":
                        logger.info("Market closed and market_closed_action=exit. Done.")
                        break
                    if waited >= max_wait:
                        logger.info("Waited %.0fs for the market to open. Exiting cleanly.",
                                    waited)
                        break
                    if waited == 0.0:
                        logger.info("Market closed. Waiting for the next open (%s).",
                                    self.next_open)
                    self._bar_event.wait(poll)
                    self._bar_event.clear()
                    waited += poll
                    continue

                waited = 0.0
                outcome = self.process_bar()
                if outcome.halted:
                    logger.error("Halted. The loop stops here until the lock is cleared.")
                    break

                self._bar_event.wait(self._seconds_to_next_bar())
                self._bar_event.clear()
        except KeyboardInterrupt:      # pragma: no cover - interactive
            logger.info("Interrupted.")
        finally:
            self.shutdown("loop exited")
        return 0

    def _run_daily(self) -> int:
        """Decide once per session after the close; monitor in between.

        The split matters. A new entry can wait for tomorrow, because the signal
        is computed from a daily bar that will not change. A stop cannot: a
        position moving against you at 11am needs its stop looked at now, not at
        16:15. So entries are gated to the daily decision and everything
        protective keeps running on `monitor_seconds`.

        Reaching the decision twice for one session is harmless. `process_bar`
        keys on the bar timestamp and returns "bar already processed", so a
        restart at any hour cannot double-enter.
        """
        from core.calendar import now_et

        # `exit` is the under-cron setting: the scheduler owns the waiting, so
        # a run that arrives before its decision is due should return rather
        # than sleep for fourteen hours inside a job with a timeout.
        action = str(self.orch.get("market_closed_action", "wait")).lower()

        logger.info(
            "Daily cadence. Entries decided once per session after %s ET, "
            "positions and stops monitored every %.0fs.",
            self.run_after.strftime("%H:%M"), self.monitor_seconds,
        )

        try:
            while self.running:
                due = self._calendar().next_run_time(self.run_after)
                wait = (due - now_et()).total_seconds()

                if wait > 0 and action == "exit":
                    logger.info(
                        "Next decision is not due until %s and "
                        "market_closed_action=exit. Nothing to do.",
                        due.strftime("%Y-%m-%d %H:%M %Z"),
                    )
                    break

                if wait <= 0:
                    outcome = self.process_bar()
                    if outcome.halted:
                        logger.error("Halted. The loop stops here until the lock "
                                     "is cleared.")
                        break
                    continue

                logger.info("Next decision %s (%s). Monitoring until then.",
                            due.strftime("%Y-%m-%d %H:%M %Z"), _fmt_duration(wait))

                while self.running and wait > 0:
                    slept = min(self.monitor_seconds, wait)
                    self._bar_event.wait(slept)
                    self._bar_event.clear()
                    wait -= slept
                    if self.running and wait > 0:
                        self.monitor_positions()
        except KeyboardInterrupt:      # pragma: no cover - interactive
            logger.info("Interrupted.")
        finally:
            self.shutdown("loop exited")
        return 0

    def monitor_positions(self) -> None:
        """The protective half of the loop, run between daily decisions.

        Deliberately cannot open a position. It refreshes the portfolio, checks
        that every position still has a stop, tightens stops the regime has
        moved, and evaluates the breakers. No signal generation, no entries.

        Runs only while the market is open. Stops do not trigger overnight, and
        a breaker computed against a stale equity figure would be noise.
        """
        if not self._market_is_open():
            return
        try:
            self.refresh_portfolio()
            self.audit_stops()
            if self.regime_state is not None and self.orch.get("update_stops_each_bar", True):
                self._update_stops(self.regime_state)
            self._check_breakers(BarOutcome())
        except Exception as exc:
            logger.warning("monitor pass failed: %s", exc)

    def _market_is_open(self) -> bool:
        try:
            clock = self._broker_retry(self.client.get_clock, what="market clock")
        except BrokerUnavailable:
            return False
        self.market_open = clock["is_open"]
        self.next_open = clock.get("next_open")
        return self.market_open

    def _seconds_to_next_bar(self) -> float:
        """How long to sleep before the next bar close.

        Daily bars are evaluated once per session, so the daily case sleeps
        until well past the next close and lets the clock check re-drive it.
        """
        if _is_daily(self.timeframe):
            return float(self.orch.get("poll_seconds", 60)) * 5
        minutes = _timeframe_minutes(self.timeframe)
        now = datetime.now(UTC)
        elapsed = (now.minute % minutes) * 60 + now.second
        return max(5.0, minutes * 60 - elapsed)

    def process_bar(self, as_of=None) -> BarOutcome:
        """One pass of the eleven-step loop.

        Wrapped so that no single step can take the process down. A failure in
        the model, the data feed or the broker degrades this bar; it does not
        end the session. Only an unhandled error does, and that path saves state
        and alerts before it exits.
        """
        outcome = BarOutcome()
        try:
            return self._process_bar_inner(outcome, as_of)
        except BrokerUnavailable as exc:
            # Data feed drop: pause signals, keep stops active. Existing stop
            # orders rest at the broker and are untouched by this path.
            outcome.skipped = "broker unavailable"
            outcome.errors.append(str(exc))
            self.consecutive_errors += 1
            self._check_error_budget()
            return outcome
        except Exception as exc:
            outcome.errors.append(str(exc))
            self.consecutive_errors += 1
            self._emit(_EventTypes().ERROR, f"Bar processing failed: {exc}",
                       error=str(exc), traceback=traceback.format_exc())
            if self.alerts is not None:
                self.alerts.alert_unhandled_error(
                    "process_bar", str(exc), traceback.format_exc())
            self._check_error_budget()
            return outcome

    def _mark_feed_healthy(self) -> None:
        """Clear the feed flag, but only at the end of a fully clean bar.

        Deliberately not cleared by the first successful call inside a bar. A
        single call succeeding says the network came back, not that the cycle
        produced trustworthy data, and clearing it early would let the same bar
        that just failed go on to place orders. The cost is one cautious cycle
        after any feed problem, which is the right side to err on.
        """
        if not self.data_feed_healthy:
            self.data_feed_healthy = True
            self._emit(_EventTypes().DATA_FEED_UP, "Data feed recovered")

    def _check_error_budget(self) -> None:
        limit = int(self.orch.get("max_consecutive_errors", 5))
        if self.consecutive_errors >= limit:
            self._emit(_EventTypes().SYSTEM_HALT,
                       f"{self.consecutive_errors} consecutive failed cycles, limit {limit}. "
                       f"Stopping the loop. Positions and their stops are left in place.",
                       consecutive_errors=self.consecutive_errors)
            if self.alerts is not None:
                self.alerts.send(
                    _AlertLevels().CRITICAL,
                    "Loop stopped after repeated failures",
                    f"{self.consecutive_errors} consecutive failed cycles.\n"
                    f"Positions are still open and their stops are still resting "
                    f"at the broker. No new orders will be placed.",
                    dedupe_key="error_budget",
                )
            self._stop.set()

    def _process_bar_inner(self, outcome: BarOutcome, as_of=None) -> BarOutcome:
        from monitoring.logger import EventType

        # ---- 1. new bar ---------------------------------------------------
        self.bars = self._fetch_bars()
        primary = self.symbols[0]
        if primary not in self.bars or self.bars[primary].empty:
            outcome.skipped = "no bars"
            return outcome

        outcome.timestamp = self.bars[primary].index[-1]
        # Keyed on symbol as well as timestamp. Two symbols share a daily bar
        # close, so a timestamp alone made a switch of universe look like a bar
        # this session had already traded.
        bar_key = f"{primary}@{outcome.timestamp}"
        if self.session.last_bar_timestamp == bar_key:
            outcome.skipped = "bar already processed"
            return outcome

        if not self._bar_has_closed(outcome.timestamp):
            # A session still in progress has a bar whose close, high and low
            # will all change before it is final. Acting on one means acting on
            # a number that has not happened yet, which is look-ahead bias
            # arriving through the data feed rather than the feature code.
            outcome.skipped = "bar has not closed yet"
            logger.info("%s: bar %s is still forming, not acting on it.",
                        primary, outcome.timestamp)
            return outcome

        # ---- 2. features, rolling window, no future data ------------------
        from data.feature_engineering import build_feature_matrix

        self.features = build_feature_matrix(self.bars[primary])
        if self.features.empty:
            outcome.skipped = "not enough history for features"
            return outcome

        # ---- 3/4/5. filtered prediction, stability, flicker ---------------
        # The tracker inside `classify` applies the 3-bar persistence rule and
        # the flicker counter, so steps 3, 4 and 5 are one call by construction.
        # Splitting them would mean two places that decide what the regime is.
        regime_state = self._classify(outcome)
        if regime_state is None:
            outcome.skipped = "no regime"
            return outcome

        outcome.regime = regime_state.label.value
        outcome.confidence = regime_state.probability
        outcome.confirmed = regime_state.is_confirmed
        outcome.uncertain = self.orchestrator.is_uncertain(regime_state)

        if self.previous_regime and self.previous_regime != outcome.regime:
            self.log.log_regime_change(
                self.previous_regime, outcome.regime, outcome.confidence,
                confirmed=outcome.confirmed, flicker_rate=regime_state.flicker_rate,
            )
            if self.alerts is not None:
                self.alerts.alert_regime_change(
                    self.previous_regime, outcome.regime, outcome.confidence,
                    confirmed=outcome.confirmed,
                )
        self.previous_regime = outcome.regime
        self.position_tracker.update_regime(outcome.regime)

        self.refresh_portfolio()
        self._update_log_context(regime_state)
        self._check_alert_conditions(regime_state, outcome)
        self._reconcile(outcome)

        # ---- 6. target allocation -----------------------------------------
        price, ema50 = self._price_context(self.bars[primary])
        target = self.orchestrator.target_allocation(regime_state, price, ema50)
        current = self.portfolio.gross_exposure
        outcome.target_allocation = target
        outcome.current_allocation = current
        self.target_allocation = target

        # ---- 6b. watchlist scan -------------------------------------------
        # Deliberately BEFORE step 7 trades. Run after, every name the system had
        # just bought came back rejected as a duplicate order, which is true and
        # useless: the watchlist must show the decision being made, not its
        # aftermath. Also runs on bars where nothing is traded, because "what
        # would it buy" still has an answer then and that is what this is for.
        self.candidates = self.scan_candidates(regime_state)
        outcome.candidates = len(self.candidates)
        outcome.top_pick = self.candidates[0].symbol if self.candidates else ""

        # ---- 7. validate and act ------------------------------------------
        if not self.data_feed_healthy:
            outcome.skipped = "data feed unhealthy: signals paused, stops untouched"
        elif self.risk_manager.is_halted():
            outcome.skipped = "halted"
        elif not self.orchestrator.needs_rebalance(target, current):
            outcome.skipped = (
                f"no rebalance: target {target:.1%} vs current {current:.1%}, "
                f"inside the {self.orchestrator.rebalance_threshold:.0%} threshold"
            )
        else:
            outcome.rebalanced = True
            if target < current:
                self._reduce_to_target(target, current, outcome)
            else:
                self._increase_to_target(regime_state, outcome)

        # ---- 8. trailing stops per regime ---------------------------------
        if self.data_feed_healthy and self.orch.get("update_stops_each_bar", True):
            outcome.stops_updated = self._update_stops(regime_state)

        # ---- 9. circuit breakers ------------------------------------------
        outcome.breaker_state = self._check_breakers(outcome)
        outcome.halted = self.risk_manager.is_halted()

        # ---- 10. dashboard ------------------------------------------------
        if self.dashboard is not None:
            try:
                self.dashboard.render(self.dashboard_state.snapshot())
            except Exception as exc:
                logger.warning("dashboard render failed: %s", exc)

        # ---- 11. weekly retrain -------------------------------------------
        self._maybe_retrain()

        self.bars_processed += 1
        self.consecutive_errors = 0
        self._mark_feed_healthy()
        self.session.bars_processed = self.bars_processed
        self.session.last_bar_timestamp = bar_key
        self.session.last_regime = outcome.regime
        self.session.last_regime_confidence = outcome.confidence
        self.position_tracker.increment_holding_periods()
        self._capture_stops()
        self._append_history(outcome)
        self._record_bar_state(outcome)
        self.save_state()
        if self.publish_path is not None:
            self.publish_snapshot()

        self._emit(EventType.BAR_PROCESSED,
                   f"bar {outcome.timestamp} {outcome.regime} p={outcome.confidence:.2f} "
                   f"target {target:.1%} current {current:.1%} "
                   f"{outcome.submitted} orders {outcome.rejected} rejected"
                   + (f" {outcome.deduped} deduped" if outcome.deduped else "")
                   + (f" [{outcome.skipped}]" if outcome.skipped else ""),
                   **{("bar_timestamp" if k == "timestamp" else k): v
                      for k, v in asdict(outcome).items() if k != "errors"})
        return outcome

    # -- loop helpers -------------------------------------------------------

    def _fetch_bars(self) -> dict[str, Any]:
        """Pull the current history window for every symbol.

        Adjusted prices, because these feed the features and the model was fit
        on adjusted data. Orders price off the raw quote instead, which the
        executor handles.
        """
        lookback = int(self.orch.get("training_bars", 954))
        out: dict[str, Any] = {}
        for symbol in self.symbols:
            try:
                frame = self._broker_retry(
                    self.market_data.get_training_window, symbol, lookback,
                    what=f"bars for {symbol}",
                )
                if frame is not None and not frame.empty:
                    out[symbol] = frame
            except BrokerUnavailable:
                if symbol == self.symbols[0]:
                    raise
                logger.warning("%s: no bars this cycle, skipping the symbol.", symbol)
        return out

    def _classify(self, outcome: BarOutcome):
        """Filtered HMM prediction. On failure, hold the current regime.

        Per the spec's error handling. Holding the last known regime is the
        conservative choice: it keeps allocation where it is rather than
        defaulting to a regime nobody computed.
        """
        try:
            state = self.hmm.classify(self.features)
            self.regime_state = state
            return state
        except Exception as exc:
            outcome.errors.append(f"hmm: {exc}")
            self._emit(_EventTypes().ERROR,
                       f"HMM classification failed ({exc}). Holding regime "
                       f"{self.regime_state.label.value if self.regime_state else 'unknown'}.",
                       error=str(exc))
            return self.regime_state

    def _update_log_context(self, regime_state) -> None:
        """Stamp the spec's six required fields onto every subsequent log entry.

        timestamp, regime, probability, equity, positions, daily_pnl. They are
        properties of the moment rather than of any one event, so they are set
        once per bar instead of being passed at forty call sites, where the
        first one anyone forgot would be the one that mattered.
        """
        portfolio = self.portfolio
        day_start = portfolio.day_start_equity or portfolio.equity
        self.log.set_context(
            regime=regime_state.label.value,
            probability=round(regime_state.probability, 4),
            equity=round(portfolio.equity, 2),
            positions=portfolio.n_positions,
            daily_pnl=round(portfolio.equity - day_start, 2),
        )

    def _check_alert_conditions(self, regime_state, outcome: BarOutcome) -> None:
        """The spec's triggers that are conditions rather than events.

        Regime change, breaker, retrain, feed and API alerts fire from the code
        paths that cause them. These two have to be looked for.
        """
        if self.alerts is None:
            return

        if regime_state.is_flickering:
            self._emit(_EventTypes().FLICKER_EXCEEDED,
                       f"Regime flickering: {regime_state.flicker_rate} changes in "
                       f"{self.hmm.flicker_window} bars",
                       flicker_rate=regime_state.flicker_rate,
                       threshold=self.hmm.flicker_threshold)
            self.alerts.alert_flicker_exceeded(
                regime_state.flicker_rate, self.hmm.flicker_threshold,
                self.hmm.flicker_window,
            )

        day_start = self.portfolio.day_start_equity or self.portfolio.equity
        daily_pnl = self.portfolio.equity - day_start
        daily_pct = daily_pnl / day_start if day_start else 0.0
        if abs(daily_pct) >= self.alerts.large_pnl_pct:
            self._emit(_EventTypes().LARGE_PNL,
                       f"Large daily move: {daily_pct:+.2%} ({daily_pnl:+,.2f})",
                       daily_pnl=daily_pnl, daily_pnl_pct=daily_pct)
            self.alerts.alert_large_pnl(daily_pnl, daily_pct, self.portfolio.equity)

    def scan_candidates(self, regime_state) -> list:
        """Rank the whole universe by what the risk layer would let you buy.

        Side-effect free: `validate_signal(record=False)` means asking the
        question cannot change the answer by tripping the duplicate window.
        """
        from core.candidates import scan

        try:
            return scan(
                self.orchestrator, self.risk_manager, self.symbols, self.bars,
                regime_state, self.portfolio, self.settings["strategy"],
            )
        except Exception as exc:
            logger.warning("candidate scan failed: %s", exc)
            return []

    def _calendar(self):
        """The trading calendar, built on first use.

        Lazy because `process_bar` can be reached without a full `startup()`,
        in tests and in the `--once` path, and a calendar that only exists when
        startup ran would make the bar-completeness check silently skip itself
        in exactly those cases.
        """
        if self.calendar is None:
            from core.calendar import MarketCalendar

            self.calendar = MarketCalendar(self.client)
        return self.calendar

    def _bar_has_closed(self, timestamp) -> bool:
        """Is this bar final, or is its session still running?

        Only meaningful on daily bars in daily_close mode. Intraday timeframes
        and `continuous` mode keep the old behaviour, where the loop is driven
        by bar-close events rather than the calendar.

        Fails open. If the calendar cannot say, the bar is treated as closed:
        Alpaca does not return a bar for a session that has not started, and
        refusing to trade whenever the calendar endpoint is down would be a
        larger failure than the one being prevented.
        """
        if self.schedule_mode != "daily_close" or not _is_daily(self.timeframe):
            return True
        if timestamp is None:
            return True
        try:
            import pandas as pd

            bar_day = pd.Timestamp(timestamp).date()
            session = self._calendar().session(bar_day)
            if session is None:
                return True         # no session that day, so nothing is forming
            from core.calendar import now_et

            return now_et() >= session.close_at
        except Exception as exc:
            logger.debug("could not check whether %s has closed (%s)", timestamp, exc)
            return True

    def _price_context(self, bars) -> tuple[float, float]:
        from data.feature_engineering import ema

        price = float(bars["close"].iloc[-1])
        ema50 = float(ema(bars["close"], int(self.settings["strategy"]["ema_span"])).iloc[-1])
        return price, ema50

    def _reconcile(self, outcome: BarOutcome) -> None:
        """Mid-session reconciliation drift is a stop-and-alert condition."""
        report = self.position_tracker.sync()
        if not any(report.values()):
            return
        detail = [f"{k}: {', '.join(v)}" for k, v in report.items() if v]
        outcome.errors.append("reconcile drift")
        self._emit(_EventTypes().RECONCILE_MISMATCH,
                   "Reconciliation drift mid-session: " + "; ".join(detail), **report)
        if self.alerts is not None:
            self.alerts.alert_reconcile_mismatch(detail)
        self.refresh_portfolio()

    # -- 7a. increasing exposure: the risk manager has the veto -------------

    def _increase_to_target(self, regime_state, outcome: BarOutcome) -> None:
        """Generate signals, validate each, submit the approved ones."""
        from monitoring.logger import EventType

        signals = self.orchestrator.generate_signals(self.symbols, self.bars, regime_state)
        outcome.signals = len(signals)

        for candidate_signal in signals:
            quote = self._quote(candidate_signal.symbol)
            decision = self.risk_manager.validate_signal(
                candidate_signal, self.portfolio, quote=quote, overnight=True
            )
            self.log.log_signal(candidate_signal, decision)
            self._record("record_signal", candidate_signal, decision)

            if not decision.approved:
                outcome.rejected += 1
                self.signals_rejected += 1
                continue

            outcome.approved += 1
            if self.dry_run:
                logger.info(
                    "  dry run: would BUY %s x%g ($%s, risk %.3f%% of equity)%s",
                    candidate_signal.symbol, decision.approved_quantity,
                    f"{decision.approved_notional:,.2f}",
                    decision.modified_signal["risk_pct_of_equity"] * 100,
                    f" [{'; '.join(decision.modifications)}]" if decision.modifications else "",
                )
                outcome.submitted += 1
                continue

            try:
                trade = self.order_executor.submit_order(candidate_signal, decision)
                if getattr(trade, "skipped_reason", None):
                    # The idempotency layer declined. Not an error, not a
                    # submission, and specifically not a stop to attach: there
                    # is no new position, and the order it deduplicated
                    # against already carries its own.
                    outcome.deduped += 1
                    logger.info("  %s: %s", candidate_signal.symbol, trade.skipped_reason)
                    continue
                outcome.submitted += 1
                self.orders_submitted += 1
                self.session.daily_trades += 1
                self.log.log_order(trade, EventType.ORDER_SUBMITTED,
                                   trade_id=trade.trade_id, regime=outcome.regime)
                self._record("record_order", trade,
                             client_order_id=trade.client_order_id)
                self._attach_stop(candidate_signal, decision, trade)
            except Exception as exc:
                outcome.errors.append(f"{candidate_signal.symbol}: {exc}")
                self._emit(EventType.ORDER_REJECTED,
                           f"{candidate_signal.symbol}: order failed, {exc}",
                           symbol=candidate_signal.symbol, error=str(exc))

    def _attach_stop(self, signal, decision, trade) -> None:
        """Record the stop, and place the resting order once shares actually exist.

        The stop order can only be placed against a filled quantity, so an entry
        that is still open has its stop recorded locally and placed by a later
        cycle once the fill lands. `audit_stops` is what makes sure that later
        cycle actually happens rather than being forgotten.
        """
        self.position_tracker.update_stop(signal.symbol, signal.stop_loss)
        self.session.stops[signal.symbol] = float(signal.stop_loss)

        filled = getattr(trade, "filled_qty", 0) or 0
        if filled <= 0:
            logger.info("%s: entry not filled yet, stop recorded at %.2f and placed "
                        "once shares exist.", signal.symbol, signal.stop_loss)
            return
        try:
            self.order_executor.place_stop(
                signal.symbol, filled, signal.stop_loss, trade_id=trade.trade_id
            )
        except Exception as exc:
            self._emit(_EventTypes().ERROR,
                       f"{signal.symbol}: FILLED but the stop order failed ({exc}). "
                       f"This position is unprotected.", symbol=signal.symbol)
            if self.alerts is not None:
                self.alerts.alert_missing_stop([signal.symbol])

    # -- 7b. reducing exposure: no veto ------------------------------------

    def _reduce_to_target(self, target: float, current: float, outcome: BarOutcome) -> None:
        """Scale every position down to the new target. Deliberately not routed
        through `validate_signal`.

        The risk manager exists to stop the system taking on more risk than it
        should. Asking it to approve a reduction inverts that: a rejection would
        leave the system holding an allocation the regime no longer supports, at
        exactly the moment it wanted out. Reducing risk never needs permission.

        Proportional rather than name-by-name, so the relative weights of the
        book are preserved and no separate decision about which name to cut is
        smuggled in here.
        """
        positions = self.position_tracker.get_open_positions()
        if not positions:
            outcome.skipped = "nothing to reduce"
            return

        ratio = 0.0 if target <= 0 else max(0.0, min(1.0, target / current))
        for symbol, position in positions.items():
            target_shares = int(position.quantity * ratio)
            delta = target_shares - int(position.quantity)
            if delta >= 0:
                continue

            outcome.signals += 1
            if self.dry_run:
                logger.info("  dry run: would SELL %s x%d (%g -> %d, target %.1f%% gross)",
                            symbol, -delta, position.quantity, target_shares,
                            target * 100)
                outcome.submitted += 1
                continue

            try:
                if target_shares <= 0:
                    self.order_executor.close_position(symbol)
                    self.session.stops.pop(symbol, None)
                else:
                    self._sell_shares(symbol, -delta)
                outcome.submitted += 1
                self.orders_submitted += 1
                self.session.daily_trades += 1
                self._emit(_EventTypes().ORDER_SUBMITTED,
                           f"{symbol}: reducing {position.quantity:g} -> {target_shares} "
                           f"(allocation {current:.1%} -> {target:.1%})",
                           symbol=symbol, quantity=-delta, reason="regime allocation cut")
            except Exception as exc:
                outcome.errors.append(f"{symbol}: {exc}")
                self._emit(_EventTypes().ERROR, f"{symbol}: reduction failed, {exc}",
                           symbol=symbol, error=str(exc))

    def _sell_shares(self, symbol: str, quantity: int) -> Any:
        """Market sell of a partial position.

        Market, not limit: this is the exit path, and an unfilled limit order
        that leaves the system over-allocated through a regime it wanted out of
        is worse than the spread.
        """
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=symbol, qty=quantity, side=AlpacaSide.SELL, time_in_force=TimeInForce.DAY
        )
        return self.client.to_order(self.client.trading_client.submit_order(request))

    # -- 8. stops -----------------------------------------------------------

    def _update_stops(self, regime_state) -> int:
        """Recompute each position's stop under the current regime, tighten only.

        The strategy in force decides where the stop belongs, so a move from a
        low-volatility to a high-volatility regime re-anchors every stop to the
        wider formula. `modify_stop` refuses to widen, so re-anchoring can only
        ever tighten in practice: the wider stop is computed, rejected, and the
        existing tighter one stays. That asymmetry is intentional.
        """
        strategy = self.orchestrator.get_strategy(regime_state.state_id)
        updated = 0

        for symbol, position in self.position_tracker.get_open_positions().items():
            bars = self.bars.get(symbol)
            if bars is None or bars.empty:
                continue
            context = strategy._indicator_context(bars)
            if context is None:
                continue

            price, ema50, atr_value, _ = context
            raw = strategy.compute_raw_stop(price, ema50, atr_value)
            new_stop, _clamped = strategy._clamp_stop(price, raw, atr_value)

            if position.stop_loss is not None and new_stop <= position.stop_loss:
                continue

            if self.dry_run:
                logger.info("  dry run: would tighten %s stop %s -> %.2f",
                            symbol,
                            f"{position.stop_loss:.2f}" if position.stop_loss else "none",
                            new_stop)
                updated += 1
                continue

            try:
                order = self.order_executor.modify_stop(symbol, new_stop)
                if order is None and position.stop_loss is None:
                    order = self.order_executor.place_stop(
                        symbol, position.quantity, new_stop)
                if order is not None:
                    self.position_tracker.update_stop(symbol, new_stop)
                    self.session.stops[symbol] = float(new_stop)
                    updated += 1
                    self._emit(_EventTypes().STOP_UPDATED,
                               f"{symbol}: stop tightened to {new_stop:.2f} "
                               f"under {strategy.name}",
                               symbol=symbol, stop=new_stop, strategy=strategy.name)
            except Exception as exc:
                logger.warning("%s: stop update failed (%s). Existing stop left in place.",
                               symbol, exc)
        return updated

    def _record_bar_state(self, outcome: BarOutcome) -> None:
        """Mirror this bar's portfolio and positions into state.db.

        Positions are reconciled against the broker rather than tracked
        incrementally. The broker is the authority on what is held, and a
        locally maintained count drifts the moment a fill happens between
        cycles or a stop triggers overnight.
        """
        if self.repo is None:
            return
        try:
            day_start = self.portfolio.day_start_equity or self.portfolio.equity
            self.repo.record_equity(
                self.portfolio.equity,
                bar_date=str(outcome.timestamp)[:10],
                cash=getattr(self.portfolio, "cash", None),
                positions_value=getattr(self.portfolio, "positions_value", None),
                peak_equity=self.session.peak_equity,
                daily_pnl=self.portfolio.equity - day_start,
                open_positions=len(self.position_tracker.positions),
                regime=outcome.regime,
                run_id=self.run_id,
            )
            self._reconcile_positions_table(outcome)
        except Exception as exc:
            logger.debug("bar state write failed: %s", exc)

    def _reconcile_positions_table(self, outcome: BarOutcome) -> None:
        """Make the positions table match what the broker actually holds."""
        held = {p.symbol: p for p in self.position_tracker.positions.values()}
        recorded = {row["symbol"]: row for row in self.repo.open_positions()}

        for symbol, position in held.items():
            if symbol not in recorded:
                self.repo.open_position(
                    symbol, position.qty, position.avg_entry_price,
                    stop_price=self.session.stops.get(symbol),
                    regime=outcome.regime,
                )
            else:
                self.repo.update_open_position(
                    symbol,
                    current_price=position.current_price,
                    stop_price=self.session.stops.get(symbol),
                    unrealised_pnl=position.market_value - position.cost_basis,
                    holding_days=getattr(position, "holding_periods", None),
                )

        # Gone from the broker means closed. The exit reason is inferred: a
        # position whose stop was at or above the last price most likely
        # stopped out. Inferred rather than known because the fill happens
        # between cycles and nothing tells us which leg took it.
        for symbol, row in recorded.items():
            if symbol in held:
                continue
            last = row["current_price"] or row["entry_price"]
            stop = row["stop_price"]
            reason = "stop" if stop and last and last <= stop * 1.01 else "closed"
            self.repo.close_position(symbol, float(last), reason)

    def _append_history(self, outcome: BarOutcome) -> None:
        """One equity and regime point per bar, capped at `history_points`."""
        stamp = str(outcome.timestamp)[:10]
        cap = int(self.orch.get("history_points", 500))
        self.session.equity_history.append({
            "t": stamp,
            "equity": round(self.portfolio.equity, 2),
            "peak": round(self.session.peak_equity, 2),
        })
        self.session.regime_history.append({"t": stamp, "regime": outcome.regime})
        del self.session.equity_history[:-cap]
        del self.session.regime_history[:-cap]

    def publish_snapshot(self, path=None):
        """Write the web dashboard's JSON. Never raises: publishing a view must
        not be able to stop the loop that produced it."""
        from monitoring.publish import DEFAULT_OUTPUT, publish_from_engine

        try:
            return publish_from_engine(self, path or self.publish_path or DEFAULT_OUTPUT)
        except Exception as exc:
            logger.warning("dashboard publish failed: %s", exc)
            return None

    def _capture_stops(self) -> None:
        """Snapshot current stops so a restart can restore them onto adopted
        positions, which come back from the broker with no stop attached."""
        self.session.stops = {
            symbol: float(p.stop_loss)
            for symbol, p in self.position_tracker.get_open_positions().items()
            if p.stop_loss is not None
        }

    # -- 9. breakers --------------------------------------------------------

    def _check_breakers(self, outcome: BarOutcome) -> str:
        """Evaluate, latch, and act. Fires on realised P&L, never on the model."""
        from core.risk_manager import BreakerState

        was_halted = self.risk_manager.is_halted()
        state = self.risk_manager.update(self.portfolio)

        self.session.breaker_daily_tripped = self.risk_manager.breaker.daily_tripped.value
        self.session.breaker_weekly_tripped = self.risk_manager.breaker.weekly_tripped.value
        self.session.breaker_peak_tripped = self.risk_manager.breaker.peak_tripped

        if state is BreakerState.HALTED:
            breaker = self.risk_manager.breaker.check(self.portfolio).value
            self._record("record_breaker", breaker, "tripped",
                         observed=self.portfolio.drawdown_from_peak,
                         equity=self.portfolio.equity, detail=outcome.regime)
            self.log.log_breaker(
                breaker, self.portfolio.drawdown_from_peak, self.portfolio.equity,
                outcome.regime,
            )
            if self.alerts is not None:
                self.alerts.alert_breaker_triggered(
                    breaker, self.portfolio.drawdown_from_peak, self.portfolio.equity
                )
            if not was_halted and not self.dry_run:
                self._close_everything(f"circuit breaker {breaker}")
            self._stop.set()

        return state.value

    def _close_everything(self, reason: str) -> None:
        """A halt closes positions. This is the one path that does.

        Distinct from shutdown, which deliberately does not: a halt means the
        system lost more money than it is allowed to, and holding through that
        is the thing the breaker exists to prevent.
        """
        try:
            orders = self.order_executor.close_all_positions(reason=reason)
            self._emit(_EventTypes().POSITION_CLOSED,
                       f"Halt: closed {len(orders)} positions ({reason})",
                       reason=reason, n_closed=len(orders))
            self.session.stops.clear()
        except Exception as exc:
            self._emit(_EventTypes().ERROR,
                       f"HALT BUT COULD NOT CLOSE POSITIONS: {exc}. Close them by hand.",
                       error=str(exc))
            if self.alerts is not None:
                self.alerts.send(
                    _AlertLevels().CRITICAL, "Halt could not close positions",
                    f"{exc}\nClose the positions manually.", dedupe_key="halt_close_failed",
                )

    # -- 11. retrain --------------------------------------------------------

    def _maybe_retrain(self) -> bool:
        """Weekly refit, per the spec's step 11.

        Refits renumber the states, so the orchestrator's regime -> strategy map
        has to be rebuilt from the new `regime_info` or it would map the new
        state ids through the old volatility ranks. That is a silent failure:
        everything keeps running and the allocations are simply wrong.
        """
        retrain, why = self.needs_retrain()
        if not retrain:
            return False

        logger.info("Retraining: %s", why)
        previous_states = getattr(self.hmm, "n_states", None)
        try:
            path = self.model_path or ROOT / self.hmm_config.get(
                "model_path", "models/hmm_model.pkl")
            self.hmm = self.train_model(save_to=path)
            self.orchestrator.update_regime_infos(self.hmm.regime_info)
            if self.dashboard_state is not None:
                self.dashboard_state.hmm_engine = self.hmm
            self.previous_regime = None      # state ids changed meaning
            self._emit(_EventTypes().MODEL_RETRAINED,
                       f"Retrained: {self.hmm.n_states} states ({why})",
                       n_states=self.hmm.n_states, previous_states=previous_states,
                       reason=why)
            if self.alerts is not None:
                self.alerts.alert_hmm_retrained(
                    self.hmm.n_states, why, previous_states=previous_states)
            return True
        except Exception as exc:
            self._emit(_EventTypes().ERROR,
                       f"Retrain failed ({exc}). Continuing on the previous model.",
                       error=str(exc))
            return False

    def _quote(self, symbol: str) -> dict[str, float] | None:
        """Live quote for the spread check, or None when there is not one.

        None rather than a fabricated quote: the risk manager skips the spread
        check when it gets None, and skipping a check is honest where inventing
        a tight spread would not be.
        """
        try:
            quote = self.market_data.get_latest_quote(symbol)
        except Exception:
            return None
        if not quote or not quote.get("bid") or not quote.get("ask"):
            return None
        return quote

    # =======================================================================
    # SHUTDOWN
    # =======================================================================

    def _install_signal_handlers(self) -> None:
        def handle(signum, _frame):  # pragma: no cover - signal path
            logger.info("Received %s. Shutting down.", signal.Signals(signum).name)
            self._stop.set()
            self._bar_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handle)
            except ValueError:
                pass       # not on the main thread, e.g. under a test runner

    def save_state(self) -> Path:
        self.session.bars_processed = self.bars_processed
        return self.session.save(self.snapshot_path)

    def shutdown(self, reason: str = "") -> None:
        """Close the sockets, save state, print the summary. Leave positions open.

        Positions stay open on purpose: their stops are resting orders at the
        broker, so they remain protected with this process dead. Liquidating on
        every restart would turn a deploy into a taxable event and would sell
        into whatever spread happened to exist at the moment of the restart.

        The promise only holds if the stops are real. `audit_stops` runs here
        and shouts if any position would be left naked, which is the difference
        between "we chose not to close" and "we forgot".
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._stop.set()

        for stream in self._streams:
            try:
                if hasattr(stream, "stop"):
                    stream.stop()
            except Exception:
                pass
        for stopper in (
            getattr(self.position_tracker, "stop_websocket", None),
            getattr(self.market_data, "stop_stream", None),
        ):
            if stopper is not None:
                try:
                    stopper()
                except Exception as exc:
                    logger.warning("stream shutdown: %s", exc)

        unprotected: list[str] = []
        if self.position_tracker is not None:
            try:
                self._capture_stops()
                unprotected = self.audit_stops(alert=True)
            except Exception as exc:
                logger.warning("stop audit at shutdown failed: %s", exc)

        try:
            path = self.save_state()
        except Exception as exc:
            logger.error("could not save state snapshot: %s", exc)
            path = self.snapshot_path

        self._close_run(reason, unprotected)

        self._emit(_EventTypes().SYSTEM_SHUTDOWN,
                   f"Shutdown ({reason}). State saved to {path.name}. "
                   f"Positions left open with stops in place.",
                   reason=reason, bars_processed=self.bars_processed,
                   unprotected=unprotected)

        if self.orch.get("print_session_summary", True):
            self.print_session_summary(unprotected)

    def _close_run(self, reason: str, unprotected: list[str]) -> None:
        """Stamp the run row with how it ended.

        A run left as 'running' means the process died without getting here,
        which the dashboard shows as a stale run rather than a successful one.
        That distinction is the reason the row is opened at startup instead of
        being written in one piece at the end.
        """
        if self.repo is None or self.run_id is None:
            return
        status = "ok"
        if self.risk_manager is not None and self.risk_manager.is_halted():
            status = "halted"
        elif self.consecutive_errors > 0 or unprotected:
            status = "failed"
        try:
            self.repo.finish_run(
                self.run_id, status,
                error=reason if status == "failed" else None,
                bars_processed=self.bars_processed,
                orders_submitted=self.orders_submitted,
                regime=self.session.last_regime,
                equity=self.account.equity if self.account else None,
            )
            self.repo.close()
        except Exception as exc:
            logger.debug("could not close the run row: %s", exc)

    def print_session_summary(self, unprotected: list[str] | None = None) -> None:
        from rich.console import Console

        console = Console()
        started = self.started_at or datetime.now(UTC)
        elapsed = datetime.now(UTC) - started
        equity = self.account.equity if self.account else 0.0
        start_equity = self.session.day_start_equity or equity
        change = equity - start_equity

        console.print("\n[bold]Session summary[/bold]")
        console.print(f"  mode              {self.mode}"
                      f"{'  (no orders were placed)' if self.dry_run else ''}")
        console.print(f"  ran for           {str(elapsed).split('.')[0]}")
        console.print(f"  bars processed    {self.bars_processed}")
        console.print(f"  orders submitted  {self.orders_submitted}")
        console.print(f"  signals rejected  {self.signals_rejected}")
        console.print(f"  equity            ${equity:,.2f} "
                      f"({change:+,.2f} since the day's open)")
        console.print(f"  peak equity       ${self.session.peak_equity:,.2f}")

        if self.position_tracker is not None:
            positions = self.position_tracker.get_open_positions()
            console.print(f"  open positions    {len(positions)} (left open, stops in place)")
            for symbol, p in positions.items():
                stop = f"stop {p.stop_loss:.2f}" if p.has_stop else "[red]NO STOP[/red]"
                console.print(f"      {symbol:<6} x{p.quantity:<8g} {stop}  "
                              f"P&L {p.unrealised_pnl:+,.2f}")

        if unprotected:
            console.print(f"  [red bold]UNPROTECTED: {', '.join(unprotected)}[/red bold]")
            console.print("  [red]Place stops by hand or close these positions.[/red]")

        if self.risk_manager is not None and self.risk_manager.is_halted():
            console.print(f"  [red bold]HALTED[/red bold]  delete "
                          f"{self.risk_manager.breaker.lock_file} to resume")

        if self.log is not None:
            counts = self.log.counts()
            if counts:
                console.print("  events            "
                              + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
            console.print(f"  log               {self.log.event_path}")


# ===========================================================================
# Module helpers
# ===========================================================================

def _AlertLevels():
    from monitoring.alerts import AlertLevel

    return AlertLevel


def _fmt_age(days: float | None) -> str:
    return "unknown" if days is None else f"{days:.1f}d"


def _week_start(day: date) -> date:
    """Monday of the week containing `day`. The weekly breaker's boundary."""
    return day - timedelta(days=day.weekday())


def _fmt_duration(seconds: float) -> str:
    """A wait the reader can sanity-check at a glance. "14h 2m", not "50520s"."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m" if minutes else f"{seconds}s"


def _is_daily(timeframe: str) -> bool:
    return str(timeframe).lower() in ("1day", "day", "1d", "daily")


def _timeframe_minutes(timeframe: str) -> int:
    """Minutes per bar. Only used to align intraday sleeps."""
    text = str(timeframe).lower().replace(" ", "")
    for suffix, factor in (("min", 1), ("m", 1), ("hour", 60), ("h", 60)):
        if text.endswith(suffix):
            head = text[: -len(suffix)] or "1"
            try:
                return max(1, int(head) * factor)
            except ValueError:
                break
    return 5


# ===========================================================================
# CLI modes
# ===========================================================================

def run_trading(args) -> int:
    """Live or paper trading, and the dry run, which is the same pipeline."""
    from config import load_settings

    settings = load_settings()
    _configure_console_logging(args.log_level)

    if args.timeframe and not _is_daily(args.timeframe):
        logging.getLogger(__name__).warning(
            "Timeframe %s: every feature in this system is calibrated for daily "
            "bars (252-day z-scores, 200-bar SMA, 50-bar EMA stops). On %s those "
            "windows mean something else entirely and nothing has been validated "
            "against them. See docs/TUTORIAL-CONFLICTS.md section 5.",
            args.timeframe, args.timeframe,
        )

    from monitoring.publish import DEFAULT_OUTPUT

    engine = TradingEngine(
        settings,
        dry_run=args.dry_run,
        symbols=args.symbols,
        timeframe=args.timeframe,
        allow_live=args.i_understand_live,
        publish_path=DEFAULT_OUTPUT if args.publish else None,
        once=args.once,
    )

    try:
        engine.startup()
    except Exception as exc:
        logging.getLogger(__name__).error("Startup failed: %s", exc)
        if not isinstance(exc, (EngineError, BrokerUnavailable)):
            traceback.print_exc()
        return 2

    if args.once:
        outcome = engine.process_bar()
        if args.publish:
            path = engine.publish_snapshot()
            if path:
                logging.getLogger(__name__).info("Published dashboard data to %s", path)
            else:
                logging.getLogger(__name__).error(
                    "--publish was requested but writing the snapshot failed. "
                    "See the warning above."
                )
                engine.shutdown("publish failed")
                return 3
        engine.shutdown("single bar requested")
        _print_bar_outcome(outcome)
        return 0

    return engine.run()


def _print_bar_outcome(outcome: BarOutcome) -> None:
    from rich.console import Console

    console = Console()
    console.print("\n[bold]Bar outcome[/bold]")
    for key, value in asdict(outcome).items():
        if value in (None, "", 0, False, []):
            continue
        console.print(f"  {key:<20} {value}")
    if outcome.skipped:
        console.print(f"  [dim]no action taken: {outcome.skipped}[/dim]")


def run_train_only(args) -> int:
    """Train the HMM and exit."""
    from config import load_settings

    settings = load_settings()
    _configure_console_logging(args.log_level)

    engine = TradingEngine(settings, symbols=args.symbols, dry_run=True)
    engine._setup_logging()
    engine._connect_broker()

    symbol = (args.symbols or settings["broker"]["symbols"])[0]
    path = ROOT / settings["hmm"].get("model_path", "models/hmm_model.pkl")
    hmm = engine.train_model(save_to=path, symbol=symbol)

    from rich.console import Console

    console = Console()
    console.print(f"\n[bold]Trained on {symbol}[/bold]  ->  {path}")
    console.print(f"  {hmm.n_states} states selected by BIC "
                  f"({hmm.metadata.n_train_samples} samples, "
                  f"{hmm.metadata.n_parameters} parameters)")
    console.print("  BIC by candidate: "
                  + ", ".join(f"{k}:{v:,.0f}" for k, v in sorted(hmm.metadata.all_bic_scores.items())))
    console.print(f"  converged: {hmm.metadata.converged} in {hmm.metadata.n_iterations} iterations\n")
    console.print(hmm.summary().round(3).to_string(index=False))
    console.print("\n[bold]Expected duration per regime (bars)[/bold]")
    console.print("  " + hmm.get_expected_durations().round(1).to_string().replace("\n", "\n  "))
    return 0


def run_dashboard(args) -> int:
    """Terminal dashboard against a running instance's state.

    Reads the snapshot and the event log rather than connecting to the engine.
    A dashboard that holds a live handle on the trading objects is one restart
    away from being the reason the trading process died.
    """
    from rich.console import Console

    from config import load_settings
    from monitoring.logger import TradingLogger

    settings = load_settings()
    console = Console()
    snapshot_path = ROOT / settings["orchestration"].get(
        "state_snapshot_path", "state_snapshot.json")

    state = SessionState.load(snapshot_path)
    if state is None:
        console.print(f"[yellow]No {snapshot_path.name} found.[/yellow] "
                      "Nothing has run yet, or it ran from another directory.")
        return 1

    console.print(f"\n[bold]regime-trader[/bold]  session {state.session_id}")
    console.print(f"  saved             {state.saved_at}")
    console.print(f"  bars processed    {state.bars_processed}")
    console.print(f"  last bar          {state.last_bar_timestamp}")
    console.print(f"  regime            {state.last_regime} "
                  f"(p={state.last_regime_confidence:.2f})")
    console.print(f"  equity            ${state.equity_at_save:,.2f}")
    console.print(f"  peak equity       ${state.peak_equity:,.2f}")
    console.print(f"  day open          ${state.day_start_equity:,.2f}")
    console.print(f"  daily trades      {state.daily_trades}")
    console.print(f"  breakers          daily={state.breaker_daily_tripped} "
                  f"weekly={state.breaker_weekly_tripped} peak={state.breaker_peak_tripped}")
    console.print("  stops             "
                  + (", ".join(f"{k} @ {v:.2f}" for k, v in state.stops.items()) or "none"))

    lock = ROOT / "trading_halted.lock"
    if lock.exists():
        console.print(f"\n  [red bold]HALTED[/red bold]  delete {lock} to resume")

    feed = TradingLogger().setup().get_trade_log(limit=15)
    if feed:
        console.print("\n[bold]Recent signals and orders[/bold]")
        for event in feed:
            console.print(f"  {event['timestamp'][11:19]}  {event['event']:<18} "
                          f"{event.get('message', '')}")

    console.print("\n[dim]Web dashboard: python main.py --publish, "
                  "then cd dashboard && npm run dev[/dim]")
    return 0


def run_status(args) -> int:
    """Build status. Touches no network and no broker."""
    done = sum(1 for *_, complete in PHASES if complete)
    print(f"regime-trader: phase {done} of {len(PHASES)} complete.\n")
    for number, name, complete in PHASES:
        print(f"  [{'x' if complete else ' '}] Phase {number}  {name}")
    print("\nAll phases built. The strategy still has no demonstrated edge:")
    print("out-of-sample it loses to buy-and-hold and to random allocation.")
    print("\n  python main.py --dry-run --once   full pipeline, no orders")
    print("  python main.py --mode backtest --compare   out-of-sample validation")
    print("  python main.py --publish          write the web dashboard's data")
    print("  python main.py --dashboard        terminal dashboard")
    print("\nSee README.md, PROGRESS.md, and docs/OPEN-QUESTIONS.md.")
    return 0


def _configure_console_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("hmmlearn", "hmmlearn.base", "urllib3", "alpaca"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def run_backtest(args) -> int:
    """Walk-forward backtest, optionally with benchmarks and stress tests."""
    from rich.console import Console

    from backtest import performance
    from backtest.backtester import WalkForwardBacktester
    from backtest.stress_test import StressTester, render_stress
    from config import load_settings, strategy_config
    from data.market_data import load_bars

    logging.basicConfig(level=logging.WARNING, format="  %(levelname)-7s %(message)s")
    for noisy in ("hmmlearn", "hmmlearn.base", "core.hmm_engine", "core.regime_strategies"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    console = Console()
    settings = load_settings()

    symbols = args.symbols or settings["broker"]["symbols"][:1]
    exit_code = 0

    for symbol in symbols:
        bars, synthetic = load_bars(symbol, args.start, args.end)
        if synthetic:
            console.print(
                f"[yellow]No Alpaca credentials: {symbol} is SYNTHETIC data.[/yellow] "
                "[dim]Plumbing check only, not evidence of edge.[/dim]"
            )

        backtester = WalkForwardBacktester(
            **settings["backtest"],
            hmm_config=dict(settings["hmm"]),
            strategy_config=strategy_config(settings),
        )
        result = backtester.run(bars, symbol)
        report = performance.analyse(
            result, bars,
            risk_free_rate=settings["backtest"]["risk_free_rate"],
            compare=args.compare,
            n_random_seeds=args.random_seeds,
        )
        performance.render(report, console)

        if args.export:
            written = performance.export(report, Path("backtest/results") / symbol)
            console.print(f"\n  Exported: {', '.join(p.name for p in written.values())}")

        if args.stress_test:
            console.print("\n[bold]Running stress tests, this takes a few minutes.[/bold]")
            tester = StressTester(backtester, settings["risk"])
            summaries = tester.run_all_crash_scenarios(bars, args.mc_sims, symbol)
            summaries += tester.gap_risk(bars, n_simulations=max(args.mc_sims // 5, 5),
                                         symbol=symbol)
            if args.misclassification:
                summaries.append(
                    tester.regime_misclassification(
                        bars, n_simulations=max(args.mc_sims // 5, 5), symbol=symbol
                    )
                )
            render_stress(summaries, report.core, console)

        if report.beats_all_benchmarks is False:
            exit_code = 1

    return exit_code


def run_demo(args) -> int:
    """Fit the regime engine and allocator on synthetic data.

    Synthetic because the regimes are constructed and therefore known, which is
    the only way to check the classifier found the right answer rather than
    merely an answer.
    """
    import numpy as np
    import pandas as pd

    from config import load_settings, strategy_config
    from core.hmm_engine import HMMEngine
    from core.regime_strategies import StrategyOrchestrator
    from data.feature_engineering import build_feature_matrix, log_returns, required_raw_bars

    _configure_console_logging(args.log_level)
    settings = load_settings()

    rng = np.random.default_rng(7)
    n = 2600
    index = pd.bdate_range("2014-01-01", periods=n)
    block = np.arange(n) // 180 % 3
    vol = np.select([block == 0, block == 1, block == 2], [0.006, 0.014, 0.032])
    mu = np.select([block == 0, block == 1, block == 2], [0.0009, 0.0002, -0.0012])
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(mu, vol))), index=index)
    span = np.abs(rng.normal(0, vol * 1.5))
    bars = pd.DataFrame(
        {
            "open": close.shift(1).bfill(),
            "high": close * (1 + span),
            "low": close * (1 - span),
            "close": close,
            "volume": rng.lognormal(15 + vol * 20, 0.35),
        },
        index=index,
    )

    features = build_feature_matrix(bars)
    returns = log_returns(bars["close"], 1)

    print("\nSynthetic data: 3 constructed volatility regimes cycling every 180 bars")
    print(f"  {len(bars)} raw bars -> {len(features)} usable feature rows "
          f"({len(bars) - len(features)} discarded to warmup)")
    print(f"  {settings['hmm']['min_train_bars']} usable rows would need "
          f"{required_raw_bars(settings['hmm']['min_train_bars'])} raw bars\n")

    print("Fitting (BIC selection across 3-7 states):")
    config = {k: v for k, v in settings["hmm"].items() if k != "model_path"}
    engine = HMMEngine(**config).fit(features, returns)

    print(f"\nSelected {engine.n_states} states. Regimes found:\n")
    print(engine.summary().round(3).to_string(index=False))

    print("\nExpected duration per regime (bars). Under ~3 would mean noise, not regimes:")
    print("  " + engine.get_expected_durations().round(1).to_string().replace("\n", "\n  "))

    classified = engine.classify_series(features)
    raw_changes = int((classified["raw_state_id"].diff() != 0).sum())
    confirmed = int((classified["state_id"].diff() != 0).sum())
    print(f"\nStability filter: {raw_changes} raw regime changes -> "
          f"{confirmed} confirmed ({raw_changes - confirmed} suppressed as noise)")

    orchestrator = StrategyOrchestrator(strategy_config(settings), engine.regime_info)
    infos = engine.regime_info
    by_return = [infos[i].regime_name for i in sorted(infos, key=lambda k: infos[k].expected_return)]
    by_vol = [infos[i].regime_name for i in sorted(infos, key=lambda k: infos[k].expected_volatility)]
    moved = sum(1 for name in by_return if by_return.index(name) != by_vol.index(name))
    print(f"\n  sorted by return:     {' < '.join(by_return)}")
    print(f"  sorted by volatility: {' < '.join(by_vol)}")
    print(f"  -> {moved}/{len(by_return)} regimes sit in a different position under the "
          f"two sorts.\n     This is why the allocator ignores labels entirely.")

    engine.tracker.reset()
    state = engine.classify(features)
    symbols = settings["broker"]["symbols"][:3]
    signals = orchestrator.generate_signals(symbols, {s: bars for s in symbols}, state)
    if signals:
        first = signals[0]
        print(f"\n  Current regime {state.label.value} -> {first.strategy_name}")
        print(f"    allocation  {first.position_size_pct:.1%} at {first.leverage}x "
              f"= {first.metadata['gross_exposure']:.1%} gross")
        print(f"    stop        {first.stop_loss:.2f} vs entry {first.entry_price:.2f} "
              f"({first.metadata['stop_distance_pct']:.2%} away, "
              f"clamped={first.metadata['stop_clamped']})")
    return 0


# ===========================================================================
# Argument parsing
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="regime-trader: HMM regime detection with volatility-based allocation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py                              live/paper trading\n"
            "  python main.py --dry-run                    full pipeline, no orders\n"
            "  python main.py --dry-run --once             one bar, then exit\n"
            "  python main.py --mode backtest --symbols SPY --start 2019-01-01\n"
            "  python main.py --mode backtest --compare --export\n"
            "  python main.py --mode backtest --stress-test --mc-sims 100\n"
            "  python main.py --train-only --symbols SPY\n"
            "  python main.py --dashboard\n"
        ),
    )

    parser.add_argument("--mode", choices=("trade", "backtest", "demo"), default="trade",
                        help="what to run (default: trade)")
    parser.add_argument("--dry-run", action="store_true",
                        help="full pipeline, no orders. The executor is replaced by one "
                             "that cannot submit.")
    parser.add_argument("--backtest", action="store_true",
                        help="alias for --mode backtest")
    parser.add_argument("--train-only", action="store_true",
                        help="train the HMM and exit")
    parser.add_argument("--dashboard", action="store_true",
                        help="show the dashboard for a running instance")
    parser.add_argument("--status", action="store_true",
                        help="build status only, touches nothing")
    parser.add_argument("--once", action="store_true",
                        help="process a single bar and exit, for cron")
    parser.add_argument("--publish", action="store_true",
                        help="write dashboard/public/data/state.json each bar, for the web UI")
    parser.add_argument("--publish-demo", action="store_true",
                        help="write a labelled demo snapshot and exit")
    parser.add_argument("--live-view", action="store_true",
                        help="with --dashboard, refresh continuously instead of once")

    parser.add_argument("--symbols", nargs="+", help="override the configured universe")
    parser.add_argument("--timeframe", help="override broker.timeframe (default 1Day)")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--i-understand-live", action="store_true",
                        help="permit a live (non-paper) account. Read the README FAQ first.")

    backtest = parser.add_argument_group("backtest options")
    backtest.add_argument("--start", help="start date, YYYY-MM-DD")
    backtest.add_argument("--end", help="end date, YYYY-MM-DD")
    backtest.add_argument("--compare", action="store_true",
                          help="run all three benchmarks (buy-hold, 200 SMA, random)")
    backtest.add_argument("--export", action="store_true", help="write CSVs to backtest/results/")
    backtest.add_argument("--stress-test", action="store_true",
                          help="crash, gap and misclassification tests")
    backtest.add_argument("--mc-sims", type=int, default=100,
                          help="Monte Carlo simulations per scenario")
    backtest.add_argument("--misclassification", action="store_true",
                          help="include the regime misclassification test")
    backtest.add_argument("--random-seeds", type=int, default=100,
                          help="seeds for the random allocation benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.status:
        return run_status(args)
    if args.publish_demo:
        from monitoring.publish import publish_demo

        print(f"Wrote demo snapshot to {publish_demo()}")
        return 0
    if args.dashboard:
        return run_dashboard(args)
    if args.train_only:
        return run_train_only(args)
    if args.backtest or args.mode == "backtest":
        return run_backtest(args)
    if args.mode == "demo":
        return run_demo(args)
    return run_trading(args)


if __name__ == "__main__":
    sys.exit(main())

