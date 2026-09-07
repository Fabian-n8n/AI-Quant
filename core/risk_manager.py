"""
The risk management layer. Position sizing, leverage, circuit breakers, veto.

Phase 5. The most important file in the system, more important than the HMM.
A mediocre strategy with good risk management loses slowly. A good strategy with
bad risk management blows up the account.

INDEPENDENCE IS THE WHOLE DESIGN
--------------------------------
This module operates independently of the HMM. Circuit breakers fire on realised
profit and loss, never on model state, so they still work when the regime engine
is confidently wrong. That is precisely the situation they exist for.

`PortfolioState` carries the regime and flicker rate, but **for logging only**.
Nothing in `check()` reads them to decide whether to fire. The regime is recorded
so that after the fact you can ask "what did the model think when this broke",
which is how you find out whether the regime layer is adding anything.

Phase 4's stress test already probed this: with allocation parameters shuffled
so regimes drive the wrong sizing, worst-case drawdown stayed at 1.0x the
baseline. Damage stays bounded when the model is wrong, which is what
independence buys.

ABSOLUTE VETO
-------------
`validate_signal` is the single choke point. Every signal passes through it, and
its verdict is final: no caller may route around it, and an approval can shrink
a position but never grow one.

TWO CONTRADICTIONS IN THE CONFIGURED LIMITS, BOTH DELIBERATE
------------------------------------------------------------
**1. 1.25x leverage is unreachable.** Leverage is gross exposure divided by
equity, so 1.25x means 125% gross exposure, and `max_exposure` caps gross
exposure at 80%. Both gates are checked as the spec requires, so the exposure
gate always binds first and no position can exceed 0.80x leverage. The entire
"only low-vol regimes may use up to 1.25x" rule is therefore dead code under
these settings. Raising `max_exposure` above 1.25 or lowering `max_leverage` to
0.80 would make the two agree; which one is correct is a decision, not a bug fix.

**2. The gap rule always binds, so risk per trade is 0.667%, not 1%.** Overnight
sizing caps the position so a `gap_multiplier` (3x) gap-through costs at most
`gap_max_loss_pct` (2%) of equity, which works out to 0.667% of equity per
trade. That is below `max_risk_per_trade` (1%), so the minimum always picks the
gap-capped figure. Since this is a swing system, every position is held
overnight, so the 1% figure never applies to anything.

Both are recorded in docs/PHASE5-NOTES.md with the arithmetic.

THE BREAKER THRESHOLDS DO NOT SURVIVE CONTACT WITH THIS STRATEGY
----------------------------------------------------------------
Phase 4's stress test measured the configured breakers firing on **100% of
simulations, in every scenario, including the mildest**, because the unshocked
baseline already draws down 44% against a `max_dd_from_peak` of 10%.

A breaker with a 100% fire rate carries no information, and in live use it
trains you to delete `trading_halted.lock` without reading it. At that point the
safety net is gone while still appearing to be in place. The thresholds ship as
specified; retuning them is a decision to make against backtest output, and
`docs/TUTORIAL-CONFLICTS.md` has swing-horizon starting points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from core.regime_strategies import Direction, Signal

logger = logging.getLogger(__name__)

DEFAULT_LOCK_FILE = Path(__file__).resolve().parent.parent / "trading_halted.lock"


class BreakerType(str, Enum):
    """Which threshold fired. Ordered mild to severe."""
    NONE = "none"
    DAILY_REDUCE = "daily_reduce"
    WEEKLY_REDUCE = "weekly_reduce"
    DAILY_HALT = "daily_halt"
    WEEKLY_HALT = "weekly_halt"
    PEAK_HALT = "peak_halt"


class BreakerState(str, Enum):
    """The action in force. Retained from the Phase 1 skeleton."""
    NORMAL = "normal"
    REDUCED_SIZE = "reduced_size"     # halve new position sizes
    NO_NEW_POSITIONS = "no_new"       # hold what you have, open nothing
    HALTED = "halted"                 # close everything, lock file written


#: Severity ordering. When several breakers fire at once the most severe action
#: wins, so a daily reduce cannot soften a peak halt that fired on the same bar.
BREAKER_SEVERITY: dict[BreakerType, int] = {
    BreakerType.NONE: 0,
    BreakerType.DAILY_REDUCE: 1,
    BreakerType.WEEKLY_REDUCE: 2,
    BreakerType.DAILY_HALT: 3,
    BreakerType.WEEKLY_HALT: 4,
    BreakerType.PEAK_HALT: 5,
}

BREAKER_ACTION: dict[BreakerType, BreakerState] = {
    BreakerType.NONE: BreakerState.NORMAL,
    BreakerType.DAILY_REDUCE: BreakerState.REDUCED_SIZE,
    BreakerType.WEEKLY_REDUCE: BreakerState.REDUCED_SIZE,
    BreakerType.DAILY_HALT: BreakerState.HALTED,
    BreakerType.WEEKLY_HALT: BreakerState.HALTED,
    BreakerType.PEAK_HALT: BreakerState.HALTED,
}


class RiskAction(str, Enum):
    APPROVE = "approve"
    APPROVE_MODIFIED = "approve_modified"
    REJECT = "reject"


class RejectionReason(str, Enum):
    """Structured rejection reasons. Enum rather than free text because these
    get counted: "why did the system stop trading in March" is only answerable
    if rejections are queryable."""
    HALT_LOCK_FILE = "halt_lock_file"
    CIRCUIT_BREAKER = "circuit_breaker"
    DAILY_TRADE_LIMIT = "daily_trade_limit"
    NO_STOP_LOSS = "no_stop_loss"
    INVALID_STOP = "invalid_stop"
    MAX_POSITIONS = "max_positions"
    INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
    NOT_TRADEABLE = "not_tradeable"
    SPREAD_TOO_WIDE = "spread_too_wide"
    DUPLICATE_ORDER = "duplicate_order"
    CORRELATION_TOO_HIGH = "correlation_too_high"
    SECTOR_LIMIT = "sector_limit"
    EXPOSURE_LIMIT = "exposure_limit"
    BELOW_MINIMUM_SIZE = "below_minimum_size"
    INVALID_SIGNAL = "invalid_signal"


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """Snapshot of the account at one moment.

    Drawdowns are computed properties rather than stored fields so they cannot
    go stale: a cached drawdown that was not refreshed is how a breaker fails to
    fire on the one day it mattered.

    `regime`, `regime_confirmed` and `flicker_rate` are carried for **logging
    and leverage eligibility only**. No breaker reads them. That separation is
    the point of the module.
    """
    equity: float
    cash: float = 0.0
    buying_power: float = 0.0
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    week_start_equity: float = 0.0
    daily_trades: int = 0
    timestamp: datetime | None = None

    # Logging and leverage eligibility only. Never read by a circuit breaker.
    regime: str = "unknown"
    regime_confirmed: bool = True
    regime_confidence: float = 1.0
    flicker_rate: int = 0

    def __post_init__(self) -> None:
        self.peak_equity = self.peak_equity or self.equity
        self.day_start_equity = self.day_start_equity or self.equity
        self.week_start_equity = self.week_start_equity or self.equity

    # -- drawdowns, all from realised P&L -----------------------------------

    @property
    def drawdown_daily(self) -> float:
        """Fraction below the day's opening equity. Zero or negative."""
        if self.day_start_equity <= 0:
            return 0.0
        return min(0.0, self.equity / self.day_start_equity - 1)

    @property
    def drawdown_weekly(self) -> float:
        if self.week_start_equity <= 0:
            return 0.0
        return min(0.0, self.equity / self.week_start_equity - 1)

    @property
    def drawdown_from_peak(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return min(0.0, self.equity / self.peak_equity - 1)

    # -- exposure -----------------------------------------------------------

    @property
    def gross_exposure(self) -> float:
        """Total position value as a fraction of equity.

        Gross, not net: this system is long-only, so the two coincide today, but
        naming it gross keeps the meaning correct if that ever changes.
        """
        if self.equity <= 0:
            return 0.0
        return sum(abs(p.get("market_value", 0.0)) for p in self.positions.values()) / self.equity

    @property
    def leverage(self) -> float:
        """Same quantity as gross exposure, named for the limit it is checked
        against. Kept distinct because the spec sets two separate caps on it."""
        return self.gross_exposure

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    def sector_exposure(self, sector_map: dict[str, str] | None = None) -> dict[str, float]:
        """Exposure grouped by sector, as a fraction of equity.

        Five positions in NVDA, AMD, META, GOOGL and MSFT is not five positions,
        it is one bet on large-cap tech wearing five hats, and the concurrent
        limit does nothing about that.
        """
        if self.equity <= 0:
            return {}
        sector_map = sector_map or {}
        out: dict[str, float] = {}
        for symbol, position in self.positions.items():
            sector = sector_map.get(symbol, position.get("sector", "unknown"))
            out[sector] = out.get(sector, 0.0) + abs(position.get("market_value", 0.0)) / self.equity
        return out


@dataclass(frozen=True)
class BreakerTrigger:
    """One breaker firing, with the context needed to review it later.

    `regime` records what the model believed at the moment the loss happened.
    Comparing that against what actually followed is how you learn whether the
    regime layer is contributing or just adding noise.
    """
    timestamp: datetime
    breaker: BreakerType
    action: BreakerState
    drawdown: float
    equity: float
    peak_equity: float
    positions_open: int
    positions_closed: int
    regime: str
    regime_confidence: float
    detail: str = ""


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Drawdown thresholds evaluated against realised P&L.

    Fires on money lost, never on model state. Daily and weekly breakers reset
    on their own boundaries; the peak breaker does not reset at all and writes
    `trading_halted.lock`, which requires manual deletion.

    The friction of that file is deliberate. It forces someone to go and look at
    what broke before the system can lose more, and it cannot be cleared by a
    restart or a scheduled job.
    """

    def __init__(self, config: dict[str, Any], lock_file: Path | None = None) -> None:
        self.config = dict(config)
        self.lock_file = Path(lock_file) if lock_file else DEFAULT_LOCK_FILE

        self.daily_dd_reduce = self.config.get("daily_dd_reduce", 0.02)
        self.daily_dd_halt = self.config.get("daily_dd_halt", 0.03)
        self.weekly_dd_reduce = self.config.get("weekly_dd_reduce", 0.05)
        self.weekly_dd_halt = self.config.get("weekly_dd_halt", 0.07)
        self.max_dd_from_peak = self.config.get("max_dd_from_peak", 0.10)

        self.daily_tripped: BreakerType = BreakerType.NONE
        self.weekly_tripped: BreakerType = BreakerType.NONE
        self.peak_tripped: bool = False
        self.history: list[BreakerTrigger] = []

    # -- evaluation ---------------------------------------------------------

    def check(self, state: PortfolioState) -> BreakerType:
        """The most severe breaker currently in force.

        Latched: once a daily breaker trips it stays tripped until
        `reset_daily()`, even if equity recovers within the same session. A
        breaker that un-fires on a bounce is not a circuit breaker, it is a
        lagging indicator.
        """
        candidates = [self._evaluate(state), self.daily_tripped, self.weekly_tripped]
        if self.peak_tripped or self.is_halted():
            candidates.append(BreakerType.PEAK_HALT)
        return max(candidates, key=lambda b: BREAKER_SEVERITY[b])

    def _evaluate(self, state: PortfolioState) -> BreakerType:
        """Fresh evaluation of the thresholds against this snapshot."""
        triggered = BreakerType.NONE

        if state.drawdown_from_peak <= -self.max_dd_from_peak:
            triggered = BreakerType.PEAK_HALT
        elif state.drawdown_weekly <= -self.weekly_dd_halt:
            triggered = BreakerType.WEEKLY_HALT
        elif state.drawdown_daily <= -self.daily_dd_halt:
            triggered = BreakerType.DAILY_HALT
        elif state.drawdown_weekly <= -self.weekly_dd_reduce:
            triggered = BreakerType.WEEKLY_REDUCE
        elif state.drawdown_daily <= -self.daily_dd_reduce:
            triggered = BreakerType.DAILY_REDUCE

        return triggered

    def update(self, state: PortfolioState, positions_closed: int = 0) -> BreakerState:
        """Evaluate, latch, log, and write the lock file if the peak breaker fires.

        Call once per bar with a fresh snapshot. Returns the action in force.
        """
        triggered = self._evaluate(state)

        if triggered is not BreakerType.NONE:
            self._latch(triggered)
            self._log_trigger(triggered, state, positions_closed)

        if triggered is BreakerType.PEAK_HALT and not self.lock_file.exists():
            self.halt(
                f"peak drawdown {state.drawdown_from_peak:.2%} breached "
                f"{-self.max_dd_from_peak:.2%}",
                state,
            )

        return BREAKER_ACTION[self.check(state)]

    def _latch(self, triggered: BreakerType) -> None:
        if triggered in (BreakerType.DAILY_REDUCE, BreakerType.DAILY_HALT):
            if BREAKER_SEVERITY[triggered] > BREAKER_SEVERITY[self.daily_tripped]:
                self.daily_tripped = triggered
        elif triggered in (BreakerType.WEEKLY_REDUCE, BreakerType.WEEKLY_HALT):
            if BREAKER_SEVERITY[triggered] > BREAKER_SEVERITY[self.weekly_tripped]:
                self.weekly_tripped = triggered
        elif triggered is BreakerType.PEAK_HALT:
            self.peak_tripped = True

    def _log_trigger(self, triggered: BreakerType, state: PortfolioState, closed: int) -> None:
        drawdown = {
            BreakerType.DAILY_REDUCE: state.drawdown_daily,
            BreakerType.DAILY_HALT: state.drawdown_daily,
            BreakerType.WEEKLY_REDUCE: state.drawdown_weekly,
            BreakerType.WEEKLY_HALT: state.drawdown_weekly,
            BreakerType.PEAK_HALT: state.drawdown_from_peak,
        }.get(triggered, 0.0)

        trigger = BreakerTrigger(
            timestamp=state.timestamp or datetime.now(UTC),
            breaker=triggered,
            action=BREAKER_ACTION[triggered],
            drawdown=drawdown,
            equity=state.equity,
            peak_equity=state.peak_equity,
            positions_open=state.n_positions,
            positions_closed=closed,
            regime=state.regime,
            regime_confidence=state.regime_confidence,
        )
        self.history.append(trigger)
        logger.warning(
            "CIRCUIT BREAKER %s: drawdown %.2f%%, equity %.0f, %d positions open, "
            "regime was %s (p=%.2f)",
            triggered.value, drawdown * 100, state.equity, state.n_positions,
            state.regime, state.regime_confidence,
        )

    # -- lock file ----------------------------------------------------------

    def halt(self, reason: str, state: PortfolioState | None = None) -> Path:
        """Write `trading_halted.lock`. Requires manual deletion to resume."""
        self.peak_tripped = True
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "TRADING HALTED",
            f"time:   {datetime.now(UTC).isoformat()}",
            f"reason: {reason}",
        ]
        if state is not None:
            lines += [
                f"equity: {state.equity:,.2f}",
                f"peak:   {state.peak_equity:,.2f}",
                f"drawdown from peak: {state.drawdown_from_peak:.2%}",
                f"positions open: {state.n_positions}",
                f"regime at time: {state.regime} (p={state.regime_confidence:.2f})",
            ]
        lines += [
            "",
            "Delete this file to resume trading.",
            "Read the logs and understand what broke before you do.",
        ]
        self.lock_file.write_text("\n".join(lines) + "\n")
        logger.error("TRADING HALTED, wrote %s: %s", self.lock_file, reason)
        return self.lock_file

    def is_halted(self) -> bool:
        """True while the lock file exists. Checked first on every signal."""
        return self.lock_file.exists()

    def clear_halt(self) -> None:
        """Delete the lock file. Deliberately not called anywhere automatically.

        Present so tests can clean up. Wiring this into a scheduled job or a
        restart path would defeat the entire mechanism.
        """
        if self.lock_file.exists():
            self.lock_file.unlink()
        self.peak_tripped = False

    # -- resets -------------------------------------------------------------

    def reset_daily(self) -> None:
        """Clear daily breakers at the session boundary. Does not touch weekly
        or peak, which are longer-horizon and reset on their own terms."""
        if self.daily_tripped is not BreakerType.NONE:
            logger.info("Reset daily breaker (%s)", self.daily_tripped.value)
        self.daily_tripped = BreakerType.NONE

    def reset_weekly(self) -> None:
        if self.weekly_tripped is not BreakerType.NONE:
            logger.info("Reset weekly breaker (%s)", self.weekly_tripped.value)
        self.weekly_tripped = BreakerType.NONE
        self.reset_daily()

    def get_history(self) -> pd.DataFrame:
        """Every trigger, with the regime at the time. Empty if none fired."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp": t.timestamp, "breaker": t.breaker.value, "action": t.action.value,
                "drawdown": t.drawdown, "equity": t.equity, "peak_equity": t.peak_equity,
                "positions_open": t.positions_open, "positions_closed": t.positions_closed,
                "regime": t.regime, "regime_confidence": t.regime_confidence,
            }
            for t in self.history
        ])

    @property
    def size_multiplier(self) -> float:
        """0.5 while a reduce breaker is latched, 0.0 while halted."""
        action = BREAKER_ACTION[
            max([self.daily_tripped, self.weekly_tripped]
                + ([BreakerType.PEAK_HALT] if self.peak_tripped else []),
                key=lambda b: BREAKER_SEVERITY[b])
        ]
        if action is BreakerState.HALTED:
            return 0.0
        if action is BreakerState.REDUCED_SIZE:
            return 0.5
        return 1.0


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class RiskDecision:
    """The verdict on one signal.

    Rejections carry a structured `rejection_reason` so they can be counted, and
    `modifications` records every shrink applied, so a position that came out at
    a third of its requested size explains why without a debugging session.
    """
    approved: bool
    action: RiskAction = RiskAction.REJECT
    modified_signal: dict[str, Any] = field(default_factory=dict)
    rejection_reason: RejectionReason | None = None
    reason: str = ""
    modifications: list[str] = field(default_factory=list)

    # Retained from the Phase 1 skeleton so the backtester keeps working.
    approved_quantity: float = 0.0
    approved_notional: float = 0.0
    size_multiplier: float = 1.0

    @classmethod
    def reject(cls, reason: RejectionReason, detail: str) -> RiskDecision:
        return cls(approved=False, action=RiskAction.REJECT,
                   rejection_reason=reason, reason=detail)


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------

class RiskManager:
    """Absolute veto over every signal.

    `validate_signal` runs a 13-step cascade. Order matters: the cheapest and
    most absolute checks come first, so a halted system rejects immediately
    without computing a correlation matrix, and sizing happens only after the
    signal is known to be legal at all.

    An approval may shrink a position. It may never grow one.
    """

    def __init__(
        self,
        config: dict[str, Any],
        lock_file: Path | None = None,
        sector_map: dict[str, str] | None = None,
        price_history: pd.DataFrame | None = None,
    ) -> None:
        self.config = dict(config)
        self.breaker = CircuitBreaker(self.config, lock_file)
        self.sector_map = dict(sector_map or {})
        self.price_history = price_history

        self.max_risk_per_trade = self.config.get("max_risk_per_trade", 0.01)
        self.max_exposure = self.config.get("max_exposure", 0.80)
        self.max_leverage = self.config.get("max_leverage", 1.25)
        self.max_single_position = self.config.get("max_single_position", 0.15)
        self.max_concurrent = self.config.get("max_concurrent", 5)
        self.max_daily_trades = self.config.get("max_daily_trades", 20)
        self.min_position_usd = self.config.get("min_position_usd", 100.0)
        self.max_sector_exposure = self.config.get("max_sector_exposure", 0.30)
        self.gap_multiplier = self.config.get("gap_multiplier", 3.0)
        self.gap_max_loss_pct = self.config.get("gap_max_loss_pct", 0.02)
        self.correlation_window = self.config.get("correlation_window", 60)
        self.correlation_reduce = self.config.get("correlation_reduce", 0.70)
        self.correlation_reject = self.config.get("correlation_reject", 0.85)
        self.max_spread_pct = self.config.get("max_spread_pct", 0.005)
        self.duplicate_window_seconds = self.config.get("duplicate_window_seconds", 60)
        self.force_1x_position_count = self.config.get("force_1x_position_count", 3)

        self._recent_orders: list[tuple[str, str, datetime]] = []

    # -- the cascade --------------------------------------------------------

    def validate_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        quote: dict[str, float] | None = None,
        overnight: bool = True,
        record: bool = True,
    ) -> RiskDecision:
        """Approve, shrink or reject. This is the veto.

        `overnight` defaults to True because this is a swing system: every
        position is held overnight, so the gap cap applies by default rather
        than as an exception.

        `record=False` runs the identical cascade without registering the signal
        for duplicate detection. That is what the watchlist scan uses: asking
        "what would happen if I traded this" must not make the real answer
        different a second later by tripping the duplicate window.
        """
        modifications: list[str] = []

        # 1. Halt lock file. Absolute, and checked before anything else so a
        #    halted system cannot be talked into a trade by any later step.
        if self.breaker.is_halted():
            return RiskDecision.reject(
                RejectionReason.HALT_LOCK_FILE,
                f"trading_halted.lock exists at {self.breaker.lock_file}. "
                "Delete it manually after reviewing what broke.",
            )

        # 2. Circuit breakers, on realised P&L only.
        breaker_type = self.breaker.check(portfolio_state)
        action = BREAKER_ACTION[breaker_type]
        if action is BreakerState.HALTED:
            return RiskDecision.reject(
                RejectionReason.CIRCUIT_BREAKER,
                f"{breaker_type.value} active: daily DD "
                f"{portfolio_state.drawdown_daily:.2%}, weekly "
                f"{portfolio_state.drawdown_weekly:.2%}, peak "
                f"{portfolio_state.drawdown_from_peak:.2%}",
            )
        breaker_multiplier = 0.5 if action is BreakerState.REDUCED_SIZE else 1.0
        if breaker_multiplier < 1.0:
            modifications.append(f"{breaker_type.value}: size halved")

        # 3. Daily trade count.
        if portfolio_state.daily_trades >= self.max_daily_trades:
            return RiskDecision.reject(
                RejectionReason.DAILY_TRADE_LIMIT,
                f"{portfolio_state.daily_trades} trades today, limit {self.max_daily_trades}",
            )

        # 4. Stop loss. Non-negotiable: sizing divides by the distance to it, so
        #    a missing or wrong-side stop is not merely risky, it is undefined.
        if signal.stop_loss is None:
            return RiskDecision.reject(
                RejectionReason.NO_STOP_LOSS, f"{signal.symbol}: no stop loss"
            )
        if signal.direction is Direction.LONG and signal.stop_loss >= signal.entry_price:
            return RiskDecision.reject(
                RejectionReason.INVALID_STOP,
                f"{signal.symbol}: stop {signal.stop_loss:.4f} is not below entry "
                f"{signal.entry_price:.4f}",
            )
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        if stop_distance <= 0 or signal.entry_price <= 0:
            return RiskDecision.reject(
                RejectionReason.INVALID_SIGNAL, f"{signal.symbol}: non-positive stop distance"
            )

        # 5. Concurrent positions. Resizing something already held is always
        #    allowed: refusing it would trap the system at max positions with no
        #    way to reduce risk.
        is_existing = signal.symbol in portfolio_state.positions
        if not is_existing and portfolio_state.n_positions >= self.max_concurrent:
            return RiskDecision.reject(
                RejectionReason.MAX_POSITIONS,
                f"{portfolio_state.n_positions} positions open, limit {self.max_concurrent}",
            )

        # 6. Risk-based sizing.
        shares = self.position_size(portfolio_state.equity, signal.entry_price, signal.stop_loss)
        notional = shares * signal.entry_price
        cap = portfolio_state.equity * self.max_single_position
        if notional > cap:
            shares = int(cap / signal.entry_price)
            notional = shares * signal.entry_price
            modifications.append(f"capped at {self.max_single_position:.0%} single position")

        # 7. Exposure and leverage. Shrink to fit rather than reject: a position
        #    at 60% of the requested size is better than no position, and the
        #    caller asked for exposure the account cannot carry, not for
        #    something illegal.
        available = max(0.0, (self.max_exposure - portfolio_state.gross_exposure)) * portfolio_state.equity
        leverage_room = max(0.0, (self.max_leverage - portfolio_state.leverage)) * portfolio_state.equity
        room = min(available, leverage_room)
        if notional > room:
            if room < self.min_position_usd:
                return RiskDecision.reject(
                    RejectionReason.EXPOSURE_LIMIT,
                    f"gross exposure {portfolio_state.gross_exposure:.2%} leaves "
                    f"${room:,.0f} of room, below the ${self.min_position_usd:,.0f} minimum",
                )
            shares = int(room / signal.entry_price)
            notional = shares * signal.entry_price
            modifications.append(
                f"shrunk to fit {self.max_exposure:.0%} exposure / "
                f"{self.max_leverage:.2f}x leverage"
            )

        # 8. Overnight gap risk.
        if overnight:
            gap_shares = self.gap_capped_size(portfolio_state.equity, stop_distance)
            if gap_shares < shares:
                shares = gap_shares
                notional = shares * signal.entry_price
                modifications.append(
                    f"gap cap: {self.gap_multiplier:g}x stop gap-through kept under "
                    f"{self.gap_max_loss_pct:.0%} of portfolio"
                )

        # 9. Leverage eligibility.
        leverage, leverage_note = self.allowed_leverage(signal, portfolio_state, breaker_type)
        if leverage_note:
            modifications.append(leverage_note)

        # 10. Correlation with what is already held.
        correlation_multiplier, correlation_note = self.check_correlation(
            signal.symbol, portfolio_state
        )
        if correlation_multiplier == 0.0:
            return RiskDecision.reject(RejectionReason.CORRELATION_TOO_HIGH, correlation_note)
        if correlation_multiplier < 1.0:
            modifications.append(correlation_note)

        # 11. Sector exposure.
        sector_ok, sector_note = self.check_sector(signal.symbol, notional, portfolio_state)
        if not sector_ok:
            return RiskDecision.reject(RejectionReason.SECTOR_LIMIT, sector_note)

        # 12. Duplicate orders.
        if self.is_duplicate(signal):
            return RiskDecision.reject(
                RejectionReason.DUPLICATE_ORDER,
                f"{signal.symbol} {signal.direction.value} already sent within "
                f"{self.duplicate_window_seconds}s",
            )

        # Apply the accumulated multipliers, then re-check the floor.
        multiplier = breaker_multiplier * correlation_multiplier
        if multiplier < 1.0:
            shares = int(shares * multiplier)
            notional = shares * signal.entry_price

        # Order validation against live quote data, when available.
        if quote is not None:
            ok, reason, detail = self.validate_order(signal, notional, portfolio_state, quote)
            if not ok:
                return RiskDecision.reject(reason, detail)

        # 13. Minimum size.
        if shares <= 0 or notional < self.min_position_usd:
            return RiskDecision.reject(
                RejectionReason.BELOW_MINIMUM_SIZE,
                f"{signal.symbol}: ${notional:,.0f} is below the "
                f"${self.min_position_usd:,.0f} minimum after all reductions",
            )

        if record:
            self._record_order(signal)
        risk_dollars = shares * stop_distance
        return RiskDecision(
            approved=True,
            action=RiskAction.APPROVE_MODIFIED if modifications else RiskAction.APPROVE,
            modified_signal={
                "symbol": signal.symbol,
                "direction": signal.direction.value,
                "shares": shares,
                "notional": notional,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "leverage": leverage,
                "risk_dollars": risk_dollars,
                "risk_pct_of_equity": risk_dollars / portfolio_state.equity
                if portfolio_state.equity else 0.0,
            },
            reason="; ".join(modifications) if modifications else "approved unmodified",
            modifications=modifications,
            approved_quantity=shares,
            approved_notional=notional,
            size_multiplier=multiplier,
        )

    # -- sizing -------------------------------------------------------------

    def position_size(self, equity: float, entry_price: float, stop_loss: float) -> int:
        """Shares to buy: `(equity * max_risk_per_trade) / abs(entry - stop)`.

        Risk a fixed fraction of the account on every trade. A wide stop means
        fewer shares, a tight stop more, so each trade loses roughly the same
        dollar amount when it is wrong. That is what makes a losing streak
        survivable, and breakout-style systems have long ones.

        Truncated to whole shares, so the position can only ever be smaller than
        the target, never larger.
        """
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0 or entry_price <= 0 or equity <= 0:
            return 0
        return int((equity * self.max_risk_per_trade) / stop_distance)

    def gap_capped_size(self, equity: float, stop_distance: float) -> int:
        """Size assuming the stop gaps through by `gap_multiplier` times its distance.

            shares = (equity * gap_max_loss_pct) / (gap_multiplier * stop_distance)

        Stops do not protect through a gap: a close at 50 with a stop at 48 and
        an open at 44 fills at 44. This sizes for that rather than for the stop
        holding.

        With the shipped 3x and 2%, this allows 0.667% of equity at risk, which
        is **below** `max_risk_per_trade` (1%). The gap cap therefore always
        binds, and since every position in a swing system is held overnight, the
        1% figure never applies to anything. See docs/PHASE5-NOTES.md.
        """
        if stop_distance <= 0 or equity <= 0:
            return 0
        worst_case_move = self.gap_multiplier * stop_distance
        return int((equity * self.gap_max_loss_pct) / worst_case_move)

    def allowed_leverage(
        self, signal: Signal, state: PortfolioState, breaker: BreakerType
    ) -> tuple[float, str]:
        """Leverage for this signal. Defaults to 1.0x and is rarely more.

        Forced to 1.0x when the regime is uncertain, any breaker is active,
        three or more positions are open, or the model is flickering. Each of
        those says the same thing: conditions are not clean enough to borrow.

        Note the ceiling is unreachable in practice. `max_leverage` is 1.25 but
        `max_exposure` caps gross exposure at 0.80, and leverage IS gross
        exposure, so step 7 binds first. Recorded here because the rule is
        specified and the interaction is not obvious.
        """
        requested = min(getattr(signal, "leverage", 1.0) or 1.0, self.max_leverage)
        if requested <= 1.0:
            return 1.0, ""

        reasons = []
        if not state.regime_confirmed:
            reasons.append("regime unconfirmed")
        if state.regime_confidence < self.config.get("min_confidence", 0.55):
            reasons.append(f"confidence {state.regime_confidence:.2f}")
        if breaker is not BreakerType.NONE:
            reasons.append(f"breaker {breaker.value} active")
        if state.n_positions >= self.force_1x_position_count:
            reasons.append(f"{state.n_positions} positions open")
        if state.flicker_rate > self.config.get("flicker_threshold", 4):
            reasons.append(f"flicker rate {state.flicker_rate}")

        if reasons:
            return 1.0, f"leverage forced to 1.0x ({'; '.join(reasons)})"
        return requested, ""

    # -- correlation and sector --------------------------------------------

    def check_correlation(
        self, symbol: str, state: PortfolioState, returns: pd.DataFrame | None = None
    ) -> tuple[float, str]:
        """Rolling correlation against open positions.

        Above `correlation_reject` (0.85) the trade is refused; above
        `correlation_reduce` (0.70) the size is halved. Returns the multiplier
        and a note, with 0.0 meaning reject.

        Returns 1.0 with no note when price history is unavailable. That is a
        deliberate fail-open: the alternative is refusing every trade whenever
        the data feed is thin, which turns a data problem into an outage. The
        concurrent-position and sector limits still apply.
        """
        history = returns if returns is not None else self.price_history
        if history is None or not state.positions:
            return 1.0, ""

        held = [s for s in state.positions if s != symbol and s in history.columns]
        if symbol not in history.columns or not held:
            return 1.0, ""

        window = history.tail(self.correlation_window)
        if len(window) < 10:
            return 1.0, ""

        correlations = window[held].corrwith(window[symbol]).dropna()
        if correlations.empty:
            return 1.0, ""

        worst = float(correlations.abs().max())
        partner = str(correlations.abs().idxmax())

        if worst > self.correlation_reject:
            return 0.0, (
                f"{symbol} correlates {worst:.2f} with {partner}, above the "
                f"{self.correlation_reject:.2f} reject threshold"
            )
        if worst > self.correlation_reduce:
            return 0.5, (
                f"correlation {worst:.2f} with {partner}: size halved"
            )
        return 1.0, ""

    def check_sector(
        self, symbol: str, notional: float, state: PortfolioState
    ) -> tuple[bool, str]:
        """Reject if this position would push a sector past its cap."""
        if not self.sector_map or state.equity <= 0:
            return True, ""
        sector = self.sector_map.get(symbol)
        if sector is None:
            return True, ""

        current = state.sector_exposure(self.sector_map).get(sector, 0.0)
        proposed = current + notional / state.equity
        if proposed > self.max_sector_exposure:
            return False, (
                f"{sector} exposure would reach {proposed:.1%}, above the "
                f"{self.max_sector_exposure:.0%} cap"
            )
        return True, ""

    # -- order validation ---------------------------------------------------

    def validate_order(
        self,
        signal: Signal,
        notional: float,
        state: PortfolioState,
        quote: dict[str, float],
    ) -> tuple[bool, RejectionReason | None, str]:
        """Buying power, tradeable status and bid-ask spread."""
        if not quote.get("tradeable", True):
            return False, RejectionReason.NOT_TRADEABLE, f"{signal.symbol} is not tradeable"

        bid, ask = quote.get("bid", 0.0), quote.get("ask", 0.0)
        if bid > 0 and ask > 0:
            spread = (ask - bid) / ((ask + bid) / 2)
            if spread > self.max_spread_pct:
                return False, RejectionReason.SPREAD_TOO_WIDE, (
                    f"{signal.symbol} spread {spread:.2%} exceeds "
                    f"{self.max_spread_pct:.2%}"
                )

        if state.buying_power and notional > state.buying_power:
            return False, RejectionReason.INSUFFICIENT_BUYING_POWER, (
                f"${notional:,.0f} exceeds ${state.buying_power:,.0f} buying power"
            )
        return True, None, ""

    def is_duplicate(self, signal: Signal) -> bool:
        """True if the same symbol and direction was sent inside the window.

        Guards against a retry loop or a double-invoked scheduler placing the
        same order twice, which on a broker API is silent and expensive.
        """
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=self.duplicate_window_seconds)
        self._recent_orders = [o for o in self._recent_orders if o[2] > cutoff]
        return any(
            symbol == signal.symbol and direction == signal.direction.value
            for symbol, direction, _ in self._recent_orders
        )

    def _record_order(self, signal: Signal) -> None:
        self._recent_orders.append(
            (signal.symbol, signal.direction.value, datetime.now(UTC))
        )

    # -- state --------------------------------------------------------------

    def update(self, state: PortfolioState, positions_closed: int = 0) -> BreakerState:
        """Advance the breakers with a fresh snapshot. Call once per bar."""
        return self.breaker.update(state, positions_closed)

    def is_halted(self) -> bool:
        return self.breaker.is_halted()

    def halt(self, reason: str, state: PortfolioState | None = None) -> Path:
        return self.breaker.halt(reason, state)

    def reset_daily(self) -> None:
        self.breaker.reset_daily()

    def reset_weekly(self) -> None:
        self.breaker.reset_weekly()

    def get_breaker_history(self) -> pd.DataFrame:
        return self.breaker.get_history()

    # -- backward compatibility ---------------------------------------------

    def check_breakers(self, equity_curve: pd.Series) -> BreakerState:
        """Evaluate breakers from an equity curve.

        Replaces `backtest.stress_test.evaluate_breakers`, which existed only
        because this class did not. Phase 4 flagged that keeping two
        implementations of the same thresholds would let them drift.
        """
        if equity_curve.empty:
            return BreakerState.NORMAL
        state = PortfolioState(
            equity=float(equity_curve.iloc[-1]),
            peak_equity=float(equity_curve.cummax().iloc[-1]),
            day_start_equity=float(equity_curve.iloc[-2]) if len(equity_curve) > 1
            else float(equity_curve.iloc[-1]),
            week_start_equity=float(equity_curve.iloc[-6]) if len(equity_curve) > 5
            else float(equity_curve.iloc[0]),
        )
        return BREAKER_ACTION[self.breaker.check(state)]

    def evaluate(
        self, signal: Signal, portfolio_value: float, open_positions: dict
    ) -> RiskDecision:
        """Legacy entry point retained for the backtester and the Phase 1 tests."""
        return self.validate_signal(
            signal,
            PortfolioState(equity=portfolio_value, positions=dict(open_positions or {})),
        )
