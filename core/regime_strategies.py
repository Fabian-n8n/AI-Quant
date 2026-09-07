"""
Volatility-based allocation strategies. The layer that turns a regime into a
position size.

Phase 3.

DESIGN INSIGHT
--------------
The HMM detects volatility environments, not market direction. Stocks trend
upward roughly 70% of the time in low-volatility periods, and the worst
drawdowns cluster in high-volatility spikes. So the allocation rule is simple:

    low vol   -> be fully invested (calm markets trend up)
    mid vol   -> stay invested if the trend is intact, reduce if it is not
    high vol  -> reduce but stay partially invested (catch V-shaped rebounds)

The edge comes from avoiding big drawdowns through volatility-based sizing.
Halving your worst drawdown lets compounding work: recovering from -50% needs
+100%, recovering from -25% needs +33%.

ALWAYS LONG. NEVER SHORT.
-------------------------
Shorting was tested extensively in walk-forward backtesting and consistently
destroyed returns:

1. Markets have long-term upward drift.
2. V-shaped recoveries happen fast and the HMM is 2-3 days late detecting them.
3. Short positions during rebounds wipe out the gains made during the crash.

The correct response to high volatility is REDUCING allocation, not reversing
direction. `Direction` has no SHORT member, so this is structurally impossible
rather than merely discouraged. A future edit that wants to short has to add the
enum member, which is a visible act rather than a passed argument.

LABELS ARE IGNORED HERE
-----------------------
The orchestrator maps regimes to strategies by **expected_volatility**, never by
label. A regime labelled "bull" is not necessarily low volatility: the labels
sort by return and this sorts by volatility, and those orderings genuinely
disagree. Crash and euphoria sit at opposite ends of the return sort and
adjacent on this one. Keying off the label would size by direction, which is
precisely what the design insight above says not to do.

THE STOP CLAMP, READ THIS BEFORE CHANGING STOPS
-----------------------------------------------
The spec's stop formulas are anchored to the 50 EMA. When price is below the
EMA, which is most of what a selloff is, those formulas return a stop **above**
the entry price. Measured on the synthetic fixture:

    HighVol  EMA50 - 1.0*ATR              stop >= entry on 25% of bars
    MidVol   EMA50 - 0.5*ATR              stop >= entry on 36% of bars
    LowVol   max(P - 3ATR, EMA50 - 0.5ATR) stop >= entry on 36% of bars

For a long position that is an instant stop-out. Worse, Phase 5 sizes positions
as `risk / abs(entry - stop)`, so a stop equal to entry is a division by zero
and a stop above entry silently sizes off a meaningless distance.

Every stop is therefore clamped to sit at least `min_stop_atr_mult` ATR below
entry. The spec's formula is preserved and recorded in `metadata["raw_stop"]`
alongside `metadata["stop_clamped"]`, so the clamp is auditable rather than
invisible.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from core.hmm_engine import (
    Regime,
    RegimeInfo,
    RegimeState,
    VolatilityRank,
    assign_volatility_ranks,
)
from data.feature_engineering import atr, ema

logger = logging.getLogger(__name__)

UNCERTAINTY_TAG = "[UNCERTAINTY — size halved]"


class Direction(str, Enum):
    """LONG or FLAT. There is deliberately no SHORT.

    See the module docstring: shorting destroyed returns in walk-forward
    testing. Omitting the member makes it a code change rather than a
    parameter, which is the right amount of friction.
    """
    LONG = "long"
    FLAT = "flat"


@dataclass(frozen=True)
class Signal:
    """One symbol's target, before the risk manager has looked at it.

    `position_size_pct` is the **portfolio-level** allocation the regime calls
    for (0.60 to 0.95), not this symbol's share of it. The per-symbol weight is
    that figure divided across the surviving signals, and is carried in
    `metadata["per_symbol_weight"]` so nothing downstream has to guess.

    `stop_loss` is always strictly below `entry_price`. See the module docstring.
    """
    symbol: str
    direction: Direction
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float | None
    position_size_pct: float
    leverage: float
    regime_id: int
    regime_name: str
    regime_probability: float
    timestamp: pd.Timestamp
    reasoning: str
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction is Direction.LONG and self.stop_loss >= self.entry_price:
            raise ValueError(
                f"{self.symbol}: stop {self.stop_loss:.4f} is not below entry "
                f"{self.entry_price:.4f}. A long position with a stop at or above "
                f"entry is an instant stop-out and divides by zero when sized."
            )

    @property
    def risk_per_share(self) -> float:
        """Distance from entry to stop. The denominator of the sizing formula."""
        return abs(self.entry_price - self.stop_loss)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """One strategy per volatility tier.

    All three share this shape so the orchestrator can swap between them without
    knowing which is active, and so Phase 4 can benchmark them like for like.
    """

    name: str = "base"
    direction: Direction = Direction.LONG

    def __init__(
        self,
        allocation: float,
        leverage: float = 1.0,
        atr_window: int = 14,
        ema_span: int = 50,
        min_stop_atr_mult: float = 0.5,
        min_stop_pct: float = 0.005,
    ) -> None:
        self.allocation = allocation
        self.leverage = leverage
        self.atr_window = atr_window
        self.ema_span = ema_span
        self.min_stop_atr_mult = min_stop_atr_mult
        self.min_stop_pct = min_stop_pct

    @abstractmethod
    def compute_allocation(self, bars: pd.DataFrame) -> tuple[float, float, str]:
        """Return (allocation, leverage, reason) for the latest bar."""

    @abstractmethod
    def compute_raw_stop(self, price: float, ema50: float, atr_value: float) -> float:
        """The spec's stop formula, before clamping."""

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Signal | None:
        """Build a signal for one symbol, or None if the bars cannot support one.

        Returns None rather than raising when indicators are still in warmup:
        a symbol with 30 bars of history is not an error, it is a symbol that
        cannot be traded yet.
        """
        context = self._indicator_context(bars)
        if context is None:
            logger.debug("%s: insufficient history for %s", symbol, self.name)
            return None

        price, ema50, atr_value, timestamp = context
        allocation, leverage, reason = self.compute_allocation(bars)
        raw_stop = self.compute_raw_stop(price, ema50, atr_value)
        stop, clamped = self._clamp_stop(price, raw_stop, atr_value)

        return Signal(
            symbol=symbol,
            direction=self.direction,
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop,
            take_profit=None,
            position_size_pct=allocation,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label.value,
            regime_probability=regime_state.probability,
            timestamp=timestamp,
            reasoning=reason,
            strategy_name=self.name,
            metadata={
                "ema50": ema50,
                "atr": atr_value,
                "raw_stop": raw_stop,
                "stop_clamped": clamped,
                "stop_distance_pct": (price - stop) / price,
                "volatility_rank": None,
                "price_above_ema": price > ema50,
            },
        )

    # -- internals ----------------------------------------------------------

    def _indicator_context(
        self, bars: pd.DataFrame
    ) -> tuple[float, float, float, pd.Timestamp] | None:
        required = {"high", "low", "close"}
        if not required <= set(bars.columns) or len(bars) < max(self.ema_span, self.atr_window) + 1:
            return None

        ema_series = ema(bars["close"], self.ema_span)
        atr_series = atr(bars["high"], bars["low"], bars["close"], self.atr_window)

        price = float(bars["close"].iloc[-1])
        ema50 = float(ema_series.iloc[-1])
        atr_value = float(atr_series.iloc[-1])

        if not all(pd.notna(v) for v in (price, ema50, atr_value)):
            return None
        if price <= 0 or atr_value <= 0:
            return None
        return price, ema50, atr_value, bars.index[-1]

    def _clamp_stop(self, price: float, raw_stop: float, atr_value: float) -> tuple[float, bool]:
        """Force the stop strictly below entry.

        Two floors, whichever is further from price: `min_stop_atr_mult` ATR, and
        `min_stop_pct` of price. The percentage floor covers the case where ATR
        has collapsed to near zero in a very quiet stretch, which would otherwise
        produce a stop a fraction of a cent below entry and a position size large
        enough to breach every limit in Phase 5.
        """
        ceiling = min(
            price - self.min_stop_atr_mult * atr_value,
            price * (1.0 - self.min_stop_pct),
        )
        if raw_stop >= ceiling:
            return ceiling, True
        return raw_stop, False


class LowVolBullStrategy(BaseStrategy):
    """Lowest third of regimes by expected volatility.

    Where most of the returns are generated: calm markets plus modest leverage
    compounds. Allocation 95%, leverage 1.25x.

    Stop is `max(price - 3*ATR, EMA50 - 0.5*ATR)`. The max picks whichever is
    *tighter*, so the stop rides up behind a trending price rather than sitting
    three ATR below it forever.
    """

    name = "LowVolBullStrategy"

    def __init__(self, allocation: float = 0.95, leverage: float = 1.25, **kwargs: Any) -> None:
        super().__init__(allocation=allocation, leverage=leverage, **kwargs)

    def compute_allocation(self, bars: pd.DataFrame) -> tuple[float, float, str]:
        return (
            self.allocation,
            self.leverage,
            f"Low volatility regime: fully invested at {self.allocation:.0%} "
            f"with {self.leverage:.2f}x leverage. Calm markets trend up.",
        )

    def compute_raw_stop(self, price: float, ema50: float, atr_value: float) -> float:
        return max(price - 3.0 * atr_value, ema50 - 0.5 * atr_value)


class MidVolCautiousStrategy(BaseStrategy):
    """Middle third by expected volatility. The only trend-sensitive tier.

    Price above the 50 EMA means the trend is intact, so stay at 95%. Below it,
    cut to 60%. Never levered either way: the trend filter is a reason to remain
    invested, not a reason to press.
    """

    name = "MidVolCautiousStrategy"

    def __init__(
        self,
        allocation_trend: float = 0.95,
        allocation_no_trend: float = 0.60,
        leverage: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(allocation=allocation_trend, leverage=leverage, **kwargs)
        self.allocation_trend = allocation_trend
        self.allocation_no_trend = allocation_no_trend

    def has_trend(self, bars: pd.DataFrame) -> bool:
        """Trend filter: is price above the 50 EMA?"""
        ema_series = ema(bars["close"], self.ema_span)
        return bool(bars["close"].iloc[-1] > ema_series.iloc[-1])

    def compute_allocation(self, bars: pd.DataFrame) -> tuple[float, float, str]:
        if self.has_trend(bars):
            return (
                self.allocation_trend,
                self.leverage,
                f"Mid volatility, price above {self.ema_span} EMA: trend intact, "
                f"staying invested at {self.allocation_trend:.0%}.",
            )
        return (
            self.allocation_no_trend,
            self.leverage,
            f"Mid volatility, price below {self.ema_span} EMA: trend broken, "
            f"reducing to {self.allocation_no_trend:.0%}.",
        )

    def compute_raw_stop(self, price: float, ema50: float, atr_value: float) -> float:
        return ema50 - 0.5 * atr_value


class HighVolDefensiveStrategy(BaseStrategy):
    """Top third by expected volatility. Reduced, never short, never flat.

    60% invested rather than 0% because V-shaped recoveries happen fast and the
    HMM is 2-3 days late detecting them. Sitting in cash misses the rebound;
    being short during it wipes out whatever the selloff earned.

    Stop is `EMA50 - 1.0*ATR`, wider than the other tiers, because a normal stop
    in a turbulent regime is just a slower way of selling the bottom.
    """

    name = "HighVolDefensiveStrategy"

    def __init__(self, allocation: float = 0.60, leverage: float = 1.0, **kwargs: Any) -> None:
        super().__init__(allocation=allocation, leverage=leverage, **kwargs)

    def compute_allocation(self, bars: pd.DataFrame) -> tuple[float, float, str]:
        return (
            self.allocation,
            self.leverage,
            f"High volatility regime: reduced to {self.allocation:.0%}, unlevered. "
            f"Staying partially invested to catch V-shaped rebounds.",
        )

    def compute_raw_stop(self, price: float, ema50: float, atr_value: float) -> float:
        return ema50 - 1.0 * atr_value


# ---------------------------------------------------------------------------
# Volatility rank -> strategy tier
# ---------------------------------------------------------------------------

#: Strategy class per volatility tier. The orchestrator's whole mapping table.
STRATEGY_BY_VOL_RANK: dict[VolatilityRank, type[BaseStrategy]] = {
    VolatilityRank.LOW: LowVolBullStrategy,
    VolatilityRank.MID: MidVolCautiousStrategy,
    VolatilityRank.HIGH: HighVolDefensiveStrategy,
}


# Backward-compatible aliases. The earlier design named strategies after regime
# labels, which was the mistake this phase corrects: a label describes return,
# and allocation keys off volatility. They are kept so older references resolve,
# but they all point at the three volatility tiers.
#
# Note what the aliases reveal: CrashDefensiveStrategy and EuphoriaCautiousStrategy
# are opposite labels, and euphoria maps to the *low* volatility tier only when a
# euphoric regime happens to be calm. That is exactly why LABEL_TO_STRATEGY below
# is a fallback and not the primary path.
CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
StrongBearStrategy = HighVolDefensiveStrategy
WeakBearStrategy = MidVolCautiousStrategy
MeanReversionStrategy = MidVolCautiousStrategy
NeutralStrategy = MidVolCautiousStrategy
WeakBullStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
StrongBullStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy

#: Label -> strategy, covering every label across 3-7 regime counts.
#:
#: FALLBACK ONLY. The orchestrator never consults this: it maps by measured
#: expected_volatility, because a "bull" regime is not necessarily calm. This
#: exists for callers holding a bare label with no RegimeInfo to measure.
LABEL_TO_STRATEGY: dict[Regime, type[BaseStrategy]] = {
    Regime.CRASH: HighVolDefensiveStrategy,
    Regime.STRONG_BEAR: HighVolDefensiveStrategy,
    Regime.BEAR: HighVolDefensiveStrategy,
    Regime.WEAK_BEAR: MidVolCautiousStrategy,
    Regime.NEUTRAL: MidVolCautiousStrategy,
    Regime.WEAK_BULL: MidVolCautiousStrategy,
    Regime.BULL: LowVolBullStrategy,
    Regime.STRONG_BULL: LowVolBullStrategy,
    Regime.EUPHORIA: LowVolBullStrategy,
    Regime.UNKNOWN: HighVolDefensiveStrategy,   # unknown is treated as dangerous
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class StrategyOrchestrator:
    """Maps regimes to strategies by volatility, and produces signals.

    Construction sorts `regime_infos` by `expected_volatility` ascending and
    assigns each a tier. That sort is independent of the HMM's label sort, which
    is by return. A regime labelled "bull" can land in the high-volatility tier
    and will be allocated 60%, which is the correct behaviour and looks wrong on
    a dashboard until you know why.

    `update_regime_infos` rebuilds the mapping after an HMM retrain. This is not
    optional: EM renumbers its states on every refit, so a mapping built against
    the previous fit points at the wrong regimes entirely.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        regime_infos: dict[int, RegimeInfo] | None = None,
    ) -> None:
        self.config = dict(config or {})
        self.min_confidence: float = self.config.get("min_confidence", 0.55)
        self.rebalance_threshold: float = self.config.get("rebalance_threshold", 0.10)
        self.uncertainty_size_mult: float = self.config.get("uncertainty_size_mult", 0.50)
        self.max_leverage: float = self.config.get("max_leverage", 1.25)

        self._strategy_kwargs = {
            "atr_window": self.config.get("atr_window", 14),
            "ema_span": self.config.get("ema_span", 50),
            "min_stop_atr_mult": self.config.get("min_stop_atr_mult", 0.5),
            "min_stop_pct": self.config.get("min_stop_pct", 0.005),
        }

        self.regime_infos: dict[int, RegimeInfo] = {}
        self.vol_ranks: dict[int, VolatilityRank] = {}
        self.strategies: dict[int, BaseStrategy] = {}
        if regime_infos:
            self.update_regime_infos(regime_infos)

    # -- mapping ------------------------------------------------------------

    def update_regime_infos(self, regime_infos: dict[int, RegimeInfo]) -> None:
        """Rebuild regime -> strategy after a fit or refit.

        Ranks are recomputed from `expected_volatility` here rather than read off
        `RegimeInfo.volatility_rank`, even though the HMM populates that field
        with the same function. Recomputing means the orchestrator stays correct
        if it is ever handed infos from another source, and the shared
        `assign_volatility_ranks` guarantees the two agree.
        """
        if not regime_infos:
            raise ValueError("regime_infos is empty: cannot map strategies")

        self.regime_infos = dict(regime_infos)
        self.vol_ranks = assign_volatility_ranks(
            {sid: info.expected_volatility for sid, info in regime_infos.items()}
        )
        self.strategies = {
            sid: self._build_strategy(rank) for sid, rank in self.vol_ranks.items()
        }

        for sid in sorted(self.vol_ranks, key=lambda s: regime_infos[s].expected_volatility):
            info = regime_infos[sid]
            logger.info(
                "regime %d (%-12s ann.vol %6.2f%%) -> %s [%s]",
                sid, info.regime_name, info.expected_volatility * 100,
                self.strategies[sid].name, self.vol_ranks[sid].value,
            )

    def _build_strategy(self, rank: VolatilityRank) -> BaseStrategy:
        cls = STRATEGY_BY_VOL_RANK[rank]
        if cls is LowVolBullStrategy:
            return cls(
                allocation=self.config.get("low_vol_allocation", 0.95),
                leverage=min(self.config.get("low_vol_leverage", 1.25), self.max_leverage),
                **self._strategy_kwargs,
            )
        if cls is MidVolCautiousStrategy:
            return cls(
                allocation_trend=self.config.get("mid_vol_allocation_trend", 0.95),
                allocation_no_trend=self.config.get("mid_vol_allocation_no_trend", 0.60),
                **self._strategy_kwargs,
            )
        return cls(
            allocation=self.config.get("high_vol_allocation", 0.60),
            **self._strategy_kwargs,
        )

    def get_strategy(self, regime_id: int) -> BaseStrategy:
        if regime_id not in self.strategies:
            raise KeyError(
                f"no strategy for regime {regime_id}. Known: {sorted(self.strategies)}. "
                f"Call update_regime_infos() after every HMM refit."
            )
        return self.strategies[regime_id]

    def get_volatility_rank(self, regime_id: int) -> VolatilityRank:
        return self.vol_ranks[regime_id]

    # -- uncertainty --------------------------------------------------------

    def is_uncertain(self, regime_state: RegimeState, is_flickering: bool | None = None) -> bool:
        """Uncertainty triggers on any of three conditions.

        - probability below `min_confidence`: the model is unsure which regime
        - flickering: the model keeps changing its mind
        - unconfirmed: a regime change has not yet held `stability_bars`

        Any one is enough. All three mean the same thing operationally, which is
        that the regime input should not be trusted at full size.
        """
        flickering = regime_state.is_flickering if is_flickering is None else is_flickering
        return (
            regime_state.probability < self.min_confidence
            or bool(flickering)
            or not regime_state.is_confirmed
        )

    def apply_uncertainty(self, signal: Signal, reason: str) -> Signal:
        """Halve the position size and force leverage to 1.0x.

        Leverage is forced rather than scaled: the point of uncertainty mode is
        that the regime call is unreliable, and leverage on an unreliable call is
        the specific thing that turns a bad week into an unrecoverable one.
        """
        from dataclasses import replace

        return replace(
            signal,
            position_size_pct=signal.position_size_pct * self.uncertainty_size_mult,
            leverage=1.0,
            reasoning=f"{signal.reasoning} {UNCERTAINTY_TAG} ({reason})",
            metadata={
                **signal.metadata,
                "uncertainty": True,
                "uncertainty_reason": reason,
                "pre_uncertainty_size": signal.position_size_pct,
                "pre_uncertainty_leverage": signal.leverage,
            },
        )

    def _uncertainty_reason(self, regime_state: RegimeState, is_flickering: bool | None) -> str:
        flickering = regime_state.is_flickering if is_flickering is None else is_flickering
        reasons = []
        if regime_state.probability < self.min_confidence:
            reasons.append(f"confidence {regime_state.probability:.2f} < {self.min_confidence:.2f}")
        if flickering:
            reasons.append("regime flickering")
        if not regime_state.is_confirmed:
            reasons.append("regime change unconfirmed")
        return "; ".join(reasons)

    # -- signals ------------------------------------------------------------

    def generate_signals(
        self,
        symbols: list[str],
        bars: dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool | None = None,
    ) -> list[Signal]:
        """One signal per tradable symbol for the current bar.

        Every signal carries the same portfolio-level `position_size_pct`,
        because the regime determines how invested the portfolio should be, not
        how invested one symbol should be. The per-symbol share is written into
        `metadata["per_symbol_weight"]` once the tradable count is known.
        """
        strategy = self.get_strategy(regime_state.state_id)
        uncertain = self.is_uncertain(regime_state, is_flickering)
        reason = self._uncertainty_reason(regime_state, is_flickering) if uncertain else ""
        rank = self.vol_ranks[regime_state.state_id]

        signals: list[Signal] = []
        for symbol in symbols:
            symbol_bars = bars.get(symbol)
            if symbol_bars is None or symbol_bars.empty:
                logger.debug("%s: no bars, skipping", symbol)
                continue

            signal = strategy.generate_signal(symbol, symbol_bars, regime_state)
            if signal is None:
                continue
            if uncertain:
                signal = self.apply_uncertainty(signal, reason)
            signals.append(signal)

        if not signals:
            return []

        per_symbol = signals[0].position_size_pct / len(signals)
        from dataclasses import replace

        return [
            replace(
                s,
                metadata={
                    **s.metadata,
                    "volatility_rank": rank.value,
                    "per_symbol_weight": per_symbol,
                    "n_symbols": len(signals),
                    "gross_exposure": s.position_size_pct * s.leverage,
                },
            )
            for s in signals
        ]

    def target_allocation(
        self,
        regime_state: RegimeState,
        price: float,
        ema50: float,
        is_flickering: bool | None = None,
    ) -> float:
        """Gross allocation target: allocation x leverage. No Signal built.

        The backtester needs only this number, and building a full `Signal` per
        bar means recomputing ATR and EMA over a growing slice, which is O(T)
        per bar and O(T^2) over a walk-forward. The spec is explicit that the
        backtester carries no per-trade stops, so the ATR work is pure waste
        there.

        Deliberately shares the allocation rules with `generate_signal` rather
        than restating them: two copies would eventually disagree, and the
        backtest would measure a strategy the live loop does not run.

        Returns above 1.0 when the low-vol tier applies its 1.25x. That is
        margin, and the backtester's allocation math handles it.
        """
        strategy = self.get_strategy(regime_state.state_id)

        if isinstance(strategy, MidVolCautiousStrategy):
            allocation = (
                strategy.allocation_trend if price > ema50 else strategy.allocation_no_trend
            )
        else:
            allocation = strategy.allocation
        leverage = strategy.leverage

        if self.is_uncertain(regime_state, is_flickering):
            allocation *= self.uncertainty_size_mult
            leverage = 1.0

        return allocation * leverage

    # -- rebalancing --------------------------------------------------------

    def needs_rebalance(self, target_allocation: float, current_allocation: float) -> bool:
        """True only when the gap exceeds `rebalance_threshold`.

        Without this the system trades on every minor probability wobble and pays
        slippage for a 1% drift that changes nothing. Fewer trades is less
        slippage is better real-world performance, and slippage is the entire
        cost model in Phase 4 since Alpaca charges no commission.

        Compared in absolute allocation terms: a move from 0.95 to 0.60 is a gap
        of 0.35 and rebalances; 0.95 to 0.90 is 0.05 and does not.
        """
        return abs(target_allocation - current_allocation) > self.rebalance_threshold


# ---------------------------------------------------------------------------
# Backward-compatible wrapper
# ---------------------------------------------------------------------------

class RegimeStrategies:
    """Thin wrapper preserving the Phase 1 skeleton's API.

    Kept so anything written against the original `compute_allocation` /
    `needs_rebalance` surface keeps working. New code should use
    `StrategyOrchestrator` directly, which carries the full signal output.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        regime_infos: dict[int, RegimeInfo] | None = None,
    ) -> None:
        self.orchestrator = StrategyOrchestrator(config, regime_infos)

    def compute_allocation(
        self,
        regime_state: RegimeState,
        bars: pd.DataFrame,
        is_flickering: bool | None = None,
    ) -> tuple[float, float]:
        """Return (allocation, leverage) for the current regime."""
        strategy = self.orchestrator.get_strategy(regime_state.state_id)
        allocation, leverage, _ = strategy.compute_allocation(bars)
        if self.orchestrator.is_uncertain(regime_state, is_flickering):
            allocation *= self.orchestrator.uncertainty_size_mult
            leverage = 1.0
        return allocation, leverage

    def needs_rebalance(self, target: float, current: float) -> bool:
        return self.orchestrator.needs_rebalance(target, current)

    def update_regime_infos(self, regime_infos: dict[int, RegimeInfo]) -> None:
        self.orchestrator.update_regime_infos(regime_infos)
