"""
Watchlist scan: what the system would actually trade right now, ranked.

## What this is, and what it is not

This is **not a stock picker**, and presenting it as one would misrepresent the
model. The HMM is fitted on a single reference symbol and classifies the
*market's* regime; that regime is the same for every name in the universe. The
system decides **how invested to be**, not **what to own**.

So the differentiation between symbols here comes from the parts that are
genuinely per-symbol:

- **trend**: price above or below its own 50 EMA, which the mid-volatility
  strategy reads directly
- **stop distance**: each symbol's own ATR sets where its stop sits, and sizing
  divides by that distance, so a tighter stop earns a larger position
- **the risk cascade**: correlation against what is already held, sector
  concentration, single-position cap, exposure headroom, the gap rule

That last group is where most of the ranking actually comes from, and it is a
real answer to "what should I buy": it is the set of names the risk layer would
let you buy today, ordered by how much capital it would commit to each.

## Ranking

Approved candidates first, then by `conviction`, with `notional` as the tiebreak.

Ordering by notional alone was the first attempt and it was wrong. Sizing is
`risk_budget / stop_distance`, so notional does encode risk-per-share — but the
15% single-position cap truncates it, and once two names are both capped their
notionals differ by rounding. On the first live scan that put a symbol trading
*below* its 50 EMA ahead of one trading above it, on a $152 difference. Ranking
has to keep discriminating exactly when several names are worth buying.

`conviction` is a transparent 0-1 blend of trend, stop quality in ATR terms, how
much of the requested size survived the risk cascade, and 20-bar momentum. It is
not a model output and not calibrated against anything. Every input is published
beside it so the ranking can be checked rather than believed.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Weights for the published conviction blend. Deliberately few and flat: a
#: finely tuned weighting would imply a precision this system has not earned.
W_TREND = 0.35        # price above its own 50 EMA
W_STOP = 0.30         # tighter stop in ATR terms is a better risk/reward setup
W_SIZE = 0.20         # how much of the requested size survived the risk cascade
W_MOMENTUM = 0.15     # 20-bar return, as a tiebreak rather than a thesis


@dataclass
class Candidate:
    """One symbol's full signal-to-decision chain, ready to publish."""
    symbol: str
    rank: int = 0
    approved: bool = False
    action: str = "blocked"           # buy | hold | blocked
    conviction: float = 0.0

    shares: float = 0.0
    notional: float = 0.0
    entry_price: float = 0.0
    stop_loss: float | None = None
    stop_distance_pct: float = 0.0
    stop_atr_mult: float = 0.0
    risk_dollars: float = 0.0
    risk_pct_of_equity: float = 0.0

    trend: str = "unknown"            # above | below
    price_vs_ema50: float = 0.0
    atr_pct: float = 0.0
    return_20d: float = 0.0

    strategy: str = ""
    regime: str = ""
    regime_confidence: float = 0.0
    volatility_rank: str = ""
    held: bool = False
    held_quantity: float = 0.0

    rejection_reason: str | None = None
    reason: str = ""
    modifications: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _series_context(bars: pd.DataFrame, ema_span: int, atr_window: int) -> dict[str, float]:
    """Per-symbol trend and volatility context, computed causally."""
    from data.feature_engineering import atr, ema

    close = bars["close"]
    price = float(close.iloc[-1])
    ema50 = float(ema(close, ema_span).iloc[-1])
    atr_value = float(atr(bars["high"], bars["low"], close, atr_window).iloc[-1])

    lookback = min(20, len(close) - 1)
    momentum = float(close.iloc[-1] / close.iloc[-1 - lookback] - 1) if lookback > 0 else 0.0

    return {
        "price": price,
        "ema50": ema50,
        "price_vs_ema50": (price / ema50 - 1) if ema50 else 0.0,
        "atr_pct": (atr_value / price) if price else 0.0,
        "return_20d": momentum,
    }


def _conviction(candidate: Candidate, requested_shares: float) -> float:
    """Transparent 0-1 blend of the components published beside it.

    Not a model output and not calibrated against anything. It exists to give
    the ranking a readable bar, and every input sits next to it in the payload
    so it can be checked rather than believed.
    """
    trend = 1.0 if candidate.trend == "above" else 0.0

    # A stop 1 ATR away scores 1.0 and decays from there. Wider stops in ATR
    # terms are worse risk/reward per unit of the symbol's own volatility.
    stop = 0.0
    if candidate.stop_atr_mult > 0:
        stop = max(0.0, min(1.0, 1.0 / max(1.0, candidate.stop_atr_mult)))

    # How much of the requested size survived the cascade. Heavy shrinkage means
    # the risk layer had objections, which is information.
    size = 0.0
    if requested_shares > 0:
        size = max(0.0, min(1.0, candidate.shares / requested_shares))

    momentum = max(0.0, min(1.0, (candidate.return_20d + 0.10) / 0.20))

    return round(W_TREND * trend + W_STOP * stop + W_SIZE * size + W_MOMENTUM * momentum, 4)


def scan(
    orchestrator,
    risk_manager,
    symbols: list[str],
    bars: dict[str, pd.DataFrame],
    regime_state,
    portfolio,
    strategy_config: dict[str, Any],
    quotes: dict[str, dict[str, float]] | None = None,
) -> list[Candidate]:
    """Run every symbol through the full pipeline without placing anything.

    Uses `record=False` so the scan cannot trip the duplicate-order window and
    change the answer for the real signal a moment later.
    """
    ema_span = int(strategy_config.get("ema_span", 50))
    atr_window = int(strategy_config.get("atr_window", 14))
    quotes = quotes or {}

    signals = orchestrator.generate_signals(symbols, bars, regime_state)
    by_symbol = {s.symbol: s for s in signals}

    try:
        vol_rank = orchestrator.get_volatility_rank(regime_state.state_id).value
    except Exception:
        vol_rank = ""

    out: list[Candidate] = []
    for symbol in symbols:
        frame = bars.get(symbol)
        if frame is None or frame.empty:
            continue

        held = portfolio.positions.get(symbol) if portfolio else None
        candidate = Candidate(
            symbol=symbol,
            regime=regime_state.label.value,
            regime_confidence=round(regime_state.probability, 4),
            volatility_rank=vol_rank,
            held=held is not None,
            held_quantity=float(held.get("quantity", 0.0)) if held else 0.0,
        )

        try:
            context = _series_context(frame, ema_span, atr_window)
        except Exception as exc:
            logger.debug("%s: no series context (%s)", symbol, exc)
            context = {}

        candidate.price_vs_ema50 = round(context.get("price_vs_ema50", 0.0), 6)
        candidate.trend = "above" if candidate.price_vs_ema50 > 0 else "below"
        candidate.atr_pct = round(context.get("atr_pct", 0.0), 6)
        candidate.return_20d = round(context.get("return_20d", 0.0), 6)

        signal = by_symbol.get(symbol)
        if signal is None:
            candidate.reason = "no signal: not enough history for this symbol"
            out.append(candidate)
            continue

        candidate.entry_price = round(signal.entry_price, 4)
        candidate.stop_loss = round(signal.stop_loss, 4)
        candidate.strategy = signal.strategy_name
        candidate.reasoning = signal.reasoning
        candidate.stop_distance_pct = round(
            abs(signal.entry_price - signal.stop_loss) / signal.entry_price, 6
        ) if signal.entry_price else 0.0
        atr_value = candidate.atr_pct * signal.entry_price
        candidate.stop_atr_mult = round(
            abs(signal.entry_price - signal.stop_loss) / atr_value, 3
        ) if atr_value > 0 else 0.0

        requested = risk_manager.position_size(
            portfolio.equity, signal.entry_price, signal.stop_loss
        ) if portfolio else 0

        decision = risk_manager.validate_signal(
            signal, portfolio, quote=quotes.get(symbol), overnight=True, record=False
        )

        candidate.approved = decision.approved
        candidate.reason = decision.reason
        candidate.modifications = list(decision.modifications)
        candidate.rejection_reason = (
            decision.rejection_reason.value if decision.rejection_reason else None
        )

        if decision.approved:
            modified = decision.modified_signal
            candidate.shares = float(modified.get("shares", 0))
            candidate.notional = round(float(modified.get("notional", 0.0)), 2)
            candidate.risk_dollars = round(float(modified.get("risk_dollars", 0.0)), 2)
            candidate.risk_pct_of_equity = round(float(modified.get("risk_pct_of_equity", 0.0)), 6)
            candidate.action = "hold" if candidate.held else "buy"
        else:
            candidate.action = "blocked"

        candidate.conviction = _conviction(candidate, requested)
        out.append(candidate)

    # Approved first, then conviction, then the dollars committed as tiebreak.
    # Blocked names keep their conviction so the table can still show why a
    # promising setup was refused, which is usually the more useful row.
    out.sort(key=lambda c: (c.approved, c.conviction, c.notional), reverse=True)
    for index, candidate in enumerate(out, start=1):
        candidate.rank = index
    return out


def top_pick(candidates: list[Candidate]) -> Candidate | None:
    """The single name the system would commit the most capital to, or None.

    None when nothing is approved, which is a real and common answer: a halted
    breaker, a regime the strategy sits out, or an exposure cap already reached
    all produce it. Showing the least-bad blocked name instead would invent a
    recommendation the system did not make.
    """
    approved = [c for c in candidates if c.approved and c.action == "buy"]
    return approved[0] if approved else None
