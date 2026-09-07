"""
Publish a dashboard snapshot as JSON for the web UI.

The web dashboard in `dashboard/` is a static Next.js app. It has no connection
to the trading process and no Alpaca credentials, by design:

- the engine runs on your machine, on your schedule, with your keys
- the dashboard is a **read-only view of a published snapshot**

That split is deliberate. Putting Alpaca credentials into a Vercel deployment
would mean a public URL that can read your account, and a dashboard that holds
live broker handles is one bad deploy away from being the reason the trading
process died. A JSON file has neither failure mode.

    python main.py --publish        # writes dashboard/public/data/state.json
    cd dashboard && npm run dev

The published file contains no credentials, no order ids and no account number.
It contains what the terminal dashboard already prints on your screen.

## Demo data

`demo_snapshot()` exists so a fresh clone and the deployed URL show a working
interface rather than six empty panels. Everything it produces is labelled
`"source": "demo"`, and the UI renders a banner saying so. Fabricated numbers
presented as real account data would be worse than an empty page.
"""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "dashboard" / "public" / "data" / "state.json"

#: Bumped when the JSON shape changes in a way the UI must handle.
SCHEMA_VERSION = 1

#: Never publish these, whatever a future panel adds to the snapshot.
REDACTED_KEYS = {
    "api_key", "secret_key", "account_number", "order_id", "stop_order_id",
    "trade_id", "client_order_id", "lock_file", "traceback",
}


def _clean(value: Any) -> Any:
    """Recursively drop redacted keys and make everything JSON-safe.

    A denylist rather than an allowlist would be the wrong default here, but the
    snapshot is assembled from typed panels whose shape is known, so the risk is
    a future field rather than an unknown one. The publisher also runs through
    `_jsonable`, so an unexpected object becomes a string rather than an error.
    """
    from monitoring.logger import _jsonable

    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in REDACTED_KEYS}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return _jsonable(value)


def regime_mix(regime_history: list) -> list[dict[str, Any]]:
    """Share of bars spent in each regime, most frequent first.

    Computed here rather than in the UI because it is a statement about the
    model, not a presentation detail. A system reporting seven regimes that sat
    in one of them for 90% of the window has not really found seven, and that is
    worth seeing on the dashboard rather than only in a fit log.
    """
    counts: dict[str, int] = {}
    for point in regime_history or []:
        label = (point or {}).get("regime")
        if label and label != "unknown":
            counts[label] = counts.get(label, 0) + 1

    total = sum(counts.values())
    if not total:
        return []
    return [
        {"regime": label, "bars": bars, "pct": bars / total}
        for label, bars in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]


def freshness(snapshot: dict[str, Any]) -> dict[str, Any]:
    """How old the data actually is, published so the UI cannot imply live prices.

    This system is not real time and should never look like it. Three separate
    delays stack up:

    1. **Bar interval.** Daily bars. A bar is only final at the session close,
       so intraday the newest complete bar is yesterday's.
    2. **Feed delay.** Alpaca's free tier will not serve the most recent 15
       minutes of SIP data, so requests deliberately stop 16 minutes short.
    3. **Publish cadence.** The engine writes this file once per processed bar.
       Between bars the file does not change no matter how often it is polled.

    A dashboard that polls every 5 seconds looks live. Publishing these numbers
    is what stops that impression from being a lie.
    """
    regime = snapshot.get("regime", {}) or {}
    system = snapshot.get("system", {}) or {}
    bar_time = regime.get("timestamp")

    age_hours = None
    if bar_time:
        try:
            stamp = datetime.fromisoformat(str(bar_time).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            age_hours = round((datetime.now(UTC) - stamp).total_seconds() / 3600, 2)
        except ValueError:
            pass

    return {
        "bar_timestamp": bar_time,
        "bar_age_hours": age_hours,
        "timeframe": system.get("timeframe"),
        "sip_delay_minutes": 16,
        "publish_cadence": "once per processed bar",
        "poll_seconds": 5,
        "realtime": False,
    }


def timing(snapshot: dict[str, Any], engine=None) -> dict[str, Any]:
    """When a signal would actually be acted on.

    "Buy this" is incomplete without "at what price, in which session, and by
    what order type". On a daily-bar system the answer is never "right now":
    the bar the decision is based on closes at 4pm New York, and the order rests
    until the next session opens.
    """
    system = snapshot.get("system", {}) or {}
    return {
        "market_open": system.get("market_open"),
        "next_open": getattr(engine, "next_open", None) if engine else None,
        "timeframe": system.get("timeframe"),
        "acts_on": "the next session open",
        "order_type": "limit",
        "limit_offset_pct": 0.001,
        "session_close_et": "16:00",
    }


def build_payload(snapshot: dict[str, Any], *, source: str = "live",
                  equity_history: list | None = None,
                  regime_history: list | None = None,
                  notes: dict[str, Any] | None = None,
                  engine=None) -> dict[str, Any]:
    """Wrap a `DashboardState.snapshot()` in the envelope the UI expects."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "published_at": datetime.now(UTC).isoformat(),
        **_clean(snapshot),
        "equity_history": _clean(equity_history or []),
        "regime_history": _clean(regime_history or []),
        "regime_mix": regime_mix(regime_history or []),
        "freshness": _clean(freshness(snapshot)),
        "timing": _clean(timing(snapshot, engine)),
        "notes": _clean(notes or {}),
    }
    return payload


def publish(snapshot: dict[str, Any], path: Path = DEFAULT_OUTPUT, *,
            source: str = "live", equity_history: list | None = None,
            regime_history: list | None = None,
            notes: dict[str, Any] | None = None, engine=None) -> Path:
    """Write the snapshot to `path`. Atomic, so the UI never reads half a file."""
    payload = build_payload(
        snapshot, source=source, equity_history=equity_history,
        regime_history=regime_history, notes=notes, engine=engine,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with open(temp, "w") as fh:
        json.dump(payload, fh, indent=2)
    temp.replace(path)
    return path


def publish_from_engine(engine, path: Path = DEFAULT_OUTPUT) -> Path:
    """Publish straight from a started `TradingEngine`."""
    notes = {
        "verdict": "no demonstrated edge out-of-sample",
        "paper": bool(getattr(engine, "is_paper", True)),
        "mode": getattr(engine, "mode", "unknown"),
    }
    return publish(
        engine.dashboard_state.snapshot(),
        path,
        source="live",
        engine=engine,
        equity_history=list(getattr(engine.session, "equity_history", []) or []),
        regime_history=list(getattr(engine.session, "regime_history", []) or []),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def _demo_candidate(symbol, rank, approved, action, conviction, shares, notional,
                    entry, stop, trend, price_vs_ema50, atr_pct, ret20, strategy,
                    modifications=None, rejection_reason=None, reason="approved unmodified",
                    held=False, held_quantity=0.0) -> dict[str, Any]:
    """One demo candidate, matching `core.candidates.Candidate` field for field.

    Written out longhand rather than generated so the demo shows the full range
    of outcomes the real scan produces: approved, approved-but-shrunk, held, and
    three different rejection reasons. A demo where everything is approved would
    hide the half of the interface that matters.
    """
    stop_distance = abs(entry - stop) / entry if entry else 0.0
    risk_dollars = shares * abs(entry - stop)
    return {
        "symbol": symbol, "rank": rank, "approved": approved, "action": action,
        "conviction": conviction, "shares": shares, "notional": notional,
        "entry_price": entry, "stop_loss": stop,
        "stop_distance_pct": round(stop_distance, 6),
        "stop_atr_mult": round(stop_distance / atr_pct, 3) if atr_pct else 0.0,
        "risk_dollars": round(risk_dollars, 2),
        "risk_pct_of_equity": round(risk_dollars / 12_565, 6),
        "trend": trend, "price_vs_ema50": price_vs_ema50, "atr_pct": atr_pct,
        "return_20d": ret20, "strategy": strategy,
        "regime": "strong_bull", "regime_confidence": 0.72, "volatility_rank": "low",
        "held": held, "held_quantity": held_quantity,
        "rejection_reason": rejection_reason, "reason": reason,
        "modifications": modifications or [],
        "reasoning": f"{strategy}: regime strong_bull at 72% confidence, "
                     f"{'above' if trend == 'above' else 'below'} the 50 EMA",
    }


def demo_snapshot(seed: int = 7) -> dict[str, Any]:
    """A plausible but explicitly fake snapshot, for a fresh clone.

    Marked `source: "demo"` and rendered behind a banner in the UI. The numbers
    are shaped like real ones so the layout can be judged, and are not real.
    """
    rng = random.Random(seed)
    now = datetime.now(UTC)

    equity, peak = 10_000.0, 10_000.0
    history, regimes = [], []
    labels = ["neutral", "weak_bull", "strong_bull", "weak_bear", "strong_bear"]
    label = "neutral"

    for i in range(180):
        when = now - timedelta(days=180 - i)
        if i % 23 == 0:
            label = rng.choice(labels)
        drift = {"strong_bull": 0.0016, "weak_bull": 0.0007, "neutral": 0.0002,
                 "weak_bear": -0.0006, "strong_bear": -0.0018}[label]
        vol = {"strong_bull": 0.006, "weak_bull": 0.008, "neutral": 0.009,
               "weak_bear": 0.013, "strong_bear": 0.021}[label]
        equity *= math.exp(rng.gauss(drift, vol))
        peak = max(peak, equity)
        history.append({"t": when.date().isoformat(), "equity": round(equity, 2),
                        "peak": round(peak, 2)})
        regimes.append({"t": when.date().isoformat(), "regime": label})

    day_start = history[-2]["equity"]
    daily_pnl = equity - day_start

    positions = [
        {"symbol": "SPY", "direction": "LONG", "quantity": 2, "entry_price": 514.20,
         "current_price": 520.30, "market_value": 1040.60, "stop_loss": 508.00,
         "has_stop": True, "unrealised_pnl": 12.20, "unrealised_pnl_pct": 0.0119,
         "distance_to_stop_pct": 0.0236, "regime_at_entry": "strong_bull",
         "regime_current": "strong_bull", "regime_changed": False,
         "holding_periods": 3, "held_for": "3h", "adopted": False},
        {"symbol": "QQQ", "direction": "LONG", "quantity": 2, "entry_price": 719.10,
         "current_price": 713.15, "market_value": 1426.30, "stop_loss": 705.78,
         "has_stop": True, "unrealised_pnl": -11.90, "unrealised_pnl_pct": -0.0083,
         "distance_to_stop_pct": 0.0195, "regime_at_entry": "weak_bull",
         "regime_current": "strong_bull", "regime_changed": True,
         "holding_periods": 9, "held_for": "2d", "adopted": False},
    ]

    signals = [
        {"timestamp": (now - timedelta(minutes=m)).isoformat(), "event": event,
         "symbol": symbol, "shares": shares, "notional": notional,
         "signal_regime": regime, "message": message,
         "rejection_reason": rejection}
        for m, event, symbol, shares, notional, regime, message, rejection in [
            (12, "signal_generated", "COIN", 3, 554.19, "strong_bull",
             "COIN: approved 3 shares ($554.19)", None),
            (12, "signal_rejected", "AAPL", 0, 0.0, "strong_bull",
             "AAPL: rejected, correlation 0.89 with SPY", "correlation_too_high"),
            (73, "signal_generated", "AAPL", 4, 1280.44, "weak_bull",
             "AAPL: approved 4 shares ($1,280.44)", None),
            (1502, "signal_rejected", "TSLA", 0, 0.0, "weak_bull",
             "TSLA: rejected, $92 is below the $100 minimum", "below_minimum_size"),
        ]
    ]

    candidates = [
        _demo_candidate("COIN", 1, True, "buy", 0.81, 3, 554.19, 184.73, 166.30,
                        "above", 0.038, 0.0412, 0.094, "LowVolBullStrategy",
                        modifications=[]),
        _demo_candidate("AAPL", 2, True, "buy", 0.74, 4, 1280.44, 320.11, 308.71,
                        "above", 0.021, 0.0118, 0.043, "LowVolBullStrategy",
                        modifications=[]),
        _demo_candidate("SMCI", 3, True, "buy", 0.73, 9, 356.13, 39.57, 32.82,
                        "above", 0.029, 0.0451, 0.077, "LowVolBullStrategy",
                        modifications=["gap cap: 3x stop gap-through kept under 2% of portfolio"]),
        _demo_candidate("QQQ", 4, True, "hold", 0.69, 2, 1438.20, 719.10, 705.78,
                        "above", 0.009, 0.0104, 0.018, "LowVolBullStrategy",
                        modifications=[], held=True, held_quantity=2),
        _demo_candidate("NVDA", 5, False, "blocked", 0.61, 0, 0.0, 230.36, 206.43,
                        "above", 0.034, 0.0295, 0.112, "LowVolBullStrategy",
                        rejection_reason="correlation_too_high",
                        reason="correlation 0.89 with SMCI, above the 0.85 reject threshold"),
        _demo_candidate("AVGO", 6, False, "blocked", 0.44, 0, 0.0, 358.02, 344.19,
                        "below", -0.008, 0.0131, -0.014, "LowVolBullStrategy",
                        rejection_reason="sector_limit",
                        reason="semiconductors already at 31% of equity, cap is 30%"),
        _demo_candidate("TSLA", 7, False, "blocked", 0.29, 0, 0.0, 354.10, 333.38,
                        "below", -0.041, 0.0384, -0.087, "LowVolBullStrategy",
                        rejection_reason="below_minimum_size",
                        reason="$71 is below the $100 minimum after all reductions")
    ]

    return {
        "timestamp": now.isoformat(),
        "candidates": candidates,
        "regime": {
            "regime": "strong_bull", "confidence": 0.72, "confirmed": True,
            "consecutive_bars": 14, "flicker_rate": 1, "flicker_window": 20,
            "flicker_threshold": 4, "is_flickering": False, "size_multiplier": 1.0,
            "volatility_rank": "low", "raw_regime": "strong_bull", "n_regimes": 7,
            "model_age_days": 2.1, "timestamp": now.isoformat(),
            "model_trained": (now - timedelta(days=2.1)).isoformat(),
        },
        "portfolio": {
            "equity": round(equity, 2), "cash": round(equity * 0.36, 2),
            "buying_power": round(equity * 2.4, 2), "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": daily_pnl / day_start, "allocation": 0.50,
            "target_allocation": 0.95, "leverage": 0.50, "gross_exposure": 0.50,
            "n_positions": len(positions), "unrealised_pnl": 0.30,
            "peak_equity": round(peak, 2), "day_start_equity": round(day_start, 2),
            "daily_trades": 2,
        },
        "positions": positions,
        "signals": signals,
        "risk": {
            "halted": False, "daily_tripped": "none", "weekly_tripped": "none",
            "peak_tripped": False, "size_multiplier": 1.0, "n_triggers": 0,
            "breaker_now": "none",
            "drawdowns": {"daily": daily_pnl / day_start if daily_pnl < 0 else 0.0,
                          "weekly": -0.008,
                          "from_peak": round(equity / peak - 1, 4)},
            "limits": {"daily_reduce": 0.02, "daily_halt": 0.03, "weekly_reduce": 0.05,
                       "weekly_halt": 0.07, "max_from_peak": 0.10, "max_exposure": 0.80,
                       "max_leverage": 1.25, "max_risk_per_trade": 0.01},
        },
        "timing": {
            "market_open": True,
            "next_open": (now + timedelta(hours=17)).isoformat(),
            "timeframe": "1Day", "acts_on": "the next session open",
            "order_type": "limit", "limit_offset_pct": 0.001,
            "session_close_et": "16:00",
        },
        "system": {
            "data_feed_healthy": True, "broker_connected": True, "api_latency_ms": 23.0,
            "model_age_days": 2.1, "paper": True, "mode": "demo", "market_open": True,
            "bars_processed": 180, "consecutive_errors": 0,
            "started_at": (now - timedelta(hours=6)).isoformat(),
            "symbols": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"], "timeframe": "1Day",
        },
        "equity_history": history,
        "regime_history": regimes,
    }


def publish_demo(path: Path = DEFAULT_OUTPUT) -> Path:
    snapshot = demo_snapshot()
    return publish(
        snapshot, path, source="demo",
        equity_history=snapshot.pop("equity_history"),
        regime_history=snapshot.pop("regime_history"),
        notes={
            "banner": "Demo data. Not a real account.",
            "verdict": "no demonstrated edge out-of-sample",
        },
    )


if __name__ == "__main__":  # pragma: no cover
    print(publish_demo())
