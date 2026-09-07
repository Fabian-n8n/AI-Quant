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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

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


def build_payload(snapshot: dict[str, Any], *, source: str = "live",
                  equity_history: Optional[list] = None,
                  regime_history: Optional[list] = None,
                  notes: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Wrap a `DashboardState.snapshot()` in the envelope the UI expects."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "published_at": datetime.now(timezone.utc).isoformat(),
        **_clean(snapshot),
        "equity_history": _clean(equity_history or []),
        "regime_history": _clean(regime_history or []),
        "regime_mix": regime_mix(regime_history or []),
        "notes": _clean(notes or {}),
    }
    return payload


def publish(snapshot: dict[str, Any], path: Path = DEFAULT_OUTPUT, *,
            source: str = "live", equity_history: Optional[list] = None,
            regime_history: Optional[list] = None,
            notes: Optional[dict[str, Any]] = None) -> Path:
    """Write the snapshot to `path`. Atomic, so the UI never reads half a file."""
    payload = build_payload(
        snapshot, source=source, equity_history=equity_history,
        regime_history=regime_history, notes=notes,
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
        equity_history=list(getattr(engine.session, "equity_history", []) or []),
        regime_history=list(getattr(engine.session, "regime_history", []) or []),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def demo_snapshot(seed: int = 7) -> dict[str, Any]:
    """A plausible but explicitly fake snapshot, for a fresh clone.

    Marked `source: "demo"` and rendered behind a banner in the UI. The numbers
    are shaped like real ones so the layout can be judged, and are not real.
    """
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)

    equity, peak = 100_000.0, 100_000.0
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
        {"symbol": "SPY", "direction": "LONG", "quantity": 62, "entry_price": 514.20,
         "current_price": 520.30, "market_value": 32258.60, "stop_loss": 508.00,
         "has_stop": True, "unrealised_pnl": 378.20, "unrealised_pnl_pct": 0.0119,
         "distance_to_stop_pct": 0.0236, "regime_at_entry": "strong_bull",
         "regime_current": "strong_bull", "regime_changed": False,
         "holding_periods": 3, "held_for": "3h", "adopted": False},
        {"symbol": "QQQ", "direction": "LONG", "quantity": 41, "entry_price": 441.80,
         "current_price": 438.15, "market_value": 17964.15, "stop_loss": 429.60,
         "has_stop": True, "unrealised_pnl": -149.65, "unrealised_pnl_pct": -0.0083,
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
            (12, "signal_generated", "SPY", 62, 31880.40, "strong_bull",
             "SPY: approved 62 shares ($31,880.40)", None),
            (12, "signal_rejected", "AAPL", 0, 0.0, "strong_bull",
             "AAPL: rejected, correlation 0.89 with SPY", "correlation_too_high"),
            (73, "signal_generated", "QQQ", 41, 18113.80, "weak_bull",
             "QQQ: approved 41 shares ($18,113.80)", None),
            (1502, "signal_rejected", "TSLA", 0, 0.0, "weak_bull",
             "TSLA: rejected, $92 is below the $100 minimum", "below_minimum_size"),
        ]
    ]

    return {
        "timestamp": now.isoformat(),
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
            "n_positions": len(positions), "unrealised_pnl": 228.55,
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
