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
import logging
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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


def _session_state(market_open, next_open, next_close) -> dict[str, Any]:
    """Market status in words, with how long until it changes.

    An order resting unfilled at 08:51 looks identical to a broken system
    unless the page says the market opens at 09:30. This is the difference
    between "nothing is happening" and "nothing is happening yet".
    """
    def hours_until(when) -> float | None:
        if not when:
            return None
        try:
            stamp = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            return round((stamp - datetime.now(UTC)).total_seconds() / 3600, 2)
        except ValueError:
            return None

    if market_open:
        return {"state": "open", "label": "Market open",
                "detail": "Resting orders can fill now.",
                "hours_until_change": hours_until(next_close)}
    if market_open is None:
        return {"state": "unknown", "label": "Market status unknown",
                "detail": "The broker clock could not be read.",
                "hours_until_change": None}

    hours = hours_until(next_open)
    if hours is None:
        detail = "Orders rest until the next session opens."
    elif hours < 1:
        detail = f"Opens in {int(hours * 60)} minutes. Orders rest until then."
    elif hours < 24:
        detail = f"Opens in {hours:.1f} hours. Orders rest until then."
    else:
        detail = f"Opens in {hours / 24:.1f} days. Orders rest until then."
    return {"state": "closed", "label": "Market closed", "detail": detail,
            "hours_until_change": hours}


def timing(snapshot: dict[str, Any], engine=None) -> dict[str, Any]:
    """When a signal would actually be acted on.

    "Buy this" is incomplete without "at what price, in which session, and by
    what order type". On a daily-bar system the answer is never "right now":
    the bar the decision is based on closes at 4pm New York, and the order rests
    until the next session opens.
    """
    system = snapshot.get("system", {}) or {}
    next_open = getattr(engine, "next_open", None) if engine else None
    next_close = getattr(engine, "next_close", None) if engine else None
    return {
        "market_open": system.get("market_open"),
        "next_open": next_open,
        "next_close": next_close,
        # Spelled out rather than left for the UI to infer from a boolean.
        # "Closed" and "closed, opens in 39 minutes" are different messages,
        # and the second is the one that stops you wondering whether the
        # system is broken while it is simply waiting.
        "session_state": _session_state(system.get("market_open"), next_open, next_close),
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
                  activity: dict[str, Any] | None = None,
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
        # Drives the dashboard's demo banner. A real boolean rather than a
        # string comparison on `source`, so the banner cannot be left showing
        # over real data because someone renamed a source label.
        "is_demo": source == "demo",
        "activity": _clean(activity or empty_activity()),
    }
    return payload


# ---------------------------------------------------------------------------
# Activity, read from state.db
# ---------------------------------------------------------------------------

def empty_activity() -> dict[str, Any]:
    """The shape the Activity page expects when there is nothing to show.

    Returned rather than omitting the key, because a UI that has to handle both
    "missing" and "empty" will eventually handle only one of them.
    """
    return {"orders": [], "open_positions": [], "closed_positions": [],
            "runs": [], "expectancy": {"trades": 0, "expectancy": 0.0,
                                       "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0}}


def activity_from_repo(repo, limit: int = 100) -> dict[str, Any]:
    """Orders, positions and runs out of SQLite, newest first.

    Orders and fills stay distinguishable all the way to the UI. A cancelled
    order is not a trade, and collapsing the two is exactly how thirty resting
    test orders came to look like a broken system.
    """
    def rows(records):
        return [dict(r) for r in records]

    try:
        return {
            "orders": rows(repo.recent_orders(limit=limit)),
            "open_positions": rows(repo.open_positions()),
            "closed_positions": rows(repo.closed_positions(limit=limit)),
            "runs": rows(repo.recent_runs(limit=20)),
            "expectancy": repo.expectancy(),
        }
    except Exception as exc:
        logger.warning("could not read activity from state.db: %s", exc)
        return empty_activity()


def publish(snapshot: dict[str, Any], path: Path = DEFAULT_OUTPUT, *,
            source: str = "live", equity_history: list | None = None,
            regime_history: list | None = None,
            notes: dict[str, Any] | None = None,
            activity: dict[str, Any] | None = None, engine=None) -> Path:
    """Write the snapshot to `path`. Atomic, so the UI never reads half a file."""
    payload = build_payload(
        snapshot, source=source, equity_history=equity_history,
        regime_history=regime_history, notes=notes, activity=activity, engine=engine,
    )

    # Refuse to replace a snapshot that says something with one that says
    # nothing. The activity block is still merged forward, because run history
    # is real even on a cycle that computed no regime.
    path = Path(path)
    if is_blank(payload) and path.exists():
        try:
            previous = json.loads(path.read_text())
        except Exception:
            previous = None
        if previous and not is_blank(previous):
            logger.info("skipping publish: this cycle produced no regime and the "
                        "existing snapshot has one. Keeping it.")
            previous["activity"] = payload["activity"]
            previous["published_at"] = payload["published_at"]
            return _write(previous, path)

    return _write(payload, path)


def is_blank(payload: dict[str, Any]) -> bool:
    """Does this snapshot say nothing?

    A bar that was skipped produces `regime: unknown`, zero candidates and no
    positions. That is not a state worth showing, and publishing it over a good
    snapshot is how the dashboard went blank: the run succeeded, the file was
    rewritten, and every panel lost its contents.
    """
    regime = payload.get("regime", {}) or {}
    return (
        regime.get("regime") in (None, "unknown")
        and not payload.get("candidates")
        and not payload.get("positions")
    )


def _write(payload: dict[str, Any], path: Path) -> Path:
    """Atomic write, so the UI never reads half a file."""
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
    repo = getattr(engine, "repo", None)
    return publish(
        engine.dashboard_state.snapshot(),
        path,
        source="live",
        engine=engine,
        equity_history=list(getattr(engine.session, "equity_history", []) or []),
        regime_history=list(getattr(engine.session, "regime_history", []) or []),
        notes=notes,
        activity=activity_from_repo(repo) if repo is not None else None,
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
        "risk_pct_of_equity": round(risk_dollars / 125_652, 6),
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
        {"symbol": "SPY", "direction": "LONG", "quantity": 3, "entry_price": 762.40,
         "current_price": 770.19, "market_value": 2310.57, "stop_loss": 746.62,
         "has_stop": True, "unrealised_pnl": 23.37, "unrealised_pnl_pct": 0.0102,
         "distance_to_stop_pct": 0.0236, "regime_at_entry": "strong_bull",
         "regime_current": "strong_bull", "regime_changed": False,
         "holding_periods": 3, "held_for": "3h", "adopted": False},
        {"symbol": "QQQ", "direction": "LONG", "quantity": 4, "entry_price": 718.96,
         "current_price": 712.99, "market_value": 2851.96, "stop_loss": 705.63,
         "has_stop": True, "unrealised_pnl": -23.88, "unrealised_pnl_pct": -0.0083,
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
            (12, "signal_generated", "COIN", 16, 2954.24, "strong_bull",
             "COIN: approved 16 shares ($2,954.24)", None),
            (12, "signal_rejected", "AAPL", 0, 0.0, "strong_bull",
             "AAPL: rejected, correlation 0.89 with SPY", "correlation_too_high"),
            (73, "signal_generated", "AAPL", 9, 2879.73, "weak_bull",
             "AAPL: approved 9 shares ($2,879.73)", None),
            (1502, "signal_rejected", "TSLA", 0, 0.0, "weak_bull",
             "TSLA: rejected, $92 is below the $100 minimum", "below_minimum_size"),
        ]
    ]

    candidates = [
        _demo_candidate("COIN", 1, True, "buy", 0.81, 16, 2954.24, 184.64, 163.03,
                        "above", 0.038, 0.0412, 0.094, "LowVolBullStrategy",
                        modifications=["capped at 3% single position"]),
        _demo_candidate("AMZN", 2, True, "buy", 0.74, 11, 2843.61, 258.51, 252.91,
                        "above", 0.019, 0.0092, 0.036, "LowVolBullStrategy",
                        modifications=["capped at 3% single position"]),
        _demo_candidate("AAPL", 3, True, "buy", 0.74, 9, 2879.73, 319.97, 308.57,
                        "above", 0.021, 0.0118, 0.043, "LowVolBullStrategy",
                        modifications=["capped at 3% single position"]),
        _demo_candidate("SMCI", 4, True, "buy", 0.73, 75, 2969.25, 39.59, 32.78,
                        "above", 0.029, 0.0451, 0.077, "LowVolBullStrategy",
                        modifications=["capped at 3% single position"]),
        _demo_candidate("QQQ", 5, True, "hold", 0.69, 4, 2875.84, 718.96, 705.63,
                        "above", 0.009, 0.0104, 0.018, "LowVolBullStrategy",
                        modifications=[], held=True, held_quantity=4),
        _demo_candidate("NVDA", 6, False, "blocked", 0.61, 0, 0.0, 230.36, 206.43,
                        "above", 0.034, 0.0295, 0.112, "LowVolBullStrategy",
                        rejection_reason="correlation_too_high",
                        reason="correlation 0.89 with SMCI, above the 0.85 reject threshold"),
        _demo_candidate("AVGO", 7, False, "blocked", 0.44, 0, 0.0, 358.02, 344.19,
                        "below", -0.008, 0.0131, -0.014, "LowVolBullStrategy",
                        rejection_reason="sector_limit",
                        reason="semiconductors already at 30% of equity, cap is 30%"),
        _demo_candidate("TSLA", 8, False, "blocked", 0.29, 0, 0.0, 354.10, 333.38,
                        "below", -0.041, 0.0384, -0.087, "LowVolBullStrategy",
                        rejection_reason="duplicate_order",
                        reason="TSLA long already sent within the 60s duplicate window")
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
            "n_positions": len(positions), "unrealised_pnl": -0.51,
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
            "next_close": (now + timedelta(hours=4)).isoformat(),
            "session_state": {"state": "open", "label": "Market open",
                              "detail": "Resting orders can fill now.",
                              "hours_until_change": 4.0},
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


def demo_activity(seed: int = 7) -> dict[str, Any]:
    """Sample orders, positions and runs for the Activity page.

    Deliberately includes a cancelled order and a skipped one alongside fills.
    A demo where everything filled would hide the distinction the page exists
    to make, and that distinction is exactly what was misread when thirty
    resting test orders looked like a broken system.
    """
    rng = random.Random(seed)
    now = datetime.now(UTC)

    def when(days: float) -> str:
        return (now - timedelta(days=days)).isoformat(timespec="seconds")

    orders = [
        ("COIN", "buy", "limit", 16, 184.64, 184.58, 16, "filled", 0.4, None),
        ("PLTR", "buy", "limit", 17, 174.40, 174.31, 17, "filled", 0.4, None),
        ("SPY", "buy", "limit", 3, 770.19, 770.05, 3, "filled", 1.4, None),
        ("NVDA", "buy", "limit", 12, 241.80, None, 0, "canceled", 2.3, None),
        ("AMD", "buy", "limit", 14, 208.15, None, 0, "canceled", 3.3,
         "an equivalent buy order is already open"),
        ("MSFT", "sell", "limit", 5, 512.40, 512.66, 5, "filled", 5.2, None),
    ]

    return {
        "orders": [
            {
                "symbol": symbol, "side": side, "order_type": kind, "quantity": qty,
                "submitted_price": submitted, "fill_price": fill, "filled_qty": filled,
                "status": status, "submitted_at": when(age),
                "filled_at": when(age) if fill else None,
                "stop_loss": round(submitted * 0.92, 2), "regime": "strong_bull",
                "skipped_reason": skipped,
            }
            for symbol, side, kind, qty, submitted, fill, filled, status, age, skipped in orders
        ],
        "open_positions": [
            {"symbol": "COIN", "quantity": 16, "entry_price": 184.58,
             "entry_at": when(0.4), "current_price": 191.20, "stop_price": 163.04,
             "unrealised_pnl": round((191.20 - 184.58) * 16, 2), "holding_days": 0,
             "regime_at_entry": "strong_bull"},
            {"symbol": "PLTR", "quantity": 17, "entry_price": 174.31,
             "entry_at": when(0.4), "current_price": 171.05, "stop_price": 157.20,
             "unrealised_pnl": round((171.05 - 174.31) * 17, 2), "holding_days": 0,
             "regime_at_entry": "strong_bull"},
            {"symbol": "SPY", "quantity": 3, "entry_price": 770.05,
             "entry_at": when(1.4), "current_price": 774.90, "stop_price": 754.41,
             "unrealised_pnl": round((774.90 - 770.05) * 3, 2), "holding_days": 1,
             "regime_at_entry": "strong_bull"},
        ],
        "closed_positions": [
            {"symbol": "MSFT", "quantity": 5, "entry_price": 486.10, "entry_at": when(19),
             "exit_price": 512.66, "exit_at": when(5.2), "exit_reason": "target",
             "realised_pnl": round((512.66 - 486.10) * 5, 2), "holding_days": 14,
             "regime_at_entry": "weak_bull"},
            {"symbol": "TSLA", "quantity": 7, "entry_price": 402.30, "entry_at": when(28),
             "exit_price": 371.15, "exit_at": when(12), "exit_reason": "trailing_stop",
             "realised_pnl": round((371.15 - 402.30) * 7, 2), "holding_days": 16,
             "regime_at_entry": "neutral"},
            {"symbol": "AVGO", "quantity": 9, "entry_price": 318.75, "entry_at": when(41),
             "exit_price": 296.40, "exit_at": when(24), "exit_reason": "stop",
             "realised_pnl": round((296.40 - 318.75) * 9, 2), "holding_days": 17,
             "regime_at_entry": "weak_bear"},
        ],
        "runs": [
            {"id": 40 - i, "started_at": when(i + 0.02), "finished_at": when(i),
             "status": "ok" if i != 3 else "failed",
             "mode": "paper", "trigger": "schedule",
             "bars_processed": 1, "orders_submitted": rng.randint(0, 3),
             "regime": "strong_bull", "equity": round(100_000 + rng.uniform(-2000, 4000), 2),
             "error": None if i != 3 else "broker unreachable after 3 attempts"}
            for i in range(8)
        ],
        "expectancy": {
            "trades": 3,
            "expectancy": round((132.80 - 218.05 - 201.15) / 3, 2),
            "win_rate": 1 / 3, "avg_win": 132.80, "avg_loss": -209.60,
        },
    }


def demo_payload() -> dict[str, Any]:
    """The exact payload `publish_demo` writes.

    Separate from `publish_demo` so the contract tests can assert against the
    same object that ships, rather than reassembling it and slowly drifting
    from it. They had already drifted once: the test fixture was building a
    payload with no activity block while the published file had one.
    """
    snapshot = demo_snapshot()
    return build_payload(
        snapshot, source="demo",
        equity_history=snapshot.pop("equity_history"),
        regime_history=snapshot.pop("regime_history"),
        activity=demo_activity(),
        notes={
            "banner": "Demo data. Not a real account.",
            "verdict": "no demonstrated edge out-of-sample",
        },
    )


def publish_demo(path: Path = DEFAULT_OUTPUT) -> Path:
    return _write(demo_payload(), path)


if __name__ == "__main__":  # pragma: no cover
    print(publish_demo())
