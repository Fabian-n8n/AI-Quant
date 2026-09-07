"""
Live dashboard.

Two surfaces, same underlying data:

- **Terminal**, via rich. Cheap, always available, good for watching a run.
  Built in Phase 7, because the main loop's step 10 needs something to refresh.
- **Streamlit**, the web dashboard with the regime banner, price chart with
  regime overlay, signal feed and risk panel. Phase 8.

`DashboardState` is the shared part and belongs to neither surface. It collects
one snapshot dict from the live objects; both renderers read that and nothing
else. The web UI in Phase 8 adds a renderer, not a second way of computing the
numbers, so the two can never disagree about what the system thinks.

Panels, per the tutorial's finished dashboard:
  1. detected regime and confidence score
  2. portfolio value and buying power
  3. number of regimes the model settled on, and active positions
  4. price chart with regime overlay
  5. volume, confidence over time, regime distribution
  6. signal feed: allocation, entries, stops, live P&L
  7. risk controls: breaker states, drawdown levels, leverage
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


class DashboardState:
    """Collects everything the dashboard displays into one snapshot.

    Read-only by construction. The dashboard must never be able to place an
    order or clear a breaker: a display that can act is no longer a display,
    and a stray click should not be able to move real money. Nothing in this
    class calls a method that mutates, and it takes no order executor at all,
    so the capability is absent rather than merely unused.
    """

    def __init__(self, position_tracker=None, risk_manager=None, hmm_engine=None,
                 logger=None, engine=None) -> None:
        self.position_tracker = position_tracker
        self.risk_manager = risk_manager
        self.hmm_engine = hmm_engine
        self.logger = logger
        self.engine = engine

    # -- panels -------------------------------------------------------------

    def regime_panel(self) -> dict[str, Any]:
        """Detected regime, confidence, state count, stability."""
        state = getattr(self.engine, "regime_state", None)
        hmm = self.hmm_engine
        panel: dict[str, Any] = {
            "n_regimes": getattr(hmm, "n_states", None),
            "model_trained": None,
            "regime": "unknown",
            "confidence": 0.0,
            "confirmed": False,
            "consecutive_bars": 0,
            "flicker_rate": 0,
            "is_flickering": False,
            "size_multiplier": 1.0,
            "volatility_rank": None,
            "raw_regime": None,
        }

        metadata = getattr(hmm, "metadata", None)
        if metadata is not None and getattr(metadata, "training_date", None):
            panel["model_trained"] = metadata.training_date

        if state is not None:
            panel.update(
                regime=state.label.value,
                confidence=state.probability,
                confirmed=state.is_confirmed,
                consecutive_bars=state.consecutive_bars,
                flicker_rate=state.flicker_rate,
                is_flickering=state.is_flickering,
                size_multiplier=state.size_multiplier,
                raw_regime=state.raw_label.value,
                timestamp=state.timestamp,
            )
            if hmm is not None and getattr(hmm, "is_fitted", False):
                try:
                    panel["volatility_rank"] = hmm.get_volatility_rank(state.state_id).value
                except Exception:
                    pass
        return panel

    def portfolio_panel(self) -> dict[str, Any]:
        """Value, buying power, exposure, leverage, open positions."""
        portfolio = getattr(self.engine, "portfolio", None)
        if portfolio is None:
            return {}
        return {
            "equity": portfolio.equity,
            "cash": portfolio.cash,
            "buying_power": portfolio.buying_power,
            "gross_exposure": portfolio.gross_exposure,
            "leverage": portfolio.leverage,
            "n_positions": portfolio.n_positions,
            "unrealised_pnl": (
                self.position_tracker.unrealised_pnl() if self.position_tracker else 0.0
            ),
            "peak_equity": portfolio.peak_equity,
            "day_start_equity": portfolio.day_start_equity,
            "daily_trades": portfolio.daily_trades,
        }

    def positions_panel(self) -> list[dict[str, Any]]:
        if self.position_tracker is None:
            return []
        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "stop_loss": p.stop_loss,
                "has_stop": p.has_stop,
                "unrealised_pnl": p.unrealised_pnl,
                "unrealised_pnl_pct": p.unrealised_pnl_pct,
                "regime_at_entry": p.regime_at_entry,
                "regime_current": p.regime_current,
                "regime_changed": p.regime_changed,
                "holding_periods": getattr(p, "holding_periods", 0),
            }
            for p in self.position_tracker.get_open_positions().values()
        ]

    def risk_panel(self) -> dict[str, Any]:
        """Breaker states, drawdown levels, leverage status."""
        if self.risk_manager is None:
            return {}
        breaker = self.risk_manager.breaker
        portfolio = getattr(self.engine, "portfolio", None)

        panel = {
            "halted": breaker.is_halted(),
            "lock_file": str(breaker.lock_file),
            "daily_tripped": breaker.daily_tripped.value,
            "weekly_tripped": breaker.weekly_tripped.value,
            "peak_tripped": breaker.peak_tripped,
            "size_multiplier": breaker.size_multiplier,
            "n_triggers": len(breaker.history),
            "limits": {
                "daily_reduce": breaker.daily_dd_reduce,
                "daily_halt": breaker.daily_dd_halt,
                "weekly_reduce": breaker.weekly_dd_reduce,
                "weekly_halt": breaker.weekly_dd_halt,
                "max_from_peak": breaker.max_dd_from_peak,
                "max_exposure": self.risk_manager.max_exposure,
                "max_leverage": self.risk_manager.max_leverage,
            },
        }
        if portfolio is not None:
            panel["drawdowns"] = {
                "daily": portfolio.drawdown_daily,
                "weekly": portfolio.drawdown_weekly,
                "from_peak": portfolio.drawdown_from_peak,
            }
            panel["breaker_now"] = breaker.check(portfolio).value
        return panel

    def signal_feed(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent signals with allocation, entry, stop and live P&L."""
        if self.logger is None:
            return []
        try:
            return self.logger.get_trade_log(limit=limit)
        except Exception:
            return []

    def snapshot(self) -> dict[str, Any]:
        """Current regime, confidence, portfolio, positions, breakers, signals."""
        return {
            "timestamp": datetime.now(timezone.utc),
            "session": {
                "mode": getattr(self.engine, "mode", "unknown"),
                "paper": getattr(self.engine, "is_paper", True),
                "bars_processed": getattr(self.engine, "bars_processed", 0),
                "started_at": getattr(self.engine, "started_at", None),
                "market_open": getattr(self.engine, "market_open", None),
                "data_feed_healthy": getattr(self.engine, "data_feed_healthy", True),
            },
            "regime": self.regime_panel(),
            "portfolio": self.portfolio_panel(),
            "positions": self.positions_panel(),
            "risk": self.risk_panel(),
            "signals": self.signal_feed(),
        }


class TerminalDashboard:
    """rich-based view. One frame per call; the main loop drives the cadence."""

    def __init__(self, refresh_seconds: int = 5, console=None) -> None:
        self.refresh_seconds = refresh_seconds
        self._console = console

    @property
    def console(self):
        if self._console is None:
            from rich.console import Console

            self._console = Console()
        return self._console

    def render(self, state: dict[str, Any]) -> None:
        """Draw one frame."""
        from rich.panel import Panel
        from rich.table import Table

        regime = state.get("regime", {})
        portfolio = state.get("portfolio", {})
        risk = state.get("risk", {})
        session = state.get("session", {})
        positions = state.get("positions", [])

        header = Table.grid(padding=(0, 2))
        header.add_column(justify="left")
        header.add_column(justify="left")

        confirmed = "confirmed" if regime.get("confirmed") else "PENDING"
        flicker = " FLICKERING" if regime.get("is_flickering") else ""
        header.add_row(
            f"[bold]{regime.get('regime', 'unknown')}[/bold] "
            f"p={regime.get('confidence', 0):.2f} ({confirmed}{flicker})",
            f"vol rank [bold]{regime.get('volatility_rank') or '?'}[/bold]  "
            f"{regime.get('n_regimes', '?')} states  "
            f"size x{regime.get('size_multiplier', 1.0):g}",
        )
        header.add_row(
            f"equity [bold]${portfolio.get('equity', 0):,.2f}[/bold]  "
            f"buying power ${portfolio.get('buying_power', 0):,.2f}",
            f"exposure {portfolio.get('gross_exposure', 0):.1%}  "
            f"{portfolio.get('n_positions', 0)} positions  "
            f"P&L ${portfolio.get('unrealised_pnl', 0):,.2f}",
        )

        mode = session.get("mode", "?")
        market = "OPEN" if session.get("market_open") else "closed"
        feed = "" if session.get("data_feed_healthy", True) else "  [red]DATA FEED DOWN[/red]"
        title = (
            f"regime-trader  [{mode}]  "
            f"{'PAPER' if session.get('paper', True) else 'LIVE'}  "
            f"market {market}  bar {session.get('bars_processed', 0)}{feed}"
        )
        self.console.print(Panel(header, title=title, border_style="cyan"))

        if positions:
            table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
            for column in ("symbol", "qty", "entry", "last", "stop", "value", "P&L", "regime"):
                table.add_column(column, justify="right" if column != "symbol" else "left")
            for p in positions:
                pnl = p["unrealised_pnl"]
                colour = "green" if pnl >= 0 else "red"
                stop = f"{p['stop_loss']:.2f}" if p["has_stop"] else "[red]NONE[/red]"
                table.add_row(
                    p["symbol"], f"{p['quantity']:g}", f"{p['entry_price']:.2f}",
                    f"{p['current_price']:.2f}", stop, f"${p['market_value']:,.0f}",
                    f"[{colour}]{pnl:+,.0f} ({p['unrealised_pnl_pct']:+.1%})[/{colour}]",
                    (p["regime_at_entry"] or "?") + (" *" if p["regime_changed"] else ""),
                )
            self.console.print(table)
        else:
            self.console.print("  [dim]no open positions[/dim]")

        drawdowns = risk.get("drawdowns", {})
        limits = risk.get("limits", {})
        if drawdowns:
            halted = risk.get("halted")
            state_text = "[red bold]HALTED[/red bold]" if halted else risk.get(
                "breaker_now", "none"
            )
            self.console.print(
                f"  risk: {state_text}  "
                f"daily {drawdowns['daily']:.2%}/{-limits.get('daily_halt', 0):.2%}  "
                f"weekly {drawdowns['weekly']:.2%}/{-limits.get('weekly_halt', 0):.2%}  "
                f"peak {drawdowns['from_peak']:.2%}/{-limits.get('max_from_peak', 0):.2%}"
            )

    def run(self, state_source, stop_event=None) -> None:  # pragma: no cover - interactive
        """Refresh loop until interrupted. `state_source` is a zero-arg callable."""
        import time

        while stop_event is None or not stop_event.is_set():
            self.console.clear()
            self.render(state_source())
            time.sleep(self.refresh_seconds)


def run_streamlit_app() -> None:
    """Entry point for the Streamlit web dashboard.

        streamlit run monitoring/dashboard.py

    Phase 8. The terminal view above covers Phase 7's needs; the web UI is the
    frontend phase, and building it before the loop it displays was working
    would have meant styling a screen full of placeholder numbers.
    """
    raise NotImplementedError(
        "Phase 8: the Streamlit dashboard. Use `python main.py --dashboard` for the "
        "terminal view in the meantime."
    )


if __name__ == "__main__":  # pragma: no cover
    run_streamlit_app()
