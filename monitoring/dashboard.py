"""
Live dashboard.

Two surfaces, one source of truth:

- **Terminal**, via rich. Built here. Refreshes every 5 seconds, colour-coded
  risk bars, the six panels the Phase 8 spec draws.
- **Web**, the Next.js app in `dashboard/`. It reads the JSON that
  `monitoring/publish.py` writes from the same `DashboardState.snapshot()`.

`DashboardState` belongs to neither surface. Both renderers read its snapshot
and nothing else, so the terminal and the web UI can never disagree about what
the system thinks. Adding a third surface means adding a renderer, never a
second way of computing the numbers.

Read-only by construction. The dashboard must never be able to place an order or
clear a breaker: a display that can act is no longer a display, and a stray
click should not be able to move real money. `DashboardState` takes no order
executor at all, so the capability is absent rather than merely unused.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

#: Risk bar width in characters. Small on purpose: this is a status indicator,
#: not a chart, and a wide bar reads as more precision than a drawdown ratio has.
BAR_WIDTH = 12


class DashboardState:
    """Collects everything the dashboard displays into one snapshot."""

    def __init__(self, position_tracker=None, risk_manager=None, hmm_engine=None,
                 logger=None, engine=None) -> None:
        self.position_tracker = position_tracker
        self.risk_manager = risk_manager
        self.hmm_engine = hmm_engine
        self.logger = logger
        self.engine = engine

    # -- panels -------------------------------------------------------------

    def regime_panel(self) -> dict[str, Any]:
        """Detected regime, confidence, stability, flicker."""
        state = getattr(self.engine, "regime_state", None)
        hmm = self.hmm_engine
        panel: dict[str, Any] = {
            "n_regimes": getattr(hmm, "n_states", None),
            "model_trained": None,
            "model_age_days": None,
            "regime": "unknown",
            "confidence": 0.0,
            "confirmed": False,
            "consecutive_bars": 0,
            "flicker_rate": 0,
            "flicker_window": getattr(hmm, "flicker_window", 20),
            "flicker_threshold": getattr(hmm, "flicker_threshold", 4),
            "is_flickering": False,
            "size_multiplier": 1.0,
            "volatility_rank": None,
            "raw_regime": None,
            "timestamp": None,
        }

        metadata = getattr(hmm, "metadata", None)
        if metadata is not None and getattr(metadata, "training_date", None):
            panel["model_trained"] = metadata.training_date
            trained = metadata.training_date
            if trained.tzinfo is None:
                trained = trained.replace(tzinfo=timezone.utc)
            panel["model_age_days"] = (
                datetime.now(timezone.utc) - trained
            ).total_seconds() / 86400.0

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
        """Equity, daily P&L, allocation, leverage, buying power."""
        portfolio = getattr(self.engine, "portfolio", None)
        if portfolio is None:
            return {}

        day_start = portfolio.day_start_equity or portfolio.equity
        daily_pnl = portfolio.equity - day_start
        return {
            "equity": portfolio.equity,
            "cash": portfolio.cash,
            "buying_power": portfolio.buying_power,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": (daily_pnl / day_start) if day_start else 0.0,
            "allocation": portfolio.gross_exposure,
            "target_allocation": getattr(self.engine, "target_allocation", None),
            "leverage": portfolio.leverage,
            "gross_exposure": portfolio.gross_exposure,
            "n_positions": portfolio.n_positions,
            "unrealised_pnl": (
                self.position_tracker.unrealised_pnl() if self.position_tracker else 0.0
            ),
            "peak_equity": portfolio.peak_equity,
            "day_start_equity": day_start,
            "daily_trades": portfolio.daily_trades,
        }

    def positions_panel(self) -> list[dict[str, Any]]:
        if self.position_tracker is None:
            return []
        return [
            {
                "symbol": p.symbol,
                "direction": "LONG",       # long-only by construction, Phase 3
                "quantity": p.quantity,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "stop_loss": p.stop_loss,
                "has_stop": p.has_stop,
                "unrealised_pnl": p.unrealised_pnl,
                "unrealised_pnl_pct": p.unrealised_pnl_pct,
                "distance_to_stop_pct": p.distance_to_stop_pct,
                "regime_at_entry": p.regime_at_entry,
                "regime_current": p.regime_current,
                "regime_changed": p.regime_changed,
                "holding_periods": getattr(p, "holding_periods", 0),
                "held_for": _held_for(getattr(p, "entry_time", None)),
                "adopted": getattr(p, "adopted", False),
            }
            for p in self.position_tracker.get_open_positions().values()
        ]

    def risk_panel(self) -> dict[str, Any]:
        """Breaker states, drawdowns against their limits, exposure headroom."""
        if self.risk_manager is None:
            return {}
        breaker = self.risk_manager.breaker
        portfolio = getattr(self.engine, "portfolio", None)

        panel: dict[str, Any] = {
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
                "max_risk_per_trade": self.risk_manager.max_risk_per_trade,
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

    def system_panel(self) -> dict[str, Any]:
        """Feed health, broker latency, model age, paper/live."""
        engine = self.engine
        regime = self.regime_panel()
        return {
            "data_feed_healthy": getattr(engine, "data_feed_healthy", True),
            "broker_connected": bool(getattr(engine, "account", None)),
            "api_latency_ms": getattr(engine, "api_latency_ms", None),
            "model_age_days": regime["model_age_days"],
            "paper": getattr(engine, "is_paper", True),
            "mode": getattr(engine, "mode", "unknown"),
            "market_open": getattr(engine, "market_open", None),
            "bars_processed": getattr(engine, "bars_processed", 0),
            "consecutive_errors": getattr(engine, "consecutive_errors", 0),
            "started_at": getattr(engine, "started_at", None),
            "symbols": list(getattr(engine, "symbols", []) or []),
            "timeframe": getattr(engine, "timeframe", None),
        }

    def signal_feed(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent signals with allocation, entry, stop and the reason."""
        if self.logger is None:
            return []
        try:
            return self.logger.get_trade_log(limit=limit)
        except Exception:
            return []

    def snapshot(self) -> dict[str, Any]:
        """Everything both renderers draw, computed once."""
        return {
            "timestamp": datetime.now(timezone.utc),
            "session": self.system_panel(),
            "regime": self.regime_panel(),
            "portfolio": self.portfolio_panel(),
            "positions": self.positions_panel(),
            "risk": self.risk_panel(),
            "system": self.system_panel(),
            "signals": self.signal_feed(),
        }


def _held_for(entry_time) -> str:
    """Human holding duration, as the spec's positions row shows ("3h")."""
    if entry_time is None:
        return "-"
    if getattr(entry_time, "tzinfo", None) is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - entry_time).total_seconds()
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def risk_bar(used: float, limit: float, width: int = BAR_WIDTH) -> tuple[str, str]:
    """A colour-coded bar for one drawdown against its limit.

    Returns (bar, colour). Colour is a ratio of the limit consumed, not of the
    drawdown itself: 2% of a 3% limit is nearly spent and should read red, while
    2% of a 10% limit has plenty of room and should read green. Grading on the
    raw drawdown would show the same colour for both, which is exactly backwards.
    """
    if limit <= 0:
        return "─" * width, "dim"
    ratio = min(1.0, abs(used) / abs(limit))
    filled = int(round(ratio * width))
    colour = "green" if ratio < 0.5 else "yellow" if ratio < 0.8 else "red"
    return "█" * filled + "░" * (width - filled), colour


class TerminalDashboard:
    """rich-based view. Six panels, refreshed on a 5-second cadence."""

    def __init__(self, refresh_seconds: int = 5, console=None) -> None:
        self.refresh_seconds = refresh_seconds
        self._console = console

    @property
    def console(self):
        if self._console is None:
            from rich.console import Console

            self._console = Console()
        return self._console

    # -- panel builders -----------------------------------------------------

    def _regime_panel(self, state: dict[str, Any]):
        from rich.table import Table

        r = state.get("regime", {})
        grid = Table.grid(padding=(0, 3))
        for _ in range(4):
            grid.add_column()

        confidence = r.get("confidence", 0.0)
        colour = "green" if confidence >= 0.7 else "yellow" if confidence >= 0.55 else "red"
        pending = "" if r.get("confirmed") else " [yellow]PENDING[/yellow]"
        flicker = f"{r.get('flicker_rate', 0)}/{r.get('flicker_window', 20)}"
        flicker_colour = "red" if r.get("is_flickering") else "dim"

        grid.add_row(
            f"[bold {colour}]{str(r.get('regime', 'unknown')).upper()}[/bold {colour}] "
            f"[{colour}]({confidence:.0%})[/{colour}]{pending}",
            f"Stability: [bold]{r.get('consecutive_bars', 0)}[/bold] bars",
            f"Flicker: [{flicker_colour}]{flicker}[/{flicker_colour}]",
            f"Vol rank: [bold]{r.get('volatility_rank') or '?'}[/bold]"
            + (f"  size x{r['size_multiplier']:g}" if r.get("size_multiplier", 1) != 1 else ""),
        )
        return grid

    def _portfolio_panel(self, state: dict[str, Any]):
        from rich.table import Table

        p = state.get("portfolio", {})
        grid = Table.grid(padding=(0, 3))
        for _ in range(4):
            grid.add_column()

        pnl = p.get("daily_pnl", 0.0)
        pnl_colour = "green" if pnl >= 0 else "red"
        leverage = p.get("leverage", 0.0)

        grid.add_row(
            f"Equity: [bold]${p.get('equity', 0):,.0f}[/bold]",
            f"Daily: [{pnl_colour}]{pnl:+,.0f} ({p.get('daily_pnl_pct', 0):+.2%})[/{pnl_colour}]",
            f"Buying power: ${p.get('buying_power', 0):,.0f}",
            f"Peak: ${p.get('peak_equity', 0):,.0f}",
        )
        grid.add_row(
            f"Allocation: [bold]{p.get('allocation', 0):.0%}[/bold]",
            f"Leverage: [bold]{leverage:.2f}x[/bold]",
            f"Positions: {p.get('n_positions', 0)}",
            f"Trades today: {p.get('daily_trades', 0)}",
        )
        return grid

    def _positions_panel(self, state: dict[str, Any]):
        from rich.table import Table
        from rich.text import Text

        positions = state.get("positions", [])
        if not positions:
            return Text("  no open positions", style="dim")

        table = Table.grid(padding=(0, 3))
        for _ in range(7):
            table.add_column()
        for p in positions:
            pnl_colour = "green" if p["unrealised_pnl"] >= 0 else "red"
            stop = (f"Stop: ${p['stop_loss']:,.2f}" if p["has_stop"]
                    else "[red bold]NO STOP[/red bold]")
            table.add_row(
                f"[bold]{p['symbol']}[/bold]",
                p["direction"],
                f"${p['current_price']:,.2f}",
                f"[{pnl_colour}]{p['unrealised_pnl_pct']:+.1%}[/{pnl_colour}]",
                stop,
                p["held_for"],
                ("[yellow]regime changed[/yellow]" if p["regime_changed"] else ""),
            )
        return table

    def _signals_panel(self, state: dict[str, Any]):
        from rich.table import Table
        from rich.text import Text

        signals = state.get("signals", [])[-5:]
        if not signals:
            return Text("  no signals yet", style="dim")

        table = Table.grid(padding=(0, 3))
        for _ in range(4):
            table.add_column()
        for event in reversed(signals):
            stamp = str(event.get("timestamp", ""))[11:16]
            rejected = event.get("event") == "signal_rejected"
            verdict = (f"[red]rejected: {event.get('rejection_reason', '')}[/red]" if rejected
                       else f"[green]{event.get('shares', 0):g} shares "
                            f"${event.get('notional', 0):,.0f}[/green]")
            table.add_row(
                f"[dim]{stamp}[/dim]",
                f"[bold]{event.get('symbol', '?')}[/bold]",
                verdict,
                f"[dim]{str(event.get('signal_regime', ''))[:18]}[/dim]",
            )
        return table

    def _risk_panel(self, state: dict[str, Any]):
        from rich.table import Table
        from rich.text import Text

        risk = state.get("risk", {})
        drawdowns = risk.get("drawdowns")
        limits = risk.get("limits", {})
        if not drawdowns:
            return Text("  no portfolio snapshot yet", style="dim")

        table = Table.grid(padding=(0, 2))
        for _ in range(4):
            table.add_column()

        for label, used, limit in (
            ("Daily DD", drawdowns["daily"], limits.get("daily_halt", 0)),
            ("Weekly DD", drawdowns["weekly"], limits.get("weekly_halt", 0)),
            ("From Peak", drawdowns["from_peak"], limits.get("max_from_peak", 0)),
        ):
            bar, colour = risk_bar(used, limit)
            ok = abs(used) < abs(limit) if limit else True
            table.add_row(
                f"{label}:",
                f"[{colour}]{abs(used):.2%}[/{colour}] / {abs(limit):.0%}",
                f"[{colour}]{bar}[/{colour}]",
                "[green]OK[/green]" if ok else "[red bold]BREACHED[/red bold]",
            )

        if risk.get("halted"):
            table.add_row("[red bold]HALTED[/red bold]",
                          f"[red]delete {risk.get('lock_file', '')} to resume[/red]", "", "")
        return table

    def _system_panel(self, state: dict[str, Any]):
        from rich.table import Table

        system = state.get("system", {})
        grid = Table.grid(padding=(0, 3))
        for _ in range(5):
            grid.add_column()

        def tick(ok: bool) -> str:
            return "[green]OK[/green]" if ok else "[red bold]DOWN[/red bold]"

        latency = system.get("api_latency_ms")
        age = system.get("model_age_days")
        grid.add_row(
            f"Data: {tick(system.get('data_feed_healthy', True))}",
            f"API: {tick(system.get('broker_connected', False))}"
            + (f" [dim]{latency:.0f}ms[/dim]" if latency is not None else ""),
            f"HMM: [bold]{age:.1f}d[/bold] ago" if age is not None else "HMM: [dim]none[/dim]",
            f"[bold]{'PAPER' if system.get('paper', True) else 'LIVE'}[/bold]",
            f"Market: {'[green]open[/green]' if system.get('market_open') else '[dim]closed[/dim]'}"
            f"  bar {system.get('bars_processed', 0)}",
        )
        return grid

    # -- frame --------------------------------------------------------------

    def build(self, state: dict[str, Any]):
        """One renderable frame. Separated from `render` so `rich.Live` can
        redraw without the flicker of clearing and reprinting."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.rule import Rule

        sections = [
            ("REGIME", self._regime_panel(state)),
            ("PORTFOLIO", self._portfolio_panel(state)),
            ("POSITIONS", self._positions_panel(state)),
            ("RECENT SIGNALS", self._signals_panel(state)),
            ("RISK STATUS", self._risk_panel(state)),
            ("SYSTEM", self._system_panel(state)),
        ]

        body = []
        for index, (title, content) in enumerate(sections):
            if index:
                body.append(Rule(f"[bold]{title}[/bold]", style="grey35", align="left"))
            body.append(content)

        system = state.get("system", {})
        stamp = str(state.get("timestamp", ""))[11:19]
        title = (f"regime-trader  [{system.get('mode', '?')}]  "
                 f"{', '.join(system.get('symbols', [])[:6]) or 'no symbols'}  "
                 f"{system.get('timeframe', '')}")
        return Panel(
            Group(Rule(f"[bold]{sections[0][0]}[/bold]", style="grey35", align="left"), *body),
            title=title, subtitle=f"[dim]{stamp} UTC[/dim]", border_style="magenta",
        )

    def render(self, state: dict[str, Any]) -> None:
        """Draw one frame."""
        self.console.print(self.build(state))

    def run(self, state_source, stop_event=None) -> None:  # pragma: no cover - interactive
        """Refresh loop. `state_source` is a zero-arg callable returning a snapshot.

        `rich.Live` rather than clear-and-reprint: a 5-second full-screen clear
        makes the terminal unusable for anything else, and any scrollback of the
        run is lost.
        """
        from rich.live import Live

        with Live(console=self.console, refresh_per_second=4, screen=False) as live:
            while stop_event is None or not stop_event.is_set():
                try:
                    live.update(self.build(state_source()))
                except Exception as exc:
                    self.console.print(f"[red]dashboard error: {exc}[/red]")
                if stop_event is not None:
                    stop_event.wait(self.refresh_seconds)
                else:
                    time.sleep(self.refresh_seconds)


def run_streamlit_app() -> None:
    """The tutorial's Streamlit dashboard is not what this project ships.

    Streamlit needs a persistent Python server, so it cannot deploy to Vercel,
    and this project's web dashboard is the Next.js app in `dashboard/`. It reads
    the JSON that `monitoring/publish.py` writes from the very same
    `DashboardState.snapshot()` the terminal view uses.

        python main.py --publish      # write dashboard/public/data/state.json
        cd dashboard && npm run dev
    """
    raise NotImplementedError(
        "This project's web dashboard is the Next.js app in dashboard/, not Streamlit. "
        "Run `python main.py --publish` then `cd dashboard && npm run dev`. "
        "Use `python main.py --dashboard` for the terminal view."
    )


if __name__ == "__main__":  # pragma: no cover
    run_streamlit_app()
