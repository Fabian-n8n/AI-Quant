"""Multi-symbol walk-forward backtest that runs the real risk layer.

WHY THIS EXISTS
---------------
`WalkForwardBacktester` rebalances a single asset straight to the strategy's
target allocation. It has no per-position cap, no position count limit and no
correlation check, so it averaged 79.6% invested on SPY.

The live system trades twelve symbols through `core.risk_manager`, which caps
every position at `max_single_position` and the count at `max_concurrent`.
Three percent times twelve is a 36% ceiling, and the account was observed
holding five positions at 14% invested while the engine asked for 119%.

So the number that says "+29.5%, Sharpe 0.77" describes a strategy the risk
layer cannot run. Every conclusion drawn from it, including whether it beats
buy-and-hold, was about a different system.

This module closes that gap. It reuses `core.candidates.scan`, which is the
same call the live engine makes, so sizing, correlation rejection, sector caps
and the position cap are the live implementations rather than a copy that
drifts. If the risk layer changes, this changes with it.

WHAT IT DOES NOT DO
-------------------
It is not a market simulator. Fills are at the next bar's open plus slippage,
there is no order book, no partial fills and no borrow cost on margin. Those
omissions flatter the result, and the honest use of the output is to compare
configurations against each other rather than to predict a dollar figure.

LOOK-AHEAD
----------
Every decision is taken on data up to a bar's close and executed at the next
bar's open. A position entered on bar t is not stop-checked until t+1: within
a single bar the order of the high and the low is unknown, and assuming the
stop was missed on the entry bar is the assumption that costs money rather
than the one that flatters.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from core.risk_manager import PortfolioState, RiskManager
from data.feature_engineering import build_feature_matrix, log_returns, required_warmup

from .backtester import Window

logger = logging.getLogger(__name__)


@dataclass
class Holding:
    """One open position, with the context needed to close it honestly."""
    symbol: str
    shares: float
    entry_price: float
    entry_time: Any
    stop_loss: float | None = None
    take_profit: float | None = None
    regime_at_entry: str = "unknown"
    bars_held: int = 0

    def value_at(self, price: float) -> float:
        return self.shares * price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price


@dataclass
class PortfolioBacktestResult:
    """Out-of-sample only. No in-sample bar appears in the equity curve."""
    symbols: list[str] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    windows: list[Window] = field(default_factory=list)
    window_summaries: list[dict] = field(default_factory=list)
    initial_capital: float = 100_000.0
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.trade_log)

    @property
    def is_significant(self) -> bool:
        return self.n_trades >= 30

    @property
    def avg_exposure(self) -> float:
        """Mean gross exposure over the OOS span.

        This is the number the whole module exists to produce. Compared against
        the single-asset backtest's 79.6% it says whether the two are measuring
        the same strategy.
        """
        if self.history.empty or "exposure" not in self.history:
            return 0.0
        return float(self.history["exposure"].mean())

    @property
    def max_exposure(self) -> float:
        if self.history.empty or "exposure" not in self.history:
            return 0.0
        return float(self.history["exposure"].max())


class PortfolioBacktester:
    """Walk-forward across many symbols, sized by the live risk manager."""

    def __init__(
        self,
        symbols: list[str],
        primary: str | None = None,
        train_window: int = 252,
        test_window: int = 126,
        step_size: int = 126,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0005,
        commission_per_share: float = 0.0,
        hmm_config: dict | None = None,
        strategy_config: dict | None = None,
        risk_config: dict | None = None,
        reward_risk_ratio: float = 2.0,
        hmm_min_train_bars: int | None = None,
        **_ignored: Any,
    ) -> None:
        self.symbols = list(symbols)
        # The regime is a property of the market, not of each name, so it is
        # classified once on the primary symbol exactly as the live engine does.
        self.primary = primary or self.symbols[0]
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_per_share = commission_per_share
        self.hmm_config = dict(hmm_config or {})
        self.strategy_config = dict(strategy_config or {})
        self.risk_config = dict(risk_config or {})
        self.reward_risk_ratio = reward_risk_ratio

        floor = hmm_min_train_bars if hmm_min_train_bars is not None else train_window
        self.hmm_config["min_train_bars"] = floor

        self._lock_dir = tempfile.mkdtemp(prefix="pbt-halt-")
        self._lock_file = Path(self._lock_dir) / "trading_halted.lock"

    # -- windows ------------------------------------------------------------

    def build_windows(self, n_rows: int) -> list[Window]:
        windows: list[Window] = []
        start = 0
        while start + self.train_window + self.test_window <= n_rows:
            train_end = start + self.train_window
            windows.append(Window(
                index=len(windows), train_start=start, train_end=train_end,
                test_start=train_end, test_end=min(train_end + self.test_window, n_rows),
            ))
            start += self.step_size
        return windows

    # -- main ---------------------------------------------------------------

    def run(self, bars: dict[str, pd.DataFrame]) -> PortfolioBacktestResult:
        """Walk every fold. `bars` maps symbol to OHLCV indexed by timestamp."""
        if self.primary not in bars:
            raise ValueError(f"primary symbol {self.primary} missing from bars")

        primary_bars = bars[self.primary]
        features = build_feature_matrix(
            primary_bars,
            zscore_window=self.hmm_config.get("zscore_lookback", 252),
            clip=self.hmm_config.get("zscore_clip", 5.0),
        )
        returns = log_returns(primary_bars["close"], 1)

        needed = self.train_window + self.test_window
        if len(features) < needed:
            raise ValueError(
                f"{self.primary}: {len(features)} usable rows, need {needed}. "
                f"Feature warmup discards {required_warmup()} bars."
            )

        windows = self.build_windows(len(features))
        logger.info("%d symbols, %d usable rows, %d folds",
                    len(self.symbols), len(features), len(windows))

        cash = self.initial_capital
        holdings: dict[str, Holding] = {}
        rows: list[dict] = []
        trades: list[dict] = []
        summaries: list[dict] = []
        peak_equity = self.initial_capital

        for window in windows:
            cash, peak_equity, summary = self._run_window(
                window, features, returns, bars, cash, holdings,
                rows, trades, peak_equity,
            )
            summaries.append(summary)

        # Mark any position still open at the end, so its P&L is counted rather
        # than quietly dropped. An unclosed winner is how a backtest lies.
        if holdings and rows:
            last_time = rows[-1]["timestamp"]
            for symbol, holding in list(holdings.items()):
                price = self._price(bars, symbol, last_time, "close")
                if price is None:
                    continue
                cash += self._sell(holding, price)
                trades.append(self._trade_record(holding, last_time, price, "open at end"))
            holdings.clear()

        history = pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()
        equity = history["equity"] if not history.empty else pd.Series(dtype=float)

        return PortfolioBacktestResult(
            symbols=self.symbols,
            equity_curve=equity,
            trade_log=pd.DataFrame(trades),
            history=history,
            windows=windows,
            window_summaries=summaries,
            initial_capital=self.initial_capital,
            config={
                "symbols": self.symbols, "primary": self.primary,
                "train_window": self.train_window, "test_window": self.test_window,
                "slippage_pct": self.slippage_pct,
                "max_single_position": self.risk_config.get("max_single_position"),
                "max_concurrent": self.risk_config.get("max_concurrent"),
                "max_risk_per_trade": self.risk_config.get("max_risk_per_trade"),
            },
        )

    # -- one fold -----------------------------------------------------------

    def _run_window(self, window, features, returns, bars, cash, holdings,
                    rows, trades, peak_equity):
        train = features.iloc[window.train_start : window.train_end]
        test_index = features.index[window.test_start : window.test_end]

        engine = HMMEngine(**self.hmm_config)
        try:
            engine.fit(train, returns)
        except Exception as exc:
            logger.warning("fold %d: HMM fit failed (%s), holding", window.index, exc)
            return cash, peak_equity, {"window": window.index, "fitted": False,
                                       "error": str(exc)}

        orchestrator = StrategyOrchestrator(self.strategy_config, engine.regime_info)
        # An isolated lock file, never the live one.
        #
        # RiskManager defaults to <project>/trading_halted.lock. A backtest
        # sharing it can halt the real account, and a pre-existing live halt
        # would silently reject every signal in every backtest with no message
        # beyond a rejection count. Neither failure announces itself.
        risk = RiskManager(self.risk_config, lock_file=self._lock_file)

        stream = engine.stream()
        warmup = features.iloc[max(window.train_start, window.train_end - 60): window.train_end]
        stream.warm(warmup)

        pending: list[dict] = []
        week_key: tuple | None = None
        week_start = self._equity(cash, holdings, bars, test_index[0], "open")
        start_equity = week_start

        for timestamp in test_index:
            # 1. Stops and targets, on bars entered before today. A position
            #    opened at today's open is not checked today: within one bar
            #    the order of the high and the low is unknown.
            for symbol, holding in list(holdings.items()):
                if holding.bars_held < 1:
                    continue
                exit_price, reason = self._exit_price(bars, symbol, timestamp, holding)
                if exit_price is None:
                    continue
                cash += self._sell(holding, exit_price)
                trades.append(self._trade_record(holding, timestamp, exit_price, reason))
                del holdings[symbol]

            # 2. Execute yesterday's decisions at today's open.
            #
            # Portfolio limits are enforced incrementally here, not by the scan.
            # `candidates.scan` evaluates every symbol against one snapshot of
            # the account, so each candidate is sized as though it were the only
            # new position. Live, orders go one at a time and the state moves
            # underneath them. Filling the whole queue against the stale
            # snapshot took gross exposure to 165% at a 15% cap, sailing past
            # max_exposure 0.80 and exhausting cash to $102.
            max_concurrent = int(self.risk_config.get("max_concurrent", 12))
            max_exposure = float(self.risk_config.get("max_exposure", 0.80))
            equity_now = self._equity(cash, holdings, bars, timestamp, "open")
            invested_now = equity_now - cash

            for order in pending:
                symbol = order["symbol"]
                if symbol in holdings or order["shares"] <= 0:
                    continue
                if len(holdings) >= max_concurrent:
                    break
                price = self._price(bars, symbol, timestamp, "open")
                if price is None or price <= 0:
                    continue
                fill = price * (1 + self.slippage_pct)
                cost = order["shares"] * fill + self.commission_per_share * order["shares"]
                if cost > cash:
                    continue          # no margin in this model
                if equity_now > 0 and (invested_now + cost) / equity_now > max_exposure:
                    continue
                cash -= cost
                invested_now += cost
                holdings[symbol] = Holding(
                    symbol=symbol, shares=order["shares"], entry_price=fill,
                    entry_time=timestamp, stop_loss=order["stop_loss"],
                    take_profit=order["take_profit"], regime_at_entry=order["regime"],
                )
            pending = []

            for holding in holdings.values():
                holding.bars_held += 1

            # 3. Mark to market on today's close.
            equity = self._equity(cash, holdings, bars, timestamp, "close")
            peak_equity = max(peak_equity, equity)
            invested = sum(
                h.value_at(self._price(bars, s, timestamp, "close") or h.entry_price)
                for s, h in holdings.items()
            )
            exposure = invested / equity if equity > 0 else 0.0

            # 4. Decide, on data up to this close only.
            regime = stream.step(features.loc[timestamp])
            sliced = self._slice(bars, timestamp)
            # Daily and weekly baselines have to be real, or the breakers that
            # read them can never fire and the backtest is more permissive than
            # the live system it claims to model. On daily bars the previous
            # close is the day's opening baseline.
            day_start = rows[-1]["equity"] if rows else equity
            iso_week = (timestamp.isocalendar().year, timestamp.isocalendar().week)
            if iso_week != week_key:
                week_key = iso_week
                week_start = rows[-1]["equity"] if rows else equity

            state = PortfolioState(
                equity=equity, cash=cash, buying_power=cash,
                positions={s: {"quantity": h.shares, "entry_price": h.entry_price}
                           for s, h in holdings.items()},
                peak_equity=peak_equity, day_start_equity=day_start,
                week_start_equity=week_start, timestamp=timestamp,
                regime=regime.label.value, regime_confirmed=regime.is_confirmed,
                regime_confidence=regime.probability,
            )

            try:
                from core import candidates as candidates_module

                found = candidates_module.scan(
                    orchestrator, risk, list(sliced), sliced, regime, state,
                    self.strategy_config,
                )
            except Exception as exc:
                logger.debug("%s: scan failed (%s)", timestamp, exc)
                found = []

            for candidate in found:
                if not candidate.approved or candidate.action != "buy":
                    continue
                if candidate.symbol in holdings or candidate.shares <= 0:
                    continue
                stop = candidate.stop_loss
                target = None
                if stop is not None and candidate.entry_price > 0:
                    distance = candidate.entry_price - stop
                    if distance > 0:
                        target = candidate.entry_price + self.reward_risk_ratio * distance
                pending.append({
                    "symbol": candidate.symbol, "shares": float(candidate.shares),
                    "stop_loss": stop, "take_profit": target,
                    "regime": regime.label.value,
                })

            rows.append({
                "timestamp": timestamp, "equity": equity, "cash": cash,
                "exposure": exposure, "n_positions": len(holdings),
                "regime": regime.label.value, "confidence": regime.probability,
                "is_confirmed": regime.is_confirmed, "window": window.index,
            })

        end_equity = rows[-1]["equity"] if rows else start_equity
        return cash, peak_equity, {
            "window": window.index, "fitted": True, "n_states": engine.n_states,
            "test_start": test_index[0], "test_end": test_index[-1],
            "start_equity": start_equity, "end_equity": end_equity,
            "return_pct": (end_equity / start_equity - 1) if start_equity else 0.0,
        }

    # -- helpers ------------------------------------------------------------

    def _slice(self, bars: dict[str, pd.DataFrame], timestamp) -> dict[str, pd.DataFrame]:
        """Every symbol's history up to and including `timestamp`.

        The slice is what prevents look-ahead: the scan cannot see a bar that
        has not happened because the frame it is handed ends at today.
        """
        out = {}
        for symbol, frame in bars.items():
            part = frame.loc[:timestamp]
            if len(part) >= required_warmup() // 2:
                out[symbol] = part
        return out

    def _price(self, bars, symbol: str, timestamp, field_name: str) -> float | None:
        frame = bars.get(symbol)
        if frame is None or timestamp not in frame.index:
            return None
        value = frame.loc[timestamp, field_name]
        return float(value) if pd.notna(value) else None

    def _equity(self, cash: float, holdings: dict[str, Holding], bars, timestamp,
                field_name: str) -> float:
        total = cash
        for symbol, holding in holdings.items():
            price = self._price(bars, symbol, timestamp, field_name)
            total += holding.value_at(price if price else holding.entry_price)
        return total

    def _exit_price(self, bars, symbol: str, timestamp, holding: Holding):
        """Did this bar take the stop or the target, and at what price?

        A gap through the stop fills at the open, not at the stop. Modelling it
        the other way is the single most common way a backtest invents money
        that a real account never sees.
        """
        frame = bars.get(symbol)
        if frame is None or timestamp not in frame.index:
            return None, ""
        bar = frame.loc[timestamp]
        low, high, open_ = float(bar["low"]), float(bar["high"]), float(bar["open"])

        if holding.stop_loss is not None and low <= holding.stop_loss:
            fill = min(open_, holding.stop_loss)       # gap fills worse
            return fill * (1 - self.slippage_pct), "stop"
        if holding.take_profit is not None and high >= holding.take_profit:
            fill = max(open_, holding.take_profit)
            return fill * (1 - self.slippage_pct), "target"
        return None, ""

    def _sell(self, holding: Holding, price: float) -> float:
        return holding.shares * price - self.commission_per_share * holding.shares

    @staticmethod
    def _trade_record(holding: Holding, exit_time, exit_price: float, reason: str) -> dict:
        pnl = (exit_price - holding.entry_price) * holding.shares
        return {
            "symbol": holding.symbol, "entry_time": holding.entry_time,
            "exit_time": exit_time, "entry_price": holding.entry_price,
            "exit_price": exit_price, "shares": holding.shares,
            "pnl": pnl,
            "return_pct": (exit_price / holding.entry_price - 1) if holding.entry_price else 0.0,
            "bars_held": holding.bars_held, "exit_reason": reason,
            "regime_at_entry": holding.regime_at_entry,
        }
