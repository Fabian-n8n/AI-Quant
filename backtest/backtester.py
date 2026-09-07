"""
Walk-forward allocation backtester.

Phase 4. The phase that decides whether anything built so far is real.

WHAT THIS IS
------------
An **allocation-based** backtester. It does not track individual trade entries
and exits. Each bar it sets a target portfolio allocation from the detected
volatility regime, and rebalances when that target drifts meaningfully from the
current position. That is how systematic allocation strategies actually work,
and it is what the Phase 3 layer produces.

Consequently a "trade" here is a **rebalance event**, not an entry/exit pair.
Its P&L is the equity change from one rebalance to the next. Win rate and
holding period read against that definition, not against round trips.

WHY WALK-FORWARD
----------------
The usual backtest fits parameters on all history then reports how they did on
that same history: the strategy grading its own homework with the answer sheet
open. Here:

    Train the HMM on 252 bars. Freeze it.
    Run the next 126 bars with that frozen model. Record.
    Slide forward 126 bars and repeat.

Every OOS bar is classified by a model that has never seen it. The stitched OOS
results are the only ones that mean anything.

The rule that keeps it honest: look at in-sample results as much as you like.
The moment you change a parameter after seeing an out-of-sample result, that
result is no longer out-of-sample. Log every variant in docs/EXPERIMENT-LOG.md.

TWO THINGS THE SPEC DOES NOT MENTION
------------------------------------
**1. The IS window is below the HMM's own minimum.** Phase 4 specifies a 252-bar
in-sample window; Phase 2 specifies `min_train_bars: 504`. `fit()` would raise on
every fold. The backtester therefore overrides the minimum to the IS window size.
That is the spec-faithful choice, but a 252-bar fit is thin: BIC's penalty spread
(492 at n=3 versus 1310 at n=7) pushes selection toward fewer states than a
longer window would choose. `backtest.hmm_min_train_bars` controls it.

**2. Features are computed once on the full series, then sliced.** This is safe
only because the Phase 2 feature layer is causal: every window is trailing and
standardisation is a rolling z-score, so row t of a full-series computation is
bit-identical to row t of any prefix computation. That is asserted by
`test_appending_future_bars_changes_nothing`. Recomputing per window would
discard the 450-bar warmup on every fold and change nothing else.

NO RISK MANAGER HERE
--------------------
Per the spec there are no per-trade stops in the backtester; stops are a live
trading concern. The consequence is that this measures the **unclamped** Phase 3
allocations. Phase 5's risk manager will cap gross exposure at
`risk.max_exposure`, and low-vol targets 118.75%, so live behaviour will differ
from these results until that conflict is settled. See docs/PHASE3-NOTES.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from data.feature_engineering import (
    build_feature_matrix,
    ema,
    log_returns,
    required_warmup,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Window:
    """One walk-forward train/test split, in positional index terms."""
    index: int
    train_start: int
    train_end: int      # exclusive
    test_start: int
    test_end: int       # exclusive

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@dataclass
class BacktestResult:
    """Everything the walk-forward produced.

    `equity_curve` and `regime_history` are pure out-of-sample: no in-sample bar
    ever appears. `is_significant` is False under 30 trades, in which case the
    metrics should be reported as "not enough data" rather than as numbers.
    """
    symbol: str
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    buy_and_hold: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
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
    def n_folds(self) -> int:
        return len(self.windows)


class WalkForwardBacktester:
    """Trains per window, walks the out-of-sample bars, marks to market."""

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 126,
        step_size: int = 126,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0005,
        commission_per_share: float = 0.0,
        rebalance_threshold: float = 0.10,
        hmm_config: dict | None = None,
        strategy_config: dict | None = None,
        hmm_min_train_bars: int | None = None,
        **_ignored: Any,
    ) -> None:
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_per_share = commission_per_share
        self.rebalance_threshold = rebalance_threshold
        self.hmm_config = dict(hmm_config or {})
        self.strategy_config = dict(strategy_config or {})

        # Phase 4's IS window is below Phase 2's min_train_bars, so every fit
        # would raise. Override to the window size and say so once.
        floor = hmm_min_train_bars if hmm_min_train_bars is not None else train_window
        if self.hmm_config.get("min_train_bars", 0) > floor:
            logger.info(
                "Lowering hmm.min_train_bars %d -> %d to match the %d-bar IS window. "
                "A fit this short biases BIC toward fewer states.",
                self.hmm_config["min_train_bars"], floor, train_window,
            )
        self.hmm_config["min_train_bars"] = floor

    # -- windows ------------------------------------------------------------

    def build_windows(self, n_rows: int) -> list[Window]:
        """Slice usable feature rows into rolling train/test splits.

        Train and test never overlap: `test_start == train_end`. Successive test
        windows are contiguous when `step_size == test_window`, which is what
        makes the stitched OOS curve a continuous equity series rather than a
        set of disconnected fragments.
        """
        windows: list[Window] = []
        start = 0
        while start + self.train_window + self.test_window <= n_rows:
            train_end = start + self.train_window
            windows.append(
                Window(
                    index=len(windows),
                    train_start=start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=min(train_end + self.test_window, n_rows),
                )
            )
            start += self.step_size
        return windows

    # -- main loop ----------------------------------------------------------

    def run(self, bars: pd.DataFrame, symbol: str = "ASSET") -> BacktestResult:
        """Run the full walk-forward on one symbol's OHLCV.

        Needs `train_window + test_window + 450` raw bars minimum, because the
        feature warmup discards 450 before the first usable row exists.
        """
        features = build_feature_matrix(
            bars,
            zscore_window=self.hmm_config.get("zscore_lookback", 252),
            clip=self.hmm_config.get("zscore_clip", 5.0),
        )
        returns = log_returns(bars["close"], 1)

        # Precomputed once over the full series. Both are causal (trailing
        # windows), so a value at t is identical to computing it on the prefix
        # ending at t. Recomputing per bar would make the loop O(T^2).
        ema50 = ema(bars["close"], self.strategy_config.get("ema_span", 50))

        needed = self.train_window + self.test_window
        if len(features) < needed:
            raise ValueError(
                f"{symbol}: {len(features)} usable feature rows, need at least {needed}. "
                f"Feature warmup discards {required_warmup()} bars, so supply at least "
                f"{needed + required_warmup()} raw bars."
            )

        windows = self.build_windows(len(features))
        logger.info(
            "%s: %d raw bars -> %d usable rows -> %d folds (%d OOS bars)",
            symbol, len(bars), len(features), len(windows),
            sum(w.test_size for w in windows),
        )

        state = _PortfolioState(cash=self.initial_capital)
        rows: list[dict] = []
        trades: list[dict] = []
        summaries: list[dict] = []

        for window in windows:
            summary = self._run_window(
                window, features, bars, returns, ema50, state, rows, trades, symbol
            )
            summaries.append(summary)

        # Close the final open trade at the last mark so its P&L is counted.
        if trades and trades[-1].get("exit_equity") is None and rows:
            _close_trade(trades[-1], rows[-1]["timestamp"], rows[-1]["equity"], len(rows) - 1)

        history = pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()
        equity = history["equity"] if not history.empty else pd.Series(dtype=float)

        return BacktestResult(
            symbol=symbol,
            equity_curve=equity,
            trade_log=pd.DataFrame(trades),
            regime_history=history,
            buy_and_hold=self._buy_and_hold(bars, equity),
            windows=windows,
            window_summaries=summaries,
            initial_capital=self.initial_capital,
            config={
                "train_window": self.train_window,
                "test_window": self.test_window,
                "step_size": self.step_size,
                "slippage_pct": self.slippage_pct,
                "rebalance_threshold": self.rebalance_threshold,
                "initial_capital": self.initial_capital,
            },
        )

    def _run_window(
        self,
        window: Window,
        features: pd.DataFrame,
        bars: pd.DataFrame,
        returns: pd.Series,
        ema50: pd.Series,
        state: _PortfolioState,
        rows: list[dict],
        trades: list[dict],
        symbol: str,
    ) -> dict:
        """Train on the in-sample slice, then walk the out-of-sample bars."""
        train = features.iloc[window.train_start : window.train_end]
        test_index = features.index[window.test_start : window.test_end]

        engine = HMMEngine(**self.hmm_config)
        try:
            engine.fit(train, returns)
        except Exception as exc:
            logger.warning("fold %d: HMM fit failed (%s), holding position", window.index, exc)
            return {"window": window.index, "fitted": False, "error": str(exc)}

        orchestrator = StrategyOrchestrator(self.strategy_config, engine.regime_info)

        # Stream rather than re-running the forward pass per bar. Warmed on the
        # tail of the training window so the filtered distribution and the
        # stability tracker have settled before the first OOS bar, instead of
        # starting cold from startprob_ exactly where results begin to count.
        stream = engine.stream()
        warmup = features.iloc[max(window.train_start, window.train_end - 60) : window.train_end]
        stream.warm(warmup)

        pending_target: float | None = None
        pending_context: dict[str, Any] = {}
        start_equity = state.equity_at(float(bars.loc[test_index[0], "open"]))

        for timestamp in test_index:
            bar = bars.loc[timestamp]
            open_price = float(bar["open"])
            close_price = float(bar["close"])

            # 1. Execute the previous bar's decision at THIS bar's open.
            #    The one-bar delay is not a detail: a target computed from a
            #    bar's close cannot be filled at that close, and pretending
            #    otherwise flatters every result.
            if pending_target is not None:
                filled = state.rebalance(
                    pending_target, open_price, self.slippage_pct, self.commission_per_share
                )
                if filled is not None:
                    if trades and trades[-1].get("exit_equity") is None:
                        _close_trade(trades[-1], timestamp, filled["equity"], len(rows))
                    trades.append(
                        {
                            "entry_time": timestamp,
                            "entry_equity": filled["equity"],
                            "entry_price": filled["fill_price"],
                            "allocation": pending_target,
                            "shares": filled["shares"],
                            "delta_shares": filled["delta"],
                            "slippage_cost": filled["slippage_cost"],
                            "window": window.index,
                            **pending_context,
                            "exit_time": None,
                            "exit_equity": None,
                            "pnl": None,
                            "return_pct": None,
                            "bars_held": None,
                        }
                    )
                pending_target = None

            # 2. Classify and decide, using only data up to this bar's close.
            regime = stream.step(features.loc[timestamp])
            target = orchestrator.target_allocation(
                regime, close_price, float(ema50.loc[timestamp])
            )
            equity = state.equity_at(close_price)
            current_allocation = (state.shares * close_price / equity) if equity > 0 else 0.0

            if abs(target - current_allocation) > self.rebalance_threshold:
                pending_target = target
                pending_context = {
                    "regime_id": regime.state_id,
                    "regime": regime.label.value,
                    "confidence": regime.probability,
                    "is_confirmed": regime.is_confirmed,
                }

            # 3. Mark to market.
            rows.append(
                {
                    "timestamp": timestamp,
                    "equity": equity,
                    "cash": state.cash,
                    "shares": state.shares,
                    "price": close_price,
                    "allocation": current_allocation,
                    "target_allocation": target,
                    "regime_id": regime.state_id,
                    "regime": regime.label.value,
                    "confidence": regime.probability,
                    "is_confirmed": regime.is_confirmed,
                    "volatility_rank": orchestrator.get_volatility_rank(regime.state_id).value,
                    "window": window.index,
                }
            )

        end_equity = rows[-1]["equity"] if rows else start_equity
        return {
            "window": window.index,
            "fitted": True,
            "n_states": engine.n_states,
            "bic": engine.metadata.bic,
            "train_start": features.index[window.train_start],
            "train_end": features.index[window.train_end - 1],
            "test_start": test_index[0],
            "test_end": test_index[-1],
            "start_equity": start_equity,
            "end_equity": end_equity,
            "return_pct": (end_equity / start_equity - 1) if start_equity else 0.0,
        }

    @staticmethod
    def _buy_and_hold(bars: pd.DataFrame, equity: pd.Series) -> pd.Series:
        """Buy-and-hold over the same OOS span, from the same starting capital."""
        if equity.empty:
            return pd.Series(dtype=float)
        prices = bars.loc[equity.index, "close"]
        return equity.iloc[0] * prices / prices.iloc[0]


class _PortfolioState:
    """Cash and shares. The allocation math lives here and nowhere else."""

    def __init__(self, cash: float) -> None:
        self.cash = cash
        self.shares = 0

    def equity_at(self, price: float) -> float:
        """equity = cash + shares * price.

        Correct even when cash is negative. Negative cash is margin debt, and
        the share value exceeds it, so equity stays positive and meaningful.
        """
        return self.cash + self.shares * price

    def rebalance(
        self, target_allocation: float, price: float, slippage_pct: float, commission: float
    ) -> dict | None:
        """Move to the target allocation at `price`, with slippage.

            equity        = cash + shares * price
            target_shares = int(equity * target_allocation / price)
            delta         = target_shares - current_shares
            cash         -= delta * fill_price
            shares        = target_shares

        `int()` truncates, so fractional shares are dropped rather than rounded.
        That is deliberate: it can only ever under-allocate, never over.

        When `target_allocation` exceeds 1.0 the position costs more than
        equity, so cash goes negative. That is margin and it is left alone;
        clamping it to zero would silently cap leverage at 1.0 and the backtest
        would stop measuring the strategy that was configured.
        """
        equity = self.equity_at(price)
        if equity <= 0 or price <= 0:
            return None

        target_shares = int(equity * target_allocation / price)
        delta = target_shares - self.shares
        if delta == 0:
            return None

        # Slippage always works against the trade: pay up to buy, down to sell.
        fill_price = price * (1 + slippage_pct) if delta > 0 else price * (1 - slippage_pct)
        slippage_cost = abs(delta) * abs(fill_price - price)

        self.cash -= delta * fill_price
        self.cash -= abs(delta) * commission
        self.shares = target_shares

        return {
            "equity": self.equity_at(price),
            "fill_price": fill_price,
            "shares": self.shares,
            "delta": delta,
            "slippage_cost": slippage_cost,
        }


def _close_trade(trade: dict, exit_time, exit_equity: float, exit_row: int) -> None:
    """Close a rebalance-to-rebalance holding period.

    A "trade" spans one rebalance to the next, so its P&L is the equity change
    across the period the allocation was held. There is no entry/exit pair to
    difference: the position was never fully closed, only resized.
    """
    trade["exit_time"] = exit_time
    trade["exit_equity"] = exit_equity
    trade["pnl"] = exit_equity - trade["entry_equity"]
    trade["return_pct"] = (
        exit_equity / trade["entry_equity"] - 1 if trade["entry_equity"] else 0.0
    )
    if isinstance(exit_time, pd.Timestamp) and isinstance(trade["entry_time"], pd.Timestamp):
        trade["bars_held"] = int(np.busday_count(
            trade["entry_time"].date(), exit_time.date()
        ))
