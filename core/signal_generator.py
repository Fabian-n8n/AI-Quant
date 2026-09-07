"""
Combines the HMM regime and the allocation strategy into concrete signals.

Phase 3.

Most of what this file was scaffolded to do now lives in
`StrategyOrchestrator.generate_signals`, which is where the Phase 3 spec puts
it. What remains here is the join: classify the current bar with the HMM, hand
the regime to the orchestrator, and return the signals.

**`Signal` is defined once, in `core.regime_strategies`, and re-exported here.**
That matters more than it looks. The Phase 1 skeleton declared its own `Signal`
and `AllocationTarget`; leaving those in place would have given the codebase two
incompatible dataclasses with the same name, and `core.risk_manager` imports one
of them. Phase 5 would have been type-checking against a class the strategy
layer never produces.
"""

from __future__ import annotations

import logging

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import (  # re-exported: one definition, imported everywhere
    Direction,
    Signal,
    StrategyOrchestrator,
)

logger = logging.getLogger(__name__)

__all__ = ["Direction", "Signal", "SignalGenerator", "StrategyOrchestrator"]


class SignalGenerator:
    """Classify the bar, then allocate against it.

    Thin by design. The regime logic belongs to the HMM and the allocation logic
    belongs to the orchestrator; this only sequences them and keeps the
    look-ahead guarantee intact by passing `as_of` through rather than letting
    the orchestrator see the full history.
    """

    def __init__(self, hmm_engine: HMMEngine, orchestrator: StrategyOrchestrator) -> None:
        self.hmm_engine = hmm_engine
        self.orchestrator = orchestrator

    def generate(
        self,
        symbols: list[str],
        bars: dict[str, pd.DataFrame],
        features: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
    ) -> list[Signal]:
        """Signals for one bar.

            1. classify the regime using only data up to `as_of`
            2. look up the strategy for that regime, by volatility not label
            3. generate one signal per symbol
            4. halve sizes and drop leverage if the regime call is unreliable

        Phase 7 adds the risk manager between steps 3 and 4 becoming orders.
        Nothing here may place a trade.
        """
        regime_state = self.hmm_engine.classify(features, as_of=as_of)
        return self.orchestrator.generate_signals(
            symbols=symbols,
            bars={s: self._truncate(b, as_of) for s, b in bars.items()},
            regime_state=regime_state,
            is_flickering=regime_state.is_flickering,
        )

    def refresh_after_retrain(self) -> None:
        """Rebuild the regime-to-strategy mapping after an HMM refit.

        EM renumbers its states on every refit, so a mapping built against the
        previous fit points at the wrong regimes. Forgetting this is silent: the
        system keeps trading, just with the strategies wired to the wrong states.
        """
        self.orchestrator.update_regime_infos(self.hmm_engine.regime_info)

    @staticmethod
    def _truncate(bars: pd.DataFrame, as_of: pd.Timestamp | None) -> pd.DataFrame:
        """Cut price history at `as_of`.

        A caller will eventually pass the full frame by mistake. Truncating here
        means the stop and trend calculations cannot see bars that had not
        happened yet.
        """
        return bars if as_of is None else bars.loc[:as_of]
