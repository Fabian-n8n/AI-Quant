"""
Hidden Markov Model regime detection. The brain of the system.

Phase 2.

DESIGN PHILOSOPHY
-----------------
The HMM is a **volatility classifier**. It detects whether the market is in a
calm, moderate or turbulent volatility environment. It does NOT predict price
direction, and nothing downstream should treat it as if it did. The strategy
layer uses the volatility classification to set portfolio allocation: fully
invested when conditions are calm, reduced when turbulent.

Regime *labels* (crash, bear, bull, euphoria) are assigned by sorting states on
mean return, purely so a human reading the dashboard can tell them apart. The
strategy layer ranks the same states by **volatility**, independently. The
labels are cosmetic; the volatility rank is what drives allocation. Those two
orderings genuinely disagree: crash and euphoria sit at opposite ends of the
return sort and next to each other on the volatility sort.

THE CRITICAL DETAIL: NO LOOK-AHEAD BIAS
---------------------------------------
`model.predict()` is never called anywhere in this file, and must never be.
hmmlearn's `predict()` runs Viterbi, which processes the entire observation
sequence and revises earlier states in light of later ones. A classification
for March would be informed by June. That is look-ahead bias, and it produces a
backtest that looks extraordinary and fails completely live.

Everything here uses the **forward algorithm** instead, computing
P(state_t | observations_1..t): filtered inference, past and present only. It is
implemented explicitly below rather than borrowed from hmmlearn, because the
distinction between the filtered and smoothed posterior is invisible at the call
site and only one of them is correct.

`tests/test_look_ahead.py` fails if this regresses.

MODEL CAPACITY, READ BEFORE CHANGING FEATURES
---------------------------------------------
A full-covariance Gaussian HMM is parameter-hungry. With d features and n states
it fits n*d*(d+1)/2 covariance terms alone. Against the spec's defaults, d=14
and 504 training rows:

    n=3   365 params    n=4   491 params    n=5   619 params
    n=6   749 params    n=7   881 params

Five states and up have more parameters than data points, so their covariance
matrices are singular and the fit is meaningless. Worse, BIC's penalty term
(n_params * log 504 = 6.22 per parameter) is large enough that it will select
n=3 every single time regardless of what the data looks like, so the automatic
model selection stops selecting anything.

`check_fittability()` catches this at fit time with an explicit message rather
than letting it surface as a LinAlgError or, worse, a plausible-looking result.
Three ways out, in order of preference:

1. Narrow the HMM's inputs to the volatility signal. `VOLATILITY_FEATURES` is
   six columns, which at n=7 costs 237 parameters. This suits the design
   philosophy above: a volatility classifier does not need momentum and mean
   reversion features to do its job.
2. Use `covariance_type="diag"`, which drops the covariance cost from
   n*d*(d+1)/2 to n*d.
3. Train on more data. The expanding window makes this free over time.

Set `hmm.feature_columns` in settings.yaml. See docs/PHASE2-NOTES.md.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from data.feature_engineering import (  # noqa: F401  (re-exported for callers)
    FEATURE_COLUMNS,
    VOLATILITY_FEATURES,
    build_feature_matrix,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    """Regime labels, ordered worst to best by mean return.

    Which subset is in play depends on how many states BIC selected. Labels are
    for human readability only. See the module docstring.
    """
    CRASH = "crash"
    STRONG_BEAR = "strong_bear"
    BEAR = "bear"
    WEAK_BEAR = "weak_bear"
    NEUTRAL = "neutral"
    WEAK_BULL = "weak_bull"
    BULL = "bull"
    STRONG_BULL = "strong_bull"
    EUPHORIA = "euphoria"
    UNKNOWN = "unknown"      # not fitted, insufficient history, or below confidence


#: Label sets by selected state count, sorted by mean return ascending.
REGIME_LABEL_SETS: dict[int, list[Regime]] = {
    3: [Regime.BEAR, Regime.NEUTRAL, Regime.BULL],
    4: [Regime.CRASH, Regime.BEAR, Regime.BULL, Regime.EUPHORIA],
    5: [Regime.CRASH, Regime.BEAR, Regime.NEUTRAL, Regime.BULL, Regime.EUPHORIA],
    6: [
        Regime.CRASH, Regime.STRONG_BEAR, Regime.WEAK_BEAR,
        Regime.WEAK_BULL, Regime.STRONG_BULL, Regime.EUPHORIA,
    ],
    7: [
        Regime.CRASH, Regime.STRONG_BEAR, Regime.WEAK_BEAR, Regime.NEUTRAL,
        Regime.WEAK_BULL, Regime.STRONG_BULL, Regime.EUPHORIA,
    ],
}


class VolatilityRank(str, Enum):
    """What the strategy layer actually keys off. Independent of the label."""
    LOW = "low"
    MID = "mid"
    HIGH = "high"


#: Boundaries for the volatility-rank split, from the Phase 3 spec.
VOL_RANK_LOW_MAX = 0.33
VOL_RANK_HIGH_MIN = 0.67


def volatility_rank_from_position(position: float) -> VolatilityRank:
    """Map a normalised volatility position in [0, 1] onto a strategy tier.

        position <= 0.33  -> LOW
        position >= 0.67  -> HIGH
        otherwise         -> MID

    Defined once and used by both the HMM (to populate RegimeInfo) and the
    strategy orchestrator. Two implementations would eventually disagree about
    which regime is "low volatility", and the dashboard and the allocator would
    silently show different answers.

    The literal 0.33 / 0.67 boundaries are the spec's, and they are not the same
    as exact thirds. At n=7 a rank of 2 gives 2/6 = 0.3333, which is above 0.33,
    so it lands in MID rather than LOW. `assign_volatility_ranks` pins the exact
    partition for every state count so this stays visible rather than surprising.
    """
    if position <= VOL_RANK_LOW_MAX:
        return VolatilityRank.LOW
    if position >= VOL_RANK_HIGH_MIN:
        return VolatilityRank.HIGH
    return VolatilityRank.MID


def assign_volatility_ranks(volatility_by_state: dict[int, float]) -> dict[int, VolatilityRank]:
    """Rank states by volatility ascending, then map each onto a tier.

        position = rank / (n_states - 1)

    Resulting partitions, worth knowing because they are not even thirds:

        n=3  low 1, mid 1, high 1
        n=4  low 1, mid 2, high 1
        n=5  low 2, mid 1, high 2
        n=6  low 2, mid 2, high 2
        n=7  low 2, mid 3, high 2
    """
    ordered = sorted(volatility_by_state, key=lambda s: volatility_by_state[s])
    n = len(ordered)
    if n == 1:
        return {ordered[0]: VolatilityRank.MID}
    return {
        state: volatility_rank_from_position(rank / (n - 1))
        for rank, state in enumerate(ordered)
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeInfo:
    """Static description of one fitted state.

    `expected_return` and `expected_volatility` are computed from the **raw**
    returns of the bars assigned to this state, not from the model's `means_`.
    The means are in standardised z-score space, so reading an expected return
    off them would give a unitless number that looks like a percentage and is
    not one.
    """
    regime_id: int
    regime_name: str
    expected_return: float          # annualised, from raw returns
    expected_volatility: float      # annualised, from raw returns
    volatility_rank: VolatilityRank
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float
    n_observations: int = 0
    frequency: float = 0.0          # share of training bars in this state


@dataclass(frozen=True)
class RegimeState:
    """One classification, for one bar.

    `label` is the regime the system should ACT on, which during an unconfirmed
    transition is still the previous one. `raw_label` is what the model said
    this bar. They differ exactly while a change is pending confirmation, and
    conflating them is how the stability filter gets accidentally disabled.
    """
    label: Regime
    state_id: int
    probability: float
    state_probabilities: dict[Regime, float]
    timestamp: pd.Timestamp
    is_confirmed: bool
    consecutive_bars: int
    raw_label: Regime = Regime.UNKNOWN
    raw_state_id: int = -1
    is_flickering: bool = False
    flicker_rate: int = 0
    size_multiplier: float = 1.0
    meets_confidence: bool = True


@dataclass
class ModelMetadata:
    """Everything needed to know what a pickled model is and whether to trust it."""
    n_regimes: int
    bic: float
    log_likelihood: float
    all_bic_scores: dict[int, float] = field(default_factory=dict)
    training_date: Optional[datetime] = None
    train_start: Optional[pd.Timestamp] = None
    train_end: Optional[pd.Timestamp] = None
    n_train_samples: int = 0
    feature_columns: list[str] = field(default_factory=list)
    labels: dict[int, str] = field(default_factory=dict)
    converged: bool = False
    n_iterations: int = 0
    covariance_type: str = "full"
    n_parameters: int = 0
    random_state: int = 42


# ---------------------------------------------------------------------------
# Fittability
# ---------------------------------------------------------------------------

def count_parameters(n_states: int, n_features: int, covariance_type: str = "full") -> int:
    """Free parameters in a Gaussian HMM. Used for BIC and the capacity check.

        startprob   n - 1              (constrained to sum to 1)
        transmat    n * (n - 1)        (each row sums to 1)
        means       n * d
        covars      n * d * (d+1) / 2  (full)   or   n * d  (diag)
    """
    start = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    if covariance_type == "full":
        covars = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        covars = n_states * n_features
    elif covariance_type == "spherical":
        covars = n_states
    elif covariance_type == "tied":
        covars = n_features * (n_features + 1) // 2
    else:
        raise ValueError(f"unsupported covariance_type: {covariance_type}")
    return start + trans + means + covars


class InsufficientDataError(ValueError):
    """Raised when the model is too large for the data, with the fix spelled out."""


def check_fittability(
    n_samples: int,
    n_features: int,
    n_candidates: Iterable[int],
    covariance_type: str = "full",
    min_samples_per_param: float = 1.0,
    strict: bool = True,
) -> dict[int, int]:
    """Verify each candidate state count can actually be fitted.

    Returns {n_states: n_params}. Raises InsufficientDataError when the largest
    candidate is overparameterised and `strict`, because the alternative is a
    singular covariance matrix surfacing as an opaque linear algebra error, or
    a fit that converges to confident nonsense.
    """
    counts = {
        n: count_parameters(n, n_features, covariance_type) for n in n_candidates
    }
    over = {n: p for n, p in counts.items() if p > n_samples * min_samples_per_param}
    if over and strict:
        worst = max(over.items(), key=lambda kv: kv[1])
        raise InsufficientDataError(
            f"Model too large for the data. With {n_features} features and "
            f"covariance_type='{covariance_type}', candidates {sorted(over)} need "
            f"{sorted(over.values())} parameters against only {n_samples} training "
            f"samples (worst: n={worst[0]} needs {worst[1]}).\n"
            f"Covariance matrices will be singular and BIC will collapse onto the "
            f"smallest candidate regardless of the data.\n"
            f"Fixes, in order of preference:\n"
            f"  1. Set hmm.feature_columns to the 6-column VOLATILITY_FEATURES set.\n"
            f"  2. Set hmm.covariance_type to 'diag'.\n"
            f"  3. Train on more bars (min_train_bars, or a longer history).\n"
            f"See docs/PHASE2-NOTES.md."
        )
    if over:
        logger.warning(
            "Overparameterised candidates %s will likely fail or overfit "
            "(%d training samples)", sorted(over), n_samples,
        )
    return counts


def resolve_feature_columns(spec: Optional[list[str] | str]) -> list[str]:
    """Turn a settings.yaml value into an explicit column list.

    Accepts the aliases `all` and `volatility` as well as an explicit list, so
    the config stays readable and a typo becomes an error here rather than a
    silently different model.
    """
    if spec is None or spec == "all":
        return list(FEATURE_COLUMNS)
    if spec == "volatility":
        return list(VOLATILITY_FEATURES)
    if isinstance(spec, str):
        raise ValueError(
            f"unknown feature_columns alias {spec!r}. Use 'all', 'volatility', "
            f"or an explicit list from {FEATURE_COLUMNS}"
        )
    unknown = set(spec) - set(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"unknown feature columns: {sorted(unknown)}")
    if not spec:
        raise ValueError("feature_columns cannot be empty")
    return list(spec)


# ---------------------------------------------------------------------------
# Forward filter
# ---------------------------------------------------------------------------

class ForwardFilter:
    """Incremental forward algorithm with a cached alpha vector.

    The live loop classifies one new bar per day. Rerunning the forward pass
    over the entire history each time is O(T) work for one bar of information,
    so this caches the previous log-alpha and advances by a single step.

    Correctness requirement: stepping bar by bar must produce exactly the same
    numbers as running the batch pass over the whole sequence. That is asserted
    in tests/test_look_ahead.py, because a divergence here would mean live
    trading and backtesting disagree about what regime it is.

    Log-alpha is renormalised at every step and the normalising constant is
    accumulated into `log_likelihood`. That keeps the vector bounded no matter
    how long the sequence runs, which plain log-space accumulation does not.
    """

    def __init__(
        self,
        log_startprob: np.ndarray,
        log_transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        covariance_type: str = "full",
    ) -> None:
        self.log_startprob = log_startprob
        self.log_transmat = log_transmat
        self.means = means
        self.covars = covars
        self.covariance_type = covariance_type
        self._chol_cache = _prepare_emission(means, covars, covariance_type)
        self.reset()

    def reset(self) -> None:
        self.log_alpha: Optional[np.ndarray] = None
        self.log_likelihood: float = 0.0
        self.n_steps: int = 0

    def step(self, observation: np.ndarray) -> np.ndarray:
        """Advance one bar. Returns P(state | observations up to and including it).

        alpha_0 = startprob * emission(obs_0)
        alpha_t = (alpha_{t-1} @ transmat) * emission(obs_t)

        both in log space, renormalised each step.
        """
        log_emission = _log_emission_single(observation, self._chol_cache)

        if self.log_alpha is None:
            log_alpha = self.log_startprob + log_emission
        else:
            log_alpha = logsumexp(
                self.log_alpha[:, None] + self.log_transmat, axis=0
            ) + log_emission

        norm = logsumexp(log_alpha)
        self.log_alpha = log_alpha - norm
        self.log_likelihood += float(norm)
        self.n_steps += 1
        return np.exp(self.log_alpha)

    def run(self, observations: np.ndarray) -> np.ndarray:
        """Filtered posteriors for a whole sequence, shape (T, n_states).

        Row t is P(state_t | obs_0..t). Row t does not depend on any row after
        it, which is the property the whole design exists to guarantee.
        """
        self.reset()
        return np.vstack([self.step(obs) for obs in observations])


def _prepare_emission(means: np.ndarray, covars: np.ndarray, covariance_type: str) -> dict:
    """Precompute Cholesky factors and log-determinants once per model.

    Gaussian log density needs Sigma^-1 and log|Sigma|. Inverting per bar is
    both slow and numerically worse than solving against a Cholesky factor, and
    this runs once per bar of every backtest.
    """
    n_states, n_features = means.shape
    full = _as_full_covariance(covars, covariance_type, n_states, n_features)

    chols, log_dets = [], []
    for k in range(n_states):
        sigma = full[k]
        try:
            chol = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            # Singular or near-singular. Nudge the diagonal rather than crash:
            # the fittability check should have caught the cause upstream, and
            # failing here would lose the more useful error message.
            jitter = 1e-6 * np.trace(sigma) / n_features
            chol = np.linalg.cholesky(sigma + jitter * np.eye(n_features))
            logger.warning("State %d covariance was singular; applied jitter %.3g", k, jitter)
        chols.append(chol)
        log_dets.append(2.0 * np.sum(np.log(np.diag(chol))))

    return {
        "means": means,
        "chols": np.array(chols),
        "log_dets": np.array(log_dets),
        "const": -0.5 * n_features * np.log(2 * np.pi),
        "n_states": n_states,
        "n_features": n_features,
    }


def _log_emission_single(observation: np.ndarray, cache: dict) -> np.ndarray:
    """log N(x | mu_k, Sigma_k) for every state k, for one observation.

        log N = -0.5 * [ d*log(2*pi) + log|Sigma| + (x-mu)' Sigma^-1 (x-mu) ]

    The quadratic form is evaluated as ||L^-1 (x-mu)||^2 via a triangular solve,
    which avoids forming Sigma^-1 at all.
    """
    diff = observation[None, :] - cache["means"]            # (n_states, d)
    out = np.empty(cache["n_states"])
    for k in range(cache["n_states"]):
        solved = np.linalg.solve(cache["chols"][k], diff[k])
        out[k] = cache["const"] - 0.5 * (cache["log_dets"][k] + solved @ solved)
    return out


def _as_full_covariance(
    covars: np.ndarray, covariance_type: str, n_states: int, n_features: int
) -> np.ndarray:
    """Normalise hmmlearn's covariance shapes to (n_states, d, d)."""
    covars = np.asarray(covars)
    if covariance_type == "full":
        return covars
    if covariance_type == "diag":
        return np.array([np.diag(covars[k]) for k in range(n_states)])
    if covariance_type == "spherical":
        return np.array([np.eye(n_features) * float(np.ravel(covars[k])[0]) for k in range(n_states)])
    if covariance_type == "tied":
        return np.array([covars for _ in range(n_states)])
    raise ValueError(f"unsupported covariance_type: {covariance_type}")


# ---------------------------------------------------------------------------
# Stability filter
# ---------------------------------------------------------------------------

class RegimeTracker:
    """Turns the model's raw per-bar output into a regime the system acts on.

    The model's argmax flips around on individual bars. Trading every flip means
    rebalancing on noise. This is the filter that sits between them, and it is
    deliberately separate from the model so the look-ahead guarantee stays a
    property of pure functions rather than of a stateful object.

    Two independent mechanisms, often confused:

    - **Transition damping.** A new regime is not acted on until it has held for
      `stability_bars` consecutive bars. Until then the previously confirmed
      regime stands and sizes are cut by 25% (`transition_size_mult`). The
      system is saying "something is changing, be smaller while I find out".

    - **Uncertainty mode.** Independently, if the raw regime changed more than
      `flicker_threshold` times in the last `flicker_window` bars, the model is
      effectively saying it does not know. Sizes are cut by
      `uncertainty_size_mult`, a deeper reduction.

    Flicker counts **raw** changes, not confirmed ones. Counting confirmed
    changes would be circular: the transition damper exists to suppress exactly
    the flips that flicker detection needs to see, so the detector would almost
    never fire.

    When both apply the smaller multiplier wins.
    """

    def __init__(
        self,
        stability_bars: int = 3,
        flicker_window: int = 20,
        flicker_threshold: int = 4,
        transition_size_mult: float = 0.75,
        uncertainty_size_mult: float = 0.50,
    ) -> None:
        self.stability_bars = stability_bars
        self.flicker_window = flicker_window
        self.flicker_threshold = flicker_threshold
        self.transition_size_mult = transition_size_mult
        self.uncertainty_size_mult = uncertainty_size_mult
        self.reset()

    def reset(self) -> None:
        self.confirmed: Optional[int] = None
        self.candidate: Optional[int] = None
        self.candidate_bars: int = 0
        self.consecutive_bars: int = 0
        self._raw_history: list[int] = []

    def update(self, raw_state: int) -> dict[str, Any]:
        """Consume one bar's raw state, return the acted-on regime and sizing.

        Returns keys: state_id, is_confirmed, consecutive_bars, in_transition,
        is_flickering, flicker_rate, size_multiplier, changed.
        """
        self._raw_history.append(raw_state)
        if len(self._raw_history) > self.flicker_window:
            self._raw_history.pop(0)

        changed = False

        if self.confirmed is None:
            # First bar: adopt immediately. There is no previous regime to hold,
            # and refusing to classify for the first three bars would just push
            # the problem into the caller.
            self.confirmed = raw_state
            self.candidate = raw_state
            self.candidate_bars = 1
            self.consecutive_bars = 1
        elif raw_state == self.confirmed:
            self.candidate = raw_state
            self.candidate_bars += 1
            self.consecutive_bars += 1
        else:
            if raw_state == self.candidate:
                self.candidate_bars += 1
            else:
                self.candidate = raw_state
                self.candidate_bars = 1
            self.consecutive_bars = 0

            if self.candidate_bars >= self.stability_bars:
                previous = self.confirmed
                self.confirmed = raw_state
                self.consecutive_bars = self.candidate_bars
                changed = True
                logger.warning(
                    "Regime change CONFIRMED: state %s -> %s after %d consecutive bars",
                    previous, raw_state, self.candidate_bars,
                )

        in_transition = self.candidate != self.confirmed
        flicker_rate = self.get_flicker_rate()
        is_flickering = flicker_rate > self.flicker_threshold

        multiplier = 1.0
        if in_transition:
            multiplier = min(multiplier, self.transition_size_mult)
        if is_flickering:
            multiplier = min(multiplier, self.uncertainty_size_mult)

        return {
            "state_id": self.confirmed,
            "is_confirmed": not in_transition,
            "consecutive_bars": self.consecutive_bars,
            "in_transition": in_transition,
            "is_flickering": is_flickering,
            "flicker_rate": flicker_rate,
            "size_multiplier": multiplier,
            "changed": changed,
        }

    def get_flicker_rate(self) -> int:
        """Raw state changes within the trailing flicker window."""
        h = self._raw_history
        return sum(1 for i in range(1, len(h)) if h[i] != h[i - 1])

    def is_flickering(self) -> bool:
        return self.get_flicker_rate() > self.flicker_threshold

    def get_regime_stability(self) -> int:
        """Consecutive bars the confirmed regime has held."""
        return self.consecutive_bars


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class HMMEngine:
    """Gaussian HMM regime detector with automatic state-count selection.

    Lifecycle:

        engine = HMMEngine(**settings["hmm"])
        engine.fit(features, returns)          # BIC selection + labelling
        engine.classify_series(features)       # backtest: causal, whole series
        engine.classify(features, as_of)       # live: one bar
        engine.save() / HMMEngine.load()

    `fit` takes raw returns alongside the standardised features because regime
    statistics have to be computed in real units. Feature space is z-scored, so
    a mean read off `means_` is a unitless number that would be mistaken for a
    return.
    """

    def __init__(
        self,
        n_candidates: Optional[list[int]] = None,
        n_init: int = 10,
        covariance_type: str = "full",
        min_train_bars: int = 504,
        stability_bars: int = 3,
        flicker_window: int = 20,
        flicker_threshold: int = 4,
        min_confidence: float = 0.55,
        transition_size_mult: float = 0.75,
        uncertainty_size_mult: float = 0.50,
        feature_columns: Optional[list[str] | str] = None,
        zscore_lookback: int = 252,
        zscore_clip: Optional[float] = 5.0,
        retrain_interval_bars: int = 21,
        random_state: int = 42,
        covars_prior: float = 0.01,
        min_covar: float = 1e-3,
        n_iter: int = 200,
        strict_fittability: bool = True,
        model_path: Optional[Path] = None,
        **_ignored: Any,
    ) -> None:
        self.n_candidates = list(n_candidates or [3, 4, 5, 6, 7])
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.min_train_bars = min_train_bars
        self.stability_bars = stability_bars
        self.flicker_window = flicker_window
        self.flicker_threshold = flicker_threshold
        self.min_confidence = min_confidence
        self.transition_size_mult = transition_size_mult
        self.uncertainty_size_mult = uncertainty_size_mult
        self.feature_columns = resolve_feature_columns(feature_columns)
        self.zscore_lookback = zscore_lookback
        self.zscore_clip = zscore_clip
        self.retrain_interval_bars = retrain_interval_bars
        self.random_state = random_state
        self.covars_prior = covars_prior
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.strict_fittability = strict_fittability
        self.model_path = Path(model_path) if model_path else MODEL_DIR / "hmm_model.pkl"

        self.model = None
        self.n_states: Optional[int] = None
        self.state_labels: dict[int, Regime] = {}
        self.regime_info: dict[int, RegimeInfo] = {}
        self.metadata: Optional[ModelMetadata] = None
        self.tracker = self._new_tracker()
        self._bars_since_fit: int = 0

    def _new_tracker(self) -> RegimeTracker:
        return RegimeTracker(
            stability_bars=self.stability_bars,
            flicker_window=self.flicker_window,
            flicker_threshold=self.flicker_threshold,
            transition_size_mult=self.transition_size_mult,
            uncertainty_size_mult=self.uncertainty_size_mult,
        )

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    # -- fitting ------------------------------------------------------------

    def fit(self, features: pd.DataFrame, returns: Optional[pd.Series] = None) -> "HMMEngine":
        """Fit with automatic state-count selection by BIC.

        For each candidate in `n_candidates`, run `n_init` random restarts and
        keep the best by log-likelihood, then score that model with BIC and
        select the lowest across candidates.

        The restarts matter more than they look: EM is a local optimiser and a
        Gaussian HMM's likelihood surface is riddled with local maxima, so a
        single fit is effectively a random draw. Seeds are derived
        deterministically from `random_state`, so the same data always gives the
        same model. That is what makes the backtest reproducible.
        """
        from hmmlearn.hmm import GaussianHMM

        X = self._prepare_matrix(features)
        n_samples, n_features = X.shape

        if n_samples < self.min_train_bars:
            raise InsufficientDataError(
                f"Need at least {self.min_train_bars} training rows, got {n_samples}. "
                f"Note the feature warmup: {self.min_train_bars} usable rows require "
                f"roughly {self.min_train_bars + 200 + self.zscore_lookback - 2} raw bars."
            )

        check_fittability(
            n_samples, n_features, self.n_candidates,
            self.covariance_type, strict=self.strict_fittability,
        )

        best: dict[str, Any] = {"bic": np.inf}
        all_bic: dict[int, float] = {}

        for n in self.n_candidates:
            candidate = self._fit_candidate(GaussianHMM, X, n)
            if candidate is None:
                logger.warning("n_states=%d: all %d initialisations failed", n, self.n_init)
                continue

            n_params = count_parameters(n, n_features, self.covariance_type)
            bic = -2.0 * candidate["log_likelihood"] + n_params * np.log(n_samples)
            all_bic[n] = float(bic)

            logger.info(
                "n_states=%d  loglik=%.2f  params=%d  BIC=%.2f  converged=%s  iters=%d",
                n, candidate["log_likelihood"], n_params, bic,
                candidate["converged"], candidate["n_iterations"],
            )

            if bic < best["bic"]:
                best = {**candidate, "bic": float(bic), "n_states": n, "n_params": n_params}

        if not all_bic:
            raise RuntimeError(
                "Every candidate failed to fit. Usually means the model is too "
                "large for the data: see check_fittability and docs/PHASE2-NOTES.md."
            )

        logger.info(
            "Selected n_states=%d (BIC %.2f). All candidates: %s",
            best["n_states"], best["bic"],
            {k: round(v, 1) for k, v in sorted(all_bic.items())},
        )

        # A selection landing on either end of the search range means the range
        # itself is the binding constraint, not the data. "Automatic model
        # selection" that always returns the boundary has stopped selecting.
        fitted = sorted(all_bic)
        if len(fitted) > 1 and best["n_states"] == fitted[-1]:
            logger.warning(
                "BIC selected n_states=%d, the LARGEST candidate. The optimum may "
                "lie above the tested range: widen hmm.n_candidates and refit.",
                best["n_states"],
            )
        elif len(fitted) > 1 and best["n_states"] == fitted[0]:
            logger.warning(
                "BIC selected n_states=%d, the SMALLEST candidate. Check the "
                "parameter counts: if the model is overparameterised, the BIC "
                "penalty picks the smallest candidate regardless of the data.",
                best["n_states"],
            )

        self.model = best["model"]
        self.n_states = best["n_states"]
        self._label_states(X, features.index, returns)

        self.metadata = ModelMetadata(
            n_regimes=self.n_states,
            bic=best["bic"],
            log_likelihood=best["log_likelihood"],
            all_bic_scores=all_bic,
            training_date=datetime.now(timezone.utc),
            train_start=features.index[0],
            train_end=features.index[-1],
            n_train_samples=n_samples,
            feature_columns=list(self.feature_columns),
            labels={k: v.value for k, v in self.state_labels.items()},
            converged=best["converged"],
            n_iterations=best["n_iterations"],
            covariance_type=self.covariance_type,
            n_parameters=best["n_params"],
            random_state=self.random_state,
        )

        self.tracker = self._new_tracker()
        self._bars_since_fit = 0
        return self

    def _fit_candidate(self, GaussianHMM, X: np.ndarray, n_states: int) -> Optional[dict]:
        """Run `n_init` restarts for one state count, keep the best log-likelihood."""
        best = None
        for i in range(self.n_init):
            seed = self.random_state + i * 1000 + n_states
            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=seed,
                    covars_prior=self.covars_prior,
                    min_covar=self.min_covar,
                )
                model.fit(X)
                score = float(model.score(X))
            except Exception as exc:  # singular covariance, non-convergence, etc.
                logger.debug("n_states=%d init=%d failed: %s", n_states, i, exc)
                continue

            if not np.isfinite(score):
                continue
            if best is None or score > best["log_likelihood"]:
                best = {
                    "model": model,
                    "log_likelihood": score,
                    "converged": bool(getattr(model.monitor_, "converged", False)),
                    "n_iterations": int(getattr(model.monitor_, "iter", 0)),
                }
        return best

    def _label_states(
        self, X: np.ndarray, index: pd.Index, returns: Optional[pd.Series]
    ) -> None:
        """Assign labels by sorting states on mean return, ascending.

        State assignment for labelling uses the same filtered posterior the live
        path uses, not Viterbi. Labelling on training data is legitimately
        in-sample, but keeping one code path means there is no second place for
        a smoothed posterior to creep in.

        Sorting is what makes labels stable. EM numbers its states arbitrarily
        and renumbers them on every refit, so without this, "state 3" would mean
        something different after each retrain and the dashboard would be
        meaningless.
        """
        posteriors = self._forward_filter().run(X)
        assignments = posteriors.argmax(axis=1)

        if returns is not None:
            aligned = returns.reindex(index)
        else:
            aligned = pd.Series(np.nan, index=index)

        stats = {}
        for state in range(self.n_states):
            mask = assignments == state
            state_returns = aligned.to_numpy()[mask]
            state_returns = state_returns[~np.isnan(state_returns)]
            if len(state_returns) > 1:
                mean_ret = float(np.mean(state_returns) * 252)
                vol = float(np.std(state_returns, ddof=1) * np.sqrt(252))
            else:
                # No usable returns: fall back to the model's own mean on the
                # first feature so ordering is at least deterministic.
                mean_ret = float(self.model.means_[state][0])
                vol = float(np.sqrt(np.trace(
                    _as_full_covariance(
                        self.model.covars_, self.covariance_type,
                        self.n_states, X.shape[1],
                    )[state]
                )))
            stats[state] = {
                "mean_return": mean_ret,
                "volatility": vol,
                "n_obs": int(mask.sum()),
                "frequency": float(mask.mean()),
            }

        by_return = sorted(stats, key=lambda s: stats[s]["mean_return"])
        labels = REGIME_LABEL_SETS[self.n_states]
        self.state_labels = {state: labels[rank] for rank, state in enumerate(by_return)}

        # Volatility rank is computed independently of the label ordering. These
        # two orderings genuinely disagree, which is the point: crash and
        # euphoria sit at opposite ends of the return sort and adjacent on this
        # one. The strategy layer keys off this, not off the label.
        vol_rank = assign_volatility_ranks(
            {state: stats[state]["volatility"] for state in stats}
        )

        self.regime_info = {
            state: RegimeInfo(
                regime_id=state,
                regime_name=self.state_labels[state].value,
                expected_return=stats[state]["mean_return"],
                expected_volatility=stats[state]["volatility"],
                volatility_rank=vol_rank[state],
                recommended_strategy_type=_STRATEGY_BY_VOL_RANK[vol_rank[state]],
                max_leverage_allowed=_MAX_LEVERAGE_BY_VOL_RANK[vol_rank[state]],
                max_position_size_pct=_MAX_POSITION_BY_VOL_RANK[vol_rank[state]],
                min_confidence_to_act=self.min_confidence,
                n_observations=stats[state]["n_obs"],
                frequency=stats[state]["frequency"],
            )
            for state in range(self.n_states)
        }

        for state in by_return:
            info = self.regime_info[state]
            logger.info(
                "state %d -> %-12s  ann.return %+7.2f%%  ann.vol %6.2f%%  "
                "vol_rank=%-4s  freq %5.1f%%",
                state, info.regime_name, info.expected_return * 100,
                info.expected_volatility * 100, info.volatility_rank.value,
                info.frequency * 100,
            )

    # -- prediction ---------------------------------------------------------

    def predict_regime_filtered(self, features_up_to_now: pd.DataFrame) -> pd.DataFrame:
        """P(state_t | observations_1..t) via the forward algorithm.

        Uses ONLY past and present data. No future data. This is the method the
        whole design is built around: `model.predict()` is never called, because
        Viterbi revises earlier states using later observations.

        Pure and stateless: no tracker involvement, no cached alpha. Row t is a
        function of rows 0..t alone, so appending future bars cannot change any
        earlier row. Returns one row per input bar with `state_id`, `label`,
        `probability` and one column per state.
        """
        self._require_fitted()
        X = self._prepare_matrix(features_up_to_now)
        posteriors = self._forward_filter().run(X)
        return self._posteriors_to_frame(posteriors, features_up_to_now.index)

    def predict_regime_proba(self, features_up_to_now: pd.DataFrame) -> pd.DataFrame:
        """Filtered probability distribution over states, one row per bar."""
        self._require_fitted()
        X = self._prepare_matrix(features_up_to_now)
        posteriors = self._forward_filter().run(X)
        return pd.DataFrame(
            posteriors,
            index=features_up_to_now.index,
            columns=[self.state_labels[k].value for k in range(self.n_states)],
        )

    def classify_series(self, features: pd.DataFrame) -> pd.DataFrame:
        """Classify every bar, each using only its own past, with the stability
        filter applied. This is what the backtester consumes.

        The tracker is reset first, so the result depends only on the input and
        is reproducible across runs. It is applied in a forward pass over the
        filtered posteriors, so it introduces no look-ahead of its own: bar t's
        confirmation status depends on bars 0..t.
        """
        filtered = self.predict_regime_filtered(features)
        tracker = self._new_tracker()

        rows = []
        for timestamp, row in filtered.iterrows():
            raw_state = int(row["state_id"])
            update = tracker.update(raw_state)
            confirmed_state = update["state_id"]
            probability = float(row[f"prob_{confirmed_state}"])
            rows.append(
                {
                    "timestamp": timestamp,
                    "state_id": confirmed_state,
                    "label": self.state_labels[confirmed_state].value,
                    "raw_state_id": raw_state,
                    "raw_label": self.state_labels[raw_state].value,
                    "probability": probability,
                    "is_confirmed": update["is_confirmed"],
                    "consecutive_bars": update["consecutive_bars"],
                    "is_flickering": update["is_flickering"],
                    "flicker_rate": update["flicker_rate"],
                    "size_multiplier": update["size_multiplier"],
                    "meets_confidence": probability >= self.min_confidence,
                    "volatility_rank": self.regime_info[confirmed_state].volatility_rank.value,
                }
            )
        return pd.DataFrame(rows).set_index("timestamp")

    def classify(self, features: pd.DataFrame, as_of: Optional[pd.Timestamp] = None) -> RegimeState:
        """Classify a single bar, for the live loop.

        `features` must contain history up to `as_of` inclusive; anything after
        it is truncated rather than trusted, so a caller passing the full frame
        by mistake cannot leak the future.

        Advances the engine's persistent tracker, so calling this is a state
        change. For a reproducible pass over history use `classify_series`.
        """
        self._require_fitted()
        if as_of is not None:
            features = features.loc[:as_of]
        if features.empty:
            raise ValueError("no feature rows at or before as_of")

        timestamp = features.index[-1]
        X = self._prepare_matrix(features)
        posteriors = self._forward_filter().run(X)
        last = posteriors[-1]
        raw_state = int(last.argmax())

        update = self.tracker.update(raw_state)
        self._bars_since_fit += 1
        confirmed_state = update["state_id"]
        probability = float(last[confirmed_state])

        if update["changed"]:
            logger.warning(
                "Regime change confirmed at %s: now %s (p=%.2f)",
                timestamp, self.state_labels[confirmed_state].value, probability,
            )
        elif update["is_confirmed"]:
            logger.info(
                "Regime %s confirmed at %s (%d consecutive bars, p=%.2f)",
                self.state_labels[confirmed_state].value, timestamp,
                update["consecutive_bars"], probability,
            )

        return RegimeState(
            label=self.state_labels[confirmed_state],
            state_id=confirmed_state,
            probability=probability,
            state_probabilities={
                self.state_labels[k]: float(last[k]) for k in range(self.n_states)
            },
            timestamp=timestamp,
            is_confirmed=update["is_confirmed"],
            consecutive_bars=update["consecutive_bars"],
            raw_label=self.state_labels[raw_state],
            raw_state_id=raw_state,
            is_flickering=update["is_flickering"],
            flicker_rate=update["flicker_rate"],
            size_multiplier=update["size_multiplier"],
            meets_confidence=probability >= self.min_confidence,
        )

    def stream(self) -> "RegimeStream":
        """A stateful bar-at-a-time classifier for the live loop and backtester.

        `classify()` re-runs the forward pass over the whole prefix every call,
        which is O(T) per bar and O(T^2) over a backtest. This advances a cached
        alpha by one step instead, and carries its own tracker so the stability
        filter stays consistent.

        Produces identical output to calling `classify()` bar by bar, which
        `test_stream_matches_classify` asserts. If it ever diverges, the live
        loop and the backtest disagree about what regime it is, and every
        backtest result stops describing the system that runs.
        """
        self._require_fitted()
        return RegimeStream(self)

    def make_live_filter(self) -> ForwardFilter:
        """A stepper for the live loop, so one new bar costs one step.

        Produces numerically identical output to `predict_regime_filtered` over
        the same sequence, which tests/test_look_ahead.py asserts. Without that
        guarantee, live trading and backtesting could disagree about the regime.
        """
        self._require_fitted()
        return self._forward_filter()

    # -- introspection ------------------------------------------------------

    def get_transition_matrix(self) -> pd.DataFrame:
        """Learned transition probabilities, labelled. Rows sum to 1.

        The diagonal is the useful part: it is the probability a regime persists
        to the next bar, so 1/(1-diagonal) is its expected duration in bars. A
        diagonal near 0.5 means the "regimes" last two days and are noise.
        """
        self._require_fitted()
        names = [self.state_labels[k].value for k in range(self.n_states)]
        return pd.DataFrame(self.model.transmat_, index=names, columns=names)

    def get_expected_durations(self) -> pd.Series:
        """Expected persistence of each regime in bars: 1 / (1 - p_stay)."""
        self._require_fitted()
        diagonal = np.diag(self.model.transmat_)
        return pd.Series(
            1.0 / np.clip(1.0 - diagonal, 1e-9, None),
            index=[self.state_labels[k].value for k in range(self.n_states)],
        )

    def get_regime_stability(self) -> int:
        """Consecutive bars the currently confirmed regime has held."""
        return self.tracker.get_regime_stability()

    def get_regime_flicker_rate(self) -> int:
        """Raw regime changes within the trailing flicker window."""
        return self.tracker.get_flicker_rate()

    def is_flickering(self) -> bool:
        """True when the model is changing its mind too often to be trusted."""
        return self.tracker.is_flickering()

    def detect_regime_change(self, features: pd.DataFrame, as_of=None) -> bool:
        """True only when a regime change is CONFIRMED on this bar.

        Deliberately not "the raw state differs from last bar". Acting on that
        is what the stability filter exists to prevent.
        """
        before = self.tracker.confirmed
        state = self.classify(features, as_of)
        return before is not None and state.state_id != before

    def get_volatility_rank(self, state_id: int) -> VolatilityRank:
        """Low / mid / high volatility for a state. What the strategy keys off."""
        self._require_fitted()
        return self.regime_info[state_id].volatility_rank

    def get_regime_info(self, state_id: int) -> RegimeInfo:
        self._require_fitted()
        return self.regime_info[state_id]

    def summary(self) -> pd.DataFrame:
        """One row per fitted state. The table to look at after every retrain."""
        self._require_fitted()
        durations = self.get_expected_durations()
        return pd.DataFrame(
            [
                {
                    "state_id": info.regime_id,
                    "label": info.regime_name,
                    "ann_return": info.expected_return,
                    "ann_volatility": info.expected_volatility,
                    "volatility_rank": info.volatility_rank.value,
                    "frequency": info.frequency,
                    "n_obs": info.n_observations,
                    "expected_duration_bars": float(durations.iloc[info.regime_id]),
                    "max_leverage": info.max_leverage_allowed,
                    "max_position_pct": info.max_position_size_pct,
                }
                for info in sorted(self.regime_info.values(), key=lambda i: i.expected_return)
            ]
        )

    # -- retraining and persistence -----------------------------------------

    def should_retrain(self) -> bool:
        """True once `retrain_interval_bars` have passed since the last fit.

        Expanding-window retraining: each refit sees all history up to that
        point. Note that refits renumber states, which is exactly why labels are
        derived by sorting rather than by index.
        """
        return (not self.is_fitted) or self._bars_since_fit >= self.retrain_interval_bars

    def save(self, path: Optional[Path] = None) -> Path:
        """Pickle the model with its metadata.

        The feature column list is stored alongside. `load` refuses to predict
        against a different set, because a reordered column would rotate every
        covariance matrix and produce confident nonsense rather than an error.
        """
        self._require_fitted()
        target = Path(path) if path else self.model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            pickle.dump(
                {
                    "model": self.model,
                    "n_states": self.n_states,
                    "state_labels": {k: v.value for k, v in self.state_labels.items()},
                    "regime_info": self.regime_info,
                    "metadata": self.metadata,
                    "config": self._config_dict(),
                    "format_version": 1,
                },
                fh,
            )
        logger.info("Saved HMM (%d states, BIC %.1f) to %s",
                    self.n_states, self.metadata.bic, target)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "HMMEngine":
        """Restore a saved model. Tracker state is not persisted and restarts."""
        target = Path(path) if path else MODEL_DIR / "hmm_model.pkl"
        with open(target, "rb") as fh:
            payload = pickle.load(fh)

        engine = cls(**payload["config"])
        engine.model = payload["model"]
        engine.n_states = payload["n_states"]
        engine.state_labels = {int(k): Regime(v) for k, v in payload["state_labels"].items()}
        engine.regime_info = payload["regime_info"]
        engine.metadata = payload["metadata"]
        engine.tracker = engine._new_tracker()
        logger.info("Loaded HMM (%d states) trained %s",
                    engine.n_states, engine.metadata.training_date)
        return engine

    # -- internals ----------------------------------------------------------

    def _config_dict(self) -> dict[str, Any]:
        return {
            "n_candidates": self.n_candidates,
            "n_init": self.n_init,
            "covariance_type": self.covariance_type,
            "min_train_bars": self.min_train_bars,
            "stability_bars": self.stability_bars,
            "flicker_window": self.flicker_window,
            "flicker_threshold": self.flicker_threshold,
            "min_confidence": self.min_confidence,
            "transition_size_mult": self.transition_size_mult,
            "uncertainty_size_mult": self.uncertainty_size_mult,
            "feature_columns": self.feature_columns,
            "zscore_lookback": self.zscore_lookback,
            "zscore_clip": self.zscore_clip,
            "retrain_interval_bars": self.retrain_interval_bars,
            "random_state": self.random_state,
            "covars_prior": self.covars_prior,
            "min_covar": self.min_covar,
            "n_iter": self.n_iter,
            "strict_fittability": self.strict_fittability,
            "model_path": str(self.model_path),
        }

    def _forward_filter(self) -> ForwardFilter:
        return ForwardFilter(
            log_startprob=np.log(np.clip(self.model.startprob_, 1e-300, None)),
            log_transmat=np.log(np.clip(self.model.transmat_, 1e-300, None)),
            means=self.model.means_,
            covars=self.model.covars_,
            covariance_type=self.covariance_type,
        )

    def _prepare_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Select the configured columns in the stored order and sanity check.

        Column order is enforced rather than assumed. Passing the same features
        in a different order would silently pair each value with another
        feature's mean and variance.
        """
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"features missing required columns: {sorted(missing)}")
        X = features[self.feature_columns].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError(
                "features contain NaN or inf. Call build_feature_matrix(dropna=True) "
                "so the warmup rows are removed before fitting or predicting."
            )
        return X

    def _posteriors_to_frame(self, posteriors: np.ndarray, index: pd.Index) -> pd.DataFrame:
        states = posteriors.argmax(axis=1)
        frame = pd.DataFrame(
            {
                "state_id": states,
                "label": [self.state_labels[int(s)].value for s in states],
                "probability": posteriors.max(axis=1),
            },
            index=index,
        )
        for k in range(self.n_states):
            frame[f"prob_{k}"] = posteriors[:, k]
        return frame

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("HMMEngine is not fitted. Call fit() or load() first.")


# Allocation guidance per volatility rank. Starting points only: the strategy
# layer in Phase 3 owns these decisions and the backtester in Phase 4 settles
# them. They live here so RegimeInfo is self-describing on the dashboard.
_STRATEGY_BY_VOL_RANK = {
    VolatilityRank.LOW: "fully_invested",
    VolatilityRank.MID: "trend_filtered",
    VolatilityRank.HIGH: "reduced_exposure",
}
_MAX_LEVERAGE_BY_VOL_RANK = {
    VolatilityRank.LOW: 1.25,
    VolatilityRank.MID: 1.0,
    VolatilityRank.HIGH: 1.0,
}
_MAX_POSITION_BY_VOL_RANK = {
    VolatilityRank.LOW: 0.15,
    VolatilityRank.MID: 0.15,
    VolatilityRank.HIGH: 0.10,
}


class RegimeStream:
    """One-bar-at-a-time regime classification with a cached forward alpha.

    Wraps a `ForwardFilter` and a `RegimeTracker` so a caller gets the same
    `RegimeState` that `HMMEngine.classify` returns, at O(1) per bar instead of
    O(T). Warm it on the training tail before stepping through out-of-sample
    bars, so the filtered distribution has settled rather than starting from
    `startprob_` on the first bar that matters.
    """

    def __init__(self, engine: "HMMEngine") -> None:
        self.engine = engine
        self.filter = engine._forward_filter()
        self.tracker = engine._new_tracker()

    def warm(self, features: pd.DataFrame) -> None:
        """Advance the filter over history without recording anything.

        The tracker is deliberately advanced too: its flicker window and
        confirmation count should reflect the bars immediately before the
        out-of-sample period, not start empty at the boundary.
        """
        for observation in self.engine._prepare_matrix(features):
            posterior = self.filter.step(observation)
            self.tracker.update(int(posterior.argmax()))

    def step(self, row: pd.Series) -> RegimeState:
        """Classify one bar. `row` is a single feature row (a Series)."""
        observation = self.engine._prepare_matrix(row.to_frame().T)[0]
        posterior = self.filter.step(observation)
        raw_state = int(posterior.argmax())
        update = self.tracker.update(raw_state)
        confirmed = update["state_id"]
        probability = float(posterior[confirmed])

        return RegimeState(
            label=self.engine.state_labels[confirmed],
            state_id=confirmed,
            probability=probability,
            state_probabilities={
                self.engine.state_labels[k]: float(posterior[k])
                for k in range(self.engine.n_states)
            },
            timestamp=row.name,
            is_confirmed=update["is_confirmed"],
            consecutive_bars=update["consecutive_bars"],
            raw_label=self.engine.state_labels[raw_state],
            raw_state_id=raw_state,
            is_flickering=update["is_flickering"],
            flicker_rate=update["flicker_rate"],
            size_multiplier=update["size_multiplier"],
            meets_confidence=probability >= self.engine.min_confidence,
        )
