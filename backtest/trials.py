"""Multiple-testing correction, and a registry that makes it unavoidable.

THE PROBLEM THIS SOLVES
-----------------------
Try enough configurations and one of them looks good by luck. Not "might look
good" -- will. If you test twenty settings on pure noise, the best of them has
an expected Sharpe well above zero, and reporting that number as though it were
the only thing you tried is how backtests come to promise returns that never
arrive.

This project produced exactly that failure in one session: roughly a dozen
combinations of position cap and circuit-breaker level were swept, the best was
+21.9%, and the sweep was reported without correction. The number was not a
lie, it was a selection.

WHAT THE CORRECTION DOES
------------------------
The Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014) asks a different
question from the Sharpe ratio. Not "how good did this look" but "given that I
searched N configurations, how surprised should I be that the best one looked
this good". It returns a probability that the true Sharpe is above zero.

Two inputs drive it and both matter:

  * How many things you tried. More trials means a higher bar.
  * How much the trials varied. If every configuration scored about the same,
    the best one is not special. If they scattered widely, the search had a lot
    of room to get lucky, and the bar rises further.

Below about 0.95 the result is not evidence. That is not a convention borrowed
from somewhere else; it is the probability the strategy has any edge at all.

WHY A REGISTRY
--------------
The correction needs the number of trials, and the honest number includes the
ones that did not work. Counting only the configurations that made it into the
final report is the same mistake with extra steps, so every run is written to
`data/trials.db` whether or not anybody liked the result.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EULER_MASCHERONI = 0.5772156649015329

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "trials.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    study        TEXT    NOT NULL,
    label        TEXT    NOT NULL,
    config       TEXT    NOT NULL,
    total_return REAL,
    sharpe       REAL,
    max_drawdown REAL,
    avg_exposure REAL,
    n_trades     INTEGER,
    n_bars       INTEGER,
    ran_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_study ON trials (study, ran_at);
"""


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF Acklam-style, accurate enough for this purpose."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Probability the true Sharpe exceeds `benchmark_sr`.

    Corrects the plain Sharpe for two things it ignores. Short samples: a high
    Sharpe over 60 bars says much less than the same figure over 2000. And
    non-normal returns: strategies with a long left tail, which is every
    strategy that sells insurance or holds through gaps, have their Sharpe
    flattered by an estimator that assumes returns are symmetric.

    All inputs are per-observation, not annualised. Annualising first would
    scale the numerator without scaling the sampling error and inflate the
    answer by roughly sqrt(252).
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return float("nan")

    sr = float(r.mean() / r.std(ddof=1))
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0          # pandas gives excess; need raw

    denominator = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    if denominator <= 0:
        return float("nan")

    z = (sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denominator)
    return _norm_cdf(z)


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """The Sharpe the best of `n_trials` reaches on skill-free strategies.

    This is the bar. Searching more configurations raises it, and so does the
    trials disagreeing with each other, because both give the search more room
    to find a lucky one.

    Per-observation, matching `probabilistic_sharpe_ratio`.
    """
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    n = float(n_trials)
    term = ((1 - EULER_MASCHERONI) * _norm_ppf(1 - 1 / n)
            + EULER_MASCHERONI * _norm_ppf(1 - 1 / (n * math.e)))
    return math.sqrt(sharpe_variance) * term


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes: list[float]) -> dict[str, float]:
    """Probability this strategy has a real edge, given everything else tried.

    `trial_sharpes` must be every configuration tested, annualised, including
    the ones that were abandoned. Passing only the survivors understates the
    search and produces a number that is wrong in the flattering direction.

    Returns the probability plus the intermediate values, because a bare 0.31
    invites the question "compared to what" and the answer should be in the
    same dictionary.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return {"dsr": float("nan"), "psr": float("nan"), "sr_annual": float("nan"),
                "hurdle_annual": float("nan"), "n_trials": len(trial_sharpes)}

    sr_obs = float(r.mean() / r.std(ddof=1))
    annualise = math.sqrt(252)

    clean = [s for s in trial_sharpes if s is not None and np.isfinite(s)]
    n_trials = max(len(clean), 1)
    # The variance is measured across trials in per-observation units, which is
    # the space the formula works in.
    variance = float(np.var([s / annualise for s in clean], ddof=1)) if len(clean) > 1 else 0.0

    hurdle = expected_max_sharpe(n_trials, variance)
    return {
        "dsr": probabilistic_sharpe_ratio(r, benchmark_sr=hurdle),
        "psr": probabilistic_sharpe_ratio(r, benchmark_sr=0.0),
        "sr_annual": sr_obs * annualise,
        "hurdle_annual": hurdle * annualise,
        "n_trials": n_trials,
    }


@dataclass
class Trial:
    """One configuration and what it scored."""
    study: str
    label: str
    config: dict[str, Any] = field(default_factory=dict)
    total_return: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    avg_exposure: float | None = None
    n_trades: int | None = None
    n_bars: int | None = None


class TrialRegistry:
    """Every configuration ever tested, so the trial count cannot be understated.

    Deliberately append-only. There is no method to delete a trial, because the
    one thing that breaks the correction is quietly dropping the attempts that
    did not work.
    """

    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def record(self, trial: Trial) -> int:
        cursor = self.conn.execute(
            "INSERT INTO trials (study, label, config, total_return, sharpe, "
            " max_drawdown, avg_exposure, n_trades, n_bars, ran_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (trial.study, trial.label, json.dumps(trial.config, default=str),
             trial.total_return, trial.sharpe, trial.max_drawdown,
             trial.avg_exposure, trial.n_trades, trial.n_bars,
             datetime.now(UTC).isoformat(timespec="seconds")),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def sharpes(self, study: str | None = None) -> list[float]:
        """Annualised Sharpe of every recorded trial. The N in the correction."""
        sql = "SELECT sharpe FROM trials WHERE sharpe IS NOT NULL"
        params: tuple = ()
        if study:
            sql += " AND study = ?"
            params = (study,)
        return [float(row["sharpe"]) for row in self.conn.execute(sql, params)]

    def all(self, study: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM trials"
        params: tuple = ()
        if study:
            sql += " WHERE study = ?"
            params = (study,)
        return list(self.conn.execute(sql + " ORDER BY id", params))

    def count(self, study: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM trials"
        params: tuple = ()
        if study:
            sql += " WHERE study = ?"
            params = (study,)
        return int(self.conn.execute(sql, params).fetchone()["n"])

    def close(self) -> None:
        self.conn.close()
