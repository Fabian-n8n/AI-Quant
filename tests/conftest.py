"""Shared fixtures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env so the Alpaca-marked tests can find credentials. Without this they
# skip silently, which looks identical to passing.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


@pytest.fixture(scope="session")
def settings() -> dict:
    import yaml

    with open(ROOT / "config" / "settings.yaml") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def synthetic_bars() -> pd.DataFrame:
    """OHLCV with three genuine volatility regimes cycling every 180 bars.

    Synthetic on purpose. Real market data cannot tell you whether the model
    found the right answer, because nobody knows the right answer. Here the
    regimes are constructed, so a detector that cannot separate a 0.6% daily
    vol regime from a 3.2% one is broken, not unlucky.
    """
    rng = np.random.default_rng(7)
    n = 2600
    index = pd.bdate_range("2014-01-01", periods=n)
    block = np.arange(n) // 180 % 3
    vol = np.select([block == 0, block == 1, block == 2], [0.006, 0.014, 0.032])
    mu = np.select([block == 0, block == 1, block == 2], [0.0009, 0.0002, -0.0012])
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(mu, vol))), index=index)
    span = np.abs(rng.normal(0, vol * 1.5))
    return pd.DataFrame(
        {
            "open": close.shift(1).bfill(),
            "high": close * (1 + span),
            "low": close * (1 - span),
            "close": close,
            "volume": rng.lognormal(15 + vol * 20, 0.35),
        },
        index=index,
    )


@pytest.fixture(scope="session")
def features(synthetic_bars) -> pd.DataFrame:
    from data.feature_engineering import build_feature_matrix

    return build_feature_matrix(synthetic_bars)


@pytest.fixture(scope="session")
def returns(synthetic_bars) -> pd.Series:
    from data.feature_engineering import log_returns

    return log_returns(synthetic_bars["close"], 1)


@pytest.fixture(scope="session")
def fitted_engine(features, returns):
    """One fitted engine shared across the suite. Fitting takes seconds."""
    from core.hmm_engine import VOLATILITY_FEATURES, HMMEngine

    return HMMEngine(
        feature_columns=VOLATILITY_FEATURES, n_init=4, random_state=42
    ).fit(features, returns)
