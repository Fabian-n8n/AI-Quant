"""
Technical indicators and feature computation. The observable inputs to the HMM.

Phase 2.

Every function here is a **pure function**: same input, same output, no state, no
I/O. That is what makes them testable and the backtest reproducible.

THE ONE RULE
------------
No function may see a bar dated after the bar it is computing for. Every window
is trailing. This is where look-ahead bias enters most often, and it does so
silently: a centred rolling window looks like a smoother and is actually a time
machine.

Two consequences that are enforced throughout:

- `min_periods` always equals the window. Pandas defaults to emitting a value as
  soon as it has one observation, so a "200-day SMA" would otherwise return a
  number on bar 3. Those partial values are not wrong-ish, they are a different
  statistic wearing the same name, and they poison the start of every
  walk-forward window.
- Standardisation is a **rolling** z-score, not a fitted scaler. Fitting a
  StandardScaler on the whole series leaks the future's mean and variance
  backwards into every earlier row. A trailing 252-bar z-score cannot, because
  row t only ever sees rows t-251..t.

WARMUP
------
The feature set has a long warmup and it is easy to underestimate. The longest
base window is the 200-bar SMA, and the z-score then needs another 252 bars on
top of that before it emits its first value:

    (200 - 1) + (252 - 1) = 450 bars discarded

So 504 usable feature rows require 954 raw bars, not 504. Call
`required_raw_bars()` rather than assuming. Getting this wrong shows up in Phase
4 as walk-forward windows that silently train on far less data than configured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Longest base lookback in the feature set, before standardisation.
MAX_BASE_WINDOW = 200


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over `periods` bars: log(close_t / close_{t-periods}).

    Log rather than simple returns because they are additive over time and
    closer to symmetric, which suits a Gaussian emission model.
    """
    return np.log(close / close.shift(periods))


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def realized_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling standard deviation of returns, annualised by default.

    The single most informative input for a volatility classifier.
    """
    vol = returns.rolling(window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def volatility_ratio(returns: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """Short-horizon volatility divided by long-horizon volatility.

    The acceleration term. Realised vol says how turbulent it has been; this
    says whether it is getting worse. A ratio above 1 means the last week has
    been rougher than the last month, which is what a regime shift looks like
    before the slower measure catches up.
    """
    fast_vol = returns.rolling(fast, min_periods=fast).std()
    slow_vol = returns.rolling(slow, min_periods=slow).std()
    return fast_vol / slow_vol.replace(0.0, np.nan)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed.

    True Range accounts for overnight gaps, which a simple high-minus-low does
    not. On a system holding positions overnight that distinction is the whole
    point.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return _wilder_smooth(tr, window)


def normalized_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ATR as a fraction of price, so it is comparable across symbols and eras.

    Raw ATR is in dollars: $4 of daily range means something very different for
    a $20 stock than a $600 one, and it means something different for the same
    stock ten years apart.
    """
    return atr(high, low, close, window) / close


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def volume_zscore(volume: pd.Series, window: int = 50) -> pd.Series:
    """Volume relative to its own recent norm.

    Price alone cannot separate a quiet drift down from a panic. Volume is what
    distinguishes them, and it has to be normalised because raw share volume
    trends over years for reasons that have nothing to do with regime.
    """
    mean = volume.rolling(window, min_periods=window).mean()
    std = volume.rolling(window, min_periods=window).std()
    return (volume - mean) / std.replace(0.0, np.nan)


def volume_trend(volume: pd.Series, sma_window: int = 10, slope_window: int = 10) -> pd.Series:
    """Slope of the volume SMA, normalised by its own level.

    Normalised so the result is a fractional change per bar rather than shares
    per bar, which keeps it comparable across symbols.

    The spec says "slope of 10-period SMA" without defining over how many bars
    the slope is measured. This uses a `slope_window`-bar least-squares fit,
    which is less jumpy than a two-point difference.
    """
    sma_vol = volume.rolling(sma_window, min_periods=sma_window).mean()
    return _rolling_slope(sma_vol, slope_window) / sma_vol.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index. Trend strength, not trend direction.

    Directionless on purpose: ADX is high in a strong downtrend and a strong
    uptrend alike, and low in a chop. That is exactly the right shape for a
    volatility classifier, which cares whether the market is organised, not
    which way it is going.

    Wilder's smoothing is implemented as an EWM with alpha = 1/window, which is
    the standard equivalence. Wilder seeded his first value with an SMA; the EWM
    seeds with the first observation. The difference decays and is gone well
    inside the warmup period discarded below.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    atr_w = _wilder_smooth(tr, window)
    plus_di = 100 * _wilder_smooth(plus_dm, window) / atr_w.replace(0.0, np.nan)
    minus_di = 100 * _wilder_smooth(minus_dm, window) / atr_w.replace(0.0, np.nan)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return _wilder_smooth(dx, window)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average. Trailing, and no partial values."""
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average. Trailing, no partial values.

    Distinct from `sma` and not interchangeable with it: every stop in the
    Phase 3 strategies is anchored to the 50 EMA specifically. An EMA reacts
    faster to a change in level, which is what you want a stop to track.

    `min_periods=span` suppresses the warmup, matching the rest of this module.
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def sma_slope(close: pd.Series, sma_window: int = 50, slope_window: int = 10) -> pd.Series:
    """Slope of a price SMA, normalised by its level.

    Normalising turns dollars-per-bar into a fractional rate, so a $600 stock
    and a $20 stock trending equally hard produce the same number.
    """
    ma = sma(close, sma_window)
    return _rolling_slope(ma, slope_window) / ma.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder-smoothed.

    Emitted raw (0-100). The global rolling z-score applied in
    `build_feature_matrix` handles standardisation, so applying a second z-score
    here would double-standardise it and quietly shrink its influence relative
    to every other feature.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _wilder_smooth(gain, window)
    avg_loss = _wilder_smooth(loss, window)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # avg_loss == 0 means an unbroken run of gains: RSI is 100 by definition.
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def distance_from_sma(close: pd.Series, window: int = 200) -> pd.Series:
    """Distance from the long moving average, as a fraction of price.

    The slowest feature in the set, and the one that sets the warmup floor.
    """
    ma = sma(close, window)
    return (close - ma) / ma.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def roc(close: pd.Series, window: int = 10) -> pd.Series:
    """Rate of change over `window` bars, as a fraction."""
    return close.pct_change(window)


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

def rolling_zscore(
    series: pd.Series, window: int = 252, clip: float | None = 5.0
) -> pd.Series:
    """Trailing z-score. The causal alternative to a fitted scaler.

    Row t uses rows t-window+1..t and nothing else, so this cannot leak the
    future no matter how it is called. That property is why the spec asks for a
    rolling z-score rather than sklearn's StandardScaler.

    `clip` bounds the result at +/- that many standard deviations. A single
    outlier at 40 sigma will otherwise dominate a full-covariance Gaussian fit,
    because the likelihood is quadratic in the residual: the model will happily
    spend an entire state describing one day in March 2020. Set clip=None to
    disable.
    """
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    z = (series - mean) / std.replace(0.0, np.nan)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

#: Canonical feature order. The HMM stores this list at fit time and refuses to
#: predict against a different one. A silently reordered column would rotate the
#: covariance matrix and produce confident nonsense rather than an error.
FEATURE_COLUMNS: list[str] = [
    "logret_1",
    "logret_5",
    "logret_20",
    "realvol_20",
    "vol_ratio_5_20",
    "volume_z_50",
    "volume_trend_10",
    "adx_14",
    "sma50_slope",
    "rsi_14",
    "dist_sma_200",
    "roc_10",
    "roc_20",
    "natr_14",
]

#: Subset carrying the volatility signal, for when the full set is too wide to
#: fit. See `check_fittability` and docs/PHASE2-NOTES.md.
VOLATILITY_FEATURES: list[str] = [
    "realvol_20",
    "vol_ratio_5_20",
    "natr_14",
    "adx_14",
    "volume_z_50",
    "logret_5",
]


def compute_raw_features(bars: pd.DataFrame) -> pd.DataFrame:
    """All 14 observable features, unstandardised.

    `bars` needs columns open, high, low, close, volume, indexed by timestamp
    ascending. Returns the same index with the feature columns; leading rows are
    NaN through the warmup.
    """
    _validate_ohlcv(bars)

    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    volume = bars["volume"]
    ret1 = log_returns(close, 1)

    return pd.DataFrame(
        {
            "logret_1": ret1,
            "logret_5": log_returns(close, 5),
            "logret_20": log_returns(close, 20),
            "realvol_20": realized_volatility(ret1, 20),
            "vol_ratio_5_20": volatility_ratio(ret1, 5, 20),
            "volume_z_50": volume_zscore(volume, 50),
            "volume_trend_10": volume_trend(volume, 10, 10),
            "adx_14": adx(high, low, close, 14),
            "sma50_slope": sma_slope(close, 50, 10),
            "rsi_14": rsi(close, 14),
            "dist_sma_200": distance_from_sma(close, 200),
            "roc_10": roc(close, 10),
            "roc_20": roc(close, 20),
            "natr_14": normalized_atr(high, low, close, 14),
        },
        index=bars.index,
    )


def build_feature_matrix(
    bars: pd.DataFrame,
    zscore_window: int = 252,
    clip: float | None = 5.0,
    columns: list[str] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """The HMM's observation matrix: raw features, rolling z-scored, warmup cut.

    This is the only function the HMM engine calls. Everything above is a
    building block for it.

    With `dropna=True` the leading warmup rows are removed, so every returned
    row is fully populated. The number of rows returned will be roughly
    `len(bars) - required_warmup(zscore_window)`, which for the default settings
    is 450 fewer than you started with.
    """
    raw = compute_raw_features(bars)
    cols = columns if columns is not None else FEATURE_COLUMNS
    _assert_known_columns(cols)

    standardised = pd.DataFrame(
        {c: rolling_zscore(raw[c], zscore_window, clip) for c in cols},
        index=raw.index,
    )[cols]

    # Infinities arise from divide-by-near-zero in the ratio features. Left in,
    # they crash the Gaussian fit with a far less obvious error than this.
    standardised = standardised.replace([np.inf, -np.inf], np.nan)
    return standardised.dropna() if dropna else standardised


def required_warmup(zscore_window: int = 252) -> int:
    """Rows discarded before the first fully-populated feature row."""
    return MAX_BASE_WINDOW + zscore_window - 2


def required_raw_bars(n_usable_rows: int, zscore_window: int = 252) -> int:
    """Raw bars needed to yield `n_usable_rows` usable feature rows.

    Use this instead of assuming. For the default 504 training rows the answer
    is 954 bars, roughly 3.8 years, not the 2 years the row count suggests.
    """
    return n_usable_rows + required_warmup(zscore_window)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing, expressed as an EWM with alpha = 1/window."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope over a trailing window, in units per bar.

    Closed form: with t centred on the window, slope = sum(t_c * y) / sum(t_c^2).
    The weights are fixed, so this is a dot product rather than a regression fit
    per bar, which matters because the backtester calls it thousands of times.
    """
    if window < 2:
        raise ValueError("slope window must be at least 2")
    t = np.arange(window, dtype=float)
    t_centred = t - t.mean()
    denominator = (t_centred**2).sum()
    weights = t_centred / denominator
    return series.rolling(window, min_periods=window).apply(
        lambda w: float(np.dot(weights, w)), raw=True
    )


def _validate_ohlcv(bars: pd.DataFrame) -> None:
    missing = {"open", "high", "low", "close", "volume"} - set(bars.columns)
    if missing:
        raise ValueError(f"bars is missing required columns: {sorted(missing)}")
    if not bars.index.is_monotonic_increasing:
        raise ValueError(
            "bars must be sorted ascending by timestamp. Out-of-order bars make "
            "every trailing window silently wrong."
        )
    if bars.index.has_duplicates:
        raise ValueError("bars index contains duplicate timestamps")


def _assert_known_columns(columns: list[str]) -> None:
    unknown = set(columns) - set(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(
            f"unknown feature columns: {sorted(unknown)}. "
            f"Available: {FEATURE_COLUMNS}"
        )
