"""
Real-time and historical data fetching.

Phase 1 stub. Implemented in Phase 2, alongside the HMM that consumes it.

Source is Alpaca. Note the free plan serves the IEX feed only, which is a
fraction of consolidated volume, so free-tier bars can differ from what you see
on a charting site. Fine for paper and for regime detection on liquid names.
Worth revisiting before any result is treated as evidence.

Two things this has to get right, both silent when wrong:

1. **Adjusted versus raw prices.** Features are computed on split and dividend
   adjusted prices, otherwise a 4-for-1 split looks like a 75% single-day crash
   and the HMM learns a "crash regime" that is really a corporate action. But
   orders, stops and position sizes use raw prices, because that is what
   actually trades. Both series must be retrievable.

2. **Caching.** Refetching years of daily bars on every backtest run is slow and
   burns rate limits. Cache to data/cache/ keyed by symbol and range.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

#: How far back from "now" a historical request may reach.
#:
#: Alpaca's free plan rejects any bar request whose `end` touches the last 15
#: minutes of SIP data: `{"message":"subscription does not permit querying
#: recent SIP data"}`. Every request therefore ends at `utc_now() - SIP_DELAY`.
#:
#: Chosen over hardcoding `feed="iex"`, which also works but pins every request
#: to the thin IEX feed even after a paid upgrade. The delay works on both tiers
#: and costs nothing on a daily-bar system.
SIP_DELAY = timedelta(minutes=16)


def utc_now() -> datetime:
    """Timezone-aware UTC. Never use a naive `datetime.now()` against Alpaca.

    Alpaca interprets a naive datetime as UTC. This machine runs UTC+8
    (Singapore), so a naive `datetime.now()` arrives at the API eight hours in
    the future: `now - 16 minutes` still lands inside the forbidden recent-SIP
    window and the request is rejected, while a `start` date silently shifts by
    eight hours.

    The failure mode is worse on the paid tier, where it would not error at all
    and would just return a slightly wrong window.
    """
    return datetime.now(UTC)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
ROOT = Path(__file__).resolve().parent.parent


def synthetic_bars(
    n_bars: int = 2600,
    start: str = "2014-01-01",
    seed: int = 7,
    regime_length: int = 180,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """OHLCV with three constructed volatility regimes, cycling.

    The fallback when Alpaca credentials are not configured, and the fixture the
    whole test suite runs against.

    Synthetic on purpose rather than as a stopgap. Real market data cannot tell
    you whether a regime detector found the right answer, because nobody knows
    the right answer. Here the regimes are built in at 0.6%, 1.4% and 3.2% daily
    volatility, so a classifier that cannot separate them is broken rather than
    unlucky.

    What it will NOT tell you: whether the strategy makes money. The drift terms
    are arbitrary. Treat every return figure from synthetic data as a check that
    the plumbing works, never as evidence of edge.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n_bars)
    block = np.arange(n_bars) // regime_length % 3

    volatility = np.select([block == 0, block == 1, block == 2], [0.006, 0.014, 0.032])
    drift = np.select([block == 0, block == 1, block == 2], [0.0009, 0.0002, -0.0012])

    close = pd.Series(
        start_price * np.exp(np.cumsum(rng.normal(drift, volatility))), index=index
    )
    span = np.abs(rng.normal(0, volatility * 1.5))

    return pd.DataFrame(
        {
            "open": close.shift(1).bfill(),
            "high": close * (1 + span),
            "low": close * (1 - span),
            "close": close,
            "volume": rng.lognormal(15 + volatility * 20, 0.35),
        },
        index=index,
    )


def load_bars(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    allow_synthetic: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Load daily bars for one symbol. Returns (bars, is_synthetic).

    Tries Alpaca first, falls back to synthetic data when credentials are
    missing. The boolean is returned rather than logged and forgotten so callers
    can label their output: a backtest on synthetic data that is reported as if
    it were SPY is worse than no backtest.

    Real bars come from the same cached Alpaca path the live engine uses, so a
    backtest and a live run see identical history.

    This used to raise NotImplementedError whenever credentials were present,
    and fall through to synthetic data whenever they were not. Nothing called
    it with credentials loaded, so every backtest this project has ever
    produced -- including the walk-forward result that preflight compares
    against buy-and-hold -- was computed on a random walk. The numbers were not
    evidence that the strategy underperforms. They were not evidence of
    anything.
    """
    import os

    # The caller is often a script that never loaded .env. Loading it here is
    # what makes "do I have credentials" mean "can I reach Alpaca" rather than
    # "did my caller remember".
    load_dotenv(ROOT / ".env", override=False)
    has_credentials = bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))

    if has_credentials:
        try:
            from broker.alpaca_client import AlpacaClient

            # MarketDataClient needs a connected broker client; constructed
            # bare it has no data client and every fetch returns None.
            broker = AlpacaClient()
            broker.connect()
            client = MarketDataClient(broker)
            bars = client.get_historical_bars(
                symbol, "1Day",
                start=pd.Timestamp(start).to_pydatetime() if start else None,
                end=pd.Timestamp(end).to_pydatetime() if end else None,
            )
            bars = _extract_symbol(bars, symbol)
            if bars is not None and not bars.empty:
                return bars, False
            logger.warning("%s: Alpaca returned no bars for the requested range.", symbol)
        except Exception as exc:
            # Falling back silently is how a synthetic backtest gets reported as
            # if it were SPY. Say so loudly, and refuse entirely when the caller
            # has said it wants real data or nothing.
            logger.error("%s: could not load real bars (%s).", symbol, exc)
            if not allow_synthetic:
                raise

    if not allow_synthetic:
        raise RuntimeError(
            f"No real bars available for {symbol}. Set ALPACA_API_KEY and "
            f"ALPACA_SECRET_KEY in .env, or allow the synthetic fallback."
        )

    logger.warning(
        "%s: falling back to SYNTHETIC bars. Any return figure from this run is "
        "a check that the plumbing works, not evidence about the strategy.", symbol)
    bars = synthetic_bars()
    if start:
        bars = bars.loc[str(start):]
    if end:
        bars = bars.loc[:str(end)]
    return bars, True


def _as_utc(value: datetime | None) -> datetime | None:
    """Attach UTC to a naive datetime rather than letting Alpaca assume it.

    A caller passing `datetime(2024, 1, 1)` means midnight, and on a UTC+8
    machine that must not become 08:00 UTC the previous day.
    """
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _match_tz(moment: pd.Timestamp, reference: pd.Timestamp) -> pd.Timestamp:
    """Align one timestamp's awareness to another's before comparing them.

    Cached parquet keeps whatever tz Alpaca returned; callers pass anything.
    Comparing an aware Timestamp to a naive one raises, and the raise would
    surface as a cache miss and a silent refetch of four years of bars.
    """
    aware = reference.tzinfo is not None
    if aware and moment.tzinfo is None:
        return moment.tz_localize("UTC")
    if not aware and moment.tzinfo is not None:
        return moment.tz_localize(None)
    return moment


def _bar_period(timeframe: str) -> timedelta:
    """How long one bar covers. The smallest gap worth a network call."""
    return {
        "1Min": timedelta(minutes=1), "5Min": timedelta(minutes=5),
        "15Min": timedelta(minutes=15), "1Hour": timedelta(hours=1),
    }.get(timeframe, timedelta(days=1))


def _extract_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """One symbol's rows out of a (timestamp, symbol) frame, flat by timestamp."""
    if frame is None or frame.empty:
        return None
    if not isinstance(frame.index, pd.MultiIndex):
        return frame
    if "symbol" not in (frame.index.names or []):
        return None
    try:
        return frame.xs(symbol, level="symbol")
    except KeyError:
        return None


class MarketDataClient:
    """Historical bars, live quotes and streaming, with a disk cache.

    Two things this has to get right, both silent when wrong:

    1. **Adjusted versus raw prices.** Features are computed on split and
       dividend adjusted prices, otherwise a 4-for-1 split looks like a 75%
       single-day crash and the HMM learns a "crash regime" that is really a
       corporate action. Orders, stops and position sizes use raw prices,
       because that is what trades. Alpaca's `adjustment` parameter controls it
       and both are retrievable.

    2. **Caching.** Refetching years of daily bars on every backtest run is slow
       and burns rate limits. Cached to `data/cache/` keyed by symbol,
       timeframe, range and adjustment, so a changed adjustment cannot silently
       serve the wrong series.

    FREE TIER CAVEATS, BOTH LOAD-BEARING
    ------------------------------------
    1. **No recent SIP data.** A request whose `end` touches the last 15 minutes
       is rejected outright. Every request therefore ends at `now - SIP_DELAY`.
       Harmless on daily bars; it would matter on an intraday system.

    2. **IEX feed only**, a fraction of consolidated volume, so bars can differ
       from a charting site and quotes commonly return a **zero ask outside
       market hours**. Callers must treat a zero as "no quote" rather than as a
       price of zero. `reference_price()` handles the fallback.

    Fine for paper and for regime detection on liquid names. Worth revisiting
    before any result is treated as evidence.
    """

    def __init__(self, alpaca_client=None, cache_dir: Path = CACHE_DIR) -> None:
        self.alpaca_client = alpaca_client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._bar_stream = None
        self._stream_thread = None
        self._stop_stream = threading.Event()

    # -- historical ---------------------------------------------------------

    def get_historical_bars(
        self,
        symbols: str | list[str],
        timeframe: str = "1Day",
        start: datetime | None = None,
        end: datetime | None = None,
        adjusted: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Daily (or other timeframe) OHLCV, indexed by (timestamp, symbol).

        `adjusted=True` returns split and dividend adjusted prices for feature
        computation. Pass `adjusted=False` for the raw prices that orders and
        stops must use.
        """
        symbol_list = [symbols] if isinstance(symbols, str) else list(symbols)
        start = start or utc_now() - timedelta(days=365 * 5)
        # Never reach into the SIP delay window: the free plan rejects it
        # outright, and on a daily-bar system the last 16 minutes are worthless.
        latest_allowed = utc_now() - SIP_DELAY
        start, end = _as_utc(start), _as_utc(end)
        end = min(end, latest_allowed) if end else latest_allowed

        if not use_cache:
            return self._fetch_bars(symbol_list, timeframe, start, end, adjusted)

        # Work out per symbol what is missing, then fetch the union in one
        # request. Asking for only the gap is the point: without it, a window
        # that has advanced by one day refetches four years of history.
        needed: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for symbol in symbol_list:
            gaps = self.missing_ranges(symbol, timeframe, start, end, adjusted)
            if gaps:
                needed[symbol] = gaps

        fresh = pd.DataFrame()
        if needed:
            fetch_start = min(g[0] for gaps in needed.values() for g in gaps)
            fetch_end = max(g[1] for gaps in needed.values() for g in gaps)
            logger.debug("cache miss for %s, fetching %s to %s",
                         ", ".join(needed), fetch_start.date(), fetch_end.date())
            fresh = self._fetch_bars(list(needed), timeframe, fetch_start, fetch_end, adjusted)
            for symbol in needed:
                part = _extract_symbol(fresh, symbol)
                if part is not None and not part.empty:
                    self._write_store(symbol, timeframe, adjusted, part)

        frames = {}
        for symbol in symbol_list:
            part = self._cached_slice(symbol, timeframe, start, end, adjusted)
            if part is None or part.empty:
                # Fall back to what was just fetched. The cache is an
                # optimisation and must never be load-bearing: without this,
                # an unreadable cache turns a successful API call into "no
                # data", which is how a missing parquet engine took down the
                # first scheduled run. A cache failure should cost speed, not
                # correctness.
                part = _extract_symbol(fresh, symbol)
            if part is not None and not part.empty:
                frames[symbol] = part

        if not frames:
            logger.warning("No bars available for %s", symbol_list)
            return pd.DataFrame()

        combined = pd.concat(frames, names=["symbol", "timestamp"])
        return combined.swaplevel().sort_index()

    def _fetch_bars(self, symbols: list[str], timeframe: str, start, end,
                    adjusted: bool) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=self._to_timeframe(timeframe),
            start=start,
            end=end,
            adjustment="all" if adjusted else "raw",
        )
        frame = self.alpaca_client.data_client.get_stock_bars(request).df
        if frame.empty:
            logger.warning("No bars returned for %s", symbols)
            return frame
        return self._normalise(frame)

    def get_historical(
        self, symbol: str, lookback_days: int = 400, adjusted: bool = True
    ) -> pd.DataFrame:
        """Single symbol, flat OHLCV frame indexed by timestamp.

        The shape the strategies and feature builder expect.
        """
        frame = self.get_historical_bars(
            symbol,
            start=utc_now() - timedelta(days=lookback_days),
            adjusted=adjusted,
        )
        return self._flatten(frame, symbol)

    def get_training_window(self, symbol: str, n_bars: int = 954) -> pd.DataFrame:
        """Enough raw bars to yield a usable feature window.

        Defaults to 954 because the feature warmup discards 450 rows, so 504
        usable rows need 954 raw bars. Calendar days are padded well beyond that
        since weekends and holidays are not trading days.
        """
        return self.get_historical(symbol, lookback_days=int(n_bars * 1.55))

    # -- live ---------------------------------------------------------------

    def get_latest_bar(self, symbol: str) -> dict[str, Any]:
        from alpaca.data.requests import StockLatestBarRequest

        bar = self.alpaca_client.data_client.get_stock_latest_bar(
            StockLatestBarRequest(symbol_or_symbols=symbol)
        )[symbol]
        return {
            "symbol": symbol, "open": float(bar.open), "high": float(bar.high),
            "low": float(bar.low), "close": float(bar.close),
            "volume": float(bar.volume), "timestamp": bar.timestamp,
        }

    def get_latest_quote(self, symbol: str) -> dict[str, Any]:
        """Best bid and ask. A zero means no quote, not a price of zero."""
        return self.alpaca_client.get_latest_quote(symbol)

    def get_snapshot(self, symbols: str | list[str]) -> dict[str, dict[str, Any]]:
        """Latest bar, quote and daily bar in one call, per symbol."""
        from alpaca.data.requests import StockSnapshotRequest

        symbol_list = [symbols] if isinstance(symbols, str) else list(symbols)
        snapshots = self.alpaca_client.data_client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=symbol_list)
        )
        out: dict[str, dict[str, Any]] = {}
        for symbol, snapshot in snapshots.items():
            quote, daily = snapshot.latest_quote, snapshot.daily_bar
            out[symbol] = {
                "bid": float(quote.bid_price or 0) if quote else 0.0,
                "ask": float(quote.ask_price or 0) if quote else 0.0,
                "last_close": float(daily.close) if daily else 0.0,
                "daily_open": float(daily.open) if daily else 0.0,
                "daily_volume": float(daily.volume) if daily else 0.0,
                "timestamp": quote.timestamp if quote else None,
            }
        return out

    def reference_price(self, symbol: str) -> float:
        """A usable price whatever the market is doing.

        Midpoint when both sides quote, otherwise whichever side does, otherwise
        the last daily close. Outside hours the IEX feed returns a zero ask, so
        without this fallback every limit order placed on a weekend would be
        priced off zero.
        """
        try:
            snapshot = self.get_snapshot(symbol)[symbol]
        except Exception as exc:
            logger.warning("%s: snapshot failed (%s)", symbol, exc)
            return 0.0

        bid, ask, close = snapshot["bid"], snapshot["ask"], snapshot["last_close"]
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return ask or bid or close

    # -- streaming ----------------------------------------------------------

    def subscribe_bars(self, symbols: list[str], callback, timeframe: str = "1Day") -> Any:
        """Stream bar updates on a background thread, reconnecting on failure.

        A convenience rather than a requirement: this is a daily-bar swing
        system, so the loop can equally poll the last completed bar. The
        orchestrator must not depend on the socket staying up.
        """
        return self._start_stream("bars", symbols, callback)

    def subscribe_quotes(self, symbols: list[str], callback) -> Any:
        """Stream quotes, for the risk manager's spread check."""
        return self._start_stream("quotes", symbols, callback)

    def _start_stream(self, kind: str, symbols: list[str], callback):
        from alpaca.data.live import StockDataStream

        def run() -> None:
            backoff = 1.0
            while not self._stop_stream.is_set():
                try:
                    stream = StockDataStream(
                        self.alpaca_client._api_key, self.alpaca_client._secret_key
                    )
                    self._bar_stream = stream
                    if kind == "bars":
                        stream.subscribe_bars(callback, *symbols)
                    else:
                        stream.subscribe_quotes(callback, *symbols)
                    logger.info("%s stream connected for %s", kind, symbols)
                    backoff = 1.0
                    stream.run()
                except Exception as exc:
                    if self._stop_stream.is_set():
                        break
                    logger.warning("%s stream dropped (%s), reconnecting in %.0fs",
                                   kind, exc, backoff)
                    self._stop_stream.wait(backoff)
                    backoff = min(backoff * 2, 60.0)

        thread = threading.Thread(target=run, name=f"alpaca-{kind}-stream", daemon=True)
        thread.start()
        self._stream_thread = thread
        return thread

    def stop_stream(self) -> None:
        self._stop_stream.set()
        if self._bar_stream is not None:
            try:
                self._bar_stream.stop()
            except Exception:
                pass

    # -- validation ---------------------------------------------------------

    def validate(self, bars: pd.DataFrame) -> list[str]:
        """Data quality problems, empty if clean.

        Run before any fit or backtest. Bad data is worse than no data because
        it produces a number you might believe.

        Weekend and holiday gaps are expected and not reported: the market being
        shut is not a data problem. Gaps *inside* a trading week are, because
        they silently change every rolling window that crosses them.
        """
        problems: list[str] = []
        if bars.empty:
            return ["no bars returned"]

        if bars.index.has_duplicates:
            problems.append(f"{int(bars.index.duplicated().sum())} duplicate timestamps")

        for column in ("open", "high", "low", "close"):
            if column in bars and (bars[column] <= 0).any():
                problems.append(f"non-positive {column} prices")

        if {"high", "low"} <= set(bars.columns) and (bars["high"] < bars["low"]).any():
            problems.append("high below low on some bars")

        if "volume" in bars and (bars["volume"] <= 0).sum() > len(bars) * 0.05:
            problems.append("more than 5% of bars have zero volume")

        if "close" in bars and len(bars) > 1:
            moves = bars["close"].pct_change().abs()
            extreme = moves > 0.35
            if extreme.any():
                problems.append(
                    f"{int(extreme.sum())} single-bar moves above 35%, "
                    "possibly an unadjusted split"
                )

        if isinstance(bars.index, pd.DatetimeIndex) and len(bars) > 5:
            expected = pd.bdate_range(bars.index[0], bars.index[-1])
            missing = len(expected) - len(bars)
            # Roughly 10 US market holidays a year. Anything well beyond that is
            # a real gap rather than the calendar.
            allowance = max(3, int(len(expected) * 0.06))
            if missing > allowance:
                problems.append(f"{missing} weekday bars missing beyond holiday allowance")

        return problems

    # -- cache --------------------------------------------------------------
    #
    # Partitioned by symbol, append-only. The previous scheme hashed the whole
    # request (symbols + range + adjustment) into one filename, which meant a
    # request one day longer than a cached one shared nothing with it and
    # refetched four years of bars. Nineteen files had accumulated, none reused.
    #
    # Keyed on (symbol, timeframe, adjusted) because those are the three things
    # that change the *content* of a bar. Ranges are not part of the key: they
    # are what `missing_ranges` works out against what is already stored.

    def _store_path(self, symbol: str, timeframe: str, adjusted: bool) -> Path:
        return (self.cache_dir / "bars" /
                f"{symbol}_{timeframe}_{'adj' if adjusted else 'raw'}.parquet")

    def _read_store(self, symbol: str, timeframe: str, adjusted: bool) -> pd.DataFrame | None:
        path = self._store_path(symbol, timeframe, adjusted)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("cache read failed for %s (%s), refetching", symbol, exc)
            return None

    def _write_store(self, symbol: str, timeframe: str, adjusted: bool,
                     frame: pd.DataFrame) -> None:
        """Merge new bars into the symbol's store and write it back.

        New rows win on collision. An adjusted series is rewritten backwards by
        every split and dividend, so the freshly fetched copy of an overlapping
        bar is the correct one and the stored copy is stale by definition.
        """
        path = self._store_path(symbol, timeframe, adjusted)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read_store(symbol, timeframe, adjusted)
        if existing is not None and not existing.empty:
            frame = pd.concat([existing, frame])
            frame = frame[~frame.index.duplicated(keep="last")]
        try:
            frame.sort_index().to_parquet(path)
        except Exception as exc:
            logger.debug("cache write skipped for %s (%s)", symbol, exc)

    def missing_ranges(self, symbol: str, timeframe: str, start, end,
                       adjusted: bool = True) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Which parts of [start, end] are not already on disk.

        Returns the head and tail gaps only. Interior holes are not detected,
        because the source is contiguous trading days: a range fetched from
        Alpaca comes back complete or does not come back. Detecting interior
        gaps would mean reconstructing the trading calendar here to tell a
        missing bar from a holiday, which is `validate`'s job, not the cache's.
        """
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        stored = self._read_store(symbol, timeframe, adjusted)
        if stored is None or stored.empty:
            return [(start, end)]

        index = pd.DatetimeIndex(stored.index)
        first, last = index.min(), index.max()
        first, last = _match_tz(first, start), _match_tz(last, start)

        # Ignore gaps shorter than one bar. `start` carries a time of day and
        # the first stored bar sits at the session open, so a naive comparison
        # reports a thirteen-hour "gap" at the head of every request and
        # refetches it forever. There is no bar in there to find.
        tolerance = _bar_period(timeframe)

        gaps = []
        if first - start >= tolerance:
            gaps.append((start, min(first, end)))
        if end - last >= tolerance:
            gaps.append((max(last, start), end))
        return gaps

    def _cached_slice(self, symbol: str, timeframe: str, start, end,
                      adjusted: bool) -> pd.DataFrame | None:
        stored = self._read_store(symbol, timeframe, adjusted)
        if stored is None or stored.empty:
            return None
        index = pd.DatetimeIndex(stored.index)
        lo, hi = _match_tz(pd.Timestamp(start), index[0]), _match_tz(pd.Timestamp(end), index[0])
        return stored.loc[(index >= lo) & (index <= hi)]

    # -- shaping ------------------------------------------------------------

    @staticmethod
    def _to_timeframe(timeframe: str):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        mapping = {
            "1Day": TimeFrame.Day, "1Hour": TimeFrame.Hour, "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Week": TimeFrame.Week,
        }
        if timeframe not in mapping:
            raise ValueError(f"unsupported timeframe {timeframe!r}. Use one of {sorted(mapping)}")
        return mapping[timeframe]

    @staticmethod
    def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
        """Alpaca returns a (symbol, timestamp) MultiIndex. Flip it to
        (timestamp, symbol) so time slicing works the obvious way."""
        if isinstance(frame.index, pd.MultiIndex) and frame.index.names == ["symbol", "timestamp"]:
            frame = frame.swaplevel().sort_index()
        return frame

    @staticmethod
    def _flatten(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Reduce a multi-symbol frame to one symbol's OHLCV, tz-naive.

        Timezones are dropped because the feature layer compares against
        `pd.bdate_range`, which is naive, and a mismatch there raises rather
        than silently misaligning.
        """
        if frame.empty:
            return frame
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.xs(symbol, level="symbol") if "symbol" in frame.index.names \
                else frame.droplevel(0)
        frame = frame[["open", "high", "low", "close", "volume"]].copy()
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        return frame.sort_index()
