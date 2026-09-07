"""
Trading calendar: which days the market is open, and when a session has closed.

Phase 10.

WHY NOT `get_clock().is_open`
----------------------------
The orchestrator used to gate on Alpaca's clock, which answers "is the market
open right now". That is the wrong question for a scheduled job. A cron firing
at 21:15 UTC needs to know whether *today was a trading day* and whether *its
session has closed*, and `is_open` is false in both the holiday case and the
five-minutes-before-the-open case without distinguishing them.

Alpaca's calendar endpoint answers the right question. It knows the holiday
schedule, early closes (the half-days after Thanksgiving and before Christmas,
when the close is 13:00 rather than 16:00), and returns the actual open and
close times per session.

EVERYTHING IS US/EASTERN
------------------------
Never local time. The machine that runs this is in Singapore, the GitHub runner
is in UTC, and the market is in New York. Two of those three shift by an hour
twice a year, on different dates. Every comparison in this module is done in
`America/New_York` and converted at the boundary, because "16:00" is only
unambiguous when it carries a timezone.

The daylight-saving consequence worth stating: 21:15 UTC is 17:15 ET in winter
and 16:15 ET in summer. A cron expression cannot express "after the close" for
both, which is why the schedule fires on the wider of the two and this module,
not cron, decides whether the session has actually finished.

WHEN THE BROKER IS UNREACHABLE
------------------------------
`is_trading_day` falls back to a weekday check and logs a warning. It will
therefore say yes on Thanksgiving. That is deliberate and safe: the fallback
only ever causes an *extra* run, and an extra run finds no new bar, so the
orchestrator's existing bar-dedupe key turns it into a no-op. The reverse
default, refusing to run when the calendar is unavailable, would silently skip
real trading days.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")

# How far either side of "now" to pull when the cache misses. Wide enough that
# a normal run never refetches, narrow enough to stay a cheap call.
_LOOKBACK_DAYS = 20
_LOOKAHEAD_DAYS = 45


@dataclass(frozen=True)
class Session:
    """One trading day, with its real open and close in US/Eastern."""
    day: date
    open_at: datetime
    close_at: datetime

    @property
    def is_early_close(self) -> bool:
        return self.close_at.time() < time(16, 0)


def now_et() -> datetime:
    """Current time in market terms. The only clock this module trusts."""
    return datetime.now(MARKET_TZ)


def to_et(moment: datetime) -> datetime:
    """Interpret a naive datetime as Eastern; convert an aware one into it.

    Naive datetimes reaching this module are a bug upstream, but assuming
    Eastern is the least surprising reading and beats raising in a scheduler.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=MARKET_TZ)
    return moment.astimezone(MARKET_TZ)


class MarketCalendar:
    """Sessions from Alpaca, cached per process.

    The cache is a plain dict of date -> Session over a fetched window. A daily
    job makes one calendar call per process, and a long-running process makes
    one every few weeks when the window runs out.
    """

    def __init__(self, client=None) -> None:
        self.client = client
        self._sessions: dict[date, Session] = {}
        self._covered: tuple[date, date] | None = None

    # -- fetching -----------------------------------------------------------

    def _fetch(self, start: date, end: date) -> None:
        from alpaca.trading.requests import GetCalendarRequest

        trading_client = getattr(self.client, "trading_client", None) or self.client
        raw = trading_client.get_calendar(GetCalendarRequest(start=start, end=end))

        for entry in raw:
            day = entry.date if isinstance(entry.date, date) else _parse_date(entry.date)
            self._sessions[day] = Session(
                day=day,
                open_at=_combine(day, entry.open),
                close_at=_combine(day, entry.close),
            )
        self._covered = (start, end)
        logger.debug("calendar: cached %d sessions, %s to %s",
                     len(self._sessions), start, end)

    def _ensure(self, day: date) -> None:
        """Load a window around `day` if it is not already covered.

        Covered-range tracking matters as much as the dict. A non-trading day
        has no entry, so "not in self._sessions" cannot distinguish a holiday
        from a date we never fetched.
        """
        if self._covered and self._covered[0] <= day <= self._covered[1]:
            return
        start = day - timedelta(days=_LOOKBACK_DAYS)
        end = day + timedelta(days=_LOOKAHEAD_DAYS)
        if self._covered:
            start = min(start, self._covered[0])
            end = max(end, self._covered[1])
        self._fetch(start, end)

    # -- queries ------------------------------------------------------------

    def session(self, day: date) -> Session | None:
        """The session on `day`, or None if the market was closed."""
        try:
            self._ensure(day)
        except Exception as exc:
            logger.warning("calendar unavailable (%s), assuming weekdays trade", exc)
            return _assumed_session(day)
        return self._sessions.get(day)

    def is_trading_day(self, day: date) -> bool:
        return self.session(day) is not None

    def last_completed_session(self, now: datetime | None = None) -> Session | None:
        """The most recent session whose close has already passed.

        The bar this returns is the only one safe to act on. A session still in
        progress has a bar that will change, and a strategy that acts on a
        partially formed bar is reading a number that has not happened yet.
        """
        moment = to_et(now) if now else now_et()
        day = moment.date()
        for _ in range(_LOOKBACK_DAYS):
            session = self.session(day)
            if session is not None and session.close_at <= moment:
                return session
            day -= timedelta(days=1)
        logger.warning("no completed session found in the last %d days", _LOOKBACK_DAYS)
        return None

    def next_run_time(self, run_after: time, now: datetime | None = None) -> datetime:
        """When the next daily run is due, in US/Eastern.

        `run_after` is a wall-clock time on a trading day, so a run scheduled
        for 16:15 lands 15 minutes after a normal close and 3h15 after an early
        one. Padding past the close rather than aiming at it leaves room for the
        last bar to settle in Alpaca's data feed.
        """
        moment = to_et(now) if now else now_et()
        day = moment.date()
        for _ in range(_LOOKAHEAD_DAYS):
            session = self.session(day)
            if session is not None:
                due = datetime.combine(day, run_after, tzinfo=MARKET_TZ)
                # Never before the session actually closed. An early close does
                # not move `run_after`, but a run_after earlier than a normal
                # close would otherwise fire mid-session.
                due = max(due, session.close_at + timedelta(minutes=1))
                if due > moment:
                    return due
            day += timedelta(days=1)
        raise RuntimeError(
            f"no trading day found within {_LOOKAHEAD_DAYS} days of {moment.date()}"
        )

    def seconds_until_next_run(self, run_after: time, now: datetime | None = None) -> float:
        moment = to_et(now) if now else now_et()
        return max(0.0, (self.next_run_time(run_after, moment) - moment).total_seconds())

    def is_after_close(self, now: datetime | None = None) -> bool:
        """Has today's session finished? False on a non-trading day."""
        moment = to_et(now) if now else now_et()
        session = self.session(moment.date())
        return session is not None and moment >= session.close_at


# -- helpers ----------------------------------------------------------------

def _combine(day: date, moment) -> datetime:
    """Alpaca returns open/close as a time, or occasionally a full datetime."""
    if isinstance(moment, datetime):
        return to_et(moment)
    return datetime.combine(day, moment, tzinfo=MARKET_TZ)


def _parse_date(value) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _assumed_session(day: date) -> Session | None:
    """Weekday fallback when the calendar cannot be reached.

    Regular hours only, and wrong on holidays. See the module docstring for why
    erring towards an extra run is the safe direction.
    """
    if day.weekday() >= 5:
        return None
    return Session(
        day=day,
        open_at=datetime.combine(day, time(9, 30), tzinfo=MARKET_TZ),
        close_at=datetime.combine(day, time(16, 0), tzinfo=MARKET_TZ),
    )


def parse_run_after(value: str | time) -> time:
    """Config gives `run_after` as "16:15". Turn it into a time."""
    if isinstance(value, time):
        return value
    hours, _, minutes = str(value).partition(":")
    return time(int(hours), int(minutes or 0))
