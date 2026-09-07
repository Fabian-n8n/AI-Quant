"""
Trading calendar and daily cadence.

The bugs this suite exists to catch are all of the same family: a scheduler
that is right in one timezone and wrong in another, or right in November and
wrong in July. They do not show up in a single-day test, so most of these pin
specific real dates.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from core.calendar import (
    MARKET_TZ,
    MarketCalendar,
    Session,
    parse_run_after,
    to_et,
)

UTC = ZoneInfo("UTC")


# -- fakes ------------------------------------------------------------------

class FakeCalendarEntry:
    def __init__(self, day: date, open_at=time(9, 30), close_at=time(16, 0)):
        self.date = day
        self.open = open_at
        self.close = close_at


class FakeTradingClient:
    """Weekdays trade, minus an explicit holiday list. Early closes supported."""

    def __init__(self, holidays=(), early_closes=(), fail=False):
        self.holidays = set(holidays)
        self.early_closes = set(early_closes)
        self.fail = fail
        self.calls = 0

    def get_calendar(self, request):
        self.calls += 1
        if self.fail:
            raise RuntimeError("calendar endpoint down")
        out = []
        day = request.start
        while day <= request.end:
            if day.weekday() < 5 and day not in self.holidays:
                close = time(13, 0) if day in self.early_closes else time(16, 0)
                out.append(FakeCalendarEntry(day, close_at=close))
            day = date.fromordinal(day.toordinal() + 1)
        return out


def calendar(**kwargs) -> MarketCalendar:
    return MarketCalendar(FakeTradingClient(**kwargs))


def et(year, month, day, hour=0, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=MARKET_TZ)


# -- trading days -----------------------------------------------------------

def test_weekends_are_not_trading_days():
    cal = calendar()
    assert cal.is_trading_day(date(2026, 9, 5)) is False   # Saturday
    assert cal.is_trading_day(date(2026, 9, 6)) is False   # Sunday
    assert cal.is_trading_day(date(2026, 9, 4)) is True    # Friday


def test_a_holiday_is_not_a_trading_day():
    """The case get_clock().is_open cannot distinguish from "before the open"."""
    labor_day = date(2026, 9, 7)
    cal = calendar(holidays=[labor_day])
    assert cal.is_trading_day(labor_day) is False
    assert cal.is_trading_day(date(2026, 9, 8)) is True


def test_july_fourth_is_skipped():
    cal = calendar(holidays=[date(2026, 7, 3)])   # observed, the 4th is a Saturday
    assert cal.is_trading_day(date(2026, 7, 3)) is False


def test_the_calendar_is_fetched_once_and_cached():
    client = FakeTradingClient()
    cal = MarketCalendar(client)
    for offset in range(10):
        cal.is_trading_day(date(2026, 9, 1 + offset))
    assert client.calls == 1


def test_a_date_outside_the_cached_window_refetches():
    client = FakeTradingClient()
    cal = MarketCalendar(client)
    cal.is_trading_day(date(2026, 9, 4))
    cal.is_trading_day(date(2027, 3, 1))
    assert client.calls == 2


def test_a_missing_day_inside_the_window_is_a_holiday_not_a_cache_miss():
    """The distinction the covered-range tracking exists for. Without it, a
    holiday and a never-fetched date look identical."""
    client = FakeTradingClient(holidays=[date(2026, 9, 7)])
    cal = MarketCalendar(client)
    assert cal.is_trading_day(date(2026, 9, 7)) is False
    assert client.calls == 1, "a holiday triggered a refetch"


# -- completed sessions -----------------------------------------------------

def test_last_completed_session_is_today_after_the_close():
    cal = calendar()
    session = cal.last_completed_session(et(2026, 9, 4, 16, 30))
    assert session.day == date(2026, 9, 4)


def test_last_completed_session_is_yesterday_before_the_close():
    """The bar for a session in progress is not final and must not be traded."""
    cal = calendar()
    session = cal.last_completed_session(et(2026, 9, 4, 11, 0))
    assert session.day == date(2026, 9, 3)


def test_last_completed_session_skips_back_over_a_weekend():
    cal = calendar()
    session = cal.last_completed_session(et(2026, 9, 6, 12, 0))   # Sunday
    assert session.day == date(2026, 9, 4)                        # Friday


def test_last_completed_session_skips_back_over_a_holiday():
    cal = calendar(holidays=[date(2026, 9, 7)])
    session = cal.last_completed_session(et(2026, 9, 7, 20, 0))
    assert session.day == date(2026, 9, 4)


def test_an_early_close_completes_early():
    """Half-days after Thanksgiving close at 13:00. At 14:00 that bar is final."""
    black_friday = date(2026, 11, 27)
    cal = calendar(early_closes=[black_friday])
    session = cal.last_completed_session(et(2026, 11, 27, 14, 0))
    assert session.day == black_friday
    assert session.is_early_close


# -- scheduling -------------------------------------------------------------

def test_next_run_is_today_when_the_time_has_not_passed():
    cal = calendar()
    due = cal.next_run_time(time(16, 15), et(2026, 9, 4, 10, 0))
    assert due == et(2026, 9, 4, 16, 15)


def test_next_run_rolls_to_the_next_trading_day_once_today_has_passed():
    cal = calendar()
    due = cal.next_run_time(time(16, 15), et(2026, 9, 4, 17, 0))   # Friday evening
    assert due == et(2026, 9, 7, 16, 15), "should skip Saturday and Sunday"


def test_next_run_skips_a_holiday():
    cal = calendar(holidays=[date(2026, 9, 7)])
    due = cal.next_run_time(time(16, 15), et(2026, 9, 6, 12, 0))
    assert due.date() == date(2026, 9, 8)


def test_a_run_time_before_the_close_is_pushed_past_it():
    """Otherwise a misconfigured run_after would fire mid-session, on a bar
    that is still forming."""
    cal = calendar()
    due = cal.next_run_time(time(11, 0), et(2026, 9, 4, 9, 0))
    assert due > et(2026, 9, 4, 16, 0)


def test_an_early_close_does_not_pull_the_run_time_earlier():
    """13:00 close, 16:15 run time. The run still happens at 16:15, because
    run_after is a wall-clock time and there is no reason to hurry."""
    cal = calendar(early_closes=[date(2026, 11, 27)])
    due = cal.next_run_time(time(16, 15), et(2026, 11, 27, 9, 0))
    assert due == et(2026, 11, 27, 16, 15)


# -- daylight saving --------------------------------------------------------
#
# The whole reason the calendar check lives inside the job rather than in cron.
# 21:15 UTC is 17:15 ET in winter and 16:15 ET in summer, so no single cron
# expression means "after the close" all year.

def test_the_utc_offset_differs_between_summer_and_winter():
    summer = et(2026, 7, 15, 16, 0).astimezone(UTC)
    winter = et(2026, 1, 15, 16, 0).astimezone(UTC)
    assert summer.hour == 20    # EDT, UTC-4
    assert winter.hour == 21    # EST, UTC-5


def test_the_scheduled_utc_time_is_after_the_close_in_both_halves_of_the_year():
    """21:15 UTC, the workflow's cron. In summer that is 17:15 ET, in winter
    16:15 ET. Both are past a 16:00 close, which is the property that matters."""
    for month in (1, 7):
        fired = datetime(2026, month, 15, 21, 15, tzinfo=UTC)
        assert to_et(fired).time() > time(16, 0)


def test_a_run_scheduled_across_the_spring_forward_boundary():
    """DST begins 2026-03-08. A Friday-evening run must land on Monday at the
    same *wall-clock* time, not the same elapsed offset."""
    cal = calendar()
    due = cal.next_run_time(time(16, 15), et(2026, 3, 6, 17, 0))
    assert due == et(2026, 3, 9, 16, 15)
    assert due.utcoffset().total_seconds() / 3600 == -4    # EDT


def test_a_run_scheduled_across_the_fall_back_boundary():
    """DST ends 2026-11-01."""
    cal = calendar()
    due = cal.next_run_time(time(16, 15), et(2026, 10, 30, 17, 0))
    assert due == et(2026, 11, 2, 16, 15)
    assert due.utcoffset().total_seconds() / 3600 == -5    # EST


def test_a_naive_datetime_is_read_as_eastern_not_local():
    """The machine running this is in Singapore. Local time is never the answer."""
    assert to_et(datetime(2026, 9, 4, 16, 0)).tzinfo is MARKET_TZ


# -- fallback ---------------------------------------------------------------

def test_an_unreachable_calendar_falls_back_to_weekdays():
    """Erring towards an extra run, not a missed one. The extra run finds no
    new bar and the orchestrator's dedupe key makes it a no-op."""
    cal = calendar(fail=True)
    assert cal.is_trading_day(date(2026, 9, 4)) is True    # Friday
    assert cal.is_trading_day(date(2026, 9, 5)) is False   # Saturday


def test_the_fallback_still_produces_a_usable_session():
    cal = calendar(fail=True)
    session = cal.last_completed_session(et(2026, 9, 4, 17, 0))
    assert session.day == date(2026, 9, 4)
    assert session.close_at.time() == time(16, 0)


# -- helpers ----------------------------------------------------------------

@pytest.mark.parametrize(("raw", "expected"), [
    ("16:15", time(16, 15)),
    ("9:30", time(9, 30)),
    ("16", time(16, 0)),
    (time(16, 15), time(16, 15)),
])
def test_parse_run_after(raw, expected):
    assert parse_run_after(raw) == expected


def test_session_knows_it_closed_early():
    early = Session(
        day=date(2026, 11, 27),
        open_at=et(2026, 11, 27, 9, 30),
        close_at=et(2026, 11, 27, 13, 0),
    )
    assert early.is_early_close
