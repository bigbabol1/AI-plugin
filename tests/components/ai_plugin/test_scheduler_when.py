"""Turning what the model says about time into a moment."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from .test_conversation import _baseline_orch  # noqa: F401  # isort: skip (stubs HA first)

from custom_components.ai_plugin.scheduler.when import (
    WhenError,
    describe,
    next_occurrence,
    parse_clock,
    parse_day,
    resolve,
)

TZ = ZoneInfo("Europe/Berlin")
# A Sunday.
NOW = datetime(2026, 11, 1, 10, 0, tzinfo=TZ)


@pytest.mark.parametrize(
    ("text", "hour", "minute"),
    [
        ("07:00", 7, 0), ("7", 7, 0), ("19.30", 19, 30), ("7:05", 7, 5), ("7 pm", 19, 0),
        ("7pm", 19, 0), ("12am", 0, 0), ("12 pm", 12, 0), ("6:30 a.m.", 6, 30), ("22h15", 22, 15),
    ],
)
def test_parse_clock(text, hour, minute) -> None:
    clock = parse_clock(text)
    assert (clock.hour, clock.minute) == (hour, minute)


@pytest.mark.parametrize("text", ["", None, "soon", "25:00", "7:75", "13pm", "0am"])
def test_parse_clock_refuses(text) -> None:
    with pytest.raises(WhenError):
        parse_clock(text)


@pytest.mark.parametrize(
    ("word", "iso"),
    [
        ("today", "2026-11-01"), ("tomorrow", "2026-11-02"), ("morgen", "2026-11-02"),
        ("Friday", "2026-11-06"), ("freitag", "2026-11-06"), ("sunday", "2026-11-01"),
        ("2026-12-24", "2026-12-24"),
    ],
)
def test_parse_day(word, iso) -> None:
    assert parse_day(word, NOW.date()).isoformat() == iso


def test_parse_day_refuses_what_it_cannot_read() -> None:
    with pytest.raises(WhenError, match="today, tomorrow"):
        parse_day("next week", NOW.date())


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"at": "11:00"}, "2026-11-01 11:00"),  # still ahead today
        ({"at": "07:00"}, "2026-11-02 07:00"),  # already passed: next morning
        ({"at": "07:00", "day": "tomorrow"}, "2026-11-02 07:00"),
        ({"at": "10:00", "day": "friday"}, "2026-11-06 10:00"),
        ({"at": "09:00", "day": "sunday"}, "2026-11-08 09:00"),  # passed today: next Sunday
        ({"at": "11:00", "day": "sunday"}, "2026-11-01 11:00"),
        ({"in_minutes": 30}, "2026-11-01 10:30"),
        ({"in_minutes": "120"}, "2026-11-01 12:00"),
        ({"at": "07:00", "repeat": "weekdays"}, "2026-11-02 07:00"),
        ({"at": "11:00", "repeat": "weekdays"}, "2026-11-02 11:00"),  # today is a Sunday
        ({"at": "08:00", "repeat": "weekends"}, "2026-11-07 08:00"),
    ],
)
def test_resolve(kwargs, expected) -> None:
    assert resolve(NOW, **kwargs).strftime("%Y-%m-%d %H:%M") == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "what time"),
        ({"day": "tomorrow"}, "what time"),
        ({"at": "09:00", "day": "today"}, "already passed"),
        ({"at": "09:00", "day": "2026-01-01"}, "already passed"),
        ({"in_minutes": 0}, "more than 0"),
        ({"in_minutes": "soon"}, "number"),
        ({"in_minutes": 5, "repeat": "daily"}, "clock time"),
        ({"at": "09:00", "repeat": "monthly"}, "'repeat'"),
    ],
)
def test_resolve_explains_what_is_wrong(kwargs, message) -> None:
    with pytest.raises(WhenError, match=message):
        resolve(NOW, **kwargs)


def test_next_occurrence_skips_the_weekend() -> None:
    friday = datetime(2026, 11, 6, 7, 0, tzinfo=TZ)
    after = friday + timedelta(minutes=1)
    assert next_occurrence(friday, "weekdays", after).date().isoformat() == "2026-11-09"
    assert next_occurrence(friday, "weekends", after).date().isoformat() == "2026-11-07"
    assert next_occurrence(friday, "daily", after).date().isoformat() == "2026-11-07"
    assert next_occurrence(friday, "weekly", after).date().isoformat() == "2026-11-13"
    assert next_occurrence(friday, "biweekly", after).date().isoformat() == "2026-11-20"


def test_next_occurrence_catches_up_after_days_of_downtime() -> None:
    old = datetime(2026, 11, 2, 7, 0, tzinfo=TZ)
    later = datetime(2026, 11, 9, 8, 0, tzinfo=TZ)
    assert next_occurrence(old, "daily", later).date().isoformat() == "2026-11-10"


def test_a_daily_alarm_keeps_its_wall_clock_across_the_clock_change() -> None:
    """Clocks went back on 2026-10-25 in Berlin."""
    before = datetime(2026, 10, 24, 7, 0, tzinfo=TZ)
    after = datetime(2026, 10, 24, 7, 1, tzinfo=TZ)
    moment = next_occurrence(before, "daily", after, "07:00")
    assert moment.strftime("%Y-%m-%d %H:%M") == "2026-10-25 07:00"
    assert moment.utcoffset() == timedelta(hours=1)
    assert before.utcoffset() == timedelta(hours=2)


@pytest.mark.parametrize(
    ("moment", "text"),
    [
        (datetime(2026, 11, 1, 18, 0, tzinfo=TZ), "today at 18:00"),
        (datetime(2026, 11, 2, 7, 0, tzinfo=TZ), "tomorrow (Monday) at 07:00"),
        (datetime(2026, 11, 6, 10, 0, tzinfo=TZ), "Friday at 10:00"),
        (datetime(2026, 12, 24, 9, 0, tzinfo=TZ), "Thursday 2026-12-24 at 09:00"),
    ],
)
def test_describe(moment, text) -> None:
    assert describe(moment, NOW) == text
