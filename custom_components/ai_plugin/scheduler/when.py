"""Turning what a small model says about time into a moment.

The model passes a clock time and a day word, and the plugin does the date
arithmetic: "friday" is resolved here, so a model that miscounts days cannot
put an alarm on the wrong date. Errors are written for the model to read and
act on. No Home Assistant imports, so all of this is plain to test.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import re

REPEATS = ("daily", "weekdays", "weekends", "weekly", "biweekly")

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAYS_DE = ("montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag")
_DAY_OFFSETS = {"today": 0, "heute": 0, "tomorrow": 1, "morgen": 1}
_CLOCK_RE = re.compile(r"^(\d{1,2})(?:[:.h](\d{2}))?\s*([ap])?\.?(?:m\.?)?$", re.IGNORECASE)
_MAX_MINUTES = 366 * 24 * 60


class WhenError(ValueError):
    """The time could not be understood; the message tells the model why."""


def parse_clock(text: object) -> time:
    """A clock time from "07:00", "7", "19.30" or "7 pm"."""
    match = _CLOCK_RE.match(str(text or "").strip())
    if match is None:
        raise WhenError(f"'at' must be a clock time like 07:00, not {text!r}.")
    hour, minute, half = int(match[1]), int(match[2] or 0), (match[3] or "").lower()
    if half:
        if not 1 <= hour <= 12:
            raise WhenError(f"{text!r} is not a clock time.")
        hour = hour % 12 + (12 if half == "p" else 0)
    if hour > 23 or minute > 59:
        raise WhenError(f"{text!r} is not a clock time.")
    return time(hour, minute)


def _weekday_index(word: str) -> int | None:
    for names in (_WEEKDAYS, _WEEKDAYS_DE):
        if word in names:
            return names.index(word)
    return None


def parse_day(text: object, today: date) -> date:
    """A date from today/tomorrow, a weekday (the next one, today included)
    or YYYY-MM-DD."""
    word = str(text or "").strip().lower()
    if word in _DAY_OFFSETS:
        return today + timedelta(days=_DAY_OFFSETS[word])
    weekday = _weekday_index(word)
    if weekday is not None:
        return today + timedelta(days=(weekday - today.weekday()) % 7)
    try:
        return date.fromisoformat(word)
    except ValueError:
        raise WhenError(
            f"'day' must be today, tomorrow, a weekday or YYYY-MM-DD, not {text!r}."
        ) from None


def _fits_repeat(day: date, repeat: str | None) -> bool:
    if repeat == "weekdays":
        return day.weekday() < 5
    if repeat == "weekends":
        return day.weekday() >= 5
    return True


def resolve(
    now: datetime,
    *,
    at: object = None,
    day: object = None,
    in_minutes: object = None,
    repeat: str | None = None,
) -> datetime:
    """The moment meant, always later than ``now``. ``now`` is timezone-aware
    and local; so is the result."""
    if repeat is not None and repeat not in REPEATS:
        raise WhenError(f"'repeat' must be one of {', '.join(REPEATS)}.")
    if in_minutes not in (None, ""):
        if repeat:
            raise WhenError("A repeat needs a clock time: use 'at', not 'in_minutes'.")
        try:
            minutes = float(in_minutes)
        except (TypeError, ValueError):
            raise WhenError(f"'in_minutes' must be a number, not {in_minutes!r}.") from None
        if not 0 < minutes <= _MAX_MINUTES:
            raise WhenError("'in_minutes' must be more than 0 and at most a year.")
        return now + timedelta(minutes=minutes)
    if at in (None, ""):
        raise WhenError("Ask the user what time: this needs 'at' or 'in_minutes'.")
    clock = parse_clock(at)
    today = now.date()
    if day in (None, ""):
        moment = datetime.combine(today, clock, tzinfo=now.tzinfo)
        if moment <= now:
            moment = datetime.combine(today + timedelta(days=1), clock, tzinfo=now.tzinfo)
    else:
        moment = datetime.combine(parse_day(day, today), clock, tzinfo=now.tzinfo)
        if moment <= now:
            if _weekday_index(str(day).strip().lower()) is None:
                raise WhenError(f"{describe(moment, now)} has already passed.")
            moment += timedelta(days=7)
    while not _fits_repeat(moment.date(), repeat):
        moment = datetime.combine(moment.date() + timedelta(days=1), clock, tzinfo=now.tzinfo)
    return moment


def next_occurrence(due: datetime, repeat: str, now: datetime, clock: str | None = None) -> datetime:
    """The next time a repeating job runs after ``now``, at its wall-clock
    time, so a daily 07:00 stays 07:00 across a daylight-saving change."""
    step = {"weekly": 7, "biweekly": 14}.get(repeat, 1)
    wall = parse_clock(clock) if clock else due.time()
    day = due.astimezone(now.tzinfo).date()
    while True:
        day += timedelta(days=step)
        moment = datetime.combine(day, wall, tzinfo=now.tzinfo)
        if moment > now and _fits_repeat(day, repeat):
            return moment


def describe(moment: datetime, now: datetime) -> str:
    """"tomorrow (Monday) at 07:00": unambiguous for the model to repeat."""
    days = (moment.date() - now.date()).days
    clock = moment.strftime("%H:%M")
    weekday = _WEEKDAYS[moment.weekday()].capitalize()
    if days == 0:
        return f"today at {clock}"
    if days == 1:
        return f"tomorrow ({weekday}) at {clock}"
    if 1 < days < 7:
        return f"{weekday} at {clock}"
    return f"{weekday} {moment.date().isoformat()} at {clock}"
