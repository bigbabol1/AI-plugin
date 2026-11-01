"""Adding an event to a calendar by voice."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from .test_conversation import _baseline_orch  # noqa: F401  # isort: skip (stubs HA first)

from custom_components.ai_plugin.shortcuts import registry
from custom_components.ai_plugin.tools.calendar_events import async_add_event

TZ = ZoneInfo("Europe/Berlin")
NOW = datetime(2026, 11, 1, 10, 0, tzinfo=TZ)


def _calendar(entity_id, name, features=7):
    return SimpleNamespace(
        entity_id=entity_id,
        attributes={"friendly_name": name, "supported_features": features},
    )


def _hass(*calendars):
    hass = MagicMock()
    hass.states.async_all = MagicMock(return_value=list(calendars))
    entities = {
        c.entity_id: SimpleNamespace(async_create_event=AsyncMock()) for c in calendars
    }
    hass.data = {"calendar": SimpleNamespace(get_entity=entities.get)}
    hass.entities = entities
    return hass


def _home():
    return _hass(_calendar("calendar.home", "Home"))


def _created(hass, entity_id="calendar.home") -> dict:
    return hass.entities[entity_id].async_create_event.await_args.kwargs


@pytest.fixture
def exposed():
    with patch.object(registry, "is_exposed", return_value=True) as mock:
        yield mock


async def test_a_timed_event(exposed) -> None:
    hass = _home()
    result = await async_add_event(hass, {"summary": "Dentist", "day": "friday", "at": "10:00"}, NOW)
    assert result == "Added to Home: Dentist, Friday 2026-11-06 at 10:00."
    assert _created(hass) == {
        "summary": "Dentist",
        "dtstart": datetime(2026, 11, 6, 10, 0, tzinfo=TZ),
        "dtend": datetime(2026, 11, 6, 11, 0, tzinfo=TZ),
    }


async def test_duration(exposed) -> None:
    hass = _home()
    await async_add_event(
        hass, {"summary": "x", "day": "tomorrow", "at": "9:00", "duration_minutes": 30}, NOW
    )
    assert _created(hass)["dtend"] == datetime(2026, 11, 2, 9, 30, tzinfo=TZ)


async def test_no_time_means_all_day(exposed) -> None:
    hass = _home()
    result = await async_add_event(hass, {"summary": "Trip", "day": "2026-12-24"}, NOW)
    assert "all day" in result and "every" not in result
    assert _created(hass) == {
        "summary": "Trip", "dtstart": date(2026, 12, 24), "dtend": date(2026, 12, 25),
    }


@pytest.mark.parametrize(
    ("summary", "message", "language"),
    [
        ("Anna's birthday", "Note down Anna's birthday on the 1st of October.", "en-US"),
        ("Birthday", "Note my birthday on the 20th of July in my calendar.", "en-US"),
        ("Anna", "Put Anna's birthday on my calendar.", "en-US"),
        ("Annas Geburtstag", "Trag Annas Geburtstag ein.", "de-DE"),
        ("Anniversaire d'Anna", "", "fr-FR"),
        ("Cumpleaños de Ana", "", "es-ES"),
        ("Aniversário da Ana", "", "pt-PT"),
        ("Urodziny Anny", "", "pl-PL"),
        ("Verjaardag van Anna", "", "nl-NL"),
    ],
)
async def test_a_birthday_repeats_every_year(exposed, summary, message, language) -> None:
    """Also when its date this year has gone by: it comes round again."""
    hass = _home()
    result = await async_add_event(
        hass, {"summary": summary, "day": "2026-10-01"}, NOW,
        user_message=message, language=language,
    )
    assert result.endswith(", every year.")
    event = _created(hass)
    assert event["rrule"] == "FREQ=YEARLY"
    assert event["dtstart"] == date(2026, 10, 1)


@pytest.mark.parametrize(
    ("summary", "message", "language"),
    [
        ("Birthday party", "Put Anna's birthday party on my calendar.", "en-US"),
        ("Birthday dinner", "", "en-US"),
        ("Geburtstagsfeier", "Trag die Geburtstagsfeier ein.", "de-DE"),
        ("Fête d'anniversaire", "", "fr-FR"),
        ("Fiesta de cumpleaños", "", "es-ES"),
        ("Verjaardagsfeest", "", "nl-NL"),
    ],
)
async def test_a_birthday_party_happens_once(exposed, summary, message, language) -> None:
    hass = _home()
    await async_add_event(
        hass, {"summary": summary, "day": "friday", "at": "15:00"}, NOW,
        user_message=message, language=language,
    )
    assert "rrule" not in _created(hass)


@pytest.mark.parametrize(
    ("repeat", "rrule", "words"),
    [("weekly", "FREQ=WEEKLY", "every week"), ("monthly", "FREQ=MONTHLY", "every month"),
     ("yearly", "FREQ=YEARLY", "every year")],
)
async def test_a_repeat_the_user_asked_for(exposed, repeat, rrule, words) -> None:
    hass = _home()
    result = await async_add_event(
        hass, {"summary": "Gym", "day": "monday", "at": "7:00", "repeat": repeat}, NOW
    )
    assert _created(hass)["rrule"] == rrule
    assert result.endswith(f", {words}.")


async def test_saying_no_repeat_beats_the_birthday_default(exposed) -> None:
    hass = _home()
    await async_add_event(hass, {"summary": "Anna's birthday", "day": "friday", "repeat": "none"}, NOW)
    assert "rrule" not in _created(hass)


async def test_a_one_off_date_gone_by_is_refused(exposed) -> None:
    """Live: "added" for an entry nobody would ever see again."""
    hass = _home()
    result = await async_add_event(hass, {"summary": "Dentist", "day": "2026-10-01"}, NOW)
    assert result.startswith("[Not added") and "already passed" in result
    hass.entities["calendar.home"].async_create_event.assert_not_awaited()


async def test_read_only_and_unexposed_calendars_are_not_offered(exposed) -> None:
    """The holiday calendar takes no events, and what was not shared with the
    assistant stays out of reach."""
    hass = _hass(_calendar("calendar.holidays", "Holidays", features=0), _calendar("calendar.private", "Private"))
    exposed.side_effect = lambda _hass, eid: eid != "calendar.private"
    result = await async_add_event(hass, {"summary": "x", "day": "today"}, NOW)
    assert result.startswith("[Not added") and "exposed to Assist" in result
    for entity in hass.entities.values():
        entity.async_create_event.assert_not_awaited()


async def test_several_calendars_ask_which(exposed) -> None:
    hass = _hass(_calendar("calendar.home", "Home"), _calendar("calendar.work", "Work"))
    args = {"summary": "x", "day": "tomorrow", "at": "9:00"}
    assert await async_add_event(hass, args, NOW) == "[Not added: ask which calendar: Home, Work.]"
    hass.entities["calendar.home"].async_create_event.assert_not_awaited()
    await async_add_event(hass, {**args, "calendar": "work"}, NOW)
    hass.entities["calendar.work"].async_create_event.assert_awaited_once()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"day": "friday"}, "summary"),
        ({"summary": "x", "day": "someday"}, "today, tomorrow"),
        ({"summary": "x", "day": "friday", "at": "soon"}, "clock time"),
        ({"summary": "x", "day": "friday", "at": "9:00", "duration_minutes": "long"}, "duration"),
    ],
)
async def test_what_it_cannot_read_goes_back_to_the_model(exposed, args, message) -> None:
    hass = _home()
    result = await async_add_event(hass, args, NOW)
    assert result.startswith("[Not added") and message in result
    hass.entities["calendar.home"].async_create_event.assert_not_awaited()


async def test_a_failing_calendar_says_so(exposed) -> None:
    hass = _home()
    hass.entities["calendar.home"].async_create_event.side_effect = RuntimeError("read only")
    result = await async_add_event(hass, {"summary": "x", "day": "today"}, NOW)
    assert result == "[Not added to Home: read only]"


async def test_a_calendar_gone_from_home_assistant_says_so(exposed) -> None:
    hass = _home()
    hass.entities.clear()
    hass.data["calendar"] = SimpleNamespace(get_entity=lambda _: None)
    result = await async_add_event(hass, {"summary": "x", "day": "today"}, NOW)
    assert result == "[Not added: Home is not available right now.]"
