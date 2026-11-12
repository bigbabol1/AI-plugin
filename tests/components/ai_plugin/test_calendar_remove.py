"""Removing an event from a calendar by voice."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from .test_conversation import _baseline_orch  # noqa: F401  # isort: skip (stubs HA first)

from custom_components.ai_plugin.orchestrator.grounding import _no_write_landed
from custom_components.ai_plugin.shortcuts import registry
from custom_components.ai_plugin.tools.calendar_events import (
    REMOVE_TOOL_NAME,
    _matches,
    async_remove_event,
)

TZ = ZoneInfo("Europe/Berlin")
NOW = datetime(2026, 11, 10, 15, 0, tzinfo=TZ)


def _event(summary, start, uid="u1", rrule=None, recurrence_id=None):
    return SimpleNamespace(
        summary=summary, start=start, uid=uid, rrule=rrule, recurrence_id=recurrence_id
    )


def _hass(events, features=7, name="Home"):
    hass = MagicMock()
    hass.states.async_all = MagicMock(return_value=[SimpleNamespace(
        entity_id="calendar.home",
        attributes={"friendly_name": name, "supported_features": features},
    )])
    entity = SimpleNamespace(
        async_get_events=AsyncMock(return_value=list(events)),
        async_delete_event=AsyncMock(),
    )
    hass.data = {"calendar": SimpleNamespace(get_entity={"calendar.home": entity}.get)}
    hass.entity = entity
    return hass


VIEWING = _event("House viewing", datetime(2026, 11, 10, 16, 30, tzinfo=TZ))


@pytest.fixture
def exposed():
    with patch.object(registry, "is_exposed", return_value=True) as mock:
        yield mock


async def test_the_named_event_is_removed(exposed) -> None:
    hass = _hass([VIEWING])
    result = await async_remove_event(hass, {"summary": "house viewing", "day": "today"}, NOW)
    assert result == "Removed from Home: House viewing, Tuesday 2026-11-10 at 16:30."
    hass.entity.async_delete_event.assert_awaited_once_with("u1", recurrence_id=None)
    start, end = hass.entity.async_get_events.await_args.args[1:]
    assert start == datetime(2026, 11, 10, 0, 0, tzinfo=TZ)
    assert end == datetime(2026, 11, 11, 0, 0, tzinfo=TZ)


async def test_a_misheard_name_removes_nothing_and_lists_the_day(exposed) -> None:
    """"The house viewing" came out of speech recognition as "the hospital"."""
    hass = _hass([VIEWING])
    result = await async_remove_event(hass, {"summary": "the hospital", "day": "today"}, NOW)
    assert result.startswith("[Not removed: nothing called 'the hospital' on Tuesday 2026-11-10.")
    assert "House viewing, Tuesday 2026-11-10 at 16:30" in result
    assert "remove nothing until they say which" in result
    hass.entity.async_delete_event.assert_not_awaited()


async def test_the_only_event_of_a_day_goes_when_only_the_day_was_named(exposed) -> None:
    hass = _hass([VIEWING])
    result = await async_remove_event(hass, {"summary": "today's appointment", "day": "today"}, NOW)
    assert result.startswith("Removed from Home: House viewing")


async def test_several_matches_ask_which(exposed) -> None:
    hass = _hass([VIEWING, _event("House viewing", datetime(2026, 11, 12, 10, 0, tzinfo=TZ), uid="u2")])
    result = await async_remove_event(hass, {"summary": "house viewing"}, NOW)
    assert result.startswith("[Not removed: several events match:")
    assert "Thursday 2026-11-12 at 10:00" in result
    hass.entity.async_delete_event.assert_not_awaited()


async def test_without_a_day_the_next_two_weeks_are_searched(exposed) -> None:
    hass = _hass([VIEWING])
    await async_remove_event(hass, {"summary": "viewing"}, NOW)
    start, end = hass.entity.async_get_events.await_args.args[1:]
    assert (start, end) == (NOW, datetime(2026, 11, 24, 15, 0, tzinfo=TZ))


async def test_a_repeating_event_loses_only_this_time(exposed) -> None:
    yoga = _event("Yoga", datetime(2026, 11, 10, 18, 0, tzinfo=TZ), rrule="FREQ=WEEKLY", recurrence_id="20261110T170000Z")
    hass = _hass([yoga])
    result = await async_remove_event(hass, {"summary": "yoga", "day": "today"}, NOW)
    hass.entity.async_delete_event.assert_awaited_once_with("u1", recurrence_id="20261110T170000Z")
    assert result.endswith("(only this time; it still repeats).")


async def test_an_all_day_event(exposed) -> None:
    hass = _hass([_event("Holiday", date(2026, 11, 10))])
    result = await async_remove_event(hass, {"summary": "holiday", "day": "today"}, NOW)
    assert result == "Removed from Home: Holiday, Tuesday 2026-11-10, all day."


async def test_an_empty_day_says_so(exposed) -> None:
    result = await async_remove_event(_hass([]), {"summary": "dentist", "day": "friday"}, NOW)
    assert result == "[Not removed: nothing called 'dentist' on Friday 2026-11-13. The calendar has no events then. Tell the user.]"


async def test_a_calendar_that_cannot_delete_or_is_not_exposed_is_left_alone() -> None:
    with patch.object(registry, "is_exposed", return_value=True):
        assert "no calendar that allows removing" in await async_remove_event(_hass([VIEWING], features=1), {"summary": "x"}, NOW)
    with patch.object(registry, "is_exposed", return_value=False):
        hass = _hass([VIEWING])
        assert "no calendar that allows removing" in await async_remove_event(hass, {"summary": "house viewing"}, NOW)
        hass.entity.async_get_events.assert_not_awaited()


async def test_a_bad_day_and_a_failing_delete_go_back_to_the_model(exposed) -> None:
    assert (await async_remove_event(_hass([VIEWING]), {"day": "someday"}, NOW)).startswith("[Not removed: 'day' must be")
    hass = _hass([VIEWING])
    hass.entity.async_delete_event.side_effect = RuntimeError("locked")
    assert await async_remove_event(hass, {"summary": "house viewing", "day": "today"}, NOW) == "[Not removed from Home: locked]"


async def test_an_event_without_an_id_cannot_be_removed(exposed) -> None:
    hass = _hass([_event("Workday", date(2026, 11, 10), uid=None)])
    assert "cannot be removed" in await async_remove_event(hass, {"summary": "workday", "day": "today"}, NOW)


@pytest.mark.parametrize(("asked", "summary", "expected"), [
    ("dentist", "Dentist Dr. Weber", True),
    ("the house viewing", "House viewing", True),
    ("Zahnarzt", "Zahnarzttermin", True),
    ("the hospital", "House viewing", False),
    ("yoga", "Dentist", False),
    ("appointment", "Dentist", True),
])
def test_what_counts_as_naming_the_event(asked, summary, expected) -> None:
    assert _matches(asked, summary) is expected


def test_a_removal_counts_as_an_action_done() -> None:
    call = {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": REMOVE_TOOL_NAME, "arguments": "{}"}}]}
    ok = {"role": "tool", "tool_call_id": "1", "name": REMOVE_TOOL_NAME, "content": "Removed from Home: X."}
    bad = dict(ok, content="[Not removed: nothing.]")
    assert _no_write_landed([call, ok]) is False
    assert _no_write_landed([call, bad]) is True


async def test_an_event_the_user_did_not_name_is_not_removed(exposed) -> None:
    """The model picked the house viewing for "the hospital" and removed it unasked."""
    hass = _hass([VIEWING])
    result = await async_remove_event(
        hass, {"summary": "House viewing", "day": "today"}, NOW,
        user_message="There the appointment for today, the hospital and cancel it.",
    )
    assert result == (
        "[Not removed: the user has not named 'House viewing'. Ask whether they mean "
        "House viewing, Tuesday 2026-11-10 at 16:30; remove it only after they say yes.]"
    )
    hass.entity.async_delete_event.assert_not_awaited()


async def test_a_yes_to_a_reply_that_named_it_removes_it(exposed) -> None:
    hass = _hass([VIEWING])
    result = await async_remove_event(
        hass, {"summary": "House viewing", "day": "today"}, NOW, user_message="Yes.",
        previous_reply="I see a house viewing at 4:30 PM. Would you like me to cancel that one?",
    )
    assert result.startswith("Removed from Home: House viewing")


async def test_naming_it_now_is_enough(exposed) -> None:
    hass = _hass([VIEWING])
    result = await async_remove_event(
        hass, {"summary": "House viewing"}, NOW, user_message="Cancel the viewing today.",
    )
    assert result.startswith("Removed from Home: House viewing")


async def test_the_orchestrator_hands_over_the_message_and_the_reply_before() -> None:
    from custom_components.ai_plugin.orchestrator import core

    orch = _baseline_orch()
    orch._hass = None
    orch._scheduler = MagicMock()
    orch._scheduler.now = MagicMock(return_value=NOW)
    remove = AsyncMock(return_value="Removed")
    token = core._PREVIOUS_REPLY.set("Shall I cancel the house viewing?")
    try:
        with patch.object(core, "async_remove_event", remove):
            assert await orch._dispatch_tool(REMOVE_TOOL_NAME, {"summary": "x"}, None, user_message="Yes.") == "Removed"
    finally:
        core._PREVIOUS_REPLY.reset(token)
    assert remove.await_args.kwargs == {"user_message": "Yes.", "previous_reply": "Shall I cancel the house viewing?"}
    assert orch._is_lookup_tool(REMOVE_TOOL_NAME) is False


SATURDAY_TEN = _event("Test appointment", datetime(2026, 11, 14, 10, 0, tzinfo=TZ), uid="u10")
SATURDAY_MIDNIGHT = _event("Test 2", datetime(2026, 11, 14, 0, 0, tzinfo=TZ), uid="u0")


@pytest.mark.parametrize(
    "arguments",
    [
        # 2026-11-12: "Delete the test appointment" matched "Test 2" as well.
        {"summary": "the test appointment", "day": "saturday"},
        # The answer to "which one?": "The one at 10."
        {"summary": "test", "day": "saturday", "at": "10:00"},
        {"day": "saturday", "at": "10"},
    ],
)
async def test_the_time_or_the_whole_name_picks_one_of_several(exposed, arguments) -> None:
    hass = _hass([SATURDAY_MIDNIGHT, SATURDAY_TEN])
    result = await async_remove_event(
        hass, arguments, NOW, user_message="The one at 10.",
        previous_reply="Which one, the test appointment at 10:00 or Test 2 at midnight?",
    )
    assert result.startswith("Removed from Home: Test appointment, Saturday 2026-11-14 at 10:00")
    hass.entity.async_delete_event.assert_awaited_once_with("u10", recurrence_id=None)


async def test_a_time_that_matches_none_still_asks(exposed) -> None:
    hass = _hass([SATURDAY_MIDNIGHT, SATURDAY_TEN])
    result = await async_remove_event(hass, {"summary": "test", "day": "saturday", "at": "15:00"}, NOW)
    assert result.startswith("[Not removed: several events match:")
    assert "call again with its time as 'at'" in result
    hass.entity.async_delete_event.assert_not_awaited()
