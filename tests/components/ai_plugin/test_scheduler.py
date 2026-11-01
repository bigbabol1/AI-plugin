"""Alarms, reminders and commands for later: the engine and its tools."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from .test_conversation import _voice_orch_with_intent  # isort: skip (stubs HA first)

from custom_components.ai_plugin import scheduler as pkg
from custom_components.ai_plugin.i18n import LOCALIZATIONS, L
from custom_components.ai_plugin.orchestrator import core
from custom_components.ai_plugin.providers import ChatResponse, ToolCall
from custom_components.ai_plugin.scheduler import engine
from custom_components.ai_plugin.scheduler.engine import (
    KIND_ALARM,
    KIND_COMMAND,
    KIND_REMINDER,
    Scheduler,
)
from custom_components.ai_plugin.scheduler.tools import (
    async_call_tool,
    names_later_time,
    strip_later_time,
)

TZ = ZoneInfo("Europe/Berlin")
NOW = datetime(2026, 11, 1, 10, 0, tzinfo=TZ)
SAT = "sat-bedroom"


@pytest.fixture
def announce():
    with patch.object(
        engine.satellite_speech, "async_announce", AsyncMock(return_value="spoken")
    ) as mock:
        yield mock


@pytest.fixture
def track():
    with patch.object(engine, "async_track_point_in_utc_time", MagicMock()) as mock:
        yield mock


@pytest.fixture
def quick_rings():
    with patch.object(engine, "RINGS", 3), patch.object(engine, "RING_GAP_S", 0.0):
        yield


def _scheduler(now=NOW) -> Scheduler:
    hass = MagicMock()
    hass.services.async_call = AsyncMock()
    scheduler = Scheduler(hass, "e1")
    scheduler.now = lambda: now
    return scheduler


async def _settle(scheduler) -> None:
    while scheduler._tasks:
        await asyncio.gather(*list(scheduler._tasks), return_exceptions=True)


async def _call(scheduler, name, args, *, device_id=SAT, person_id=None, message="", language="en-US"):
    return await async_call_tool(
        scheduler, scheduler._hass, name, args,
        device_id=device_id, user_id=None, person_id=person_id,
        language=language, user_message=message,
    )


# ── storage and restarts ─────────────────────────────────────────────────────


async def _restarted(scheduler, now) -> Scheduler:
    fresh = _scheduler(now)
    fresh._store = scheduler._store
    await fresh.async_load()
    return fresh


async def test_jobs_survive_a_restart(track) -> None:
    scheduler = _scheduler()
    due = NOW + timedelta(hours=2)
    await scheduler.async_add(KIND_ALARM, due, text="gym", device_id=SAT, language="de-DE")
    fresh = await _restarted(scheduler, NOW + timedelta(hours=1))
    [job] = fresh.jobs()
    assert (job.kind, job.text, job.device_id, job.language) == (KIND_ALARM, "gym", SAT, "de-DE")
    assert job.due == due
    assert track.call_args.args[2] == due


@pytest.mark.parametrize(
    ("kind", "late", "kept"),
    [
        (KIND_ALARM, timedelta(minutes=5), True),
        (KIND_ALARM, timedelta(hours=1), False),
        # A light switching off hours late would be a surprise.
        (KIND_COMMAND, timedelta(minutes=5), True),
        (KIND_COMMAND, timedelta(hours=1), False),
        # A reminder is still worth hearing.
        (KIND_REMINDER, timedelta(hours=20), True),
        (KIND_REMINDER, timedelta(hours=30), False),
    ],
)
async def test_what_came_due_during_downtime(track, kind, late, kept) -> None:
    scheduler = _scheduler()
    await scheduler.async_add(kind, NOW + timedelta(minutes=1), text="x", device_id=SAT)
    now = NOW + timedelta(minutes=1) + late
    fresh = await _restarted(scheduler, now)
    if not kept:
        assert fresh.jobs() == []
        return
    # Not at once: the satellites are not back yet.
    assert fresh.jobs()[0].due == now + timedelta(seconds=engine.AFTER_RESTART_S)


async def test_a_missed_repeating_alarm_moves_to_its_next_time(track) -> None:
    scheduler = _scheduler()
    due = NOW + timedelta(hours=1)
    await scheduler.async_add(KIND_ALARM, due, device_id=SAT, repeat="daily", clock="11:00")
    fresh = await _restarted(scheduler, NOW + timedelta(days=2))
    assert fresh.jobs()[0].due == datetime(2026, 11, 3, 11, 0, tzinfo=TZ)


async def test_unreadable_stored_job_is_dropped(track) -> None:
    scheduler = _scheduler()
    await scheduler._store.async_save({"jobs": [{"kind": "alarm"}]})
    await scheduler.async_load()
    assert scheduler.jobs() == []


async def test_timer_stays_on_the_earliest_job(track) -> None:
    scheduler = _scheduler()
    later, sooner = NOW + timedelta(hours=5), NOW + timedelta(hours=1)
    await scheduler.async_add(KIND_REMINDER, later, text="a", device_id=SAT)
    first_timer = track.return_value
    await scheduler.async_add(KIND_REMINDER, sooner, text="b", device_id=SAT)
    assert track.call_args.args[2] == sooner
    first_timer.assert_called()
    await scheduler.async_cancel([j for j in scheduler.jobs() if j.text == "b"])
    assert track.call_args.args[2] == later


# ── firing ───────────────────────────────────────────────────────────────────


async def test_a_one_shot_job_runs_once_and_is_gone(track, announce) -> None:
    scheduler = _scheduler()
    await scheduler.async_add(KIND_REMINDER, NOW, text="trash", device_id=SAT, language="en-US")
    await scheduler._async_on_timer()
    await _settle(scheduler)
    announce.assert_awaited_once_with(scheduler._hass, SAT, "reminder_due", "en", label="trash")
    assert scheduler.jobs() == []
    assert (await scheduler._store.async_load())["jobs"] == []


async def test_a_repeating_job_moves_on(track, announce) -> None:
    scheduler = _scheduler()
    await scheduler.async_add(
        KIND_REMINDER, NOW, text="pills", device_id=SAT, repeat="daily", clock="10:00"
    )
    await scheduler._async_on_timer()
    await _settle(scheduler)
    assert scheduler.jobs()[0].due == NOW + timedelta(days=1)
    announce.assert_awaited_once()


async def test_a_job_not_yet_due_is_left_alone(track, announce) -> None:
    scheduler = _scheduler()
    await scheduler.async_add(KIND_REMINDER, NOW + timedelta(minutes=1), text="x", device_id=SAT)
    await scheduler._async_on_timer()
    await _settle(scheduler)
    announce.assert_not_awaited()
    assert len(scheduler.jobs()) == 1


async def test_one_failing_job_does_not_stop_the_next(track, announce) -> None:
    scheduler = _scheduler()
    announce.side_effect = [RuntimeError("boom"), "spoken"]
    await scheduler.async_add(KIND_REMINDER, NOW, text="a", device_id=SAT)
    await scheduler.async_add(KIND_REMINDER, NOW, text="b", device_id=SAT)
    await scheduler._async_on_timer()
    await _settle(scheduler)
    assert announce.await_count == 2


# ── reminders ────────────────────────────────────────────────────────────────


async def test_reminder_also_goes_to_the_persons_phone(track, announce) -> None:
    scheduler = _scheduler()
    phone = SimpleNamespace(service="mobile_app_lena", name="Lena")
    await scheduler.async_add(
        KIND_REMINDER, NOW, text="trash", device_id=SAT, person_id="person.lena", language="de-DE"
    )
    with patch("custom_components.ai_plugin.shortcuts.phone.phones_of", return_value=[phone]):
        await scheduler._async_on_timer()
        await _settle(scheduler)
    announce.assert_awaited_once()
    call = scheduler._hass.services.async_call.await_args
    assert call.args[:2] == ("notify", "mobile_app_lena")
    assert call.args[2] == {"title": "Erinnerung", "message": "trash"}


async def test_reminder_nowhere_to_go_is_left_as_a_notification(track, announce) -> None:
    announce.return_value = None
    scheduler = _scheduler()
    await scheduler.async_add(KIND_REMINDER, NOW, text="trash", device_id=SAT)
    await scheduler._async_on_timer()
    await _settle(scheduler)
    assert scheduler._hass.services.async_call.await_args.args[:2] == (
        "persistent_notification", "create",
    )


# ── alarms ───────────────────────────────────────────────────────────────────


async def test_an_alarm_rings_several_times_then_gives_up(track, announce, quick_rings) -> None:
    scheduler = _scheduler()
    await scheduler.async_add(KIND_ALARM, NOW, device_id=SAT, language="en-US")
    await scheduler._async_on_timer()
    await _settle(scheduler)
    assert announce.await_count == 3
    assert announce.await_args.args[2] == "alarm_ring"
    assert scheduler._rings == {}


def test_it_rings_often_enough_to_read_as_one_alarm() -> None:
    """At 45 s apart it was taken for several separate alarms."""
    assert engine.RING_GAP_S == 10.0
    assert engine.RINGS * engine.RING_GAP_S == 180.0


async def test_an_alarm_with_a_purpose_says_it(track, announce, quick_rings) -> None:
    scheduler = _scheduler()
    await scheduler.async_add(KIND_ALARM, NOW, text="Time to wake up", device_id=SAT)
    await scheduler._async_on_timer()
    await _settle(scheduler)
    assert announce.await_args.args[2] == "scheduled_say"
    assert announce.await_args.kwargs["label"] == "Time to wake up"


@pytest.mark.parametrize("lang", sorted(LOCALIZATIONS))
def test_a_plain_alarm_assumes_nothing(lang) -> None:
    """It may have nothing to do with getting up, and it names no time."""
    text = L.template("alarm_ring", lang).lower()
    assert not any(c.isdigit() for c in text)
    for word in ("get up", "wake", "aufstehen", "wecker", "lever", "levantarse", "acordar", "wstawać", "opstaan"):
        assert word not in text


async def _ringing(scheduler) -> None:
    """An alarm caught between two rings."""
    await scheduler.async_add(KIND_ALARM, NOW, text="", device_id=SAT, language="en-US")
    with patch.object(engine, "RING_GAP_S", 60.0):
        await scheduler._async_on_timer()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    assert SAT in scheduler._rings


async def test_stop_ends_the_ringing(track, announce) -> None:
    scheduler = _scheduler()
    await _ringing(scheduler)
    task = scheduler._rings[SAT].task
    assert await scheduler.async_handle_spoken("Stop.", SAT, "en") == "Alarm off."
    assert scheduler._rings == {}
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert scheduler.jobs() == []


async def test_stop_from_another_room_stops_it_too(track, announce) -> None:
    scheduler = _scheduler()
    await _ringing(scheduler)
    assert await scheduler.async_handle_spoken("turn off the alarm", "sat-kitchen", "en")
    assert scheduler._rings == {}


@pytest.mark.parametrize(
    ("words", "lang", "minutes"),
    [("Snooze.", "en", 9), ("snooze for 5 minutes", "en", 5), ("3 more minutes", "en", 3),
     ("Schlummern", "de", 9), ("noch 5 Minuten", "de", 5)],
)
async def test_snooze_rings_again_later(track, announce, words, lang, minutes) -> None:
    scheduler = _scheduler()
    await _ringing(scheduler)
    reply = await scheduler.async_handle_spoken(words, SAT, lang)
    assert str(minutes) in reply
    assert scheduler._rings == {}
    [job] = scheduler.jobs()
    assert job.kind == KIND_ALARM and job.device_id == SAT
    assert job.due == NOW + timedelta(minutes=minutes)


async def test_other_words_while_ringing_are_not_taken(track, announce) -> None:
    scheduler = _scheduler()
    await _ringing(scheduler)
    assert await scheduler.async_handle_spoken("What's the weather like?", SAT, "en") is None
    assert SAT in scheduler._rings
    scheduler.async_unload()


async def test_stop_with_no_alarm_ringing_is_not_ours() -> None:
    """Otherwise "stop" could no longer stop the music."""
    assert await _scheduler().async_handle_spoken("stop", SAT, "en") is None


async def test_an_alarm_that_cannot_play_tries_the_phone(track, announce) -> None:
    announce.return_value = None
    scheduler = _scheduler()
    phone = SimpleNamespace(service="mobile_app_lena", name="Lena")
    await scheduler.async_add(KIND_ALARM, NOW, device_id=SAT, person_id="person.lena")
    with patch("custom_components.ai_plugin.shortcuts.phone.phones_of", return_value=[phone]):
        await scheduler._async_on_timer()
        await _settle(scheduler)
    assert announce.await_count == 1
    assert scheduler._hass.services.async_call.await_args.args[1] == "mobile_app_lena"


# ── commands ─────────────────────────────────────────────────────────────────


async def test_a_command_runs_and_a_visible_change_stays_silent(track, announce) -> None:
    scheduler = _scheduler()
    runner = AsyncMock(return_value="")
    scheduler.set_command_runner(runner)
    await scheduler.async_add(KIND_COMMAND, NOW, text="turn off the lights", device_id=SAT)
    await scheduler._async_on_timer()
    await _settle(scheduler)
    assert runner.await_args.args[0].text == "turn off the lights"
    announce.assert_not_awaited()


async def test_a_command_with_words_says_them(track, announce) -> None:
    scheduler = _scheduler()
    scheduler.set_command_runner(AsyncMock(return_value="There is no such device."))
    await scheduler.async_add(KIND_COMMAND, NOW, text="x", device_id=SAT, language="en-US")
    await scheduler._async_on_timer()
    await _settle(scheduler)
    announce.assert_awaited_once_with(
        scheduler._hass, SAT, "scheduled_say", "en", label="There is no such device."
    )


async def test_a_failing_command_says_so(track, announce) -> None:
    scheduler = _scheduler()
    scheduler.set_command_runner(AsyncMock(side_effect=RuntimeError("boom")))
    await scheduler.async_add(KIND_COMMAND, NOW, text="dim the lights", device_id=SAT)
    await scheduler._async_on_timer()
    await _settle(scheduler)
    assert announce.await_args.kwargs["label"] == "I couldn't carry out: dim the lights."


# ── tools ────────────────────────────────────────────────────────────────────


async def test_set_alarm(track) -> None:
    scheduler = _scheduler()
    assert await _call(scheduler, "set_alarm", {"at": "07:00"}) == (
        "Alarm set for tomorrow (Monday) at 07:00."
    )
    [job] = scheduler.jobs()
    assert (job.kind, job.device_id, job.repeat, job.clock) == (KIND_ALARM, SAT, None, None)


@pytest.mark.parametrize(
    ("message", "language", "label", "says"),
    [
        ("Set an alarm for 7.", "en-US", None, ""),
        # Replayed: the model invented these for a plain alarm.
        ("Set an alarm in 30 seconds.", "en-US", "Alarm in 30 seconds", ""),
        ("Set an alarm in 10 seconds.", "en-US", "10 seconds", ""),
        ("Set an alarm.", "en-US", "Alarm", ""),
        # Replayed: it echoed the request instead of saying what it is for.
        ("Wake me up in 10 minutes.", "en-US", "Wake me up", "Time to wake up."),
        ("Wake me up at 7.", "en-US", None, "Time to wake up."),
        ("Wecke mich um 7 Uhr.", "de-DE", "Wecke mich", "Zeit aufzuwachen."),
        ("Alarm for the pasta.", "en-US", "The pasta is ready", "The pasta is ready"),
        ("Stelle einen Alarm für die Nudeln.", "de-DE", "Die Nudeln sind fertig", "Die Nudeln sind fertig"),
    ],
)
async def test_what_an_alarm_says(track, message, language, label, says) -> None:
    scheduler = _scheduler()
    args = {"at": "07:00"} if label is None else {"at": "07:00", "label": label}
    await _call(scheduler, "set_alarm", args, message=message, language=language)
    assert scheduler.jobs()[0].text == says


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("wake me up at 7", "en"), ("get me up at 7", "en"), ("Weck mich um 7", "de"),
        ("Réveille-moi à 7 heures", "fr"), ("Despiértame a las 7", "es"),
        ("Acorda-me às 7", "pt"), ("Obudź mnie o 7", "pl"), ("Maak me om 7 uur wakker", "nl"),
    ],
)
def test_being_woken_is_recognised_in_every_language(message, lang) -> None:
    assert any(rx.search(message.lower()) for rx in L.pattern_list("alarm_wake_request", lang))
    assert L.template("alarm_wake", lang)


async def test_a_repeating_alarm_keeps_its_clock_time(track) -> None:
    scheduler = _scheduler()
    await _call(scheduler, "set_alarm", {"at": "07:00", "repeat": "weekdays"})
    [job] = scheduler.jobs()
    assert (job.repeat, job.clock) == ("weekdays", "07:00")


async def test_an_alarm_from_the_app_needs_a_room(track) -> None:
    scheduler = _scheduler()
    result = await _call(scheduler, "set_alarm", {"at": "07:00"}, device_id=None)
    assert result.startswith("[Not set") and "which room" in result
    assert scheduler.jobs() == []


async def test_an_alarm_in_a_named_room(track) -> None:
    scheduler = _scheduler()
    tools = "custom_components.ai_plugin.scheduler.tools.satellite_in_area"
    with patch(tools, return_value="sat-kitchen"):
        await _call(scheduler, "set_alarm", {"at": "07:00", "area": "kitchen"})
    assert scheduler.jobs()[0].device_id == "sat-kitchen"
    with patch(tools, return_value=None):
        result = await _call(scheduler, "set_alarm", {"at": "07:00", "area": "attic"})
    assert "no satellite in attic" in result


async def test_a_time_it_cannot_read_goes_back_to_the_model(track) -> None:
    scheduler = _scheduler()
    result = await _call(scheduler, "set_reminder", {"text": "x"})
    assert result.startswith("[Not set") and "what time" in result
    assert scheduler.jobs() == []


async def test_reminder_from_the_app_goes_to_the_phone(track) -> None:
    scheduler = _scheduler()
    result = await _call(
        scheduler, "set_reminder", {"text": "trash", "in_minutes": 5},
        device_id=None, person_id="person.lena",
    )
    assert result == "Reminder set for today at 10:05: trash."
    assert scheduler.jobs()[0].person_id == "person.lena"
    nowhere = await _call(scheduler, "set_reminder", {"text": "x", "in_minutes": 5}, device_id=None)
    assert nowhere.startswith("[Not set: nowhere")


async def test_a_reminder_for_someone_else_goes_to_their_phone(track, announce) -> None:
    """Live replay: "a reminder for Anna" went to whoever was speaking."""
    scheduler = _scheduler()
    phone = SimpleNamespace(service="mobile_app_anna", name="Anna")
    with patch("custom_components.ai_plugin.shortcuts.phone.phones_of", return_value=[phone]):
        result = await async_call_tool(
            scheduler, scheduler._hass, "set_reminder",
            {"text": "buy milk", "at": "17:00", "person": "Anna"},
            device_id=SAT, user_id=None, person_id="person.ben",
            language="en-US", resolve_person=lambda _: "person.anna",
        )
        assert result == "Reminder set for Anna for today at 17:00: buy milk."
        [job] = scheduler.jobs()
        assert (job.person_id, job.device_id) == ("person.anna", None)

        job.due = NOW
        await scheduler._async_on_timer()
        await _settle(scheduler)
    announce.assert_not_awaited()
    assert scheduler._hass.services.async_call.await_args.args[:2] == ("notify", "mobile_app_anna")


async def test_a_reminder_for_someone_without_a_phone_stays_in_this_room(track) -> None:
    scheduler = _scheduler()
    with patch("custom_components.ai_plugin.shortcuts.phone.phones_of", return_value=[]):
        await async_call_tool(
            scheduler, scheduler._hass, "set_reminder",
            {"text": "x", "in_minutes": 5, "person": "Anna"},
            device_id=SAT, user_id=None, person_id=None,
            language="en", resolve_person=lambda _: "person.anna",
        )
    assert scheduler.jobs()[0].device_id == SAT


async def test_a_reminder_for_an_unknown_name_asks_who(track) -> None:
    scheduler = _scheduler()
    result = await async_call_tool(
        scheduler, scheduler._hass, "set_reminder",
        {"text": "x", "in_minutes": 5, "person": "Zed"},
        device_id=SAT, user_id=None, person_id=None,
        language="en", resolve_person=lambda _: None,
    )
    assert "do not know who Zed is" in result
    assert scheduler.jobs() == []


async def test_a_reminder_for_the_speaker_by_name_is_an_ordinary_one(track) -> None:
    scheduler = _scheduler()
    result = await async_call_tool(
        scheduler, scheduler._hass, "set_reminder",
        {"text": "x", "in_minutes": 5, "person": "Anna"},
        device_id=SAT, user_id=None, person_id="person.anna",
        language="en", resolve_person=lambda _: "person.anna",
    )
    assert "for Anna" not in result
    assert scheduler.jobs()[0].device_id == SAT


async def test_the_spoken_duration_beats_the_models_arithmetic(track) -> None:
    scheduler = _scheduler()
    await _call(
        scheduler, "schedule_action", {"command": "turn off the lights", "in_minutes": 2},
        message="Turn off the lights in 2 hours.",
    )
    assert scheduler.jobs()[0].due == NOW + timedelta(hours=2)


@pytest.mark.parametrize(
    ("command", "lang", "kept"),
    [
        ("turn off the lights in 30 minutes", "en", "turn off the lights"),
        ("at 10 dim the lights", "en", "dim the lights"),
        ("set the heating to 21 degrees", "en", "set the heating to 21 degrees"),
        ("schalte in 30 Minuten das Licht aus", "de", "schalte das Licht aus"),
        ("mach morgen früh das Licht an", "de", "mach das Licht an"),
    ],
)
def test_a_command_is_stored_without_its_time(command, lang, kept) -> None:
    """Run later with the time still in it, it would schedule itself again."""
    assert strip_later_time(command, lang) == kept


async def test_a_command_of_only_a_time_is_refused(track) -> None:
    scheduler = _scheduler()
    result = await _call(scheduler, "schedule_action", {"command": "in 30 minutes", "in_minutes": 30})
    assert result.startswith("[Not scheduled")
    assert scheduler.jobs() == []


async def test_list_scheduled(track) -> None:
    scheduler = _scheduler()
    assert await _call(scheduler, "list_scheduled", {}) == "Nothing is scheduled."
    await _call(scheduler, "set_alarm", {"at": "07:00", "repeat": "daily"})
    await _call(scheduler, "set_reminder", {"text": "trash", "at": "18:00"})
    await _call(scheduler, "schedule_action", {"command": "dim the lights", "in_minutes": 30})
    assert (await _call(scheduler, "list_scheduled", {})).splitlines() == [
        "1. scheduled action today at 10:30: dim the lights",
        "2. reminder today at 18:00: trash",
        "3. alarm tomorrow (Monday) at 07:00, repeats daily",
    ]


async def test_cancel_one_several_all_and_none(track) -> None:
    scheduler = _scheduler()
    await _call(scheduler, "set_alarm", {"at": "07:00"})
    await _call(scheduler, "set_alarm", {"at": "08:00"})
    await _call(scheduler, "set_reminder", {"text": "trash", "at": "18:00"})

    assert "Nothing matching" in await _call(scheduler, "cancel_scheduled", {"text": "milk"})
    several = await _call(scheduler, "cancel_scheduled", {"kind": "alarm"})
    assert several.startswith("Several match") and len(scheduler.jobs()) == 3
    assert "trash" in await _call(scheduler, "cancel_scheduled", {"text": "TRASH"})
    assert len(scheduler.jobs()) == 2
    assert "no number 7" in await _call(scheduler, "cancel_scheduled", {"number": 7})
    await _call(scheduler, "cancel_scheduled", {"number": 1})
    assert scheduler.jobs()[0].due.hour == 8
    await _call(scheduler, "set_alarm", {"at": "09:00"})
    await _call(scheduler, "cancel_scheduled", {"kind": "alarm", "all": True})
    assert scheduler.jobs() == []


# ── recognising a later command ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("Turn off the lights in 30 minutes.", "en"),
        ("Turn off the lights in an hour", "en"),
        ("in half an hour turn off the TV", "en"),
        ("At 10 dim the lights", "en"),
        ("Turn on the heating tomorrow", "en"),
        ("dim the lights every evening", "en"),
        ("Schalte in 30 Minuten das Licht aus", "de"),
        ("Licht aus um 22 Uhr", "de"),
        ("Mach morgen früh das Licht an", "de"),
        ("Éteins la lumière dans 10 minutes", "fr"),
        ("Apaga la luz en 10 minutos", "es"),
        ("Desliga a luz daqui a 10 minutos", "pt"),
        ("Wyłącz światło za 10 minut", "pl"),
        ("Zet het licht over 10 minuten uit", "nl"),
    ],
)
def test_later_commands_are_recognised(message, lang) -> None:
    assert names_later_time(message, lang)


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("Turn off the lights", "en"),
        ("Dim the lights at 50%", "en"),
        ("Dim the lights at 50 percent", "en"),
        ("Set the heating to 21 degrees", "en"),
        ("Turn on the lights in the kitchen", "en"),
        ("Turn on every light", "en"),
        ("Guten Morgen", "de"),
        ("Dimm das Licht um 10 Prozent", "de"),
        ("Goede morgen", "nl"),
    ],
)
def test_commands_for_now_are_not_mistaken_for_later(message, lang) -> None:
    assert not names_later_time(message, lang)


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("When I say good night, turn off all the lights.", "en"),
        ("If I say movie time dim the lights", "en"),
        ("Wenn ich gute Nacht sage, mach alle Lichter aus.", "de"),
        ("Quand je dis bonne nuit, éteins tout", "fr"),
        ("Cuando digo buenas noches apaga todo", "es"),
        ("Quando eu disser boa noite, apaga tudo", "pt"),
        ("Kiedy powiem dobranoc, wyłącz światła", "pl"),
        ("Als ik welterusten zeg, doe de lichten uit", "nl"),
    ],
)
def test_a_routine_by_phrase_is_recognised(message, lang) -> None:
    assert pkg.asks_for_phrase_routine(message, lang)


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("Good night.", "en"),
        ("When is sunset?", "en"),
        ("If it rains, tell me", "en"),
        ("Wann ist Sonnenuntergang?", "de"),
    ],
)
def test_other_sentences_are_not_a_routine_by_phrase(message, lang) -> None:
    assert not pkg.asks_for_phrase_routine(message, lang)


async def test_a_routine_by_phrase_is_refused_honestly(track) -> None:
    """Live replay: "I have scheduled that", with no tool, and nothing ever ran."""
    orch, scheduler = _wired()
    _chatting(orch, ChatResponse(content="I have scheduled that.", tool_calls=[]))
    reply = await orch.async_process(
        "When I say good night, turn off all the lights and lock the door.", "c7", "en"
    )
    assert "nothing was set" in reply
    orch._provider.async_chat.assert_not_awaited()
    assert scheduler.jobs() == []


@pytest.mark.parametrize("lang", sorted(LOCALIZATIONS))
def test_every_language_has_the_words_and_sentences(lang) -> None:
    data = LOCALIZATIONS[lang]
    for key in (
        "alarm_ring", "alarm_stopped", "alarm_snoozed",
        "reminder_due", "reminder_title", "scheduled_say", "scheduled_failed",
        "alarm_wake", "message_title", "message_title_from", "phrase_routine_refused",
    ):
        assert key in data.templates
    for key in ("alarm_stop", "alarm_snooze", "later_time", "alarm_wake_request", "yearly_event", "phrase_routine"):
        assert L.pattern_list(key, lang)
    assert "{val}" not in L.template("alarm_snoozed", lang, val=5)


# ── the orchestrator ─────────────────────────────────────────────────────────


def _names(orch) -> list[str]:
    return [s["function"]["name"] for s in orch._tool_schemas_for("hi", True, None)]


def _wired(tool: str = "HassLightSet", reply: str = "Ok.") -> tuple:
    orch = _voice_orch_with_intent(tool, reply)
    orch._hass = None
    orch._scheduler = _scheduler()
    return orch, orch._scheduler


async def test_the_new_tools_come_last(track) -> None:
    """This model reacts to its neighbours in the list."""
    orch, _ = _wired()
    names = _names(orch)
    assert names[-8:] == [
        "set_alarm", "set_reminder", "schedule_action", "list_scheduled",
        "cancel_scheduled", "add_calendar_event", "tell", "move_music",
    ]


async def test_a_command_running_later_cannot_put_itself_off_again(track) -> None:
    orch, _ = _wired()
    orch._mcp.get_tool_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "start_timer"}},
        {"type": "function", "function": {"name": "HassLightSet"}},
    ])
    before = _names(orch)
    token = core._IN_SCHEDULED_RUN.set(True)
    try:
        # The list stays the same, or the next voice turn reads the prompt in again.
        assert _names(orch) == before
        for name, args in (
            ("set_alarm", {"at": "07:00"}),
            ("schedule_action", {"command": "x", "in_minutes": 5}),
            ("start_timer", {"minutes": 5}),
        ):
            assert (await orch._dispatch_tool(name, args, None)).startswith("[This is due now")
    finally:
        core._IN_SCHEDULED_RUN.reset(token)
    assert not orch._scheduler.jobs()


async def test_a_scheduled_command_runs_as_its_own_conversation(track) -> None:
    orch, _ = _wired()
    orch.async_process = AsyncMock(return_value="")
    job = pkg.Job(id="abc", kind=KIND_COMMAND, due=NOW, text="dim", device_id=SAT, language="de-DE")
    await orch._run_scheduled_command(job)
    args, kwargs = orch.async_process.await_args
    assert args[0] == "dim" and args[1].startswith("scheduled-abc-") and args[2] == "de-DE"
    assert kwargs["device_id"] == SAT
    assert core._IN_SCHEDULED_RUN.get() is False


def _chatting(orch, *responses) -> None:
    orch._provider.async_chat = AsyncMock(side_effect=list(responses))


async def test_scheduling_counts_as_acting_and_is_spoken(track) -> None:
    """No "execute now" retry: that would switch the lights straight away."""
    orch, scheduler = _wired()
    _chatting(
        orch,
        ChatResponse(content=None, tool_calls=[ToolCall(
            id="c1", name="schedule_action",
            arguments={"command": "turn off the lights", "in_minutes": 30},
        )]),
        ChatResponse(content="Lights off in 30 minutes.", tool_calls=[]),
    )
    reply = await orch.async_process("Turn off the lights in 30 minutes.", "c1", "en")
    assert reply == "Lights off in 30 minutes."
    assert orch._provider.async_chat.await_count == 2
    [job] = scheduler.jobs()
    assert job.kind == KIND_COMMAND and job.text == "turn off the lights"


async def test_a_later_command_skips_the_shortcuts_that_act_at_once(track) -> None:
    orch, _ = _wired()
    orch._hass = MagicMock()
    _chatting(orch, ChatResponse(content="Sure.", tool_calls=[]), ChatResponse(content="Sure.", tool_calls=[]))
    sweep = AsyncMock(return_value=(True, ""))
    action = AsyncMock(return_value=(True, ""))
    media = AsyncMock(return_value=(True, ""))
    with patch.object(core, "async_try_domain_sweep_shortcut", sweep), \
         patch.object(core, "async_try_action_shortcut", action), \
         patch.object(core, "async_try_media_shortcut", media), \
         patch.object(core, "try_shortcut", return_value=None), \
         patch.object(core, "person_for_user", return_value=None):
        await orch.async_process("Turn off the lights in 30 minutes.", "c2", "en")
    sweep.assert_not_awaited()
    action.assert_not_awaited()
    media.assert_not_awaited()


async def test_the_retry_for_a_later_command_can_only_schedule(track) -> None:
    orch, _ = _wired()
    orch._mcp.get_tool_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "HassTurnOff"}},
    ])
    _chatting(
        orch,
        ChatResponse(content="Okay.", tool_calls=[]),
        ChatResponse(content=None, tool_calls=[ToolCall(
            id="c1", name="schedule_action",
            arguments={"command": "turn off the tv", "in_minutes": 30},
        )]),
        ChatResponse(content="TV off in 30 minutes.", tool_calls=[]),
    )
    reply = await orch.async_process("Turn off the TV in 30 minutes.", "c3", "en")
    assert reply == "TV off in 30 minutes."
    retry = orch._provider.async_chat.await_args_list[1]
    offered = {t["function"]["name"] for t in retry.args[1]}
    assert "schedule_action" in offered and "HassTurnOff" not in offered
    assert any("LATER" in (m.get("content") or "") for m in retry.args[0])


async def test_a_lights_turn_that_also_set_a_reminder_keeps_its_voice(track) -> None:
    orch, _ = _wired()
    _chatting(
        orch,
        ChatResponse(content=None, tool_calls=[
            ToolCall(id="c1", name="HassLightSet", arguments={}),
            ToolCall(id="c2", name="set_reminder", arguments={"text": "tea", "in_minutes": 5}),
        ]),
        ChatResponse(content="Dimmed, and I will remind you.", tool_calls=[]),
    )
    with patch.object(core, "_any_actuator_success", return_value=True):
        reply = await orch.async_process("Dim the lights and remind me about tea in 5 minutes", "c4", "en")
    assert reply == "Dimmed, and I will remind you."


async def test_stop_reaches_the_alarm_before_the_media_shortcut(track, announce) -> None:
    orch, scheduler = _wired()
    await _ringing(scheduler)
    media = AsyncMock(return_value=(True, ""))
    with patch.object(core, "async_try_media_shortcut", media):
        reply = await orch.async_process("Stop.", "c5", "en", device_id=SAT)
    assert reply == "Alarm off."
    media.assert_not_awaited()


@pytest.mark.parametrize(
    ("message", "tool"),
    [("Are any timers running?", "timer_status"), ("What alarms are on?", "list_scheduled")],
)
async def test_a_status_question_answered_by_its_tool_is_not_retried(track, message, tool) -> None:
    orch, _ = _wired()
    orch._mcp.get_tool_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "timer_status"}},
        {"type": "function", "function": {"name": "list_entities"}},
    ])
    orch._mcp.call_tool = AsyncMock(return_value="None.")
    _chatting(
        orch,
        ChatResponse(content=None, tool_calls=[ToolCall(id="c1", name=tool, arguments={})]),
        ChatResponse(content="None.", tool_calls=[]),
    )
    await orch.async_process(message, "c6", "en")
    assert orch._provider.async_chat.await_count == 2


async def test_scheduler_tools_say_no_one_second(track) -> None:
    orch, _ = _wired()
    for name in ("set_alarm", "list_scheduled", "add_calendar_event"):
        assert orch._is_lookup_tool(name) is False


async def test_a_retry_cannot_set_it_twice() -> None:
    from custom_components.ai_plugin.orchestrator.toolcalls import _readonly_schemas

    kept = _readonly_schemas(pkg.TOOL_SCHEMAS)
    assert [s["function"]["name"] for s in kept] == ["list_scheduled"]
