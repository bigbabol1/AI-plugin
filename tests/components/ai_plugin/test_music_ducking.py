"""Quieter music while a satellite listens, with or without Mic to MediaPlayer."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .test_conversation import _baseline_orch  # noqa: F401  # isort: skip (stubs HA first)

from custom_components.ai_plugin import music_ducking as ducking

SAT = "assist_satellite.hobby"
SPEAKER = "media_player.hobby"


def _state(state="playing", volume=0.4, **attrs):
    return SimpleNamespace(state=state, attributes={"volume_level": volume, **attrs})


def _hass(**states):
    hass = MagicMock()
    hass.data = {}
    hass.states.get = lambda eid: states.get(eid.replace("media_player.", ""))
    hass.services.async_call = AsyncMock()
    hass.services.has_service = MagicMock(return_value=False)
    hass.tasks = []
    hass.async_create_task = lambda coro: hass.tasks.append(coro)
    return hass


async def _volumes(hass) -> list[tuple[str, float]]:
    """Run what was scheduled and return the volume calls, in order."""
    for coro in hass.tasks:
        await coro
    hass.tasks.clear()
    calls = [
        (c.kwargs["target"]["entity_id"], c.args[2]["volume_level"])
        for c in hass.services.async_call.await_args_list
    ]
    hass.services.async_call.reset_mock()
    return calls


async def test_it_lowers_while_listening_and_restores_after() -> None:
    hass = _hass(hobby=_state())
    ducking.duck(hass, SAT, [SPEAKER])
    assert await _volumes(hass) == [(SPEAKER, 0.08)]
    ducking.release(hass, SAT)
    assert await _volumes(hass) == [(SPEAKER, 0.4)]


async def test_two_satellites_waking_together_restore_the_real_volume() -> None:
    """The second must not save the lowered volume as the one to restore."""
    hass = _hass(hobby=_state())
    ducking.duck(hass, "sat.a", [SPEAKER])
    await _volumes(hass)
    hass.states.get = lambda eid: _state(volume=0.12)
    ducking.duck(hass, "sat.b", [SPEAKER])
    assert await _volumes(hass) == []
    ducking.release(hass, "sat.a")
    assert await _volumes(hass) == []
    ducking.release(hass, "sat.b")
    assert await _volumes(hass) == [(SPEAKER, 0.4)]


async def test_a_quick_next_turn_does_not_save_a_lagging_low_volume() -> None:
    hass = _hass(hobby=_state())
    ducking.duck(hass, SAT, [SPEAKER])
    ducking.release(hass, SAT)
    await _volumes(hass)
    hass.states.get = lambda eid: _state(volume=0.12)
    ducking.duck(hass, SAT, [SPEAKER])
    assert await _volumes(hass) == [(SPEAKER, 0.08)]
    assert hass.data[ducking.DATA_KEY][SPEAKER]["level"] == 0.4


@pytest.mark.parametrize("volume", [None, 0, 0.0])
async def test_a_speaker_with_no_volume_is_left_alone(volume) -> None:
    hass = _hass(hobby=_state(volume=volume))
    ducking.duck(hass, SAT, [SPEAKER])
    ducking.release(hass, SAT)
    assert await _volumes(hass) == []


async def test_releasing_without_ducking_does_nothing() -> None:
    hass = _hass()
    ducking.release(hass, SAT)
    assert await _volumes(hass) == []


async def test_a_failing_speaker_does_not_break_the_voice_turn() -> None:
    hass = _hass(hobby=_state())
    hass.services.async_call.side_effect = RuntimeError("offline")
    ducking.duck(hass, SAT, [SPEAKER])
    await _volumes(hass)


def _entry(entity_id, device, area=None):
    return SimpleNamespace(entity_id=entity_id, device_id=device, area_id=area)


def _registries(*entries, satellite_area="hobby"):
    ents = {e.entity_id: e for e in entries}
    ents[SAT] = _entry(SAT, "sat-device", satellite_area)
    reg = SimpleNamespace(entities=ents, async_get=ents.get)
    devs = SimpleNamespace(async_get=lambda _: SimpleNamespace(area_id=None))
    return (
        patch.object(ducking.er, "async_get", return_value=reg),
        patch.object(ducking.dr, "async_get", return_value=devs),
    )


def test_what_is_playing_in_the_room_is_lowered() -> None:
    hass = _hass(
        cast=_state(),
        quiet=_state(state="idle"),
        kitchen=_state(),
        group=_state(mass_player_type="group"),
        tv=_state(device_class="tv"),
    )
    a, b = _registries(
        _entry("media_player.cast", "d1", "hobby"),
        _entry("media_player.quiet", "d2", "hobby"),
        _entry("media_player.kitchen", "d3", "kitchen"),
        _entry("media_player.group", "d4", "hobby"),
        _entry("media_player.tv", "d5", "hobby"),
    )
    with a, b:
        assert ducking.room_players(hass, SAT) == ["media_player.cast"]


def test_without_mic_to_mediaplayer_the_satellites_own_speaker_counts() -> None:
    """Speech in and out on one device, also with no room set."""
    hass = _hass(own=_state())
    a, b = _registries(_entry("media_player.own", "sat-device"), satellite_area=None)
    with a, b:
        assert ducking.room_players(hass, SAT) == ["media_player.own"]


def test_a_satellites_own_wake_sound_is_not_music() -> None:
    """It reports "playing" for under a second, just before listening."""
    now = ducking.dt_util.utcnow()
    hass = _hass(
        own=_state(last_changed=None),
    )
    hass.states.get = lambda _: SimpleNamespace(
        state="playing", attributes={"volume_level": 0.4}, last_changed=now
    )
    a, b = _registries(_entry("media_player.own", "sat-device"))
    with a, b:
        assert ducking.room_players(hass, SAT) == []

    hass.states.get = lambda _: SimpleNamespace(
        state="playing", attributes={"volume_level": 0.4}, last_changed=now - timedelta(seconds=60)
    )
    a, b = _registries(_entry("media_player.own", "sat-device"))
    with a, b:
        assert ducking.room_players(hass, SAT) == ["media_player.own"]


def test_music_just_started_on_another_speaker_still_counts() -> None:
    now = ducking.dt_util.utcnow()
    hass = _hass()
    hass.states.get = lambda _: SimpleNamespace(
        state="playing", attributes={"volume_level": 0.4}, last_changed=now
    )
    a, b = _registries(_entry("media_player.cast", "d1", "hobby"))
    with a, b:
        assert ducking.room_players(hass, SAT) == ["media_player.cast"]


def test_the_speaker_mic_to_mediaplayer_routes_to_counts_in_any_room() -> None:
    hass = _hass(hobby=_state())
    routed = SimpleNamespace(media_player_entity_id=SPEAKER)
    a, b = _registries(satellite_area=None)
    with a, b, patch.object(ducking, "mic2mp_for", return_value=routed):
        assert ducking.room_players(hass, SAT) == [SPEAKER]


def test_nothing_playing_means_nothing_to_lower() -> None:
    hass = _hass(own=_state(state="idle"))
    a, b = _registries(_entry("media_player.own", "sat-device"))
    with a, b:
        assert ducking.room_players(hass, SAT) == []


# ── following the satellite ─────────────────────────────────────────────────


def _follow(hass):
    """Set up the listener; returns a function that delivers a state change."""
    hass.bus.async_listen = MagicMock(return_value=MagicMock())
    undo = ducking.async_setup_listening_duck(hass)
    args, kwargs = hass.bus.async_listen.call_args
    listener, is_satellite = args[1], kwargs["event_filter"]

    def change(old, new, entity_id=SAT):
        data = {
            "entity_id": entity_id,
            "old_state": SimpleNamespace(state=old) if old else None,
            "new_state": SimpleNamespace(state=new) if new else None,
        }
        if is_satellite(data):
            listener(SimpleNamespace(data=data))

    return change, undo


@pytest.fixture
def playing():
    with patch.object(ducking, "room_players", return_value=[SPEAKER]) as mock, \
         patch.object(ducking, "async_call_later", return_value=MagicMock()) as later:
        mock.later = later
        yield mock


async def test_it_follows_the_satellite_state(playing) -> None:
    hass = _hass(hobby=_state())
    change, _ = _follow(hass)
    change("idle", "listening")
    assert await _volumes(hass) == [(SPEAKER, 0.08)]
    change("listening", "processing")
    assert await _volumes(hass) == [(SPEAKER, 0.4)]
    change("processing", "responding")
    change("responding", "idle")
    assert await _volumes(hass) == []


async def test_a_run_that_ends_without_words_restores_too(playing) -> None:
    hass = _hass(hobby=_state())
    change, _ = _follow(hass)
    change("idle", "listening")
    await _volumes(hass)
    change("listening", "idle")
    assert await _volumes(hass) == [(SPEAKER, 0.4)]


async def test_only_satellites_count(playing) -> None:
    hass = _hass(hobby=_state())
    change, _ = _follow(hass)
    change("idle", "listening", entity_id="sensor.mode")
    assert await _volumes(hass) == []


async def test_a_satellite_stuck_listening_does_not_leave_it_quiet(playing) -> None:
    hass = _hass(hobby=_state())
    change, _ = _follow(hass)
    change("idle", "listening")
    await _volumes(hass)
    seconds, limit = playing.later.call_args.args[1:]
    assert seconds == ducking.MAX_DUCK_S
    limit(None)
    assert await _volumes(hass) == [(SPEAKER, 0.4)]


async def test_unloading_restores_what_is_lowered(playing) -> None:
    hass = _hass(hobby=_state())
    change, undo = _follow(hass)
    change("idle", "listening")
    await _volumes(hass)
    undo()
    assert await _volumes(hass) == [(SPEAKER, 0.4)]
    playing.later.return_value.assert_called()


async def test_a_failure_never_reaches_the_voice_turn() -> None:
    hass = _hass()
    change, _ = _follow(hass)
    with patch.object(ducking, "room_players", side_effect=RuntimeError("boom")):
        change("idle", "listening")
