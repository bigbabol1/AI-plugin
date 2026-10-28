"""The "one second" line on slow look-ups (voice turns)."""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin import satellite_speech
from custom_components.ai_plugin.orchestrator import core

from .test_conversation import _baseline_orch


def _call(name):
    return SimpleNamespace(
        name=name, arguments={},
        to_assistant_message=lambda: {"role": "assistant", "content": None},
        to_tool_result_message=lambda r: {"role": "tool", "content": r},
    )


def _orch(tool_names, tool_seconds: float, ha_local_names=()):
    orch = _baseline_orch()
    orch._hass = MagicMock()
    orch._hass.async_create_task = lambda coro: asyncio.ensure_future(coro)
    if ha_local_names:
        orch._ha_local = SimpleNamespace(tool_names=set(ha_local_names))
    calls = [_call(n) for n in tool_names]
    replies = [
        SimpleNamespace(has_tool_calls=True, tool_calls=calls, reply_text=lambda: ""),
        SimpleNamespace(has_tool_calls=False, tool_calls=[], reply_text=lambda: "It is 12 degrees."),
    ]
    orch._provider = MagicMock()
    orch._provider.async_chat = AsyncMock(side_effect=replies)
    events: list[str] = []

    async def _dispatch(name, *a, **kw):
        await asyncio.sleep(tool_seconds)
        events.append(f"tool {name}")
        return "result"

    orch._dispatch_tool = _dispatch
    return orch, events


async def _loop(orch, events, voice_mode=True):
    async def _say(hass, device_id, key):
        events.append(f"say {device_id} {key}")
        await asyncio.sleep(0.05)
        events.append("said")
        return True

    with patch.object(core, "TOOL_WAIT_LINE_DELAY_S", 0.02), patch.object(
        satellite_speech, "async_say", _say
    ):
        reply, _ = await orch._tool_loop(
            [{"role": "user", "content": "q"}], [], voice_mode=voice_mode,
            device_id="sat1", language="en",
        )
    events.append("reply")
    return reply


async def test_slow_lookup_says_the_line_before_the_answer() -> None:
    orch, events = _orch(["wikipedia_search"], tool_seconds=0.1)
    assert await _loop(orch, events) == "It is 12 degrees."
    assert events == ["say sat1 tool_wait", "said", "tool wikipedia_search", "reply"]


async def test_fast_tool_says_nothing() -> None:
    orch, events = _orch(["get_current_time"], tool_seconds=0.0)
    await _loop(orch, events)
    assert events == ["tool get_current_time", "reply"]


async def test_ha_and_text_turns_say_nothing() -> None:
    orch, events = _orch(["get_entity"], tool_seconds=0.1, ha_local_names=["get_entity"])
    await _loop(orch, events)
    assert not any(e.startswith("say") for e in events)

    orch, events = _orch(["web_search"], tool_seconds=0.1)
    await _loop(orch, events, voice_mode=False)
    assert not any(e.startswith("say") for e in events)


async def test_home_assistant_mcp_tools_say_nothing() -> None:
    """A slow HassTurnOn is a device action, not a look-up; its reply is not
    spoken, so "one second" would be all the user heard."""
    orch, events = _orch(["HassTurnOn"], tool_seconds=0.1)
    orch._mcp = SimpleNamespace(is_home_assistant_tool=lambda name: name == "HassTurnOn")
    await _loop(orch, events)
    assert not any(e.startswith("say") for e in events)

    orch, events = _orch(["search"], tool_seconds=0.1)
    orch._mcp = SimpleNamespace(is_home_assistant_tool=lambda name: name == "HassTurnOn")
    await _loop(orch, events)
    assert any(e.startswith("say") for e in events)


async def test_line_is_said_once_per_turn() -> None:
    orch, events = _orch(["web_search", "browse_url"], tool_seconds=0.1)
    await _loop(orch, events)
    assert sum(e.startswith("say") for e in events) == 1


async def test_reply_waits_for_the_line_to_finish() -> None:
    orch, events = _orch(["web_search"], tool_seconds=0.03)
    await _loop(orch, events)
    assert events.index("said") < events.index("reply")


# ── where the line is played ─────────────────────────────────────────────────


@pytest.fixture
def speech(monkeypatch):
    hass = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states.get = MagicMock(return_value=None)
    pipeline = SimpleNamespace(language="de", tts_engine="tts.cloud", tts_voice="KatjaNeural")
    monkeypatch.setattr(satellite_speech, "satellite_entity", lambda h, d: "assist_satellite.bad")
    monkeypatch.setattr(satellite_speech, "satellite_pipeline", lambda h, d: pipeline)
    monkeypatch.setattr(satellite_speech, "speech_url", lambda h, t, p: "/api/tts_proxy/x.flac")
    monkeypatch.setattr(satellite_speech, "_wait_until_played", AsyncMock())
    network = types.ModuleType("homeassistant.helpers.network")
    network.get_url = lambda h: "http://ha:8123"
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.network", network)
    return hass


async def test_routed_satellite_announces_through_mic_to_mediaplayer(speech, monkeypatch) -> None:
    routed = SimpleNamespace(media_player_entity_id="media_player.badezimmer_2")
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: routed)
    assert await satellite_speech.async_say(speech, "dev", "tool_wait")
    speech.services.async_call.assert_awaited_once_with(
        "mic_to_mediaplayer", "announce",
        {"satellite_entity_id": "assist_satellite.bad", "audio_url": "/api/tts_proxy/x.flac"},
        blocking=True,
    )
    satellite_speech._wait_until_played.assert_awaited_once_with(speech, "media_player.badezimmer_2")


async def test_call_that_blocked_through_playback_is_not_watched(speech, monkeypatch) -> None:
    """A blocking play call that returned only once the sentence was played
    needs no state watching afterwards."""
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: SimpleNamespace())
    monkeypatch.setattr(satellite_speech, "_BLOCKED_THROUGH_S", 0.02)

    async def _plays(*a, **kw):
        await asyncio.sleep(0.05)

    speech.services.async_call = _plays
    assert await satellite_speech.async_say(speech, "dev", "tool_wait")
    satellite_speech._wait_until_played.assert_not_awaited()


def _speaker_states(speech, states: list[str]) -> None:
    """The routed speaker reports ``states`` on successive polls (last one sticks)."""
    seq = iter(states)
    last = {"s": states[0]}

    def _get(entity_id):
        last["s"] = next(seq, last["s"])
        return SimpleNamespace(state=last["s"])

    speech.states.get = _get


async def test_reply_is_released_once_the_speaker_went_quiet(speech, monkeypatch) -> None:
    """Measured 2026-10-26: the speaker played the line 48.97-51.74 but the
    blocking call came back at 53.66; the reply waited for nothing."""
    routed = SimpleNamespace(media_player_entity_id="media_player.hobbyraum")
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: routed)
    monkeypatch.setattr(satellite_speech, "_POLL_S", 0.01)
    _speaker_states(speech, ["idle", "idle", "playing", "playing", "idle"])
    returned = asyncio.Event()

    async def _slow_call(*a, **kw):
        await asyncio.sleep(1.0)
        returned.set()

    speech.services.async_call = _slow_call
    assert await asyncio.wait_for(satellite_speech.async_say(speech, "dev", "tool_wait"), 0.5)
    assert not returned.is_set()


async def test_speaker_already_playing_music_waits_for_the_call(speech, monkeypatch) -> None:
    """Music pausing around the line would look like its end."""
    routed = SimpleNamespace(media_player_entity_id="media_player.hobbyraum")
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: routed)
    monkeypatch.setattr(satellite_speech, "_POLL_S", 0.01)
    _speaker_states(speech, ["playing", "paused", "playing", "playing"])
    returned = asyncio.Event()

    async def _call(*a, **kw):
        await asyncio.sleep(0.1)
        returned.set()

    speech.services.async_call = _call
    assert await satellite_speech.async_say(speech, "dev", "tool_wait")
    assert returned.is_set()


async def test_speaker_that_never_finishes_does_not_hold_the_reply(speech, monkeypatch) -> None:
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: SimpleNamespace())
    monkeypatch.setattr(satellite_speech, "_START_WAIT_S", 0.02)
    monkeypatch.setattr(satellite_speech, "_PLAY_WAIT_S", 0.02)
    stuck = asyncio.Event()

    async def _hangs(*a, **kw):
        await stuck.wait()

    speech.services.async_call = _hangs
    assert await asyncio.wait_for(satellite_speech.async_say(speech, "dev", "tool_wait"), 1)
    stuck.set()


async def test_other_satellite_plays_on_its_own_media_player(speech, monkeypatch) -> None:
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: None)
    monkeypatch.setattr(satellite_speech, "_device_media_player", lambda h, d: "media_player.pe")
    assert await satellite_speech.async_say(speech, "dev", "tool_wait")
    domain, service, data = speech.services.async_call.await_args.args
    assert (domain, service) == ("media_player", "play_media")
    assert data["media_content_id"] == "http://ha:8123/api/tts_proxy/x.flac"
    assert data["announce"] is True


async def test_nowhere_to_play_is_quietly_skipped(speech, monkeypatch) -> None:
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: None)
    monkeypatch.setattr(satellite_speech, "_device_media_player", lambda h, d: None)
    assert await satellite_speech.async_say(speech, "dev", "tool_wait") is False
    speech.services.async_call.assert_not_awaited()


async def test_line_is_spoken_in_the_pipeline_language(speech, monkeypatch) -> None:
    said = []
    monkeypatch.setattr(satellite_speech, "speech_url", lambda h, t, p: said.append(t) or "/u")
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: SimpleNamespace())
    await satellite_speech.async_say(speech, "dev", "tool_wait")
    assert said == ["Einen Moment."]
