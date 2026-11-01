"""Tests for announce-mode timers (v0.9.33)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin import satellite_speech
from custom_components.ai_plugin.const import TIMER_DONE_SENTINEL
from custom_components.ai_plugin.tools.ha_local import HALocalToolRegistry

from .test_conversation import _baseline_orch


async def test_start_timer_plants_conversation_command() -> None:
    hass = MagicMock()
    hass.config.language = "en"
    reg = HALocalToolRegistry(hass)

    captured: dict = {}

    async def fake_handle(*args, **kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.speech = {"plain": {"speech": "Timer started."}}
        return resp

    with patch(
        "custom_components.ai_plugin.tools.ha_local.timers.intent.async_handle",
        side_effect=fake_handle,
    ):
        out = await reg.call_tool(
            "start_timer",
            {"minutes": 5, "name": "pasta"},
            device_id="dev1",
            language="en",
            user_message="set a pasta timer for 5 minutes",
            announce_agent_id="conversation.ai_plugin",
        )

    assert "Timer started" in out
    slots = captured["slots"]
    assert slots["conversation_command"]["value"] == f"{TIMER_DONE_SENTINEL} pasta"
    assert captured["conversation_agent_id"] == "conversation.ai_plugin"


async def test_start_timer_without_announce_keeps_device_ring() -> None:
    hass = MagicMock()
    hass.config.language = "en"
    reg = HALocalToolRegistry(hass)

    captured: dict = {}

    async def fake_handle(*args, **kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.speech = {"plain": {"speech": "ok"}}
        return resp

    with patch(
        "custom_components.ai_plugin.tools.ha_local.timers.intent.async_handle",
        side_effect=fake_handle,
    ):
        await reg.call_tool(
            "start_timer", {"minutes": 5},
            device_id="dev1", language="en",
            user_message="set a timer for 5 minutes",
        )

    assert "conversation_command" not in captured["slots"]
    assert "conversation_agent_id" not in captured


@pytest.fixture
def speech(monkeypatch):
    """A satellite whose main pipeline is English (Luna) and whose second
    wake word runs German (Maja)."""
    hass = MagicMock()
    hass.services.async_call = AsyncMock()
    en = SimpleNamespace(language="en", tts_engine="tts.cloud", tts_voice="LunaNeural")
    de = SimpleNamespace(language="de", tts_engine="tts.cloud", tts_voice="MajaNeural")
    rendered: list = []

    def _pipeline(h, d, lang=None):
        return de if lang == "de" else en

    def _url(h, text, pipeline):
        rendered.append((text, pipeline.tts_voice))
        return "/api/tts_proxy/x.flac"

    monkeypatch.setattr(satellite_speech, "satellite_entity", lambda h, d: "assist_satellite.bad")
    monkeypatch.setattr(satellite_speech, "satellite_pipeline", _pipeline)
    monkeypatch.setattr(satellite_speech, "speech_url", _url)
    network = types.ModuleType("homeassistant.helpers.network")
    network.get_url = lambda h: "http://ha:8123"
    monkeypatch.setitem(sys.modules, "homeassistant.helpers.network", network)
    hass.rendered = rendered
    return hass


async def _expire(hass, message: str, language: str) -> str:
    orch = _baseline_orch()
    orch._hass = hass
    return await orch.async_process(message, "conv-t", language, device_id="dev1")


async def test_routed_timer_speaks_in_the_pipeline_voice(speech, monkeypatch) -> None:
    """Text alone was read by HA Cloud's default voice, a German one, so
    "Timer pasta is up." came out with a German accent."""
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: SimpleNamespace())
    reply = await _expire(speech, f"{TIMER_DONE_SENTINEL} pasta", "en-US")

    assert reply == ""
    assert speech.rendered == [("Timer pasta is up.", "LunaNeural")]
    speech.services.async_call.assert_awaited_once_with(
        "mic_to_mediaplayer", "announce",
        {"satellite_entity_id": "assist_satellite.bad", "audio_url": "/api/tts_proxy/x.flac"},
        blocking=False,
    )


async def test_german_timer_uses_the_german_pipeline(speech, monkeypatch) -> None:
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: SimpleNamespace())
    await _expire(speech, f"{TIMER_DONE_SENTINEL} pasta", "de-DE")
    assert speech.rendered == [("Der Timer pasta ist abgelaufen.", "MajaNeural")]


async def test_text_follows_the_voice_when_no_pipeline_speaks_the_language(
    speech, monkeypatch
) -> None:
    """A French timer on a satellite with only English and German
    pipelines: the English voice reads English, never French."""
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: SimpleNamespace())
    await _expire(speech, TIMER_DONE_SENTINEL, "fr")
    assert speech.rendered == [("Timer is up.", "LunaNeural")]


async def test_unrouted_satellite_announces_the_rendered_audio(speech, monkeypatch) -> None:
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: None)
    await _expire(speech, TIMER_DONE_SENTINEL, "en")
    speech.services.async_call.assert_awaited_once_with(
        "assist_satellite", "announce",
        {
            "entity_id": "assist_satellite.bad",
            "message": "Timer is up.",
            "media_id": "http://ha:8123/api/tts_proxy/x.flac",
        },
        blocking=False,
    )


async def test_without_a_pipeline_voice_the_text_is_sent(speech, monkeypatch) -> None:
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: SimpleNamespace())
    monkeypatch.setattr(satellite_speech, "speech_url", lambda h, t, p: None)
    await _expire(speech, f"{TIMER_DONE_SENTINEL} pasta", "de")
    speech.services.async_call.assert_awaited_once_with(
        "mic_to_mediaplayer", "announce",
        {"satellite_entity_id": "assist_satellite.bad", "message": "Der Timer pasta ist abgelaufen."},
        blocking=False,
    )


async def test_sentinel_never_enters_history(speech, monkeypatch) -> None:
    monkeypatch.setattr(satellite_speech, "mic2mp_for", lambda h, s: SimpleNamespace())
    orch = _baseline_orch()
    orch._hass = speech
    await orch.async_process(f"{TIMER_DONE_SENTINEL} pasta", "conv-t", "en", device_id="dev1")
    assert orch._context_mgr.get_history("conv-t") == []


async def test_no_satellite_drops_the_announcement(speech, monkeypatch) -> None:
    monkeypatch.setattr(satellite_speech, "satellite_entity", lambda h, d: None)
    assert await _expire(speech, TIMER_DONE_SENTINEL, "en") == ""
    speech.services.async_call.assert_not_awaited()


# ── which pipeline a satellite speaks with ───────────────────────────────────


def _select(entity_id: str, unique_id: str) -> SimpleNamespace:
    return SimpleNamespace(entity_id=entity_id, unique_id=unique_id, device_id="dev1")


def _pipelines(monkeypatch, selects: dict[str, str]):
    """Satellite dev1 with pipeline selects {entity_id: chosen pipeline}."""
    import homeassistant.components.assist_pipeline as ap
    import homeassistant.helpers.entity_registry as er

    buddy = SimpleNamespace(name="Buddy", language="en")
    buddy_de = SimpleNamespace(name="Buddy (DE)", language="de")
    monkeypatch.setattr(ap, "async_get_pipeline", lambda h, pid=None: buddy)
    monkeypatch.setattr(ap, "async_get_pipelines", lambda h: [buddy, buddy_de])
    entries = {
        eid: _select(eid, "abc-pipeline_2" if eid.endswith("_2") else "abc-pipeline")
        for eid in selects
    }
    monkeypatch.setattr(er, "async_get", lambda h: SimpleNamespace(entities=entries))
    hass = MagicMock()
    hass.states.get.side_effect = lambda eid: SimpleNamespace(
        state=selects[eid], attributes={"options": ["preferred", "Buddy", "Buddy (DE)"]}
    )
    return hass


def test_main_pipeline_without_a_language(monkeypatch) -> None:
    hass = _pipelines(monkeypatch, {
        "select.sat_assistant_2": "Buddy (DE)", "select.sat_assistant": "preferred",
    })
    assert satellite_speech.satellite_pipeline(hass, "dev1").name == "Buddy"


def test_second_wake_word_pipeline_in_the_timer_language(monkeypatch) -> None:
    hass = _pipelines(monkeypatch, {
        "select.sat_assistant": "Buddy", "select.sat_assistant_2": "Buddy (DE)",
    })
    assert satellite_speech.satellite_pipeline(hass, "dev1", "de").name == "Buddy (DE)"
    assert satellite_speech.satellite_pipeline(hass, "dev1", "fr").name == "Buddy"
