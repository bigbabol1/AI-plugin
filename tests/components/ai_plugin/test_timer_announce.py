"""Tests for announce-mode timers (v0.9.33)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
        "custom_components.ai_plugin.tools.ha_local.intent.async_handle",
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
        "custom_components.ai_plugin.tools.ha_local.intent.async_handle",
        side_effect=fake_handle,
    ):
        await reg.call_tool(
            "start_timer", {"minutes": 5},
            device_id="dev1", language="en",
            user_message="set a timer for 5 minutes",
        )

    assert "conversation_command" not in captured["slots"]
    assert "conversation_agent_id" not in captured


async def test_sentinel_triggers_mic_to_mediaplayer_announce() -> None:
    orch = _baseline_orch()
    hass = MagicMock()
    hass.services.has_service.return_value = True
    hass.services.async_call = AsyncMock()
    orch._hass = hass

    sat = SimpleNamespace(entity_id="assist_satellite.wohnzimmer_sat")
    with patch(
        "custom_components.ai_plugin.orchestrator.er.async_get",
        return_value=MagicMock(),
    ), patch(
        "custom_components.ai_plugin.orchestrator.er.async_entries_for_device",
        return_value=[sat],
    ):
        reply = await orch.async_process(
            f"{TIMER_DONE_SENTINEL} pasta", "conv-t", "de-DE", device_id="dev1"
        )

    assert reply == ""
    args, kwargs = hass.services.async_call.await_args
    assert args[0] == "mic_to_mediaplayer" and args[1] == "announce"
    assert args[2]["satellite_entity_id"] == "assist_satellite.wohnzimmer_sat"
    assert args[2]["message"] == "Der Timer pasta ist abgelaufen."
    # sentinel turns never pollute conversation history
    assert orch._context_mgr.get_history("conv-t") == []


async def test_sentinel_falls_back_to_assist_satellite_announce() -> None:
    orch = _baseline_orch()
    hass = MagicMock()
    hass.services.has_service.return_value = False
    hass.services.async_call = AsyncMock()
    orch._hass = hass

    sat = SimpleNamespace(entity_id="assist_satellite.sat1")
    with patch(
        "custom_components.ai_plugin.orchestrator.er.async_get",
        return_value=MagicMock(),
    ), patch(
        "custom_components.ai_plugin.orchestrator.er.async_entries_for_device",
        return_value=[sat],
    ):
        reply = await orch.async_process(
            TIMER_DONE_SENTINEL, "conv-t2", "en", device_id="dev1"
        )

    assert reply == ""
    args, _ = hass.services.async_call.await_args
    assert args[0] == "assist_satellite" and args[1] == "announce"
    assert args[2]["message"] == "Timer is up."
