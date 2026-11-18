"""With the model server down, Home Assistant's own Assist agent answers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from .test_tool_switches import _off, _scheduled_orch, _tool_call

from custom_components.ai_plugin import const as c
from custom_components.ai_plugin.exceptions import BackendUnreachable, OrchestratorError


# ── the orchestrator says whether anything happened ──────────────────────────


async def test_a_first_request_that_fails_has_done_nothing() -> None:
    orch = _scheduled_orch(_off(), [])
    orch._provider.async_chat = AsyncMock(side_effect=BackendUnreachable("Cannot connect"))
    with pytest.raises(BackendUnreachable) as caught:
        await orch.async_process("What's the weather like?", "c-down", "en")
    assert caught.value.nothing_done is True


async def test_a_failure_after_a_tool_ran_is_not_handed_over() -> None:
    """The tool may have switched something; Assist would do it twice."""
    orch = _scheduled_orch(_off(), ["some_tool"])
    orch._mcp.call_tool = AsyncMock(return_value="ok")
    orch._provider.async_chat = AsyncMock(side_effect=[
        _tool_call("some_tool"), BackendUnreachable("timed out"),
    ])
    with pytest.raises(BackendUnreachable) as caught:
        await orch.async_process("Do the thing.", "c-half", "en")
    assert caught.value.nothing_done is False


# ── the conversation entity hands the turn over ─────────────────────────────


def _entity(hass, error, options=None):
    from custom_components.ai_plugin.conversation import AIPluginConversationEntity

    entity = AIPluginConversationEntity.__new__(AIPluginConversationEntity)
    entity.hass = hass
    entity._entry = MagicMock()
    entity._entry.entry_id = "e1"
    entity._entry.options = options if options is not None else {
        c.CONF_BASE_URL: "http://localhost:11434/v1", c.CONF_MODEL: "m",
    }
    entity._attr_unique_id = "test"
    entity._orchestrator = MagicMock()
    entity._orchestrator.async_process = AsyncMock(side_effect=error)
    return entity


def _unreachable(nothing_done=True):
    exc = BackendUnreachable("Cannot connect to http://localhost:11434")
    exc.nothing_done = nothing_done
    return exc


def _input(text="Turn on the kitchen light."):
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    return ConversationInput(
        text=text, context=Context(), conversation_id="conv-1",
        device_id="sat-1", language="en", agent_id=None,
    )


def _assist_says(text):
    response = MagicMock()
    response.speech = {"plain": {"speech": text}}
    return AsyncMock(return_value=SimpleNamespace(response=response, continue_conversation=False))


def _speech(result):
    return result.response.speech["plain"]["speech"]


async def test_an_unreachable_server_hands_the_turn_to_assist(hass: HomeAssistant) -> None:
    entity = _entity(hass, _unreachable())
    converse = _assist_says("Turned on the light.")
    with patch("custom_components.ai_plugin.conversation.conversation.async_converse", converse, create=True):
        result = await entity.async_process(_input())
    assert _speech(result) == "Turned on the light."
    kwargs = converse.await_args.kwargs
    assert kwargs["agent_id"] == "conversation.home_assistant"
    assert kwargs["text"] == "Turn on the kitchen light."
    assert kwargs["device_id"] == "sat-1"
    assert result.continue_conversation is False


async def test_for_a_minute_turns_go_straight_to_assist(hass: HomeAssistant) -> None:
    entity = _entity(hass, _unreachable())
    converse = _assist_says("Done.")
    with patch("custom_components.ai_plugin.conversation.conversation.async_converse", converse, create=True):
        await entity.async_process(_input())
        await entity.async_process(_input("Set a timer for 5 minutes."))
        assert entity._orchestrator.async_process.await_count == 1
        assert converse.await_count == 2
        entity._backend_down_until = 0.0
        await entity.async_process(_input("And now?"))
    assert entity._orchestrator.async_process.await_count == 2


@pytest.mark.parametrize(
    ("error", "options"),
    [
        (_unreachable(nothing_done=False), None),
        (OrchestratorError("Invalid API key (HTTP 401)"), None),
        (_unreachable(), {c.CONF_OFFLINE_FALLBACK: False}),
    ],
)
async def test_these_keep_the_plugins_own_error_reply(hass, error, options) -> None:
    entity = _entity(hass, error, options)
    converse = _assist_says("Done.")
    with patch("custom_components.ai_plugin.conversation.conversation.async_converse", converse, create=True):
        result = await entity.async_process(_input())
    converse.assert_not_awaited()
    assert "Sorry" in _speech(result)


async def test_a_failing_assist_keeps_the_plugins_own_error_reply(hass) -> None:
    entity = _entity(hass, _unreachable())
    converse = AsyncMock(side_effect=RuntimeError("no agent"))
    with patch("custom_components.ai_plugin.conversation.conversation.async_converse", converse, create=True):
        result = await entity.async_process(_input())
    assert "Sorry" in _speech(result)
