"""Tests for the AI Hub conversation entity and orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.core import HomeAssistant

from custom_components.ai_hub.const import (
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_VOICE_MODE,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
)
from custom_components.ai_hub.exceptions import OrchestratorError
from custom_components.ai_hub.orchestrator import Orchestrator


def _make_mock_entry(options: dict | None = None) -> MagicMock:
    """Build a minimal mock ConfigEntry."""
    entry = MagicMock()
    entry.entry_id = "test-entry-id"
    entry.title = "AI Hub (llama3.2:3b)"
    entry.options = options or {
        CONF_BASE_URL: "http://localhost:11434/v1",
        CONF_MODEL: "llama3.2:3b",
    }
    return entry


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator: system prompt selection
# ══════════════════════════════════════════════════════════════════════════════


def test_system_prompt_default() -> None:
    """Default system prompt is returned when no custom prompt and voice=False."""
    entry = _make_mock_entry()
    orch = Orchestrator.__new__(Orchestrator)
    orch._entry = entry
    prompt = orch._build_system_prompt(voice_mode=False)
    assert prompt == SYSTEM_PROMPT_DEFAULT


def test_system_prompt_voice() -> None:
    """Voice system prompt is returned when voice_mode=True and no custom prompt."""
    entry = _make_mock_entry()
    orch = Orchestrator.__new__(Orchestrator)
    orch._entry = entry
    prompt = orch._build_system_prompt(voice_mode=True)
    assert prompt == SYSTEM_PROMPT_VOICE


def test_system_prompt_custom_overrides_voice() -> None:
    """Custom instructions take priority over voice mode."""
    entry = _make_mock_entry(
        options={
            CONF_BASE_URL: "http://localhost:11434/v1",
            CONF_MODEL: "llama3.2:3b",
            "system_prompt": "You are a pirate.",
            CONF_VOICE_MODE: True,
        }
    )
    orch = Orchestrator.__new__(Orchestrator)
    orch._entry = entry
    prompt = orch._build_system_prompt(voice_mode=True)
    assert prompt == "You are a pirate."


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator: process
# ══════════════════════════════════════════════════════════════════════════════


async def test_process_happy_path() -> None:
    """async_process returns the provider reply and stores history."""
    from custom_components.ai_hub.context_manager import ContextManager

    entry = _make_mock_entry()
    orch = Orchestrator.__new__(Orchestrator)
    orch._entry = entry
    orch._context_mgr = ContextManager(max_tokens=8192)
    orch._summarization_enabled = False  # keep test simple

    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="Lights are on.")
    orch._provider = mock_provider

    reply = await orch.async_process(
        message="Turn on the lights",
        conversation_id="conv-abc",
        language="en",
    )

    assert reply == "Lights are on."
    hist = orch._context_mgr.get_history("conv-abc")
    assert len(hist) == 2
    assert hist[0] == {"role": "user", "content": "Turn on the lights"}
    assert hist[1] == {"role": "assistant", "content": "Lights are on."}


async def test_process_multi_turn_maintains_history() -> None:
    """Multi-turn conversation accumulates history correctly."""
    from custom_components.ai_hub.context_manager import ContextManager

    entry = _make_mock_entry()
    orch = Orchestrator.__new__(Orchestrator)
    orch._entry = entry
    orch._context_mgr = ContextManager(max_tokens=8192)
    orch._summarization_enabled = False

    responses = ["Hello!", "The temperature is 72°F."]
    call_count = 0

    async def mock_complete(messages):
        nonlocal call_count
        result = responses[call_count]
        call_count += 1
        return result

    mock_provider = MagicMock()
    mock_provider.async_complete = mock_complete
    orch._provider = mock_provider

    await orch.async_process("Hi", "conv-1", "en")
    await orch.async_process("What's the temperature?", "conv-1", "en")

    history = orch._context_mgr.get_history("conv-1")
    assert len(history) == 4  # 2 user + 2 assistant
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[2]["role"] == "user"
    assert history[3]["role"] == "assistant"


async def test_process_isolates_conversation_ids() -> None:
    """Different conversation_ids maintain separate histories."""
    from custom_components.ai_hub.context_manager import ContextManager

    entry = _make_mock_entry()
    orch = Orchestrator.__new__(Orchestrator)
    orch._entry = entry
    orch._context_mgr = ContextManager(max_tokens=8192)
    orch._summarization_enabled = False

    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="Reply")
    orch._provider = mock_provider

    await orch.async_process("Hello from conv A", "conv-a", "en")
    await orch.async_process("Hello from conv B", "conv-b", "en")

    assert len(orch._context_mgr.get_history("conv-a")) == 2
    assert len(orch._context_mgr.get_history("conv-b")) == 2
    assert orch._context_mgr.get_history("conv-a")[0]["content"] == "Hello from conv A"
    assert orch._context_mgr.get_history("conv-b")[0]["content"] == "Hello from conv B"


async def test_process_propagates_orchestrator_error() -> None:
    """OrchestratorError from provider propagates out of async_process."""
    from custom_components.ai_hub.context_manager import ContextManager

    entry = _make_mock_entry()
    orch = Orchestrator.__new__(Orchestrator)
    orch._entry = entry
    orch._context_mgr = ContextManager(max_tokens=8192)
    orch._summarization_enabled = False

    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(
        side_effect=OrchestratorError("Provider down")
    )
    orch._provider = mock_provider

    with pytest.raises(OrchestratorError, match="Provider down"):
        await orch.async_process("Hello", "conv-1", "en")


# ══════════════════════════════════════════════════════════════════════════════
# Conversation entity: error handling
# ══════════════════════════════════════════════════════════════════════════════


async def test_conversation_entity_returns_graceful_reply_on_error(
    hass: HomeAssistant,
) -> None:
    """OrchestratorError is caught and a graceful string reply is returned."""
    from custom_components.ai_hub.conversation import AIHubConversationEntity

    entry = _make_mock_entry()
    entity = AIHubConversationEntity.__new__(AIHubConversationEntity)
    entity.hass = hass
    entity._entry = entry
    entity._attr_unique_id = "test"

    mock_orch = MagicMock()
    mock_orch.async_process = AsyncMock(
        side_effect=OrchestratorError("Connection refused")
    )
    entity._orchestrator = mock_orch

    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    user_input = ConversationInput(
        text="turn on lights",
        context=Context(),
        conversation_id="conv-1",
        device_id=None,
        language="en",
        agent_id=None,
    )
    result = await entity.async_process(user_input)

    # Should return a ConversationResult, not raise
    assert result is not None
    assert "Sorry" in result.response.speech["plain"]["speech"]
