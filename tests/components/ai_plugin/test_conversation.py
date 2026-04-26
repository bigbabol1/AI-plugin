"""Tests for the AI Hub conversation entity and orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.core import HomeAssistant

from custom_components.ai_plugin.const import (
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_VOICE_MODE,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
)
from custom_components.ai_plugin.exceptions import OrchestratorError
from custom_components.ai_plugin.orchestrator import Orchestrator


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


def _baseline_orch(entry=None) -> Orchestrator:
    """Build a bare Orchestrator with all attributes async_process touches."""
    from custom_components.ai_plugin.context_manager import ContextManager

    orch = Orchestrator.__new__(Orchestrator)
    orch._entry = entry or _make_mock_entry()
    orch._context_mgr = ContextManager(max_tokens=8192)
    orch._summarization_enabled = False
    orch._mcp = None
    orch._memory = None
    orch._max_tool_iterations = 5
    orch._web_search = None
    orch._ha_local = None
    orch._last_entities = {}
    orch._conv_locks = {}
    loc = MagicMock()
    loc.async_resolve = AsyncMock(return_value={})
    orch._location = loc
    return orch


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator: system prompt selection
# ══════════════════════════════════════════════════════════════════════════════


async def test_system_prompt_default() -> None:
    """Default system prompt is returned when no custom prompt and voice=False."""
    orch = _baseline_orch()
    prompt = await orch._build_system_prompt(voice_mode=False)
    assert prompt == SYSTEM_PROMPT_DEFAULT


async def test_system_prompt_voice() -> None:
    """Voice system prompt is returned when voice_mode=True and no custom prompt."""
    orch = _baseline_orch()
    prompt = await orch._build_system_prompt(voice_mode=True)
    assert prompt == SYSTEM_PROMPT_VOICE


async def test_system_prompt_custom_appended_to_voice() -> None:
    """Custom instructions are appended after the voice base prompt."""
    entry = _make_mock_entry(
        options={
            CONF_BASE_URL: "http://localhost:11434/v1",
            CONF_MODEL: "llama3.2:3b",
            "system_prompt": "You are a pirate.",
            CONF_VOICE_MODE: True,
        }
    )
    orch = _baseline_orch(entry)
    prompt = await orch._build_system_prompt(voice_mode=True)
    assert prompt == f"{SYSTEM_PROMPT_VOICE}\n\nYou are a pirate."


async def test_system_prompt_custom_appended_to_default() -> None:
    """Custom instructions are appended after the default base prompt."""
    entry = _make_mock_entry(
        options={
            CONF_BASE_URL: "http://localhost:11434/v1",
            CONF_MODEL: "llama3.2:3b",
            "system_prompt": "You are a pirate.",
        }
    )
    orch = _baseline_orch(entry)
    prompt = await orch._build_system_prompt(voice_mode=False)
    assert prompt == f"{SYSTEM_PROMPT_DEFAULT}\n\nYou are a pirate."


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator: process
# ══════════════════════════════════════════════════════════════════════════════


async def test_process_happy_path() -> None:
    """async_process returns the provider reply and stores history."""
    orch = _baseline_orch()

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
    orch = _baseline_orch()

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
    orch = _baseline_orch()

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
    orch = _baseline_orch()

    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(
        side_effect=OrchestratorError("Provider down")
    )
    orch._provider = mock_provider

    with pytest.raises(OrchestratorError, match="Provider down"):
        await orch.async_process("Hello", "conv-1", "en")


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator: tool calling loop
# ══════════════════════════════════════════════════════════════════════════════


def _make_orch_with_mcp(mock_mcp, mock_provider):
    """Helper: build a minimal Orchestrator with MCP and provider pre-wired."""
    orch = _baseline_orch()
    orch._mcp = mock_mcp
    orch._provider = mock_provider
    return orch


async def test_tool_loop_single_tool_call() -> None:
    """Tool loop executes one tool call and returns the final text reply."""
    from custom_components.ai_plugin.providers import ChatResponse, ToolCall

    tool_call = ToolCall(id="c1", name="get_time", arguments={})
    responses = [
        ChatResponse(content=None, tool_calls=[tool_call]),   # first: tool call
        ChatResponse(content="It is 22:00.", tool_calls=[]),  # second: final answer
    ]
    call_count = 0

    async def mock_chat(messages, tools=None):
        nonlocal call_count
        r = responses[call_count]
        call_count += 1
        return r

    mock_provider = MagicMock()
    mock_provider.async_chat = mock_chat

    mock_mcp = MagicMock()
    mock_mcp.get_tool_schemas = MagicMock(return_value=[{"type": "function", "function": {"name": "get_time"}}])
    mock_mcp.call_tool = AsyncMock(return_value="22:00")

    orch = _make_orch_with_mcp(mock_mcp, mock_provider)
    reply = await orch.async_process("What time is it?", "c", "en")

    assert reply == "It is 22:00."
    mock_mcp.call_tool.assert_awaited_once_with("get_time", {})


async def test_process_with_summarization_enabled() -> None:
    """When summarization is enabled, summarize_if_needed is called."""
    orch = _baseline_orch()
    orch._summarization_enabled = True

    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="Summarized reply.")
    orch._provider = mock_provider

    # Patch summarize_if_needed to verify it's called
    with patch.object(
        orch._context_mgr, "summarize_if_needed", new_callable=AsyncMock
    ) as mock_summarize:
        reply = await orch.async_process("Hello", "conv-sum", "en")

    assert reply == "Summarized reply."
    mock_summarize.assert_awaited_once()


async def test_dispatch_tool_no_handler_returns_unavailable_message() -> None:
    """_dispatch_tool returns unavailable message when no MCP and no web_search."""
    orch = _baseline_orch()
    result = await orch._dispatch_tool("unknown_tool", {"arg": "value"})

    assert "unavailable" in result
    assert "unknown_tool" in result


async def test_async_close_delegates_to_provider() -> None:
    """async_close calls provider.async_close."""
    orch = _baseline_orch()

    mock_provider = MagicMock()
    mock_provider.async_close = AsyncMock()
    orch._provider = mock_provider

    await orch.async_close()

    mock_provider.async_close.assert_awaited_once()


async def test_tool_loop_max_iterations_appends_note() -> None:
    """When max_tool_iterations is reached, a note is appended to the reply."""
    from custom_components.ai_plugin.providers import ChatResponse, ToolCall

    async def always_tool_call(messages, tools=None):
        return ChatResponse(
            content="partial",
            tool_calls=[ToolCall(id="x", name="loop_tool", arguments={})],
        )

    mock_provider = MagicMock()
    mock_provider.async_chat = always_tool_call

    mock_mcp = MagicMock()
    mock_mcp.get_tool_schemas = MagicMock(return_value=[{"type": "function", "function": {"name": "loop_tool"}}])
    mock_mcp.call_tool = AsyncMock(return_value="result")

    orch = _make_orch_with_mcp(mock_mcp, mock_provider)
    orch._max_tool_iterations = 2
    reply = await orch.async_process("Loop forever", "c", "en")

    assert "tool call limit" in reply.lower()


async def test_no_tools_skips_tool_loop() -> None:
    """When MCP has no tools, async_complete is used directly."""
    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(return_value="Direct reply.")
    mock_provider.async_chat = AsyncMock()

    mock_mcp = MagicMock()
    mock_mcp.get_tool_schemas = MagicMock(return_value=[])  # empty = no tools

    orch = _make_orch_with_mcp(mock_mcp, mock_provider)
    reply = await orch.async_process("Hello", "c", "en")

    assert reply == "Direct reply."
    mock_provider.async_chat.assert_not_awaited()


# ══════════════════════════════════════════════════════════════════════════════
# Conversation entity: error handling
# ══════════════════════════════════════════════════════════════════════════════


async def test_conversation_entity_returns_graceful_reply_on_error(
    hass: HomeAssistant,
) -> None:
    """OrchestratorError is caught and a graceful string reply is returned."""
    from custom_components.ai_plugin.conversation import AIPluginConversationEntity

    entry = _make_mock_entry()
    entity = AIPluginConversationEntity.__new__(AIPluginConversationEntity)
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


async def test_conversation_entity_passes_context_user_id_through(
    hass: HomeAssistant,
) -> None:
    """Context user_id is forwarded to the orchestrator verbatim."""
    from custom_components.ai_plugin.conversation import AIPluginConversationEntity

    entry = _make_mock_entry(
        options={
            CONF_BASE_URL: "http://localhost:11434/v1",
            CONF_MODEL: "llama3.2:3b",
        }
    )
    entity = AIPluginConversationEntity.__new__(AIPluginConversationEntity)
    entity.hass = hass
    entity._entry = entry
    entity._attr_unique_id = "test"

    mock_orch = MagicMock()
    mock_orch.async_process = AsyncMock(return_value="ok")
    entity._orchestrator = mock_orch

    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    user_input = ConversationInput(
        text="hi",
        context=Context(user_id="real-typed-user"),
        conversation_id="conv-typed",
        device_id=None,
        language="en",
        agent_id=None,
    )
    await entity.async_process(user_input)

    kwargs = mock_orch.async_process.await_args.kwargs
    assert kwargs["user_id"] == "real-typed-user"


async def test_conversation_entity_anonymous_context_keeps_user_id_none(
    hass: HomeAssistant,
) -> None:
    """Voice Assist with no authenticated user stays anonymous (no fallback)."""
    from custom_components.ai_plugin.conversation import AIPluginConversationEntity

    entry = _make_mock_entry()
    entity = AIPluginConversationEntity.__new__(AIPluginConversationEntity)
    entity.hass = hass
    entity._entry = entry
    entity._attr_unique_id = "test"

    mock_orch = MagicMock()
    mock_orch.async_process = AsyncMock(return_value="ok")
    entity._orchestrator = mock_orch

    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    user_input = ConversationInput(
        text="hi",
        context=Context(),
        conversation_id="conv-voice",
        device_id="media_player.satellite",
        language="en",
        agent_id=None,
    )
    await entity.async_process(user_input)

    kwargs = mock_orch.async_process.await_args.kwargs
    assert kwargs["user_id"] is None


async def test_conversation_entity_will_remove_closes_orchestrator(
    hass: HomeAssistant,
) -> None:
    """async_will_remove_from_hass delegates to orchestrator.async_close."""
    from custom_components.ai_plugin.conversation import AIPluginConversationEntity

    entry = _make_mock_entry()
    entity = AIPluginConversationEntity.__new__(AIPluginConversationEntity)
    entity.hass = hass
    entity._entry = entry
    entity._attr_unique_id = "test"

    mock_orch = MagicMock()
    mock_orch.async_close = AsyncMock()
    entity._orchestrator = mock_orch

    await entity.async_will_remove_from_hass()

    mock_orch.async_close.assert_awaited_once()
