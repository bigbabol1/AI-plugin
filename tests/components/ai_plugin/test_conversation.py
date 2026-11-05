"""Tests for the AI Plugin conversation entity and orchestrator."""

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
    entry.title = "AI Plugin (llama3.2:3b)"
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
    orch._prune_schemas = False
    orch._bg_tasks = set()
    orch._mcp = None
    orch._memory = None
    orch._max_tool_iterations = 5
    orch._web_search = None
    orch._browse_url = None
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

    # Patch summarize_if_needed to verify it's called. Since v0.9.27 it
    # runs post-reply in a background task — drain it before asserting.
    import asyncio as _aio

    with patch.object(
        orch._context_mgr, "summarize_if_needed", new_callable=AsyncMock
    ) as mock_summarize:
        reply = await orch.async_process("Hello", "conv-sum", "en")
        if orch._bg_tasks:
            await _aio.gather(*orch._bg_tasks)

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


# ── v0.9.27: cache-stable prompt layout ───────────────────────────────────────


async def test_volatile_context_injected_before_user_turn() -> None:
    """[LAST ACTION] rides in a late system message, not the prompt head,
    so the static system prompt stays byte-stable for prefix caching."""
    orch = _baseline_orch()
    orch._last_entities["conv-vol"] = "HassTurnOn(lamp → light.lamp @ kitchen)"

    captured: dict = {}

    async def _capture(messages):
        captured["messages"] = [dict(m) for m in messages]
        return "ok"

    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(side_effect=_capture)
    orch._provider = mock_provider

    reply = await orch.async_process("turn it off please", "conv-vol", "en")

    assert reply == "ok"
    msgs = captured["messages"]
    assert [m["role"] for m in msgs] == ["system", "system", "user"]
    assert "[LAST ACTION]" in msgs[1]["content"]
    # The tracked entity must ride in the volatile message, not the static
    # head (the head may *mention* [LAST ACTION] in its instructions).
    assert "light.lamp @ kitchen" in msgs[1]["content"]
    assert "light.lamp" not in msgs[0]["content"]


async def test_follow_up_request_extends_the_previous_one() -> None:
    """The context block stays in history where it was sent, so the next
    request repeats the previous one exactly and only the new turn has to be
    read. Dropped from history, it made every follow-up re-read the whole
    conversation (~1.1 s for one web search's worth, 2026-10-28)."""
    orch = _baseline_orch()
    orch._last_entities["conv-pfx"] = "HassTurnOn(lamp → light.lamp @ kitchen)"
    sent: list[list[dict]] = []

    async def _capture(messages):
        sent.append([dict(m) for m in messages])
        return f"reply {len(sent)}"

    orch._provider = MagicMock()
    orch._provider.async_complete = AsyncMock(side_effect=_capture)

    await orch.async_process("turn it off please", "conv-pfx", "en")
    orch._last_entities["conv-pfx"] = "HassTurnOff(lamp → light.lamp @ kitchen)"
    await orch.async_process("and the other one", "conv-pfx", "en")

    first, second = sent
    assert second[: len(first)] == first
    assert [m["role"] for m in second[len(first):]] == ["assistant", "system", "user"]
    assert "HassTurnOff" in second[-2]["content"]


async def test_no_volatile_block_means_no_extra_message() -> None:
    orch = _baseline_orch()

    captured: dict = {}

    async def _capture(messages):
        captured["messages"] = [dict(m) for m in messages]
        return "ok"

    mock_provider = MagicMock()
    mock_provider.async_complete = AsyncMock(side_effect=_capture)
    orch._provider = mock_provider

    await orch.async_process("hello", "conv-novol", "en")
    assert [m["role"] for m in captured["messages"]] == ["system", "user"]


# ── v0.9.28: kaomoji stripping ────────────────────────────────────────────────


def test_strip_emoji_removes_kaomoji() -> None:
    from custom_components.ai_plugin.orchestrator.textproc import _strip_emoji

    assert _strip_emoji("Sorry! (╯°□°）╯︵ ┻━┻") == "Sorry!"
    assert _strip_emoji("Nothing to cancel (・_・)") == "Nothing to cancel"
    assert _strip_emoji("shrug ¯\\_(ツ)_/¯ done") != ""  # must not crash
    # Said by gemma-4 on a voice satellite, 2026-10-26.
    assert _strip_emoji("It could crush a small car. ᕦ(ò_ó)ᕤ") == "It could crush a small car."
    for face in ("(T_T)", "(-_-)", "(>_<)", "(^^)", "(^o^)", "(^_^)"):
        assert _strip_emoji(f"Okay {face}") == "Okay", face
    assert _strip_emoji("Der Wecker ist gestellt. ⏰") == "Der Wecker ist gestellt."
    # The face goes first; its arms must not stay behind to be read out.
    assert _strip_emoji("Looking fabulous! ٩(◕‿◕)۶") == "Looking fabulous!"
    assert _strip_emoji("Looking fabulous! ٩۶") == "Looking fabulous!"
    # Live 2026-11-05: what gemma left behind, the second one spoken by TTS.
    assert _strip_emoji("You're very welcome! ٩ⅴ") == "You're very welcome!"
    assert _strip_emoji("You're very welcome! ⸜⸜") == "You're very welcome!"
    assert _strip_emoji("You're very welcome! ⸜(◕‿◕)⸝") == "You're very welcome!"


def test_strip_emoji_preserves_degrees_and_parens() -> None:
    from custom_components.ai_plugin.orchestrator.textproc import _strip_emoji

    # In Arabic these are digits.
    assert _strip_emoji("الساعة ٩٥") == "الساعة ٩٥"
    assert _strip_emoji("समय ५ बजे है ।") == "समय ५ बजे है ।"
    # Ordinary typography and symbols stand on their own.
    assert _strip_emoji("Tea – or coffee … „gern“ « oui » ¿sí?") == "Tea – or coffee … „gern“ « oui » ¿sí?"
    assert _strip_emoji("• Milk\n• Eggs") == "• Milk\n• Eggs"
    assert _strip_emoji("It is 21 ℃ and 5 € ½ off.") == "It is 21 ℃ and 5 € ½ off."
    assert _strip_emoji("It is 21°C outside.") == "It is 21°C outside."
    assert _strip_emoji("Timer (eval) is running.") == "Timer (eval) is running."
    assert _strip_emoji("Five squared (5^2) is 25.") == "Five squared (5^2) is 25."
    assert _strip_emoji("The ratio (3:2) holds.") == "The ratio (3:2) holds."


# ── v0.9.32: follow-up honoured on voice; per-device feedback guard ──────────


def _entity_for_continue(options: dict, reply="ok"):
    entity = object.__new__(
        __import__(
            "custom_components.ai_plugin.conversation", fromlist=["x"]
        ).AIPluginConversationEntity
    )
    entity._entry = _make_mock_entry(
        options={CONF_BASE_URL: "http://x/v1", CONF_MODEL: "m", **options}
    )
    entity._attr_unique_id = "test"
    mock_orch = MagicMock()
    mock_orch.async_process = AsyncMock(return_value=reply)
    mock_orch.is_voice_device = MagicMock(return_value=True)
    entity._orchestrator = mock_orch
    return entity


def _voice_input(text="hello", device_id="satdev"):
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    return ConversationInput(
        text=text, context=Context(), conversation_id="c",
        device_id=device_id, language="en", agent_id=None,
    )


async def test_voice_satellite_honours_follow_up_toggle() -> None:
    entity = _entity_for_continue({})  # continue_conversation defaults True
    result = await entity.async_process(_voice_input())
    assert result.continue_conversation is True


async def test_feedback_guard_device_force_ends() -> None:
    entity = _entity_for_continue({"feedback_loop_devices": ["satdev"]})
    result = await entity.async_process(_voice_input())
    assert result.continue_conversation is False


async def test_close_phrase_still_ends_on_voice() -> None:
    entity = _entity_for_continue({})
    result = await entity.async_process(_voice_input(text="thanks jarvis"))
    assert result.continue_conversation is False


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator: action commands must never be confirmed without a tool call
# ══════════════════════════════════════════════════════════════════════════════


def _provider_saying(text: str) -> MagicMock:
    """Provider that always answers with prose and never calls a tool."""
    from custom_components.ai_plugin.providers import ChatResponse

    provider = MagicMock()
    provider.async_chat = AsyncMock(
        return_value=ChatResponse(content=text, tool_calls=[])
    )
    provider.async_complete = AsyncMock(return_value=text)
    return provider


def _orch_with_actuator_tools() -> tuple[Orchestrator, MagicMock]:
    """Baseline orchestrator whose ha_local exposes one actuator schema."""
    orch = _baseline_orch()
    ha_local = MagicMock()
    ha_local.get_schemas.return_value = [
        {
            "type": "function",
            "function": {
                "name": "set_area_state",
                "description": "sweep",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    ha_local.tool_names = ("set_area_state",)
    ha_local.call_tool = AsyncMock(return_value="OK")
    orch._ha_local = ha_local
    return orch, ha_local


async def test_promise_without_tool_call_is_replaced_with_failure_notice() -> None:
    """The exact observed failure: a spoken promise while nothing happened.

    qwen3.5:9b answered " Switch all lights off." with "I'll turn off all
    the lights in your home now!" and called no tool, so the user heard a
    confirmation while every lamp stayed on.
    """
    from custom_components.ai_plugin.i18n import L

    orch, _ = _orch_with_actuator_tools()
    orch._provider = _provider_saying(
        "I'll turn off all the lights in your home now!"
    )

    reply = await orch.async_process(
        message="Switch all lights off.",
        conversation_id="conv-promise",
        language="en",
    )

    assert reply == L.template("err_action_failed", "en")
    # The grounding retry must have been attempted before giving up.
    assert orch._provider.async_chat.await_count >= 2


def _voice_orch_with_intent(tool_name: str, reply: str) -> Orchestrator:
    """Voice-mode orchestrator whose model calls one MCP intent, then replies."""
    from custom_components.ai_plugin.providers import ChatResponse, ToolCall

    orch = _baseline_orch(
        _make_mock_entry({
            CONF_BASE_URL: "http://localhost:11434/v1",
            CONF_MODEL: "llama3.2:3b",
            CONF_VOICE_MODE: True,
        })
    )
    provider = MagicMock()
    provider.async_chat = AsyncMock(side_effect=[
        ChatResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name=tool_name, arguments={})],
        ),
        ChatResponse(content=reply, tool_calls=[]),
    ])
    orch._provider = provider
    mcp = MagicMock()
    mcp.get_tool_schemas = MagicMock(
        return_value=[{"type": "function", "function": {"name": tool_name}}]
    )
    mcp.call_tool = AsyncMock(return_value='{"response_type": "action_done"}')
    orch._mcp = mcp
    return orch


async def test_started_timer_is_not_retried_as_an_unexecuted_command() -> None:
    """Live 2026-10-29: after start_timer answered, "Set a timer for 13
    minutes." still got the action-command retry, a second model call
    told to switch lights."""
    orch = _voice_orch_with_intent("start_timer", "13 minute timer running.")

    reply = await orch.async_process(
        message="Set a timer for 13 minutes.",
        conversation_id="conv-timer",
        language="en",
    )

    assert reply == "13 minute timer running."
    assert orch._provider.async_chat.await_count == 2


async def test_reasoning_lifts_the_cap_on_an_action_command() -> None:
    """Reasoning counts against the limit: with the cap, "Turn on the light."
    ran out before its tool call (4 of 4 in replay on 2026-11-03)."""
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._entry.options = {**orch._entry.options, "enable_thinking": True}
    orch._provider.thinking_refused = False
    await orch.async_process(
        message="Dim the light.", conversation_id="conv-think", language="en",
    )
    first = orch._provider.async_chat.await_args_list[0]
    assert "max_tokens" not in first.kwargs


async def test_a_model_that_refused_reasoning_keeps_the_cap() -> None:
    from custom_components.ai_plugin.orchestrator.core import (
        ACTION_FIRST_REPLY_MAX_TOKENS,
    )

    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._entry.options = {**orch._entry.options, "enable_thinking": True}
    orch._provider.thinking_refused = True
    orch._hass = None
    await orch.async_process(
        message="Dim the light.", conversation_id="conv-refused", language="en",
    )
    first = orch._provider.async_chat.await_args_list[0]
    assert first.kwargs.get("max_tokens") == ACTION_FIRST_REPLY_MAX_TOKENS


async def test_first_reply_to_an_action_command_is_capped() -> None:
    """Live 2026-10-31: "Dim the light." looped in hidden reasoning up to
    512 tokens, 18.9 s of silence before the retry. Only the first reply is
    capped; the one after the tool result is not."""
    from custom_components.ai_plugin.orchestrator.core import (
        ACTION_FIRST_REPLY_MAX_TOKENS,
    )

    orch = _voice_orch_with_intent("HassLightSet", "")
    await orch.async_process(
        message="Dim the light.", conversation_id="conv-cap", language="en",
    )
    first, second = orch._provider.async_chat.await_args_list
    assert first.kwargs.get("max_tokens") == ACTION_FIRST_REPLY_MAX_TOKENS
    assert "max_tokens" not in second.kwargs


async def test_the_action_command_retry_is_capped_too() -> None:
    orch, _ = _orch_with_actuator_tools()
    orch._provider = _provider_saying("I'll turn off all the lights now!")
    await orch.async_process(
        message="Switch all lights off.", conversation_id="conv-cap2", language="en",
    )
    caps = [c.kwargs.get("max_tokens") for c in orch._provider.async_chat.await_args_list]
    assert len(caps) >= 2 and all(caps), caps


async def test_a_question_is_not_capped() -> None:
    orch = _voice_orch_with_intent("HassGetWeather", "Sunny, 14 degrees.")
    await orch.async_process(
        message="What's the weather like?", conversation_id="conv-cap3", language="en",
    )
    assert all(
        "max_tokens" not in c.kwargs for c in orch._provider.async_chat.await_args_list
    )


async def test_voice_thermostat_change_keeps_its_confirmation() -> None:
    """A new set-point is neither visible nor audible, so it is spoken.

    Observed: "Set the heating to 21 degrees." on a kitchen satellite set
    the thermostat in silence, and the user repeated the command.
    """
    orch = _voice_orch_with_intent(
        "HassClimateSetTemperature", "Heating set to 21 degrees."
    )

    reply = await orch.async_process(
        message="Set the heating to 21 degrees.",
        conversation_id="conv-climate",
        language="en",
    )

    assert reply == "Heating set to 21 degrees."


async def test_voice_light_switch_stays_silent() -> None:
    """A light switch is its own confirmation; the reply stays suppressed."""
    orch = _voice_orch_with_intent("HassTurnOn", "The kitchen light is on.")

    reply = await orch.async_process(
        message="Turn on the kitchen light.",
        conversation_id="conv-light",
        language="en",
    )

    assert reply == ""


async def test_a_silenced_confirmation_stays_in_history() -> None:
    """Live 2026-10-31: after a silent "Dim the lights.", a repeated "Dim
    the light." got no tool call from the first reply. With an empty
    assistant turn in history the voice model treated the previous turn as
    unfinished and answered it; the confirmation it gave there is kept in
    history now, only the speech is suppressed."""
    orch = _voice_orch_with_intent("HassLightSet", "I have dimmed the lights for you.")

    reply = await orch.async_process(
        message="Dim the lights.", conversation_id="conv-hist", language="en",
    )

    assert reply == ""
    history = orch._context_mgr.get_history("conv-hist")
    assert history[-1] == {
        "role": "assistant", "content": "I have dimmed the lights for you.",
    }


async def test_a_silenced_media_confirmation_stays_in_history() -> None:
    orch = _voice_orch_with_intent("HassMediaNext", "Skipped to the next song.")

    reply = await orch.async_process(
        message="Next song.", conversation_id="conv-hist-media", language="en",
    )

    assert reply == ""
    history = orch._context_mgr.get_history("conv-hist-media")
    assert history[-1]["content"] == "Skipped to the next song."


async def test_informative_reply_without_tool_call_is_kept() -> None:
    """A truthful 'can't find it' answer is not a promise — keep it."""
    orch, _ = _orch_with_actuator_tools()
    orch._provider = _provider_saying(
        "There is no device called kettle in the bedroom."
    )

    reply = await orch.async_process(
        message="turn the kettle off",
        conversation_id="conv-informative",
        language="en",
    )

    assert reply == "There is no device called kettle in the bedroom."


# ── a satellite keeps its conversation for 15 minutes ────────────────────────


def _satellite_entity(voice: bool = True):
    entity = _entity_for_continue({})
    entity._orchestrator.is_voice_device = MagicMock(return_value=voice)
    return entity


def _at(
    entity, seconds: float, device: str | None, conversation: str, language: str = "en-US"
) -> str:
    from custom_components.ai_plugin import conversation as module

    with patch.object(module.time, "monotonic", return_value=1000.0 + seconds):
        return entity._satellite_history_id(device, conversation, language)


def test_a_new_ha_conversation_within_15_minutes_continues_the_old_one() -> None:
    entity = _satellite_entity()
    assert _at(entity, 0, "sat", "A") == "A"
    assert _at(entity, 300, "sat", "B") == "A"
    assert _at(entity, 300 + 899, "sat", "C") == "A"


def test_after_15_quiet_minutes_a_satellite_starts_over() -> None:
    entity = _satellite_entity()
    _at(entity, 0, "sat", "A")
    assert _at(entity, 901, "sat", "B") == "B"
    assert _at(entity, 902, "sat", "C") == "B"


def test_switching_the_pipeline_language_starts_a_new_history() -> None:
    """HA can keep its conversation id across the switch, as it did on
    2026-11-03 when English turns were answered in German."""
    entity = _satellite_entity()
    assert _at(entity, 0, "sat", "A", "de-DE") == "A"
    assert _at(entity, 30, "sat", "A", "de-DE") == "A"
    english = _at(entity, 60, "sat", "A", "en-US")
    assert english != "A"
    assert _at(entity, 90, "sat", "B", "en-GB") == english


def test_each_satellite_has_its_own_conversation() -> None:
    entity = _satellite_entity()
    _at(entity, 0, "bedroom", "A")
    assert _at(entity, 1, "kitchen", "B") == "B"
    assert _at(entity, 2, "bedroom", "C") == "A"


@pytest.mark.parametrize("device", [None, "browser"])
def test_the_assist_panel_still_starts_over_when_reopened(device) -> None:
    entity = _satellite_entity(voice=False)
    assert _at(entity, 0, device, "A") == "A"
    assert _at(entity, 1, device, "B") == "B"


@pytest.mark.parametrize("text", ["Enough for today.", "Genug für heute."])
def test_enough_for_today_ends_the_conversation(text) -> None:
    from custom_components.ai_plugin.conversation import _match_close_phrase

    assert _match_close_phrase(text)


@pytest.mark.parametrize(
    "text",
    [
        "Can you recommend me a show on Netflix?",
        "Can you recommend me a show on Netflix? I wanna watch something new.",
        "Recommend a movie on Disney Plus.",
        "I watched a show on Netflix.",
        "Tell me a story about a man on the moon.",
    ],
)
def test_on_as_a_preposition_is_not_a_command(text) -> None:
    from custom_components.ai_plugin.orchestrator.grounding import _is_action_command

    assert not _is_action_command(text)


@pytest.mark.parametrize(
    "text",
    [
        "Lights on.",
        "Lights on",
        "Bedroom off please",
        "Kitchen light on now",
        "Lights on in the kitchen",
        "Can you turn on the lights?",
        "Could you turn off the TV?",
        "Please switch the fan on.",
        "Turn on the light.",
        "Licht an",
        "Licht aus im Bad",
        "Schalte das Licht aus",
        "Mach das Licht aus.",
        "Set a timer for 5 minutes.",
    ],
)
def test_real_commands_still_count(text) -> None:
    from custom_components.ai_plugin.orchestrator.grounding import _is_action_command

    assert _is_action_command(text)


def test_only_a_colour_for_a_whole_room_has_a_room_scope() -> None:
    from custom_components.ai_plugin.orchestrator.core import _room_colour_scope

    assert _room_colour_scope("set_brightness", {"color": "Red", "area": "Kitchen"}) == ("kitchen", "red")
    assert _room_colour_scope("set_brightness", {"color": "red"}) == ("", "red")
    assert _room_colour_scope("set_brightness", {"color": "red", "name": "Lamp"}) is None
    assert _room_colour_scope("set_brightness", {"command": "dimmer"}) is None
    assert _room_colour_scope("set_area_state", {"color": "red"}) is None


async def test_a_second_room_colour_in_one_turn_is_not_run() -> None:
    from types import SimpleNamespace

    from custom_components.ai_plugin.orchestrator.core import ROOM_COLOUR_REPLACED_RESULT

    def call(args):
        return SimpleNamespace(
            name="set_brightness", arguments=args,
            to_assistant_message=lambda: {
                "role": "assistant", "content": None,
                "tool_calls": [{"function": {"name": "set_brightness", "arguments": "{}"}}],
            },
            to_tool_result_message=lambda r: {"role": "tool", "content": r},
        )

    def reply_with(*calls):
        return SimpleNamespace(has_tool_calls=True, tool_calls=list(calls), reply_text=lambda: "")

    orch = _baseline_orch()
    orch._hass = MagicMock()
    orch._provider = MagicMock()
    orch._provider.async_chat = AsyncMock(side_effect=[
        reply_with(call({"command": "set", "color": "red"})),
        reply_with(call({"command": "set", "color": "green"})),
        reply_with(call({"command": "set", "color": "green", "name": "Lamp A"})),
        SimpleNamespace(has_tool_calls=False, tool_calls=[], reply_text=lambda: "Done."),
    ])
    orch._dispatch_tool = AsyncMock(return_value="OK — colour in room: 1 light(s)")
    orch._tool_schemas_for = lambda *a: [
        {"type": "function", "function": {"name": "set_brightness", "parameters": {}}},
    ]
    await orch.async_process("Give each lamp a colour.", "c-colour", "en-US")
    ran = [c.args[1] for c in orch._dispatch_tool.await_args_list]
    assert {"command": "set", "color": "green"} not in ran
    assert {"command": "set", "color": "red"} in ran
    assert {"command": "set", "color": "green", "name": "Lamp A"} in ran
    tool_texts = [
        m["content"] for m in orch._provider.async_chat.await_args_list[-1].args[0]
        if m.get("role") == "tool"
    ]
    assert ROOM_COLOUR_REPLACED_RESULT in tool_texts


@pytest.mark.parametrize(
    "text",
    [
        "Switch every lamp in here to a different color.",
        "Turn off the lights in here.",
        "Turn the lights off here.",
        "Dim this room.",
        "Mach das Licht hier aus.",
        "Mach alle Lampen in diesem Raum blau.",
        "Allume la lumière ici.",
    ],
)
def test_says_this_room(text) -> None:
    from custom_components.ai_plugin.orchestrator.core import says_this_room

    assert says_this_room(text)


@pytest.mark.parametrize(
    "text",
    [
        "Turn off all the lights except in here.",
        "Turn off every light but not here.",
        "Mach alle Lampen aus, außer hier.",
        "Turn off the lights.",
        "Turn off the lights in the kitchen.",
        "Where is Anna?",
    ],
)
def test_does_not_say_this_room(text) -> None:
    from custom_components.ai_plugin.orchestrator.core import says_this_room

    assert not says_this_room(text)


def _confining_orch(room="Living Room", elsewhere=False):
    orch = _baseline_orch()
    orch._this_room = lambda device_id, message: room
    orch._named_device_elsewhere = lambda device, where: elsewhere
    return orch


def test_in_here_replaces_the_area_the_model_sent() -> None:
    orch = _confining_orch()
    args, refused = orch._confine_to_this_room(
        "set_brightness", {"command": "set", "color": "red", "area": "Kitchen"},
        "Switch every lamp in here to a different color.", "sat",
    )
    assert args["area"] == "Living Room" and refused is None


def test_in_here_sets_the_area_the_model_left_out() -> None:
    """Otherwise "all the lights in here" reads as the whole home further on."""
    orch = _confining_orch()
    args, _ = orch._confine_to_this_room(
        "set_area_state", {"domain": "light", "action": "turn_off"},
        "Turn off all the lights in here.", "sat",
    )
    assert args["area"] == "Living Room"


def test_in_here_refuses_a_device_in_another_room() -> None:
    orch = _confining_orch(elsewhere=True)
    args, refused = orch._confine_to_this_room(
        "set_brightness", {"command": "set", "color": "red", "name": "Kitchen Light"},
        "Make every lamp in here red.", "sat",
    )
    assert refused.startswith("[Not changed")
    assert "Kitchen Light" in refused


def test_in_here_lets_a_device_in_this_room_through() -> None:
    orch = _confining_orch()
    call = {"command": "set", "color": "red", "name": "Floor Lamp"}
    args, refused = orch._confine_to_this_room(
        "set_brightness", call, "Make the lamp in here red.", "sat",
    )
    assert args == call and refused is None


@pytest.mark.parametrize(
    ("tool", "message", "room"),
    [
        ("list_entities", "Which lights are on in here?", "Living Room"),
        ("set_brightness", "Make the kitchen red.", "Living Room"),
        # No satellite room, or the message names another room.
        ("set_brightness", "Turn off the lights in here.", None),
    ],
)
def test_nothing_changes_otherwise(tool, message, room) -> None:
    orch = _confining_orch(room=room)
    call = {"area": "Kitchen"}
    args, refused = orch._confine_to_this_room(tool, call, message, "sat")
    assert args == call and refused is None


def test_a_plain_lights_command_stays_in_this_room() -> None:
    orch = _confining_orch()
    args, refused = orch._confine_to_this_room(
        "set_area_state", {"area": "all", "domain": "light", "action": "turn_off"},
        "Turn off the lights.", "sat",
    )
    assert args["area"] == "Living Room" and refused is None


def test_a_plain_lights_command_refuses_a_lamp_in_another_room() -> None:
    orch = _confining_orch(elsewhere=True)
    _, refused = orch._confine_to_this_room(
        "HassTurnOff", {"name": "light.kitchen"}, "Lights off please.", "sat",
    )
    assert refused.startswith("[Not changed")


@pytest.mark.parametrize(
    "message", ["Turn off all the lights.", "Turn off the kitchen lights.", "Turn off Bertha."]
)
def test_a_wider_or_named_request_is_left_alone(message) -> None:
    orch = _confining_orch()
    call = {"area": "all", "domain": "light", "action": "turn_off"}
    args, refused = orch._confine_to_this_room("set_area_state", call, message, "sat")
    assert args == call and refused is None

