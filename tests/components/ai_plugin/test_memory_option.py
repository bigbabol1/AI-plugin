"""The long-term memory switch, for setups that bring their own memory."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .test_conversation import _baseline_orch, _make_mock_entry  # isort: skip (stubs HA first)

import custom_components.ai_plugin as integration
from custom_components.ai_plugin.const import (
    CONF_BASE_URL,
    CONF_MEMORY_ENABLED,
    CONF_MODEL,
    DOMAIN,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
)
from custom_components.ai_plugin.orchestrator.prompts import MEMORY_OFF_RULE
from custom_components.ai_plugin.tools.memory import asks_to_remember, claims_saved
from custom_components.ai_plugin.tools.memory import MemoryTool


def _load_schemas():
    """Import config_flow/schemas.py alone: the package's other modules need
    HA's real flow machinery, absent under the test stubs."""
    helpers = sys.modules["homeassistant.helpers"]
    if not hasattr(helpers, "selector"):
        helpers.selector = MagicMock()
        sys.modules["homeassistant.helpers.selector"] = helpers.selector
    name = "custom_components.ai_plugin._schemas_pkg"
    package = ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    path = Path(integration.__file__).parent / "config_flow" / "schemas.py"
    spec = importlib.util.spec_from_file_location(f"{name}.schemas", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _orch(enabled: bool | None):
    options = {CONF_BASE_URL: "http://x/v1", CONF_MODEL: "m"}
    if enabled is not None:
        options[CONF_MEMORY_ENABLED] = enabled
    return _baseline_orch(_make_mock_entry(options=options))


async def _set_up(enabled: bool | None) -> dict:
    """Run the real async_setup_entry and return what it stored."""
    options = {} if enabled is None else {CONF_MEMORY_ENABLED: enabled}
    hass = MagicMock()
    hass.data = {}
    hass.config.config_dir = "/tmp/never-written"
    hass.config_entries.async_forward_entry_setups = AsyncMock()
    entry = SimpleNamespace(
        entry_id="e1",
        options=options,
        add_update_listener=MagicMock(),
        async_on_unload=MagicMock(),
    )
    mcp = MagicMock(async_setup=AsyncMock())
    voice_id = MagicMock(async_load=AsyncMock())
    with (
        patch.object(integration, "MCPToolRegistry", return_value=mcp),
        patch.object(integration, "VoiceIDManager", return_value=voice_id),
        patch.object(integration, "HALocalToolRegistry"),
        patch.object(integration, "_schedule_pipeline_sync"),
        patch.object(integration, "_refresh_tools_on_entity_changes"),
    ):
        assert await integration.async_setup_entry(hass, entry) is True
    return hass.data[DOMAIN]["e1"]


async def test_setup_creates_the_memory_unless_it_is_switched_off() -> None:
    assert isinstance((await _set_up(None))["memory"], MemoryTool)
    assert isinstance((await _set_up(True))["memory"], MemoryTool)
    assert (await _set_up(False))["memory"] is None


@pytest.mark.parametrize("enabled", [None, True])
async def test_memory_on_changes_no_byte_of_the_prompts(enabled) -> None:
    orch = _orch(enabled)
    assert await orch._build_system_prompt(voice_mode=False) == SYSTEM_PROMPT_DEFAULT
    assert await orch._build_system_prompt(voice_mode=True) == SYSTEM_PROMPT_VOICE


@pytest.mark.parametrize(
    ("voice_mode", "next_section"),
    [(False, "[COMPOUND REQUESTS]"), (True, "Weather (strict order):")],
)
async def test_memory_off_replaces_only_its_rules_in_the_prompt(
    voice_mode, next_section
) -> None:
    """No tool rules, and no false "I have noted that" either."""
    prompt = await _orch(False)._build_system_prompt(voice_mode=voice_mode)
    for gone in ("recall", "remember saves", "forget removes", "Memory tools:", "[MEMORY TOOLS]"):
        assert gone not in prompt
    assert "Never say it was noted, saved, remembered or removed" in prompt
    assert "\n\n\n" not in prompt
    original = SYSTEM_PROMPT_VOICE if voice_mode else SYSTEM_PROMPT_DEFAULT
    head, _, _ = original.partition("[MEMORY TOOLS]" if not voice_mode else "Memory tools:")
    _, _, tail = original.partition(next_section)
    assert prompt == head + MEMORY_OFF_RULE + next_section + tail


async def test_memory_off_leaves_memory_out_of_the_speaker_block() -> None:
    """With a memory of their own, "nothing can be saved" would be false."""
    orch = _orch(False)
    known = await orch._build_volatile_block(None, "c1", speaker="Anna")
    unknown = await orch._build_volatile_block(None, "c1", speaker_checked=True)
    assert "recognised by voice as Anna" in known
    assert "[USER FACTS]" not in known
    assert "could not tell who is speaking" in unknown
    assert "saved to memory" not in unknown


async def test_memory_on_keeps_the_speaker_block() -> None:
    orch = _orch(True)
    known = await orch._build_volatile_block(None, "c1", speaker="Anna")
    unknown = await orch._build_volatile_block(None, "c1", speaker_checked=True)
    assert "The newest [USER FACTS] are Anna's." in known
    assert "Nothing can be saved to memory" in unknown


async def test_memory_off_sends_no_memory_tools_and_reads_no_facts() -> None:
    orch = _orch(False)
    names = {
        s["function"]["name"]
        for s in orch._tool_schemas_for("Remember this.", True, None)
    }
    assert not names & {"recall", "remember", "forget"}
    assert await orch._build_user_facts_block(None) == ""


async def test_memory_off_hands_a_tool_of_the_same_name_to_its_mcp_server() -> None:
    orch = _orch(False)
    orch._mcp = MagicMock(call_tool=AsyncMock(return_value="stored elsewhere"))
    orch._mcp.is_home_assistant_tool = MagicMock(return_value=False)
    orch._ha_local = None

    result = await orch._dispatch_tool("remember", {"fact": "x"}, None)

    assert result == "stored elsewhere"
    orch._mcp.call_tool.assert_awaited_once()


def test_the_option_is_on_by_default_and_keeps_its_value() -> None:
    def default(current):
        schema = _load_schemas()._memory_schema(current).schema
        key = next(k for k in schema if str(k) == CONF_MEMORY_ENABLED)
        return key.default()

    assert default({}) is True
    assert default({CONF_MEMORY_ENABLED: False}) is False


@pytest.mark.parametrize("name", ["strings.json", "translations/en.json"])
def test_the_option_has_a_label_and_a_description(name) -> None:
    path = Path(integration.__file__).parent / name
    strings = json.loads(path.read_text(encoding="utf-8"))
    step = strings["options"]["step"]["memory"]
    assert step["data"][CONF_MEMORY_ENABLED]
    assert step["data_description"][CONF_MEMORY_ENABLED]
    assert "memory" in strings["options"]["step"]["init"]["menu_options"]
    for flow in ("config", "options"):
        assert CONF_MEMORY_ENABLED not in strings[flow]["step"]["advanced"]["data"]


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("Remember that I like tea.", "en"),
        ("Don't forget that I like tea.", "en"),
        ("Merk dir, dass ich Tee mag.", "de"),
        ("Vergiss nicht, dass ich Tee mag.", "de"),
        ("Souviens-toi que j'aime le thé.", "fr"),
        ("Recuerda que me gusta el té.", "es"),
        ("Onthoud dat ik van thee houd.", "nl"),
        ("Zapamiętaj, że lubię herbatę.", "pl"),
        ("Lembra-te de que gosto de chá.", "pt"),
        # The words need not match the pipeline's language.
        ("Merk dir, dass ich Tee mag.", "en"),
        # A language with no patterns cannot be judged.
        ("Ricorda che mi piace il tè.", "it"),
    ],
)
def test_explicit_requests_to_remember(message, lang) -> None:
    assert asks_to_remember(message, lang)


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("I live in Utah.", "en"),
        ("I like tea.", "en"),
        ("Ich mag Tee.", "de"),
        ("Das Essen ist fertig.", "de"),
        ("J'aime le thé.", "fr"),
    ],
)
def test_a_statement_is_not_a_request_to_remember(message, lang) -> None:
    assert not asks_to_remember(message, lang)


async def test_remember_refuses_a_statement_without_saving() -> None:
    orch = _baseline_orch()
    orch._memory = SimpleNamespace(call_tool=AsyncMock(return_value="Remembered: x"))
    orch._speaker_unrecognised = lambda device_id: False
    result = await orch._dispatch_tool(
        "remember", {"fact": "the user lives in Utah"}, None, True,
        "I live in Utah.", language="en-US",
    )
    assert result.startswith("[Not saved")
    orch._memory.call_tool.assert_not_awaited()


async def test_remember_saves_on_an_explicit_request() -> None:
    orch = _baseline_orch()
    orch._memory = SimpleNamespace(call_tool=AsyncMock(return_value="Remembered: x"))
    orch._speaker_unrecognised = lambda device_id: False
    result = await orch._dispatch_tool(
        "remember", {"fact": "the user likes tea"}, None, True,
        "Remember that I like tea.", language="en-US",
    )
    assert result == "Remembered: x"


@pytest.mark.parametrize(
    "reply",
    [
        "I've noted that you live in Utah.",
        "I have remembered that.",
        "That sounds lovely, I'll make sure to keep that in mind for our future chats!",
        "Ich habe mir gemerkt, dass du in Utah lebst.",
        "Das habe ich mir gemerkt.",
        "C'est noté, je me souviendrai que tu aimes le thé.",
    ],
)
def test_replies_that_claim_a_save(reply) -> None:
    assert claims_saved(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "Yes, I remember that you like tea.",
        "Do you want me to remember that?",
        "That sounds lovely, I hope you enjoy your tea!",
        "I haven't saved that. If I should keep it, say remember and then the fact.",
        "Das habe ich nicht gespeichert. Wenn ich es behalten soll, sag merk dir und dann, was.",
        "Não guardei isso. Se quiseres que me lembre, diz lembra e depois o quê.",
        "Nie zapisałem tego. Jeśli mam to zapamiętać, powiedz zapamiętaj, a potem co.",
        "No lo he guardado. Si quieres que lo recuerde, di recuerda y luego el dato.",
    ],
)
def test_replies_that_do_not_claim_a_save(reply) -> None:
    assert not claims_saved(reply)


async def test_a_claim_with_nothing_saved_is_replaced_in_speech_and_history() -> None:
    orch = _baseline_orch()
    orch._hass = MagicMock()
    orch._memory = SimpleNamespace(call_tool=AsyncMock(return_value="x"))
    orch._provider = MagicMock()
    orch._provider.async_chat = AsyncMock(return_value=SimpleNamespace(
        has_tool_calls=False, tool_calls=[],
        reply_text=lambda: "I've noted that you live in Utah.",
    ))
    reply = await orch.async_process("I live in Utah.", "c-utah", "en-US")
    assert reply.startswith("I haven't saved that.")
    last = orch._context_mgr.get_history("c-utah")[-1]
    assert last["content"].startswith("I haven't saved that.")


def _orch_replying(text: str, tool_rounds: int = 0):
    orch = _baseline_orch()
    orch._hass = MagicMock()
    orch._memory = None
    replies = [
        SimpleNamespace(
            has_tool_calls=True,
            tool_calls=[SimpleNamespace(
                name="remember", arguments={"fact": "x"},
                to_assistant_message=lambda: {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"function": {"name": "remember", "arguments": "{}"}}],
                },
                to_tool_result_message=lambda r: {"role": "tool", "content": r},
            )],
            reply_text=lambda: "",
        )
    ] * tool_rounds
    replies.append(SimpleNamespace(
        has_tool_calls=False, tool_calls=[], reply_text=lambda: text,
    ))
    orch._provider = MagicMock()
    orch._provider.async_chat = AsyncMock(side_effect=replies)
    orch._dispatch_tool = AsyncMock(return_value="Saved.")
    orch._tool_schemas_for = lambda *a: [
        {"type": "function", "function": {"name": "remember", "parameters": {}}},
    ]
    return orch


async def test_memory_off_replaces_a_claim_without_the_remember_tip() -> None:
    orch = _orch_replying("I've noted that you live in Utah.")
    reply = await orch.async_process("I live in Utah.", "c-off", "en-US")
    assert reply == "I haven't saved that."


async def test_memory_off_leaves_a_reply_after_the_users_own_tool_alone() -> None:
    orch = _orch_replying("I've noted that you like tea.", tool_rounds=1)
    reply = await orch.async_process("Remember that I like tea.", "c-own", "en-US")
    assert reply == "I've noted that you like tea."


async def test_refused_reasoning_is_announced_once() -> None:
    orch = _orch_replying("Hello.")
    orch._hass.services.async_call = AsyncMock()
    orch._provider.thinking_refused = True
    orch._entry.options = {**orch._entry.options, CONF_MODEL: "gemma:test"}
    await orch.async_process("Hi.", "c-a", "en-US")
    orch._provider.async_chat = AsyncMock(return_value=SimpleNamespace(
        has_tool_calls=False, tool_calls=[], reply_text=lambda: "Hello again.",
    ))
    await orch.async_process("Hi again.", "c-a", "en-US")
    calls = [
        c for c in orch._hass.services.async_call.await_args_list
        if c.args[0] == "persistent_notification"
    ]
    assert len(calls) == 1
    assert "gemma:test" in calls[0].args[2]["message"]


async def test_a_refused_forget_does_not_claim_the_fact_is_missing(tmp_path) -> None:
    from custom_components.ai_plugin.tools.memory import MemoryTool

    memory = MemoryTool(str(tmp_path))
    await memory.call_tool("remember", {"fact": "the user likes tea"}, user_id="u")
    out = await memory.call_tool(
        "forget", {"fact": "likes tea"}, user_id="u",
        user_message="Bitte lösche diese Info wieder.",
    )
    assert out.startswith("[Not removed")
    assert "Ask the user" in out
    assert "the user likes tea" in await memory.call_tool("recall", {}, user_id="u")

