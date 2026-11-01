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
    assert "Never claim it was saved or removed." in prompt
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
    assert "[USER FACTS] below are Anna's." in known
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
        schema = _load_schemas()._advanced_schema(current, MagicMock()).schema
        key = next(k for k in schema if str(k) == CONF_MEMORY_ENABLED)
        return key.default()

    assert default({}) is True
    assert default({CONF_MEMORY_ENABLED: False}) is False


@pytest.mark.parametrize("name", ["strings.json", "translations/en.json"])
def test_the_option_has_a_label_and_a_description(name) -> None:
    path = Path(integration.__file__).parent / name
    strings = json.loads(path.read_text(encoding="utf-8"))
    for flow in ("config", "options"):
        step = strings[flow]["step"]["advanced"]
        assert step["data"][CONF_MEMORY_ENABLED]
        assert step["data_description"][CONF_MEMORY_ENABLED]
