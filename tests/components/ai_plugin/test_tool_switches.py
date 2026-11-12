"""Configure -> Tools: built-in features that can be switched off.

Asked for on the simon42 forum (2026-11-11) by a user with his own timers,
alarms and announcements: "Weck mich in einer Stunde" never reached his own
alarm script, and the plugin sometimes answered his announcements.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from .test_conversation import _voice_orch_with_intent  # isort: skip (stubs HA first)

from custom_components.ai_plugin import const as c
from custom_components.ai_plugin.i18n import L
from custom_components.ai_plugin.orchestrator.prompts import prune_feature_rules
from custom_components.ai_plugin.scheduler import Scheduler
from custom_components.ai_plugin.scheduler.tools import TOOL_NAMES as SCHEDULER_NAMES
from custom_components.ai_plugin.shortcuts.phone import RING_PHONE_TOOL
from custom_components.ai_plugin.tools import calendar_events, tell
from custom_components.ai_plugin.tools.ha_local.schemas import TOOL_NAMES as LOCAL_NAMES

NOW = datetime(2026, 11, 1, 10, 0, tzinfo=ZoneInfo("Europe/Berlin"))
PROMPTS = (c.SYSTEM_PROMPT_DEFAULT, c.SYSTEM_PROMPT_VOICE)


def _off(*keys: str) -> dict:
    return {
        c.CONF_BASE_URL: "http://localhost:11434/v1",
        c.CONF_MODEL: "m",
        c.CONF_VOICE_MODE: True,
        **{key: False for key in keys},
    }


def test_the_feature_tool_names_are_the_real_ones() -> None:
    tools = c.FEATURE_TOOLS
    assert tools[c.CONF_TOOL_TIMERS] < LOCAL_NAMES
    assert tools[c.CONF_TOOL_ALARMS] == SCHEDULER_NAMES
    assert tools[c.CONF_TOOL_CALENDAR] == {
        calendar_events.TOOL_NAME, calendar_events.REMOVE_TOOL_NAME,
    }
    assert tools[c.CONF_TOOL_MESSAGES] == {tell.TOOL_NAME}
    assert tools[c.CONF_TOOL_BROADCAST] == {tell.BROADCAST_TOOL_NAME}
    assert tools[c.CONF_TOOL_FIND_PHONE] == {RING_PHONE_TOOL}


def test_everything_is_on_unless_switched_off() -> None:
    assert c.features_off({}) == frozenset()
    assert c.tools_off({c.CONF_TOOL_ALARMS: True}) == frozenset()
    assert c.tools_off({c.CONF_TOOL_CALENDAR: False}) == c.FEATURE_TOOLS[c.CONF_TOOL_CALENDAR]


@pytest.mark.parametrize("prompt", PROMPTS)
def test_with_everything_on_the_prompt_is_unchanged(prompt: str) -> None:
    """Not one byte, so the runtime's prompt cache keeps working."""
    assert prune_feature_rules(prompt, frozenset()) is prompt


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("key", list(c.FEATURE_TOOLS))
def test_a_feature_switched_off_leaves_the_prompt(prompt: str, key: str) -> None:
    out = prune_feature_rules(prompt, frozenset({key}))
    for name in c.FEATURE_TOOLS[key]:
        assert f"{name}(" not in out
    if key == c.CONF_TOOL_BROADCAST:
        assert "HassBroadcast" not in out
    # The rest stays.
    for other, names in c.FEATURE_TOOLS.items():
        if other != key:
            for name in names:
                if f"{name}(" in prompt:
                    assert f"{name}(" in out, (key, name)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_messages_off_keeps_announcements_and_the_reverse(prompt: str) -> None:
    out = prune_feature_rules(prompt, frozenset({c.CONF_TOOL_MESSAGES}))
    assert "tell(" not in out and "'announce': HassBroadcast." in out
    out = prune_feature_rules(prompt, frozenset({c.CONF_TOOL_BROADCAST}))
    assert "tell(person: 'Anna'" in out and "HassBroadcast" not in out


@pytest.mark.parametrize("prompt", PROMPTS)
def test_with_all_off_the_sections_go_too(prompt: str) -> None:
    out = prune_feature_rules(prompt, frozenset(c.FEATURE_TOOLS))
    for gone in (
        "[TIMERS]", "Timers:", "[ALARMS, REMINDERS, LATER]", "Alarms, reminders, later:",
        "Say what these tools answered", "A timer counts down",
    ):
        assert gone not in out
    assert "\n\n\n" not in out
    assert "Music playback" in out or "[MEDIA PLAYBACK" in out


async def test_the_prompt_built_for_a_turn_is_pruned() -> None:
    orch = _voice_orch_with_intent("HassTurnOn", "Ok.")
    orch._entry.options = _off(c.CONF_TOOL_ALARMS)
    prompt = await orch._build_system_prompt(True)
    assert "set_alarm(" not in prompt and "cancel_timer(" in prompt


def _scheduled_orch(options: dict, tools: list[str]):
    orch = _voice_orch_with_intent("HassTurnOn", "Ok.")
    orch._entry.options = options
    orch._hass = None
    sched = Scheduler(MagicMock(), "e1")
    sched.now = lambda: NOW
    orch._scheduler = sched
    orch._mcp.get_tool_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": name}} for name in tools
    ])
    return orch


def _names(orch) -> list[str]:
    return [s["function"]["name"] for s in orch._tool_schemas_for("hi", True, None)]


def test_switched_off_tools_leave_the_tool_list() -> None:
    orch = _scheduled_orch(
        _off(c.CONF_TOOL_ALARMS, c.CONF_TOOL_BROADCAST),
        ["HassTurnOn", "HassBroadcast", "alice_wecker"],
    )
    names = _names(orch)
    assert not set(names) & (SCHEDULER_NAMES | {"HassBroadcast"})
    assert {"HassTurnOn", "alice_wecker", "tell", "add_calendar_event"} <= set(names)
    orch._entry.options = _off()
    assert {"set_alarm", "HassBroadcast"} <= set(_names(orch))


async def test_a_switched_off_tool_goes_to_the_mcp_server() -> None:
    """A tool of the same name on another server gets the call, like memory."""
    orch = _scheduled_orch(_off(c.CONF_TOOL_ALARMS), ["set_alarm"])
    orch._mcp.call_tool = AsyncMock(return_value="your alarm is set")
    out = await orch._dispatch_tool("set_alarm", {"at": "07:00"}, None)
    assert out == "your alarm is set"
    assert not orch._scheduler.jobs()


async def test_a_switched_off_timer_does_not_start() -> None:
    orch = _scheduled_orch(_off(c.CONF_TOOL_TIMERS), [])
    orch._ha_local = MagicMock()
    orch._ha_local.tool_names = LOCAL_NAMES
    orch._ha_local.call_tool = AsyncMock(return_value="started")
    orch._mcp.call_tool = AsyncMock(return_value="[Unknown tool: 'start_timer']")
    out = await orch._dispatch_tool("start_timer", {"minutes": 5}, None)
    assert out.startswith("[Unknown tool")
    orch._ha_local.call_tool.assert_not_awaited()


async def test_broadcast_off_leaves_announcements_alone() -> None:
    orch = _scheduled_orch(_off(c.CONF_TOOL_BROADCAST), [])
    orch._hass = MagicMock()
    assert await orch._broadcast_turn("Durchsage", "c1", "sat", "de") is None


def test_find_phone_off() -> None:
    orch = _scheduled_orch(_off(c.CONF_TOOL_FIND_PHONE), [])
    orch._hass = MagicMock()
    assert not orch._can_ring_phones()
    orch._entry.options = _off()
    assert orch._can_ring_phones()


def _calls(*names: str) -> list[dict]:
    return [{"tool_calls": [{"function": {"name": n}} for n in names]}]


def test_which_tools_count_as_brought() -> None:
    orch = _scheduled_orch(_off(), [])
    assert orch._brought_tool_ran(_calls("alice_wecker"))
    for own in ("set_alarm", "HassTurnOn", "GetLiveContext", "tell", "web_search"):
        assert not orch._brought_tool_ran(_calls(own))
    # Switched off, a tool of that name can only be the user's.
    orch._entry.options = _off(c.CONF_TOOL_ALARMS)
    assert orch._brought_tool_ran(_calls("set_alarm"))


@pytest.mark.parametrize("alarms_on", [True, False])
async def test_the_users_own_alarm_script_is_believed(alarms_on: bool) -> None:
    """ "Weck mich in einer Stunde" ran the user's script. The reply must
    stand, and no retry may push the model to schedule_action as well."""
    orch = _scheduled_orch(
        _off() if alarms_on else _off(c.CONF_TOOL_ALARMS), ["alice_wecker"]
    )
    orch._provider.async_chat = AsyncMock(side_effect=[
        _tool_call("alice_wecker"),
        _text("Ich habe den Wecker auf 11 Uhr gestellt."),
        _text("A retry ran."),
    ])
    orch._mcp.call_tool = AsyncMock(return_value="ok")
    reply = await orch.async_process("Weck mich in einer Stunde.", "c-own", "de")
    assert reply == "Ich habe den Wecker auf 11 Uhr gestellt."
    assert orch._provider.async_chat.await_count == 2
    assert not orch._scheduler.jobs()


async def test_a_claim_with_no_tool_at_all_is_still_replaced() -> None:
    orch = _scheduled_orch(_off(c.CONF_TOOL_ALARMS), [])
    orch._provider.async_chat = AsyncMock(side_effect=[
        _text("I've updated the reminder to 15:30."),
    ])
    reply = await orch.async_process("It's at half past four.", "c-none", "en")
    assert reply == L.template("schedule_switched_off", "en")


def _tool_call(name: str):
    from custom_components.ai_plugin.providers import ChatResponse, ToolCall

    return ChatResponse(content=None, tool_calls=[ToolCall(id="c1", name=name, arguments={})])


def _text(text: str):
    from custom_components.ai_plugin.providers import ChatResponse

    return ChatResponse(content=text, tool_calls=[])


async def test_broadcast_off_is_not_offered_or_carried_out() -> None:
    orch = _scheduled_orch(_off(c.CONF_TOOL_BROADCAST), ["HassBroadcast"])
    tell_desc = next(
        s["function"]["description"] for s in orch._tool_schemas_for("hi", True, None)
        if s["function"]["name"] == "tell"
    )
    assert "HassBroadcast" not in tell_desc
    orch._mcp.is_home_assistant_tool = MagicMock(return_value=True)
    orch._mcp.call_tool = AsyncMock(return_value="broadcast")
    out = await orch._dispatch_tool("HassBroadcast", {"message": "Essen"}, None)
    assert out.startswith("[HassBroadcast is switched off")
    orch._mcp.call_tool.assert_not_awaited()
    # On, the hint stays word for word.
    orch._entry.options = _off()
    assert any(
        "For every room use HassBroadcast." in s["function"].get("description", "")
        for s in orch._tool_schemas_for("hi", True, None)
    )
