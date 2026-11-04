"""What the model is offered: every tool costs tokens on each prompt read."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.ai_plugin.orchestrator.core import Orchestrator
from custom_components.ai_plugin.tools.ha_local.schemas import (
    TOOL_SCHEMAS,
    TOOLS_THE_PLUGIN_COVERS,
)


def _schema(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def _orch(ha_local) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch._mcp = MagicMock()
    orch._mcp.get_tool_schemas.return_value = [
        _schema("HassTurnOn"),
        *(_schema(name) for name in sorted(TOOLS_THE_PLUGIN_COVERS)),
        _schema("HassClimateSetTemperature"),
    ]
    orch._web_search = orch._browse_url = orch._memory = orch._voice_id = None
    orch._scheduler = None
    orch._prune_schemas = False
    orch._can_ring_phones = lambda: False
    orch._ha_local = ha_local
    return orch


def _names(orch: Orchestrator) -> list[str]:
    return [s["function"]["name"] for s in orch._tool_schemas_for("hi", True, None)]


def test_tools_the_plugin_does_itself_are_left_out() -> None:
    local = MagicMock()
    local.get_schemas.return_value = [_schema("media_command")]
    assert _names(_orch(local)) == [
        "media_command", "HassTurnOn", "HassClimateSetTemperature",
    ]


def test_without_the_local_tools_home_assistants_own_stay() -> None:
    names = _names(_orch(None))
    assert TOOLS_THE_PLUGIN_COVERS <= set(names)


async def test_a_covered_tool_still_runs_if_the_model_calls_it() -> None:
    """Nothing refuses it, so an old habit in the history does no harm."""
    local = MagicMock()
    local.tool_names = {"media_command"}
    orch = _orch(local)
    orch._hass = MagicMock()
    orch._mcp.call_tool = AsyncMock(return_value="done")
    result = await orch._dispatch_tool("HassSetVolume", {"volume_level": 40})
    assert result == "done"
    orch._mcp.call_tool.assert_awaited_once_with("HassSetVolume", {"volume_level": 40})


def test_exposed_only_is_offered_only_where_the_rules_use_it() -> None:
    offering = {
        s["function"]["name"]
        for s in TOOL_SCHEMAS
        if "exposed_only" in s["function"]["parameters"]["properties"]
    }
    assert offering == {"list_entities", "get_entity"}


def _called_with(arguments) -> dict:
    from custom_components.ai_plugin.providers import normalize_tool_calls

    return normalize_tool_calls(
        {"tool_calls": [{"function": {"name": "start_timer", "arguments": arguments}}]}
    )[0].arguments


@pytest.mark.parametrize(
    "glued, plain",
    [
        ({"minutes=1,name": "one minute"}, {"minutes": 1, "name": "one minute"}),
        ({"seconds=60,name": "one minute"}, {"seconds": 60, "name": "one minute"}),
        ({"hours=1,minutes=30,name": "tea"}, {"hours": 1, "minutes": 30, "name": "tea"}),
        ({"level=0.5,area": "kitchen"}, {"level": 0.5, "area": "kitchen"}),
        ({"exposed_only=false,name_or_id": "sun"}, {"exposed_only": False, "name_or_id": "sun"}),
        ({"command='set',level": 40}, {"command": "set", "level": 40}),
    ],
)
def test_arguments_glued_into_one_key_come_apart(glued, plain) -> None:
    assert _called_with(glued) == plain


def test_a_key_spelled_out_wins_over_a_recovered_one() -> None:
    assert _called_with({"minutes=1,name": "tea", "minutes": 5}) == {"minutes": 5, "name": "tea"}


@pytest.mark.parametrize(
    "arguments",
    [
        {"minutes": 1, "name": "tea"},
        {},
        {"query": "a=b, c=d"},
        {"not a name=1": "x"},
    ],
)
def test_other_arguments_are_left_alone(arguments) -> None:
    assert _called_with(arguments) == arguments


def test_arguments_as_a_json_string_come_apart_too() -> None:
    assert _called_with('{"minutes=1,name": "tea"}') == {"minutes": 1, "name": "tea"}


@pytest.mark.parametrize(
    "said, name",
    [
        ("TV @ Living Room", "TV"),
        ("switch.tv — TV @ Living Room", "TV"),
        ("TV", "TV"),
        ("Rock @ Roll Lamp @ Kitchen", "Rock"),
    ],
)
async def test_a_room_copied_into_the_name_is_left_out(said, name) -> None:
    local = MagicMock()
    local.tool_names = {"media_command"}
    orch = _orch(local)
    orch._hass = MagicMock()
    orch._mcp.call_tool = AsyncMock(return_value="done")
    await orch._dispatch_tool("HassTurnOn", {"name": said, "domain": ["switch"]})
    orch._mcp.call_tool.assert_awaited_once_with("HassTurnOn", {"name": name, "domain": ["switch"]})


def test_examples_are_written_the_way_the_model_writes_a_tool_call() -> None:
    """Shown forget(index=N), gemma writes index=4 in a call, cannot close it, and
    runs digits until the request times out: 8 of 8 for "Forget that I like X"."""
    import re

    from custom_components.ai_plugin.const import SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE

    assignment = re.compile(r"(?<![\\w.=!<>\\[])[a-z_]+=[<'\\w\\[.\"]")
    for prompt in (SYSTEM_PROMPT_VOICE, SYSTEM_PROMPT_DEFAULT):
        assert not assignment.findall(prompt)
    for schema in TOOL_SCHEMAS:
        assert not assignment.findall(schema["function"]["description"])


def test_nothing_asks_the_model_for_an_empty_reply() -> None:
    """Told to return nothing after play_music, gemma with thinking off has no token
    to send and repeats its thought marker until the request times out: 9 of 20 on
    a captured "Play jazz.", 0 of 19 without the wording. Speech is silenced by code,
    on the tool result, so the model need not be silent."""
    import json

    from custom_components.ai_plugin.const import SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE

    for prompt in (SYSTEM_PROMPT_VOICE, SYSTEM_PROMPT_DEFAULT):
        assert "EMPTY reply" not in prompt
        assert "empty string" not in prompt
        assert "except after a playback" not in prompt
    assert "empty string" not in json.dumps(TOOL_SCHEMAS)
