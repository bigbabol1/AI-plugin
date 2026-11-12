"""Home Assistant's own tools, built in instead of through its MCP server."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .test_conversation import _baseline_orch  # noqa: F401  # isort: skip (stubs HA first)

from custom_components.ai_plugin.tools import ha_llm
from custom_components.ai_plugin.tools.ha_llm import HassToolSource
from custom_components.ai_plugin.tools.mcp_client import (
    MCPToolRegistry,
    _MCPServerConnection,
    is_own_ha_mcp,
)

OWN = frozenset({"localhost", "192.168.1.20"})


@dataclasses.dataclass
class _Ctx:
    platform: str
    context: object
    language: str | None
    assistant: str
    device_id: str | None
    agent_id: str | None = None


@dataclasses.dataclass
class _Input:
    tool_name: str
    tool_args: dict


class _Tool:
    def __init__(self, name, required=(), read_only=False):
        self.name = name
        self.description = f"{name} does it"
        self.parameters = {"properties": {"name": {"type": "string"}}, "required": list(required)}
        self.annotations = SimpleNamespace(read_only=read_only)


def _fake_llm(tools, result=None, raises=None):
    seen = {}

    async def get_api(hass, apis, ctx):
        seen.setdefault("apis", []).append(apis)
        seen.setdefault("contexts", []).append(ctx)

        async def call(tool_input):
            seen["input"] = tool_input
            if raises:
                raise raises
            return result

        return SimpleNamespace(
            tools=list(tools), custom_serializer=None, async_call_tool=call
        )

    fake = SimpleNamespace(
        LLMContext=_Ctx, ToolInput=_Input, LLM_API_ASSIST="assist", async_get_api=get_api
    )
    return fake, seen


def _source(fake, mcp_entries=()):
    hass = MagicMock()
    hass.config_entries.async_entries = MagicMock(return_value=list(mcp_entries))
    return HassToolSource(hass)


@pytest.fixture
def plain_openapi():
    with patch.object(ha_llm, "_to_openapi", lambda params, ser: dict(params)):
        yield


async def test_the_tools_are_listed_as_the_mcp_server_lists_them(plain_openapi) -> None:
    fake, seen = _fake_llm([_Tool("intent__HassTurnOn", required=["name"]), _Tool("homeassistant__GetLiveContext")])
    with patch.object(ha_llm, "_llm", lambda: fake):
        source = _source(fake)
        assert await source.async_refresh_tools() is True
        assert await source.async_refresh_tools() is False
    assert [t.name for t in source.tools] == ["intent__HassTurnOn", "homeassistant__GetLiveContext"]
    assert source.tools[0].input_schema == {
        "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
    }
    assert "required" not in source.tools[1].input_schema
    # The list is made without a device, so it is the same on every satellite.
    assert seen["contexts"][0].device_id is None and seen["contexts"][0].language == "*"


async def test_a_call_knows_the_satellite_and_the_person(plain_openapi) -> None:
    result = SimpleNamespace(data={"response_type": "action_done"}, error=False)
    fake, seen = _fake_llm([_Tool("intent__HassTurnOn")], result=result)
    with patch.object(ha_llm, "_llm", lambda: fake):
        out = await _source(fake).call_tool(
            "intent__HassTurnOn", {"name": "Lamp"}, device_id="sat1", user_id="u1"
        )
    assert json.loads(out) == {"response_type": "action_done"}
    ctx = seen["contexts"][-1]
    assert ctx.device_id == "sat1" and ctx.context.user_id == "u1"
    assert seen["input"].tool_name == "intent__HassTurnOn"


@pytest.mark.parametrize(
    ("result", "raises", "expected"),
    [
        (SimpleNamespace(data={"error": "no match"}, error=True), None,
         "[Tool 'HassTurnOn' failed: {\"error\": \"no match\"}]"),
        (None, ValueError("No entity named Lamp"),
         "[Tool 'HassTurnOn' failed: Error calling tool: No entity named Lamp]"),
        # Before Home Assistant 2026.11 a tool returned the data itself.
        ({"success": True}, None, '{"success": true}'),
    ],
)
async def test_results_read_as_they_did_from_the_mcp_server(plain_openapi, result, raises, expected) -> None:
    fake, _ = _fake_llm([_Tool("HassTurnOn")], result=result, raises=raises)
    with patch.object(ha_llm, "_llm", lambda: fake):
        assert await _source(fake).call_tool("HassTurnOn", {}) == expected


async def test_the_mcp_server_entrys_apis_and_read_only_are_kept(plain_openapi) -> None:
    entry = SimpleNamespace(data={"llm_hass_api": ["assist", "other"], "read_only": True}, options={})
    fake, seen = _fake_llm([_Tool("HassTurnOn"), _Tool("GetLiveContext", read_only=True)])
    with patch.object(ha_llm, "_llm", lambda: fake):
        source = _source(fake, [entry])
        await source.async_refresh_tools()
    assert seen["apis"] == [["assist", "other"]]
    assert [t.name for t in source.tools] == ["GetLiveContext"]


async def test_unavailable_tools_never_break_setup() -> None:
    fake = SimpleNamespace(LLM_API_ASSIST="assist", async_get_api=AsyncMock(side_effect=RuntimeError("x")))
    with patch.object(ha_llm, "_llm", lambda: fake):
        source = _source(fake)
        assert await source.async_refresh_tools() is False
    assert not source.ok and not source.connected


@pytest.mark.parametrize(
    ("config", "own"),
    [
        ({"transport": "http", "url": "http://localhost:8123/api/mcp"}, True),
        ({"transport": "sse", "url": "http://192.168.1.20:8123/mcp_server/sse"}, True),
        ({"transport": "http", "url": "http://192.168.1.99:8123/api/mcp"}, False),
        ({"transport": "http", "url": "http://localhost:9000/mcp"}, False),
        ({"transport": "stdio", "command": "uvx", "args": ["mcp-server-time"]}, False),
    ],
)
def test_only_this_home_assistants_own_server_is_replaced(config, own) -> None:
    assert is_own_ha_mcp(config, OWN) is own


def _builtin(ok=True, tools=("HassTurnOn",)):
    b = MagicMock()
    b.ok = ok
    b.connected = ok
    b.is_home_assistant = True
    b.tools = [SimpleNamespace(name=n, description="", input_schema={"type": "object", "properties": {}}) for n in tools]
    b.async_start = AsyncMock()
    b.call_tool = AsyncMock(return_value="done")
    return b


def test_the_built_in_tools_take_the_mcp_servers_place() -> None:
    configs = [
        {"transport": "stdio", "command": "uvx", "args": ["mcp-server-time"]},
        {"transport": "http", "url": "http://localhost:8123/api/mcp", "token": "t"},
        {"transport": "stdio", "command": "uvx", "args": ["mcp-server-calculator"]},
    ]
    builtin = _builtin()
    reg = MCPToolRegistry(configs, builtin=builtin, own_hosts=OWN)
    kinds = [s if s is builtin else s._config.get("args", [None])[-1] for s in reg._servers]
    assert kinds == ["mcp-server-time", builtin, "mcp-server-calculator"]
    # Without one configured, they come first.
    reg = MCPToolRegistry(configs[:1], builtin=builtin, own_hosts=OWN)
    assert reg._servers[0] is builtin and len(reg._servers) == 2


async def test_if_the_built_in_tools_fail_the_configured_server_is_used() -> None:
    builtin = _builtin(ok=False)
    reg = MCPToolRegistry(
        [{"transport": "http", "url": "http://localhost:8123/api/mcp"}], builtin=builtin, own_hosts=OWN
    )
    with patch.object(_MCPServerConnection, "async_start", AsyncMock()):
        await reg.async_setup()
    assert builtin not in reg._servers
    assert isinstance(reg._servers[0], _MCPServerConnection)


async def test_calls_route_to_the_built_in_tools_with_the_satellite() -> None:
    builtin = _builtin()
    reg = MCPToolRegistry([], builtin=builtin, own_hosts=OWN)
    await reg.async_setup()
    assert reg.is_home_assistant_tool("HassTurnOn")
    assert [s["function"]["name"] for s in reg.get_tool_schemas()] == ["HassTurnOn"]
    assert await reg.call_tool("HassTurnOn", {"name": "x"}, device_id="sat1", user_id="u1") == "done"
    builtin.call_tool.assert_awaited_once_with("HassTurnOn", {"name": "x"}, device_id="sat1", user_id="u1")


async def test_the_dispatch_passes_the_satellite_and_person() -> None:
    orch = _baseline_orch()
    orch._mcp = MagicMock()
    orch._mcp.call_tool = AsyncMock(return_value="ok")
    orch._mcp.is_home_assistant_tool = MagicMock(return_value=True)
    orch._hass = None
    await orch._dispatch_tool("HassTurnOn", {"name": "Lamp"}, "u1", device_id="sat1")
    orch._mcp.call_tool.assert_awaited_once_with(
        "HassTurnOn", {"name": "Lamp"}, device_id="sat1", user_id="u1"
    )
