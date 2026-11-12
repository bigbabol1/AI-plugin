"""Home Assistant's own Assist tools, called inside Home Assistant.

These are the tools Home Assistant's MCP server offers (HassTurnOn,
GetLiveContext, exposed scripts, to-do lists, calendar reads): that server
is a thin layer over Home Assistant's LLM API, which this plugin can call
directly because it runs in the same process, the way Home Assistant's own
conversation agents do. Nobody has to add the MCP Server integration or
create a token, a call is a function call instead of an HTTP round trip to
the same machine, and the call knows which satellite and which person it is
for, which an MCP call never did ("turn on the light" could not prefer the
satellite's room).

The tools and their schemas are made the way Home Assistant's MCP server
makes them, so the model sees the same tool list as before. Tool results
come back as the same JSON text. The list is built without a device, as the
MCP server does, so it stays the same from turn to turn and the runtime's
prompt cache keeps working; the device goes with each call.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any

from homeassistant.core import Context, HomeAssistant

_LOGGER = logging.getLogger(__name__)

# The same cap as an MCP call: a hung tool must not stall a voice reply.
_CALL_TIMEOUT = 15.0
PLATFORM = "ai_plugin"


@dataclasses.dataclass(frozen=True)
class HassTool:
    """One tool as the registry sees it: name, description, JSON Schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


def _llm():
    from homeassistant.helpers import llm  # noqa: PLC0415

    return llm


def _to_openapi(parameters: Any, custom_serializer: Any) -> dict[str, Any]:
    """The tool's parameters as JSON Schema, converted as the MCP server does.

    Home Assistant 2026.11 converts with probatio; earlier versions with
    voluptuous_openapi.
    """
    try:
        import probatio  # noqa: PLC0415

        return probatio.to_openapi(
            parameters, custom_serializer=custom_serializer, openapi_version="3.1.0"
        )
    except ImportError:
        from voluptuous_openapi import convert  # noqa: PLC0415

        return convert(parameters, custom_serializer=custom_serializer)


def tool_schema(tool: Any, custom_serializer: Any) -> HassTool:
    """A Home Assistant LLM tool as the MCP server lists it."""
    converted = _to_openapi(tool.parameters, custom_serializer)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": converted.get("properties", {}),
    }
    if converted.get("required"):
        schema["required"] = converted["required"]
    return HassTool(tool.name, tool.description or "", schema)


def llm_context(
    language: str | None = None,
    device_id: str | None = None,
    user_id: str | None = None,
) -> Any:
    """The context a call runs in. Fields differ between Home Assistant
    versions, so only the ones this version has are passed."""
    llm = _llm()
    values: dict[str, Any] = {
        "platform": PLATFORM,
        "context": Context(user_id=user_id),
        # "*" is what the MCP server sends: tool results keep the wording the
        # plugin's checks and the model have seen until now.
        "language": "*" if language is None else language,
        "assistant": "conversation",
        "device_id": device_id,
        "user_prompt": None,
    }
    fields = {f.name for f in dataclasses.fields(llm.LLMContext)}
    return llm.LLMContext(**{k: v for k, v in values.items() if k in fields})


def mcp_server_settings(hass: HomeAssistant) -> tuple[list[str] | str, bool]:
    """The APIs and the read-only switch of Home Assistant's MCP Server
    integration, if it is set up, so its users keep the tools they chose.
    Otherwise Assist, read and write."""
    llm = _llm()
    try:
        entries = hass.config_entries.async_entries("mcp_server")
    except Exception:  # noqa: BLE001
        entries = []
    for entry in entries:
        conf = {**entry.data, **entry.options}
        apis = conf.get("llm_hass_api") or llm.LLM_API_ASSIST
        if apis == "stateless_assist":  # the MCP server's name for it before 2025.6
            apis = llm.LLM_API_ASSIST
        return apis, bool(conf.get("read_only", False))
    return llm.LLM_API_ASSIST, False


class HassToolSource:
    """Home Assistant's Assist tools as one source of the tool registry.

    Has the parts of an MCP server connection the registry uses: ``tools``,
    ``connected``, ``async_refresh_tools`` and ``call_tool``.
    """

    is_home_assistant = True
    name = "Home Assistant"

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        self._tools: list[HassTool] = []
        self.ok = False

    @property
    def tools(self) -> list[HassTool]:
        return self._tools

    @property
    def connected(self) -> bool:
        return self.ok

    async def async_start(self) -> None:
        await self.async_refresh_tools()

    async def async_stop(self) -> None:
        return None

    async def _api(self, context: Any) -> Any:
        llm = _llm()
        apis, read_only = mcp_server_settings(self._hass)
        try:
            api = await llm.async_get_api(self._hass, apis, context)
        except Exception:  # noqa: BLE001 — an API that no longer exists
            if apis == llm.LLM_API_ASSIST:
                raise
            api = await llm.async_get_api(self._hass, llm.LLM_API_ASSIST, context)
        if read_only:
            api.tools = [t for t in api.tools if t.annotations.read_only]
        return api

    async def async_refresh_tools(self) -> bool:
        """Read the tool list again; True when it changed."""
        try:
            api = await self._api(llm_context())
            tools = [tool_schema(t, api.custom_serializer) for t in api.tools]
        except Exception:  # noqa: BLE001 — never break setup over this
            _LOGGER.warning(
                "AI Plugin: Home Assistant's own tools are unavailable", exc_info=True
            )
            self.ok = False
            return False
        changed = tools != self._tools or not self.ok
        self._tools, self.ok = tools, True
        if changed:
            _LOGGER.debug(
                "AI Plugin: %d Home Assistant tools: %s",
                len(tools), ", ".join(t.name for t in tools),
            )
        return changed

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        device_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Call one tool. Never raises: errors come back as bracketed text,
        as they did from the MCP server."""
        llm = _llm()
        # Home Assistant logs only which intent ran, not what it was given.
        _LOGGER.debug(
            "AI Plugin: Home Assistant tool %s(%s) for device %s, user %s",
            name, json.dumps(arguments or {}, ensure_ascii=False), device_id, user_id,
        )
        try:
            api = await self._api(llm_context(device_id=device_id, user_id=user_id))
            result = await asyncio.wait_for(
                api.async_call_tool(llm.ToolInput(tool_name=name, tool_args=arguments or {})),
                timeout=_CALL_TIMEOUT,
            )
        except TimeoutError:
            _LOGGER.warning("AI Plugin: tool %r timed out after %.0fs", name, _CALL_TIMEOUT)
            return (
                f"[Tool {name!r} timed out after {_CALL_TIMEOUT:.0f}s — "
                "tell the user the action may not have completed]"
            )
        except Exception as exc:  # noqa: BLE001 — HomeAssistantError, Invalid, ...
            return f"[Tool {name!r} failed: Error calling tool: {exc}]"
        # Home Assistant 2026.11 returns a ToolResult; earlier versions the data.
        data, error = (
            (result.data, bool(getattr(result, "error", False)))
            if hasattr(result, "data") and not isinstance(result, dict)
            else (result, False)
        )
        text = json.dumps(data, ensure_ascii=False)
        if error:
            return f"[Tool {name!r} failed: {text}]"
        return text
