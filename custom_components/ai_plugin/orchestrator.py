"""Orchestrator: the core message-processing engine for AI Plugin.

Request flow:
  1.  Determine system prompt (voice / custom / default)
  2.  add_turn(user)
  3.  summarize_if_needed (Week 2)
  4.  get_messages → [system] + trimmed history
  5.  Collect tool schemas: MCP tools + web_search (if enabled)
  6a. Native tool loop (OpenAI function calling):
        LLM → tool calls → execute → repeat → final text
  6b. XML fallback (opt-in, experimental):
        Inject XML tool spec into system prompt, parse <tool_call> tags
  7.  add_turn(assistant)
  8.  Return reply text

Errors: OrchestratorError propagates to conversation.py → graceful reply.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CONTEXT_WINDOW,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_MODEL,
    CONF_RESPONSE_TIMEOUT,
    CONF_SUMMARIZATION_ENABLED,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VOICE_MODE,
    CONF_WEB_SEARCH_ENABLED,
    CONF_XML_FALLBACK,
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_RESPONSE_TIMEOUT,
    DEFAULT_SUMMARIZATION_ENABLED,
    DEFAULT_XML_FALLBACK,
    DOMAIN,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
)
from .context_manager import ContextManager
from .exceptions import OrchestratorError
from .providers.openai_compat import OpenAICompatProvider
from .tools.memory import TOOL_NAMES as MEMORY_TOOL_NAMES, TOOL_SCHEMAS as MEMORY_TOOL_SCHEMAS, MemoryTool
from .tools.web_search import TOOL_SCHEMA as WEB_SEARCH_SCHEMA, WebSearchTool

if TYPE_CHECKING:
    from .providers.base import AbstractProvider
    from .tools.mcp_client import MCPToolRegistry

_LOGGER = logging.getLogger(__name__)

# Strips <tool_call name="...">...</tool_call> tags from LLM replies before
# they are written to canonical history (XML fallback path).
_XML_TOOL_CALL_RE = re.compile(r"<tool_call[^>]*>.*?</tool_call>", re.DOTALL)


class Orchestrator:
    """Processes messages: history → system prompt → LLM → reply.

    One Orchestrator instance per config entry. Created at platform
    setup; replaced on options change via full entry reload.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._hass = hass
        self._entry = entry
        self._provider: AbstractProvider = self._build_provider()

        # Retrieve MCPToolRegistry and MemoryTool created by async_setup_entry.
        entry_data: dict = (
            hass.data.get(DOMAIN, {}).get(entry.entry_id) or {}
            if hass is not None
            else {}
        )
        self._mcp: MCPToolRegistry | None = entry_data.get("mcp")
        self._memory: MemoryTool | None = entry_data.get("memory")

        opts = entry.options
        # Use real schema token budget if MCPToolRegistry is available.
        tool_budget = (
            self._mcp.estimate_schema_tokens() if self._mcp else 2000
        )
        self._context_mgr = ContextManager(
            max_tokens=opts.get(CONF_CONTEXT_WINDOW, DEFAULT_CONTEXT_WINDOW),
            tool_token_budget=tool_budget,
        )
        self._summarization_enabled: bool = opts.get(
            CONF_SUMMARIZATION_ENABLED, DEFAULT_SUMMARIZATION_ENABLED
        )
        self._max_tool_iterations: int = opts.get(
            CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS
        )
        self._xml_fallback: bool = opts.get(CONF_XML_FALLBACK, DEFAULT_XML_FALLBACK)

        # Web search tool (None when disabled in config).
        self._web_search: WebSearchTool | None = (
            WebSearchTool(opts) if opts.get(CONF_WEB_SEARCH_ENABLED, False) else None
        )

        # Per-conversation locks: serialise concurrent requests for the same
        # conversation_id to prevent interleaved history corruption.
        self._conv_locks: dict[str, asyncio.Lock] = {}

        # Tracks most recently controlled HA entities per conversation.
        # Injected into system prompt so models resolve 'it'/'that' even
        # if message history is unavailable.
        self._last_entities: dict[str, str] = {}

    def _get_conv_lock(self, conv_id: str) -> asyncio.Lock:
        if not hasattr(self, "_conv_locks"):
            self._conv_locks = {}
        if conv_id not in self._conv_locks:
            self._conv_locks[conv_id] = asyncio.Lock()
        return self._conv_locks[conv_id]

    def _build_provider(self) -> AbstractProvider:
        """Build provider from current entry options."""
        opts = self._entry.options
        raw_temp = opts.get(CONF_TEMPERATURE)
        raw_top_p = opts.get(CONF_TOP_P)
        raw_max_tokens = opts.get(CONF_MAX_TOKENS, 0)
        return OpenAICompatProvider(
            base_url=opts.get(CONF_BASE_URL, DEFAULT_BASE_URL),
            model=opts.get(CONF_MODEL, ""),
            api_key=opts.get(CONF_API_KEY) or None,
            timeout=opts.get(CONF_RESPONSE_TIMEOUT, DEFAULT_RESPONSE_TIMEOUT),
            temperature=float(raw_temp) if raw_temp is not None else None,
            top_p=float(raw_top_p) if raw_top_p is not None else None,
            max_tokens=int(raw_max_tokens) if raw_max_tokens else None,
            context_window=int(opts.get(CONF_CONTEXT_WINDOW, DEFAULT_CONTEXT_WINDOW)),
        )

    def _build_system_prompt(self, voice_mode: bool) -> str:
        """Return the system prompt for this request.

        Priority:
        1. User-configured custom instructions (if non-empty)
        2. Voice-mode compact prompt (if voice_mode=True)
        3. Default standard prompt
        """
        custom = self._entry.options.get(CONF_SYSTEM_PROMPT, "").strip()
        return custom or (SYSTEM_PROMPT_VOICE if voice_mode else SYSTEM_PROMPT_DEFAULT)

    async def async_process(
        self,
        message: str,
        conversation_id: str,
        language: str,
        device_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Process a user message and return the assistant's reply.

        Args:
            message: The user's text input.
            conversation_id: Stable ID for this conversation thread.
            language: BCP-47 language code (passed through from HA Assist).
            device_id: Optional device ID (used for voice mode detection
                       in future; currently unused in Week 1).

        Returns:
            The assistant's reply text.

        Raises:
            OrchestratorError: on provider failure.
        """
        # Heuristic: if a device_id is present the request came from a voice
        # assistant pipeline (Assist mic → media player → device).  We also
        # honour the explicit CONF_VOICE_MODE config toggle as a fallback for
        # setups where the device_id is unavailable but the user always wants
        # the compact voice prompt.  Note: HA does not expose a dedicated
        # is_voice flag on ConversationInput (Context is not a dict), so
        # device_id presence is the closest reliable signal available.
        voice_mode: bool = (device_id is not None) or bool(
            self._entry.options.get(CONF_VOICE_MODE, False)
        )
        base_prompt = self._build_system_prompt(voice_mode)

        # Inject per-server static context (e.g. HA Assist device inventory).
        # Must precede [LAST ACTION] so the inventory is established first.
        sections: list[str] = [base_prompt]
        if self._mcp is not None:
            for ctx in self._mcp.get_static_contexts():
                sections.append(f"[HOME CONTEXT]\n{ctx}")
        last = self._last_entities.get(conversation_id)
        if last:
            sections.append(f"[LAST ACTION]\n{last}")
        system_prompt = "\n\n".join(sections)

        # Serialise concurrent requests for the same conversation to prevent
        # interleaved history and summarisation races.
        async with self._get_conv_lock(conversation_id):
            # 1. Add user turn to history.
            await self._context_mgr.add_turn(conversation_id, "user", message)

            # 2. Summarize old turns if we're approaching the soft token limit.
            if self._summarization_enabled:
                await self._context_mgr.summarize_if_needed(
                    conversation_id, system_prompt, self._provider
                )

            # 3. Build the message list (system prompt + trimmed history).
            messages = await self._context_mgr.get_messages(conversation_id, system_prompt)

            # 4. Collect tool schemas: memory + web_search + MCP tools.
            tool_schemas = self._mcp.get_tool_schemas() if self._mcp else []
            if self._web_search is not None:
                tool_schemas = [WEB_SEARCH_SCHEMA, *tool_schemas]
            if self._memory is not None:
                tool_schemas = [*MEMORY_TOOL_SCHEMAS, *tool_schemas]

            _LOGGER.debug(
                "Processing message for conv_id=%s lang=%s voice=%s msg_count=%d tools=%d",
                conversation_id,
                language,
                voice_mode,
                len(messages),
                len(tool_schemas),
            )

            # 5. Call LLM — three paths based on config and tool availability:
            #    a) XML fallback (opt-in, experimental) — tool spec injected into prompt
            #    b) Native tool loop (function calling)
            #    c) Plain completion (no tools)
            try:
                if tool_schemas and self._xml_fallback:
                    reply, tool_msgs = await self._xml_tool_loop(messages, tool_schemas, user_id)
                elif tool_schemas:
                    reply, tool_msgs = await self._tool_loop(messages, tool_schemas, user_id)
                else:
                    reply = await self._provider.async_complete(messages)
                    tool_msgs = []
            except Exception:
                # Roll back the user turn so history stays consistent.
                await self._context_mgr.remove_last_turn(conversation_id)
                raise

            # 6. Append tool call/result messages then the assistant reply to history.
            for msg in tool_msgs:
                await self._context_mgr.add_raw_message(conversation_id, msg)
            stored_reply = _XML_TOOL_CALL_RE.sub("", reply).strip() or reply
            await self._context_mgr.add_turn(conversation_id, "assistant", stored_reply)

            # Track last-controlled entities for explicit context injection.
            # Defensive: registry quirks (e.g. ComputedNameType) must never
            # propagate up and kill the user-facing reply.
            try:
                entity_ctx = self._extract_entity_context(tool_msgs)
            except Exception:  # noqa: BLE001
                _LOGGER.exception(
                    "AI Plugin: _extract_entity_context failed — skipping [LAST ACTION] update"
                )
                entity_ctx = None
            if entity_ctx:
                self._last_entities[conversation_id] = entity_ctx
            history_depth = len(self._context_mgr.get_history(conversation_id))
            _LOGGER.info(
                "AI Plugin: conv_id=%s history_depth=%d last_entity=%s",
                conversation_id, history_depth, self._last_entities.get(conversation_id),
            )

        return reply

    async def _tool_loop(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        user_id: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Run the native function-calling loop: call LLM → execute tools → repeat.

        Returns (reply_text, history_messages) where history_messages is the
        list of tool-call assistant messages and tool-result messages in native
        OpenAI format.  Storing these in history gives the LLM full structured
        context on subsequent turns (entity IDs, results, etc.) — far more
        reliable than a text summary for pronoun/reference resolution.
        """
        last_response = None
        history_messages: list[dict] = []

        for iteration in range(self._max_tool_iterations):
            response = await self._provider.async_chat(messages, tool_schemas)
            last_response = response

            if not response.has_tool_calls:
                return response.reply_text(), history_messages

            _LOGGER.debug(
                "Tool loop iteration %d/%d: %d call(s): %s",
                iteration + 1,
                self._max_tool_iterations,
                len(response.tool_calls),
                [tc.name for tc in response.tool_calls],
            )

            for tc in response.tool_calls:
                msg = tc.to_assistant_message()
                messages.append(msg)
                history_messages.append(msg)

            for tc in response.tool_calls:
                result = await self._dispatch_tool(tc.name, tc.arguments, user_id)
                msg = tc.to_tool_result_message(result)
                messages.append(msg)
                history_messages.append(msg)

        suffix = " [Note: reached tool call limit — response may be incomplete.]"
        return (last_response.reply_text() if last_response else "") + suffix, history_messages

    async def _xml_tool_loop(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        user_id: str | None = None,
    ) -> tuple[str, list[dict]]:
        """XML fallback tool loop for models without native function calling.

        Returns (reply_text, []) — XML path doesn't produce structured tool
        messages, so history_messages is always empty.
        """
        import json as _json
        import re as _re

        # Build XML tool spec to append to system prompt.
        tool_spec_lines = ["<tools>"]
        for schema in tool_schemas:
            fn = schema.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = _json.dumps(fn.get("parameters", {}))
            tool_spec_lines.append(
                f'  <tool name="{name}"><description>{desc}</description>'
                f"<parameters>{params}</parameters></tool>"
            )
        tool_spec_lines.append("</tools>")
        tool_spec_lines.append(
            'To call a tool respond with: <tool_call name="tool_name">{"arg": "value"}</tool_call>'
        )
        xml_suffix = "\n".join(tool_spec_lines)

        # Patch system message to include XML spec.
        patched = list(messages)
        if patched and patched[0]["role"] == "system":
            patched[0] = {
                "role": "system",
                "content": patched[0]["content"] + "\n\n" + xml_suffix,
            }

        tool_call_re = _re.compile(
            r'<tool_call\s+name="([^"]+)">(.*?)</tool_call>',
            _re.DOTALL,
        )

        for iteration in range(self._max_tool_iterations):
            reply = await self._provider.async_complete(patched)

            matches = tool_call_re.findall(reply)
            if not matches:
                return reply, []

            _LOGGER.debug(
                "XML tool loop iteration %d/%d: %d call(s)",
                iteration + 1, self._max_tool_iterations, len(matches),
            )

            patched.append({"role": "assistant", "content": reply})
            for name, raw_args in matches:
                try:
                    args = _json.loads(raw_args.strip())
                except _json.JSONDecodeError:
                    args = {}
                result = await self._dispatch_tool(name, args, user_id)
                patched.append({"role": "user", "content": f"Tool result for {name}: {result}"})

        return reply + " [Note: reached tool call limit — response may be incomplete.]", []

    async def _dispatch_tool(self, name: str, arguments: dict, user_id: str | None = None) -> str:
        """Route a tool call to memory, web search, or MCP, never raises."""
        if name in MEMORY_TOOL_NAMES and self._memory is not None:
            return await self._memory.call_tool(name, arguments, user_id=user_id)
        if name == "web_search" and self._web_search is not None:
            query = arguments.get("query", "")
            return await self._web_search.async_search(query)
        if self._mcp is not None:
            return await self._mcp.call_tool(name, arguments)
        return f"[Tool {name!r} unavailable — no handler configured]"

    def _resolve_entity_area(self, name_or_id: str) -> tuple[str | None, str | None]:
        """Resolve a user-facing name (or entity_id) to (entity_id, area_name).

        Reads HA's entity_registry, device_registry, area_registry, and state
        machine to find the matching entity and its containing area. Used so
        [LAST ACTION] can carry the area for pronoun resolution ('that room',
        'there', 'in here').
        """
        hass = self._hass
        if hass is None:
            return (None, None)
        ent_reg = er.async_get(hass)
        area_reg = ar.async_get(hass)
        dev_reg = dr.async_get(hass)

        def _area_for(entry) -> str | None:
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if not area_id:
                return None
            area = area_reg.async_get_area(area_id)
            return area.name if area else None

        # Direct entity_id match
        if "." in name_or_id:
            entry = ent_reg.async_get(name_or_id)
            if entry:
                return (name_or_id, _area_for(entry))

        # Name / alias / friendly_name match — case-insensitive exact.
        # HA's newer registries return ComputedNameType (lazy-evaluated) for
        # some .name fields, which is not a plain str and has no .strip().
        # Coerce via str() so the comparison never blows up on registry quirks.
        def _s(val: object) -> str:
            if val is None:
                return ""
            return val if isinstance(val, str) else str(val)

        needle = name_or_id.strip().lower()
        for entity_id, entry in ent_reg.entities.items():
            candidates = [_s(entry.name), _s(entry.original_name)]
            candidates.extend(_s(a) for a in (entry.aliases or ()))
            state = hass.states.get(entity_id)
            if state:
                candidates.append(_s(state.attributes.get("friendly_name")))
            if any(c and c.strip().lower() == needle for c in candidates):
                return (entity_id, _area_for(entry))
        return (None, None)

    def _extract_entity_context(self, tool_msgs: list[dict]) -> str | None:
        """Build a [LAST ACTION] string including entity_id and area.

        Pronouns like 'that room', 'there', 'in here' require the model to
        know the area of the last-controlled entity. Storing just the name
        is not enough — HA accepts user-facing names but areas live in the
        registry.
        """
        import json as _json
        parts: list[str] = []
        for msg in tool_msgs:
            if msg.get("role") != "assistant" or "tool_calls" not in msg:
                continue
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = (
                        _json.loads(args_raw)
                        if isinstance(args_raw, str)
                        else args_raw
                    )
                except Exception:  # noqa: BLE001
                    continue
                target = (
                    args.get("entity_id")
                    or args.get("name")
                    or args.get("entity")
                )
                area_arg = args.get("area")
                if not target and not area_arg:
                    continue
                if target:
                    entity_id, resolved_area = self._resolve_entity_area(target)
                    area = resolved_area or area_arg
                    canonical = entity_id or target
                    suffix = f" @ {area}" if area else ""
                    parts.append(f"{name}({target} → {canonical}{suffix})")
                else:
                    parts.append(f"{name}(area={area_arg})")
        return "; ".join(parts) if parts else None

    async def async_close(self) -> None:
        """Close provider sessions. Called on integration unload."""
        await self._provider.async_close()
