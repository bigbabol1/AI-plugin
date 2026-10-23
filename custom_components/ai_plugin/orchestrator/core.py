"""The Orchestrator itself: one user turn, start to finish.

Request flow:
  1.  Determine system prompt (voice / custom / default)
  2.  add_turn(user)
  3.  summarize_if_needed (Week 2)
  4.  get_messages → [system] + trimmed history
  5.  Collect tool schemas: MCP tools + web_search (if enabled)
  6.  Native tool loop (OpenAI function calling):
        LLM → tool calls → execute → repeat → final text
  7.  add_turn(assistant)
  8.  Return reply text

Errors: OrchestratorError propagates to conversation.py → graceful reply.

The judgement calls this class makes along the way live in sibling
modules: grounding (did the model act?), online (does this need the
web?), toolcalls (which schemas, and malformed calls back), textproc
(cleaning the reply), delta_gate (streaming), location.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date as _date
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    entity_registry as er,
)

from ..const import (
    CONF_CONTEXT_WINDOW,
    CONF_ENABLE_THINKING,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PRUNE_TOOL_SCHEMAS,
    CONF_SUMMARIZATION_ENABLED,
    CONF_VOICE_MODE,
    CONF_WEB_SEARCH_ENABLED,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_ENABLE_THINKING,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_PRUNE_TOOL_SCHEMAS,
    DEFAULT_SUMMARIZATION_ENABLED,
    DOMAIN,
)
from ..context_manager import ContextManager
from ..i18n import L
from ..providers.openai_compat import OpenAICompatProvider
from ..shortcuts import (
    async_try_action_shortcut,
    async_try_domain_sweep_shortcut,
    async_try_media_shortcut,
    try_shortcut,
)
from ..tools.ha_local import HALocalToolRegistry
from ..tools.memory import TOOL_NAMES as MEMORY_TOOL_NAMES, TOOL_SCHEMAS as MEMORY_TOOL_SCHEMAS, MemoryTool
from ..tools.web_search import TOOL_SCHEMA as WEB_SEARCH_SCHEMA, WebSearchTool
from ..tools.browse_url import TOOL_SCHEMA as BROWSE_URL_SCHEMA, BrowseUrlTool

if TYPE_CHECKING:
    from ..providers.base import AbstractProvider
    from ..tools.mcp_client import MCPToolRegistry

from .delta_gate import _DeltaGate
from .grounding import (
    _ACTUATOR_TOOL_NAMES,
    _MISS_RESULT_PATTERNS,
    _any_actuator_call,
    _any_actuator_success,
    _any_list_entities_call,
    _any_media_status_call,
    _any_media_success,
    _any_tool_call,
    _get_entity_missed,
    _is_action_command,
    _is_action_promise,
    _is_state_set_query,
)
from .entity_context import EntityContextMixin
from .location import LocationProvider
from .prompts import PromptBlocksMixin
from .online import _is_online_followup, _is_online_query
from .textproc import (
    _flatten_for_voice,
    _strip_emoji,
    _strip_filler_closing,
    _strip_narration,
)
from .timers import TimerAnnounceMixin
from .toolcalls import _parse_raw_tool_call, _prune_ha_local_schemas, _readonly_schemas

_LOGGER = logging.getLogger(__name__)


class Orchestrator(EntityContextMixin, PromptBlocksMixin, TimerAnnounceMixin):
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
        self._ha_local: HALocalToolRegistry | None = entry_data.get("ha_local")

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
        self._prune_schemas: bool = opts.get(
            CONF_PRUNE_TOOL_SCHEMAS, DEFAULT_PRUNE_TOOL_SCHEMAS
        )
        # Background summarization tasks — kept referenced so they aren't GC'd.
        self._bg_tasks: set[asyncio.Task] = set()
        self._max_tool_iterations: int = opts.get(
            CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS
        )
        # Web search tool (None when disabled in config).
        self._web_search: WebSearchTool | None = (
            WebSearchTool(opts) if opts.get(CONF_WEB_SEARCH_ENABLED, False) else None
        )
        # Browse URL tool — enabled alongside web search (no extra config needed).
        self._browse_url: BrowseUrlTool | None = (
            BrowseUrlTool() if opts.get(CONF_WEB_SEARCH_ENABLED, False) else None
        )

        # Resolves the user's home (entity → hass.config → nothing) and
        # reverse-geocodes coordinates to a city / region label. Lazy:
        # the first await on async_resolve() does the work; subsequent
        # calls return the cached payload.
        self._location: LocationProvider = LocationProvider(hass, entry)

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


    def _is_voice_device(self, device_id: str | None) -> bool:
        """True only when the device_id has an assist_satellite entity.

        Plain-text Assist (sidebar) and REST API calls also carry a
        device_id, but they are not voice — they want the full default
        prompt with detailed replies. Only assist_satellite indicates a
        real STT→TTS voice pipeline.
        """
        if not device_id or self._hass is None:
            return False
        try:
            ent_reg = er.async_get(self._hass)
            for entry in er.async_entries_for_device(ent_reg, device_id):
                if entry.entity_id.startswith("assist_satellite."):
                    return True
        except Exception:  # noqa: BLE001
            _LOGGER.debug("voice-device check failed", exc_info=True)
        return False

    def _build_provider(self) -> AbstractProvider:
        """Build provider from current entry options."""
        return OpenAICompatProvider.from_options(dict(self._entry.options))


    async def async_process(
        self,
        message: str,
        conversation_id: str,
        language: str,
        device_id: str | None = None,
        user_id: str | None = None,
        on_delta=None,
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
        # Voice mode = device has an assist_satellite entity (real voice
        # pipeline: mic → STT → conversation → TTS → speaker). Plain text
        # Assist (sidebar) and REST API calls also carry a device_id but
        # are NOT voice — they expect detailed, listable replies. The old
        # heuristic ``device_id is not None`` over-detected and forced the
        # voice prompt's "one short sentence" rule on every text query,
        # turning rich web_search results into vague summaries.
        # CONF_VOICE_MODE remains as an explicit override for installs
        # that want the compact prompt always.
        voice_mode: bool = self._is_voice_device(device_id) or bool(
            self._entry.options.get(CONF_VOICE_MODE, False)
        )
        # Normalize HA's BCP-47 language ("de-DE", "fr-CA") to bare ISO 639-1.
        lang = (language or "en").split("-")[0].lower()

        # Machine callback from HA's TimerManager (announce-mode timers set
        # a conversation_command that routes back to this agent at expiry).
        # Handle before shortcuts/history — this is not a user utterance.
        from ..const import TIMER_DONE_SENTINEL  # noqa: PLC0415
        if message.startswith(TIMER_DONE_SENTINEL):
            return await self._handle_timer_done(message, device_id, lang)

        # Streaming gate: forward sentence-complete deltas to the voice
        # pipeline for early TTS — but only on turns where no verifier can
        # replace the reply and no suppression can blank it. Everything
        # else keeps the fully-buffered behaviour.
        stream_safe = (
            on_delta is not None
            and not self._entry.options.get(CONF_ENABLE_THINKING, DEFAULT_ENABLE_THINKING)
            and not _is_state_set_query(message)
            and not _is_action_command(message)
            and not (self._web_search is not None and _is_online_query(message))
        )
        gate = _DeltaGate(
            on_delta if stream_safe else None,
            lang=lang,
            voice_mode=voice_mode,
        )
        # v0.5.15: no more [HOME CONTEXT] YAML dump. Small LLMs drown in it and
        # hallucinate their way through inventory questions. The model instead
        # calls discovery tools (list_areas / list_entities / get_entity /
        # search_entities) on demand. Volatile blocks ([CURRENT TIME],
        # [USER FACTS], [LAST ACTION]) ride in a late system message so this
        # prompt stays byte-stable for the runtime's prefix cache.
        system_prompt = await self._build_system_prompt(voice_mode)

        # Evict conversations idle for hours; HA mints a fresh id per Assist
        # session, so these maps otherwise grow for the process lifetime.
        for stale_id in self._context_mgr.evict_idle():
            self._conv_locks.pop(stale_id, None)
            self._last_entities.pop(stale_id, None)

        # Serialise concurrent requests for the same conversation to prevent
        # interleaved history and summarisation races.
        async with self._get_conv_lock(conversation_id):
            # 1. Add user turn to history.
            await self._context_mgr.add_turn(conversation_id, "user", message)

            # 1b. Deterministic pre-LLM shortcut for "<attr> in <area>" questions.
            # Bypasses the tool loop entirely when we can answer from the
            # entity/area registry directly. ~50ms vs ~2-4s via the LLM, and
            # avoids fuzzy-resolve misses on multi-sensor rooms. Falls through
            # on any miss so the LLM still handles ambiguous/novel phrasings.
            try:
                shortcut_reply = try_shortcut(self._hass, message, lang=lang)
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin: shortcut raised", exc_info=True)
                shortcut_reply = None
            if shortcut_reply:
                _LOGGER.info(
                    "AI Plugin: shortcut hit for conv=%s", conversation_id
                )
                await self._context_mgr.add_turn(
                    conversation_id, "assistant", shortcut_reply
                )
                return shortcut_reply

            # 1c. Pre-LLM shortcut for media playback commands (pause, next,
            # skip, resume, stop). Small models hallucinate prose
            # confirmations without invoking media_command; this bypass
            # dispatches the service call deterministically and returns an
            # empty reply for TTS suppression.
            try:
                media_result = await async_try_media_shortcut(self._hass, message, lang=lang)
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin: media shortcut raised", exc_info=True)
                media_result = None
            if media_result is not None:
                handled, media_reply = media_result
                if handled:
                    _LOGGER.info(
                        "AI Plugin: media shortcut hit for conv=%s",
                        conversation_id,
                    )
                    await self._context_mgr.add_turn(
                        conversation_id, "assistant", media_reply
                    )
                    return media_reply

            # 1c2. Pre-LLM shortcut for plural-domain sweeps ("all lights
            # off", "alle Lichter aus"). Runs BEFORE the single-device
            # shortcut so a plural noun is never fuzzy-matched to one lamp.
            # Small models either promise the action without calling a tool
            # or omit the area (= "this room only"), leaving the rest of the
            # home untouched; this dispatches the sweep deterministically.
            try:
                sweep_result = await async_try_domain_sweep_shortcut(
                    self._hass, message, lang=lang, device_id=device_id
                )
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin: sweep shortcut raised", exc_info=True)
                sweep_result = None
            if sweep_result is not None:
                handled, sweep_reply = sweep_result
                if handled:
                    _LOGGER.info(
                        "AI Plugin: sweep shortcut hit for conv=%s",
                        conversation_id,
                    )
                    await self._context_mgr.add_turn(
                        conversation_id, "assistant", sweep_reply
                    )
                    return sweep_reply

            # 1d. Pre-LLM shortcut for single named-device on/off commands
            # (all i18n languages). Small models mis-route "switch X on" /
            # "schalte X ein" (verb vs domain, or only "turn on X" word
            # order); this resolves the named device and dispatches the
            # service directly, returning an empty reply for TTS
            # suppression. Misses / ambiguous devices fall through to the LLM.
            try:
                action_result = await async_try_action_shortcut(
                    self._hass, message, lang=lang, device_id=device_id
                )
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin: action shortcut raised", exc_info=True)
                action_result = None
            if action_result is not None:
                handled, action_reply = action_result
                if handled:
                    _LOGGER.info(
                        "AI Plugin: action shortcut hit for conv=%s",
                        conversation_id,
                    )
                    await self._context_mgr.add_turn(
                        conversation_id, "assistant", action_reply
                    )
                    return action_reply

            # 2. Collect tool schemas: ha_local + memory + web_search + MCP
            # tools. Done before history budgeting so the context manager can
            # reserve space for what is actually sent this request — the
            # constructor-time estimate ran before MCP connected and never
            # covered the built-in schemas, so history could overrun num_ctx
            # and the runtime silently dropped the system prompt.
            tool_schemas = self._mcp.get_tool_schemas() if self._mcp else []
            if self._web_search is not None:
                tool_schemas = [WEB_SEARCH_SCHEMA, *tool_schemas]
            if self._browse_url is not None:
                tool_schemas = [BROWSE_URL_SCHEMA, *tool_schemas]
            if self._memory is not None:
                tool_schemas = [*MEMORY_TOOL_SCHEMAS, *tool_schemas]
            if self._ha_local is not None:
                ha_schemas = self._ha_local.get_schemas()
                if self._prune_schemas:
                    # Opt-in since v0.9.27: pruning changes the tool list per
                    # message, which busts the runtime's prompt prefix cache.
                    ha_schemas = _prune_ha_local_schemas(message, ha_schemas)
                tool_schemas = [*ha_schemas, *tool_schemas]
            schema_tokens = self._context_mgr.estimate_tokens(str(tool_schemas))

            # 3. Volatile per-turn context (time, user facts, last action).
            volatile = await self._build_volatile_block(user_id, conversation_id)
            reserved_tokens = schema_tokens + self._context_mgr.estimate_tokens(volatile)

            # 4. Build the message list (static system prompt + trimmed
            # history), then slot the volatile block in directly before the
            # newest user turn: everything ahead of it is byte-identical to
            # the previous request, so the runtime's prefix cache applies.
            messages = await self._context_mgr.get_messages(
                conversation_id, system_prompt, tool_tokens=reserved_tokens
            )
            if volatile:
                vol_msg = {"role": "system", "content": volatile}
                if messages and messages[-1].get("role") == "user":
                    messages.insert(len(messages) - 1, vol_msg)
                else:
                    messages.append(vol_msg)

            _LOGGER.debug(
                "Processing message for conv_id=%s lang=%s voice=%s msg_count=%d tools=%d",
                conversation_id,
                language,
                voice_mode,
                len(messages),
                len(tool_schemas),
            )

            # Token budget snapshot — one INFO line per user turn for observability.
            if _LOGGER.isEnabledFor(logging.INFO):
                report = self._context_mgr.budget_report(
                    system_prompt, messages, schema_tokens
                )
                log_fn = _LOGGER.warning if report["pct"] >= 0.80 else _LOGGER.info
                log_fn(
                    "AI Plugin budget conv=%s sys=%d tools=%d hist=%d total=%d/%d (%.0f%%)",
                    conversation_id,
                    report["system"],
                    report["tools"],
                    report["history"],
                    report["total"],
                    report["cap"],
                    report["pct"] * 100,
                )

            # 5. Call LLM — native tool loop when tools available, plain
            #    completion otherwise.
            try:
                if tool_schemas:
                    reply, tool_msgs = await self._tool_loop(
                        messages, tool_schemas, user_id, voice_mode, message,
                        device_id=device_id, language=language, gate=gate,
                    )
                elif gate.active:
                    response = await self._provider.async_chat_stream(
                        messages, on_delta=gate.feed
                    )
                    reply = response.reply_text()
                    tool_msgs = []
                else:
                    reply = await self._provider.async_complete(messages)
                    tool_msgs = []
            except Exception:
                # Roll back the user turn so history stays consistent.
                await self._context_mgr.remove_last_turn(conversation_id)
                raise

            # 5b. Grounding verifier: if the user asked a state-set question
            # ("any lights on?", "which switches are off?") and the tool loop
            # never called list_entities, the model is answering from imagination.
            # Inject a corrective system turn and re-run the tool loop ONCE.
            #
            # Grounding retries (5b-5d) are informational — they re-run the
            # loop WITHOUT actuator schemas so a retry can never re-execute
            # the turn's action (double next-track, double toggle). 5e keeps
            # the full set: forcing the action is its purpose.
            readonly_schemas = _readonly_schemas(tool_schemas)
            if (
                readonly_schemas
                and _is_state_set_query(message)
                and not _any_list_entities_call(tool_msgs)
                # media_command('status') already grounds media questions
                # ("what's playing?") deterministically — a list_entities
                # retry would only add tool-loop latency.
                and not _any_media_status_call(tool_msgs)
            ):
                _LOGGER.info(
                    "AI Plugin: grounding retry — state-set query had no list_entities call"
                )
                messages.append({
                    "role": "system",
                    "content": (
                        "CRITICAL: The user asked about the CURRENT state across a domain "
                        "(e.g. which lights are on, any windows open). You did NOT call "
                        "list_entities. Answering from memory is wrong. Call "
                        "list_entities(domain=..., state=...) now, then answer from the "
                        "returned rows."
                    ),
                })
                try:
                    reply2, tool_msgs2 = await self._tool_loop(
                        messages, readonly_schemas, user_id, voice_mode, message,
                        device_id=device_id, language=language,
                    )
                    if _any_list_entities_call(tool_msgs2):
                        reply = reply2
                        tool_msgs = tool_msgs + tool_msgs2
                except Exception:  # noqa: BLE001
                    _LOGGER.exception("AI Plugin: grounding retry failed — keeping original reply")

            # 5c. Fuzzy-resolve verifier: if get_entity returned a no-match
            # ("No entity matches 'bedroom temperature'. Try search_entities.")
            # and the model never fell back to search_entities, force the
            # fallback. Small models accept the miss and tell the user "can't
            # find" instead of broadening the search.
            missed_name = _get_entity_missed(tool_msgs)
            if (
                missed_name is not None
                and readonly_schemas
                and not _any_tool_call(tool_msgs, "search_entities")
            ):
                _LOGGER.info(
                    "AI Plugin: fuzzy-resolve retry — get_entity(%r) missed, "
                    "search_entities was not called",
                    missed_name,
                )
                hint = (missed_name or "").strip() or "the requested entity"
                messages.append({
                    "role": "system",
                    "content": (
                        f"CRITICAL: get_entity returned no match for {hint!r}. "
                        "Do NOT tell the user you cannot find it. CALL "
                        "search_entities with a BROADER, single-keyword query "
                        "(e.g. 'temperature', 'bedroom', 'door', 'stehlampe') "
                        "— never the full phrase. Read the returned list, pick "
                        "the best match, then CALL get_entity with its "
                        "entity_id. Only after both tools have run, answer "
                        "the user from the returned state."
                    ),
                })
                try:
                    reply3, tool_msgs3 = await self._tool_loop(
                        messages, readonly_schemas, user_id, voice_mode, message,
                        device_id=device_id, language=language,
                    )
                    if _any_tool_call(tool_msgs3, "search_entities"):
                        reply = reply3
                        tool_msgs = tool_msgs + tool_msgs3
                except Exception:  # noqa: BLE001
                    _LOGGER.exception(
                        "AI Plugin: fuzzy-resolve retry failed — keeping original reply"
                    )

            # 5d. Online-query grounding verifier: if the message clearly needs
            # live web data and web_search is available but was never called,
            # the model answered from stale training data. Orchestrator calls
            # web_search directly and injects results — no model cooperation.
            # Also fires on short follow-up questions referencing a prior
            # web search result ("when exactly did that happen?").
            _needs_search = (
                self._web_search is not None
                and readonly_schemas
                and not _any_tool_call(tool_msgs, "web_search")
                and (
                    _is_online_query(message)
                    or _is_online_followup(message, tool_msgs)
                )
            )
            if _needs_search:
                _LOGGER.info(
                    "AI Plugin: web-search grounding — orchestrator injecting mandatory search"
                )
                try:
                    _loc = await self._location.async_resolve()
                except Exception:  # noqa: BLE001
                    _loc = {}
                # Append today's date so DDG returns current results instead
                # of evergreen "check Eventbrite" style pages.
                _today = _date.today().strftime("%Y-%m-%d")
                _search_query = f"{message[:260]} ({_today})"
                _search_result = await self._web_search.async_search(
                    _search_query,
                    strip_urls=voice_mode,
                    near_user=False,
                    location=_loc or None,
                    language=(_loc or {}).get("language"),
                )
                messages.append({
                    "role": "system",
                    "content": (
                        "The user's question requires current information. "
                        f"Today's date is {_today}. "
                        "A web search was run automatically. Results:\n\n"
                        f"{_search_result}\n\n"
                        "Answer the user's question using ONLY the above search "
                        "results. Do not use your training data. State what the "
                        "results say; if they do not answer the question, say so."
                    ),
                })
                try:
                    reply4, tool_msgs4 = await self._tool_loop(
                        messages, readonly_schemas, user_id, voice_mode, message,
                        device_id=device_id, language=language,
                    )
                    reply = reply4
                    tool_msgs = tool_msgs + tool_msgs4
                except Exception:  # noqa: BLE001
                    _LOGGER.exception(
                        "AI Plugin: web-search grounding retry failed — keeping original reply"
                    )

            # 5e. Action-command grounding verifier: short STT inputs like
            # "Lights on." / "Bedroom off." make weak local models narrate
            # ("I'll turn the lights on.") instead of calling an actuator
            # tool. _strip_narration then produces an empty reply and the
            # user gets the "couldn't produce" fallback while the device
            # stays off. Inject a corrective and re-run once.
            if (
                tool_schemas
                and _is_action_command(message)
                and not _any_actuator_call(tool_msgs)
            ):
                _LOGGER.info(
                    "AI Plugin: action-command grounding retry — %r had no actuator call",
                    message[:80],
                )
                messages.append({
                    "role": "system",
                    "content": (
                        "CRITICAL: The user issued an ACTION command "
                        f"({message!r}). You must CALL a tool to execute it — "
                        "do NOT just say 'I'll turn it on'. For whole-home "
                        "actions ('all lights on', 'all lights off', 'all "
                        "off'): CALL set_area_state(area='all', "
                        "domain='light', action='turn_on') (or 'turn_off') — "
                        "area='all' is required, omitting it means this room "
                        "only. For one room: pass that room as area. "
                        "For a specific device: CALL "
                        "search_entities then HassTurnOn/HassTurnOff with the "
                        "returned entity_id. Pick the most likely interpretation "
                        "from context — do not ask for clarification. Execute now."
                    ),
                })
                try:
                    reply5, tool_msgs5 = await self._tool_loop(
                        messages, tool_schemas, user_id, voice_mode, message,
                        device_id=device_id, language=language,
                    )
                    if _any_actuator_call(tool_msgs5):
                        reply = reply5
                        tool_msgs = tool_msgs + tool_msgs5
                    elif _is_action_promise(reply5) or _is_action_promise(reply):
                        # Both attempts narrated instead of acting. Speaking
                        # "I'll turn off all the lights" now would tell the
                        # user the action succeeded while nothing moved —
                        # replace it with the truth.
                        _LOGGER.warning(
                            "AI Plugin: action command %r produced no actuator "
                            "call — replacing promise reply %r with failure "
                            "notice",
                            message[:80], (reply5 or reply)[:120],
                        )
                        reply = L.template("err_action_failed", lang)
                except Exception:  # noqa: BLE001
                    _LOGGER.exception(
                        "AI Plugin: action-command grounding retry failed — keeping original reply"
                    )

            # 6. Append tool call/result messages then the assistant reply to history.
            for msg in tool_msgs:
                await self._context_mgr.add_raw_message(conversation_id, msg)
            narration_stripped = _strip_narration(reply, lang=lang)
            # When the entire reply was narration, prefer surfacing the
            # "couldn't produce" fallback (handled below in the empty-reply
            # branch) over playing back useless filler like "I'm checking
            # the temperature for you". Keep the partial-strip result when
            # there is real content left.
            if narration_stripped:
                stored_reply = narration_stripped
            elif reply.strip():
                _LOGGER.warning(
                    "AI Plugin: reply was pure narration (%r) — "
                    "downgrading to fallback so user isn't told an action "
                    "was taken when it wasn't",
                    reply[:120],
                )
                stored_reply = ""
            else:
                stored_reply = reply
            stored_reply = _strip_emoji(stored_reply) or stored_reply
            stored_reply = _strip_filler_closing(stored_reply) or stored_reply
            if voice_mode:
                stored_reply = _flatten_for_voice(stored_reply) or stored_reply

            # TTS suppression for actuator tool calls — the visible/audible
            # change IS the confirmation. Speaking "OK lights on" while the
            # lights are visibly on is redundant and annoying for voice
            # users. Only suppress for actual voice satellites; text Assist
            # still gets a confirmation since the user can't see the device.
            # All actuators are result-gated: a FAILED call performed no
            # visible change, so the model's explanation (or the empty-reply
            # fallback below) must reach the user instead of silence that
            # reads as success.
            media_acted = _any_media_success(tool_msgs)
            actuator_acted = _any_actuator_success(tool_msgs)
            if actuator_acted and voice_mode:
                stored_reply = ""
                reply = ""
            elif media_acted:
                # Music tools always suppress regardless of voice/text —
                # the audio change IS the confirmation and TTS would talk
                # over it on the same speaker.
                stored_reply = ""
                reply = ""
            elif not stored_reply.strip():
                # Defensive: small models occasionally return empty content
                # with no tool calls (prompt/tool confusion). Surface a
                # user-facing fallback so TTS doesn't play silence.
                _LOGGER.warning(
                    "AI Plugin: empty reply from model (conv=%s, tools_called=%d) — substituting fallback",
                    conversation_id, len(tool_msgs),
                )
                reply = L.template("err_no_answer", lang)
                stored_reply = reply

            # Streaming epilogue: send whatever part of the processed reply
            # was not yet forwarded (or the whole reply when nothing was).
            # TTS-suppressed turns arrive here with an empty stored_reply —
            # flush_final sends nothing for those.
            gate.flush_final(stored_reply)

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

        # Summarize AFTER replying, in the background. Inline summarization
        # added a full LLM round-trip to whichever unlucky turn crossed the
        # soft limit; the hard truncation in get_messages covers the gap
        # until the background pass lands.
        if self._summarization_enabled:
            self._schedule_summarization(
                conversation_id, system_prompt, reserved_tokens
            )

        return stored_reply

    def _schedule_summarization(
        self, conv_id: str, system_prompt: str, tool_tokens: int
    ) -> None:
        """Run summarize_if_needed as a background task under the conv lock."""

        async def _run() -> None:
            try:
                async with self._get_conv_lock(conv_id):
                    await self._context_mgr.summarize_if_needed(
                        conv_id, system_prompt, self._provider,
                        tool_tokens=tool_tokens,
                    )
            except Exception:  # noqa: BLE001
                _LOGGER.warning(
                    "AI Plugin: background summarization failed", exc_info=True
                )

        task = asyncio.get_running_loop().create_task(_run())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _tool_loop(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        user_id: str | None = None,
        voice_mode: bool = False,
        user_message: str = "",
        device_id: str | None = None,
        language: str | None = None,
        gate: "_DeltaGate | None" = None,
    ) -> tuple[str, list[dict]]:
        """Run the native function-calling loop: call LLM → execute tools → repeat.

        Returns (reply_text, history_messages) where history_messages is the
        list of tool-call assistant messages and tool-result messages in native
        OpenAI format.  Storing these in history gives the LLM full structured
        context on subsequent turns (entity IDs, results, etc.) — far more
        reliable than a text summary for pronoun/reference resolution.
        """
        import json as _json

        last_response = None
        history_messages: list[dict] = []
        valid_tool_names = {
            s.get("function", {}).get("name")
            for s in tool_schemas
            if s.get("function", {}).get("name")
        }

        for iteration in range(self._max_tool_iterations):
            if gate is not None and gate.active:
                # Drop any unemitted tail from the previous iteration —
                # prose preceding a tool call is narration, never speech.
                gate.new_response()
                response = await self._provider.async_chat_stream(
                    messages, tool_schemas, on_delta=gate.feed
                )
            else:
                response = await self._provider.async_chat(messages, tool_schemas)
            last_response = response

            if not response.has_tool_calls:
                reply_text = response.reply_text()
                # Raw-tool-syntax catcher: small models sometimes write
                # `tool_name(args)` into content instead of emitting a
                # structured tool_call. Recover by parsing and executing.
                recovered = _parse_raw_tool_call(reply_text, valid_tool_names)
                if recovered is not None:
                    # Model wrote a pseudo tool call as prose — nothing it
                    # says this turn is trustworthy enough to stream.
                    if gate is not None:
                        gate.close()
                    rec_name, rec_args = recovered
                    _LOGGER.info(
                        "AI Plugin: recovered raw tool-call from content: %s(%s)",
                        rec_name, rec_args,
                    )
                    call_id = f"call_recovered_{iteration}"
                    assistant_msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": rec_name,
                                "arguments": _json.dumps(rec_args),
                            },
                        }],
                    }
                    messages.append(assistant_msg)
                    history_messages.append(assistant_msg)
                    result = await self._dispatch_tool(
                        rec_name, rec_args, user_id, voice_mode, user_message,
                        device_id=device_id, language=language,
                        available_tools=valid_tool_names,
                    )
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result,
                    }
                    messages.append(tool_msg)
                    history_messages.append(tool_msg)
                    continue
                return reply_text, history_messages

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
                result = await self._dispatch_tool(
                    tc.name, tc.arguments, user_id, voice_mode, user_message,
                    device_id=device_id, language=language,
                    available_tools=valid_tool_names,
                )
                if gate is not None and gate.active:
                    # Actuator turns get TTS-suppressed; get_entity misses
                    # trigger a reply-replacing verifier. Either way the
                    # final text is no longer streamable.
                    if tc.name in _ACTUATOR_TOOL_NAMES:
                        gate.close()
                    elif any(
                        p in (result or "").lower() for p in _MISS_RESULT_PATTERNS
                    ):
                        gate.close()
                msg = tc.to_tool_result_message(result)
                messages.append(msg)
                history_messages.append(msg)

        _lang = (language or "en").split("-")[0].lower()
        suffix = " " + L.template("note_tool_limit", _lang)
        return (last_response.reply_text() if last_response else "") + suffix, history_messages

    async def _dispatch_tool(
        self,
        name: str,
        arguments: dict,
        user_id: str | None = None,
        voice_mode: bool = False,
        user_message: str = "",
        device_id: str | None = None,
        language: str | None = None,
        available_tools: set[str] | None = None,
    ) -> str:
        """Route a tool call to ha_local, memory, web search, or MCP. Never raises."""
        if self._ha_local is not None and name in self._ha_local.tool_names:
            return await self._ha_local.call_tool(
                name, arguments,
                device_id=device_id,
                language=language,
                user_message=user_message,
                available_tools=available_tools,
                announce_agent_id=self._timer_announce_agent_id(),
            )
        if name in MEMORY_TOOL_NAMES and self._memory is not None:
            return await self._memory.call_tool(
                name, arguments, user_id=user_id, user_message=user_message
            )
        if name == "web_search" and self._web_search is not None:
            query = arguments.get("query", "")
            near_user = bool(arguments.get("near_user", False))
            try:
                location = await self._location.async_resolve()
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin: location resolve failed", exc_info=True)
                location = {}
            return await self._web_search.async_search(
                query,
                strip_urls=voice_mode,
                near_user=near_user,
                location=location or None,
                language=(location or {}).get("language"),
            )
        if name == "browse_url" and self._browse_url is not None:
            url = arguments.get("url", "")
            return await self._browse_url.async_browse(url, strip_urls=voice_mode)
        if self._mcp is not None:
            return await self._mcp.call_tool(name, arguments)
        return f"[Tool {name!r} unavailable — no handler configured]"


    async def async_close(self) -> None:
        """Close provider/tool sessions and cancel background work."""
        for task in list(getattr(self, "_bg_tasks", ())):
            task.cancel()
        if self._web_search is not None:
            await self._web_search.async_close()
        if self._browse_url is not None:
            await self._browse_url.async_close()
        await self._provider.async_close()
