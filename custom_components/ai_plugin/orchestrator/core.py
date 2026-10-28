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
import time
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
    CONF_FIND_PHONE,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PRUNE_TOOL_SCHEMAS,
    CONF_SUMMARIZATION_ENABLED,
    CONF_VOICE_MODE,
    CONF_WEB_SEARCH_ENABLED,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_ENABLE_THINKING,
    DEFAULT_FIND_PHONE,
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
from ..shortcuts.phone import (
    RING_PHONE_SCHEMA,
    RING_PHONE_TOOL,
    async_ring_phone,
    match_find_phone,
    people_with_phones,
    person_for_user,
    resolve_owner,
)
from ..tools.ha_local import HALocalToolRegistry
from ..tools.memory import (
    TOOL_NAMES as MEMORY_TOOL_NAMES,
    TOOL_SCHEMAS as MEMORY_TOOL_SCHEMAS,
    WRITE_TOOL_NAMES as MEMORY_WRITE_TOOLS,
    MemoryTool,
)
from ..tools.web_search import TOOL_SCHEMA as WEB_SEARCH_SCHEMA, WebSearchTool
from ..tools.browse_url import TOOL_SCHEMA as BROWSE_URL_SCHEMA, BrowseUrlTool
from ..voice_id import VOICE_ENROLL_SCHEMA, VOICE_ENROLL_TOOL
from .. import satellite_speech

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
from .warmup import PromptWarmupMixin
from .toolcalls import _parse_raw_tool_call, _prune_ha_local_schemas, _readonly_schemas

_LOGGER = logging.getLogger(__name__)

# A look-up tool still running after this long gets the "one second" line on
# voice turns. The line is heard only ~3 s after it is
# started (speech rendering and the speaker starting; measured 2026-10-26),
# and the reply then waits for it to finish. Started at 0.8 s it fired on
# every web search and held answers that were ready before it was even
# heard, so only look-ups still running at 2 s get it.
TOOL_WAIT_LINE_DELAY_S = 2.0

# Returned instead of running a look-up the model already ran, with the same
# arguments, earlier in the same turn. A DHL drop-off question on 2026-10-27
# ran one web search four times over (six searches, ~7 s more) because the
# results named no address; repeating it could never have found one.
REPEATED_LOOKUP_RESULT = (
    "You already ran this exact {name} call in this turn; its result is above "
    "and running it again returns the same thing. Do not repeat it. Answer the "
    "user now with what you found, and say plainly what you could not find."
)


class _WaitLine:
    """Per-tool-loop state of the wait line: pending timer, speaking task."""

    __slots__ = ("task", "timer")

    def __init__(self) -> None:
        self.timer: asyncio.TimerHandle | None = None
        self.task: asyncio.Task | None = None


class Orchestrator(
    EntityContextMixin, PromptBlocksMixin, PromptWarmupMixin, TimerAnnounceMixin
):
    """Processes messages: history → system prompt → LLM → reply.

    One Orchestrator instance per config entry. Created at platform
    setup; replaced on options change via full entry reload.
    """

    # Voice recognition manager (None when the entry has none).
    _voice_id = None
    # device_id → person recognised on the turn being processed, and the
    # enrollment introduction to speak verbatim when the tool started one.
    _turn_speakers: dict | None = None
    _enroll_replies: dict | None = None

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
        self._voice_id = entry_data.get("voice_id")

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


    def _tool_schemas_for(
        self, message: str, voice_mode: bool, device_id: str | None
    ) -> list[dict]:
        """The tool schemas sent with a turn, in the order they are sent."""
        tool_schemas = self._mcp.get_tool_schemas() if self._mcp else []
        if self._web_search is not None:
            tool_schemas = [WEB_SEARCH_SCHEMA, *tool_schemas]
        if self._browse_url is not None:
            tool_schemas = [BROWSE_URL_SCHEMA, *tool_schemas]
        if self._memory is not None:
            tool_schemas = [*MEMORY_TOOL_SCHEMAS, *tool_schemas]
        if (
            voice_mode
            and device_id
            and self._voice_id is not None
            and self._voice_id.enabled
            and self._voice_id.voice_enroll_allowed
        ):
            tool_schemas = [VOICE_ENROLL_SCHEMA, *tool_schemas]
        if self._find_phone_enabled():
            tool_schemas = [RING_PHONE_SCHEMA, *tool_schemas]
        if self._ha_local is not None:
            ha_schemas = self._ha_local.get_schemas()
            if self._prune_schemas:
                # Opt-in since v0.9.27: pruning changes the tool list per
                # message, which busts the runtime's prompt prefix cache.
                ha_schemas = _prune_ha_local_schemas(message, ha_schemas)
            tool_schemas = [*ha_schemas, *tool_schemas]
        return tool_schemas

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

        # A voice-enrollment session on this satellite owns the turn: the
        # answer is a voice sample, never a command (runs before shortcuts,
        # so "turn everything off" is recorded, not executed).
        if self._voice_id is not None and self._voice_id.enrollment_active(device_id):
            enroll_reply = await self._voice_id.async_enrollment_turn(message, device_id)
            if enroll_reply is not None:
                return enroll_reply

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

            # 1d2. "Where is my phone": ring it (Find my phone option). Whose
            # phone it is needs the speaker, so this is the one shortcut that
            # starts voice ID; if the turn still goes to the LLM (nobody
            # recognised), step 3 reuses the same task.
            speaker_task: asyncio.Task | None = None
            owner = (
                match_find_phone(message, lang) if self._find_phone_enabled() else None
            )
            if owner is not None:
                if (
                    not owner
                    and self._voice_id is not None
                    and voice_mode
                    and self._voice_id.should_identify(message, device_id, user_id)
                ):
                    speaker_task = asyncio.get_running_loop().create_task(
                        self._voice_id.async_identify(message, device_id)
                    )
                try:
                    phone_reply = await self._find_phone_shortcut(
                        owner, user_id, speaker_task, lang
                    )
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("AI Plugin: phone shortcut raised", exc_info=True)
                    phone_reply = None
                if phone_reply:
                    _LOGGER.info(
                        "AI Plugin: phone shortcut hit for conv=%s", conversation_id
                    )
                    await self._context_mgr.add_turn(
                        conversation_id, "assistant", phone_reply
                    )
                    return phone_reply

            # 1e. Every shortcut missed, so this turn goes to the LLM: only
            # now is the voice fingerprinted (HA- and shortcut-handled turns
            # never are). It runs while the tool list and history are built
            # and is awaited just before the per-turn context needs it.
            if (
                speaker_task is None
                and self._voice_id is not None
                and voice_mode
                and self._voice_id.should_identify(message, device_id, user_id)
            ):
                speaker_task = asyncio.get_running_loop().create_task(
                    self._voice_id.async_identify(message, device_id)
                )

            # 2. Collect tool schemas: ha_local + memory + web_search + MCP
            # tools. Done before history budgeting so the context manager can
            # reserve space for what is actually sent this request — the
            # constructor-time estimate ran before MCP connected and never
            # covered the built-in schemas, so history could overrun num_ctx
            # and the runtime silently dropped the system prompt.
            tool_schemas = self._tool_schemas_for(message, voice_mode, device_id)
            self._note_prompt_head(system_prompt, tool_schemas, voice_mode)
            schema_tokens = self._context_mgr.estimate_tokens(str(tool_schemas))

            # 3. Volatile per-turn context (time, user facts, last action).
            speaker = None
            if speaker_task is not None:
                try:
                    speaker = await speaker_task
                except Exception:  # noqa: BLE001 — unknown speaker, carry on
                    _LOGGER.debug("AI Plugin: voice ID failed", exc_info=True)
                if speaker is not None:
                    # Memory reads and writes for this turn use the
                    # recognised person's store instead of the anonymous one.
                    user_id = speaker.memory_key
            if self._turn_speakers is None:
                self._turn_speakers = {}
            if device_id:
                if speaker is not None:
                    self._turn_speakers[device_id] = speaker.person_id
                elif speaker_task is not None:
                    # Checked and not recognised: memory writes are refused.
                    self._turn_speakers[device_id] = None
                else:
                    self._turn_speakers.pop(device_id, None)
            volatile = await self._build_volatile_block(
                user_id,
                conversation_id,
                speaker=speaker.name if speaker else None,
                speaker_checked=speaker_task is not None,
            )
            reserved_tokens = schema_tokens + self._context_mgr.estimate_tokens(volatile)

            # 4. The volatile block goes into history directly before the
            # newest user turn and stays there, so the next request repeats
            # this one byte for byte up to its new block and the runtime's
            # prefix cache covers the whole conversation. Sent but not kept,
            # the block made every follow-up re-read all history after the
            # system prompt: ~1.1 s for one web search's worth of history on
            # the voice model (2026-10-28), growing with every turn.
            if volatile:
                await self._context_mgr.add_before_last_user(
                    conversation_id, {"role": "system", "content": volatile}
                )
            messages = await self._context_mgr.get_messages(
                conversation_id, system_prompt, tool_tokens=schema_tokens
            )

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
            # Only when the model used no tool at all: having read the house
            # (the weather entity, the to-do list), it did not answer from
            # training data, and the forced search replaced a grounded answer
            # with other cities' results (2026-10-27, 10-28).
            _needs_search = (
                self._web_search is not None
                and readonly_schemas
                and not tool_msgs
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

    def _is_lookup_tool(self, name: str) -> bool:
        """Tools that reach out to the network (web search, page fetch, MCP
        servers such as Wikipedia) rather than HA, memory or enrollment.

        Home Assistant's own MCP tools are not look-ups: a slow HassTurnOn
        would otherwise say "one second", and since actuator
        replies are not spoken, that line would be all the user heard."""
        if self._ha_local is not None and name in self._ha_local.tool_names:
            return False
        if self._mcp is not None and self._mcp.is_home_assistant_tool(name):
            return False
        return name not in MEMORY_TOOL_NAMES and name != VOICE_ENROLL_TOOL

    def _lookup_key(self, name: str, arguments: object) -> str | None:
        """Identity of a look-up call within one turn, or None for tools that
        may rightly run twice (a second "volume up" is a second step)."""
        if not self._is_lookup_tool(name):
            return None
        import json as _json

        try:
            args = _json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
        return f"{name}:{' '.join(args.lower().split())}"

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
        """Run the tool loop; on a voice turn whose look-up is slow, the
        satellite first says "one second".

        The line starts when a look-up tool is still running after
        TOOL_WAIT_LINE_DELAY_S, so quick tools (time, calculator) never
        trigger it. The reply is held until the line has played: otherwise
        the answer would talk over it, and the follow-up microphone would
        take the end of the line for the end of the answer.
        """
        wait = _WaitLine()
        try:
            return await self._run_tool_loop(
                messages, tool_schemas, user_id, voice_mode, user_message,
                device_id, language, gate, wait,
            )
        finally:
            if wait.timer is not None:
                wait.timer.cancel()
            if wait.task is not None and not wait.task.done():
                try:
                    await asyncio.shield(wait.task)
                except Exception:  # noqa: BLE001 — never fail the turn over it
                    pass

    def _arm_wait_line(
        self, wait: "_WaitLine", name: str, voice_mode: bool,
        device_id: str | None, gate: "_DeltaGate | None",
    ) -> None:
        if (
            wait.task is not None
            or not voice_mode
            or not device_id
            or self._hass is None
            or not self._is_lookup_tool(name)
        ):
            return

        def _speak() -> None:
            wait.timer = None
            if wait.task is not None:
                return
            _LOGGER.debug("AI Plugin: %s is slow — saying the wait line", name)
            # The answer must not stream into TTS while the line plays.
            if gate is not None:
                gate.close()
            wait.task = self._hass.async_create_task(
                satellite_speech.async_say(self._hass, device_id, "tool_wait")
            )

        wait.timer = asyncio.get_running_loop().call_later(TOOL_WAIT_LINE_DELAY_S, _speak)

    async def _run_tool_loop(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        user_id: str | None,
        voice_mode: bool,
        user_message: str,
        device_id: str | None,
        language: str | None,
        gate: "_DeltaGate | None",
        wait: "_WaitLine",
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
        seen_lookups: set[str] = set()
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
                lookup_key = self._lookup_key(tc.name, tc.arguments)
                if lookup_key is not None and lookup_key in seen_lookups:
                    _LOGGER.debug("AI Plugin: %s repeated with the same arguments — not run", tc.name)
                    msg = tc.to_tool_result_message(REPEATED_LOOKUP_RESULT.format(name=tc.name))
                    messages.append(msg)
                    history_messages.append(msg)
                    continue
                if lookup_key is not None:
                    seen_lookups.add(lookup_key)
                self._arm_wait_line(wait, tc.name, voice_mode, device_id, gate)
                tool_started = time.monotonic()
                result = await self._dispatch_tool(
                    tc.name, tc.arguments, user_id, voice_mode, user_message,
                    device_id=device_id, language=language,
                    available_tools=valid_tool_names,
                )
                _LOGGER.debug(
                    "AI Plugin: tool %s took %.2f s", tc.name, time.monotonic() - tool_started
                )
                if wait.timer is not None:
                    wait.timer.cancel()
                    wait.timer = None
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

            # A started voice enrollment speaks its own first question,
            # word for word: a paraphrase is not a question to answer, and a
            # failed start must never be reported as a success.
            enroll_reply = (self._enroll_replies or {}).pop(device_id, None) if device_id else None
            if enroll_reply:
                if gate is not None:
                    gate.close()
                return enroll_reply, history_messages

        _lang = (language or "en").split("-")[0].lower()
        suffix = " " + L.template("note_tool_limit", _lang)
        return (last_response.reply_text() if last_response else "") + suffix, history_messages

    def _speaker_unrecognised(self, device_id: str | None) -> bool:
        """True when voice ID checked this device's turn and knew nobody:
        facts said then would land in a store that is no one's."""
        turn = self._turn_speakers or {}
        return bool(device_id) and device_id in turn and turn[device_id] is None

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
        if name == VOICE_ENROLL_TOOL and self._voice_id is not None:
            return self._start_voice_enrollment(arguments, device_id, language)
        if name == RING_PHONE_TOOL and self._find_phone_enabled():
            return await self._ring_phone_tool(arguments, user_id, device_id, language)
        if name in MEMORY_TOOL_NAMES and self._memory is not None:
            if name in MEMORY_WRITE_TOOLS and self._speaker_unrecognised(device_id):
                return (
                    "[Not saved: the speaker's voice was not recognised, so nothing "
                    "can be stored for them. Tell the user nothing was saved.]"
                )
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


    def _find_phone_enabled(self) -> bool:
        entry = getattr(self, "_entry", None)
        return bool(
            getattr(self, "_hass", None) is not None
            and entry is not None
            and entry.options.get(CONF_FIND_PHONE, DEFAULT_FIND_PHONE) is True
        )

    def _find_person(self, name: str) -> str | None:
        return self._voice_id.find_person(name) if self._voice_id is not None else None

    async def _find_phone_shortcut(
        self,
        owner: str,
        user_id: str | None,
        speaker_task: asyncio.Task | None,
        lang: str,
    ) -> str | None:
        """Ring the phone a find-my-phone turn asks for; None hands the turn
        to the LLM, whose ring_phone tool asks whose phone it is."""
        speaker = person_for_user(self._hass, user_id)
        if speaker is None and speaker_task is not None:
            try:
                identity = await speaker_task
            except Exception:  # noqa: BLE001 — unknown speaker
                identity = None
            speaker = identity.person_id if identity is not None else None
        person_id = resolve_owner(
            self._hass, owner, speaker_person=speaker, find_person=self._find_person
        )
        if person_id is None:
            return None
        return await async_ring_phone(
            self._hass, person_id, lang, own=not owner and person_id == speaker
        )

    async def _ring_phone_tool(
        self,
        arguments: dict,
        user_id: str | None,
        device_id: str | None,
        language: str | None,
    ) -> str:
        """Tool handler: ring the named person's phone, else the speaker's.

        An unknown speaker gets a result naming the people who have a phone,
        so the model asks whose and calls again; nothing rings on a guess.
        """
        lang = (language or "en").split("-")[0].lower()
        name = str(arguments.get("person") or "").strip()
        speaker = (self._turn_speakers or {}).get(device_id) if device_id else None
        speaker = speaker or person_for_user(self._hass, user_id)
        person_id = resolve_owner(
            self._hass, name, speaker_person=speaker, find_person=self._find_person
        )
        if person_id is None:
            known = ", ".join(sorted(people_with_phones(self._hass).values()))
            heard = (
                f"The name {name!r} matches nobody. " if name
                else "The speaker's voice was not recognised. "
            )
            return (
                f"[No phone rang. {heard}Ask the user whose phone to ring "
                f"({known or 'nobody has a phone linked'}), then call ring_phone "
                "with that name.]"
            )
        reply = await async_ring_phone(
            self._hass, person_id, lang, own=not name and person_id == speaker
        )
        if reply is None:
            return (
                f"[No phone rang: {person_id} has no phone with the Home Assistant "
                "app linked to them. Tell the user.]"
            )
        return f"[Done. Tell the user, in their language: {reply}]"

    def _start_voice_enrollment(
        self, arguments: dict, device_id: str | None, language: str | None
    ) -> str:
        """Tool handler: begin a guided enrollment on the calling satellite.

        Without a usable name it falls back to the person voice recognition
        named on this very turn (enrolling another microphone for someone
        already known). On success the first question becomes the reply
        verbatim (see _tool_loop); on failure the result says plainly that
        nothing started, so the model cannot announce a success.
        """
        vid = self._voice_id
        if not device_id or not vid.enabled or not vid.voice_enroll_allowed:
            return "[Voice enrollment is not available here. Nothing was started.]"
        name = str(arguments.get("person") or "").strip()
        person_id = vid.find_person(name) if name else None
        if person_id is None and not name:
            person_id = (self._turn_speakers or {}).get(device_id)
        if person_id is None:
            known = ", ".join(sorted(str(p["name"]) for p in vid.persons().values()))
            heard = f"The name {name!r} matches nobody. " if name else ""
            return (
                f"[Voice enrollment NOT started. {heard}Ask the user which of these "
                f"people they are: {known or 'none'}. Then call this tool again with "
                "that name.]"
            )
        lang = (language or "en").split("-")[0].lower()
        text = vid.start_enrollment(device_id, person_id, lang)
        if self._enroll_replies is None:
            self._enroll_replies = {}
        self._enroll_replies[device_id] = text
        return f"Voice enrollment started. Say exactly: {text}"

    async def async_close(self) -> None:
        """Close provider/tool sessions and cancel background work."""
        for task in list(getattr(self, "_bg_tasks", ())):
            task.cancel()
        if self._web_search is not None:
            await self._web_search.async_close()
        if self._browse_url is not None:
            await self._browse_url.async_close()
        await self._provider.async_close()
