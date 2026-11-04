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
import contextvars
import logging
import time
from datetime import date as _date
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

from ..const import (
    CONF_CONTEXT_WINDOW,
    CONF_ENABLE_THINKING,
    CONF_MODEL,
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
from ..i18n import L, SUPPORTED_LANGS
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
from ..shortcuts.registry import _caller_area_id
from ..tools.ha_local import HALocalToolRegistry
from ..tools.ha_local.formatting import _s
from ..tools.ha_local.lights import is_generic_lights_command
from ..tools.ha_local.lights import hass_light_set_args
from ..scheduler import (
    asks_for_phrase_routine,
    TOOL_NAMES as SCHEDULER_TOOL_NAMES,
    TOOL_SCHEMAS as SCHEDULER_TOOL_SCHEMAS,
    WRITE_TOOL_NAMES as SCHEDULER_WRITE_TOOL_NAMES,
    Job,
    async_call_tool as async_call_scheduler_tool,
    names_later_time,
)
from ..tools.calendar_events import (
    TOOL_NAME as CALENDAR_TOOL,
    TOOL_SCHEMA as CALENDAR_SCHEMA,
    async_add_event,
)
from ..tools.ha_local.schemas import MOVE_MUSIC_SCHEMA, TOOLS_THE_PLUGIN_COVERS
from ..tools.tell import TOOL_NAME as TELL_TOOL, TOOL_SCHEMA as TELL_SCHEMA, async_tell
from ..tools.memory import (
    asks_to_remember,
    claims_saved,
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
    _TIMER_WRITE_TOOL_NAMES,
    _ACTUATOR_TOOL_NAMES,
    _MISS_RESULT_PATTERNS,
    _any_actuator_call,
    _any_actuator_success,
    _any_list_entities_call,
    _any_media_status_call,
    _any_media_success,
    _any_tool_call,
    _any_unobservable_action,
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
from .toolcalls import (
    _parse_raw_tool_call,
    _prune_ha_local_schemas,
    _readonly_schemas,
    name_without_room,
)

_LOGGER = logging.getLogger(__name__)

# A look-up tool still running after this long gets the "one second" line on
# voice turns. The line is heard only ~3 s after it is
# started (speech rendering and the speaker starting; measured 2026-10-26),
# and the reply then waits for it to finish. Started at 0.8 s it fired on
# every web search and held answers that were ready before it was even
# heard, so only look-ups still running at 2 s get it.
TOOL_WAIT_LINE_DELAY_S = 2.0

# Token limit on the model's first reply to an action command ("dim the
# lights"). The right reply there is a tool call of 10-40 tokens; the longest
# of 750 replies to HA from 2026-10-23 to 2026-10-31 was 116 tokens. On
# 2026-10-31 a "Dim the light." reply instead looped in hidden reasoning up
# to the configured 512 tokens, 18.9 s of silence before the action-command
# retry could step in. Capped, such a reply ends after ~4 s.
ACTION_FIRST_REPLY_MAX_TOKENS = 128

# Returned instead of a second room-wide colour in one turn. Asked for "each lamp
# a different colour", K2 Horizon sent three colours to the whole room and the last
# replaced the others, in silence, since each call succeeded (2026-11-03).
ROOM_COLOUR_REPLACED_RESULT = (
    "[Not changed: another colour for the whole room would only replace the one "
    "just set. To give lamps different colours, call set_brightness once per lamp "
    "with name and color; list_entities(domain: 'light') names them. If one colour "
    "was all that was asked for, stop here.]"
)

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


# True while a scheduled command runs. It runs in its own task, so a context
# variable cannot leak into a turn someone is speaking at the same moment.
_IN_SCHEDULED_RUN: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ai_plugin_in_scheduled_run", default=False
)


def says_this_room(message: str) -> bool:
    """True when a command is for the room the speaker is in: "in here", "hier"."""
    text = (message or "").lower()

    def found(key: str) -> bool:
        return any(
            rx.search(text)
            for code in SUPPORTED_LANGS
            for rx in L.pattern_list(key, code)
        )

    return found("this_room") and not found("this_room_except")


def _room_colour_scope(name: str, arguments: dict) -> tuple[str, str] | None:
    """(room, colour) when this call colours a whole room, else None."""
    if name != "set_brightness" or arguments.get("name"):
        return None
    colour = str(arguments.get("color") or "").strip().lower()
    if not colour:
        return None
    return str(arguments.get("area") or "").strip().lower(), colour


def _remember_was_refused(tool_msgs: list[dict]) -> bool:
    """True when a tool result in this turn says nothing was saved."""
    return any(
        str(m.get("content") or "").startswith("[Not saved")
        for m in tool_msgs
        if m.get("role") == "tool"
    )


class Orchestrator(
    EntityContextMixin, PromptBlocksMixin, PromptWarmupMixin, TimerAnnounceMixin
):
    """Processes messages: history → system prompt → LLM → reply.

    One Orchestrator instance per config entry. Created at platform
    setup; replaced on options change via full entry reload.
    """

    _thinking_announced = False

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
        self._scheduler = entry_data.get("scheduler")
        if self._scheduler is not None:
            self._scheduler.set_command_runner(self._run_scheduled_command)

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
        if self._ha_local is not None:
            tool_schemas = [
                s for s in tool_schemas
                if s.get("function", {}).get("name") not in TOOLS_THE_PLUGIN_COVERS
            ]
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
        if self._can_ring_phones():
            tool_schemas = [RING_PHONE_SCHEMA, *tool_schemas]
        if self._ha_local is not None:
            ha_schemas = self._ha_local.get_schemas()
            if self._prune_schemas:
                # Opt-in since v0.9.27: pruning changes the tool list per
                # message, which busts the runtime's prompt prefix cache.
                ha_schemas = _prune_ha_local_schemas(message, ha_schemas)
            tool_schemas = [*ha_schemas, *tool_schemas]
        # Last on purpose: this model reacts to its neighbours in the list. A
        # tool placed next to set_area_state left a repeated light command
        # uncalled (12 of 12 replays, 2026-10-29).
        if getattr(self, "_scheduler", None) is not None:
            tool_schemas = [
                *tool_schemas, *SCHEDULER_TOOL_SCHEMAS, CALENDAR_SCHEMA, TELL_SCHEMA,
                MOVE_MUSIC_SCHEMA,
            ]
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

        # "Stop" or "snooze" while an alarm rings: before everything else, or
        # the media shortcut takes "stop" and the alarm rings on.
        if getattr(self, "_scheduler", None) is not None:
            alarm_reply = await self._scheduler.async_handle_spoken(message, device_id, lang)
            if alarm_reply is not None:
                return alarm_reply

        # Streaming gate: forward sentence-complete deltas to the voice
        # pipeline for early TTS — but only on turns where no verifier can
        # replace the reply and no suppression can blank it. Everything
        # else keeps the fully-buffered behaviour.
        stream_safe = (
            on_delta is not None
            and not self._reasoning_active()
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
            # A command for later must not be carried out now: the shortcuts
            # below all act at once and none of them reads a time.
            later = names_later_time(message, lang)

            # "When I say good night, ...": nothing can keep that yet. Left to
            # the model it answered "I have scheduled that" with no tool at all.
            if asks_for_phrase_routine(message, lang):
                refused = L.template("phrase_routine_refused", lang)
                await self._context_mgr.add_turn(conversation_id, "assistant", refused)
                return refused

            try:
                media_result = (
                    None if later
                    else await async_try_media_shortcut(
                        self._hass, message, lang=lang, device_id=device_id
                    )
                )
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
                sweep_result = None if later else await async_try_domain_sweep_shortcut(
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
                action_result = None if later else await async_try_action_shortcut(
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

            # 1d2. "Where is my phone": ring it (Find my phone). Whose
            # phone it is needs the speaker, so this is the one shortcut that
            # starts voice ID; if the turn still goes to the LLM (nobody
            # recognised), step 3 reuses the same task.
            speaker_task: asyncio.Task | None = None
            owner = (
                match_find_phone(message, lang) if self._can_ring_phones() else None
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
                # "Are any timers running?" and "What alarms are on?" have tools
                # of their own. The retry made the model read its whole prompt
                # in again (6-8 s), to answer from the wrong tool.
                and not _any_tool_call(tool_msgs, "timer_status")
                and not _any_tool_call(tool_msgs, "list_scheduled")
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
                        "list_entities(domain: ..., state: ...) now, then answer from the "
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
                # "Execute now" would be exactly wrong for "in 30 minutes", so
                # that retry can only schedule; the actuators are not offered.
                retry_schemas = tool_schemas
                if later:
                    retry_schemas = [
                        *_readonly_schemas(tool_schemas),
                        *[
                            t for t in SCHEDULER_TOOL_SCHEMAS
                            if t["function"]["name"] == "schedule_action"
                        ],
                    ]
                messages.append({
                    "role": "system",
                    "content": (
                        "CRITICAL: The user wants this done LATER "
                        f"({message!r}). CALL schedule_action with the command "
                        "and in_minutes or at. Do not do it now and do not "
                        "start a timer."
                    ) if later else (
                        "CRITICAL: The user issued an ACTION command "
                        f"({message!r}). You must CALL a tool to execute it — "
                        "do NOT just say 'I'll turn it on'. For whole-home "
                        "actions ('all lights on', 'all lights off', 'all "
                        "off'): CALL set_area_state(area: 'all', "
                        "domain: 'light', action: 'turn_on') (or 'turn_off') — "
                        "area: 'all' is required, omitting it means this room "
                        "only. For one room: pass that room as area. "
                        "For a specific device: CALL "
                        "search_entities then HassTurnOn/HassTurnOff with the "
                        "returned entity_id. Pick the most likely interpretation "
                        "from context — do not ask for clarification. Execute now."
                    ),
                })
                try:
                    reply5, tool_msgs5 = await self._tool_loop(
                        messages, retry_schemas, user_id, voice_mode, message,
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
            # reads as success. A thermostat change is neither visible nor
            # audible, so a turn that touched one keeps its spoken reply.
            #
            # Suppression silences the reply only: history keeps what the
            # model said. With an empty assistant turn after the tool result,
            # the voice model treated the previous turn as unfinished, and a
            # repeated "Dim the light." got the missing confirmation ("I have
            # dimmed the lights for you.") or a stalled reply instead of a
            # tool call: 0 of 10 replays acted, the action-command retry had
            # to. With the confirmation kept, 10 of 10 acted on the first
            # reply. "Done." in its place did not help (0 of 10, all stalls).
            media_acted = _any_media_success(tool_msgs)
            actuator_acted = _any_actuator_success(tool_msgs)
            # The rule in the prompt alone left "I live in Utah." answered with
            # "I've noted that you live in Utah." and no tool call, 8 of 8 in
            # replay. A claim that nothing backs must not reach the user or
            # history, where the model would take it as true. Turns with other
            # tools stay out of this: "noted" can be about a calendar entry, or
            # come from a memory tool the user brought. With long-term memory
            # off it happened 18 of 18, and "say remember" would be no help.
            if (
                not tool_msgs or _remember_was_refused(tool_msgs)
            ) and claims_saved(stored_reply):
                _LOGGER.info(
                    "AI Plugin: reply claimed a fact was saved, but none was: %r",
                    stored_reply[:120],
                )
                stored_reply = reply = L.template(
                    "remember_not_saved"
                    if self._memory is not None
                    else "remember_not_saved_short",
                    lang,
                )
            history_reply = stored_reply
            if (
                actuator_acted
                and voice_mode
                and not _any_unobservable_action(tool_msgs)
            ):
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
                history_reply = reply

            # Streaming epilogue: send whatever part of the processed reply
            # was not yet forwarded (or the whole reply when nothing was).
            # TTS-suppressed turns arrive here with an empty stored_reply —
            # flush_final sends nothing for those.
            gate.flush_final(stored_reply)

            await self._context_mgr.add_turn(conversation_id, "assistant", history_reply)
            await self._announce_thinking_refused()

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
        return (
            name not in MEMORY_TOOL_NAMES
            and name not in SCHEDULER_TOOL_NAMES
            and name not in (VOICE_ENROLL_TOOL, CALENDAR_TOOL, TELL_TOOL, "move_music")
        )

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
        room_colour: tuple[str, str] | None = None
        valid_tool_names = {
            s.get("function", {}).get("name")
            for s in tool_schemas
            if s.get("function", {}).get("name")
        }

        action_command = _is_action_command(user_message)
        for iteration in range(self._max_tool_iterations):
            # Only the first reply to an action command is capped: later
            # ones answer from tool results and may need to say more.
            # Reasoning counts against the same limit, so with it on the cap ran
            # out before the tool call: "Turn on the light." failed 4 of 4.
            cap = (
                {"max_tokens": ACTION_FIRST_REPLY_MAX_TOKENS}
                if action_command and iteration == 0 and not self._reasoning_active()
                else {}
            )
            if gate is not None and gate.active:
                # Drop any unemitted tail from the previous iteration —
                # prose preceding a tool call is narration, never speech.
                gate.new_response()
                response = await self._provider.async_chat_stream(
                    messages, tool_schemas, on_delta=gate.feed, **cap
                )
            else:
                response = await self._provider.async_chat(messages, tool_schemas, **cap)
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
                colour_scope = _room_colour_scope(tc.name, tc.arguments)
                if colour_scope is not None and room_colour is not None and (
                    colour_scope[0] == room_colour[0] and colour_scope[1] != room_colour[1]
                ):
                    _LOGGER.info(
                        "AI Plugin: second colour %r for the same room in one turn — not run",
                        colour_scope[1],
                    )
                    msg = tc.to_tool_result_message(ROOM_COLOUR_REPLACED_RESULT)
                    messages.append(msg)
                    history_messages.append(msg)
                    continue
                if colour_scope is not None and room_colour is None:
                    room_colour = colour_scope
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

    def _reasoning_active(self) -> bool:
        """The option is on and the model has not refused it."""
        return bool(
            self._entry.options.get(CONF_ENABLE_THINKING, DEFAULT_ENABLE_THINKING)
        ) and getattr(self._provider, "thinking_refused", False) is not True

    async def _announce_thinking_refused(self) -> None:
        """Say once, where it will be seen, that the option has no effect.

        Settings refuse it for such a model, but a model chosen later or an
        option saved earlier still gets here. The turn is answered regardless.
        """
        if (
            getattr(self._provider, "thinking_refused", False) is not True
            or self._thinking_announced
            or self._hass is None
        ):
            return
        self._thinking_announced = True
        await self._hass.services.async_call(
            "persistent_notification",
            "create",
            {
                "title": "AI Plugin",
                "message": (
                    f"The model {self._entry.options.get(CONF_MODEL, '')} cannot "
                    "reason, so 'Enable model reasoning' has no effect. Replies "
                    "work as usual. Turn the option off or choose a model that can."
                ),
                "notification_id": "ai_plugin_no_reasoning",
            },
            blocking=False,
        )

    def _confine_to_this_room(
        self, name: str, arguments: dict, user_message: str, device_id: str | None
    ) -> tuple[dict, str | None]:
        """Hold "in here" to the satellite's room, whatever the model sent.

        "Switch every lamp in here to a different color" came back in replay
        with four rooms as the area (2026-11-03). A plain "Turn off the lights."
        means this room too: K2 Horizon sent area 'all' 25 of 49 times, and even
        after a success went on to lamps in other rooms (5 of 16). Returns the
        arguments to use, and a refusal when the named device is in another room.
        """
        if name not in _ACTUATOR_TOOL_NAMES or not (
            says_this_room(user_message) or is_generic_lights_command(user_message)
        ):
            return arguments, None
        room = self._this_room(device_id, user_message)
        if room is None:
            return arguments, None
        device = _s(arguments.get("name")).strip()
        if device:
            if self._named_device_elsewhere(device, room):
                return arguments, (
                    f"[Not changed: {device} is not in this room. Tell the user, "
                    "and name a device that is here.]"
                )
            return arguments, None
        if "area" in arguments or name in ("set_brightness", "set_area_state"):
            return {**arguments, "area": room}, None
        return arguments, None

    def _this_room(self, device_id: str | None, user_message: str) -> str | None:
        """The satellite's room, unless the message names a different one, as in
        "turn off the kitchen light from here"."""
        if self._hass is None or not device_id:
            return None
        area_id = _caller_area_id(self._hass, device_id)
        if not area_id:
            return None
        area = ar.async_get(self._hass).async_get_area(area_id)
        if area is None:
            return None
        named = self._ha_local._area_named_in_message(user_message) if self._ha_local else None
        if named and named.lower() != area.name.lower():
            return None
        return area.name

    def _named_device_elsewhere(self, device: str, room: str) -> bool:
        """True when the name matches devices and none of them is in this room.
        A name that matches nothing is left to the tool, which says so."""
        ent_reg = er.async_get(self._hass)
        dev_reg = dr.async_get(self._hass)
        area_reg = ar.async_get(self._hass)
        needle = device.lower()
        found = False
        for eid, entry in ent_reg.entities.items():
            state = self._hass.states.get(eid)
            friendly = _s(state.attributes.get("friendly_name") if state else "").lower()
            if needle != eid.lower() and needle not in friendly:
                continue
            found = True
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                area_id = dev.area_id if dev else None
            area = area_reg.async_get_area(area_id) if area_id else None
            if area is not None and area.name.lower() == room.lower():
                return False
        return found

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
        arguments, refused = self._confine_to_this_room(
            name, arguments, user_message, device_id
        )
        if refused:
            return refused
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
        if name == RING_PHONE_TOOL and self._can_ring_phones():
            return await self._ring_phone_tool(arguments, user_id, device_id, language)
        # A command that has come due is carried out now; putting it off again
        # could go on forever. The tool list stays the same: a different one
        # would make the next voice turn read the whole prompt in again (7-8 s).
        if _IN_SCHEDULED_RUN.get() and (
            name in SCHEDULER_WRITE_TOOL_NAMES or name in _TIMER_WRITE_TOOL_NAMES
        ):
            return "[This is due now. Carry out the command now with the device tools.]"
        if name in SCHEDULER_TOOL_NAMES and getattr(self, "_scheduler", None) is not None:
            return await async_call_scheduler_tool(
                self._scheduler,
                self._hass,
                name,
                arguments,
                device_id=device_id,
                user_id=user_id,
                person_id=self._speaking_person(device_id, user_id),
                language=language,
                user_message=user_message,
                resolve_person=self._resolve_person,
            )
        if name == TELL_TOOL and getattr(self, "_scheduler", None) is not None:
            speaker = self._speaking_person(device_id, user_id)
            state = self._hass.states.get(speaker) if speaker else None
            return await async_tell(
                self._hass,
                arguments,
                language=language,
                resolve_person=self._resolve_person,
                sender_name=state.attributes.get("friendly_name") if state else None,
            )
        if name == "move_music" and self._ha_local is not None:
            return await self._ha_local.move_music(arguments.get("area") or "", device_id)
        if name == CALENDAR_TOOL and getattr(self, "_scheduler", None) is not None:
            return await async_add_event(
                self._hass, arguments, self._scheduler.now(),
                user_message=user_message, language=language,
            )
        if name in MEMORY_TOOL_NAMES and self._memory is not None:
            if name in MEMORY_WRITE_TOOLS and self._speaker_unrecognised(device_id):
                _LOGGER.debug("AI Plugin: %s refused: voice not recognised", name)
                # Named for what it is: "nothing was saved" on a forget came out
                # as "I couldn't find that in my memory", with the fact listed
                # (2026-11-03), and the user could not know to try again.
                if name == "forget":
                    return (
                        "[Not removed: the speaker's voice was not recognised, so "
                        "their facts cannot be changed. Tell the user you did not "
                        "recognise their voice, so nothing was removed, and to try "
                        "again.]"
                    )
                return (
                    "[Not saved: the speaker's voice was not recognised, so nothing "
                    "can be stored for them. Tell the user you did not recognise "
                    "their voice, so nothing was saved, and to try again.]"
                )
            if (
                name == "remember"
                and user_message
                and not asks_to_remember(
                    user_message, (language or "en").split("-")[0].lower()
                )
            ):
                return (
                    "[Not saved: the user did not ask you to remember this. Do not "
                    "say it was saved or noted. Reply to what they said.]"
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
            if name == "HassLightSet":
                arguments = hass_light_set_args(arguments)
            return await self._mcp.call_tool(name, name_without_room(arguments))
        return f"[Tool {name!r} unavailable — no handler configured]"


    def _resolve_person(self, name: str) -> str | None:
        """The person entity meant by a spoken name, also "Anna's"."""
        return resolve_owner(
            self._hass, name, speaker_person=None, find_person=self._find_person
        ) if name else None

    def _speaking_person(self, device_id: str | None, user_id: str | None) -> str | None:
        """The person entity of who is speaking: by voice, else by HA login."""
        speaker = (self._turn_speakers or {}).get(device_id) if device_id else None
        return speaker or person_for_user(self._hass, user_id)

    async def _run_scheduled_command(self, job: Job) -> str:
        """Carry out a command that has come due, as if said right now."""
        token = _IN_SCHEDULED_RUN.set(True)
        try:
            return await self.async_process(
                job.text,
                f"scheduled-{job.id}-{int(time.time())}",
                job.language,
                device_id=job.device_id,
                user_id=job.user_id,
            )
        finally:
            _IN_SCHEDULED_RUN.reset(token)

    def _can_ring_phones(self) -> bool:
        return getattr(self, "_hass", None) is not None

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
