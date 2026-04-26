"""Orchestrator: the core message-processing engine for AI Plugin.

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
    CONF_LOCATION_BIAS,
    CONF_LOCATION_ENTITY,
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
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_LOCATION_BIAS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_RESPONSE_TIMEOUT,
    DEFAULT_SUMMARIZATION_ENABLED,
    DOMAIN,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
)
from .context_manager import ContextManager
from .exceptions import OrchestratorError
from .providers.openai_compat import OpenAICompatProvider
from .shortcuts import try_shortcut
from .tools.ha_local import HALocalToolRegistry
from .tools.memory import TOOL_NAMES as MEMORY_TOOL_NAMES, TOOL_SCHEMAS as MEMORY_TOOL_SCHEMAS, MemoryTool
from .tools.web_search import TOOL_SCHEMA as WEB_SEARCH_SCHEMA, WebSearchTool
from .tools._geocode import GeocodeResult, reverse_geocode

if TYPE_CHECKING:
    from .providers.base import AbstractProvider
    from .tools.mcp_client import MCPToolRegistry

_LOGGER = logging.getLogger(__name__)

# Strip narration sentences like "Calling list_entities(...)..." that small
# models emit even when the system prompt forbids it. Matches a full line
# or a leading sentence up to the next newline / sentence break.
_NARRATION_PATTERNS = [
    r"^\s*Calling\s+\w+\([^\n]*?\)\s*\.{0,6}\s*$",
    r"^\s*(?:Let me|I will|I'll|I am going to|Let's)\s+(?:call|check|see|look|run|use|try|invoke)[^\n]*$",
    r"^\s*(?:Checking|Searching|Looking up|Querying)[^\n]*\.{0,6}\s*$",
    r"^\s*Found\s+(?:a\s+|the\s+|one\s+)?\w+\s+entit(?:y|ies)[^\n]*$",
    r"^\s*No\s+\w+\s+entit(?:y|ies)\s+found[^\n]*$",
]
_NARRATION_RE = re.compile("|".join(_NARRATION_PATTERNS), re.MULTILINE | re.IGNORECASE)


# Trigger phrases for questions that MUST be grounded in list_entities.
# "any X on", "which X are off", "welche X sind an", etc. — used by the
# grounding verifier to detect when the model answered from memory instead
# of calling list_entities. Keep conservative to avoid false positives
# (e.g. on plain greetings or commands).
_STATE_SET_QUERY_RE = re.compile(
    r"\b(?:any|are\s+any|is\s+any|is\s+anything|which|what(?:'s|\s+is)?|"
    r"what's|welche|sind)\b.*?\b"
    r"(?:on|off|open|closed|playing|active|running|an|aus|offen|auf)\b",
    re.IGNORECASE,
)


def _is_state_set_query(text: str) -> bool:
    return bool(_STATE_SET_QUERY_RE.search(text or ""))


def _any_list_entities_call(tool_msgs: list[dict]) -> bool:
    """True if the tool-loop made at least one list_entities call."""
    for m in tool_msgs:
        for tc in m.get("tool_calls") or []:
            if tc.get("function", {}).get("name") == "list_entities":
                return True
    return False


def _any_tool_call(tool_msgs: list[dict], tool_name: str) -> bool:
    """True if the tool-loop invoked a given tool at least once."""
    for m in tool_msgs:
        for tc in m.get("tool_calls") or []:
            if tc.get("function", {}).get("name") == tool_name:
                return True
    return False


# Phrases get_entity / search_entities return when an entity can't be
# resolved. When the model sees these in a tool result it is supposed
# to broaden the search (search_entities) or list_entities; small models
# sometimes accept the miss and tell the user "can't find X" instead.
_MISS_RESULT_PATTERNS = (
    "no entity matches",
    "no matches for",
    "no results",
    "try search_entities",
)


def _get_entity_missed(tool_msgs: list[dict]) -> str | None:
    """Return the original get_entity argument if a call returned a miss.

    Scans tool result messages; if a get_entity call returned a
    no-match string, return the name/entity_id that was asked about so
    the retry can pass it straight to search_entities.
    """
    import json as _json
    last_get_args: dict | None = None
    for m in tool_msgs:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                if fn.get("name") == "get_entity":
                    raw = fn.get("arguments", "{}")
                    try:
                        last_get_args = (
                            _json.loads(raw) if isinstance(raw, str) else (raw or {})
                        )
                    except Exception:  # noqa: BLE001
                        last_get_args = {}
        elif m.get("role") == "tool":
            content = (m.get("content") or "").lower()
            if any(p in content for p in _MISS_RESULT_PATTERNS):
                if last_get_args:
                    return (
                        last_get_args.get("name")
                        or last_get_args.get("entity_id")
                        or last_get_args.get("entity")
                        or ""
                    )
                return ""
    return None


# Detect when a model emits a tool call as plain text ("content") instead of
# using the API's structured tool_calls field. Matches first call in text.
# Accepts optional "CALL " prefix and optional trailing period.
_RAW_TOOL_CALL_RE = re.compile(
    r"(?:^|\n)\s*(?:CALL\s+)?([a-zA-Z_][\w]*)\s*\(([^()\n]*)\)",
    re.IGNORECASE,
)


def _parse_raw_tool_call(text: str, valid_names: set[str]) -> tuple[str, dict] | None:
    """Parse 'tool_name(kwargs)' pseudo-syntax out of a model reply.

    Returns (name, args_dict) on success, None otherwise. Handles:
    - list_entities(domain="light", state="on")
    - CALL web_search('price of bitcoin')  (positional → first arg name)
    - list_entities()
    Only matches if the parsed tool name is in valid_names.
    """
    if not text:
        return None
    m = _RAW_TOOL_CALL_RE.search(text)
    if not m:
        return None
    name = m.group(1)
    if name not in valid_names:
        return None
    arg_str = m.group(2).strip()
    if not arg_str:
        return (name, {})
    args: dict = {}
    # kwargs: key="value" | key='value' | key=bareword
    kw_pairs = re.findall(
        r"(\w+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,\s][^,]*?)\s*(?:,|$)",
        arg_str,
    )
    if kw_pairs:
        for k, v in kw_pairs:
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (
                v.startswith("'") and v.endswith("'")
            ):
                v = v[1:-1]
            args[k] = v
        return (name, args)
    # Positional single value: map to "query" for web_search, else "name".
    val = arg_str
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        val = val[1:-1]
    if name == "web_search":
        return (name, {"query": val})
    return (name, {"name": val})


def _strip_narration(text: str) -> str:
    """Remove tool-call narration lines from a model reply.

    Small models (qwen2.5:7B and similar) sometimes ignore the
    "don't narrate" prompt rule and prefix their real answer with
    progress chatter. Strip those lines before handing the reply to
    the user / TTS.
    """
    cleaned = _NARRATION_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# Strip emoji and pictographs. TTS either mispronounces or reads them
# ("smiling face with sunglasses"), and chatty models like qwen3 love
# sprinkling them into greetings. Unicode ranges cover emoji blocks,
# misc symbols, dingbats, transport, and common decorative chars.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # symbols & pictographs, emoticons, transport, extended-A/B
    "\U00002600-\U000027BF"   # misc symbols, dingbats
    "\U0001F000-\U0001F2FF"   # mahjong, domino, playing cards, enclosed alphanumerics
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "]",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    """Remove emoji/pictographs. Safe for TTS and voice output."""
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    # Collapse residual double-spaces introduced by stripped emoji.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()

# Domain/device words that suggest the user is asking about a specific entity.
_ENTITY_WORDS = frozenset(
    "light lights switch switches lamp lamps bulb bulbs sensor sensors "
    "thermostat thermostats blinds cover covers lock locks fan fans "
    "vacuum vacuums plug plugs media speaker tv television climate".split()
)
# Words that suggest an area-level question.
_AREA_WORDS = frozenset("room rooms area areas house zones zone".split())
# Words that suggest a domain sweep ("list all lights", "show switches").
_SWEEP_WORDS = frozenset("list show which all every what".split())
# Pronouns that reference a prior action — keep all schemas so model is flexible.
_PRONOUN_RE = re.compile(r"\b(it|that|them|those|there|here)\b", re.IGNORECASE)


def _prune_ha_local_schemas(
    user_message: str, schemas: list[dict]
) -> list[dict]:
    """Drop ha_local tool schemas that are unlikely to be useful for this turn.

    Heuristic-only — cheap, never perfect. When in doubt, keep the schema.
    Minimum 2 schemas always returned so the model always has search + get.
    """
    if len(schemas) <= 2:
        return schemas

    text = user_message.lower()
    words = set(re.findall(r"[a-z_]+", text))

    # Pronouns → keep everything; prior-turn entity is in [LAST ACTION].
    if _PRONOUN_RE.search(text):
        return schemas

    # State-set questions ("any lights on?", "are any lights turned on?")
    # must retain list_entities — that's the only tool that can answer them.
    # Without this check the entity-word heuristic below drops list_entities
    # for short-form queries whose trigger word ("any", "are") isn't in the
    # sweep set, leaving the model tool-less and producing empty replies.
    if _is_state_set_query(user_message):
        return schemas

    has_entity_word = bool(words & _ENTITY_WORDS)
    # entity_id pattern like "light.something"
    has_entity_id = bool(re.search(r"[a-z_]+\.[a-z0-9_]+", text))
    has_area_word = bool(words & _AREA_WORDS)
    has_sweep = bool(words & _SWEEP_WORDS)

    names_to_drop: set[str] = set()

    # If message is clearly about a specific entity (no area question, no sweep):
    if (has_entity_word or has_entity_id) and not has_area_word and not has_sweep:
        names_to_drop.add("list_areas")
        names_to_drop.add("list_entities")

    if not names_to_drop:
        return schemas

    pruned = [s for s in schemas if s.get("function", {}).get("name") not in names_to_drop]
    # Safety: always keep at least 2 ha_local schemas (search_entities + get_entity).
    if len(pruned) < 2:
        return schemas

    if names_to_drop:
        _LOGGER.debug(
            "AI Plugin: pruned ha_local schemas %s for message %r",
            names_to_drop,
            user_message[:60],
        )
    return pruned


class LocationProvider:
    """Resolve the user's home location across heterogeneous installs.

    Source priority: configured live entity > hass.config home coords >
    nothing. Reverse-geocoded once per coordinate pair via OSM Nominatim
    (cached on disk so we never repeat lookups across HA restarts).

    Designed to fail open: missing internet, missing country, missing
    coords all degrade to the next available signal. When *no* signal
    resolves, callers receive an empty payload — the plugin must then
    skip location injection entirely rather than guess a city.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._hass = hass
        self._entry = entry
        self._cached: dict | None = None
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._entry.options.get(CONF_LOCATION_BIAS, DEFAULT_LOCATION_BIAS))

    def _read_entity_coords(self) -> tuple[float | None, float | None]:
        """Return live coords from the configured location entity, if any."""
        entity_id = self._entry.options.get(CONF_LOCATION_ENTITY)
        if not entity_id or self._hass is None:
            return (None, None)
        state = self._hass.states.get(entity_id)
        if state is None:
            return (None, None)
        attrs = state.attributes or {}
        lat = attrs.get("latitude")
        lon = attrs.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return (float(lat), float(lon))
        return (None, None)

    def _read_config_coords(self) -> tuple[float | None, float | None]:
        if self._hass is None:
            return (None, None)
        cfg = self._hass.config
        lat = getattr(cfg, "latitude", None)
        lon = getattr(cfg, "longitude", None)
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return (float(lat), float(lon))
        return (None, None)

    async def async_resolve(self) -> dict:
        """Return a location dict — possibly empty, never None.

        Result keys (any may be absent):
          * ``city``, ``region``, ``country_name``, ``country_iso``
          * ``lat``, ``lon`` (always pair, or both absent)
          * ``timezone``
          * ``language`` (BCP-47 from hass.config)
          * ``source``: "entity" or "config" — None when nothing resolved.

        Cached after first call; cache is invalidated on entry reload
        because a new Orchestrator (and Provider) is built then.
        """
        if not self.enabled:
            return {}
        if self._cached is not None:
            return self._cached

        async with self._lock:
            if self._cached is not None:
                return self._cached

            payload: dict = {}
            cfg = self._hass.config if self._hass is not None else None

            # Coordinates: entity first, then hass.config.
            lat, lon = self._read_entity_coords()
            source = "entity" if lat is not None and lon is not None else None
            if lat is None or lon is None:
                lat, lon = self._read_config_coords()
                if lat is not None and lon is not None:
                    source = "config"

            if lat is not None and lon is not None:
                payload["lat"] = lat
                payload["lon"] = lon

            # Country / timezone / language from hass.config — always
            # available even when geocoding fails.
            if cfg is not None:
                country = str(getattr(cfg, "country", "") or "").strip()
                if country:
                    payload["country_iso"] = country.upper()
                tz = str(getattr(cfg, "time_zone", "") or "").strip()
                if tz:
                    payload["timezone"] = tz
                lang = str(getattr(cfg, "language", "") or "").strip()
                if lang:
                    payload["language"] = lang

            # Reverse geocode if we have coords. Failure is silent.
            if lat is not None and lon is not None and cfg is not None:
                try:
                    geo: GeocodeResult | None = await reverse_geocode(
                        lat=lat,
                        lon=lon,
                        language=payload.get("language"),
                        config_dir=cfg.config_dir,
                    )
                except Exception:  # noqa: BLE001
                    _LOGGER.debug(
                        "AI Plugin: reverse_geocode raised unexpectedly",
                        exc_info=True,
                    )
                    geo = None
                if geo is not None:
                    if geo.city:
                        payload["city"] = geo.city
                    if geo.region:
                        payload["region"] = geo.region
                    if geo.country_name:
                        payload["country_name"] = geo.country_name
                    if geo.country_iso and "country_iso" not in payload:
                        payload["country_iso"] = geo.country_iso

            if source is not None:
                payload["source"] = source

            self._cached = payload
            return payload


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
        self._max_tool_iterations: int = opts.get(
            CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS
        )
        # Web search tool (None when disabled in config).
        self._web_search: WebSearchTool | None = (
            WebSearchTool(opts) if opts.get(CONF_WEB_SEARCH_ENABLED, False) else None
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

    async def _build_system_prompt(
        self, voice_mode: bool, user_id: str | None = None
    ) -> str:
        """Return the system prompt for this request.

        Base prompt (voice-compact or default) is always sent so the plugin's
        entity-discovery, grounding, and speech rules apply. User's custom
        instructions are appended on top — persona, location, etc. A
        [HOME LOCATION] block is always injected from hass.config so the
        model can enrich web_search queries with the user's real location.
        A [USER FACTS] block is auto-injected from the memory file so small
        LLMs that ignore the recall tool still see stored facts.
        """
        base = SYSTEM_PROMPT_VOICE if voice_mode else SYSTEM_PROMPT_DEFAULT
        location_block = await self._build_location_block()
        facts_block = await self._build_user_facts_block(user_id)
        custom = self._entry.options.get(CONF_SYSTEM_PROMPT, "").strip()
        parts = [base, location_block, facts_block]
        if custom:
            parts.append(custom)
        return "\n\n".join(p for p in parts if p)

    async def _build_user_facts_block(self, user_id: str | None) -> str:
        """Render stored user facts as a numbered prompt block.

        Auto-injected on every turn so small LLMs that fail to call the
        recall tool can still answer questions about previously-saved
        facts (name, preferences, etc.). The numbered list also doubles
        as the index reference for the two-step forget(index=N) flow.
        File I/O runs in the executor to keep the event loop unblocked.
        """
        if self._memory is None:
            return ""
        try:
            from .tools.memory import _memory_path  # noqa: PLC0415
            path = _memory_path(self._memory._config_dir, user_id)
            facts = await asyncio.get_running_loop().run_in_executor(
                None, self._memory._load, path
            )
        except Exception:  # noqa: BLE001
            _LOGGER.debug("AI Plugin: could not read user facts", exc_info=True)
            return ""
        if not facts:
            return ""
        numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
        return (
            "[USER FACTS]\n"
            "Facts previously saved by the user. Use them when answering "
            "questions about who they are or their preferences. To remove "
            "a fact, call forget(index=N, fact='<keyword>') with the matching "
            "number. Never call forget for a fact that is not listed here.\n"
            f"{numbered}"
        )

    async def _build_location_block(self) -> str:
        """Render the resolved home location as a prompt block.

        The block is informational only — it does NOT instruct the model
        to inject the city into queries; that is handled deterministically
        in ``web_search.async_search`` based on the ``near_user`` flag and
        a locality regex. The block tells the model where the user lives
        so it can answer "where am I?" accurately.

        Renders only fields that resolved. When nothing resolves (no
        coords, no country, no entity, network offline) the block is
        omitted entirely — better silence than fabricated geography.
        """
        try:
            data = await self._location.async_resolve()
        except Exception:  # noqa: BLE001
            _LOGGER.debug("AI Plugin: location resolve failed", exc_info=True)
            return ""

        if not data:
            return ""

        parts: list[str] = []
        if data.get("city"):
            parts.append(f"city={data['city']}")
        if data.get("region"):
            parts.append(f"region={data['region']}")
        if data.get("country_name"):
            parts.append(f"country={data['country_name']}")
        elif data.get("country_iso"):
            parts.append(f"country={data['country_iso']}")
        if "lat" in data and "lon" in data and "city" not in data:
            parts.append(f"coords={data['lat']:.4f},{data['lon']:.4f}")
        if data.get("timezone"):
            parts.append(f"timezone={data['timezone']}")

        if not parts:
            return ""

        joined = ", ".join(parts)
        return (
            "[HOME LOCATION]\n"
            f"- {joined}\n"
            "- For questions about the user's immediate area, set "
            "near_user=true on web_search. The plugin will scope the "
            "query to this area; do not invent a city name."
        )

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
        base_prompt = await self._build_system_prompt(voice_mode, user_id)

        # v0.5.15: no more [HOME CONTEXT] YAML dump. Small LLMs drown in it and
        # hallucinate their way through inventory questions. The model instead
        # calls discovery tools (list_areas / list_entities / get_entity /
        # search_entities) on demand.
        sections: list[str] = [base_prompt]
        last = self._last_entities.get(conversation_id)
        if last:
            sections.append(f"[LAST ACTION]\n{last}")
        system_prompt = "\n\n".join(sections)

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
                shortcut_reply = try_shortcut(self._hass, message)
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

            # 2. Summarize old turns if we're approaching the soft token limit.
            if self._summarization_enabled:
                await self._context_mgr.summarize_if_needed(
                    conversation_id, system_prompt, self._provider
                )

            # 3. Build the message list (system prompt + trimmed history).
            messages = await self._context_mgr.get_messages(conversation_id, system_prompt)

            # 4. Collect tool schemas: ha_local + memory + web_search + MCP tools.
            tool_schemas = self._mcp.get_tool_schemas() if self._mcp else []
            if self._web_search is not None:
                tool_schemas = [WEB_SEARCH_SCHEMA, *tool_schemas]
            if self._memory is not None:
                tool_schemas = [*MEMORY_TOOL_SCHEMAS, *tool_schemas]
            if self._ha_local is not None:
                ha_schemas = _prune_ha_local_schemas(message, self._ha_local.get_schemas())
                tool_schemas = [*ha_schemas, *tool_schemas]

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
                schema_tokens = self._context_mgr.estimate_tokens(str(tool_schemas))
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
                    reply, tool_msgs = await self._tool_loop(messages, tool_schemas, user_id, voice_mode, message)
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
            if (
                tool_schemas
                and _is_state_set_query(message)
                and not _any_list_entities_call(tool_msgs)
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
                        messages, tool_schemas, user_id, voice_mode, message
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
                and tool_schemas
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
                        messages, tool_schemas, user_id, voice_mode, message
                    )
                    if _any_tool_call(tool_msgs3, "search_entities"):
                        reply = reply3
                        tool_msgs = tool_msgs + tool_msgs3
                except Exception:  # noqa: BLE001
                    _LOGGER.exception(
                        "AI Plugin: fuzzy-resolve retry failed — keeping original reply"
                    )

            # 6. Append tool call/result messages then the assistant reply to history.
            for msg in tool_msgs:
                await self._context_mgr.add_raw_message(conversation_id, msg)
            stored_reply = _strip_narration(reply) or reply
            stored_reply = _strip_emoji(stored_reply) or stored_reply

            # Defensive: small models occasionally return empty content with no
            # tool calls (prompt/tool confusion). Surface a user-facing fallback
            # so TTS doesn't play silence.
            if not stored_reply.strip():
                _LOGGER.warning(
                    "AI Plugin: empty reply from model (conv=%s, tools_called=%d) — substituting fallback",
                    conversation_id, len(tool_msgs),
                )
                reply = "I couldn't produce an answer for that. Try rephrasing."
                stored_reply = reply

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

        return stored_reply

    async def _tool_loop(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        user_id: str | None = None,
        voice_mode: bool = False,
        user_message: str = "",
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
            response = await self._provider.async_chat(messages, tool_schemas)
            last_response = response

            if not response.has_tool_calls:
                reply_text = response.reply_text()
                # Raw-tool-syntax catcher: small models sometimes write
                # `tool_name(args)` into content instead of emitting a
                # structured tool_call. Recover by parsing and executing.
                recovered = _parse_raw_tool_call(reply_text, valid_tool_names)
                if recovered is not None:
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
                        rec_name, rec_args, user_id, voice_mode, user_message
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
                result = await self._dispatch_tool(tc.name, tc.arguments, user_id, voice_mode, user_message)
                msg = tc.to_tool_result_message(result)
                messages.append(msg)
                history_messages.append(msg)

        suffix = " [Note: reached tool call limit — response may be incomplete.]"
        return (last_response.reply_text() if last_response else "") + suffix, history_messages

    async def _dispatch_tool(
        self,
        name: str,
        arguments: dict,
        user_id: str | None = None,
        voice_mode: bool = False,
        user_message: str = "",
    ) -> str:
        """Route a tool call to ha_local, memory, web search, or MCP. Never raises."""
        if self._ha_local is not None and name in self._ha_local.tool_names:
            return await self._ha_local.call_tool(name, arguments)
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
