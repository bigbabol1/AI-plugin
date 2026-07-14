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
from datetime import date as _date
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
    CONF_ENABLE_THINKING,
    CONF_KEEP_ALIVE,
    CONF_LOCATION_BIAS,
    CONF_LOCATION_ENTITY,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_MODEL,
    CONF_PRUNE_TOOL_SCHEMAS,
    CONF_RESPONSE_TIMEOUT,
    CONF_SUMMARIZATION_ENABLED,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VOICE_MODE,
    CONF_WEB_SEARCH_ENABLED,
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_ENABLE_THINKING,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_LOCATION_BIAS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_PRUNE_TOOL_SCHEMAS,
    DEFAULT_RESPONSE_TIMEOUT,
    DEFAULT_SUMMARIZATION_ENABLED,
    DOMAIN,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
)
from .context_manager import ContextManager
from .exceptions import OrchestratorError
from .i18n import L
from .providers.openai_compat import OpenAICompatProvider
from .shortcuts import async_try_action_shortcut, async_try_media_shortcut, try_shortcut
from .tools.ha_local import HALocalToolRegistry
from .tools.memory import TOOL_NAMES as MEMORY_TOOL_NAMES, TOOL_SCHEMAS as MEMORY_TOOL_SCHEMAS, MemoryTool
from .tools.web_search import TOOL_SCHEMA as WEB_SEARCH_SCHEMA, WebSearchTool
from .tools.browse_url import TOOL_SCHEMA as BROWSE_URL_SCHEMA, BrowseUrlTool
from .tools._geocode import GeocodeResult, reverse_geocode

if TYPE_CHECKING:
    from .providers.base import AbstractProvider
    from .tools.mcp_client import MCPToolRegistry

_LOGGER = logging.getLogger(__name__)

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


# Action-command grounding verifier.
# Short STT inputs like "Lights on." / "Bedroom off." make weak local
# models narrate ("I'll turn the lights on.") instead of calling an
# actuator tool. _strip_narration then produces an empty reply and the
# user gets the "couldn't produce" fallback while the device stays off.
# Detect imperative-action shape: noun+state, or turn/switch/dim verb.
_ACTION_COMMAND_RE = re.compile(
    r"^\s*(?:"
    # English: "Lights on", "Bedroom off", "X on/off/dim", verbs
    r"(?:[\w\s]{2,40}\s+(?:on|off|aus|an)|"
    r"(?:turn|switch|toggle|put|set|dim|brighten|fade)\s+\S+)"
    # German: "Licht an/aus", "Schalte X ein/aus"
    r"|(?:licht|lichter|lampe|lampen|fan|ventilator|stehlampe|"
    r"flurlicht|mood\s*light)\s+(?:an|aus|ein|on|off)"
    r"|(?:schalte|mach)\s+\S+\s+(?:ein|aus|an)"
    r")\b.*$",
    re.IGNORECASE,
)
# Words that signal an information question, not an action — exclude.
_QUESTION_INTENT_RE = re.compile(
    r"\b(?:any|are|is|which|what|where|when|why|how|"
    r"welche|sind|ist|wann|wo|warum|wie)\b",
    re.IGNORECASE,
)
# Tool names whose invocation = action taken.
_ACTUATOR_TOOL_NAMES = (
    "set_area_state", "HassTurnOn", "HassTurnOff",
    "HassLightSet", "HassClimateSetTemperature",
    "HassClimateSetMode", "HassMediaPause", "HassMediaUnpause",
    "HassMediaNext", "HassMediaPrevious",
    "play_music", "media_command",
)


def _is_action_command(text: str) -> bool:
    """True when the user message is a short imperative action.

    Conservative: must be under 80 chars (commands tend to be terse),
    must match _ACTION_COMMAND_RE, must NOT be a question.
    """
    if not text:
        return False
    text = text.strip()
    if len(text) > 80:
        return False
    if _QUESTION_INTENT_RE.search(text):
        return False
    return bool(_ACTION_COMMAND_RE.search(text))


def _any_actuator_call(tool_msgs: list[dict]) -> bool:
    """True if any actuator tool was invoked in this turn."""
    return any(_any_tool_call(tool_msgs, t) for t in _ACTUATOR_TOOL_NAMES)


# Online-query grounding verifier.
#
# Rather than enumerating every possible trigger phrase (a losing battle),
# detect two broad signal classes:
#
# 1. EXPLICIT TEMPORAL ANCHOR — the message pins itself to a time window
#    the model cannot know: "this weekend", "last week", "right now", etc.
#    Any relative or recent-absolute time reference is a safe trigger.
#
# 2. REAL-WORLD PLACE QUERY — "in <City>" / "in <Country>" combined with
#    a question word or event verb. The model's training snapshot of any
#    given city's current events is always stale.
#
# Both classes are still excluded when the query is clearly about an HA
# sensor (temperature, humidity, thermostat) to avoid false positives.

_TEMPORAL_ANCHOR_RE = re.compile(
    r"\b("
    r"today|tonight|yesterday|last\s+night|this\s+morning|this\s+evening|"
    r"this\s+week(?:end)?|last\s+week(?:end)?|this\s+month|last\s+month|"
    r"this\s+year|last\s+year|right\s+now|at\s+the\s+moment|currently|"
    r"just\s+(?:now|happened|announced)|recently|"
    # German
    r"heute|gestern|heute\s+(?:nacht|abend|morgen)|"
    r"diese[sn]?\s+wochenende?|letzte[sn]?\s+wochenende?|"
    r"diese\s+woche|letzte\s+woche|diesen\s+monat|letzten\s+monat|"
    r"gerade\s+jetzt|im\s+moment|zur(?:zeit|zeit)|gerade\s+eben"
    r")\b",
    re.IGNORECASE,
)

# "in Berlin", "in Germany", "in New York" — proper noun after preposition
_PLACE_QUERY_RE = re.compile(
    r"\b(?:in|at|near|around|from|about)\s+[A-ZÄÖÜ][a-zäöüß]"
    r"(?:[a-zäöüß\-]*\s+[A-ZÄÖÜ][a-zäöüß][a-zäöüß]*)?",
)

_HA_SENSOR_RE = re.compile(
    r"\b(?:temperature|humidity|sensor|thermostat|climate|heizung|"
    r"temperatur|luftfeuchtigkeit|co2|pm2|lux|pressure|druck)\b",
    re.IGNORECASE,
)

# Verbs / question words that, combined with a place, signal a real-world query
_EVENT_VERB_RE = re.compile(
    r"\b(?:happen(?:ed|ing)?|going\s+on|event|news|what|who|when|where|"
    r"passier|geschah|los\s+(?:in|war)|was\s+(?:ist|war))\b",
    re.IGNORECASE,
)


# Meta-conversation patterns: user is talking ABOUT the previous reply,
# not asking a fresh question of the world. These should never trigger a
# web search even if a temporal token like "just now" appears in them.
_META_CONVO_RE = re.compile(
    r"\b(?:"
    # English
    r"you\s+(?:said|told|mentioned|wrote|just\s+(?:said|told|wrote))|"
    r"(?:tell|say|repeat|show)\s+(?:me\s+)?(?:that|it|this)\s+again|"
    r"what\s+did\s+you\s+(?:say|tell|mean)|"
    r"repeat\s+(?:that|it|this|yourself)|"
    r"already\s+(?:said|told|mentioned)|"
    # German
    r"du\s+(?:hast|sagtest)\s+(?:gerade|eben|vorhin)|"
    r"sagst?\s+(?:das|es)\s+(?:nochmal|noch\s+einmal)|"
    r"wiederhol(?:e|st)|"
    r"hast\s+du\s+gerade"
    r")\b",
    re.IGNORECASE,
)


def _is_online_query(text: str) -> bool:
    """Return True when the message almost certainly needs live web data.

    Triggers on:
    - Any explicit temporal anchor (this weekend, yesterday, right now…)
    - A named place combined with an event/question verb (what happened in Berlin)

    Excluded when:
    - Query is clearly about a local HA sensor.
    - Query is meta-conversation about the previous reply ("you said that
      just now"). The "just now" inside _TEMPORAL_ANCHOR_RE would otherwise
      cause a useless web_search on chat-about-chat messages.
    """
    if not text:
        return False
    if _HA_SENSOR_RE.search(text):
        return False
    if _META_CONVO_RE.search(text):
        return False
    if _TEMPORAL_ANCHOR_RE.search(text):
        return True
    if _PLACE_QUERY_RE.search(text) and _EVENT_VERB_RE.search(text):
        return True
    return False


# Short follow-up pronouns/question words that reference prior context.
_FOLLOWUP_RE = re.compile(
    r"^\s*(?:"
    r"when\??|where\??|who\??|what\??|how\??|why\??|"
    r"wann\??|wo\??|wer\??|was\??|wie\??|warum\??|"
    r"(?:when|where|what|how|why|wann|wo|wer|was|wie)\s+.{1,60}"
    r"|(?:and\s+)?(?:the|that|this|it|those|these|das|die|den|dieser?)\b.{0,80}"
    r")\s*\??$",
    re.IGNORECASE,
)


def _is_online_followup(text: str, tool_msgs: list[dict]) -> bool:
    """True when the current message is a short follow-up to a prior web search.

    Catches 'when exactly did that happen?', 'and the evacuation?', 'who was
    involved?' etc. after the model already retrieved online data in this turn
    or when recent tool messages contain a web_search result.
    """
    if not text or len(text.strip()) > 120:
        return False
    if _HA_SENSOR_RE.search(text):
        return False
    if not _FOLLOWUP_RE.search(text):
        return False
    # Only treat as online follow-up when there is evidence of a prior
    # web search in the current conversation turn's tool messages.
    return _any_tool_call(tool_msgs, "web_search")


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


def _strip_narration(text: str, lang: str = "en") -> str:
    """Remove tool-call narration lines from a model reply, in any
    supported language.

    Reads keyword + raw-pattern lists from L (i18n module). When the
    entire reply is narration, the empty result triggers the
    'I couldn't produce' fallback in async_process — see the empty-reply
    branch downstream of this call.
    """
    if not text:
        return text
    cleaned = text
    keyword_re = L.keyword_re("narration", lang)
    if keyword_re is not None:
        # Strip any line containing a narration keyword.
        cleaned = re.sub(
            rf"^.*(?:{keyword_re.pattern}).*$",
            "",
            cleaned,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    for pattern in L.pattern_list("narration_full", lang):
        cleaned = pattern.sub("", cleaned)
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


# Kaomoji ("(╯°□°）╯︵ ┻━┻", "(・_・)") are built from box-drawing, geometric,
# and CJK punctuation glyphs that sit outside the emoji ranges above, so they
# slip through and TTS reads them as garbage. Two-stage strip: bracketed
# groups containing such glyphs go first (this also removes enclosed ° and _
# that belong to the face), then any leftover glyphs of those classes.
# The degree sign itself is NOT a trigger — "21°C" must survive.
# ಠ/ಥ are Kannada letters that live almost exclusively in kaomoji when they
# appear in the plugin's shipped languages (en/de/fr/es/pt/pl).
_KAOMOJI_TRIGGER = "─-◿︰-﹏・･ಠಥ"
_KAOMOJI_GROUP_RE = re.compile(
    rf"[(（\[][^()（）\[\]]*[{_KAOMOJI_TRIGGER}][^()（）\[\]]*[)）\]]"
)
_KAOMOJI_CHAR_RE = re.compile(r"[─-◿︰-﹏]+|[ಠಥ]\s*_\s*[ಠಥ]")


def _strip_emoji(text: str) -> str:
    """Remove emoji/pictographs/kaomoji. Safe for TTS and voice output."""
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = _KAOMOJI_GROUP_RE.sub("", cleaned)
    cleaned = _KAOMOJI_CHAR_RE.sub("", cleaned)
    # Collapse residual double-spaces introduced by stripped glyphs.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


# Trailing "check X for details" / "you can find more at Y" filler sentences.
# Models love padding factual answers with redirects to external sites; the
# system prompt forbids it but small local models still emit them. Strip them
# out post-hoc to keep replies tight.
_FILLER_CLOSING_RE = re.compile(
    r"(?:^|(?<=[.!?]\s))"
    r"(?:"
    # English patterns
    r"check\s+(?:the\s+|local\s+|specific\s+|out\s+)?"
    r"(?:event\s+)?(?:calendars?|listings?|details?|websites?|sites?|news\s+links?|providers?)"
    r"[^\n]*"
    r"|(?:specific\s+|detailed?\s+|more\s+|further\s+|additional\s+)?"
    r"(?:details?|information|info)\s+(?:can\s+be\s+found|are\s+available|is\s+available)"
    r"[^\n]*"
    r"|(?:you\s+(?:can|may|might|could|should)(?:\s+want\s+to|\s+wish\s+to|\s+like\s+to)?\s+"
    r"(?:check|find|visit|see|refer|consult|browse|explore|consider|look))"
    r"[^\n]*"
    r"|(?:consider\s+(?:checking|visiting|consulting|looking))[^\n]*"
    r"|(?:it(?:'s|\s+is)\s+(?:advisable|recommended|best)\s+to\s+(?:check|visit|consult))[^\n]*"
    r"|for\s+(?:[\w'-]+\s+){0,8}?"
    r"(?:information|info|details?|coverage|listings?|updates?|news)"
    r"[^\n]*"
    r"|(?:visit|refer\s+to|consult|see|check\s+out)\s+(?:the\s+)?"
    r"(?:website|site|page|link|url|provided\s+links?)[^\n]*"
    # German patterns
    r"|weitere\s+informationen[^\n]*"
    r"|f[uü]r\s+(?:weitere|spezifische|detaillierte?|mehr)\s+"
    r"(?:informationen|details?|infos?)[^\n]*"
    r"|schauen\s+sie[^\n]*"
    r"|besuchen\s+sie[^\n]*"
    r"|(?:einzelheiten|details?)\s+(?:finden\s+sie|sind\s+verf[uü]gbar)[^\n]*"
    r")\s*$",
    re.IGNORECASE,
)


# Markdown / formatting tokens that read awfully via TTS or that some
# announce-type integrations parse oddly when present in a single utterance.
_MD_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_MD_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+", re.MULTILINE)
_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)


def _flatten_for_voice(text: str) -> str:
    """Strip markdown + collapse newlines for voice TTS.

    Multi-line replies with bullets or **bold** read poorly via TTS
    ("asterisk asterisk Title asterisk asterisk") and may interact badly
    with announce-style media routing. Voice-mode replies should be
    plain flowing prose. Idempotent.
    """
    if not text:
        return text
    cleaned = _MD_BOLD_RE.sub(r"\1", text)
    cleaned = _MD_ITALIC_RE.sub(r"\1", cleaned)
    cleaned = _MD_HEADING_RE.sub("", cleaned)
    cleaned = _MD_BULLET_RE.sub("", cleaned)
    # Collapse newlines into sentence breaks
    cleaned = re.sub(r"\n+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _strip_filler_closing(text: str) -> str:
    """Remove trailing 'check X for details' filler sentences.

    Iterative: a reply may contain two filler sentences in a row.
    Stops after a fixed number of passes so a pathological input
    cannot loop.
    """
    if not text:
        return text
    cleaned = text.strip()
    for _ in range(3):
        new = _FILLER_CLOSING_RE.sub("", cleaned).strip()
        if new == cleaned:
            break
        cleaned = new
    return cleaned

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


# Sentence boundary for the streaming gate. The trailing sentence is always
# held back so the filler-closing strip can still act on it.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class _DeltaGate:
    """Sentence-buffered forwarder between provider deltas and HA's chat log.

    Streamed speech cannot be recalled, so the gate only forwards sentences
    that survive the same sanitation the final reply gets, holds back the
    trailing sentence, and shuts permanently the moment the turn stops being
    stream-safe (actuator call, get_entity miss, thinking leak). A gate with
    no listener is inert — all methods are cheap no-ops.
    """

    def __init__(
        self,
        listener,  # Callable[[str], None] | None
        lang: str = "en",
        voice_mode: bool = False,
    ) -> None:
        self._listener = listener
        self._lang = lang
        self._voice = voice_mode
        self._buf = ""
        self._forwarded: list[str] = []

    @property
    def active(self) -> bool:
        return self._listener is not None

    @property
    def forwarded_text(self) -> str:
        return " ".join(self._forwarded)

    def close(self) -> None:
        """Stop forwarding for the rest of the turn (safety trigger hit)."""
        self._listener = None
        self._buf = ""

    def new_response(self) -> None:
        """Discard the unemitted tail between LLM calls in the tool loop.

        Prose preceding a tool call (usually narration) must never be
        spoken — it stays in the buffer thanks to the hold-back rule and
        is dropped here.
        """
        self._buf = ""

    def feed(self, fragment: str) -> None:
        """Provider callback: accumulate and emit completed sentences."""
        if self._listener is None or not fragment:
            return
        self._buf += fragment
        if "<think" in self._buf:
            self.close()
            return
        parts = _SENTENCE_SPLIT_RE.split(self._buf)
        if len(parts) <= 1:
            return
        *complete, self._buf = parts
        for sentence in complete:
            self._emit(sentence)

    def _emit(self, sentence: str) -> None:
        s = _strip_narration(sentence, lang=self._lang)
        s = _strip_emoji(s)
        if self._voice:
            s = _flatten_for_voice(s)
        s = s.strip()
        if not s:
            return
        self._forwarded.append(s)
        try:
            self._listener(s + " ")
        except Exception:  # noqa: BLE001
            _LOGGER.debug("AI Plugin: delta listener raised", exc_info=True)

    def flush_final(self, final_text: str) -> None:
        """Send whatever part of the finished reply was not yet streamed.

        When nothing streamed, the whole reply goes out in one delta (chat
        log stays the single source of what was said). When the processed
        reply no longer starts with the streamed prefix (a post-processor
        rewrote it), the tail is withheld rather than double-spoken.
        """
        if self._listener is None:
            return
        listener = self._listener
        self._listener = None
        final_text = (final_text or "").strip()
        if not self._forwarded:
            if final_text:
                try:
                    listener(final_text)
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("AI Plugin: delta listener raised", exc_info=True)
            return
        norm_fwd = " ".join(self.forwarded_text.split())
        norm_final = " ".join(final_text.split())
        if norm_final.startswith(norm_fwd):
            rest = norm_final[len(norm_fwd):].strip()
            if rest:
                try:
                    listener(rest)
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("AI Plugin: delta listener raised", exc_info=True)
        else:
            _LOGGER.debug(
                "AI Plugin: streamed prefix diverged from final reply — tail withheld"
            )


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

    def is_voice_device(self, device_id: str | None) -> bool:
        """Public alias of _is_voice_device for callers in other modules."""
        return self._is_voice_device(device_id)

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
            enable_thinking=bool(opts.get(CONF_ENABLE_THINKING, DEFAULT_ENABLE_THINKING)),
            keep_alive=str(opts.get(CONF_KEEP_ALIVE, DEFAULT_KEEP_ALIVE) or ""),
        )

    async def _build_system_prompt(self, voice_mode: bool) -> str:
        """Return the STATIC system prompt for this request.

        Only content that is byte-stable across turns belongs here (base
        prompt, custom prompt, home location — cached after first resolve).
        Volatile blocks ([CURRENT TIME], [USER FACTS], [LAST ACTION]) go
        into a late system message via _build_volatile_block instead: a
        stable prompt head lets Ollama reuse its prompt prefix cache, so
        each turn only pre-fills the new tokens instead of the whole
        system prompt + tool schemas + history.

        v0.9.0: per-language trigger hints removed. Modern multilingual
        LLMs route non-English utterances correctly without pinned hints,
        and deterministic shortcuts in shortcuts.py handle the
        load-bearing language-specific behaviour.
        """
        base = SYSTEM_PROMPT_VOICE if voice_mode else SYSTEM_PROMPT_DEFAULT
        location_block = await self._build_location_block()
        custom = self._entry.options.get(CONF_SYSTEM_PROMPT, "").strip()
        parts = [base, location_block]
        if custom:
            parts.append(custom)
        return "\n\n".join(p for p in parts if p)

    async def _build_volatile_block(
        self, user_id: str | None, conversation_id: str
    ) -> str:
        """Per-turn context ([CURRENT TIME], [USER FACTS], [LAST ACTION]).

        Injected as a system message directly before the newest user turn
        so the static prompt head above stays cache-stable.
        """
        parts = [
            self._build_time_block(),
            await self._build_user_facts_block(user_id),
        ]
        last = self._last_entities.get(conversation_id)
        if last:
            parts.append(f"[LAST ACTION]\n{last}")
        return "\n\n".join(p for p in parts if p)

    def _build_time_block(self) -> str:
        """Return a fresh [CURRENT TIME] block in the user's local timezone.

        Computed per-request so small models that ignore tools still answer
        time questions correctly. Without this block the LLM hallucinates
        clock readings.
        """
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo
            cfg = self._hass.config if self._hass is not None else None
            tz_name = (getattr(cfg, "time_zone", None) or "").strip() if cfg else ""
            now = datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.now().astimezone()
            stamp = now.strftime("%A, %Y-%m-%d %H:%M %Z")
            return (
                f"[CURRENT TIME]\n{stamp}\n"
                "Use this for any question about the time, date, day-of-week, "
                "or 'how long until X'. Do NOT guess the time."
            )
        except Exception:  # noqa: BLE001
            _LOGGER.debug("AI Plugin: time block build failed", exc_info=True)
            return ""

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
                        messages, tool_schemas, user_id, voice_mode, message,
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
                        messages, tool_schemas, user_id, voice_mode, message,
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
                and tool_schemas
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
                        messages, tool_schemas, user_id, voice_mode, message,
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
                        "actions ('lights on', 'lights off', 'all off'): "
                        "CALL set_area_state(domain='light', action='turn_on') "
                        "(or 'turn_off'). For a specific device: CALL "
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
            # users. Force-blank the reply when any of these tools was
            # invoked, regardless of what the model produced. Only suppress
            # for actual voice satellites; text Assist still gets a
            # confirmation since the user can't see the device.
            _ACTUATOR_TOOLS = (
                "play_music", "media_command",
                "HassTurnOn", "HassTurnOff", "HassLightSet",
                "HassClimateSetTemperature", "HassClimateSetMode",
                "HassMediaPause", "HassMediaUnpause", "HassMediaNext",
                "HassMediaPrevious", "set_area_state",
            )
            actuator_invoked = any(
                _any_tool_call(tool_msgs, t) for t in _ACTUATOR_TOOLS
            )
            if actuator_invoked and voice_mode:
                stored_reply = ""
                reply = ""
            elif actuator_invoked and any(
                _any_tool_call(tool_msgs, t) for t in ("play_music", "media_command")
            ):
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
        """Close provider/tool sessions and cancel background work."""
        for task in list(getattr(self, "_bg_tasks", ())):
            task.cancel()
        if self._web_search is not None:
            await self._web_search.async_close()
        if self._browse_url is not None:
            await self._browse_url.async_close()
        await self._provider.async_close()
