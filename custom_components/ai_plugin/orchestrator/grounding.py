"""Did the model actually DO what it said it did?

Small models answer a command with a confident sentence and no tool call
("I'll turn off all the lights now!") while nothing moves. Every
predicate here reads the turn's real tool traffic, so the orchestrator
can catch a promise, re-run once with a corrective nudge, and — if the
second attempt also acts on nothing — replace the promise with the
truth.
"""

from __future__ import annotations

import re

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

_MEDIA_TOOL_NAMES = ("play_music", "media_command")

def _any_media_status_call(tool_msgs: list[dict]) -> bool:
    """True if the tool-loop queried playback via media_command('status')."""
    import json as _json  # noqa: PLC0415

    for m in tool_msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name") != "media_command":
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except ValueError:
                    continue
            if isinstance(args, dict) and args.get("command") == "status":
                return True
    return False

def _tool_results(tool_msgs: list[dict], tool_names: tuple[str, ...]) -> list[str]:
    """Result contents of every completed call of the named tools."""
    ids = {
        tc.get("id")
        for m in tool_msgs
        for tc in (m.get("tool_calls") or [])
        if tc.get("function", {}).get("name") in tool_names
    }
    if not ids:
        return []
    return [
        str(m.get("content") or "")
        for m in tool_msgs
        if m.get("role") == "tool" and m.get("tool_call_id") in ids
    ]

def _any_media_success(tool_msgs: list[dict]) -> bool:
    """True if a media tool call in this turn actually changed playback.

    Media TTS suppression keys on the RESULT, not the attempt: a rejected
    call (unknown command, nothing playing) or the read-only 'status'
    command produces no audible change, so the reply must not be blanked.
    Mutating media results are prefixed "OK" by ha_local.
    """
    return any(
        r.lstrip().startswith("OK")
        for r in _tool_results(tool_msgs, _MEDIA_TOOL_NAMES)
    )

# Actuators dispatched to HA's MCP intent tools. Their failure convention
# is a "["-bracketed result (mcp_client brackets transport errors and
# isError results); any other non-empty result means the intent ran.
_INTENT_ACTUATOR_NAMES = (
    "HassTurnOn", "HassTurnOff", "HassLightSet",
    "HassClimateSetTemperature", "HassClimateSetMode",
    "HassMediaPause", "HassMediaUnpause", "HassMediaNext",
    "HassMediaPrevious",
)

def _any_actuator_success(tool_msgs: list[dict]) -> bool:
    """True if any actuator call in this turn actually performed its action.

    Suppression must never blank the model's explanation of a FAILED
    action — the user would hear silence and assume success. ha_local
    actuators mark success with an "OK" prefix; MCP intent actuators mark
    failure with a "[" prefix.
    """
    if _any_media_success(tool_msgs):
        return True
    if any(
        r.lstrip().startswith("OK")
        for r in _tool_results(tool_msgs, ("set_area_state", "set_brightness"))
    ):
        return True
    return any(
        r.strip() and not r.lstrip().startswith("[")
        for r in _tool_results(tool_msgs, _INTENT_ACTUATOR_NAMES)
    )

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
    "set_area_state", "set_brightness", "HassTurnOn", "HassTurnOff",
    "HassLightSet", "HassClimateSetTemperature",
    "HassClimateSetMode", "HassMediaPause", "HassMediaUnpause",
    "HassMediaNext", "HassMediaPrevious",
    "play_music", "media_command",
)

# Replies that CLAIM an action is being taken. When no actuator ran this
# turn, such a reply is a lie the user can act on ("I'll turn off all the
# lights in your home now!" while every lamp stays lit — observed on
# qwen3.5:9b). Only consulted on the no-actuator path, so genuine
# post-action confirmations are never touched.
_ACTION_PROMISE_RE = re.compile(
    r"(?:"
    # English: "I'll turn …", "I will switch …", "let me dim …",
    # "I'm turning …", "going to turn …"
    r"(?:i\s*(?:'|’)?ll|i\s+will|i\s+am\s+going\s+to|i'?m\s+going\s+to|"
    r"let\s+me|i'?m|i\s+am)\s+(?:now\s+|just\s+)?"
    r"(?:turn(?:ing)?|switch(?:ing)?|put(?:ting)?|set(?:ting)?|"
    r"dim(?:ming)?|start(?:ing)?|stopp?(?:ing)?|shut(?:ting)?)"
    # German: "ich schalte … aus", "ich mache das Licht an", "ich werde …"
    r"|ich\s+(?:werde\s+)?(?:schalte|schalt|mache|mach|stelle|drehe)"
    r"|ich\s+werde\s+"
    # French / Spanish / Portuguese / Polish
    r"|je\s+vais\s+(?:allumer|éteindre|eteindre|activer|désactiver)"
    r"|voy\s+a\s+(?:encender|apagar|activar|desactivar)"
    r"|vou\s+(?:ligar|desligar|acender|apagar|ativar)"
    r"|(?:zaraz\s+)?(?:włącz|wlacz|wyłącz|wylacz)\w*"
    r")",
    re.IGNORECASE,
)

def _is_action_promise(text: str) -> bool:
    """True when the reply claims an action that no tool actually performed."""
    return bool(text) and bool(_ACTION_PROMISE_RE.search(text))

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
