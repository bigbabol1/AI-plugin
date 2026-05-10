"""Local HA discovery and area-action tools for AI Plugin (mcp-assist-style).

Small LLMs drown in a dumped YAML inventory of all exposed devices. This
module provides five on-demand tools that execute inside the HA process
against entity/area/device registries directly:

    list_areas()                          → every area name
    list_entities(area?, domain?)         → entities, optionally filtered
    get_entity(name_or_id)                → one entity: id, name, area, state, attrs
    search_entities(query)                → substring match across names/aliases
    set_area_state(area, domain, action)  → bulk action on all entities in an area

Discovery responses are capped at ~1500 characters so they fit comfortably in
the model's context; overflow reports a truncation count so the model can
narrow the filter and retry.

Single-device actions (set brightness, set temperature, etc.) still flow
through HA's built-in MCP server tools (HassTurnOn, HassLightSet, etc.).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    intent,
)

try:
    from homeassistant.components.homeassistant.exposed_entities import (
        async_should_expose as _ha_should_expose,
    )
    _EXPOSURE_API_AVAILABLE = True
except ImportError:
    _ha_should_expose = None  # type: ignore[assignment]
    _EXPOSURE_API_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)

_MAX_RESPONSE_CHARS = 1500

# Attributes worth surfacing to the LLM for common entity types.
_INTERESTING_ATTRS = (
    "brightness",
    "color_mode",
    "rgb_color",
    "color_temp",
    "color_temp_kelvin",
    "current_temperature",
    "temperature",
    "target_temp_low",
    "target_temp_high",
    "current_position",
    "percentage",
    "media_title",
    "media_artist",
    "volume_level",
    "hvac_mode",
    "preset_mode",
    "battery_level",
    "humidity",
    "unit_of_measurement",
)

_ACTION_DOMAINS = {"light", "switch", "fan", "cover", "climate"}

# Patterns that confirm the user explicitly meant "every device, every area"
# rather than a specific device. Used to gate set_area_state(area='all')
# against the model's tendency to fall back to it after a failed specific
# call. Multilingual; conservative.
_USER_SAID_ALL_RE = re.compile(
    r"\b(?:"
    r"all|every|everywhere|anywhere|whole\s+(?:house|flat|home)|"
    r"entire\s+(?:house|flat|home)|every\s+room|"
    r"alle|jede(?:[srn])?|j[eé]des|"
    r"[uü]berall|im\s+ganzen?\s+haus|in\s+der\s+ganzen\s+wohnung|"
    r"komplett(?:e[srn]?)?"
    r")\b",
    re.IGNORECASE,
)

# Magic values that set_area_state treats as "every area".
# Empty string is intentionally NOT here: omitting area means "default
# to caller's area" (when device_id resolves), with sweep-all only as
# the explicit-keyword path.
_SWEEP_ALL_KEYWORDS = {
    "*",
    "all",
    "any",
    "every",
    "everywhere",
    "anywhere",
    "whole house",
    "entire house",
    "house",
    # German aliases — primary user locale.
    "alle",
    "alles",
    "überall",
    "ueberall",
    "ganzes haus",
}

_DOMAIN_SERVICE_MAP: dict[str, dict[str, str]] = {
    "light":        {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "switch":       {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "fan":          {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "cover":        {"turn_on": "open_cover", "turn_off": "close_cover", "toggle": "toggle"},
    "climate":      {"turn_on": "turn_on", "turn_off": "turn_off"},
}

_EXPOSED_ONLY_PROP = {
    "exposed_only": {
        "type": "boolean",
        "description": (
            "Only include entities exposed to the conversation assistant. "
            "Set to false to inspect hidden or diagnostic entities. Default true."
        ),
    },
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_areas",
            "description": (
                "List every area (room) defined in Home Assistant. Call this "
                "to answer 'what rooms do you have' or to discover valid area "
                "names for use with list_entities."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_entities",
            "description": (
                "List entities in Home Assistant. Optionally filter by area "
                "(e.g. 'hobbyroom'), domain (e.g. 'light', 'switch', "
                "'sensor'), and/or state (e.g. 'on', 'off', 'home'). Call "
                "this to answer 'which lights can you see', 'what is in the "
                "kitchen', 'list all switches', 'which lights are on', "
                "'any windows open?'. Each row includes the live state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "Optional area name to filter by.",
                    },
                    "domain": {
                        "type": "string",
                        "description": (
                            "Optional HA domain to filter by (light, switch, "
                            "sensor, climate, media_player, cover, etc.)."
                        ),
                    },
                    "state": {
                        "type": "string",
                        "description": (
                            "Optional state filter (case-insensitive). "
                            "Examples: 'on', 'off', 'home', 'open', 'unavailable'. "
                            "Use this to answer 'which lights are on', "
                            "'any windows open', 'welche Lichter sind an'."
                        ),
                    },
                    **_EXPOSED_ONLY_PROP,
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity",
            "description": (
                "Get the current state, area, and key attributes of a single "
                "entity. Accepts an entity_id (e.g. 'light.smartbulb') or a "
                "user-facing name / alias. Call this to answer 'is X on', "
                "'what temperature in Y', 'in which room is Z', 'what is the "
                "brightness of the smart bulb'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name_or_id": {
                        "type": "string",
                        "description": "Entity id, friendly name, or alias.",
                    },
                    **_EXPOSED_ONLY_PROP,
                },
                "required": ["name_or_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_entities",
            "description": (
                "Substring search across entity_id, friendly_name, and aliases. "
                "Use when the user refers to a device by partial or fuzzy name "
                "and you need to find its canonical entity_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Substring to match (case-insensitive).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10).",
                    },
                    **_EXPOSED_ONLY_PROP,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_area_state",
            "description": (
                "Turn devices in an area on/off/toggle. Use for commands like "
                "'lights in kitchen off', 'fans in bedroom on'. For a single "
                "named device use HassTurnOn/HassTurnOff instead. "
                "When the user does NOT name a room (e.g. 'lights on'), OMIT "
                "the area parameter — the plugin defaults to the calling "
                "satellite's room. Use 'all'/'everywhere' ONLY when the user "
                "explicitly says 'all', 'every', 'whole house', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": (
                            "Area name when the user names a specific room "
                            "(e.g. 'kitchen', 'bedroom'). OMIT when the user "
                            "did not name a room — plugin uses caller's "
                            "satellite area. Pass 'all'/'everywhere'/'*' "
                            "ONLY when the user explicitly says 'all', "
                            "'every', 'everywhere', 'whole house', etc."
                        ),
                    },
                    "domain": {
                        "type": "string",
                        "enum": sorted(_ACTION_DOMAINS),
                    },
                    "action": {
                        "type": "string",
                        "enum": ["turn_on", "turn_off", "toggle"],
                    },
                    **_EXPOSED_ONLY_PROP,
                },
                "required": ["domain", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": (
                "Search and play music on a speaker via Music Assistant. "
                "Use for music-playback requests like 'play Enya', 'play "
                "jazz', 'shuffle my workout playlist', 'spiel jazz im "
                "wohnzimmer'. If the user names a room, pass that exact "
                "room as area. If the user does NOT name a room, OMIT "
                "the area parameter entirely — the plugin will fall back "
                "to the area of the satellite that received the request. "
                "Do NOT guess or default to a specific room when none "
                "was mentioned. The audio starting on the speaker IS "
                "the confirmation — reply with an empty string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to play — track, album, artist, playlist, "
                            "or genre. Free text. Examples: 'Enya', 'Hotel "
                            "California', 'jazz', 'workout playlist'."
                        ),
                    },
                    "area": {
                        "type": "string",
                        "description": (
                            "Area name when the user explicitly names a "
                            "room (e.g. 'kitchen', 'living room', "
                            "'wohnzimmer'). OMIT when the user did not "
                            "specify a room — the plugin defaults to "
                            "the calling satellite's own area."
                        ),
                    },
                    "media_type": {
                        "type": "string",
                        "enum": ["track", "album", "artist", "playlist", "radio"],
                        "description": "Optional. Default 'track'.",
                    },
                    **_EXPOSED_ONLY_PROP,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "media_command",
            "description": (
                "Control playback: pause, resume, skip to next track, go "
                "back to previous track, or stop. Examples: 'pause' / 'pause "
                "the music' → media_command('pause'). 'next track' / 'skip "
                "this song' → media_command('next'). 'previous' → "
                "media_command('previous'). 'resume' / 'continue' / 'weiter' "
                "→ media_command('resume'). 'stop the music' → media_command"
                "('stop'). Pass area only when the user names a room ('pause "
                "hobby room', 'next track in the kitchen'); otherwise the "
                "plugin auto-targets whichever media_player is currently in "
                "the matching state. The audio change IS the confirmation — "
                "reply with an empty string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["pause", "resume", "next", "previous", "stop"],
                        "description": (
                            "Playback action. 'resume' un-pauses; 'next' / "
                            "'previous' move within the current queue; 'stop' "
                            "halts and clears."
                        ),
                    },
                    "area": {
                        "type": "string",
                        "description": (
                            "Optional area name (e.g. 'hobby room'). Omit for "
                            "global commands like bare 'pause' or 'next "
                            "track' — the plugin will find the currently-"
                            "playing speaker automatically."
                        ),
                    },
                    **_EXPOSED_ONLY_PROP,
                },
                "required": ["command"],
            },
        },
    },
]

_TIMER_DURATION_PROPS = {
    "hours": {
        "type": "integer",
        "description": "Hours portion of the duration (optional, default 0).",
    },
    "minutes": {
        "type": "integer",
        "description": "Minutes portion of the duration (optional, default 0).",
    },
    "seconds": {
        "type": "integer",
        "description": "Seconds portion of the duration (optional, default 0).",
    },
}

TOOL_SCHEMAS.extend(
    [
        {
            "type": "function",
            "function": {
                "name": "start_timer",
                "description": (
                    "Start a voice timer on the calling assist_satellite device. "
                    "Use for 'set a 5 minute timer', 'timer for 10 minutes called pasta', "
                    "'wecker auf 3 minuten'. Provide at least one of hours / minutes / "
                    "seconds. The optional name lets the user reference this specific "
                    "timer later (cancel, status). When the timer ends, the satellite "
                    "fires its on_timer_finished hook which speaks the announcement on "
                    "the configured Mic2MP speaker."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Optional human label for this timer (e.g. 'pasta', "
                                "'eggs', 'workout'). Used by the user to refer to the "
                                "timer later."
                            ),
                        },
                        **_TIMER_DURATION_PROPS,
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_timer",
                "description": (
                    "Cancel a running voice timer on the calling assist_satellite. "
                    "Provide the timer name if multiple are running; omit to cancel "
                    "the only / most recent one."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to cancel.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "pause_timer",
                "description": "Pause a running voice timer on the calling satellite.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to pause.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "unpause_timer",
                "description": "Resume a paused voice timer on the calling satellite.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to resume.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "increase_timer",
                "description": (
                    "Add time to a running voice timer (e.g. 'add 2 minutes to the "
                    "pasta timer')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to extend.",
                        },
                        **_TIMER_DURATION_PROPS,
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "decrease_timer",
                "description": (
                    "Subtract time from a running voice timer (e.g. 'take 30 seconds "
                    "off the eggs timer')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to shorten.",
                        },
                        **_TIMER_DURATION_PROPS,
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "timer_status",
                "description": (
                    "Report status of voice timers on the calling satellite "
                    "('how much time on the pasta timer', 'wie lange noch')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name of the timer to report on.",
                        },
                    },
                    "required": [],
                },
            },
        },
    ]
)

_TIMER_INTENTS = {
    "start_timer": "HassStartTimer",
    "cancel_timer": "HassCancelTimer",
    "pause_timer": "HassPauseTimer",
    "unpause_timer": "HassUnpauseTimer",
    "increase_timer": "HassIncreaseTimer",
    "decrease_timer": "HassDecreaseTimer",
    "timer_status": "HassTimerStatus",
}

TOOL_NAMES = {
    "list_areas",
    "list_entities",
    "get_entity",
    "search_entities",
    "set_area_state",
    "play_music",
    "media_command",
    *_TIMER_INTENTS.keys(),
}

_MUSIC_MEDIA_TYPES = {"track", "album", "artist", "playlist", "radio"}

_MEDIA_COMMAND_SERVICES = {
    "pause": "media_pause",
    "resume": "media_play",
    "next": "media_next_track",
    "previous": "media_previous_track",
    "stop": "media_stop",
}


def _s(val: object) -> str:
    """Coerce anything (including HA's ComputedNameType) to a plain str."""
    if val is None:
        return ""
    return val if isinstance(val, str) else str(val)


def _cap(lines: list[str], header: str = "") -> str:
    """Join lines and cap to _MAX_RESPONSE_CHARS, appending a truncation note."""
    out = header
    kept: list[str] = []
    for i, line in enumerate(lines):
        candidate = out + "\n".join(kept + [line])
        if len(candidate) > _MAX_RESPONSE_CHARS and kept:
            remaining = len(lines) - i
            kept.append(f"…truncated, {remaining} more — narrow the filter")
            break
        kept.append(line)
    return (header + "\n".join(kept)).strip() or "(empty)"


class HALocalToolRegistry:
    """Registry of discovery tools running against HA's in-process registries."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        if not _EXPOSURE_API_AVAILABLE:
            _LOGGER.warning(
                "AI Plugin: exposed_entities API unavailable on this HA version — "
                "exposed_only filter disabled, all registry entities will be shown"
            )

    @property
    def tool_names(self) -> set[str]:
        return TOOL_NAMES

    def get_schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def _is_exposed(self, entity_id: str) -> bool:
        """Return True if entity is exposed to the conversation assistant."""
        if not _EXPOSURE_API_AVAILABLE:
            return True  # Fallback: treat all as exposed on older HA.
        try:
            return _ha_should_expose(self._hass, "conversation", entity_id)
        except Exception:  # noqa: BLE001
            return True  # Fail open — never silently hide entities on error.

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        device_id: str | None = None,
        language: str | None = None,
        user_message: str = "",
    ) -> str:
        """Dispatch one tool call. Never raises; errors become string replies."""
        try:
            exposed_only = arguments.get("exposed_only", True)
            if not isinstance(exposed_only, bool):
                exposed_only = True

            if name in _TIMER_INTENTS:
                return await self._call_timer_intent(
                    name, arguments, device_id=device_id, language=language
                )

            if name == "list_areas":
                return self._list_areas()
            if name == "list_entities":
                return self._list_entities(
                    area=_s(arguments.get("area")).strip() or None,
                    domain=_s(arguments.get("domain")).strip() or None,
                    state=_s(arguments.get("state")).strip() or None,
                    exposed_only=exposed_only,
                )
            if name == "get_entity":
                return await self._get_entity(
                    _s(arguments.get("name_or_id")).strip(),
                    exposed_only=exposed_only,
                )
            if name == "search_entities":
                limit = arguments.get("limit")
                try:
                    limit = int(limit) if limit is not None else 10
                except (TypeError, ValueError):
                    limit = 10
                return self._search_entities(
                    query=_s(arguments.get("query")).strip(),
                    limit=max(1, min(50, limit)),
                    exposed_only=exposed_only,
                )
            if name == "set_area_state":
                raw_area = arguments.get("area")
                area_val = _s(raw_area).strip() if raw_area is not None else ""
                # Safety guard: model often falls back to area='all' after a
                # failed HassTurnOff for a specific device, which then turns
                # off everything in the flat (lights + thermostats + fans).
                # Reject area='all' / sweep keywords unless the user
                # explicitly said 'all'/'every'/'whole house'/etc.
                if area_val.lower() in _SWEEP_ALL_KEYWORDS:
                    if not _USER_SAID_ALL_RE.search(user_message or ""):
                        _LOGGER.warning(
                            "AI Plugin: rejected set_area_state(area=%r) — "
                            "user message %r does not contain explicit sweep "
                            "keyword. Likely model fallback after a failed "
                            "specific-device call.",
                            area_val, (user_message or "")[:80],
                        )
                        return (
                            "[set_area_state(area='all') refused — user did "
                            "not say 'all', 'every', or 'whole house'. For a "
                            "specific device call search_entities to find it, "
                            "then HassTurnOn/HassTurnOff with the entity_id.]"
                        )
                return await self._set_area_state(
                    area=area_val or None,
                    domain=_s(arguments.get("domain")).strip(),
                    action=_s(arguments.get("action")).strip(),
                    exposed_only=exposed_only,
                    device_id=device_id,
                )
            if name == "play_music":
                return await self._play_music(
                    query=_s(arguments.get("query")).strip(),
                    area=_s(arguments.get("area")).strip() or None,
                    media_type=_s(arguments.get("media_type")).strip() or "track",
                    exposed_only=exposed_only,
                    device_id=device_id,
                )
            if name == "media_command":
                return await self._media_command(
                    command=_s(arguments.get("command")).strip().lower(),
                    area=_s(arguments.get("area")).strip() or None,
                    exposed_only=exposed_only,
                )
            return f"[Unknown local tool: {name!r}]"
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin local tool %r failed", name)
            return f"[Local tool error: {exc}]"

    # ── implementations ────────────────────────────────────────────────────────

    def _list_areas(self) -> str:
        area_reg = ar.async_get(self._hass)
        names = sorted({_s(a.name) for a in area_reg.async_list_areas() if _s(a.name)})
        if not names:
            return "No areas defined."
        return _cap([f"- {n}" for n in names], header=f"{len(names)} areas:\n")

    def _list_entities(
        self,
        area: str | None = None,
        domain: str | None = None,
        state: str | None = None,
        exposed_only: bool = True,
    ) -> str:
        hass = self._hass
        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)
        area_reg = ar.async_get(hass)

        target_area = area.lower() if area else None
        target_domain = domain.lower().strip(".") if domain else None
        target_state = state.lower().strip() if state else None

        def _area_name(entry) -> str | None:
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if not area_id:
                return None
            a = area_reg.async_get_area(area_id)
            return _s(a.name) if a else None

        rows: list[tuple[str, str, str, str]] = []
        for entity_id, entry in ent_reg.entities.items():
            if target_domain and not entity_id.startswith(f"{target_domain}."):
                continue
            if exposed_only and not self._is_exposed(entity_id):
                continue
            a_name = _area_name(entry)
            if target_area:
                if not a_name or a_name.lower() != target_area:
                    continue
            ent_state = hass.states.get(entity_id)
            state_val = ent_state.state if ent_state else "unknown"
            if target_state and state_val.lower() != target_state:
                continue
            friendly = _s(ent_state.attributes.get("friendly_name")) if ent_state else ""
            friendly = friendly or _s(entry.name) or _s(entry.original_name) or entity_id
            rows.append((entity_id, friendly, a_name or "-", state_val))

        if not rows:
            filt = ", ".join(
                f"{k}={v}" for k, v in [("area", area), ("domain", domain), ("state", state)] if v
            )
            exp_note = "" if not exposed_only else ", exposed only"
            return f"No entities match ({filt or 'no filter'}{exp_note})."

        rows.sort(key=lambda r: (r[2], r[0]))
        lines = [f"- {eid} — {fn} [{st}] @ {a}" for eid, fn, a, st in rows]
        header = f"{len(rows)} entities"
        if domain:
            header += f" domain={domain}"
        if area:
            header += f" area={area}"
        if state:
            header += f" state={state}"
        return _cap(lines, header=header + ":\n")

    async def _get_entity(self, name_or_id: str, exposed_only: bool = True) -> str:
        if not name_or_id:
            return "get_entity: empty name_or_id."
        hass = self._hass
        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)
        area_reg = ar.async_get(hass)

        def _area_for(entry) -> str | None:
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if not area_id:
                return None
            a = area_reg.async_get_area(area_id)
            return _s(a.name) if a else None

        entry = None
        if "." in name_or_id:
            entry = ent_reg.async_get(name_or_id)
        if entry is None:
            needle = name_or_id.strip().lower()
            for eid, e in ent_reg.entities.items():
                candidates = [_s(e.name), _s(e.original_name)]
                candidates.extend(_s(a) for a in (e.aliases or ()))
                st = hass.states.get(eid)
                if st:
                    candidates.append(_s(st.attributes.get("friendly_name")))
                if any(c and c.strip().lower() == needle for c in candidates):
                    entry = e
                    break
            if entry is None:
                for eid, e in ent_reg.entities.items():
                    candidates = [_s(e.name), _s(e.original_name), eid]
                    candidates.extend(_s(a) for a in (e.aliases or ()))
                    st = hass.states.get(eid)
                    if st:
                        candidates.append(_s(st.attributes.get("friendly_name")))
                    if any(c and needle in c.lower() for c in candidates):
                        entry = e
                        break
        if entry is None:
            return f"No entity matches {name_or_id!r}. Try search_entities."

        entity_id = entry.entity_id
        if exposed_only and not self._is_exposed(entity_id):
            return (
                f"Entity {entity_id!r} exists but is not exposed to the conversation assistant. "
                f"Use exposed_only=false if you need to inspect it."
            )

        state = hass.states.get(entity_id)
        friendly = _s(state.attributes.get("friendly_name")) if state else ""
        friendly = friendly or _s(entry.name) or _s(entry.original_name) or entity_id
        area = _area_for(entry) or "-"
        state_val = state.state if state else "unknown"

        attrs: list[str] = []
        if state:
            for key in _INTERESTING_ATTRS:
                if key not in state.attributes:
                    continue
                val = state.attributes[key]
                if key == "brightness":
                    try:
                        pct = round(int(val) / 255 * 100)
                        attrs.append(f"  brightness: {pct}%")
                        continue
                    except (TypeError, ValueError):
                        pass
                attrs.append(f"  {key}: {val}")

        lines = [
            f"entity_id: {entity_id}",
            f"name: {friendly}",
            f"area: {area}",
            f"state: {state_val}",
        ]
        if attrs:
            lines.append("attributes:")
            lines.extend(attrs)
        if entity_id.startswith("weather."):
            forecast_block = await self._fetch_weather_forecast(entity_id)
            if forecast_block:
                lines.append(forecast_block)
        return "\n".join(lines)

    async def _fetch_weather_forecast(self, entity_id: str) -> str:
        """Call weather.get_forecasts service and format daily forecast block.

        Returns "" if entity does not support forecast service or call errors.
        """
        try:
            resp = await self._hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": entity_id, "type": "daily"},
                blocking=True,
                return_response=True,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("weather.get_forecasts failed for %s: %s", entity_id, exc)
            return ""
        data = (resp or {}).get(entity_id) or {}
        forecast = data.get("forecast") or []
        if not forecast:
            return ""
        out = ["forecast (daily, next 5):"]
        for item in forecast[:5]:
            dt = item.get("datetime", "?")
            date = dt.split("T")[0] if isinstance(dt, str) else "?"
            cond = item.get("condition", "?")
            parts = [f"  {date} {cond}"]
            hi = item.get("temperature")
            lo = item.get("templow")
            precip = item.get("precipitation_probability")
            if hi is not None:
                parts.append(f"hi:{hi}°")
            if lo is not None:
                parts.append(f"lo:{lo}°")
            if precip is not None:
                parts.append(f"precip:{precip}%")
            out.append(" ".join(parts))
        return "\n".join(out)

    def _search_entities(
        self, query: str, limit: int = 10, exposed_only: bool = True
    ) -> str:
        if not query:
            return "search_entities: empty query."
        hass = self._hass
        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)
        area_reg = ar.async_get(hass)
        needle = query.lower()

        def _area_name(entry) -> str | None:
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if not area_id:
                return None
            a = area_reg.async_get_area(area_id)
            return _s(a.name) if a else None

        hits: list[tuple[str, str, str]] = []
        for entity_id, entry in ent_reg.entities.items():
            if exposed_only and not self._is_exposed(entity_id):
                continue
            candidates = [entity_id, _s(entry.name), _s(entry.original_name)]
            candidates.extend(_s(a) for a in (entry.aliases or ()))
            state = hass.states.get(entity_id)
            if state:
                candidates.append(_s(state.attributes.get("friendly_name")))
            if any(c and needle in c.lower() for c in candidates):
                friendly = _s(state.attributes.get("friendly_name")) if state else ""
                friendly = friendly or _s(entry.name) or _s(entry.original_name) or entity_id
                hits.append((entity_id, friendly, _area_name(entry) or "-"))
                if len(hits) >= limit:
                    break

        if not hits:
            return f"No match for {query!r}."
        lines = [f"- {eid} — {fn} @ {a}" for eid, fn, a in hits]
        return _cap(lines, header=f"{len(hits)} match(es) for {query!r}:\n")

    async def _set_area_state(
        self,
        area: str | None,
        domain: str,
        action: str,
        exposed_only: bool = True,
        device_id: str | None = None,
    ) -> str:
        """Turn devices in an area on/off/toggle via a single service call.

        Area resolution:
        - Explicit area name → that area only.
        - Explicit wildcard ('all', 'every', 'everywhere', '*', etc.) →
          sweep every area.
        - Empty/None area + voice satellite device_id → caller's room.
        - Empty/None area + no device_id → sweep every area (text Assist
          / REST API: assume whole-home).
        """
        domain = domain.lower().strip(".")
        action = action.lower()

        if domain not in _ACTION_DOMAINS:
            allowed = ", ".join(sorted(_ACTION_DOMAINS))
            return f"Domain {domain!r} not supported. Allowed: {allowed}."

        svc_map = _DOMAIN_SERVICE_MAP[domain]
        if action not in svc_map:
            allowed = " or ".join(svc_map)
            return f"{domain} does not support {action}. Use {allowed}."

        needle = (area or "").strip().lower()
        # When area is empty AND a device_id is provided, default to the
        # calling satellite's area before falling through to sweep-all.
        # User in living room saying 'lights on' should affect the living
        # room only, not every light in the flat.
        if not needle and device_id:
            try:
                ent_reg = er.async_get(self._hass)
                dev_reg = dr.async_get(self._hass)
                area_reg = ar.async_get(self._hass)
                dev = dev_reg.async_get(device_id)
                area_id = dev.area_id if dev else None
                if not area_id:
                    for entry in er.async_entries_for_device(ent_reg, device_id):
                        if entry.area_id:
                            area_id = entry.area_id
                            break
                if area_id:
                    a = area_reg.async_get_area(area_id)
                    if a:
                        needle = _s(a.name).lower()
                        _LOGGER.debug(
                            "set_area_state: area unspecified, defaulted to "
                            "caller's area %r (device_id=%s)",
                            a.name, device_id,
                        )
            except Exception:  # noqa: BLE001
                _LOGGER.debug("set_area_state device→area lookup failed", exc_info=True)
        sweep_all = needle in _SWEEP_ALL_KEYWORDS or needle == ""

        target_area = None
        if not sweep_all:
            area_reg = ar.async_get(self._hass)
            for a in area_reg.async_list_areas():
                if _s(a.name).lower() == needle:
                    target_area = a
                    break
                aliases = getattr(a, "aliases", None) or ()
                if any(_s(al).lower() == needle for al in aliases):
                    target_area = a
                    break
            if target_area is None:
                return f"Unknown area {area!r}. Try list_areas."

        ent_reg = er.async_get(self._hass)
        dev_reg = dr.async_get(self._hass)

        ids: list[str] = []
        for eid, entry in ent_reg.entities.items():
            if not eid.startswith(f"{domain}."):
                continue
            if not sweep_all:
                area_id = entry.area_id
                if not area_id and entry.device_id:
                    dev = dev_reg.async_get(entry.device_id)
                    if dev:
                        area_id = dev.area_id
                if area_id != target_area.id:
                    continue
            if exposed_only and not self._is_exposed(eid):
                continue
            ids.append(eid)

        scope_label = "all areas" if sweep_all else target_area.name
        if not ids:
            scope = "exposed " if exposed_only else ""
            return f"No {scope}{domain} in {scope_label}."

        ids.sort()
        service = svc_map[action]
        try:
            await self._hass.services.async_call(
                domain,
                service,
                target={"entity_id": ids},
                blocking=True,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin set_area_state failed")
            return f"[set_area_state failed: {exc}]"

        preview = ", ".join(ids[:6])
        more = f" (+{len(ids) - 6} more)" if len(ids) > 6 else ""
        return (
            f"OK — {action} {len(ids)} {domain}(s) in {scope_label}: "
            f"{preview}{more}"
        )

    async def _play_music(
        self,
        query: str,
        area: str | None,
        media_type: str = "track",
        exposed_only: bool = True,
        device_id: str | None = None,
    ) -> str:
        """Search and play music in an area via Music Assistant.

        Replaces the long-broken ``set_area_state(media_player, turn_on)`` path:
        Music-Assistant entities don't support the generic turn_on action, and
        even when they did it would just power on the speaker without any
        media — which is what happens when the LLM "confirms" but nothing
        plays. This tool calls ``music_assistant.play_media`` with the user's
        free-text query, which MA resolves against its providers (Spotify,
        Tidal, local library, etc.).
        """
        query = (query or "").strip()
        if not query:
            return "[play_music needs a query — what should I play?]"

        # No area specified → resolve to the calling satellite's area.
        # Voice users in the living room saying 'play jazz' should not
        # have music start somewhere else just because the model picked
        # an example area name. With device_id, look up which area the
        # satellite is in and play there.
        if not area and device_id:
            try:
                ent_reg = er.async_get(self._hass)
                dev_reg = dr.async_get(self._hass)
                area_reg = ar.async_get(self._hass)
                # Get the device's area (either directly or via any of
                # its entities). Prefer device.area_id; fall back to
                # entity.area_id of the assist_satellite entity itself.
                dev = dev_reg.async_get(device_id)
                area_id = dev.area_id if dev else None
                if not area_id:
                    for entry in er.async_entries_for_device(ent_reg, device_id):
                        if entry.area_id:
                            area_id = entry.area_id
                            break
                if area_id:
                    a = area_reg.async_get_area(area_id)
                    if a:
                        area = a.name
                        _LOGGER.debug(
                            "play_music: no area specified, defaulted to "
                            "calling satellite's area %r (device_id=%s)",
                            area, device_id,
                        )
            except Exception:  # noqa: BLE001
                _LOGGER.debug("play_music device→area lookup failed", exc_info=True)

        if not area:
            return (
                "[play_music needs an area — say which speaker, e.g. "
                "'in the kitchen' or 'in the living room'. Call "
                "list_areas to see options.]"
            )

        media_type = (media_type or "track").lower().strip()
        if media_type not in _MUSIC_MEDIA_TYPES:
            media_type = "track"

        if not self._hass.services.has_service("music_assistant", "play_media"):
            return (
                "[Music Assistant integration not installed. play_music "
                "requires Music Assistant to resolve free-text queries against "
                "music providers.]"
            )

        # Resolve area
        needle = area.strip().lower()
        area_reg = ar.async_get(self._hass)
        target_area = None
        for a in area_reg.async_list_areas():
            if _s(a.name).lower() == needle:
                target_area = a
                break
            aliases = getattr(a, "aliases", None) or ()
            if any(_s(al).lower() == needle for al in aliases):
                target_area = a
                break
        if target_area is None:
            return f"Unknown area {area!r}. Try list_areas."

        # Pick the area's primary Music-Assistant media_player.
        # Filter by REGISTRY platform == "music_assistant" (stable; works
        # whether the speaker is on or off). The previous heuristic of
        # checking state.attributes.app_id only matched when the speaker
        # was already actively playing — players with state=off were
        # silently demoted to the fallback list, which often picked a
        # surprising speaker.
        ent_reg = er.async_get(self._hass)
        dev_reg = dr.async_get(self._hass)
        ma_native: list[tuple[str, str, list[str]]] = []
        other: list[str] = []
        for eid, entry in ent_reg.entities.items():
            if not eid.startswith("media_player."):
                continue
            if entry.disabled_by is not None or entry.hidden_by is not None:
                continue
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if area_id != target_area.id:
                continue
            if exposed_only and not self._is_exposed(eid):
                continue
            state = self._hass.states.get(eid)
            fn = _s(state.attributes.get("friendly_name") if state else "") or ""
            aliases = [_s(a) for a in (getattr(entry, "aliases", None) or ())]
            if entry.platform == "music_assistant":
                ma_native.append((eid, fn, aliases))
            else:
                other.append(eid)

        if not ma_native and not other:
            scope = "exposed " if exposed_only else ""
            return f"No {scope}media_player in {target_area.name}."

        entity_id: str | None = None
        if ma_native:
            # Prefer the player whose friendly_name or alias matches the
            # area's name — that's almost always the user-blessed primary
            # speaker (e.g. media_player.wohnzimmer_2 with friendly_name
            # 'Wohnzimmer' in area 'living room' / alias 'wohnzimmer').
            area_keys = {_s(target_area.name).lower()}
            for al in (getattr(target_area, "aliases", None) or ()):
                area_keys.add(_s(al).lower())
            for eid, fn, aliases in ma_native:
                hay = {fn.lower()} | {a.lower() for a in aliases}
                if hay & area_keys:
                    entity_id = eid
                    break
            if entity_id is None:
                # Fallback: alphabetical first MA player.
                entity_id = sorted(eid for eid, _, _ in ma_native)[0]
        else:
            # No MA-platform player found — last-resort fallback to any
            # other media_player. play_media may not work via MA service
            # in this case, but the call will fail loudly.
            entity_id = sorted(other)[0]
        _LOGGER.debug(
            "play_music: target area=%r, ma_native=%d, other=%d, picked=%r",
            target_area.name, len(ma_native), len(other), entity_id,
        )

        try:
            await self._hass.services.async_call(
                "music_assistant",
                "play_media",
                target={"entity_id": entity_id},
                service_data={"media_id": query, "media_type": media_type},
                blocking=True,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin play_music failed")
            return f"[play_music failed: {exc}]"

        return f"OK — playing {query!r} ({media_type}) on {entity_id}."

    def _resolve_area(self, area: str):
        """Match an area by exact name or alias (case-insensitive). Returns
        the AreaEntry or None."""
        needle = (area or "").strip().lower()
        if not needle:
            return None
        area_reg = ar.async_get(self._hass)
        for a in area_reg.async_list_areas():
            if _s(a.name).lower() == needle:
                return a
            aliases = getattr(a, "aliases", None) or ()
            if any(_s(al).lower() == needle for al in aliases):
                return a
        return None

    def _media_players_in_area(
        self, target_area, exposed_only: bool = True
    ) -> list[str]:
        """Return media_player entity_ids assigned to ``target_area`` (directly
        or via the parent device)."""
        ent_reg = er.async_get(self._hass)
        dev_reg = dr.async_get(self._hass)
        out: list[str] = []
        for eid, entry in ent_reg.entities.items():
            if not eid.startswith("media_player."):
                continue
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if area_id != target_area.id:
                continue
            if exposed_only and not self._is_exposed(eid):
                continue
            out.append(eid)
        return out

    async def _media_command(
        self,
        command: str,
        area: str | None,
        exposed_only: bool = True,
    ) -> str:
        """Pause, resume, skip, or stop playback.

        With ``area``: targets media_players in that area, preferring those
        already in a state the command applies to (playing for pause/next/
        previous/stop, paused/idle for resume).

        Without ``area``: scans every media_player and acts on the ones
        currently in the relevant state. This covers terse phrasings like
        bare "pause" or "next track" where the LLM has no area to pass —
        the plugin finds whatever is actually playing right now.
        """
        if command not in _MEDIA_COMMAND_SERVICES:
            allowed = ", ".join(sorted(_MEDIA_COMMAND_SERVICES))
            return f"Unknown media command {command!r}. Use: {allowed}."

        # Prefer entities in a state the command actually applies to.
        relevant_states = {
            "pause": {"playing"},
            "resume": {"paused", "idle"},
            "next": {"playing", "paused"},
            "previous": {"playing", "paused"},
            "stop": {"playing", "paused"},
        }[command]

        if area:
            target_area = self._resolve_area(area)
            if target_area is None:
                return f"Unknown area {area!r}. Try list_areas."
            candidates = self._media_players_in_area(target_area, exposed_only)
            if not candidates:
                scope = "exposed " if exposed_only else ""
                return f"No {scope}media_player in {target_area.name}."
            active = [
                eid
                for eid in candidates
                if (st := self._hass.states.get(eid)) and st.state in relevant_states
            ]
            target_ids = sorted(active or candidates)
        else:
            # No area — find every media_player in the relevant state.
            ent_reg = er.async_get(self._hass)
            target_ids = []
            for eid in ent_reg.entities:
                if not eid.startswith("media_player."):
                    continue
                if exposed_only and not self._is_exposed(eid):
                    continue
                st = self._hass.states.get(eid)
                if st and st.state in relevant_states:
                    target_ids.append(eid)
            if not target_ids:
                state_label = " / ".join(sorted(relevant_states))
                return (
                    f"Nothing is {state_label} right now — pass an area to "
                    f"target a specific speaker."
                )
            target_ids.sort()

        service = _MEDIA_COMMAND_SERVICES[command]
        try:
            await self._hass.services.async_call(
                "media_player",
                service,
                target={"entity_id": target_ids},
                blocking=True,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin media_command %s failed", command)
            return f"[media_command {command} failed: {exc}]"

        preview = ", ".join(target_ids[:3])
        more = f" (+{len(target_ids) - 3} more)" if len(target_ids) > 3 else ""
        return f"OK — {command} on {preview}{more}."

    async def _call_timer_intent(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        device_id: str | None = None,
        language: str | None = None,
    ) -> str:
        """Invoke a HA built-in voice-timer intent.

        Voice timers are bound to the calling assist_satellite via ``device_id``.
        Without a device_id HA cannot ring the timer on the right satellite, so
        we refuse early with a useful message rather than letting the intent
        return an opaque error.
        """
        intent_type = _TIMER_INTENTS.get(tool_name)
        if intent_type is None:
            return f"[Unknown timer tool: {tool_name!r}]"

        if not device_id:
            return (
                "[Timer requires a satellite device — this tool is only available "
                "when the request comes from a voice pipeline. Try again from the "
                "voice satellite.]"
            )

        slots: dict[str, Any] = {}
        for key in ("name", "hours", "minutes", "seconds"):
            if key not in arguments:
                continue
            val = arguments[key]
            if val is None:
                continue
            if key == "name":
                name_val = _s(val).strip()
                if name_val:
                    slots["name"] = {"value": name_val}
            else:
                try:
                    int_val = int(val)
                except (TypeError, ValueError):
                    continue
                slots[key] = {"value": int_val}

        if tool_name == "start_timer":
            has_duration = any(k in slots for k in ("hours", "minutes", "seconds"))
            if not has_duration:
                return (
                    "[start_timer needs a duration — provide at least one of "
                    "hours / minutes / seconds.]"
                )

        try:
            response = await intent.async_handle(
                self._hass,
                "ai_plugin",
                intent_type,
                slots=slots,
                text_input=None,
                context=Context(),
                language=language or self._hass.config.language,
                assistant="conversation",
                device_id=device_id,
            )
        except intent.IntentHandleError as exc:
            return f"[{intent_type} failed: {exc}]"
        except intent.UnknownIntent:
            return (
                f"[{intent_type} is not registered in this Home Assistant version. "
                "Voice timers require HA 2024.7 or newer.]"
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin timer intent %s failed", intent_type)
            return f"[{intent_type} error: {exc}]"

        speech = ""
        try:
            speech_block = response.speech.get("plain", {}) if response.speech else {}
            speech = _s(speech_block.get("speech", "")).strip()
        except Exception:  # noqa: BLE001
            pass
        return speech or f"OK — {intent_type} dispatched."
