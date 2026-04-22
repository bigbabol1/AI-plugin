"""Local HA discovery tools for AI Plugin (mcp-assist-style).

Small LLMs drown in a dumped YAML inventory of all exposed devices. This
module replaces the dump with four on-demand discovery tools that execute
inside the HA process against entity/area/device registries directly:

    list_areas()                        → every area name
    list_entities(area?, domain?)       → entities, optionally filtered
    get_entity(name_or_id)              → one entity: id, name, area, state, attrs
    search_entities(query)              → substring match across names/aliases

Responses are capped at ~1500 characters so they fit comfortably in the
model's context; overflow reports a truncation count so the model can
narrow the filter and retry.

Actions (turn on/off, set brightness, etc.) are NOT handled here — those
still flow through HA's built-in MCP server tools (HassTurnOn, HassTurnOff,
HassLightSet, etc.). This module is pure discovery.
"""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
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

_ACTION_DOMAINS = {"light", "switch", "fan", "media_player", "cover", "climate"}

_DOMAIN_SERVICE_MAP: dict[str, dict[str, str]] = {
    "light":        {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "switch":       {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "fan":          {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "media_player": {"turn_on": "turn_on", "turn_off": "turn_off"},
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
                "(e.g. 'hobbyroom') and/or domain (e.g. 'light', 'switch', "
                "'sensor', 'media_player', 'climate'). Call this to answer "
                "'which lights can you see', 'what is in the kitchen', "
                "'list all switches'."
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
                "named device use HassTurnOn/HassTurnOff instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "Area name (e.g. 'kitchen'). Call list_areas if unsure.",
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
                "required": ["area", "domain", "action"],
            },
        },
    },
]

TOOL_NAMES = {"list_areas", "list_entities", "get_entity", "search_entities", "set_area_state"}


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

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch one tool call. Never raises; errors become string replies."""
        try:
            exposed_only = arguments.get("exposed_only", True)
            if not isinstance(exposed_only, bool):
                exposed_only = True

            if name == "list_areas":
                return self._list_areas()
            if name == "list_entities":
                return self._list_entities(
                    area=_s(arguments.get("area")).strip() or None,
                    domain=_s(arguments.get("domain")).strip() or None,
                    exposed_only=exposed_only,
                )
            if name == "get_entity":
                return self._get_entity(
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
                return await self._set_area_state(
                    area=_s(arguments.get("area")).strip(),
                    domain=_s(arguments.get("domain")).strip(),
                    action=_s(arguments.get("action")).strip(),
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
        exposed_only: bool = True,
    ) -> str:
        hass = self._hass
        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)
        area_reg = ar.async_get(hass)

        target_area = area.lower() if area else None
        target_domain = domain.lower().strip(".") if domain else None

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

        rows: list[tuple[str, str, str]] = []
        for entity_id, entry in ent_reg.entities.items():
            if target_domain and not entity_id.startswith(f"{target_domain}."):
                continue
            if exposed_only and not self._is_exposed(entity_id):
                continue
            a_name = _area_name(entry)
            if target_area:
                if not a_name or a_name.lower() != target_area:
                    continue
            state = hass.states.get(entity_id)
            friendly = _s(state.attributes.get("friendly_name")) if state else ""
            friendly = friendly or _s(entry.name) or _s(entry.original_name) or entity_id
            rows.append((entity_id, friendly, a_name or "-"))

        if not rows:
            filt = ", ".join(f"{k}={v}" for k, v in [("area", area), ("domain", domain)] if v)
            exp_note = "" if not exposed_only else ", exposed only"
            return f"No entities match ({filt or 'no filter'}{exp_note})."

        rows.sort(key=lambda r: (r[2], r[0]))
        lines = [f"- {eid} — {fn} @ {a}" for eid, fn, a in rows]
        header = f"{len(rows)} entities"
        if domain:
            header += f" domain={domain}"
        if area:
            header += f" area={area}"
        return _cap(lines, header=header + ":\n")

    def _get_entity(self, name_or_id: str, exposed_only: bool = True) -> str:
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
                if key in state.attributes:
                    attrs.append(f"  {key}: {state.attributes[key]}")

        lines = [
            f"entity_id: {entity_id}",
            f"name: {friendly}",
            f"area: {area}",
            f"state: {state_val}",
        ]
        if attrs:
            lines.append("attributes:")
            lines.extend(attrs)
        return "\n".join(lines)

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
        area: str,
        domain: str,
        action: str,
        exposed_only: bool = True,
    ) -> str:
        """Turn devices in an area on/off/toggle via a single service call."""
        domain = domain.lower().strip(".")
        action = action.lower()

        if domain not in _ACTION_DOMAINS:
            allowed = ", ".join(sorted(_ACTION_DOMAINS))
            return f"Domain {domain!r} not supported. Allowed: {allowed}."

        svc_map = _DOMAIN_SERVICE_MAP[domain]
        if action not in svc_map:
            allowed = " or ".join(svc_map)
            return f"{domain} does not support {action}. Use {allowed}."

        # resolve area (name or alias, case-insensitive)
        area_reg = ar.async_get(self._hass)
        target_area = None
        needle = area.lower()
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

        if not ids:
            scope = "exposed " if exposed_only else ""
            return f"No {scope}{domain} in {target_area.name}."

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
            f"OK — {action} {len(ids)} {domain}(s) in {target_area.name}: "
            f"{preview}{more}"
        )
