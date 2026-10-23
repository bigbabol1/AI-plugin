"""Registry access shared by every shortcut module.

One place resolves areas, walks the entity registry and applies Home
Assistant's exposure rules, so the shortcut modules do not each carry
their own copy of the ``async_should_expose`` dance (four copies before
this module existed, two of which logged and two of which did not).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

try:
    from homeassistant.components.homeassistant.exposed_entities import (
        async_should_expose,
    )
except ImportError:  # pragma: no cover — older HA cores
    async_should_expose = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)
_CONVERSATION_ASSISTANT = "conversation"


def is_exposed(hass: HomeAssistant, entity_id: str) -> bool:
    """True when the user exposed ``entity_id`` to the conversation agent.

    Fails OPEN: on an older core without the helper, and on any registry
    quirk that raises, the entity stays eligible. A shortcut that silently
    stopped seeing devices would be worse than one that sees one too many.
    """
    if async_should_expose is None:
        return True
    try:
        return bool(async_should_expose(hass, _CONVERSATION_ASSISTANT, entity_id))
    except Exception:  # noqa: BLE001 — never block on registry quirks
        _LOGGER.debug(
            "AI Plugin: should_expose check failed for %s", entity_id, exc_info=True
        )
        return True


# German → English area name synonyms. Helps when the user says
# "schlafzimmer" but HA area is registered as "bedroom".
_AREA_SYNONYMS: dict[str, tuple[str, ...]] = {
    "bedroom": ("schlafzimmer",),
    "living room": ("wohnzimmer", "living", "livingroom"),
    "kitchen": ("küche", "kueche"),
    "bathroom": ("bad", "badezimmer"),
    "hobby room": ("hobbyraum", "hobby"),
    "aisle": ("flur", "diele", "hallway"),
    "flat": ("wohnung", "home"),
}


def _resolve_area(hass: HomeAssistant, raw: str) -> Any | None:
    """Return AreaEntry matching raw name (case-insensitive, synonym-aware)."""
    needle = raw.strip().lower()
    if not needle:
        return None
    area_reg = ar.async_get(hass)
    # Direct name / alias match.
    for area in area_reg.async_list_areas():
        if (area.name or "").strip().lower() == needle:
            return area
        for alias in (area.aliases or ()):
            if (alias or "").strip().lower() == needle:
                return area
    # Substring fallback.
    for area in area_reg.async_list_areas():
        if needle in (area.name or "").lower():
            return area
    # Synonym map.
    for canonical, syns in _AREA_SYNONYMS.items():
        if needle == canonical or needle in syns:
            for area in area_reg.async_list_areas():
                if canonical in (area.name or "").lower():
                    return area
    return None


def _entities_in_area(hass: HomeAssistant, area_id: str) -> list[Any]:
    """Entities in an area that the user has exposed to the conversation agent.

    The exposure check matches HA's own intent / Assist pipeline so unexposed
    entities (e.g. voice-satellite chip sensors that share an area with the
    real room sensors) cannot leak into shortcut answers. If the
    exposed_entities helper is unavailable on older cores, fall back to
    returning all area-matching entities.
    """
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)
    results = []
    for entry in ent_reg.entities.values():
        eid_area = entry.area_id
        if not eid_area and entry.device_id:
            dev = dev_reg.async_get(entry.device_id)
            if dev:
                eid_area = dev.area_id
        if eid_area != area_id:
            continue
        if not is_exposed(hass, entry.entity_id):
            continue
        results.append(entry)
    return results


# German definite articles must match BEFORE the bare "in " branch —
# regex alternation is ordered, and "in\s+(?:the\s+)?" wins otherwise,
# swallowing the article into the area name ("in der küche" → "der küche"
# → no area match → command silently broadens to the whole home).
_AREA_SUFFIX_RE = re.compile(
    r"\b(?:in\s+der\s+|in\s+dem\s+|im\s+|in\s+(?:the\s+)?)"
    r"(?P<area>[\w\s\-]+?)\s*[.!?]?\s*$",
    re.IGNORECASE,
)


def _extract_area_from_media_message(hass: HomeAssistant, message: str):
    """Find an area mentioned via 'in <area>' / 'im <area>' suffix."""
    m = _AREA_SUFFIX_RE.search(message)
    if not m:
        return None
    raw = m.group("area").strip().lower()
    # Defensive: strip a leading article that slipped past the suffix
    # regex (case variants like "in die küche") so the name still resolves.
    dearticled = re.sub(
        r"^(?:the|a|an|der|die|das|dem|den)\s+", "", raw, flags=re.IGNORECASE
    )
    area_reg = ar.async_get(hass)
    for a in area_reg.async_list_areas():
        names = [(a.name or "").lower()]
        names.extend((al or "").lower() for al in (getattr(a, "aliases", None) or ()))
        if raw in names or dearticled in names:
            return a
    return None


def _caller_area_id(hass: HomeAssistant, device_id: str | None) -> str | None:
    """Resolve the area of the calling voice satellite, if any."""
    if not device_id:
        return None
    dev = dr.async_get(hass).async_get(device_id)
    if dev and dev.area_id:
        return dev.area_id
    for entry in er.async_entries_for_device(er.async_get(hass), device_id):
        if entry.area_id:
            return entry.area_id
    return None


def _entry_area_id(hass: HomeAssistant, entry: Any, dev_reg: Any) -> str | None:
    """Area of an entity entry, falling back to its device's area."""
    if entry.area_id:
        return entry.area_id
    if entry.device_id:
        dev = dev_reg.async_get(entry.device_id)
        return dev.area_id if dev else None
    return None
