"""Deterministic pre-LLM shortcuts for common `<attr> in <area>` questions.

Small LLMs are unreliable at resolving descriptive sensor queries against
HA's registry — they often call get_entity with the full phrase ("humidity
in the living room"), get a miss, and tell the user "I can't find it"
instead of decomposing into area + attribute. Multi-sensor rooms also
trip the fuzzy retry (returns the first match, not the right one).

This module bypasses the LLM entirely for queries that match a short
catalogue of templates. When a pattern hits:
  1. Parse the area name out of the query.
  2. Resolve it against area_registry (case-insensitive, aliases, DE↔EN).
  3. Walk entities in the area, match by device_class (for sensors) or
     domain + attribute (for lights / climate).
  4. Return a short speech reply directly, skipping tool_loop.

The orchestrator invokes `try_shortcut` before the LLM path. If it
returns None, the normal tool loop runs as before. No prompt changes,
no model changes — a pure deterministic shortcut.
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
)

from .i18n import L

try:
    from homeassistant.components.homeassistant.exposed_entities import (
        async_should_expose,
    )
except ImportError:  # pragma: no cover — older HA cores
    async_should_expose = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)
_CONVERSATION_ASSISTANT = "conversation"


# Attribute → (domain, device_class, attribute_key, unit_hint, label).
# device_class may be None for domain-only matches (e.g. light → light).
# attribute_key is consulted when the primary value is an attribute
# (brightness on a light) rather than the state itself (sensor.state).
_ATTR_SPECS: dict[str, dict[str, Any]] = {
    "temperature": {"domain": "sensor", "device_class": "temperature", "label": "temperature", "unit_fallback": "°C"},
    "humidity": {"domain": "sensor", "device_class": "humidity", "label": "humidity", "unit_fallback": "%"},
    "air_quality": {"domain": "sensor", "device_classes": ("pm25", "pm10", "aqi", "volatile_organic_compounds"), "label": "air quality"},
    "co2": {"domain": "sensor", "device_class": "carbon_dioxide", "label": "CO₂"},
    "illuminance": {"domain": "sensor", "device_class": "illuminance", "label": "illuminance"},
    "pressure": {"domain": "sensor", "device_class": "atmospheric_pressure", "label": "pressure"},
    "power": {"domain": "sensor", "device_class": "power", "label": "power"},
    "energy": {"domain": "sensor", "device_class": "energy", "label": "energy"},
    "battery": {"domain": "sensor", "device_class": "battery", "label": "battery"},
    "brightness": {"domain": "light", "attribute": "brightness", "label": "brightness", "transform": "brightness_pct"},
}

# Keyword → canonical attribute. First match wins. Multilingual.
_KEYWORD_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:temperature|temp|warm|warmth|warmly|hot|cold)\b", re.IGNORECASE), "temperature"),
    (re.compile(r"\b(?:temperatur|wärme|warm|kalt)\b", re.IGNORECASE), "temperature"),
    (re.compile(r"\b(?:humidity|humid|moisture)\b", re.IGNORECASE), "humidity"),
    (re.compile(r"\b(?:feuchtigkeit|luftfeuchtigkeit|feucht)\b", re.IGNORECASE), "humidity"),
    (re.compile(r"\bair[\s-]?quality|pm2\.?5|pm10|aqi\b", re.IGNORECASE), "air_quality"),
    (re.compile(r"\b(?:luftqualität|feinstaub)\b", re.IGNORECASE), "air_quality"),
    (re.compile(r"\bco2|carbon[\s-]?dioxide\b", re.IGNORECASE), "co2"),
    (re.compile(r"\b(?:illuminance|lux|light\s+level)\b", re.IGNORECASE), "illuminance"),
    (re.compile(r"\b(?:helligkeit|beleuchtungsstärke)\b", re.IGNORECASE), "illuminance"),
    (re.compile(r"\b(?:pressure|atmospheric)\b", re.IGNORECASE), "pressure"),
    (re.compile(r"\b(?:luftdruck|druck)\b", re.IGNORECASE), "pressure"),
    (re.compile(r"\b(?:power\s+(?:usage|consumption|draw)|watts?|wattage)\b", re.IGNORECASE), "power"),
    (re.compile(r"\bstromverbrauch|leistung\b", re.IGNORECASE), "power"),
    (re.compile(r"\benergy\s+(?:usage|consumption)\b", re.IGNORECASE), "energy"),
    (re.compile(r"\bbattery\s+(?:level|percentage|percent)?\b", re.IGNORECASE), "battery"),
    (re.compile(r"\bakku(?:stand)?|batterie(?:stand)?\b", re.IGNORECASE), "battery"),
    (re.compile(r"\bbrightness\b", re.IGNORECASE), "brightness"),
]

# Query patterns. Each captures an area group. Order matters — more
# specific patterns first. All are case-insensitive.
#
# Examples we want to hit:
#   "what's the temperature in the bedroom?"
#   "what's the humidity in the living room?"
#   "how warm is the living room?"
#   "tell me the temperature of the bedroom"
#   "temperature reading for the bathroom"
#   "wie warm ist das wohnzimmer?"
#   "wie ist die luftfeuchtigkeit im schlafzimmer?"
_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "... in the <area>" / "... in <area>" (EN)
    re.compile(r"\bin\s+(?:the\s+)?(?P<area>[a-zA-Z][\w \-]+?)\s*[\?\.!]*\s*$", re.IGNORECASE),
    # "... of the <area>"
    re.compile(r"\bof\s+(?:the\s+)?(?P<area>[a-zA-Z][\w \-]+?)\s*[\?\.!]*\s*$", re.IGNORECASE),
    # "... for the <area>" / "... reading for <area>"
    re.compile(r"\bfor\s+(?:the\s+)?(?P<area>[a-zA-Z][\w \-]+?)\s*[\?\.!]*\s*$", re.IGNORECASE),
    # "how warm / humid / bright is the <area>"
    re.compile(r"\bhow\s+\w+\s+is\s+(?:the\s+)?(?P<area>[a-zA-Z][\w \-]+?)\s*[\?\.!]*\s*$", re.IGNORECASE),
    # German "... im <area>" / "... in der/im <area>"
    re.compile(r"\bim\s+(?P<area>[a-zA-ZäöüÄÖÜß][\w \-]+?)\s*[\?\.!]*\s*$", re.IGNORECASE),
    re.compile(r"\bin\s+(?:der|dem)\s+(?P<area>[a-zA-ZäöüÄÖÜß][\w \-]+?)\s*[\?\.!]*\s*$", re.IGNORECASE),
    # "wie <adj> ist (das|die|der) <area>"
    re.compile(r"\bwie\s+\w+\s+ist\s+(?:der|die|das)\s+(?P<area>[a-zA-ZäöüÄÖÜß][\w \-]+?)\s*[\?\.!]*\s*$", re.IGNORECASE),
)


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

# Stop-words a user might include after the area that we want to trim.
_AREA_TAIL_NOISE = re.compile(
    r"\s+(?:today|now|right\s+now|currently|at\s+the\s+moment|please|gerade|jetzt)\s*$",
    re.IGNORECASE,
)


def _detect_attribute(message: str) -> str | None:
    for rx, attr in _KEYWORD_MAP:
        if rx.search(message):
            return attr
    return None


def _extract_area_name(message: str) -> str | None:
    for rx in _QUERY_PATTERNS:
        m = rx.search(message)
        if not m:
            continue
        area = m.group("area").strip()
        area = _AREA_TAIL_NOISE.sub("", area).strip()
        # Trim trailing articles / small words that leaked in.
        area = re.sub(r"^(?:the|a|an|der|die|das|dem)\s+", "", area, flags=re.IGNORECASE)
        if area:
            return area
    return None


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
        if async_should_expose is not None:
            try:
                if not async_should_expose(
                    hass, _CONVERSATION_ASSISTANT, entry.entity_id
                ):
                    continue
            except Exception:  # noqa: BLE001 — never block on registry quirks
                _LOGGER.debug(
                    "AI Plugin: should_expose check failed for %s", entry.entity_id,
                    exc_info=True,
                )
        results.append(entry)
    return results


def _round_numeric(val: Any, decimals: int = 1) -> Any:
    """Round to N decimals if value is a float-parseable number; else return as-is.

    Avoids speech artefacts like '25.0617828369141°C' from raw sensor floats.
    Integers stay integer-shaped ('42' not '42.0'). Strings that don't parse
    as numbers pass through unchanged ('on', 'open', etc).
    """
    if val is None:
        return val
    try:
        f = float(val)
    except (TypeError, ValueError):
        return val
    if f == int(f):
        return int(f)
    return round(f, decimals)


def _format_state(state_obj: Any, spec: dict, lang: str = "en") -> str | None:
    """Render a state object into a localized speech string."""
    if state_obj is None:
        return None
    label_key = spec["label"]
    label = L.label(label_key, lang)
    transform = spec.get("transform")

    if "attribute" in spec:
        val = state_obj.attributes.get(spec["attribute"])
        if val is None:
            return None
        if transform == "brightness_pct":
            try:
                pct = round(int(val) / 255 * 100)
                return L.template("attr_pct", lang, label=label, val=pct)
            except Exception:  # noqa: BLE001
                return L.template("attr_state", lang, label=label,
                                  val=_round_numeric(val), unit="")
        return L.template("attr_state", lang, label=label,
                          val=_round_numeric(val), unit="")

    raw = state_obj.state
    if raw in (None, "", "unknown", "unavailable", "none"):
        return None
    unit = state_obj.attributes.get("unit_of_measurement") or spec.get("unit_fallback") or ""
    return L.template("attr_state", lang, label=label,
                      val=_round_numeric(raw), unit=unit)


def _pick_best_sensor(entities: list[Any], hass: HomeAssistant, spec: dict) -> Any | None:
    """From entities in an area, pick the sensor whose device_class matches spec."""
    wanted = spec.get("device_class")
    wanted_many = spec.get("device_classes")
    domain = spec["domain"]

    candidates: list[tuple[int, Any]] = []  # (score, state_obj)
    for entry in entities:
        eid = entry.entity_id
        if not eid.startswith(f"{domain}."):
            continue
        state = hass.states.get(eid)
        if state is None:
            continue
        ent_dc = (
            state.attributes.get("device_class")
            or getattr(entry, "device_class", None)
            or getattr(entry, "original_device_class", None)
        )
        if wanted and ent_dc == wanted:
            score = 0 if state.state not in (None, "", "unknown", "unavailable") else 10
            candidates.append((score, state))
        elif wanted_many and ent_dc in wanted_many:
            score = 0 if state.state not in (None, "", "unknown", "unavailable") else 10
            candidates.append((score, state))
        elif domain == "light" and not wanted and not wanted_many:
            score = 0 if state.state == "on" else 5
            candidates.append((score, state))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _pick_climate_temperature(entities: list[Any], hass: HomeAssistant) -> tuple[Any, Any] | None:
    """Fallback for temperature queries: read current_temperature from climate.* in area.

    Many rooms only have a thermostat (e.g. Tado, Nest) and no separate
    temperature sensor exposed. The thermostat publishes the room's actual
    temperature in its current_temperature attribute. Returns
    (state_obj, current_temperature_value) or None.
    """
    for entry in entities:
        if not entry.entity_id.startswith("climate."):
            continue
        state = hass.states.get(entry.entity_id)
        if state is None:
            continue
        val = state.attributes.get("current_temperature")
        if val in (None, "", "unknown", "unavailable"):
            continue
        return state, val
    return None


def _pick_climate_humidity(entities: list[Any], hass: HomeAssistant) -> tuple[Any, Any] | None:
    """Fallback for humidity queries: read current_humidity from climate.* in area.

    Mirrors _pick_climate_temperature for rooms whose thermostat exposes
    current_humidity but where no dedicated humidity sensor is exposed
    to the conversation assistant.
    """
    for entry in entities:
        if not entry.entity_id.startswith("climate."):
            continue
        state = hass.states.get(entry.entity_id)
        if state is None:
            continue
        val = state.attributes.get("current_humidity")
        if val in (None, "", "unknown", "unavailable"):
            continue
        return state, val
    return None


def _try_sun_shortcut(hass: HomeAssistant, message: str, lang: str = "en") -> str | None:
    """Deterministic reply for sun/daylight questions, in the user's language.

    Reads ``sun.sun`` directly. Bypasses the LLM which often refuses these
    queries. ``lang`` selects the keyword regex set and the response
    template; English is the universal fallback.
    """
    msg_lower = (message or "").lower()
    sun_set_re = L.keyword_re("sun_set", lang)
    sun_rise_re = L.keyword_re("sun_rise", lang)
    sun_dark_re = L.keyword_re("sun_dark", lang)
    sun_is_up_re = L.keyword_re("sun_is_up", lang)

    matched = (
        (sun_set_re and sun_set_re.search(msg_lower))
        or (sun_rise_re and sun_rise_re.search(msg_lower))
        or (sun_dark_re and sun_dark_re.search(msg_lower))
        or (sun_is_up_re and sun_is_up_re.search(msg_lower))
    )
    if not matched:
        return None

    state = hass.states.get("sun.sun")
    if state is None:
        return None

    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        tz_name = (getattr(hass.config, "time_zone", None) or "").strip()
        tz = ZoneInfo(tz_name) if tz_name else None
    except Exception:  # noqa: BLE001
        tz = None

    def _fmt(iso: str | None) -> str | None:
        if not iso:
            return None
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if tz is not None:
                dt = dt.astimezone(tz)
            return dt.strftime("%H:%M")
        except Exception:  # noqa: BLE001
            return None

    attrs = state.attributes or {}
    next_setting = _fmt(attrs.get("next_setting"))
    next_rising = _fmt(attrs.get("next_rising"))
    is_up = state.state == "above_horizon"

    # Boolean queries first so 'dark' / 'fait-il nuit' don't shadow the
    # sunset time branch.
    if (sun_dark_re and sun_dark_re.search(msg_lower)) or (
        sun_is_up_re and sun_is_up_re.search(msg_lower)
    ):
        if is_up and next_setting:
            return L.template("sun_is_up", lang, time=next_setting)
        if not is_up and next_rising:
            return L.template("sun_is_down", lang, time=next_rising)

    if sun_set_re and sun_set_re.search(msg_lower) and next_setting:
        _LOGGER.info("AI Plugin shortcut hit: sunset → %s (lang=%s)", next_setting, lang)
        return L.template("sun_set_at", lang, time=next_setting)
    if sun_rise_re and sun_rise_re.search(msg_lower) and next_rising:
        _LOGGER.info("AI Plugin shortcut hit: sunrise → %s (lang=%s)", next_rising, lang)
        return L.template("sun_rise_at", lang, time=next_rising)

    return None


def _try_time_shortcut(hass: HomeAssistant, message: str, lang: str) -> str | None:
    """Answer 'what time is it' from the HA clock — no LLM round-trip.

    The LLM answers this correctly from the [CURRENT TIME] block, but at
    5-8s per turn for a deterministic fact. Date questions stay with the
    LLM (localized weekday/month names aren't worth the i18n surface).
    """
    time_re = L.keyword_re("time_now", lang)
    if time_re is None or not time_re.search(message.lower()):
        return None
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        tz_name = (getattr(hass.config, "time_zone", None) or "").strip()
        now = datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.now().astimezone()
        stamp = now.strftime("%H:%M")
    except Exception:  # noqa: BLE001
        _LOGGER.debug("AI Plugin time shortcut: clock read failed", exc_info=True)
        return None
    _LOGGER.info("AI Plugin shortcut hit: time → %s (lang=%s)", stamp, lang)
    return L.template("time_is", lang, time=stamp)


def try_shortcut(hass: HomeAssistant, message: str, *, lang: str = "en") -> str | None:
    """Return a deterministic reply for supported patterns, else None.

    Entry point used by the orchestrator before falling through to the
    LLM tool loop. Always returns either:
      - a short speech string (shortcut hit) → orchestrator persists it
        and skips the tool loop entirely, OR
      - None (no pattern matched OR data not present) → normal path.

    Misses intentionally fall through rather than apologising so the
    LLM can still try via search_entities / web_search.
    """
    if not message or not hass:
        return None

    time_reply = _try_time_shortcut(hass, message, lang)
    if time_reply:
        return time_reply

    sun_reply = _try_sun_shortcut(hass, message, lang)
    if sun_reply:
        return sun_reply

    attr_key = _detect_attribute(message)
    if attr_key is None:
        return None
    spec = _ATTR_SPECS.get(attr_key)
    if spec is None:
        return None

    raw_area = _extract_area_name(message)
    if not raw_area:
        return None

    area = _resolve_area(hass, raw_area)
    if area is None:
        return None

    entities = _entities_in_area(hass, area.id)
    if not entities:
        return None

    # For light brightness we target the brightest-on light, else any.
    if attr_key == "brightness":
        best = None
        for entry in entities:
            if not entry.entity_id.startswith("light."):
                continue
            s = hass.states.get(entry.entity_id)
            if s and s.state == "on" and s.attributes.get("brightness") is not None:
                best = s
                break
        if best is None:
            return None
        reply = _format_state(best, spec, lang=lang)
        if reply:
            _LOGGER.info(
                "AI Plugin shortcut hit: %s in %s → %s", attr_key, area.name, best.entity_id
            )
            return reply
        return None

    best = _pick_best_sensor(entities, hass, spec)
    if best is not None:
        reply = _format_state(best, spec, lang=lang)
        if reply:
            _LOGGER.info(
                "AI Plugin shortcut hit: %s in %s → %s", attr_key, area.name, best.entity_id
            )
            return reply

    # Temperature fallback: rooms with only a thermostat (no exposed
    # temperature sensor). Read current_temperature from climate.*.
    if attr_key == "temperature":
        fallback = _pick_climate_temperature(entities, hass)
        if fallback:
            state, val = fallback
            unit = state.attributes.get("unit_of_measurement") or spec.get("unit_fallback") or "°C"
            _LOGGER.info(
                "AI Plugin shortcut hit: temperature in %s → %s (climate fallback)",
                area.name, state.entity_id,
            )
            return L.template(
                "attr_state", lang,
                label=L.label("temperature", lang),
                val=_round_numeric(val), unit=unit,
            )

    # Humidity fallback: same pattern, climate.* current_humidity.
    if attr_key == "humidity":
        fallback = _pick_climate_humidity(entities, hass)
        if fallback:
            state, val = fallback
            _LOGGER.info(
                "AI Plugin shortcut hit: humidity in %s → %s (climate fallback)",
                area.name, state.entity_id,
            )
            return L.template(
                "attr_humidity_fb", lang,
                label=L.label("humidity", lang),
                val=_round_numeric(val),
            )

    return None


# ── media playback shortcut ────────────────────────────────────────────────

_MEDIA_TRIGGERS: list[tuple[str, re.Pattern[str]]] = [
    # Order matters: match more specific phrases first so bare "weiter"
    # doesn't shadow "weiter spielen" — though both map to resume anyway.
    (
        "resume",
        re.compile(
            r"\b(?:resume|unpause|continue|fortsetzen|weiter\s*(?:spiel\w*|h[öo]r\w*|machen)?|spiel(?:e)?\s+weiter)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "next",
        re.compile(
            r"\b(?:next(?:\s+(?:track|song))?|skip(?:\s+(?:this\s+)?(?:song|track))?|"
            r"n[äa]chst(?:er|es)?(?:\s+(?:song|titel|track|st[üu]ck))?|"
            r"[üu]berspring(?:e|en)?|skippen)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "previous",
        re.compile(
            r"\b(?:previous(?:\s+(?:track|song))?|prev|go\s+back|"
            r"zur[üu]ck(?:\s+zum\s+vorherig\w*)?|vorherig(?:er|es)?(?:\s+(?:song|titel|track))?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "pause",
        re.compile(r"\b(?:paus(?:e|ier\w*))\b", re.IGNORECASE),
    ),
    # Volume — set (with percent capture) must precede up/down; unmute
    # must precede mute so the prefix isn't shadowed.
    (
        "volume_set",
        re.compile(
            r"\b(?:volume|lautst[äa]rke)\s+(?:to|auf)\s+(?P<pct>\d{1,3})\s*"
            r"(?:%|percent|prozent)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "volume_up",
        re.compile(
            r"\b(?:volume\s+up|turn\s+(?:it|the\s+volume)\s+up|louder|lauter)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "volume_down",
        re.compile(
            r"\b(?:volume\s+down|turn\s+(?:it|the\s+volume)\s+down|"
            r"quieter|softer|leiser)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "unmute",
        re.compile(
            r"\b(?:unmute|ton\s+(?:wieder\s+)?an|stumm(?:schaltung)?\s+aus\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "mute",
        re.compile(r"\b(?:mute|stumm(?:schalten)?)\b", re.IGNORECASE),
    ),
    (
        "stop",
        re.compile(
            r"\b(?:stop(?:\s+(?:the\s+)?music)?|stopp(?:\s+die\s+musik)?|halt(?:\s+die\s+musik)?)\b",
            re.IGNORECASE,
        ),
    ),
]

_AREA_SUFFIX_RE = re.compile(
    r"\b(?:in\s+(?:the\s+)?|im\s+|in\s+der\s+|in\s+dem\s+|in\s+der\s+the\s+)"
    r"(?P<area>[\w\s\-]+?)\s*[.!?]?\s*$",
    re.IGNORECASE,
)

_MEDIA_SERVICE_MAP = {
    "pause": "media_pause",
    "resume": "media_play",
    "next": "media_next_track",
    "previous": "media_previous_track",
    "stop": "media_stop",
    "volume_set": "volume_set",
    "volume_up": "volume_up",
    "volume_down": "volume_down",
    "mute": "volume_mute",
    "unmute": "volume_mute",
}

_MEDIA_RELEVANT_STATES = {
    "pause": {"playing"},
    "resume": {"paused", "idle"},
    "next": {"playing", "paused"},
    "previous": {"playing", "paused"},
    "stop": {"playing", "paused"},
    "volume_set": {"playing", "paused", "on"},
    "volume_up": {"playing", "paused", "on"},
    "volume_down": {"playing", "paused", "on"},
    "mute": {"playing", "paused", "on"},
    "unmute": {"playing", "paused", "on"},
}


def _detect_media_command(message: str) -> tuple[str, "re.Match[str]"] | None:
    for cmd, rx in _MEDIA_TRIGGERS:
        if m := rx.search(message):
            return cmd, m
    return None


def _extract_area_from_media_message(hass: HomeAssistant, message: str):
    """Find an area mentioned via 'in <area>' / 'im <area>' suffix."""
    m = _AREA_SUFFIX_RE.search(message)
    if not m:
        return None
    raw = m.group("area").strip().lower()
    area_reg = ar.async_get(hass)
    for a in area_reg.async_list_areas():
        if (a.name or "").lower() == raw:
            return a
        for al in (getattr(a, "aliases", None) or ()):
            if (al or "").lower() == raw:
                return a
    return None


async def async_try_media_shortcut(
    hass: HomeAssistant, message: str, *, lang: str = "en"
) -> tuple[bool, str] | None:
    """Pre-LLM shortcut for media playback commands.

    Small LLMs (qwen3.5-9b on the 8 GB tier) won't reliably translate
    terse phrasings like "next track" or "pause music" into a
    media_command tool call — they tend to produce a confident-sounding
    prose reply ("Skipping to the next track in the hobby room") with
    zero actual playback change. This shortcut pattern-matches common
    English/German playback verbs and dispatches the matching
    media_player service directly, bypassing the LLM.

    Return value: ``(handled, reply)`` when the message was consumed
    (``reply=""`` for TTS suppression), or ``None`` to fall through to
    the normal LLM path.
    """
    if not message:
        return None
    msg = message.strip()
    if not msg or len(msg.split()) > 10:
        return None

    detected = _detect_media_command(msg)
    if detected is None:
        return None
    cmd, match = detected

    area = _extract_area_from_media_message(hass, msg)

    relevant_states = _MEDIA_RELEVANT_STATES[cmd]
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)

    target_ids: list[str] = []
    for eid, entry in ent_reg.entities.items():
        if not eid.startswith("media_player."):
            continue
        if area is not None:
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if area_id != area.id:
                continue
        st = hass.states.get(eid)
        if st and st.state in relevant_states:
            target_ids.append(eid)

    if not target_ids:
        # Nothing matching — fall through so the LLM can produce a
        # useful "nothing is playing" reply rather than a silent no-op.
        return None

    target_ids.sort()
    service = _MEDIA_SERVICE_MAP[cmd]
    data: dict[str, Any] = {}
    if cmd in ("mute", "unmute"):
        data["is_volume_muted"] = cmd == "mute"
    elif cmd == "volume_set":
        pct = int(match.group("pct"))
        if not 0 <= pct <= 100:
            return None
        data["volume_level"] = pct / 100
    try:
        await hass.services.async_call(
            "media_player",
            service,
            data or None,
            target={"entity_id": target_ids},
            blocking=True,
        )
    except Exception:  # noqa: BLE001
        _LOGGER.exception(
            "AI Plugin media shortcut: %s on %s failed", cmd, target_ids
        )
        return None

    _LOGGER.info(
        "AI Plugin media shortcut: %s → %s on %s",
        cmd,
        service,
        ", ".join(target_ids),
    )
    return (True, "")


# ── On/off shortcut for a single named device (all i18n languages) ─────────────

# HassTurnOn-equivalent domains a deterministic on/off may actuate.
_ACTION_DOMAINS = ("switch", "light", "fan", "input_boolean", "humidifier", "siren")


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


def _resolve_named_entity(
    hass: HomeAssistant,
    name: str,
    *,
    device_id: str | None = None,
    domains: tuple[str, ...] = _ACTION_DOMAINS,
) -> str | None:
    """Resolve a spoken device name to ONE exposed, actuatable entity_id.

    Matches name / original_name / aliases / friendly_name (exact first,
    then substring), restricted to exposed entities in ``domains``.
    When the match is ambiguous, the caller's area breaks the tie; if it
    still can't, returns None so the LLM can disambiguate rather than the
    shortcut actuating the wrong device.
    """
    needle = name.strip().lower().strip(".?!,")
    if not needle:
        return None
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)
    caller = _caller_area_id(hass, device_id)

    exact: list[tuple[str, str | None]] = []
    substr: list[tuple[str, str | None]] = []
    for eid, entry in ent_reg.entities.items():
        if eid.split(".", 1)[0] not in domains:
            continue
        if async_should_expose is not None:
            try:
                if not async_should_expose(hass, _CONVERSATION_ASSISTANT, eid):
                    continue
            except Exception:  # noqa: BLE001 — fail open; never let one entity abort resolution
                pass
        raw = [entry.name, entry.original_name, *(entry.aliases or ())]
        st = hass.states.get(eid)
        if st:
            raw.append(st.attributes.get("friendly_name"))
        # entry.name can be a non-str sentinel (HA ComputedNameType) and
        # state attributes are arbitrary types — keep only real strings.
        cands = [c.lower() for c in raw if isinstance(c, str) and c]
        if not cands:
            continue
        area = _entry_area_id(hass, entry, dev_reg)
        if needle in cands:
            exact.append((eid, area))
        elif any(needle in c for c in cands):
            substr.append((eid, area))

    for pool in (exact, substr):
        if not pool:
            continue
        if len(pool) == 1:
            return pool[0][0]
        if caller:
            same = [eid for eid, area in pool if area == caller]
            if len(same) == 1:
                return same[0]
        return None  # ambiguous, no clean winner → fall through to the LLM
    return None


async def async_try_action_shortcut(
    hass: HomeAssistant,
    message: str,
    *,
    lang: str = "en",
    device_id: str | None = None,
) -> tuple[bool, str] | None:
    """Pre-LLM shortcut: turn a single named device on/off, in any language.

    Small models mis-route phrasings like "switch TV on" or German
    "schalte den Fernseher ein" (treating "switch"/"schalte" as a domain,
    or only handling "turn on X" word order). This matches the per-language
    action_on / action_off regexes from i18n, resolves the captured device
    name to one exposed entity, and dispatches homeassistant.turn_on/off
    directly. 'switch'/'schalte' is the verb here, never a domain.

    Returns (handled, reply) with reply="" for TTS suppression, or None to
    fall through to the LLM (no match / unresolved / ambiguous device).
    """
    if not message:
        return None
    msg = message.strip().rstrip(".?!").lower()
    if not msg or len(msg.split()) > 7:
        return None

    # (pattern key, service domain, service, allowed entity domains).
    # off/close before on/open is a safe tiebreak; cover verbs are checked
    # first so "open the blinds" never reaches the on/off patterns.
    kinds = (
        ("action_close", "cover", "close_cover", ("cover",)),
        ("action_open", "cover", "open_cover", ("cover",)),
        ("action_off", "homeassistant", "turn_off", _ACTION_DOMAINS),
        ("action_on", "homeassistant", "turn_on", _ACTION_DOMAINS),
    )
    matched = None
    for key, service_domain, service, domains in kinds:
        for rx in L.pattern_list(key, lang):
            m = rx.match(msg)
            if m and (m.group("name") or "").strip():
                matched = (service_domain, service, domains, m.group("name").strip())
                break
        if matched:
            break
    if matched is None:
        return None
    service_domain, service, domains, name = matched

    entity_id = _resolve_named_entity(
        hass, name, device_id=device_id, domains=domains
    )
    if not entity_id:
        return None

    try:
        await hass.services.async_call(
            service_domain,
            service,
            {"entity_id": entity_id},
            blocking=True,
            context=Context(),
        )
    except Exception:  # noqa: BLE001
        _LOGGER.exception(
            "AI Plugin action shortcut: %s on %s failed", service, entity_id
        )
        return None

    _LOGGER.info(
        "AI Plugin action shortcut: %s.%s → %s (lang=%s)",
        service_domain, service, entity_id, lang,
    )
    return (True, "")
