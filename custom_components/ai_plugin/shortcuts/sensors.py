"""Sensor readings for "<attribute> in <area>" questions.

The catalogue of attributes the shortcut layer can answer without the
model, the phrase patterns that name an area, and the entry point
(``try_shortcut``) the orchestrator calls before the LLM path.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from homeassistant.core import HomeAssistant

from ..i18n import L
from .registry import _entities_in_area, _resolve_area
from .sun_time import _try_sun_shortcut, _try_time_shortcut

_LOGGER = logging.getLogger(__name__)

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

# Verbs that mark a message as a COMMAND, not a question. try_shortcut's
# answers (time, sun, sensor readings) are informational — firing them on
# a command swallows the action the user asked for.
_ACTION_VERB_RE = re.compile(
    r"\b(?:turn|switch|toggle|dim|brighten|set|start|stop|open|close|"
    r"play|pause|mach|schalt\w*|stell\w*|dreh\w*|starte?|"
    r"öffne\w*|schließ\w*|spiel\w*)\b",
    re.IGNORECASE,
)

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

    # Q&A shortcuts only: a message carrying an action verb is a COMMAND
    # ("turn on the heating, it's cold in the living room" / "mach es warm
    # im schlafzimmer") — answering it with a sensor reading swallows the
    # action. Long messages are compound requests; both go to the LLM.
    if len(message.split()) > 12 or _ACTION_VERB_RE.search(message):
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
