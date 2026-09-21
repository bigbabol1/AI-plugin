"""Turning ONE named device on or off, in any shipped language.

The per-language action_on / action_off / action_open / action_close
patterns live in i18n; this module resolves the captured device name to a
single exposed entity and calls the service.
"""

from __future__ import annotations

import logging

from homeassistant.core import Context, HomeAssistant

from ..i18n import L
from . import registry
from .registry import _caller_area_id, _entry_area_id, dr, er

_LOGGER = logging.getLogger(__name__)

# ── On/off shortcut for a single named device (all i18n languages) ─────────────

# HassTurnOn-equivalent domains a deterministic on/off may actuate.
_ACTION_DOMAINS = ("switch", "light", "fan", "input_boolean", "humidifier", "siren")

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
        if not registry.is_exposed(hass, eid):
            continue
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

def _match_action_intent(
    msg: str, lang: str, *, on_off_only: bool = False
) -> tuple[str, str, tuple[str, ...], str] | None:
    """Match an action command against the per-language i18n patterns.

    Returns (service_domain, service, allowed entity domains, spoken object
    name) or None. off/close before on/open is a safe tiebreak; cover verbs
    are checked first so "open the blinds" never reaches the on/off
    patterns. ``on_off_only`` skips the cover verbs for callers that only
    handle turn_on/turn_off.
    """
    kinds = (
        ("action_close", "cover", "close_cover", ("cover",)),
        ("action_open", "cover", "open_cover", ("cover",)),
        ("action_off", "homeassistant", "turn_off", _ACTION_DOMAINS),
        ("action_on", "homeassistant", "turn_on", _ACTION_DOMAINS),
    )
    for key, service_domain, service, domains in kinds:
        if on_off_only and key in ("action_open", "action_close"):
            continue
        for rx in L.pattern_list(key, lang):
            m = rx.match(msg)
            if m and (m.group("name") or "").strip():
                return (service_domain, service, domains, m.group("name").strip())
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

    matched = _match_action_intent(msg, lang)
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
