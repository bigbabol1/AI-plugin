"""Whole-room and whole-home sweeps: "switch all the lights off".

The most common voice command and the one small models fumble worst —
they promise the action and never call a tool. The plural-noun + on/off
shape is unambiguous, so it never reaches the model.
"""

from __future__ import annotations

import logging
import re

from homeassistant.core import Context, HomeAssistant

from ..i18n import L
from . import registry
from .actions import _match_action_intent
from .registry import (
    _AREA_SUFFIX_RE,
    _caller_area_id,
    _entry_area_id,
    _extract_area_from_media_message,
    dr,
    er,
)

_LOGGER = logging.getLogger(__name__)

# Plural domain nouns for the whole-area sweep shortcut, per domain.
# PLURAL ONLY, deliberately: a singular "das Licht aus" stays on the
# single-named-device path (unchanged behaviour), while "alle Lichter aus"
# is unambiguously a domain sweep.
_SWEEP_DOMAIN_NOUNS: tuple[tuple[str, str], ...] = (
    (
        "light",
        r"lights|lamps|lichter|lampen|lumi[eè]res|luces|l[aá]mparas|"
        r"l[aâ]mpadas|luzes|[sś]wiat[lł]a",
    ),
    (
        "fan",
        r"fans|ventilatoren|ventilateurs|ventiladores|ventoinhas|wentylatory",
    ),
    (
        "switch",
        r"switches|sockets|plugs|steckdosen|prisen|prises|enchufes|tomadas|"
        r"gniazdka",
    ),
)

# i18n label key per domain, for the whole-home spoken confirmation.
_SWEEP_LABEL_KEYS = {"light": "lights", "fan": "fans", "switch": "switches"}

_SWEEP_DOMAIN_RES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (domain, re.compile(rf"^(?:{alt})$", re.IGNORECASE))
    for domain, alt in _SWEEP_DOMAIN_NOUNS
)

# "all"/"every" as a determiner in front of the noun. Narrower on purpose
# than ha_local._USER_SAID_ALL_RE, which scans the whole message: here the
# word must actually qualify the noun ("all lights", "alle Lichter").
_SWEEP_ALL_PREFIX_RE = re.compile(
    r"^(?:all|every|alle[srnm]?|s[äa]mtliche[srn]?|tous|toutes|todos|todas|"
    r"wszystkie|wszystkich)\s+",
    re.IGNORECASE,
)

_SWEEP_ARTICLE_RE = re.compile(
    r"^(?:the|der|die|das|den|dem|les|le|la|l['’]|los|las|el|os|as|o|a)\s+",
    re.IGNORECASE,
)

def _parse_sweep_noun(name: str) -> tuple[str, bool] | None:
    """('light', said_all) when ``name`` is a plural domain noun, else None.

    Strips leading articles and an "all"/"alle" determiner (in either
    order, e.g. "all the lights" / "die alle Lichter").
    """
    rest = name.strip().strip(".?!,")
    said_all = False
    for _ in range(2):
        stripped = _SWEEP_ALL_PREFIX_RE.sub("", rest, count=1)
        if stripped != rest:
            said_all = True
            rest = stripped
        rest = _SWEEP_ARTICLE_RE.sub("", rest, count=1)
    rest = rest.strip()
    if not rest:
        return None
    for domain, rx in _SWEEP_DOMAIN_RES:
        if rx.match(rest):
            return (domain, said_all)
    return None

def _sweep_entities(
    hass: HomeAssistant, domain: str, area_id: str | None
) -> list[str]:
    """Exposed entity_ids of ``domain`` in ``area_id`` (all areas if None)."""
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)
    out: list[str] = []
    for eid, entry in ent_reg.entities.items():
        if not eid.startswith(f"{domain}."):
            continue
        if area_id is not None and _entry_area_id(hass, entry, dev_reg) != area_id:
            continue
        if not registry.is_exposed(hass, eid):
            continue
        out.append(eid)
    out.sort()
    return out

async def async_try_domain_sweep_shortcut(
    hass: HomeAssistant,
    message: str,
    *,
    lang: str = "en",
    device_id: str | None = None,
) -> tuple[bool, str] | None:
    """Pre-LLM shortcut: switch every light/fan/socket in a scope on or off.

    "Switch all lights off" is the single most common voice command and the
    one small local models fumble worst: they answer with a promise ("I'll
    turn off all the lights in your home now!") and never call a tool, or
    they call set_area_state with no area — which means "this room only".
    Either way the user's flat stays lit. The plural-noun + on/off shape is
    unambiguous, so resolve it here and never involve the model.

    Scope precedence mirrors ha_local.set_area_state:
      1. a room named in the utterance ("all lights in the kitchen")
      2. an explicit "all"/"alle" determiner → every area
      3. neither → the calling satellite's room
    With no room, no "all" and no satellite (text chat / REST) the scope is
    genuinely ambiguous, so fall through to the LLM instead of guessing.

    Returns (handled, reply), or None. A whole-home sweep gets a short spoken
    confirmation — the user cannot see the rooms they are not in, so silence
    there is indistinguishable from "nothing happened". Room-scoped and
    caller-room sweeps stay silent: the change is right in front of them.
    """
    if not message:
        return None
    msg = message.strip().rstrip(".?!").lower()
    if not msg or len(msg.split()) > 8:
        return None

    matched = _match_action_intent(msg, lang, on_off_only=True)
    if matched is None:
        return None
    _, service, _, name = matched

    # A trailing "in <room>" belongs to the scope, not to the noun. The
    # helper is media-named but purely generic ("in <area>" suffix → area).
    area = _extract_area_from_media_message(hass, name)
    if area is not None:
        name = _AREA_SUFFIX_RE.sub("", name).strip()

    parsed = _parse_sweep_noun(name)
    if parsed is None:
        return None
    domain, said_all = parsed

    if area is not None:
        area_id: str | None = area.id
        scope = area.name
    elif said_all:
        area_id = None
        scope = "all areas"
    else:
        area_id = _caller_area_id(hass, device_id)
        if not area_id:
            return None
        scope = area_id

    ids = _sweep_entities(hass, domain, area_id)
    if not ids:
        # Nothing exposed to act on — let the LLM explain rather than
        # silently swallowing the command.
        return None

    try:
        await hass.services.async_call(
            "homeassistant",
            service,
            {"entity_id": ids},
            blocking=True,
            context=Context(),
        )
    except Exception:  # noqa: BLE001
        _LOGGER.exception(
            "AI Plugin sweep shortcut: %s on %d %s(s) failed",
            service, len(ids), domain,
        )
        return None

    _LOGGER.info(
        "AI Plugin sweep shortcut: %s %d %s(s) in %s (lang=%s)",
        service, len(ids), domain, scope, lang,
    )
    if area_id is None:
        key = "sweep_all_on" if service == "turn_on" else "sweep_all_off"
        label = L.label(_SWEEP_LABEL_KEYS[domain], lang)
        return (True, L.template(key, lang, label=label))
    return (True, "")
