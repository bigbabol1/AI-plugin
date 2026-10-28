"""Ringing a person's phone: "where is my phone".

The phone is found through Home Assistant's own links, never through
exposure: a person entity lists its device trackers, a tracker made by the
Companion app belongs to a mobile_app config entry, and that entry names
the phone (its notify service is ``notify.mobile_app_<device name>``) and
its operating system. Only phones linked to a person can ring, and only
with the "Find my phone" option on.

Android speaks a short line on the alarm stream at full volume, which
sounds on silent and restores the volume afterwards. iOS offers only a
critical alert: one short sound that plays on mute and through Focus once
the app has the Critical Alerts permission. Either is sent a few times, a
few seconds apart, so the sound can be followed.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from homeassistant.core import Context, HomeAssistant
from homeassistant.util import slugify

from ..i18n import L
from .registry import er

_LOGGER = logging.getLogger(__name__)

RINGS = 3
RING_GAP_S = 6.0

RING_PHONE_TOOL = "ring_phone"
RING_PHONE_SCHEMA = {
    "type": "function",
    "function": {
        "name": RING_PHONE_TOOL,
        "description": (
            "Make a person's phone ring loudly so it can be found, also when it is "
            "on silent. Call it when the user asks where their phone is, cannot "
            "find it, or wants it to ring. Without a name it rings the speaker's "
            "own phone. Its result says what happened; never claim a phone rang "
            "unless the result says so."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "person": {
                    "type": "string",
                    "description": "Whose phone, if the user named someone else.",
                }
            },
        },
    },
}
_APPLE_OS = ("ios", "ipados", "macos")
_RING_TAG = "ai_plugin_find_phone"
# Words a "<name>'s phone" pattern can capture that are not a name.
_NOT_A_NAME = {
    "my", "the", "a", "your", "mein", "meine", "das", "dein", "mon", "le", "ton",
    "mi", "el", "tu", "mijn", "de", "je", "meu", "o", "teu",
}


@dataclass(frozen=True)
class Phone:
    """One Companion-app device linked to a person."""

    service: str  # notify service name, without the "notify." domain
    name: str
    apple: bool
    tracker: str


def match_find_phone(message: str, lang: str) -> str | None:
    """The owner named in a find-my-phone request.

    Returns "" for the speaker's own phone ("where is my phone"), the
    spoken name for someone else's ("where is Anna's phone"), or None when
    the message is not such a request.
    """
    msg = re.sub(r"\s+", " ", (message or "").strip(" .?!¿¡").lower())
    if not msg or len(msg.split()) > 9:
        return None
    for rx in L.pattern_list("find_phone", lang):
        m = rx.match(msg)
        if not m:
            continue
        owner = (m.groupdict().get("owner") or "").strip(" '’")
        if owner in _NOT_A_NAME:
            owner = ""
        return owner
    return None


def phones_of(hass: HomeAssistant, person_id: str) -> list[Phone]:
    """The Companion-app devices among a person's device trackers."""
    state = hass.states.get(person_id)
    if state is None:
        return []
    ent_reg = er.async_get(hass)
    phones: list[Phone] = []
    for tracker in state.attributes.get("device_trackers") or ():
        entry = ent_reg.async_get(tracker)
        if entry is None or entry.platform != "mobile_app":
            continue
        config = hass.config_entries.async_get_entry(entry.config_entry_id)
        if config is None:
            continue
        device_name = str(config.data.get("device_name") or "")
        service = f"mobile_app_{slugify(device_name)}"
        if not device_name or not hass.services.has_service("notify", service):
            _LOGGER.debug("AI Plugin find phone: no notify service for %s", tracker)
            continue
        os_name = str(config.data.get("os_name") or "").lower()
        phones.append(Phone(service, device_name, os_name in _APPLE_OS, tracker))
    return phones


def people_with_phones(hass: HomeAssistant) -> dict[str, str]:
    """person_id → friendly name, for every person with a ringable phone."""
    return {
        state.entity_id: str(state.attributes.get("friendly_name") or state.entity_id)
        for state in hass.states.async_all("person")
        if phones_of(hass, state.entity_id)
    }


def person_for_user(hass: HomeAssistant, user_id: str | None) -> str | None:
    """The person entity of an HA user, or of a voice-ID memory key."""
    if not user_id:
        return None
    if user_id.startswith("person_"):
        candidate = "person." + user_id[len("person_"):]
        return candidate if hass.states.get(candidate) is not None else None
    for state in hass.states.async_all("person"):
        if state.attributes.get("user_id") == user_id:
            return state.entity_id
    return None


def resolve_owner(
    hass: HomeAssistant,
    owner: str,
    *,
    speaker_person: str | None,
    find_person: Callable[[str], str | None],
) -> str | None:
    """Whose phone to ring: the named person, else the speaker.

    With nobody named and no speaker known, a home where only one person
    has a phone rings that one; otherwise None, and the caller asks.
    """
    if owner:
        for name in (owner, re.sub(r"(?:'s|’s|s')$", "", owner)):
            person = find_person(name)
            if person is not None:
                return person
        return None
    if speaker_person is not None:
        return speaker_person
    candidates = people_with_phones(hass)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _payload(phone: Phone, lang: str) -> dict[str, Any]:
    text = L.template("phone_ring_text", lang)
    if phone.apple:
        return {
            "message": text,
            "data": {
                "tag": _RING_TAG,
                "push": {"sound": {"name": "default", "critical": 1, "volume": 1.0}},
            },
        }
    return {
        "message": "TTS",
        "data": {
            "tts_text": text,
            "media_stream": "alarm_stream_max",
            "ttl": 0,
            "priority": "high",
        },
    }


async def _send(hass: HomeAssistant, phone: Phone, lang: str) -> bool:
    try:
        await hass.services.async_call(
            "notify", phone.service, _payload(phone, lang), blocking=True, context=Context()
        )
    except Exception:  # noqa: BLE001 — one phone failing must not stop the others
        _LOGGER.warning("AI Plugin find phone: ringing %s failed", phone.name, exc_info=True)
        return False
    return True


async def _ring_again(hass: HomeAssistant, phones: list[Phone], lang: str) -> None:
    for _ in range(RINGS - 1):
        await asyncio.sleep(RING_GAP_S)
        for phone in phones:
            await _send(hass, phone, lang)


def _last_seen(hass: HomeAssistant, phones: list[Phone], lang: str) -> str:
    """A sentence when HA last placed the phone away from home, else ""."""
    for phone in phones:
        state = hass.states.get(phone.tracker)
        if state is None or state.state in ("home", "unknown", "unavailable"):
            continue
        if "zone.home" in (state.attributes.get("in_zones") or ()):
            continue
        if state.state == "not_home":
            return L.template("phone_seen_away", lang)
        return L.template("phone_seen_at", lang, place=state.state)
    return ""


async def async_ring_phone(
    hass: HomeAssistant, person_id: str, lang: str, *, own: bool
) -> str | None:
    """Ring every phone of ``person_id`` and return what to say.

    The first ring is sent before replying, so a failure is reported, not
    announced as a success; the remaining rings follow in the background.
    Returns None when the person has no phone that can ring.
    """
    phones = phones_of(hass, person_id)
    if not phones:
        return None
    sent = [phone for phone in phones if await _send(hass, phone, lang)]
    if not sent:
        return L.template("phone_failed", lang)
    if RINGS > 1:
        hass.async_create_task(_ring_again(hass, sent, lang))
    _LOGGER.info(
        "AI Plugin find phone: ringing %s (%s)",
        person_id, ", ".join(phone.name for phone in sent),
    )
    state = hass.states.get(person_id)
    first_name = str(
        (state.attributes.get("friendly_name") if state else None) or person_id
    ).split()[0]
    if own:
        reply = L.template("phone_ringing", lang)
    else:
        reply = L.template("phone_ringing_named", lang, name=first_name)
    seen = _last_seen(hass, sent, lang)
    return f"{reply} {seen}" if seen else reply
