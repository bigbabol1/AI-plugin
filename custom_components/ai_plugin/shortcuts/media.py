"""Playback, volume and mute commands, dispatched without the model.

Small local models turn "next track" into a confident prose reply and no
tool call, so the music keeps playing. These patterns dispatch the
media_player service directly.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from homeassistant.core import HomeAssistant

from . import registry
from .registry import _extract_area_from_media_message, dr, er

_LOGGER = logging.getLogger(__name__)

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

# Questions must never trigger playback changes: "what does stop mean?",
# "when is the next bus?" contain trigger words but ask, not command.
# Ambiguity falls through to the LLM, which can still act via tools.
_MEDIA_QUESTION_RE = re.compile(
    r"\?\s*$|\b(?:"
    r"what|what's|who|who's|whose|when|where|why|how|which|"
    r"was|wer|wann|wo|warum|wieso|weshalb|wie|welche\w*"
    r")\b",
    re.IGNORECASE,
)

# No word boundary on the left: German compounds ("Nudeltimer").
_TIMER_MENTION_RE = re.compile(r"timer|wecker|countdown|stoppuhr|eieruhr", re.IGNORECASE)

# German homographs of playback verbs: bare "halt" is usually a modal
# particle ("das ist halt so") and bare "weiter" continues *speech* as
# often as music ("erzähl weiter", "und so weiter"). Only trust them as
# playback commands in very short utterances.
_AMBIGUOUS_BARE_TRIGGERS = {"halt", "weiter"}

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

    # Questions fall through to the LLM — a trigger word inside a question
    # ("what does stop mean?") must not change playback with an empty reply.
    if _MEDIA_QUESTION_RE.search(msg):
        return None

    # "Resume the timer" / "Pause the timer" share the playback verbs and
    # belong to the timer tools. Matched here, "resume the timer" sent
    # media_play to every idle exposed speaker and never reached the timer.
    if _TIMER_MENTION_RE.search(msg):
        return None

    detected = _detect_media_command(msg)
    if detected is None:
        return None
    cmd, match = detected

    matched_text = match.group(0).strip().lower()
    if matched_text in _AMBIGUOUS_BARE_TRIGGERS and len(msg.split()) > 3:
        return None

    area = _extract_area_from_media_message(hass, msg)

    relevant_states = _MEDIA_RELEVANT_STATES[cmd]
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)

    target_ids: list[str] = []
    for eid, entry in ent_reg.entities.items():
        if not eid.startswith("media_player."):
            continue
        if not registry.is_exposed(hass, eid):
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
