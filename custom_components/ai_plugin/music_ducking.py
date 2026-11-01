"""Lowering the music while a satellite listens.

A satellite cannot cancel music from a speaker that is not its own, so it hears
it as someone still talking: a one-second command kept the microphone open for
7 seconds (measured 2026-11-01). Speakers in the room play quieter from the
moment it starts listening until the words are in.

This follows the satellite's own state, which Home Assistant keeps for every
satellite, so it works with or without Mic to MediaPlayer. A satellite whose
firmware lowers its own speaker as well just gets that speaker quieter still,
and only while it listens.

Two satellites can wake on the same word. Each one holds the duck and the volume
comes back when the last one lets go; the level is saved once, or the second
would save the lowered volume as the one to restore.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import time

from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util

from .satellite_speech import mic2mp_for

_LOGGER = logging.getLogger(__name__)

DATA_KEY = "ai_plugin_music_ducks"
DUCK_FACTOR = 0.2
# A satellite stuck listening must not leave the music quiet.
MAX_DUCK_S = 30.0
# A restored speaker can report its lowered volume for a moment.
STATE_LAG_S = 10.0
_LISTENING = "listening"
# A satellite's own speaker reports "playing" for its wake sound, for under a
# second, just before it listens (measured 2026-11-01). That is not music.
_WAKE_SOUND_S = 1.5


def _ducks(hass) -> dict[str, dict]:
    return hass.data.setdefault(DATA_KEY, {})


def room_players(hass, satellite_entity_id: str) -> list[str]:
    """The speakers playing in the satellite's room, its own included, and the
    one Mic to MediaPlayer routes it to. Not a TV. Not a group either: its
    members play too and are lowered one by one, and both would compound."""
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)

    def _area(entry) -> str | None:
        if entry.area_id:
            return entry.area_id
        device = dev_reg.async_get(entry.device_id) if entry.device_id else None
        return device.area_id if device else None

    candidates: set[str] = set()
    own: set[str] = set()
    routed = mic2mp_for(hass, satellite_entity_id)
    if routed is not None:
        candidates.add(routed.media_player_entity_id)
    satellite = ent_reg.async_get(satellite_entity_id)
    if satellite is not None:
        area = _area(satellite)
        for entry in ent_reg.entities.values():
            if entry.entity_id.startswith("media_player.") and (
                entry.device_id == satellite.device_id or (area and _area(entry) == area)
            ):
                candidates.add(entry.entity_id)
                if entry.device_id == satellite.device_id:
                    own.add(entry.entity_id)
    playing = []
    for entity_id in sorted(candidates):
        state = hass.states.get(entity_id)
        if (
            state is not None
            and state.state == "playing"
            and state.attributes.get("device_class") != "tv"
            and state.attributes.get("mass_player_type") != "group"
            and not (entity_id in own and _just_started(state))
        ):
            playing.append(entity_id)
    return playing


def _just_started(state) -> bool:
    changed = getattr(state, "last_changed", None)
    return changed is not None and (dt_util.utcnow() - changed).total_seconds() < _WAKE_SOUND_S


def _set_volume(hass, entity_id: str, level: float) -> None:
    async def _call() -> None:
        try:
            await hass.services.async_call(
                "media_player",
                "volume_set",
                {"volume_level": round(level, 3)},
                target={"entity_id": entity_id},
                blocking=True,
            )
        except Exception:  # noqa: BLE001 — quiet music must never break a voice turn
            _LOGGER.debug("AI Plugin: could not set the volume of %s", entity_id, exc_info=True)

    hass.async_create_task(_call())


def duck(hass, holder: str, players: list[str]) -> None:
    """Lower these speakers on behalf of ``holder``."""
    ducks = _ducks(hass)
    now = time.monotonic()
    for entity_id in players:
        entry = ducks.get(entity_id)
        if entry is not None and entry["holders"]:
            entry["holders"].add(holder)
            continue
        state = hass.states.get(entity_id)
        level = state.attributes.get("volume_level") if state is not None else None
        if entry is not None and now - entry["restored_at"] < STATE_LAG_S:
            level = entry["level"]
        if not level:
            continue
        ducks[entity_id] = {"level": level, "holders": {holder}, "restored_at": 0.0}
        _set_volume(hass, entity_id, level * DUCK_FACTOR)


def release(hass, holder: str) -> None:
    """Let go; a speaker nobody holds any more goes back to its volume."""
    for entity_id, entry in _ducks(hass).items():
        if holder not in entry["holders"]:
            continue
        entry["holders"].discard(holder)
        if not entry["holders"]:
            entry["restored_at"] = time.monotonic()
            _set_volume(hass, entity_id, entry["level"])


def real_volume(hass, entity_id: str) -> float | None:
    """A speaker's volume as the user set it, whatever it is lowered to now."""
    ducks = hass.data.get(DATA_KEY)
    entry = ducks.get(entity_id) if isinstance(ducks, dict) else None
    if entry is not None and (
        entry["holders"] or time.monotonic() - entry["restored_at"] < STATE_LAG_S
    ):
        return entry["level"]
    state = hass.states.get(entity_id)
    level = state.attributes.get("volume_level") if state is not None else None
    return level if isinstance(level, (int, float)) else None


def async_setup_listening_duck(hass) -> Callable[[], None]:
    """Follow every satellite's state; returns what undoes it."""
    limits: dict[str, Callable[[], None]] = {}

    def _end(satellite: str) -> None:
        cancel = limits.pop(satellite, None)
        if cancel is not None:
            cancel()
        release(hass, satellite)

    def _start(satellite: str) -> None:
        players = room_players(hass, satellite)
        if not players:
            return
        _LOGGER.debug("AI Plugin: lowering %s while %s listens", ", ".join(players), satellite)
        duck(hass, satellite, players)

        @callback
        def _limit(_now) -> None:
            limits.pop(satellite, None)
            release(hass, satellite)

        limits[satellite] = async_call_later(hass, MAX_DUCK_S, _limit)

    @callback
    def _is_satellite(event_data) -> bool:
        return str(event_data.get("entity_id", "")).startswith("assist_satellite.")

    @callback
    def _on_state(event) -> None:
        old = event.data.get("old_state")
        new = event.data.get("new_state")
        was = old is not None and old.state == _LISTENING
        now = new is not None and new.state == _LISTENING
        satellite = event.data["entity_id"]
        try:
            if now and not was:
                _start(satellite)
            elif was and not now:
                _end(satellite)
        except Exception:  # noqa: BLE001 — must never disturb the voice turn
            _LOGGER.debug("AI Plugin: music ducking failed", exc_info=True)

    unsubscribe = hass.bus.async_listen(EVENT_STATE_CHANGED, _on_state, event_filter=_is_satellite)

    def _undo() -> None:
        unsubscribe()
        for satellite in list(limits):
            _end(satellite)

    return _undo
