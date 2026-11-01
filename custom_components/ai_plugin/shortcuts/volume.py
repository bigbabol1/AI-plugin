"""Louder and quieter by a step you can hear.

The players' own volume_up and volume_down are too small: "Turn down the volume"
went from 40% to 37% (measured 2026-11-01).
"""

from __future__ import annotations

import logging

from ..music_ducking import real_volume

_LOGGER = logging.getLogger(__name__)

VOLUME_STEP = 0.10


async def async_step_volume(hass, entity_ids: list[str], command: str) -> None:
    """Step these speakers up or down. A player that reports no volume gets its
    own volume_up or volume_down."""
    sign = 1 if command == "volume_up" else -1
    for entity_id in entity_ids:
        # Not what it reports: while a satellite listens it is lowered, and
        # "turn it down" must not start from there.
        level = real_volume(hass, entity_id)
        if level is None:
            await hass.services.async_call(
                "media_player", command, None, target={"entity_id": entity_id}, blocking=True,
            )
            continue
        await hass.services.async_call(
            "media_player",
            "volume_set",
            {"volume_level": round(min(1.0, max(0.0, level + sign * VOLUME_STEP)), 2)},
            target={"entity_id": entity_id},
            blocking=True,
        )
