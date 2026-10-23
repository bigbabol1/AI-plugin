"""Timer completions that arrive as a conversation turn.

In announce mode every timer carries a conversation_command, so HA calls
the agent back at expiry with a sentinel instead of ringing on the
satellite. This mixin recognises that call and plays the announcement
through a media player.
"""

from __future__ import annotations

import logging

from homeassistant.helpers import entity_registry as er

from ..const import CONF_TIMER_ANNOUNCE, DEFAULT_TIMER_ANNOUNCE, DOMAIN
from ..i18n import L

_LOGGER = logging.getLogger(__name__)


class TimerAnnounceMixin:
    """Timer-completion handling. Mixed into Orchestrator."""

    def is_voice_device(self, device_id: str | None) -> bool:
        """Public alias of _is_voice_device for callers in other modules."""
        return self._is_voice_device(device_id)

    def _timer_announce_agent_id(self) -> str | None:
        """Return this integration's conversation entity_id when timer
        announcements are enabled, else None (default off = ring on device)."""
        if not self._entry.options.get(CONF_TIMER_ANNOUNCE, DEFAULT_TIMER_ANNOUNCE):
            return None
        if self._hass is None:
            return None
        try:
            return er.async_get(self._hass).async_get_entity_id(
                "conversation", DOMAIN, self._entry.entry_id
            )
        except Exception:  # noqa: BLE001
            _LOGGER.debug("AI Plugin: agent entity lookup failed", exc_info=True)
            return None

    async def _handle_timer_done(
        self, message: str, device_id: str | None, lang: str
    ) -> str:
        """Announce an expired timer on the satellite's speaker path.

        How announce-mode timers work, end to end:
          1. With the ``timer_announce`` option on, ``start_timer`` attaches
             a ``conversation_command`` ("AI_PLUGIN_TIMER_DONE <name>") and
             this agent's id to the HA timer it creates.
          2. HA's own TimerManager does all the scheduling (cancel, pause,
             "add two minutes" — everything stays one HA-managed timer).
             Because a conversation_command is set, HA never notifies the
             device (no on-device ring/LEDs) and doesn't require the device
             to support timers at all.
          3. At expiry, HA calls this agent back with the sentinel text and
             the original device_id. ``async_process`` routes it here —
             before shortcuts and history, since it's not a user utterance.
          4. This method resolves the device's assist_satellite entity and
             plays a localized announcement via ``mic_to_mediaplayer.announce``
             (TTS routed to a separate speaker), falling back to HA's native
             ``assist_satellite.announce`` for satellites with their own audio.

        Returns "" — the announcement IS the reply; nothing is spoken twice.
        """
        from ..const import TIMER_DONE_SENTINEL  # noqa: PLC0415

        timer_name = message[len(TIMER_DONE_SENTINEL):].strip()
        if timer_name:
            text = L.template("timer_done_named", lang, label=timer_name)
        else:
            text = L.template("timer_done", lang)

        satellite = None
        if device_id and self._hass is not None:
            try:
                ent_reg = er.async_get(self._hass)
                for entry in er.async_entries_for_device(ent_reg, device_id):
                    if entry.entity_id.startswith("assist_satellite."):
                        satellite = entry.entity_id
                        break
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin: satellite lookup failed", exc_info=True)
        if satellite is None or self._hass is None:
            _LOGGER.warning(
                "AI Plugin: timer expired but no satellite found for device %s — "
                "announcement dropped", device_id,
            )
            return ""

        try:
            if self._hass.services.has_service("mic_to_mediaplayer", "announce"):
                await self._hass.services.async_call(
                    "mic_to_mediaplayer",
                    "announce",
                    {"satellite_entity_id": satellite, "message": text},
                    blocking=False,
                )
            else:
                await self._hass.services.async_call(
                    "assist_satellite",
                    "announce",
                    {"entity_id": satellite, "message": text},
                    blocking=False,
                )
            _LOGGER.info(
                "AI Plugin: timer announcement %r → %s", text, satellite
            )
        except Exception:  # noqa: BLE001
            _LOGGER.exception("AI Plugin: timer announcement failed")
        return ""
