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
          4. This method speaks a localized announcement in the voice of the
             satellite's Assist pipeline, like every reply there: through
             ``mic_to_mediaplayer.announce`` where that routes the satellite
             to another speaker, else HA's native ``assist_satellite.announce``.

        Returns "" — the announcement IS the reply; nothing is spoken twice.
        """
        from .. import satellite_speech  # noqa: PLC0415
        from ..const import TIMER_DONE_SENTINEL  # noqa: PLC0415

        timer_name = message[len(TIMER_DONE_SENTINEL):].strip()
        text = None
        if self._hass is not None and device_id:
            if timer_name:
                text = await satellite_speech.async_announce(
                    self._hass, device_id, "timer_done_named", lang, label=timer_name
                )
            else:
                text = await satellite_speech.async_announce(
                    self._hass, device_id, "timer_done", lang
                )
        if text is None:
            _LOGGER.warning(
                "AI Plugin: timer expired but could not be announced on device "
                "%s", device_id,
            )
        else:
            _LOGGER.info("AI Plugin: timer announcement %r on %s", text, device_id)
        return ""
