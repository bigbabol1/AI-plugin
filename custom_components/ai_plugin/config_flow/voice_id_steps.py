"""Voice recognition screens of the options flow.

  Menu ─ voice_id            Settings / Enroll a voice / Delete a voice
  Step ─ voice_id_settings   on/off, voice-started enrollment, thresholds
  Step ─ voice_id_enroll     person + satellite → the satellite starts asking
  Step ─ voice_id_delete     remove one person's fingerprints

Enrolling and deleting end the flow with an abort message instead of
saving: saving options reloads the entry, which would end the enrollment
session that was just started.
"""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.helpers import selector

from ..const import (
    CONF_VOICE_ID_ENABLED,
    CONF_VOICE_ID_MARGIN,
    CONF_VOICE_ID_THRESHOLD,
    CONF_VOICE_ID_VOICE_ENROLL,
    DEFAULT_VOICE_ID_ENABLED,
    DEFAULT_VOICE_ID_VOICE_ENROLL,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


class VoiceIDStepsMixin:
    """Voice recognition screens. Mixed into AIPluginOptionsFlow."""

    def _voice_manager(self):
        return (
            self.hass.data.get(DOMAIN, {})
            .get(self._config_entry.entry_id, {})
            .get("voice_id")
        )

    def _voice_status(self) -> str:
        """One line per enrolled person, plus model and pipeline state."""
        mgr = self._voice_manager()
        if mgr is None:
            return "-"
        lines = []
        persons = mgr.persons()
        groups = self._mic_group_rooms()
        for pid, counts in mgr.profile_summary().items():
            name = (persons.get(pid) or {}).get("name") or pid
            where = ", ".join(
                f"{groups.get(g, g or '?')} ({n})" for g, n in sorted(counts.items())
            )
            lines.append(f"- {name}: {sum(counts.values())} samples — {where}")
        missing = sorted(set(groups) - {g for c in mgr.profile_summary().values() for g in c})
        if lines and missing:
            lines.append(
                "Tip: a few samples on " + ", ".join(groups[g] for g in missing)
                + " improve recognition there."
            )
        if not lines:
            lines.append("No voices enrolled yet.")
        if mgr.enabled:
            from ..voice_id import pipelines  # noqa: PLC0415

            pending = pipelines.unwrapped_pipelines(self.hass, self._config_entry)
            if pending:
                lines.append("Pipelines not yet using voice recognition: " + ", ".join(pending))
        if mgr.last_error:
            lines.append(f"Model unavailable: {mgr.last_error}")
        return "\n".join(lines)

    def _satellites(self) -> dict[str, tuple[str, str]]:
        """device_id → (assist_satellite entity_id, display name)."""
        from homeassistant.helpers import entity_registry as er  # noqa: PLC0415

        out: dict[str, tuple[str, str]] = {}
        for entry in er.async_get(self.hass).entities.values():
            if entry.entity_id.startswith("assist_satellite.") and entry.device_id:
                state = self.hass.states.get(entry.entity_id)
                name = (state.attributes.get("friendly_name") if state else None) or entry.entity_id
                out[entry.device_id] = (entry.entity_id, name)
        return out

    def _mic_group_rooms(self) -> dict[str, str]:
        """mic group → names of the satellites in it, for the status lines."""
        mgr = self._voice_manager()
        rooms: dict[str, list[str]] = {}
        for device_id, (_eid, name) in self._satellites().items():
            rooms.setdefault(mgr.mic_group(device_id), []).append(name)
        return {g: " / ".join(sorted(n)) for g, n in rooms.items()}

    async def async_step_voice_id(self, user_input: dict[str, Any] | None = None):
        return self.async_show_menu(
            step_id="voice_id",
            menu_options=["voice_id_settings", "voice_id_enroll", "voice_id_delete"],
            description_placeholders={"status": self._voice_status()},
        )

    async def async_step_voice_id_settings(self, user_input: dict[str, Any] | None = None):
        from ..voice_id.matching import DEFAULT_MARGIN, DEFAULT_THRESHOLD  # noqa: PLC0415

        if user_input is not None:
            self._options.update(user_input)
            return self.async_create_entry(title="", data=self._options)
        opts = self._options
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_VOICE_ID_ENABLED,
                    default=opts.get(CONF_VOICE_ID_ENABLED, DEFAULT_VOICE_ID_ENABLED),
                ): bool,
                vol.Required(
                    CONF_VOICE_ID_VOICE_ENROLL,
                    default=opts.get(CONF_VOICE_ID_VOICE_ENROLL, DEFAULT_VOICE_ID_VOICE_ENROLL),
                ): bool,
                vol.Required(
                    CONF_VOICE_ID_THRESHOLD,
                    default=opts.get(CONF_VOICE_ID_THRESHOLD, DEFAULT_THRESHOLD),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.2, max=0.8, step=0.01, mode="box")
                ),
                vol.Required(
                    CONF_VOICE_ID_MARGIN,
                    default=opts.get(CONF_VOICE_ID_MARGIN, DEFAULT_MARGIN),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.0, max=0.4, step=0.01, mode="box")
                ),
            }
        )
        return self.async_show_form(step_id="voice_id_settings", data_schema=schema)

    async def async_step_voice_id_enroll(self, user_input: dict[str, Any] | None = None):
        mgr = self._voice_manager()
        if mgr is None or not mgr.enabled:
            return self.async_abort(reason="voice_id_disabled")
        persons = mgr.persons()
        satellites = self._satellites()
        if not persons:
            return self.async_abort(reason="voice_id_no_persons")
        if not satellites:
            return self.async_abort(reason="voice_id_no_satellites")
        if user_input is not None:
            device_id = user_input["satellite"]
            entity_id = satellites[device_id][0]
            fallback = (self.hass.config.language or "en").split("-")[0].lower()
            try:
                await mgr.async_start_on_satellite(device_id, entity_id, user_input["person"], fallback)
            except Exception:  # noqa: BLE001
                # Could not make the satellite speak: fall back to waiting for
                # the wake word, which always reaches the user audibly.
                _LOGGER.warning("AI Plugin voice ID: could not start the satellite", exc_info=True)
                mgr.arm_enrollment(device_id, user_input["person"], fallback)
                return self.async_abort(
                    reason="voice_id_enroll_armed",
                    description_placeholders={"satellite": satellites[device_id][1]},
                )
            return self.async_abort(
                reason="voice_id_enroll_started",
                description_placeholders={"satellite": satellites[user_input["satellite"]][1]},
            )
        # Label each person with what is stored, so this list of everyone
        # who CAN be enrolled is not mistaken for who already is.
        summary = mgr.profile_summary()

        def _label(pid: str, person: dict) -> str:
            count = sum(summary.get(pid, {}).values())
            state = f"{count} samples" if count else "not enrolled yet"
            return f"{person['name']} ({state})"

        schema = vol.Schema(
            {
                vol.Required("person"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": pid, "label": _label(pid, p)} for pid, p in persons.items()
                        ]
                    )
                ),
                vol.Required("satellite"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": did, "label": name}
                            for did, (_eid, name) in sorted(satellites.items(), key=lambda kv: kv[1][1])
                        ]
                    )
                ),
            }
        )
        return self.async_show_form(step_id="voice_id_enroll", data_schema=schema)

    async def async_step_voice_id_delete(self, user_input: dict[str, Any] | None = None):
        mgr = self._voice_manager()
        enrolled = list(mgr.profile_summary()) if mgr else []
        if not enrolled:
            return self.async_abort(reason="voice_id_nothing_enrolled")
        if user_input is not None:
            await mgr.async_delete_profile(user_input["person"])
            return self.async_abort(reason="voice_id_deleted")
        persons = mgr.persons()
        schema = vol.Schema(
            {
                vol.Required("person"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": pid, "label": str((persons.get(pid) or {}).get("name") or pid)}
                            for pid in enrolled
                        ]
                    )
                )
            }
        )
        return self.async_show_form(step_id="voice_id_delete", data_schema=schema)
