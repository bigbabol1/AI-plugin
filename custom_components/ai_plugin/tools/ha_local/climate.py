"""set_climate_mode: switch a thermostat's operating mode.

Home Assistant's MCP server offers HassClimateSetTemperature but nothing
that changes the HVAC mode, so "put the heating back to auto" had no tool
to land on. The model improvised: it set the temperature to 0 (refused by
HA's range check), then "turned on" the thermostat, which was already
heating, and confirmed a switch to auto that never happened (2026-10-29).
"""

from __future__ import annotations

import logging

from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

from .formatting import _s

_LOGGER = logging.getLogger(__name__)

# HA's HVACMode values, which is what climate.set_hvac_mode accepts.
_HVAC_MODES = ("off", "heat", "cool", "heat_cool", "auto", "dry", "fan_only")

# What people say for a mode, in the languages the plugin speaks. "auto" is
# the thermostat following its own schedule; "manual" is the fixed-target
# mode, which HA calls heat on a heating-only thermostat.
_MODE_ALIASES = {
    "automatic": "auto", "schedule": "auto", "program": "auto",
    "automatik": "auto", "zeitplan": "auto", "programm": "auto",
    "heating": "heat", "manual": "heat", "heizen": "heat", "manuell": "heat",
    "cooling": "cool", "kühlen": "cool",
    "aus": "off",
}


def _hvac_mode(mode: str) -> str | None:
    """The HVACMode a spoken mode word means, or None."""
    m = mode.strip().lower().replace("-", "_").replace(" ", "_")
    m = _MODE_ALIASES.get(m, m)
    return m if m in _HVAC_MODES else None


class ClimateToolsMixin:
    """set_climate_mode."""

    async def _set_climate_mode(
        self,
        mode: str,
        name: str | None = None,
        area: str | None = None,
        exposed_only: bool = True,
        device_id: str | None = None,
    ) -> str:
        """Set the HVAC mode of one named thermostat, a room's, or the home's only one."""
        hvac_mode = _hvac_mode(mode)
        if hvac_mode is None:
            return (
                f"set_climate_mode: unknown mode {mode!r}. Use one of "
                f"{', '.join(_HVAC_MODES)}."
            )

        hass = self._hass
        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)
        area_reg = ar.async_get(hass)

        def _area_id(entry) -> str | None:
            if entry.area_id:
                return entry.area_id
            dev = dev_reg.async_get(entry.device_id) if entry.device_id else None
            return dev.area_id if dev else None

        climates = [
            entry for eid, entry in ent_reg.entities.items()
            if eid.startswith("climate.")
            and (not exposed_only or self._is_exposed(eid))
        ]
        if not climates:
            return "No exposed thermostat in this home."

        if name:
            needle = name.strip().lower()
            targets = []
            for entry in climates:
                st = hass.states.get(entry.entity_id)
                friendly = _s(st.attributes.get("friendly_name") if st else "").lower()
                if needle == entry.entity_id.lower() or needle in friendly:
                    targets.append(entry)
            # With one thermostat in the home, "the heating" or "Heizung"
            # can only mean that one.
            if not targets and len(climates) == 1:
                targets = climates
            if not targets:
                return f"No exposed thermostat matches {name!r}. Try search_entities."
        elif area:
            needle = area.strip().lower()
            target_area = next(
                (
                    a for a in area_reg.async_list_areas()
                    if _s(a.name).lower() == needle
                    or any(_s(al).lower() == needle for al in (a.aliases or ()))
                ),
                None,
            )
            if target_area is None:
                return f"Unknown area {area!r}. Try list_areas."
            targets = [e for e in climates if _area_id(e) == target_area.id]
            # The model often passes the caller's room as the area; with
            # one thermostat in the home that is still the heating meant.
            if not targets and len(climates) == 1:
                targets = climates
            if not targets:
                return f"No exposed thermostat in {target_area.name}."
        else:
            # "The heating" means the caller's room when it has a
            # thermostat, else the home's only one. Several thermostats
            # and none here: ask rather than switch them all.
            targets = []
            dev = dev_reg.async_get(device_id) if device_id else None
            here = dev.area_id if dev else None
            if here:
                targets = [e for e in climates if _area_id(e) == here]
            if not targets:
                if len(climates) > 1:
                    names = ", ".join(sorted(e.entity_id for e in climates))
                    return (
                        f"Several thermostats ({names}). Ask the user which "
                        "one, then pass name or area."
                    )
                targets = climates

        changed: list[str] = []
        already: list[str] = []
        for entry in sorted(targets, key=lambda e: e.entity_id):
            eid = entry.entity_id
            st = hass.states.get(eid)
            if st is None or st.state in ("unavailable", "unknown"):
                return f"[set_climate_mode failed: {eid} is unavailable]"
            supported = [_s(m) for m in st.attributes.get("hvac_modes") or ()]
            if supported and hvac_mode not in supported:
                return (
                    f"[set_climate_mode failed: {eid} has no {hvac_mode!r} "
                    f"mode; it supports {', '.join(supported)}]"
                )
            if st.state == hvac_mode:
                already.append(eid)
                continue
            try:
                await hass.services.async_call(
                    "climate",
                    "set_hvac_mode",
                    {"hvac_mode": hvac_mode},
                    target={"entity_id": eid},
                    blocking=True,
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.exception("AI Plugin set_climate_mode failed")
                return f"[set_climate_mode failed: {exc}]"
            changed.append(f"{eid} (was {st.state})")

        parts = []
        if changed:
            parts.append(f"set {', '.join(changed)} to {hvac_mode}")
        if already:
            parts.append(f"{', '.join(already)} already {hvac_mode}")
        out = "OK — " + "; ".join(parts)
        _LOGGER.debug("set_climate_mode -> %s", out)
        return out
