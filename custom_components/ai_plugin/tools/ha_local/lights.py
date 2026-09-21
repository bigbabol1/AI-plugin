"""Area actions and relative brightness.

set_area_state is the bulk switch; set_brightness exists because HA's
own HassLightSet is absolute only and cannot step, so "a bit brighter"
had nowhere to go.
"""

from __future__ import annotations

import logging
import re

from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

from .formatting import _s

_LOGGER = logging.getLogger(__name__)


_ACTION_DOMAINS = {"light", "switch", "fan", "cover", "climate"}


# Relative brightness stepping. HA core has no relative light intent:
# HassLightSet takes an ABSOLUTE 0-100 and brightness 0 turns the light
# OFF. Asked to turn "brighter" into a number with no knowledge of the
# current one, the model picked the extremes — "brighter" set every
# living-room lamp to 100% and "dim the lights" turned them all off
# (measured 2026-09-07 18:46:02Z, four lights off in the same second).
# So the read-modify-write lives here, in Python, not in the prompt.
_BRIGHTNESS_STEP_PP = 20   # percentage POINTS per brighter/dimmer step

_BRIGHTNESS_FLOOR_PCT = 10  # dimmer never goes below this, so it can

                            # never reach 0 and turn the light off

# Small models drop enum arguments constantly — the same defect the area
# ladder in set_area_state exists for. Measured 2026-09-10: the model
# called set_brightness(area='Wohnzimmer') with NO command at all, the
# handler rejected it, and the model fell straight back to HassLightSet
# and set every lamp to 100% — the exact bug this tool exists to stop.
# So an omitted or unrecognised command is recovered from the utterance
# instead of refused.
_BRIGHTER_RE = re.compile(
    r"\b(?:heller|aufhellen|helligkeit\s+(?:hoch|rauf|erh\u00f6h\w*)|"
    r"brighter|brighten|lighter|turn\s+up\s+the\s+light\w*|"
    r"plus\s+lumineux|m\u00e1s\s+brillante|mais\s+brilhante|ja\u015bniej)\b",
    re.IGNORECASE,
)

_DIMMER_RE = re.compile(
    r"\b(?:dunkler|abdunkeln|dimm\w*|ged\u00e4mpft|helligkeit\s+(?:runter|"
    r"reduzier\w*)|dim|dimmer|darker|turn\s+down\s+the\s+light\w*|"
    r"moins\s+lumineux|m\u00e1s\s+tenue|mais\s+escuro|ciemniej)\b",
    re.IGNORECASE,
)


# Patterns that confirm the user explicitly meant "every device, every area"
# rather than a specific device. Used to gate set_area_state(area='all')
# against the model's tendency to fall back to it after a failed specific
# call. Multilingual; conservative.
_USER_SAID_ALL_RE = re.compile(
    r"\b(?:"
    r"all|every(?:thing)?|everywhere|anywhere|whole\s+(?:house|flat|home)|"
    r"entire\s+(?:house|flat|home)|every\s+room|"
    r"alles?|jede(?:[srn])?|j[eé]des|s[äa]mtliche[srn]?|"
    r"[uü]berall|im\s+ganzen?\s+haus|in\s+der\s+ganzen\s+wohnung|"
    r"komplett(?:e[srn]?)?"
    r")\b",
    re.IGNORECASE,
)


# Specific device nouns: "air purifier", "kettle", etc. When the user
# message contains one of these, set_area_state must be refused — the
# user means a NAMED PHYSICAL DEVICE, not a domain category. Force
# search_entities + HassTurnOn/Off path.
_DEVICE_NOUN_RE = re.compile(
    r"\b(?:"
    r"air\s+purifier|air\s+filter|humidifier|dehumidifier|"
    r"kettle|toaster|coffee\s+(?:machine|maker)|espresso|"
    r"reading\s+lamp|mood\s+light|night\s+light|floor\s+lamp|"
    r"tv|television|monitor|soundbar|"
    r"vacuum|robot|hoover|diffuser|aroma|"
    r"luftreiniger|luftfilter|luftbefeuchter|"
    r"wasserkocher|kaffeemaschine|"
    r"stehlampe|nachtlicht|leselampe|moodlight|flurlicht|"
    r"fernseher|lautsprecher|"
    r"staubsauger|saugroboter"
    r")\b",
    re.IGNORECASE,
)


# Magic values that set_area_state treats as "every area".
# Empty string is intentionally NOT here: omitting area means "default
# to caller's area" (when device_id resolves), with sweep-all only as
# the explicit-keyword path.
_SWEEP_ALL_KEYWORDS = {
    "*",
    "all",
    "any",
    "every",
    "everywhere",
    "anywhere",
    "whole house",
    "entire house",
    "house",
    # German aliases — primary user locale.
    "alle",
    "alles",
    "überall",
    "ueberall",
    "ganzes haus",
}


_DOMAIN_SERVICE_MAP: dict[str, dict[str, str]] = {
    "light":        {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "switch":       {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "fan":          {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
    "cover":        {"turn_on": "open_cover", "turn_off": "close_cover", "toggle": "toggle"},
    "climate":      {"turn_on": "turn_on", "turn_off": "turn_off"},
}


class LightToolsMixin:
    """set_area_state / set_brightness."""

    async def _set_area_state(
        self,
        area: str | None,
        domain: str,
        action: str,
        exposed_only: bool = True,
        device_id: str | None = None,
    ) -> str:
        """Turn devices in an area on/off/toggle via a single service call.

        Area resolution:
        - Explicit area name → that area only.
        - Explicit wildcard ('all', 'every', 'everywhere', '*', etc.) →
          sweep every area.
        - Empty/None area + voice satellite device_id → caller's room.
        - Empty/None area + no device_id → sweep every area. call_tool
          only lets this through when the user explicitly said
          'all'/'everything'/'whole house' — an omitted area alone must
          not empty the whole home from a text chat.
        """
        domain = domain.lower().strip(".")
        action = action.lower()

        if domain not in _ACTION_DOMAINS:
            allowed = ", ".join(sorted(_ACTION_DOMAINS))
            return f"Domain {domain!r} not supported. Allowed: {allowed}."

        svc_map = _DOMAIN_SERVICE_MAP[domain]
        if action not in svc_map:
            allowed = " or ".join(svc_map)
            return f"{domain} does not support {action}. Use {allowed}."

        needle = (area or "").strip().lower()
        # When area is empty AND a device_id is provided, default to the
        # calling satellite's area before falling through to sweep-all.
        # User in living room saying 'lights on' should affect the living
        # room only, not every light in the flat.
        if not needle and device_id:
            try:
                ent_reg = er.async_get(self._hass)
                dev_reg = dr.async_get(self._hass)
                area_reg = ar.async_get(self._hass)
                dev = dev_reg.async_get(device_id)
                area_id = dev.area_id if dev else None
                if not area_id:
                    for entry in er.async_entries_for_device(ent_reg, device_id):
                        if entry.area_id:
                            area_id = entry.area_id
                            break
                if area_id:
                    a = area_reg.async_get_area(area_id)
                    if a:
                        needle = _s(a.name).lower()
                        _LOGGER.debug(
                            "set_area_state: area unspecified, defaulted to "
                            "caller's area %r (device_id=%s)",
                            a.name, device_id,
                        )
            except Exception:  # noqa: BLE001
                _LOGGER.debug("set_area_state device→area lookup failed", exc_info=True)
        sweep_all = needle in _SWEEP_ALL_KEYWORDS or needle == ""

        target_area = None
        if not sweep_all:
            area_reg = ar.async_get(self._hass)
            for a in area_reg.async_list_areas():
                if _s(a.name).lower() == needle:
                    target_area = a
                    break
                aliases = getattr(a, "aliases", None) or ()
                if any(_s(al).lower() == needle for al in aliases):
                    target_area = a
                    break
            if target_area is None:
                return f"Unknown area {area!r}. Try list_areas."

        ent_reg = er.async_get(self._hass)
        dev_reg = dr.async_get(self._hass)

        ids: list[str] = []
        for eid, entry in ent_reg.entities.items():
            if not eid.startswith(f"{domain}."):
                continue
            if not sweep_all:
                area_id = entry.area_id
                if not area_id and entry.device_id:
                    dev = dev_reg.async_get(entry.device_id)
                    if dev:
                        area_id = dev.area_id
                if area_id != target_area.id:
                    continue
            if exposed_only and not self._is_exposed(eid):
                continue
            ids.append(eid)

        scope_label = "all areas" if sweep_all else target_area.name
        if not ids:
            scope = "exposed " if exposed_only else ""
            return f"No {scope}{domain} in {scope_label}."

        ids.sort()
        service = svc_map[action]
        try:
            await self._hass.services.async_call(
                domain,
                service,
                target={"entity_id": ids},
                blocking=True,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("AI Plugin set_area_state failed")
            return f"[set_area_state failed: {exc}]"

        preview = ", ".join(ids[:6])
        more = f" (+{len(ids) - 6} more)" if len(ids) > 6 else ""
        return (
            f"OK — {action} {len(ids)} {domain}(s) in {scope_label}: "
            f"{preview}{more}"
        )

    async def _set_brightness(
        self,
        command: str,
        level: int | None = None,
        area: str | None = None,
        name: str | None = None,
        exposed_only: bool = True,
        device_id: str | None = None,
        allow_sweep: bool = False,
    ) -> str:
        """Relative or absolute brightness for lights in an area or one lamp.

        Read-modify-write happens HERE, not in the model. 'dimmer' clamps
        at _BRIGHTNESS_FLOOR_PCT so it can never reach 0 and switch a
        light off — turning off is HassTurnOff/set_area_state's job.
        """
        _LOGGER.debug(
            "set_brightness: command=%r level=%r area=%r name=%r",
            command, level, area, name,
        )
        if command not in ("brighter", "dimmer", "set"):
            return (
                "set_brightness: command must be 'brighter', 'dimmer' or "
                "'set'."
            )
        if command == "set":
            if level is None:
                return "set_brightness: 'set' needs level (1-100)."
            level = max(1, min(100, level))

        hass = self._hass
        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)

        ids: list[str] = []
        scope_label = ""

        if name:
            needle = name.strip().lower()
            for eid, entry in ent_reg.entities.items():
                if not eid.startswith("light."):
                    continue
                if exposed_only and not self._is_exposed(eid):
                    continue
                st = hass.states.get(eid)
                friendly = _s((st.attributes.get("friendly_name") if st else "")).lower()
                if needle in (eid.lower(), friendly) or needle in friendly:
                    ids.append(eid)
            if not ids:
                return (
                    f"No exposed light matches {name!r}. Try search_entities."
                )
            ids.sort()
            scope_label = name
        else:
            needle = (area or "").strip().lower()
            if not needle and device_id:
                try:
                    area_reg = ar.async_get(hass)
                    dev = dev_reg.async_get(device_id)
                    area_id = dev.area_id if dev else None
                    if not area_id:
                        for entry in er.async_entries_for_device(ent_reg, device_id):
                            if entry.area_id:
                                area_id = entry.area_id
                                break
                    if area_id:
                        a = area_reg.async_get_area(area_id)
                        if a:
                            needle = _s(a.name).lower()
                except Exception:  # noqa: BLE001
                    _LOGGER.debug(
                        "set_brightness device->area lookup failed", exc_info=True
                    )
            # An OMITTED area must never mean "every light in the flat".
            # set_area_state guards this in its dispatcher; do the same here.
            # With a satellite device_id the needle was already resolved to
            # the caller's room above, so reaching this empty means a text
            # chat with no room named — refuse rather than sweep the home.
            if not needle and not allow_sweep:
                return (
                    "set_brightness: name a room (area='living room'), or "
                    "say 'all' for the whole home."
                )
            sweep_all = needle in _SWEEP_ALL_KEYWORDS or needle == ""

            target_area = None
            if not sweep_all:
                area_reg = ar.async_get(hass)
                for a in area_reg.async_list_areas():
                    if _s(a.name).lower() == needle:
                        target_area = a
                        break
                    aliases = getattr(a, "aliases", None) or ()
                    if any(_s(al).lower() == needle for al in aliases):
                        target_area = a
                        break
                if target_area is None:
                    return f"Unknown area {area!r}. Try list_areas."

            for eid, entry in ent_reg.entities.items():
                if not eid.startswith("light."):
                    continue
                if not sweep_all:
                    area_id = entry.area_id
                    if not area_id and entry.device_id:
                        dev = dev_reg.async_get(entry.device_id)
                        if dev:
                            area_id = dev.area_id
                    if area_id != target_area.id:
                        continue
                if exposed_only and not self._is_exposed(eid):
                    continue
                ids.append(eid)
            ids.sort()
            scope_label = "all areas" if sweep_all else target_area.name
            if not ids:
                scope = "exposed " if exposed_only else ""
                return f"No {scope}lights in {scope_label}."

        # Read current brightness, compute the target, group by value so
        # lights sharing a target take one service call.
        by_target: dict[int, list[str]] = {}
        skipped_off: list[str] = []
        unavailable: list[str] = []
        for eid in ids:
            st = hass.states.get(eid)
            if st is None or st.state in ("unavailable", "unknown"):
                unavailable.append(eid)
                continue
            is_on = st.state == "on"
            raw = st.attributes.get("brightness")
            cur_pct = 0
            if is_on and raw is not None:
                try:
                    cur_pct = max(0, min(100, round(float(raw) / 255 * 100)))
                except (TypeError, ValueError):
                    cur_pct = 0
            elif is_on:
                # On but no brightness attribute (on/off-only light).
                cur_pct = 100

            if command == "set":
                new_pct = level
            elif command == "brighter":
                new_pct = min(100, max(_BRIGHTNESS_FLOOR_PCT, cur_pct + _BRIGHTNESS_STEP_PP))
            else:  # dimmer
                if not is_on:
                    skipped_off.append(eid)
                    continue
                new_pct = max(_BRIGHTNESS_FLOOR_PCT, cur_pct - _BRIGHTNESS_STEP_PP)
            by_target.setdefault(int(new_pct), []).append(eid)

        if not by_target:
            if skipped_off:
                return (
                    f"No lights on in {scope_label} to dim "
                    f"({len(skipped_off)} already off)."
                )
            return f"No usable lights in {scope_label}."

        applied: list[str] = []
        for target_pct, group in sorted(by_target.items()):
            try:
                await hass.services.async_call(
                    "light",
                    "turn_on",
                    {"brightness_pct": target_pct},
                    target={"entity_id": group},
                    blocking=True,
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.exception("AI Plugin set_brightness failed")
                return f"[set_brightness failed: {exc}]"
            applied.append(f"{len(group)} to {target_pct}%")

        note = ""
        if skipped_off:
            note += f", {len(skipped_off)} already off"
        if unavailable:
            note += f", {len(unavailable)} unavailable"
        _out = f"OK — {command} in {scope_label}: " + ", ".join(applied) + note
        _LOGGER.debug("set_brightness -> %s", _out)
        return _out
