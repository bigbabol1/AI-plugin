"""Discovery: areas, entities, one entity, and search.

On-demand lookups against the registries, so the prompt never carries a
YAML dump of the house.
"""

from __future__ import annotations

import logging
import re

from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

from . import exposure  # noqa: F401 — re-exported for callers that patch it
from .formatting import _INTERESTING_ATTRS, _cap, _s
from .lights import _SWEEP_ALL_KEYWORDS

_LOGGER = logging.getLogger(__name__)


class EntityToolsMixin:
    """list_areas / list_entities / get_entity / search_entities."""

    def _list_areas(self) -> str:
        area_reg = ar.async_get(self._hass)
        names = sorted({_s(a.name) for a in area_reg.async_list_areas() if _s(a.name)})
        if not names:
            return "No areas defined."
        return _cap([f"- {n}" for n in names], header=f"{len(names)} areas:\n")

    def _list_entities(
        self,
        area: str | None = None,
        domain: str | None = None,
        state: str | None = None,
        exposed_only: bool = True,
    ) -> str:
        hass = self._hass
        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)
        area_reg = ar.async_get(hass)

        target_domain = domain.lower().strip(".") if domain else None
        target_state = state.lower().strip() if state else None

        # Alias-aware, like every other area resolver in this module —
        # list_entities(area='wohnzimmer') must work when 'wohnzimmer' is
        # an alias of "Living room".
        target_area_entry = None
        if area:
            target_area_entry = self._resolve_area(area)
            if target_area_entry is None:
                return f"Unknown area {area!r}. Try list_areas."

        def _area_of(entry) -> tuple[str | None, str | None]:
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if not area_id:
                return None, None
            a = area_reg.async_get_area(area_id)
            return area_id, (_s(a.name) if a else None)

        rows: list[tuple[str, str, str, str]] = []
        for entity_id, entry in ent_reg.entities.items():
            if target_domain and not entity_id.startswith(f"{target_domain}."):
                continue
            if exposed_only and not self._is_exposed(entity_id):
                continue
            a_id, a_name = _area_of(entry)
            if target_area_entry is not None and a_id != target_area_entry.id:
                continue
            ent_state = hass.states.get(entity_id)
            state_val = ent_state.state if ent_state else "unknown"
            if target_state and state_val.lower() != target_state:
                continue
            friendly = _s(ent_state.attributes.get("friendly_name")) if ent_state else ""
            friendly = friendly or _s(entry.name) or _s(entry.original_name) or entity_id
            rows.append((entity_id, friendly, a_name or "-", state_val))

        if not rows:
            filt = ", ".join(
                f"{k}={v}" for k, v in [("area", area), ("domain", domain), ("state", state)] if v
            )
            exp_note = "" if not exposed_only else ", exposed only"
            return f"No entities match ({filt or 'no filter'}{exp_note})."

        rows.sort(key=lambda r: (r[2], r[0]))
        lines = [f"- {eid} — {fn} [{st}] @ {a}" for eid, fn, a, st in rows]
        header = f"{len(rows)} entities"
        if domain:
            header += f" domain={domain}"
        if area:
            header += f" area={area}"
        if state:
            header += f" state={state}"
        return _cap(lines, header=header + ":\n")

    async def _get_entity(self, name_or_id: str, exposed_only: bool = True) -> str:
        if not name_or_id:
            return "get_entity: empty name_or_id."
        hass = self._hass
        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)
        area_reg = ar.async_get(hass)

        def _area_for(entry) -> str | None:
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if not area_id:
                return None
            a = area_reg.async_get_area(area_id)
            return _s(a.name) if a else None

        entry = None
        shadowed_id: str | None = None
        if "." in name_or_id:
            entry = ent_reg.async_get(name_or_id)
        if entry is None:
            needle = name_or_id.strip().lower()
            # Prefer EXPOSED matches: registry order is arbitrary, and an
            # unexposed diagnostic entity matching first must not shadow an
            # exposed entity carrying the same name fragment.
            for eid, e in ent_reg.entities.items():
                candidates = [_s(e.name), _s(e.original_name)]
                candidates.extend(_s(a) for a in (e.aliases or ()))
                st = hass.states.get(eid)
                if st:
                    candidates.append(_s(st.attributes.get("friendly_name")))
                if any(c and c.strip().lower() == needle for c in candidates):
                    if exposed_only and not self._is_exposed(eid):
                        shadowed_id = shadowed_id or eid
                        continue
                    entry = e
                    break
            if entry is None:
                for eid, e in ent_reg.entities.items():
                    candidates = [_s(e.name), _s(e.original_name), eid]
                    candidates.extend(_s(a) for a in (e.aliases or ()))
                    st = hass.states.get(eid)
                    if st:
                        candidates.append(_s(st.attributes.get("friendly_name")))
                    if any(c and needle in c.lower() for c in candidates):
                        if exposed_only and not self._is_exposed(eid):
                            shadowed_id = shadowed_id or eid
                            continue
                        entry = e
                        break
        if entry is None:
            if shadowed_id is not None:
                return (
                    f"Entity {shadowed_id!r} exists but is not exposed to the "
                    "conversation assistant. Use exposed_only=false if you "
                    "need to inspect it."
                )
            return f"No entity matches {name_or_id!r}. Try search_entities."

        entity_id = entry.entity_id
        if exposed_only and not self._is_exposed(entity_id):
            return (
                f"Entity {entity_id!r} exists but is not exposed to the conversation assistant. "
                f"Use exposed_only=false if you need to inspect it."
            )

        state = hass.states.get(entity_id)
        friendly = _s(state.attributes.get("friendly_name")) if state else ""
        friendly = friendly or _s(entry.name) or _s(entry.original_name) or entity_id
        area = _area_for(entry) or "-"
        state_val = state.state if state else "unknown"

        attrs: list[str] = []
        if state:
            for key in _INTERESTING_ATTRS:
                if key not in state.attributes:
                    continue
                val = state.attributes[key]
                if key == "brightness":
                    try:
                        pct = round(int(val) / 255 * 100)
                        attrs.append(f"  brightness: {pct}%")
                        continue
                    except (TypeError, ValueError):
                        pass
                attrs.append(f"  {key}: {val}")

        lines = [
            f"entity_id: {entity_id}",
            f"name: {friendly}",
            f"area: {area}",
            f"state: {state_val}",
        ]
        if attrs:
            lines.append("attributes:")
            lines.extend(attrs)
        if entity_id.startswith("weather."):
            forecast_block = await self._fetch_weather_forecast(entity_id)
            if forecast_block:
                lines.append(forecast_block)
        return "\n".join(lines)

    async def _fetch_weather_forecast(self, entity_id: str) -> str:
        """Call weather.get_forecasts service and format daily forecast block.

        Returns "" if entity does not support forecast service or call errors.
        """
        try:
            resp = await self._hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": entity_id, "type": "daily"},
                blocking=True,
                return_response=True,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("weather.get_forecasts failed for %s: %s", entity_id, exc)
            return ""
        data = (resp or {}).get(entity_id) or {}
        forecast = data.get("forecast") or []
        if not forecast:
            return ""
        out = ["forecast (daily, next 5):"]
        for item in forecast[:5]:
            dt = item.get("datetime", "?")
            date = dt.split("T")[0] if isinstance(dt, str) else "?"
            cond = item.get("condition", "?")
            parts = [f"  {date} {cond}"]
            hi = item.get("temperature")
            lo = item.get("templow")
            precip = item.get("precipitation_probability")
            if hi is not None:
                parts.append(f"hi:{hi}°")
            if lo is not None:
                parts.append(f"lo:{lo}°")
            if precip is not None:
                parts.append(f"precip:{precip}%")
            out.append(" ".join(parts))
        return "\n".join(out)

    def _auto_search_recovery(
        self,
        user_message: str,
        device_id: str | None,
        exposed_only: bool = True,
        has_hass_actuators: bool = True,
    ) -> str:
        """When set_area_state(area='all') is rejected, run a search using
        keywords from the user message and bias toward the caller's area.

        Returns a formatted hit list ready to feed back to the model.
        """
        # Strip filler: "Turn off the air purifier." → "air purifier".
        # Word-bounded on purpose: raw substring strips mangled device
        # nouns (" an" ate the head of "Anlage" → query 'lage').
        msg = (user_message or "").lower()
        msg = re.sub(
            r"\b(?:turn|switch|toggle|schalte?|mach|on|off|an|aus|ein|"
            r"the|a|der|die|das|den|eine?|bitte|please)\b",
            " ",
            msg,
        )
        query = " ".join(msg.split()).strip(".?! ")
        if not query:
            query = (user_message or "").strip()

        # Resolve caller's area name if possible
        caller_area: str | None = None
        if device_id and self._hass is not None:
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
                    caller_area = _s(a.name) if a else None
            except Exception:  # noqa: BLE001
                pass

        raw = self._search_entities(query, limit=10, exposed_only=exposed_only)
        if caller_area:
            head, _, body = raw.partition("\n")
            lines = body.splitlines()
            same = [l for l in lines if l.endswith(f"@ {caller_area}")]
            other = [l for l in lines if not l.endswith(f"@ {caller_area}")]
            if same:
                # Strong narrowing: when there's a hit in the caller's
                # area, return ONLY that one. Prevents the model from
                # firing HassTurnOff on every match across the flat.
                if has_hass_actuators:
                    directive = (
                        f"Caller is in {caller_area!r}. Use ONLY this one "
                        f"entity. Call HassTurnOn or HassTurnOff EXACTLY ONCE."
                    )
                else:
                    directive = (
                        f"Caller is in {caller_area!r}. This is the matching "
                        "entity, but no direct device-control tool is "
                        "available — report it to the user instead of "
                        "calling set_area_state."
                    )
                return f"1 match (caller-area filtered):\n{directive}\n" + same[0]
            head_extra = (
                f"  (caller is in {caller_area!r} — no matches there; "
                f"the entries below are in OTHER areas — pick AT MOST ONE "
                f"and confirm with the user, do NOT actuate all of them)"
            )
            return f"{head}\n{head_extra}\n" + "\n".join(other)
        return raw

    def _search_entities(
        self, query: str, limit: int = 10, exposed_only: bool = True
    ) -> str:
        if not query:
            return "search_entities: empty query."
        hass = self._hass
        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)
        area_reg = ar.async_get(hass)
        needle = query.lower()

        def _area_name(entry) -> str | None:
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if not area_id:
                return None
            a = area_reg.async_get_area(area_id)
            return _s(a.name) if a else None

        hits: list[tuple[str, str, str]] = []
        for entity_id, entry in ent_reg.entities.items():
            if exposed_only and not self._is_exposed(entity_id):
                continue
            candidates = [entity_id, _s(entry.name), _s(entry.original_name)]
            candidates.extend(_s(a) for a in (entry.aliases or ()))
            state = hass.states.get(entity_id)
            if state:
                candidates.append(_s(state.attributes.get("friendly_name")))
            if any(c and needle in c.lower() for c in candidates):
                friendly = _s(state.attributes.get("friendly_name")) if state else ""
                friendly = friendly or _s(entry.name) or _s(entry.original_name) or entity_id
                hits.append((entity_id, friendly, _area_name(entry) or "-"))
                if len(hits) > limit:
                    break

        if not hits:
            return f"No match for {query!r}."
        truncated = len(hits) > limit
        hits = hits[:limit]
        lines = [f"- {eid} — {fn} @ {a}" for eid, fn, a in hits]
        header = f"{len(hits)} match(es) for {query!r}"
        if truncated:
            header += " (more exist — narrow the query or raise limit)"
        return _cap(lines, header=header + ":\n")

    def _resolve_area(self, area: str):
        """Match an area by exact name or alias (case-insensitive). Returns
        the AreaEntry or None."""
        needle = (area or "").strip().lower()
        if not needle:
            return None
        area_reg = ar.async_get(self._hass)
        for a in area_reg.async_list_areas():
            if _s(a.name).lower() == needle:
                return a
            aliases = getattr(a, "aliases", None) or ()
            if any(_s(al).lower() == needle for al in aliases):
                return a
        return None

    def _area_named_in_message(self, message: str) -> str | None:
        """Return the area name/alias the user actually said, if any.

        Small models routinely omit the ``area`` argument even when the
        user named a room ("switch the kitchen lights off"), which used to
        silently fall back to the calling satellite's room — the WRONG
        room. Longest match wins so "living room" beats a hypothetical
        "room" alias. Whole-home keywords are skipped: they are handled by
        the explicit-sweep branch, not as an area name.
        """
        text = (message or "").lower()
        if not text:
            return None
        best: str | None = None
        try:
            area_reg = ar.async_get(self._hass)
            for a in area_reg.async_list_areas():
                aliases = getattr(a, "aliases", None) or ()
                for cand in (_s(a.name), *(_s(al) for al in aliases)):
                    cand = cand.strip().lower()
                    if not cand or cand in _SWEEP_ALL_KEYWORDS:
                        continue
                    if best is not None and len(cand) <= len(best):
                        continue
                    if re.search(rf"\b{re.escape(cand)}\b", text):
                        best = cand
        except Exception:  # noqa: BLE001
            _LOGGER.debug("area-in-message scan failed", exc_info=True)
        return best
