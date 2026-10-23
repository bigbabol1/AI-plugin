"""What the last action touched, so a pronoun can find it.

"Turn it back on", "make it warmer in there" — the model needs the
entity and the area of the action it just took. This resolves both from
the registries and renders the [LAST ACTION] block.
"""

from __future__ import annotations

import logging

from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

_LOGGER = logging.getLogger(__name__)


class EntityContextMixin:
    """Entity / area resolution for pronoun context. Mixed into Orchestrator."""

    def _resolve_entity_area(self, name_or_id: str) -> tuple[str | None, str | None]:
        """Resolve a user-facing name (or entity_id) to (entity_id, area_name).

        Reads HA's entity_registry, device_registry, area_registry, and state
        machine to find the matching entity and its containing area. Used so
        [LAST ACTION] can carry the area for pronoun resolution ('that room',
        'there', 'in here').
        """
        hass = self._hass
        if hass is None:
            return (None, None)
        ent_reg = er.async_get(hass)
        area_reg = ar.async_get(hass)
        dev_reg = dr.async_get(hass)

        def _area_for(entry) -> str | None:
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
            if not area_id:
                return None
            area = area_reg.async_get_area(area_id)
            return area.name if area else None

        # Direct entity_id match
        if "." in name_or_id:
            entry = ent_reg.async_get(name_or_id)
            if entry:
                return (name_or_id, _area_for(entry))

        # Name / alias / friendly_name match — case-insensitive exact.
        # HA's newer registries return ComputedNameType (lazy-evaluated) for
        # some .name fields, which is not a plain str and has no .strip().
        # Coerce via str() so the comparison never blows up on registry quirks.
        def _s(val: object) -> str:
            if val is None:
                return ""
            return val if isinstance(val, str) else str(val)

        needle = name_or_id.strip().lower()
        for entity_id, entry in ent_reg.entities.items():
            candidates = [_s(entry.name), _s(entry.original_name)]
            candidates.extend(_s(a) for a in (entry.aliases or ()))
            state = hass.states.get(entity_id)
            if state:
                candidates.append(_s(state.attributes.get("friendly_name")))
            if any(c and c.strip().lower() == needle for c in candidates):
                return (entity_id, _area_for(entry))
        return (None, None)

    def _extract_entity_context(self, tool_msgs: list[dict]) -> str | None:
        """Build a [LAST ACTION] string including entity_id and area.

        Pronouns like 'that room', 'there', 'in here' require the model to
        know the area of the last-controlled entity. Storing just the name
        is not enough — HA accepts user-facing names but areas live in the
        registry.
        """
        import json as _json
        parts: list[str] = []
        for msg in tool_msgs:
            if msg.get("role") != "assistant" or "tool_calls" not in msg:
                continue
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = (
                        _json.loads(args_raw)
                        if isinstance(args_raw, str)
                        else args_raw
                    )
                except Exception:  # noqa: BLE001
                    continue
                target = (
                    args.get("entity_id")
                    or args.get("name")
                    or args.get("entity")
                )
                area_arg = args.get("area")
                if not target and not area_arg:
                    continue
                if target:
                    entity_id, resolved_area = self._resolve_entity_area(target)
                    area = resolved_area or area_arg
                    canonical = entity_id or target
                    suffix = f" @ {area}" if area else ""
                    parts.append(f"{name}({target} → {canonical}{suffix})")
                else:
                    parts.append(f"{name}(area={area_arg})")
        return "; ".join(parts) if parts else None
