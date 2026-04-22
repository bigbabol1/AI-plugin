"""Tests for tools/ha_local.py (entity discovery + set_area_state)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin.tools.ha_local import (
    HALocalToolRegistry,
    TOOL_NAMES,
    TOOL_SCHEMAS,
)


# ── registry stubs ────────────────────────────────────────────────────────────


def _area(area_id: str, name: str, aliases: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(id=area_id, name=name, aliases=set(aliases))


def _entity(
    entity_id: str,
    *,
    area_id: str | None = None,
    device_id: str | None = None,
    name: str = "",
    original_name: str = "",
    aliases: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        entity_id=entity_id,
        area_id=area_id,
        device_id=device_id,
        name=name,
        original_name=original_name,
        aliases=set(aliases),
    )


def _device(device_id: str, area_id: str | None) -> SimpleNamespace:
    return SimpleNamespace(id=device_id, area_id=area_id)


def _make_hass(
    areas: list[SimpleNamespace],
    entities: list[SimpleNamespace],
    devices: list[SimpleNamespace] | None = None,
    exposed: set[str] | None = None,
    service_side_effect=None,
) -> MagicMock:
    """Build a hass mock with area/entity/device registries and a services layer.

    `exposed` is the set of entity_ids treated as exposed to the conversation
    assistant; if None, all entities are exposed.
    """
    hass = MagicMock()

    # area registry
    area_reg = MagicMock()
    area_reg.async_list_areas.return_value = list(areas)
    area_reg.async_get_area.side_effect = lambda aid: next(
        (a for a in areas if a.id == aid), None
    )

    # entity registry — `.entities` is a mapping-like {entity_id: entry}
    ent_reg = MagicMock()
    ent_reg.entities = {e.entity_id: e for e in entities}
    ent_reg.async_get.side_effect = lambda eid: ent_reg.entities.get(eid)

    # device registry
    dev_reg = MagicMock()
    dev_map = {d.id: d for d in (devices or [])}
    dev_reg.async_get.side_effect = lambda did: dev_map.get(did)

    hass._registries = (area_reg, ent_reg, dev_reg)  # keep refs alive

    # services
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock(side_effect=service_side_effect)

    # states (not needed for set_area_state, but stub to None)
    hass.states = MagicMock()
    hass.states.get.return_value = None

    return hass, area_reg, ent_reg, dev_reg


@pytest.fixture
def patched_registries():
    """Patch the module-level `ar`, `er`, `dr` references so `ar.async_get(hass)`
    etc. return the registry mocks we attached to the hass stub."""

    def _apply(hass, area_reg, ent_reg, dev_reg):
        p1 = patch(
            "custom_components.ai_plugin.tools.ha_local.ar.async_get",
            return_value=area_reg,
        )
        p2 = patch(
            "custom_components.ai_plugin.tools.ha_local.er.async_get",
            return_value=ent_reg,
        )
        p3 = patch(
            "custom_components.ai_plugin.tools.ha_local.dr.async_get",
            return_value=dev_reg,
        )
        p1.start()
        p2.start()
        p3.start()
        return (p1, p2, p3)

    started: list = []

    def start(hass, area_reg, ent_reg, dev_reg):
        started.extend(_apply(hass, area_reg, ent_reg, dev_reg))

    yield start
    for p in started:
        p.stop()


def test_set_area_state_schema_registered() -> None:
    """set_area_state must be exposed in TOOL_NAMES and TOOL_SCHEMAS."""
    assert "set_area_state" in TOOL_NAMES
    schema = next(
        (s for s in TOOL_SCHEMAS if s["function"]["name"] == "set_area_state"),
        None,
    )
    assert schema is not None, "set_area_state schema missing"
    params = schema["function"]["parameters"]
    assert set(params["required"]) == {"area", "domain", "action"}
    assert set(params["properties"]["domain"]["enum"]) == {
        "light", "switch", "fan", "media_player", "cover", "climate",
    }
    assert set(params["properties"]["action"]["enum"]) == {
        "turn_on", "turn_off", "toggle",
    }


@pytest.mark.asyncio
async def test_set_area_state_bad_domain(patched_registries) -> None:
    """Unknown domain returns a friendly error without touching services."""
    hass, ar_r, er_r, dr_r = _make_hass(areas=[_area("a1", "Kitchen")], entities=[])
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "kitchen", "domain": "heater", "action": "turn_on"},
    )
    assert out.startswith("Domain 'heater' not supported")
    hass.services.async_call.assert_not_called()


@pytest.mark.asyncio
async def test_set_area_state_bad_action_for_climate(patched_registries) -> None:
    """climate + toggle returns the domain/action-mismatch error."""
    hass, ar_r, er_r, dr_r = _make_hass(areas=[_area("a1", "Kitchen")], entities=[])
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "kitchen", "domain": "climate", "action": "toggle"},
    )
    assert "climate does not support toggle" in out
    hass.services.async_call.assert_not_called()


@pytest.mark.asyncio
async def test_set_area_state_unknown_area(patched_registries) -> None:
    """Unknown area returns 'Unknown area ...'."""
    hass, ar_r, er_r, dr_r = _make_hass(areas=[_area("a1", "Kitchen")], entities=[])
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "garden", "domain": "light", "action": "turn_off"},
    )
    assert out.startswith("Unknown area 'garden'")
    hass.services.async_call.assert_not_called()
