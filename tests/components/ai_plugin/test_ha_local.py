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


@pytest.mark.asyncio
async def test_set_area_state_success_light_turn_off(patched_registries) -> None:
    """Happy path: all exposed lights in the area get a single service call."""
    areas = [_area("a_kit", "Kitchen"), _area("a_bed", "Bedroom")]
    entities = [
        _entity("light.ceiling", area_id="a_kit"),
        _entity("light.counter", area_id="a_kit"),
        _entity("light.dining", area_id="a_kit"),
        _entity("light.bedroom_lamp", area_id="a_bed"),  # different area
        _entity("switch.kettle", area_id="a_kit"),        # different domain
    ]
    hass, ar_r, er_r, dr_r = _make_hass(areas=areas, entities=entities)
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "kitchen", "domain": "light", "action": "turn_off"},
    )
    assert out.startswith("OK — turn_off 3 light(s) in Kitchen")
    assert "light.ceiling" in out and "light.counter" in out and "light.dining" in out

    hass.services.async_call.assert_awaited_once()
    args, kwargs = hass.services.async_call.call_args
    assert args[0] == "light"
    assert args[1] == "turn_off"
    assert kwargs["target"] == {
        "entity_id": ["light.ceiling", "light.counter", "light.dining"],
    }
    assert kwargs["blocking"] is True


@pytest.mark.asyncio
async def test_set_area_state_cover_open_uses_open_cover(patched_registries) -> None:
    """cover+turn_on maps to the open_cover service."""
    areas = [_area("a1", "Hobbyroom")]
    entities = [_entity("cover.garage", area_id="a1")]
    hass, ar_r, er_r, dr_r = _make_hass(areas=areas, entities=entities)
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "hobbyroom", "domain": "cover", "action": "turn_on"},
    )
    assert "OK — turn_on 1 cover(s) in Hobbyroom" in out

    args, _kw = hass.services.async_call.call_args
    assert args[0] == "cover"
    assert args[1] == "open_cover"


@pytest.mark.asyncio
async def test_set_area_state_no_exposed_entities(patched_registries) -> None:
    """Area and domain exist but no entity is exposed → friendly message."""
    areas = [_area("a1", "Kitchen")]
    entities = [_entity("light.hidden", area_id="a1")]
    hass, ar_r, er_r, dr_r = _make_hass(areas=areas, entities=entities)
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    # force _is_exposed to return False for this test
    with patch.object(HALocalToolRegistry, "_is_exposed", return_value=False):
        out = await reg.call_tool(
            "set_area_state",
            {"area": "kitchen", "domain": "light", "action": "turn_off"},
        )
    assert out == "No exposed light in Kitchen."
    hass.services.async_call.assert_not_called()


@pytest.mark.asyncio
async def test_set_area_state_alias_match(patched_registries) -> None:
    """Alias match resolves to the canonical area name."""
    areas = [_area("a1", "Kitchen", aliases=("Küche",))]
    entities = [_entity("light.ceiling", area_id="a1")]
    hass, ar_r, er_r, dr_r = _make_hass(areas=areas, entities=entities)
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "küche", "domain": "light", "action": "turn_on"},
    )
    assert "in Kitchen" in out
    hass.services.async_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_area_state_device_inherits_area(patched_registries) -> None:
    """Entity with no area_id inherits its device's area_id."""
    areas = [_area("a1", "Kitchen")]
    devices = [_device("d1", area_id="a1")]
    entities = [_entity("light.stove", area_id=None, device_id="d1")]
    hass, ar_r, er_r, dr_r = _make_hass(areas=areas, entities=entities, devices=devices)
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "kitchen", "domain": "light", "action": "turn_on"},
    )
    assert "1 light(s) in Kitchen" in out
    args, _kw = hass.services.async_call.call_args
    assert _kw["target"] == {"entity_id": ["light.stove"]}


@pytest.mark.asyncio
async def test_set_area_state_service_exception_caught(patched_registries) -> None:
    """Service failure returns a bracketed error string, does not raise."""
    areas = [_area("a1", "Kitchen")]
    entities = [_entity("light.ceiling", area_id="a1")]
    hass, ar_r, er_r, dr_r = _make_hass(
        areas=areas,
        entities=entities,
        service_side_effect=RuntimeError("boom"),
    )
    patched_registries(hass, ar_r, er_r, dr_r)
    reg = HALocalToolRegistry(hass)

    out = await reg.call_tool(
        "set_area_state",
        {"area": "kitchen", "domain": "light", "action": "turn_on"},
    )
    assert out.startswith("[set_area_state failed:")
    assert "boom" in out
