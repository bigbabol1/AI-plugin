"""Tests for set_climate_mode — the thermostat mode HA's MCP server lacks.

HA 2026.7 offers HassClimateSetTemperature but nothing that changes the
HVAC mode. Asked to put the heating "back to auto", the voice model set
the temperature to 0 (refused by HA), then "turned on" the already
heating thermostat and confirmed a switch to auto that never happened.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin.tools.ha_local import (
    HALocalToolRegistry,
    TOOL_NAMES,
    TOOL_SCHEMAS,
)

_MODES = ["heat", "off", "auto"]


def _area(area_id: str, name: str, aliases: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(id=area_id, name=name, aliases=set(aliases))


def _entity(entity_id: str, *, area_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(entity_id=entity_id, area_id=area_id, device_id=None)


def _state(state: str, name: str, modes=_MODES) -> SimpleNamespace:
    return SimpleNamespace(
        state=state, attributes={"friendly_name": name, "hvac_modes": list(modes)}
    )


def _make_hass(areas, entities, states: dict, caller_area: str | None = None):
    hass = MagicMock()
    area_reg = MagicMock()
    area_reg.async_list_areas.return_value = list(areas)
    ent_reg = MagicMock()
    ent_reg.entities = {e.entity_id: e for e in entities}
    dev_reg = MagicMock()
    dev_reg.async_get.side_effect = lambda did: (
        SimpleNamespace(area_id=caller_area) if did == "sat" else None
    )
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states = MagicMock()
    hass.states.get.side_effect = lambda eid: states.get(eid)
    return hass, area_reg, ent_reg, dev_reg


@pytest.fixture
def registries():
    started: list = []

    def start(area_reg, ent_reg, dev_reg):
        for target, val in (
            ("ar.async_get", area_reg),
            ("er.async_get", ent_reg),
            ("dr.async_get", dev_reg),
        ):
            p = patch(
                f"custom_components.ai_plugin.tools.ha_local.climate.{target}",
                return_value=val,
            )
            p.start()
            started.append(p)

    yield start
    for p in started:
        p.stop()


def _mode_calls(hass) -> list[tuple[str, str]]:
    out = []
    for args, kwargs in hass.services.async_call.await_args_list:
        assert args[:2] == ("climate", "set_hvac_mode")
        out.append((kwargs["target"]["entity_id"], args[2]["hvac_mode"]))
    return out


def _one_thermostat(state: str = "heat", caller_area: str | None = "bath"):
    areas = [_area("living", "Wohnzimmer", ("living room",)), _area("bath", "Badezimmer")]
    ents = [_entity("climate.thermostat", area_id="living")]
    states = {"climate.thermostat": _state(state, "Thermostat")}
    return _make_hass(areas, ents, states, caller_area)


def test_set_climate_mode_registered_in_both_places() -> None:
    """In TOOL_SCHEMAS but not TOOL_NAMES, the call would be routed past
    ha_local to MCP and fail; that shipped once for set_brightness."""
    assert "set_climate_mode" in TOOL_NAMES
    schema = next(
        s for s in TOOL_SCHEMAS if s["function"]["name"] == "set_climate_mode"
    )
    params = schema["function"]["parameters"]
    assert "auto" in params["properties"]["mode"]["enum"]
    assert params["required"] == ["mode"]


async def test_back_to_auto_from_the_bathroom(registries) -> None:
    """The live turn: asked from a room without a thermostat, 'the
    heating' is the home's only one."""
    hass, *regs = _one_thermostat()
    registries(*regs)
    out = await HALocalToolRegistry(hass).call_tool(
        "set_climate_mode", {"mode": "auto"}, device_id="sat"
    )
    assert out.startswith("OK")
    assert "was heat" in out
    assert _mode_calls(hass) == [("climate.thermostat", "auto")]


@pytest.mark.parametrize("spoken", ["schedule", "Automatik", "Zeitplan", "automatic"])
async def test_schedule_words_mean_auto(registries, spoken) -> None:
    hass, *regs = _one_thermostat()
    registries(*regs)
    await HALocalToolRegistry(hass)._set_climate_mode(mode=spoken)
    assert _mode_calls(hass) == [("climate.thermostat", "auto")]


async def test_unsupported_mode_is_refused_without_a_call(registries) -> None:
    hass, *regs = _one_thermostat()
    registries(*regs)
    out = await HALocalToolRegistry(hass)._set_climate_mode(mode="cool")
    assert out.startswith("[")
    assert "heat, off, auto" in out
    assert _mode_calls(hass) == []


async def test_unknown_mode_word_is_refused(registries) -> None:
    hass, *regs = _one_thermostat()
    registries(*regs)
    out = await HALocalToolRegistry(hass)._set_climate_mode(mode="turbo")
    assert "unknown mode" in out
    assert _mode_calls(hass) == []


async def test_already_in_mode_reports_it_without_a_call(registries) -> None:
    hass, *regs = _one_thermostat(state="auto")
    registries(*regs)
    out = await HALocalToolRegistry(hass)._set_climate_mode(mode="auto")
    assert out == "OK — climate.thermostat already auto"
    assert _mode_calls(hass) == []


async def test_name_that_matches_nothing_means_the_only_thermostat(registries) -> None:
    hass, *regs = _one_thermostat()
    registries(*regs)
    await HALocalToolRegistry(hass)._set_climate_mode(mode="off", name="heating")
    assert _mode_calls(hass) == [("climate.thermostat", "off")]


async def test_room_alias_selects_its_thermostat(registries) -> None:
    hass, *regs = _one_thermostat()
    registries(*regs)
    await HALocalToolRegistry(hass)._set_climate_mode(mode="auto", area="living room")
    assert _mode_calls(hass) == [("climate.thermostat", "auto")]


def _two_thermostats(caller_area: str | None):
    areas = [_area("living", "Living room"), _area("bed", "Bedroom"), _area("bath", "Bathroom")]
    ents = [
        _entity("climate.living", area_id="living"),
        _entity("climate.bedroom", area_id="bed"),
    ]
    states = {
        "climate.living": _state("heat", "Living Thermostat"),
        "climate.bedroom": _state("heat", "Bedroom Thermostat"),
    }
    return _make_hass(areas, ents, states, caller_area)


async def test_several_thermostats_none_here_asks_instead_of_switching_all(
    registries,
) -> None:
    hass, *regs = _two_thermostats(caller_area="bath")
    registries(*regs)
    out = await HALocalToolRegistry(hass)._set_climate_mode(
        mode="off", device_id="sat"
    )
    assert "Several thermostats" in out
    assert _mode_calls(hass) == []


async def test_several_thermostats_uses_the_callers_room(registries) -> None:
    hass, *regs = _two_thermostats(caller_area="bed")
    registries(*regs)
    await HALocalToolRegistry(hass)._set_climate_mode(mode="off", device_id="sat")
    assert _mode_calls(hass) == [("climate.bedroom", "off")]


async def test_several_thermostats_named_room_without_one_is_refused(
    registries,
) -> None:
    hass, *regs = _two_thermostats(caller_area=None)
    registries(*regs)
    out = await HALocalToolRegistry(hass)._set_climate_mode(mode="off", area="Bathroom")
    assert out == "No exposed thermostat in Bathroom."
    assert _mode_calls(hass) == []


async def test_service_error_is_reported_as_a_failure(registries) -> None:
    hass, *regs = _one_thermostat()
    registries(*regs)
    hass.services.async_call.side_effect = RuntimeError("cloud timeout")
    out = await HALocalToolRegistry(hass)._set_climate_mode(mode="auto")
    assert out == "[set_climate_mode failed: cloud timeout]"
