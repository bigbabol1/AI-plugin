"""Doors and windows in list_entities; covers: classes, "except", awnings."""
# ruff: noqa: F811  (patched_registries is a fixture imported from test_ha_local)

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from .test_ha_local import _area, _entity, _make_hass, patched_registries  # noqa: F401

from custom_components.ai_plugin.tools.ha_local import HALocalToolRegistry
from custom_components.ai_plugin.tools.ha_local.lights import (
    WINDOW_COVERINGS,
    widen_cover_classes,
)


def _with_class(entity, device_class):
    entity.device_class = device_class
    return entity


def _states(hass, states: dict[str, str]):
    hass.states.get.side_effect = lambda eid: (
        SimpleNamespace(state=states[eid], attributes={"friendly_name": eid.split(".")[1].replace("_", " ").title()})
        if eid in states else None
    )


def _home(patched, states=None):
    areas = [
        _area("a_liv", "Living Room", ("Wohnzimmer",)),
        _area("a_bed", "Bedroom", ("Schlafzimmer",)),
        _area("a_bal", "Balcony", ("Balkon",)),
        _area("a_stu", "Studio"),
    ]
    entities = [
        _with_class(_entity("binary_sensor.front_door", area_id="a_liv"), "door"),
        _with_class(_entity("binary_sensor.bedroom_window", area_id="a_bed"), "window"),
        _with_class(_entity("binary_sensor.motion", area_id="a_liv"), "motion"),
        _with_class(_entity("cover.jalousie", area_id="a_liv"), "blind"),
        _with_class(_entity("cover.rollo_bed", area_id="a_bed"), "shutter"),
        _with_class(_entity("cover.rollo_studio", area_id="a_stu"), None),
        _with_class(_entity("cover.markise", area_id="a_bal"), "awning"),
    ]
    hass, ar_r, er_r, dr_r = _make_hass(areas=areas, entities=entities)
    patched(hass, ar_r, er_r, dr_r)
    _states(hass, states or {
        "binary_sensor.front_door": "off", "binary_sensor.bedroom_window": "on",
        "binary_sensor.motion": "on", "cover.jalousie": "open", "cover.rollo_bed": "open",
        "cover.rollo_studio": "open", "cover.markise": "closed",
    })
    reg = HALocalToolRegistry(hass)
    reg._is_exposed = lambda eid: True
    return hass, reg


async def test_an_open_window_is_found_by_asking_for_windows(patched_registries) -> None:
    """ "Are all doors and windows closed?" said yes with a window open."""
    _, reg = _home(patched_registries)
    out = await reg.call_tool("list_entities", {"domain": "window", "state": "open"})
    assert "binary_sensor.bedroom_window" in out and "[open]" in out
    out = await reg.call_tool("list_entities", {"domain": "door", "state": "open"})
    assert out.startswith("None of the 1 door sensor(s) is open")
    out = await reg.call_tool("list_entities", {"domain": "door"})
    assert "binary_sensor.front_door" in out and "[closed]" in out
    assert "motion" not in out


async def test_without_any_sensor_closed_is_never_implied(patched_registries) -> None:
    _, reg = _home(patched_registries)
    reg._is_exposed = lambda eid: not eid.startswith("binary_sensor.")
    out = await reg.call_tool("list_entities", {"domain": "window", "state": "open"})
    assert "UNKNOWN" in out and "never say they are closed" in out


async def test_other_binary_sensors_keep_on_and_off(patched_registries) -> None:
    _, reg = _home(patched_registries)
    out = await reg.call_tool("list_entities", {"domain": "binary_sensor"})
    assert "binary_sensor.motion" in out and "[on]" in out
    assert "binary_sensor.bedroom_window" in out and "[open]" in out


async def test_all_blinds_leave_the_awning_alone(patched_registries) -> None:
    hass, reg = _home(patched_registries)
    out = await reg.call_tool(
        "set_area_state", {"area": "all", "domain": "cover", "action": "turn_off"},
        user_message="Close all the blinds.",
    )
    ids = hass.services.async_call.call_args.kwargs["target"]["entity_id"]
    assert ids == ["cover.jalousie", "cover.rollo_bed", "cover.rollo_studio"]
    assert "Left as they are" in out and "Markise" in out


@pytest.mark.parametrize("message", [
    "Close all blinds except in the bedroom.",
    "Mach alle Rollos zu außer im Schlafzimmer.",
])
async def test_except_leaves_the_room_out_in_one_call(patched_registries, message) -> None:
    hass, reg = _home(patched_registries)
    out = await reg.call_tool(
        "set_area_state", {"area": "all", "domain": "cover", "action": "turn_off"},
        user_message=message,
    )
    ids = hass.services.async_call.call_args.kwargs["target"]["entity_id"]
    assert ids == ["cover.jalousie", "cover.rollo_studio"]
    assert "except Bedroom" in out


async def test_except_area_from_the_model_counts_too(patched_registries) -> None:
    hass, reg = _home(patched_registries)
    await reg.call_tool(
        "set_area_state",
        {"area": "all", "domain": "cover", "action": "turn_off", "except_area": "Studio"},
        user_message="Close all the blinds but leave the studio.",
    )
    ids = hass.services.async_call.call_args.kwargs["target"]["entity_id"]
    assert ids == ["cover.jalousie", "cover.rollo_bed"]


async def test_the_excluded_room_is_never_undone_afterwards(patched_registries) -> None:
    """The model closed everything, then opened the bedroom again."""
    hass, reg = _home(patched_registries)
    out = await reg.call_tool(
        "set_area_state", {"area": "Bedroom", "domain": "cover", "action": "turn_on"},
        user_message="Close all blinds except in the bedroom.",
    )
    assert out.startswith("[Not done: the user asked to leave Bedroom out.")
    hass.services.async_call.assert_not_awaited()


async def test_an_omitted_area_is_not_the_excluded_room(patched_registries) -> None:
    hass, reg = _home(patched_registries)
    await reg.call_tool(
        "set_area_state", {"domain": "cover", "action": "turn_off"},
        user_message="Close all blinds except in the bedroom.",
    )
    ids = hass.services.async_call.call_args.kwargs["target"]["entity_id"]
    assert "cover.rollo_bed" not in ids and ids


async def test_a_position_fallback_moves_window_coverings_of_the_room(patched_registries) -> None:
    hass, reg = _home(patched_registries)
    out = await reg._set_cover_position("Wohnzimmer", 50)
    assert out.startswith("OK — 1 cover(s) in Living Room to 50%")
    args, kwargs = hass.services.async_call.call_args
    assert args[:3] == ("cover", "set_cover_position", {"position": 50})
    assert kwargs["target"] == {"entity_id": ["cover.jalousie"]}
    assert (await reg._set_cover_position("Balkon", 20)) == "No window covering in Balcony."


@pytest.mark.parametrize(("name", "classes", "widened"), [
    ("HassTurnOff", ["blind"], True),
    ("HassSetPosition", ["shade"], True),
    ("HassTurnOn", "shutter", True),
    ("HassTurnOn", ["awning"], False),
    ("HassTurnOn", None, False),
    ("HassLightSet", ["blind"], False),
])
def test_one_window_covering_class_stands_for_all(name, classes, widened) -> None:
    args = {"area": "Studio"} if classes is None else {"area": "Studio", "device_class": classes}
    out = widen_cover_classes(name, args)
    if widened:
        assert set(out["device_class"]) == WINDOW_COVERINGS
    else:
        assert out == args


async def test_when_home_assistant_matches_nothing_the_plugin_moves_them() -> None:
    from .test_conversation import _baseline_orch
    from custom_components.ai_plugin.orchestrator import core

    orch = _baseline_orch()
    orch._mcp = MagicMock()
    orch._mcp.is_home_assistant_tool = MagicMock(return_value=True)
    orch._mcp.call_tool = AsyncMock(return_value="[Tool 'HassTurnOff' failed: no entities matched]")
    orch._ha_local = MagicMock()
    orch._ha_local.tool_names = set()
    orch._ha_local._set_area_state = AsyncMock(return_value="OK — turn_off 1 cover(s) in Studio")
    orch._hass = None
    out = await orch._dispatch_tool("HassTurnOff", {"area": "Studio", "device_class": ["blind"]}, None)
    sent = orch._mcp.call_tool.await_args.args[1]
    assert set(sent["device_class"]) == WINDOW_COVERINGS
    assert out == "OK — turn_off 1 cover(s) in Studio"
    orch._ha_local._set_area_state.assert_awaited_once_with("Studio", "cover", "turn_off", device_id=None)
    # No room and no satellite: never every cover in the home.
    orch._ha_local._set_area_state.reset_mock()
    out = await orch._dispatch_tool("HassTurnOff", {"device_class": ["blind"]}, None)
    assert out.startswith("[Tool")
    orch._ha_local._set_area_state.assert_not_awaited()
    assert core.WINDOW_COVERINGS == WINDOW_COVERINGS


async def test_doors_and_windows_in_one_call(patched_registries) -> None:
    _, reg = _home(patched_registries)
    out = await reg.call_tool("list_entities", {"domain": ["door", "window"], "state": "open"})
    assert out.startswith("None of the 1 door sensor(s) is open")
    assert "binary_sensor.bedroom_window" in out
