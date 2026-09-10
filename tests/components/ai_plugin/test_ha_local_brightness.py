"""Tests for set_brightness — relative light stepping (v0.9.47).

HA core has no relative brightness intent: HassLightSet is absolute and
brightness 0 turns the light OFF. Asked to turn "brighter" into a number
with no knowledge of the current one, the model picked the extremes —
"brighter" set every living-room lamp to 100% and "dim the lights"
turned them all off. These tests pin the behaviour that replaced it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin.tools.ha_local import (
    HALocalToolRegistry,
    TOOL_NAMES,
    TOOL_SCHEMAS,
    _BRIGHTNESS_FLOOR_PCT,
    _BRIGHTNESS_STEP_PP,
)


def _area(area_id: str, name: str, aliases: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(id=area_id, name=name, aliases=set(aliases))


def _entity(entity_id: str, *, area_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        entity_id=entity_id, area_id=area_id, device_id=None,
        name="", original_name="", aliases=set(),
    )


def _state(state: str, brightness: int | None):
    attrs = {}
    if brightness is not None:
        attrs["brightness"] = brightness
    return SimpleNamespace(state=state, attributes=attrs)


def _make_hass(areas, entities, states: dict):
    hass = MagicMock()
    area_reg = MagicMock()
    area_reg.async_list_areas.return_value = list(areas)
    area_reg.async_get_area.side_effect = lambda aid: next(
        (a for a in areas if a.id == aid), None
    )
    ent_reg = MagicMock()
    ent_reg.entities = {e.entity_id: e for e in entities}
    ent_reg.async_get.side_effect = lambda eid: ent_reg.entities.get(eid)
    dev_reg = MagicMock()
    dev_reg.async_get.return_value = None
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states = MagicMock()
    hass.states.get.side_effect = lambda eid: states.get(eid)
    return hass, area_reg, ent_reg, dev_reg


@pytest.fixture
def patched_registries():
    started: list = []

    def start(area_reg, ent_reg, dev_reg):
        for target, val in (
            ("ar.async_get", area_reg),
            ("er.async_get", ent_reg),
            ("dr.async_get", dev_reg),
        ):
            p = patch(
                f"custom_components.ai_plugin.tools.ha_local.{target}",
                return_value=val,
            )
            p.start()
            started.append(p)

    yield start
    for p in started:
        p.stop()


def _pct(raw: int) -> int:
    return round(raw / 255 * 100)


def _calls(hass):
    """Return {brightness_pct: sorted(entity_ids)} for every turn_on issued."""
    out = {}
    for call in hass.services.async_call.await_args_list:
        args, kwargs = call
        assert args[0] == "light"
        assert args[1] == "turn_on", f"unexpected service {args[1]!r}"
        pct = args[2]["brightness_pct"]
        out[pct] = sorted(kwargs["target"]["entity_id"])
    return out


# ── registration ─────────────────────────────────────────────────────────────


def test_set_brightness_registered_in_both_places() -> None:
    """A tool in TOOL_SCHEMAS but missing from TOOL_NAMES is SENT to the
    model, called by it, then routed past ha_local to MCP where it fails —
    and the model silently falls back to HassLightSet. That shipped once."""
    assert "set_brightness" in TOOL_NAMES
    schema = next(
        (s for s in TOOL_SCHEMAS if s["function"]["name"] == "set_brightness"),
        None,
    )
    assert schema is not None, "set_brightness schema missing"
    params = schema["function"]["parameters"]
    assert set(params["properties"]["command"]["enum"]) == {
        "brighter", "dimmer", "set",
    }


# ── stepping ─────────────────────────────────────────────────────────────────


async def test_brighter_steps_each_light_from_its_own_level(
    patched_registries,
) -> None:
    areas = [_area("a1", "living room", ("Wohnzimmer",))]
    ents = [
        _entity("light.one", area_id="a1"),
        _entity("light.two", area_id="a1"),
        _entity("light.high", area_id="a1"),
    ]
    states = {
        "light.one": _state("on", 51),    # 20%
        "light.two": _state("on", 128),   # 50%
        "light.high": _state("on", 242),  # 95% → clamps at 100
    }
    hass, ar_, er_, dr_ = _make_hass(areas, ents, states)
    patched_registries(ar_, er_, dr_)

    out = await HALocalToolRegistry(hass)._set_brightness(
        command="brighter", area="Wohnzimmer"
    )
    assert out.startswith("OK")
    assert _calls(hass) == {
        20 + _BRIGHTNESS_STEP_PP: ["light.one"],
        50 + _BRIGHTNESS_STEP_PP: ["light.two"],
        100: ["light.high"],
    }


async def test_dimmer_never_turns_a_light_off(patched_registries) -> None:
    """The bug this tool exists for: 'dim the lights' must not mean off."""
    areas = [_area("a1", "living room")]
    ents = [_entity("light.low", area_id="a1")]
    # 12% — one full step below the floor.
    states = {"light.low": _state("on", 31)}
    hass, ar_, er_, dr_ = _make_hass(areas, ents, states)
    patched_registries(ar_, er_, dr_)

    await HALocalToolRegistry(hass)._set_brightness(
        command="dimmer", area="living room"
    )
    calls = _calls(hass)
    assert calls == {_BRIGHTNESS_FLOOR_PCT: ["light.low"]}
    assert all(pct >= _BRIGHTNESS_FLOOR_PCT for pct in calls)
    services = [c[0][1] for c in hass.services.async_call.await_args_list]
    assert "turn_off" not in services


async def test_dimmer_skips_lights_already_off(patched_registries) -> None:
    areas = [_area("a1", "living room")]
    ents = [_entity("light.on", area_id="a1"), _entity("light.off", area_id="a1")]
    states = {
        "light.on": _state("on", 128),
        "light.off": _state("off", None),
    }
    hass, ar_, er_, dr_ = _make_hass(areas, ents, states)
    patched_registries(ar_, er_, dr_)

    out = await HALocalToolRegistry(hass)._set_brightness(
        command="dimmer", area="living room"
    )
    assert _calls(hass) == {30: ["light.on"]}
    assert "already off" in out


async def test_brighter_turns_an_off_light_on_at_one_step(
    patched_registries,
) -> None:
    areas = [_area("a1", "living room")]
    ents = [_entity("light.off", area_id="a1")]
    states = {"light.off": _state("off", None)}
    hass, ar_, er_, dr_ = _make_hass(areas, ents, states)
    patched_registries(ar_, er_, dr_)

    await HALocalToolRegistry(hass)._set_brightness(
        command="brighter", area="living room"
    )
    assert _calls(hass) == {_BRIGHTNESS_STEP_PP: ["light.off"]}


# ── scope safety ─────────────────────────────────────────────────────────────


async def test_omitted_area_does_not_sweep_the_whole_home(
    patched_registries,
) -> None:
    """An absent scope must never widen to every light in the house."""
    areas = [_area("a1", "living room"), _area("a2", "bedroom")]
    ents = [_entity("light.lr", area_id="a1"), _entity("light.bed", area_id="a2")]
    states = {
        "light.lr": _state("on", 128),
        "light.bed": _state("on", 128),
    }
    hass, ar_, er_, dr_ = _make_hass(areas, ents, states)
    patched_registries(ar_, er_, dr_)

    out = await HALocalToolRegistry(hass)._set_brightness(
        command="brighter", area=None, device_id=None, allow_sweep=False
    )
    assert "name a room" in out
    hass.services.async_call.assert_not_awaited()


async def test_explicit_all_may_sweep(patched_registries) -> None:
    areas = [_area("a1", "living room"), _area("a2", "bedroom")]
    ents = [_entity("light.lr", area_id="a1"), _entity("light.bed", area_id="a2")]
    states = {
        "light.lr": _state("on", 128),
        "light.bed": _state("on", 128),
    }
    hass, ar_, er_, dr_ = _make_hass(areas, ents, states)
    patched_registries(ar_, er_, dr_)

    out = await HALocalToolRegistry(hass)._set_brightness(
        command="brighter", area="all", allow_sweep=True
    )
    assert out.startswith("OK")
    assert _calls(hass) == {70: ["light.bed", "light.lr"]}


# ── absolute ─────────────────────────────────────────────────────────────────


async def test_set_requires_a_level(patched_registries) -> None:
    hass, ar_, er_, dr_ = _make_hass([_area("a1", "living room")], [], {})
    patched_registries(ar_, er_, dr_)
    out = await HALocalToolRegistry(hass)._set_brightness(
        command="set", area="living room"
    )
    assert "needs level" in out
    hass.services.async_call.assert_not_awaited()
