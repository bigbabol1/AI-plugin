"""Tests for the deterministic plural-domain sweep shortcut.

Regression cover for the reported bug: "switch all lights off" spoken to a
bedroom satellite did nothing useful — the model either promised the action
without calling a tool, or called set_area_state with no area (= that room
only), leaving the rest of the flat lit. The sweep shortcut takes the
command away from the model entirely.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.ai_plugin import shortcuts
from custom_components.ai_plugin.shortcuts import async_try_domain_sweep_shortcut


# ── stubs ─────────────────────────────────────────────────────────────────────


def _area(area_id, name, aliases=()):
    return SimpleNamespace(id=area_id, name=name, aliases=set(aliases))


def _entity(entity_id, *, area_id=None, device_id=None, name=""):
    return SimpleNamespace(
        entity_id=entity_id, name=name, original_name=name,
        aliases=set(), area_id=area_id, device_id=device_id,
    )


AREAS = [
    _area("a_kit", "Kitchen", aliases=("Küche",)),
    _area("a_bed", "Bedroom", aliases=("Schlafzimmer",)),
]
ENTITIES = [
    _entity("light.kitchen_ceiling", area_id="a_kit"),
    _entity("light.kitchen_counter", area_id="a_kit"),
    _entity("light.bed_left", area_id="a_bed"),
    _entity("fan.bedroom_fan", area_id="a_bed"),
    _entity("switch.kettle", area_id="a_kit"),
    _entity("light.no_area"),
]


def _make_hass(entities=None, areas=None, devices=None):
    hass = MagicMock()
    ent_reg = MagicMock()
    ent_reg.entities = {e.entity_id: e for e in (entities or ENTITIES)}
    dev_reg = MagicMock()
    dev_map = {d.id: d for d in (devices or [])}
    dev_reg.async_get.side_effect = lambda did: dev_map.get(did)
    area_reg = MagicMock()
    area_reg.async_list_areas.return_value = list(areas or AREAS)
    hass._regs = (ent_reg, dev_reg, area_reg)
    hass.states = MagicMock()
    hass.states.get.return_value = None
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass, ent_reg, dev_reg, area_reg


@pytest.fixture
def patched(monkeypatch):
    def _start(hass, ent_reg, dev_reg, area_reg):
        monkeypatch.setattr(shortcuts.er, "async_get", lambda h: ent_reg)
        monkeypatch.setattr(shortcuts.dr, "async_get", lambda h: dev_reg)
        monkeypatch.setattr(shortcuts.ar, "async_get", lambda h: area_reg)
        monkeypatch.setattr(shortcuts.er, "async_entries_for_device",
                            lambda reg, did: [])
        monkeypatch.setattr(shortcuts, "async_should_expose", lambda h, a, e: True)
    return _start


def _call(hass):
    assert hass.services.async_call.await_count == 1
    args, kwargs = hass.services.async_call.await_args
    data = args[2] if len(args) > 2 else kwargs.get("service_data") or {}
    return args[0], args[1], data["entity_id"]


# ── whole-home sweeps ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "msg,lang,service",
    [
        ("Switch all lights off.", "en", "turn_off"),   # the reported phrasing
        ("switch off all lights", "en", "turn_off"),
        ("turn off all the lights", "en", "turn_off"),
        ("turn on all lights", "en", "turn_on"),
        ("all lights off", "en", "turn_off"),
        ("mach alle Lichter aus", "de", "turn_off"),
        ("schalte alle Lichter ein", "de", "turn_on"),
        ("alle Lichter aus", "de", "turn_off"),
        ("éteins toutes les lumières", "fr", "turn_off"),
        ("apaga todas las luces", "es", "turn_off"),
        ("desliga todas as luzes", "pt", "turn_off"),
    ],
)
async def test_all_lights_sweeps_every_area(patched, msg, lang, service):
    """An explicit 'all' hits every exposed light, in every area."""
    hass, ent_reg, dev_reg, area_reg = _make_hass()
    patched(hass, ent_reg, dev_reg, area_reg)

    result = await async_try_domain_sweep_shortcut(hass, msg, lang=lang)

    assert result == (True, ""), f"{msg!r} ({lang}) should sweep silently"
    domain, svc, ids = _call(hass)
    assert (domain, svc) == ("homeassistant", service)
    assert ids == [
        "light.bed_left",
        "light.kitchen_ceiling",
        "light.kitchen_counter",
        "light.no_area",
    ]


@pytest.mark.asyncio
async def test_all_lights_beats_caller_area(patched):
    """The satellite's own room must not shrink an explicit 'all'."""
    devices = [SimpleNamespace(id="sat_bed", area_id="a_bed")]
    hass, ent_reg, dev_reg, area_reg = _make_hass(devices=devices)
    patched(hass, ent_reg, dev_reg, area_reg)

    result = await async_try_domain_sweep_shortcut(
        hass, "switch all lights off", lang="en", device_id="sat_bed"
    )

    assert result == (True, "")
    _, _, ids = _call(hass)
    assert len(ids) == 4


# ── scoped sweeps ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_named_room_scopes_the_sweep(patched):
    hass, ent_reg, dev_reg, area_reg = _make_hass()
    patched(hass, ent_reg, dev_reg, area_reg)

    result = await async_try_domain_sweep_shortcut(
        hass, "turn off all lights in the kitchen", lang="en"
    )

    assert result == (True, "")
    _, svc, ids = _call(hass)
    assert svc == "turn_off"
    assert ids == ["light.kitchen_ceiling", "light.kitchen_counter"]


@pytest.mark.asyncio
async def test_named_room_alias_de(patched):
    hass, ent_reg, dev_reg, area_reg = _make_hass()
    patched(hass, ent_reg, dev_reg, area_reg)

    result = await async_try_domain_sweep_shortcut(
        hass, "mach die Lichter in der Küche aus", lang="de"
    )

    assert result == (True, "")
    _, _, ids = _call(hass)
    assert ids == ["light.kitchen_ceiling", "light.kitchen_counter"]


@pytest.mark.asyncio
async def test_plural_without_all_uses_caller_room(patched):
    """"lights off" from a bedroom satellite stays in the bedroom."""
    devices = [SimpleNamespace(id="sat_bed", area_id="a_bed")]
    hass, ent_reg, dev_reg, area_reg = _make_hass(devices=devices)
    patched(hass, ent_reg, dev_reg, area_reg)

    result = await async_try_domain_sweep_shortcut(
        hass, "lights off", lang="en", device_id="sat_bed"
    )

    assert result == (True, "")
    _, svc, ids = _call(hass)
    assert (svc, ids) == ("turn_off", ["light.bed_left"])


@pytest.mark.asyncio
async def test_fans_sweep_only_fan_domain(patched):
    devices = [SimpleNamespace(id="sat_bed", area_id="a_bed")]
    hass, ent_reg, dev_reg, area_reg = _make_hass(devices=devices)
    patched(hass, ent_reg, dev_reg, area_reg)

    result = await async_try_domain_sweep_shortcut(
        hass, "turn the fans off", lang="en", device_id="sat_bed"
    )

    assert result == (True, "")
    _, _, ids = _call(hass)
    assert ids == ["fan.bedroom_fan"]


# ── fall-through: never guess ─────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "msg,lang",
    [
        ("turn off the light", "en"),          # singular → single-device path
        ("mach das Licht aus", "de"),          # singular → single-device path
        ("turn off the TV", "en"),             # named device, not a domain
        ("what time is it", "en"),             # not an action at all
        ("open the blinds", "en"),             # cover verbs are out of scope
        ("lights off", "en"),                  # no room, no 'all', no caller
    ],
)
async def test_fall_through_cases(patched, msg, lang):
    hass, ent_reg, dev_reg, area_reg = _make_hass()
    patched(hass, ent_reg, dev_reg, area_reg)

    result = await async_try_domain_sweep_shortcut(hass, msg, lang=lang)

    assert result is None
    hass.services.async_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_unexposed_lights_are_never_swept(monkeypatch):
    hass, ent_reg, dev_reg, area_reg = _make_hass()
    monkeypatch.setattr(shortcuts.er, "async_get", lambda h: ent_reg)
    monkeypatch.setattr(shortcuts.dr, "async_get", lambda h: dev_reg)
    monkeypatch.setattr(shortcuts.ar, "async_get", lambda h: area_reg)
    monkeypatch.setattr(shortcuts, "async_should_expose", lambda h, a, e: False)

    result = await async_try_domain_sweep_shortcut(
        hass, "switch all lights off", lang="en"
    )

    # Nothing exposed to act on → defer to the LLM, do not actuate.
    assert result is None
    hass.services.async_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_service_failure_falls_through(patched):
    hass, ent_reg, dev_reg, area_reg = _make_hass()
    patched(hass, ent_reg, dev_reg, area_reg)
    hass.services.async_call = AsyncMock(side_effect=RuntimeError("boom"))

    result = await async_try_domain_sweep_shortcut(
        hass, "switch all lights off", lang="en"
    )

    assert result is None
