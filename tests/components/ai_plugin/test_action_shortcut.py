"""Tests for the deterministic on/off shortcut (shortcuts.async_try_action_shortcut).

Covers the per-language action_on/action_off regexes, name/alias resolution,
the exact→substring→caller-area tiebreak, and fall-through on miss/ambiguity.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin import shortcuts
from custom_components.ai_plugin.shortcuts import async_try_action_shortcut


# ── stubs ─────────────────────────────────────────────────────────────────────


def _entity(entity_id, *, name="", original_name="", aliases=(), area_id=None, device_id=None):
    return SimpleNamespace(
        entity_id=entity_id, name=name, original_name=original_name,
        aliases=set(aliases), area_id=area_id, device_id=device_id,
    )


def _make_hass(entities, devices=None):
    hass = MagicMock()
    ent_reg = MagicMock()
    ent_reg.entities = {e.entity_id: e for e in entities}
    dev_reg = MagicMock()
    dev_map = {d.id: d for d in (devices or [])}
    dev_reg.async_get.side_effect = lambda did: dev_map.get(did)
    hass._regs = (ent_reg, dev_reg)
    hass.states = MagicMock()
    hass.states.get.return_value = None  # resolve by registry name/alias only
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass, ent_reg, dev_reg


@pytest.fixture
def patched(monkeypatch):
    """Patch shortcuts.er/dr registries + exposure gate (expose everything)."""
    def _start(hass, ent_reg, dev_reg):
        monkeypatch.setattr(shortcuts.er, "async_get", lambda h: ent_reg)
        monkeypatch.setattr(shortcuts.dr, "async_get", lambda h: dev_reg)
        monkeypatch.setattr(shortcuts, "async_should_expose", lambda h, a, e: True)
    return _start


def _last_call(hass):
    """Return (service, entity_id) of the last homeassistant service call."""
    assert hass.services.async_call.await_count == 1
    args, kwargs = hass.services.async_call.await_args
    domain, service = args[0], args[1]
    data = args[2] if len(args) > 2 else kwargs.get("service_data") or {}
    return domain, service, data["entity_id"]


# ── on/off resolution across languages ────────────────────────────────────────


TV = lambda: _entity("switch.tv", name="TV switch", aliases=("TV", "Fernseher"))
LAMP = lambda: _entity("light.lamp", name="Lamp", aliases=("Lampe", "lámpara", "lâmpada", "lampa"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "msg,lang,expect_service,expect_eid",
    [
        # English — the phrasings that previously failed via voice
        ("switch TV on", "en", "turn_on", "switch.tv"),
        ("turn the TV on", "en", "turn_on", "switch.tv"),
        ("turn on the TV", "en", "turn_on", "switch.tv"),
        ("TV on", "en", "turn_on", "switch.tv"),
        ("switch TV off", "en", "turn_off", "switch.tv"),
        ("turn the TV off", "en", "turn_off", "switch.tv"),
        # German
        ("schalte den Fernseher ein", "de", "turn_on", "switch.tv"),
        ("mach den Fernseher an", "de", "turn_on", "switch.tv"),
        ("Fernseher an", "de", "turn_on", "switch.tv"),
        ("schalte den Fernseher aus", "de", "turn_off", "switch.tv"),
        ("Fernseher aus", "de", "turn_off", "switch.tv"),
        # French / Spanish / Portuguese / Polish (verb-first, on the lamp)
        ("allume la lampe", "fr", "turn_on", "light.lamp"),
        ("éteins la lampe", "fr", "turn_off", "light.lamp"),
        ("enciende la lámpara", "es", "turn_on", "light.lamp"),
        ("apaga la lámpara", "es", "turn_off", "light.lamp"),
        ("liga a lâmpada", "pt", "turn_on", "light.lamp"),
        ("desliga a lâmpada", "pt", "turn_off", "light.lamp"),
        ("włącz lampa", "pl", "turn_on", "light.lamp"),
        ("wyłącz lampa", "pl", "turn_off", "light.lamp"),
    ],
)
async def test_action_resolves_and_dispatches(patched, msg, lang, expect_service, expect_eid):
    hass, ent_reg, dev_reg = _make_hass([TV(), LAMP()])
    patched(hass, ent_reg, dev_reg)
    result = await async_try_action_shortcut(hass, msg, lang=lang)
    assert result == (True, ""), f"{msg!r} ({lang}) should be handled silently"
    domain, service, eid = _last_call(hass)
    assert domain == "homeassistant"
    assert service == expect_service
    assert eid == expect_eid


# ── fall-through cases (return None, never actuate) ────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "msg,lang",
    [
        ("what time is it", "en"),            # not an action
        ("turn on the kettle", "en"),         # unknown device
        ("how warm is the living room", "en"),
        ("schalte das radio ein", "de"),      # unknown device
    ],
)
async def test_action_misses_fall_through(patched, msg, lang):
    hass, ent_reg, dev_reg = _make_hass([TV(), LAMP()])
    patched(hass, ent_reg, dev_reg)
    result = await async_try_action_shortcut(hass, msg, lang=lang)
    assert result is None
    hass.services.async_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_ambiguous_without_caller_area_falls_through(patched):
    """Two entities both named 'lamp' and no caller area → defer to the LLM."""
    ents = [
        _entity("light.lamp_a", name="Lamp", area_id="kitchen"),
        _entity("light.lamp_b", name="Lamp", area_id="bedroom"),
    ]
    hass, ent_reg, dev_reg = _make_hass(ents)
    patched(hass, ent_reg, dev_reg)
    result = await async_try_action_shortcut(hass, "turn on the lamp", lang="en")
    assert result is None
    hass.services.async_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_caller_area_breaks_ambiguity(patched):
    """Same ambiguous match, but the caller's satellite is in the kitchen."""
    ents = [
        _entity("light.lamp_a", name="Lamp", area_id="kitchen"),
        _entity("light.lamp_b", name="Lamp", area_id="bedroom"),
    ]
    devices = [SimpleNamespace(id="sat1", area_id="kitchen")]
    hass, ent_reg, dev_reg = _make_hass(ents, devices)
    patched(hass, ent_reg, dev_reg)
    result = await async_try_action_shortcut(hass, "turn on the lamp", lang="en", device_id="sat1")
    assert result == (True, "")
    _, service, eid = _last_call(hass)
    assert service == "turn_on"
    assert eid == "light.lamp_a"


@pytest.mark.asyncio
async def test_unexposed_entity_not_actuated(monkeypatch):
    """An entity hidden from the conversation assistant must not be matched."""
    hass, ent_reg, dev_reg = _make_hass([TV()])
    monkeypatch.setattr(shortcuts.er, "async_get", lambda h: ent_reg)
    monkeypatch.setattr(shortcuts.dr, "async_get", lambda h: dev_reg)
    monkeypatch.setattr(shortcuts, "async_should_expose", lambda h, a, e: False)
    result = await async_try_action_shortcut(hass, "switch TV on", lang="en")
    assert result is None
    hass.services.async_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_string_registry_name_does_not_crash(patched):
    """HA 2026.6 entry.name can be a non-str sentinel (ComputedNameType) with
    no .lower(); iterating it must not abort resolution of the real device."""
    class _Computed:  # mimics HA's ComputedNameType
        pass
    ents = [_entity("light.weird", name=_Computed()), TV()]
    hass, ent_reg, dev_reg = _make_hass(ents)
    patched(hass, ent_reg, dev_reg)
    result = await async_try_action_shortcut(hass, "switch TV on", lang="en")
    assert result == (True, "")
    _, service, eid = _last_call(hass)
    assert service == "turn_on" and eid == "switch.tv"


# ── v0.9.31: cover open/close ─────────────────────────────────────────────────


async def test_cover_open_named(patched) -> None:
    hass, ent_reg, dev_reg = _make_hass(
        [_entity("cover.blinds", name="Blinds"),
         _entity("light.blinds_light", name="Blinds Light")]
    )
    patched(hass, ent_reg, dev_reg)

    result = await async_try_action_shortcut(hass, "open the blinds", lang="en")

    assert result == (True, "")
    domain, service, eid = _last_call(hass)
    assert (domain, service, eid) == ("cover", "open_cover", "cover.blinds")


async def test_cover_close_de(patched) -> None:
    hass, ent_reg, dev_reg = _make_hass([_entity("cover.rollladen", name="Rollladen")])
    patched(hass, ent_reg, dev_reg)

    result = await async_try_action_shortcut(
        hass, "Mach den Rollladen zu", lang="de"
    )

    assert result == (True, "")
    domain, service, eid = _last_call(hass)
    assert (domain, service, eid) == ("cover", "close_cover", "cover.rollladen")


async def test_cover_verbs_never_actuate_non_cover(patched) -> None:
    """'open X' must resolve only in the cover domain — no lights/switches."""
    hass, ent_reg, dev_reg = _make_hass([_entity("light.spotify", name="Spotify")])
    patched(hass, ent_reg, dev_reg)

    result = await async_try_action_shortcut(hass, "open spotify", lang="en")

    assert result is None
    hass.services.async_call.assert_not_awaited()


async def test_on_off_still_works_after_refactor(patched) -> None:
    hass, ent_reg, dev_reg = _make_hass([_entity("light.mood", name="Mood Light")])
    patched(hass, ent_reg, dev_reg)

    result = await async_try_action_shortcut(hass, "turn on the mood light", lang="en")

    assert result == (True, "")
    domain, service, eid = _last_call(hass)
    assert (domain, service, eid) == ("homeassistant", "turn_on", "light.mood")
