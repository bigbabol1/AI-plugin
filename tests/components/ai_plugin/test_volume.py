"""Louder and quieter by a step you can hear."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from .test_conversation import _baseline_orch  # noqa: F401  # isort: skip (stubs HA first)

from custom_components.ai_plugin.shortcuts.volume import async_step_volume

SPEAKER = "media_player.hobby"


def _hass(volume=0.4, ducks=None):
    hass = MagicMock()
    hass.data = {} if ducks is None else {"ai_plugin_music_ducks": {SPEAKER: ducks}}
    hass.states.get = lambda _: SimpleNamespace(
        state="playing", attributes={} if volume is None else {"volume_level": volume}
    )
    hass.services.async_call = AsyncMock()
    return hass


def _set_to(hass) -> float:
    call = hass.services.async_call.await_args
    assert call.args[:2] == ("media_player", "volume_set")
    assert call.kwargs["target"] == {"entity_id": SPEAKER}
    return call.args[2]["volume_level"]


@pytest.mark.parametrize(
    ("volume", "command", "expected"),
    [
        (0.4, "volume_down", 0.3),
        (0.4, "volume_up", 0.5),
        (0.05, "volume_down", 0.0),
        (0.95, "volume_up", 1.0),
    ],
)
async def test_a_step_you_can_hear(volume, command, expected) -> None:
    """The players' own step took 40% to 37%, which nobody hears."""
    hass = _hass(volume)
    await async_step_volume(hass, [SPEAKER], command)
    assert _set_to(hass) == expected


async def test_while_the_music_is_lowered_it_steps_from_the_real_volume() -> None:
    hass = _hass(0.12, {"level": 0.4, "holders": {"sat"}, "restored_at": 0.0})
    await async_step_volume(hass, [SPEAKER], "volume_down")
    assert _set_to(hass) == 0.3


async def test_just_after_the_music_came_back_too() -> None:
    hass = _hass(0.12, {"level": 0.4, "holders": set(), "restored_at": time.monotonic()})
    await async_step_volume(hass, [SPEAKER], "volume_down")
    assert _set_to(hass) == 0.3


async def test_an_old_saved_volume_no_longer_counts() -> None:
    """The volume may have been changed since."""
    hass = _hass(0.7, {"level": 0.4, "holders": set(), "restored_at": time.monotonic() - 60})
    await async_step_volume(hass, [SPEAKER], "volume_down")
    assert _set_to(hass) == 0.6


async def test_a_player_without_a_volume_uses_its_own_step() -> None:
    hass = _hass(None)
    await async_step_volume(hass, [SPEAKER], "volume_up")
    call = hass.services.async_call.await_args
    assert call.args[:2] == ("media_player", "volume_up")
    assert call.kwargs["target"] == {"entity_id": SPEAKER}


async def test_each_speaker_steps_from_its_own_volume() -> None:
    hass = _hass()
    levels = {"media_player.a": 0.2, "media_player.b": 0.8}
    hass.states.get = lambda eid: SimpleNamespace(state="playing", attributes={"volume_level": levels[eid]})
    await async_step_volume(hass, list(levels), "volume_down")
    calls = {
        c.kwargs["target"]["entity_id"]: c.args[2]["volume_level"]
        for c in hass.services.async_call.await_args_list
    }
    assert calls == {"media_player.a": 0.1, "media_player.b": 0.7}


async def test_the_shortcut_uses_the_same_step(monkeypatch) -> None:
    from custom_components.ai_plugin.shortcuts import async_try_media_shortcut, registry

    hass = _hass(0.4)
    ents = SimpleNamespace(
        entities={SPEAKER: SimpleNamespace(entity_id=SPEAKER, area_id=None, device_id=None)}
    )
    monkeypatch.setattr(registry.er, "async_get", lambda _: ents)
    monkeypatch.setattr(registry.dr, "async_get", lambda _: MagicMock())
    monkeypatch.setattr(registry.ar, "async_get", lambda _: MagicMock())
    monkeypatch.setattr(registry, "is_exposed", lambda *_: True)
    assert await async_try_media_shortcut(hass, "volume down", lang="en") == (True, "")
    assert _set_to(hass) == 0.3
