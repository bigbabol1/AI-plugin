"""Tests for the volume additions to the media shortcut (v0.9.31)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.ai_plugin import shortcuts
from custom_components.ai_plugin.shortcuts import async_try_media_shortcut


def _make_hass(state="playing"):
    hass = MagicMock()
    ent_reg = MagicMock()
    ent_reg.entities = {
        "media_player.kitchen": SimpleNamespace(
            entity_id="media_player.kitchen", area_id=None, device_id=None
        )
    }
    dev_reg = MagicMock()
    hass.states.get.return_value = SimpleNamespace(state=state)
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass, ent_reg, dev_reg


@pytest.fixture
def patched(monkeypatch):
    def _start(hass, ent_reg, dev_reg):
        monkeypatch.setattr(shortcuts.er, "async_get", lambda h: ent_reg)
        monkeypatch.setattr(shortcuts.dr, "async_get", lambda h: dev_reg)
        monkeypatch.setattr(shortcuts.ar, "async_get", lambda h: MagicMock())
    return _start


def _call(hass):
    args, kwargs = hass.services.async_call.await_args
    return args[1], (args[2] if len(args) > 2 else None), kwargs.get("target")


async def test_volume_up(patched) -> None:
    hass, e, d = _make_hass()
    patched(hass, e, d)
    assert await async_try_media_shortcut(hass, "volume up", lang="en") == (True, "")
    service, data, target = _call(hass)
    assert service == "volume_up" and data is None
    assert target == {"entity_id": ["media_player.kitchen"]}


async def test_volume_set_percent(patched) -> None:
    hass, e, d = _make_hass()
    patched(hass, e, d)
    result = await async_try_media_shortcut(
        hass, "set the volume to 40 percent", lang="en"
    )
    assert result == (True, "")
    service, data, _ = _call(hass)
    assert service == "volume_set"
    assert data == {"volume_level": 0.4}


async def test_mute_and_unmute(patched) -> None:
    hass, e, d = _make_hass()
    patched(hass, e, d)
    assert await async_try_media_shortcut(hass, "mute", lang="en") == (True, "")
    service, data, _ = _call(hass)
    assert (service, data) == ("volume_mute", {"is_volume_muted": True})

    hass.services.async_call.reset_mock()
    assert await async_try_media_shortcut(hass, "unmute", lang="en") == (True, "")
    service, data, _ = _call(hass)
    assert (service, data) == ("volume_mute", {"is_volume_muted": False})


async def test_volume_down_de(patched) -> None:
    hass, e, d = _make_hass()
    patched(hass, e, d)
    assert await async_try_media_shortcut(hass, "mach leiser", lang="de") == (True, "")
    service, data, _ = _call(hass)
    assert service == "volume_down"


async def test_volume_ignored_when_nothing_active(patched) -> None:
    hass, e, d = _make_hass(state="off")
    patched(hass, e, d)
    assert await async_try_media_shortcut(hass, "volume up", lang="en") is None
    hass.services.async_call.assert_not_awaited()
