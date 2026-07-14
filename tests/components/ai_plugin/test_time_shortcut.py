"""Tests for the deterministic time shortcut (v0.9.28)."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

from custom_components.ai_plugin.shortcuts import try_shortcut


def _hass(tz: str = "Europe/Berlin") -> MagicMock:
    hass = MagicMock()
    hass.config.time_zone = tz
    # sun shortcut must not fire: no sun.sun state
    hass.states.get.return_value = None
    return hass


def test_time_shortcut_en() -> None:
    reply = try_shortcut(_hass(), "what time is it?", lang="en")
    assert reply is not None
    assert re.search(r"\b\d{1,2}:\d{2}\b", reply)
    assert reply.startswith("It is")


def test_time_shortcut_de() -> None:
    reply = try_shortcut(_hass(), "Wie spät ist es?", lang="de")
    assert reply is not None
    assert reply.startswith("Es ist")
    assert reply.endswith("Uhr.")


def test_time_shortcut_does_not_match_timers() -> None:
    assert try_shortcut(_hass(), "set a timer for 10 minutes", lang="en") is None
    assert try_shortcut(_hass(), "cancel the timer", lang="en") is None


def test_time_shortcut_does_not_match_sun() -> None:
    # sun questions must keep falling through to the sun shortcut / LLM
    assert try_shortcut(_hass(), "when does the sun set?", lang="en") is None
