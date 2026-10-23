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


# ── false-positive guards (v0.9.38) ──────────────────────────────────────────


def test_time_with_place_falls_through():
    """'what time is it in Tokyo' is a timezone question — the local clock
    must not answer it."""
    from unittest.mock import MagicMock
    from custom_components.ai_plugin.shortcuts import try_shortcut

    hass = MagicMock()
    assert try_shortcut(hass, "what time is it in tokyo", lang="en") is None
    assert try_shortcut(hass, "wie spät ist es in new york", lang="de") is None


def test_sunset_with_place_falls_through():
    from unittest.mock import MagicMock
    from custom_components.ai_plugin.shortcuts import try_shortcut

    hass = MagicMock()
    assert try_shortcut(hass, "when is sunset in tokyo", lang="en") is None


def test_command_with_sensor_adjective_falls_through():
    """Commands mentioning warm/cold must not be answered with a sensor
    reading — the action would be swallowed."""
    from unittest.mock import MagicMock
    from custom_components.ai_plugin.shortcuts import try_shortcut

    hass = MagicMock()
    assert (
        try_shortcut(
            hass, "turn on the heating, it's cold in the living room", lang="en"
        )
        is None
    )
    assert try_shortcut(hass, "mach es warm im schlafzimmer", lang="de") is None


def test_alarm_for_sunset_falls_through():
    from unittest.mock import MagicMock
    from custom_components.ai_plugin.shortcuts import try_shortcut

    hass = MagicMock()
    assert try_shortcut(hass, "set an alarm for sunset", lang="en") is None


def test_long_message_falls_through():
    from unittest.mock import MagicMock
    from custom_components.ai_plugin.shortcuts import try_shortcut

    hass = MagicMock()
    msg = (
        "could you please tell me what the temperature is like in the "
        "living room compared to yesterday evening"
    )
    assert try_shortcut(hass, msg, lang="en") is None
