"""Timer status / cancel / pause / add on HA's TimerManager (v0.9.51).

HA's timer intents skip every timer with a conversation_command, which
announce mode puts on each timer — so these tools read the manager directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin import shortcuts
from custom_components.ai_plugin.const import TIMER_DONE_SENTINEL
from custom_components.ai_plugin.tools.ha_local import (
    _TIMER_DATA_KEY,
    HALocalToolRegistry,
)


@dataclass
class FakeTimer:
    id: str
    name: str | None
    device_id: str | None
    seconds_left: int
    is_active: bool = True
    conversation_command: str | None = None
    start_hours: int | None = None
    start_minutes: int | None = None
    start_seconds: int | None = None


class FakeManager:
    """Mirrors the TimerManager methods the plugin calls."""

    def __init__(self, *timers: FakeTimer) -> None:
        self.timers = {t.id: t for t in timers}
        self.calls: list[tuple] = []

    def cancel_timer(self, timer_id: str) -> None:
        self.calls.append(("cancel", timer_id))
        self.timers.pop(timer_id)

    def pause_timer(self, timer_id: str) -> None:
        self.calls.append(("pause", timer_id))
        self.timers[timer_id].is_active = False

    def unpause_timer(self, timer_id: str) -> None:
        self.calls.append(("unpause", timer_id))
        self.timers[timer_id].is_active = True

    def add_time(self, timer_id: str, seconds: int) -> None:
        self.calls.append(("add", timer_id, seconds))
        self.timers[timer_id].seconds_left += seconds

    def remove_time(self, timer_id: str, seconds: int) -> None:
        self.calls.append(("remove", timer_id, seconds))
        self.timers[timer_id].seconds_left -= seconds


def _announce(name: str = "") -> str:
    return f"{TIMER_DONE_SENTINEL} {name}".strip()


def _reg(manager: FakeManager) -> HALocalToolRegistry:
    hass = MagicMock()
    hass.config.language = "en"
    hass.data = {_TIMER_DATA_KEY: manager}
    return HALocalToolRegistry(hass)


async def _call(reg, tool, args=None, device_id="dev1", msg=""):
    with patch(
        "custom_components.ai_plugin.tools.ha_local.intent.async_handle",
    ) as handle:
        out = await reg.call_tool(
            tool, args or {}, device_id=device_id, language="en", user_message=msg
        )
    handle.assert_not_called()
    return out


async def test_status_sees_announce_mode_timer() -> None:
    mgr = FakeManager(FakeTimer(
        "t1", None, "dev1", 18 * 60 + 32,
        conversation_command=_announce(), start_minutes=20,
    ))
    out = await _call(_reg(mgr), "timer_status", msg="how much is left of the timer?")
    assert out == "unnamed timer: 18 min 32 s left (set for 20 min)"


async def test_status_ignores_other_delayed_commands() -> None:
    mgr = FakeManager(
        FakeTimer("t1", "pasta", "dev1", 90, conversation_command=_announce("pasta")),
        FakeTimer("t2", None, "dev1", 600, conversation_command="turn off the lights"),
    )
    out = await _call(_reg(mgr), "timer_status")
    assert "pasta" in out and "10 min" not in out


async def test_status_without_device_and_without_timers() -> None:
    reg = _reg(FakeManager())
    assert await _call(reg, "timer_status", device_id=None) == "No timers are running."


async def test_status_unknown_name_lists_all() -> None:
    mgr = FakeManager(FakeTimer("t1", "eggs", "dev1", 45, is_active=False))
    out = await _call(_reg(mgr), "timer_status", {"name": "pasta"})
    assert out.startswith("No timer is named 'pasta'")
    assert "timer 'eggs': 45 s left, paused" in out


async def test_cancel_single_timer_despite_generic_name() -> None:
    mgr = FakeManager(FakeTimer("t1", None, "dev1", 300, conversation_command=_announce()))
    out = await _call(_reg(mgr), "cancel_timer", {"name": "timer"})
    assert mgr.calls == [("cancel", "t1")]
    assert out == "Cancelled the timer (5 min were left)."


async def test_cancel_by_name() -> None:
    mgr = FakeManager(
        FakeTimer("t1", "pasta", "dev1", 300),
        FakeTimer("t2", "eggs", "dev1", 60),
    )
    await _call(_reg(mgr), "cancel_timer", {"name": "Eggs"})
    assert mgr.calls == [("cancel", "t2")]


async def test_name_matches_without_timer_suffix() -> None:
    mgr = FakeManager(
        FakeTimer("t1", "Nudel", "dev1", 300),
        FakeTimer("t2", None, "dev1", 60),
    )
    reg = _reg(mgr)
    assert (await _call(reg, "timer_status", {"name": "Nudel-Timer"})).startswith(
        "timer 'Nudel'"
    )
    await _call(reg, "cancel_timer", {"name": "nudel timer"})
    assert mgr.calls == [("cancel", "t1")]


async def test_ambiguous_cancel_changes_nothing() -> None:
    mgr = FakeManager(
        FakeTimer("t1", None, "dev2", 300),
        FakeTimer("t2", None, "dev3", 60),
    )
    out = await _call(_reg(mgr), "cancel_timer")
    assert mgr.calls == []
    assert "several timers match" in out


async def test_calling_satellite_timer_wins_when_unnamed() -> None:
    mgr = FakeManager(
        FakeTimer("t1", None, "dev2", 300),
        FakeTimer("t2", None, "dev1", 60),
    )
    await _call(_reg(mgr), "cancel_timer")
    assert mgr.calls == [("cancel", "t2")]


async def test_increase_uses_utterance_duration() -> None:
    mgr = FakeManager(FakeTimer("t1", "pasta", "dev1", 60, conversation_command=_announce("pasta")))
    out = await _call(
        _reg(mgr), "increase_timer", {"name": "pasta", "seconds": 2},
        msg="add 2 minutes to the pasta timer",
    )
    assert mgr.calls == [("add", "t1", 120)]
    assert out == "Added 2 min; timer 'pasta' now has 3 min left."


async def test_decrease_and_missing_duration() -> None:
    mgr = FakeManager(FakeTimer("t1", None, "dev1", 600))
    reg = _reg(mgr)
    assert "needs a duration" in await _call(reg, "decrease_timer")
    await _call(reg, "decrease_timer", {"minutes": 1}, msg="take a minute off")
    assert mgr.calls == [("remove", "t1", 60)]


async def test_pause_and_unpause_pick_by_state() -> None:
    mgr = FakeManager(
        FakeTimer("t1", None, "dev1", 300, is_active=False),
        FakeTimer("t2", None, "dev1", 60),
    )
    reg = _reg(mgr)
    await _call(reg, "pause_timer")
    assert mgr.calls == [("pause", "t2")]
    mgr.timers["t2"].is_active = True
    await _call(reg, "unpause_timer")
    assert mgr.calls[-1] == ("unpause", "t1")


@pytest.mark.parametrize(
    "text", ["Resume the timer.", "Pause the timer", "Nudeltimer fortsetzen"]
)
async def test_media_shortcut_leaves_timer_commands_alone(monkeypatch, text) -> None:
    """'Resume the timer.' sent media_play to every idle exposed speaker."""
    hass = MagicMock()
    hass.states.get.side_effect = lambda eid: SimpleNamespace(state="idle")
    hass.services.async_call = AsyncMock()
    ent_reg = MagicMock()
    ent_reg.entities = {
        "media_player.kitchen": SimpleNamespace(area_id=None, device_id=None)
    }
    monkeypatch.setattr(shortcuts.er, "async_get", lambda h: ent_reg)
    monkeypatch.setattr(shortcuts.dr, "async_get", lambda h: MagicMock())
    monkeypatch.setattr(shortcuts.ar, "async_get", lambda h: MagicMock())

    assert await shortcuts.async_try_media_shortcut(hass, text, lang="en") is None
    hass.services.async_call.assert_not_awaited()


async def test_start_timer_still_needs_a_device() -> None:
    reg = _reg(FakeManager())
    out = await reg.call_tool("start_timer", {"minutes": 5}, device_id=None)
    assert "requires a satellite device" in out
