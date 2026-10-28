"""The MCP tool lists follow the entities behind them."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from custom_components.ai_plugin import _refresh_tools_on_entity_changes


def _setup():
    hass = MagicMock()
    listeners = {}
    hass.bus.async_listen = lambda event_type, cb: listeners.setdefault(event_type, cb)
    timers = []

    def call_later(delay, cb):
        handle = MagicMock()
        timers.append((handle, cb))
        return handle

    hass.loop.call_later = call_later
    entry = MagicMock()
    mcp = MagicMock()
    mcp.async_refresh_tools = AsyncMock()
    _refresh_tools_on_entity_changes(hass, entry, mcp)
    return hass, listeners["state_changed"], timers


def _event(old, new):
    return SimpleNamespace(data={"entity_id": "todo.x", "old_state": old, "new_state": new})


def test_a_new_entity_schedules_one_refresh_after_the_burst() -> None:
    hass, on_state, timers = _setup()
    on_state(_event(None, object()))
    on_state(_event(None, object()))
    assert len(timers) == 2
    timers[0][0].cancel.assert_called_once()  # the burst keeps pushing it back
    timers[-1][1]()
    hass.async_create_task.assert_called_once()


def test_a_removed_entity_schedules_a_refresh() -> None:
    _hass, on_state, timers = _setup()
    on_state(_event(object(), None))
    assert len(timers) == 1


def test_an_ordinary_state_change_does_not() -> None:
    _hass, on_state, timers = _setup()
    on_state(_event(object(), object()))
    assert timers == []
