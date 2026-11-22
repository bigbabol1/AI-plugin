"""The plugin's own sensors: reply time, counters, next alarm, model health."""

from __future__ import annotations

from datetime import datetime, timedelta
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from .test_conversation import _voice_orch_with_intent  # noqa: F401  isort: skip (stubs HA first)


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = sys.modules.get(name) or types.ModuleType(name)
    for key, value in attrs.items():
        if not hasattr(mod, key):
            setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class _Entity:
    hass = None
    _attr_extra_state_attributes = None
    _attr_native_value = None

    writes = 0

    async def async_added_to_hass(self):
        return None

    @property
    def removers(self):
        return self.__dict__.setdefault("_removers", [])

    def async_on_remove(self, fn):
        self.removers.append(fn)

    def async_write_ha_state(self):
        self.writes += 1


class _RestoreSensor(_Entity):
    last_value = None
    last_attributes: dict = {}

    async def async_get_last_sensor_data(self):
        if self.last_value is None:
            return None
        return SimpleNamespace(native_value=self.last_value)

    async def async_get_last_state(self):
        if not self.last_attributes:
            return None
        return SimpleNamespace(attributes=self.last_attributes)


class _CoordinatorEntity(_Entity):
    def __init__(self, coordinator, *args, **kwargs):
        self.coordinator = coordinator

    @property
    def available(self):
        return True


_stub(
    "homeassistant.components.sensor",
    RestoreSensor=_RestoreSensor,
    SensorEntity=_Entity,
    SensorDeviceClass=SimpleNamespace(DURATION="duration", TIMESTAMP="timestamp"),
    SensorStateClass=SimpleNamespace(MEASUREMENT="measurement", TOTAL_INCREASING="total_increasing"),
)
_stub(
    "homeassistant.components.binary_sensor",
    BinarySensorEntity=_Entity,
    BinarySensorDeviceClass=SimpleNamespace(CONNECTIVITY="connectivity"),
)
_stub(
    "homeassistant.helpers.update_coordinator",
    CoordinatorEntity=_CoordinatorEntity,
    DataUpdateCoordinator=MagicMock(),
)
_stub("homeassistant.helpers.entity_platform", AddEntitiesCallback=object)
_stub(
    "homeassistant.const",
    EntityCategory=SimpleNamespace(DIAGNOSTIC="diagnostic"),
    UnitOfTime=SimpleNamespace(SECONDS="s"),
)

from custom_components.ai_plugin import binary_sensor, sensor  # noqa: E402
from custom_components.ai_plugin.activity import ActivityLog  # noqa: E402
from custom_components.ai_plugin.scheduler import engine  # noqa: E402
from custom_components.ai_plugin.scheduler.engine import (  # noqa: E402
    KIND_ALARM,
    KIND_REMINDER,
    Scheduler,
)

TZ = ZoneInfo("Europe/Berlin")
NOW = datetime(2026, 11, 1, 10, 0, tzinfo=TZ)
ENTRY = SimpleNamespace(entry_id="e1", title="AI Plugin")


def _hass_with_device(name="Kitchen satellite"):
    hass = MagicMock()
    device = SimpleNamespace(name="kitchen_sat", name_by_user=name)
    registry = MagicMock()
    registry.async_get.side_effect = lambda device_id: device if device_id == "dev1" else None
    return hass, registry


async def _added(entity, hass=None):
    entity.hass = hass or MagicMock()
    await entity.async_added_to_hass()
    return entity


def _turn(log, **kw):
    defaults = {"text": "turn on the lamp", "reply": "Done.", "seconds": 1.234}
    log.record(**{**defaults, **kw})


# ── the activity log tells its listeners ─────────────────────────────────────


def test_activity_listener_gets_each_turn_and_can_leave():
    log = ActivityLog()
    seen = []
    remove = log.add_listener(seen.append)
    _turn(log, kind="answered")
    remove()
    _turn(log, kind="answered")
    assert len(seen) == 1
    assert seen[0]["seconds"] == 1.23


def test_a_failing_listener_does_not_fail_the_turn():
    log = ActivityLog()
    log.add_listener(MagicMock(side_effect=RuntimeError("boom")))
    _turn(log)
    assert log.counts["turns"] == 1


# ── reply time ───────────────────────────────────────────────────────────────


async def test_last_reply_time_follows_turns_and_names_the_satellite():
    log = ActivityLog()
    hass, registry = _hass_with_device()
    entity = await _added(sensor.LastReplySensor(ENTRY, log), hass)
    with patch.object(sensor.dr, "async_get", return_value=registry):
        _turn(log, seconds=0.42, device="dev1", tools=["HassTurnOn"])
    assert entity._attr_native_value == 0.42
    assert entity._attr_extra_state_attributes == {
        "kind": "answered", "satellite": "Kitchen satellite", "tools": ["HassTurnOn"],
    }
    assert entity.writes == 1


async def test_an_echo_is_not_a_reply():
    log = ActivityLog()
    entity = await _added(sensor.LastReplySensor(ENTRY, log))
    _turn(log, kind="echo", seconds=9.0)
    assert entity._attr_native_value is None
    assert entity.writes == 0


async def test_last_reply_time_survives_a_restart():
    entity = sensor.LastReplySensor(ENTRY, ActivityLog())
    entity.last_value = 2.5
    entity.last_attributes = {"kind": "answered", "friendly_name": "x"}
    await _added(entity)
    assert entity._attr_native_value == 2.5
    assert entity._attr_extra_state_attributes == {"kind": "answered"}


async def test_removing_the_sensor_stops_its_updates():
    log = ActivityLog()
    entity = await _added(sensor.LastReplySensor(ENTRY, log))
    for remove in entity.removers:
        remove()
    _turn(log)
    assert entity.writes == 0


# ── counters ─────────────────────────────────────────────────────────────────


async def test_turns_count_by_kind_and_skip_echoes():
    log = ActivityLog()
    entity = await _added(sensor.TurnsSensor(ENTRY, log))
    _turn(log, kind="answered")
    _turn(log, kind="answered")
    _turn(log, kind="automation")
    _turn(log, kind="echo")
    assert entity._attr_native_value == 3
    assert entity._attr_extra_state_attributes == {"answered": 2, "automation": 1}


async def test_turns_keep_counting_after_a_restart():
    log = ActivityLog()
    entity = sensor.TurnsSensor(ENTRY, log)
    entity.last_value = 40
    entity.last_attributes = {
        "answered": 38, "offline": 2, "state_class": "total_increasing", "friendly_name": "Turns",
    }
    await _added(entity)
    _turn(log, kind="offline")
    assert entity._attr_native_value == 41
    assert entity._attr_extra_state_attributes == {"answered": 38, "offline": 3}


async def test_safeguards_count_each_one_that_stepped_in():
    log = ActivityLog()
    entity = await _added(sensor.SafeguardsSensor(ENTRY, log))
    _turn(log)
    assert entity.writes == 0
    _turn(log, safeguards=["unverified_claim", "retry"])
    _turn(log, safeguards=["unverified_claim"])
    assert entity._attr_native_value == 3
    assert entity._attr_extra_state_attributes == {"unverified_claim": 2, "retry": 1}


# ── next alarm ───────────────────────────────────────────────────────────────


@pytest.fixture
def scheduler():
    with patch.object(engine, "async_track_point_in_utc_time", MagicMock()):
        hass = MagicMock()
        s = Scheduler(hass, "e1")
        s._store = MagicMock(async_save=AsyncMock())
        s.now = lambda: NOW
        yield s


async def test_next_alarm_is_the_earliest_alarm_not_a_reminder(scheduler):
    hass, registry = _hass_with_device()
    entity = await _added(sensor.NextAlarmSensor(ENTRY, scheduler), hass)
    assert entity._attr_native_value is None
    with patch.object(sensor.dr, "async_get", return_value=registry):
        await scheduler.async_add(KIND_REMINDER, NOW + timedelta(minutes=5), text="tea")
        late = await scheduler.async_add(KIND_ALARM, NOW + timedelta(hours=20), device_id="dev1")
        early = await scheduler.async_add(KIND_ALARM, NOW + timedelta(hours=8), text="gym", repeat="daily")
    assert entity._attr_native_value == early.due
    assert entity._attr_extra_state_attributes == {"label": "gym", "repeat": "daily", "satellite": None}
    with patch.object(sensor.dr, "async_get", return_value=registry):
        await scheduler.async_cancel([early])
    assert entity._attr_native_value == late.due
    assert entity._attr_extra_state_attributes["satellite"] == "Kitchen satellite"
    await scheduler.async_cancel([late])
    assert entity._attr_native_value is None
    assert entity.writes == 5


# ── model server health ──────────────────────────────────────────────────────


def _health_entities(data):
    coordinator = SimpleNamespace(data=data)
    return (
        binary_sensor.ModelServerReachable(coordinator, ENTRY),
        binary_sensor.ModelLoaded(coordinator, ENTRY),
    )


def test_model_server_up_with_the_model_in_memory():
    reachable, loaded = _health_entities({
        "reachable": True, "ms": 18, "model_listed": True,
        "ollama": {"loaded": True, "gpu_pct": 100, "size_gb": 9.1, "others": 0},
    })
    assert reachable.is_on and reachable.extra_state_attributes == {"response_ms": 18, "model_listed": True}
    assert loaded.available and loaded.is_on
    assert loaded.extra_state_attributes == {"gpu_pct": 100, "size_gb": 9.1, "others": 0}


def test_model_pushed_out_of_memory():
    _, loaded = _health_entities({"reachable": True, "ollama": {"loaded": False, "others": 1}})
    assert loaded.available and not loaded.is_on


def test_model_server_down():
    reachable, loaded = _health_entities({"reachable": False, "error": "Cannot connect"})
    assert not reachable.is_on
    assert reachable.extra_state_attributes == {"error": "Cannot connect"}
    assert not loaded.available


def test_model_loaded_is_unavailable_on_a_backend_that_cannot_say():
    _, loaded = _health_entities({"reachable": True, "ms": 40, "ollama": None})
    assert not loaded.available
