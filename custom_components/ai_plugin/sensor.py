"""The plugin's own sensors: reply time, turn and safeguard counters, next alarm.

They are diagnostics for dashboards and automations, not something for the
voice model to read: none of them is exposed to Assist by default.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

# A turn of this kind was the plugin hearing its own reply, not a request.
_NOT_A_REQUEST = "echo"


def device_info(entry: ConfigEntry) -> DeviceInfo:
    """The device the conversation agent created, so everything sits together."""
    return DeviceInfo(identifiers={(DOMAIN, entry.entry_id)}, name=entry.title)


def _device_name(hass: HomeAssistant, device_id: str | None) -> str | None:
    if not device_id:
        return None
    device = dr.async_get(hass).async_get(device_id)
    if device is None:
        return None
    return device.name_by_user or device.name


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    data = hass.data[DOMAIN][entry.entry_id]
    entities: list[SensorEntity] = [
        LastReplySensor(entry, data["activity"]),
        TurnsSensor(entry, data["activity"]),
        SafeguardsSensor(entry, data["activity"]),
    ]
    if data.get("scheduler") is not None:
        entities.append(NextAlarmSensor(entry, data["scheduler"]))
    async_add_entities(entities)


class _ActivitySensor(RestoreSensor):
    """Updated on every turn the activity log records; keeps its value over
    a restart."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, activity, key: str) -> None:
        self._activity = activity
        self._attr_translation_key = key
        self._attr_unique_id = f"{entry.entry_id}-{key}"
        self._attr_device_info = device_info(entry)

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_sensor_data()
        state = await self.async_get_last_state()
        self._restore(
            last.native_value if last is not None else None,
            dict(state.attributes) if state is not None else {},
        )
        self.async_on_remove(self._activity.add_listener(self._on_turn))

    def _restore(self, value: Any, attributes: dict[str, Any]) -> None:
        if value is not None:
            self._attr_native_value = value

    @callback
    def _on_turn(self, turn: dict[str, Any]) -> None:
        if self._apply(turn):
            self.async_write_ha_state()

    def _apply(self, turn: dict[str, Any]) -> bool:
        raise NotImplementedError


class LastReplySensor(_ActivitySensor):
    """Seconds from the request to the finished reply, for the last turn."""

    _attr_device_class = SensorDeviceClass.DURATION
    _attr_native_unit_of_measurement = UnitOfTime.SECONDS
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 2
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, entry: ConfigEntry, activity) -> None:
        super().__init__(entry, activity, "last_reply_time")

    def _restore(self, value: Any, attributes: dict[str, Any]) -> None:
        super()._restore(value, attributes)
        self._attr_extra_state_attributes = {
            k: attributes.get(k) for k in ("kind", "satellite", "tools") if k in attributes
        }

    def _apply(self, turn: dict[str, Any]) -> bool:
        if turn.get("kind") == _NOT_A_REQUEST:
            return False
        self._attr_native_value = turn.get("seconds")
        self._attr_extra_state_attributes = {
            "kind": turn.get("kind"),
            "satellite": _device_name(self.hass, turn.get("device")),
            "tools": list(turn.get("tools") or []),
        }
        return True


class _CounterSensor(_ActivitySensor):
    """A count that only grows; per-kind counts ride along as attributes."""

    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_value = 0

    def _restore(self, value: Any, attributes: dict[str, Any]) -> None:
        try:
            self._attr_native_value = int(float(value)) if value is not None else 0
        except (TypeError, ValueError):
            self._attr_native_value = 0
        self._attr_extra_state_attributes = {
            k: v for k, v in attributes.items() if isinstance(v, int) and not isinstance(v, bool)
        }

    def _add(self, key: str, amount: int = 1) -> None:
        self._attr_native_value = int(self._attr_native_value or 0) + amount
        attrs = dict(self._attr_extra_state_attributes or {})
        attrs[key] = int(attrs.get(key, 0)) + amount
        self._attr_extra_state_attributes = attrs


class TurnsSensor(_CounterSensor):
    """Requests handled, split by how they ended (answered, offline, error,
    automation)."""

    def __init__(self, entry: ConfigEntry, activity) -> None:
        super().__init__(entry, activity, "turns")

    def _apply(self, turn: dict[str, Any]) -> bool:
        kind = turn.get("kind") or "answered"
        if kind == _NOT_A_REQUEST:
            return False
        self._add(kind)
        return True


class SafeguardsSensor(_CounterSensor):
    """Times a safeguard stepped in, split by safeguard."""

    def __init__(self, entry: ConfigEntry, activity) -> None:
        super().__init__(entry, activity, "safeguards")

    def _apply(self, turn: dict[str, Any]) -> bool:
        names = [s for s in turn.get("safeguards") or [] if s]
        for name in names:
            self._add(name)
        return bool(names)


class NextAlarmSensor(SensorEntity):
    """When the next alarm set through the assistant rings."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_translation_key = "next_alarm"

    def __init__(self, entry: ConfigEntry, scheduler) -> None:
        self._scheduler = scheduler
        self._attr_unique_id = f"{entry.entry_id}-next_alarm"
        self._attr_device_info = device_info(entry)

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(self._scheduler.add_listener(self._on_change))
        self._refresh()

    @callback
    def _on_change(self) -> None:
        self._refresh()
        self.async_write_ha_state()

    def _refresh(self) -> None:
        job = self._scheduler.next_alarm()
        due: datetime | None = job.due if job is not None else None
        self._attr_native_value = due
        self._attr_extra_state_attributes = (
            {
                "label": job.text or None,
                "repeat": job.repeat,
                "satellite": _device_name(self.hass, job.device_id),
            }
            if job is not None
            else {}
        )
