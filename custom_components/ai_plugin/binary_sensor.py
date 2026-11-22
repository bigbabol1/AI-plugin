"""Is the model server reachable, and does it hold the model in memory?

Polled once a minute with the same check the sidebar panel's status tab
uses, so the panel's latency history fills even while nobody looks at it.
"""

from __future__ import annotations

from datetime import timedelta
import logging
from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

from .sensor import device_info

_LOGGER = logging.getLogger(__name__)
_INTERVAL = timedelta(seconds=60)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    from .panel.api import async_model_health  # noqa: PLC0415

    async def _update() -> dict[str, Any]:
        return await async_model_health(hass, entry, refresh=True)

    coordinator = DataUpdateCoordinator(
        hass,
        _LOGGER,
        config_entry=entry,
        name="AI Plugin model server",
        update_method=_update,
        update_interval=_INTERVAL,
    )
    await coordinator.async_refresh()
    async_add_entities([
        ModelServerReachable(coordinator, entry),
        ModelLoaded(coordinator, entry),
    ])


class _HealthSensor(CoordinatorEntity, BinarySensorEntity):
    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, entry: ConfigEntry, key: str) -> None:
        super().__init__(coordinator)
        self._attr_translation_key = key
        self._attr_unique_id = f"{entry.entry_id}-{key}"
        self._attr_device_info = device_info(entry)

    @property
    def _health(self) -> dict[str, Any]:
        return self.coordinator.data or {}


class ModelServerReachable(_HealthSensor):
    """On while the model server answers its model list."""

    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY

    def __init__(self, coordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "model_server")

    @property
    def is_on(self) -> bool:
        return bool(self._health.get("reachable"))

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        health = self._health
        if not health.get("reachable"):
            return {"error": health.get("error")} if health.get("error") else {}
        return {"response_ms": health.get("ms"), "model_listed": health.get("model_listed")}


class ModelLoaded(_HealthSensor):
    """On while Ollama holds the configured model in memory. Unavailable on
    other backends, which cannot say."""

    def __init__(self, coordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "model_loaded")

    @property
    def _ollama(self) -> dict[str, Any] | None:
        ollama = self._health.get("ollama")
        return ollama if isinstance(ollama, dict) else None

    @property
    def available(self) -> bool:
        return super().available and self._ollama is not None

    @property
    def is_on(self) -> bool:
        return bool((self._ollama or {}).get("loaded"))

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ollama = self._ollama or {}
        keys = ("gpu_pct", "size_gb", "context", "until", "others")
        return {k: ollama[k] for k in keys if ollama.get(k) is not None}
