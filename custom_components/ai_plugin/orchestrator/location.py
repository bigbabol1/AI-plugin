"""Where the house is, and how much of that the model is told.

The location block is built from a device tracker or the HA config, with
reverse geocoding, and is deliberately coarse: a bias for search
results, not a coordinate handed to a cloud model.
"""

from __future__ import annotations

import asyncio
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from ..const import (
    CONF_LOCATION_BIAS,
    CONF_LOCATION_ENTITY,
    DEFAULT_LOCATION_BIAS,
)
from ..tools._geocode import GeocodeResult, reverse_geocode

_LOGGER = logging.getLogger(__name__)

class LocationProvider:
    """Resolve the user's home location across heterogeneous installs.

    Source priority: configured live entity > hass.config home coords >
    nothing. Reverse-geocoded once per coordinate pair via OSM Nominatim
    (cached on disk so we never repeat lookups across HA restarts).

    Designed to fail open: missing internet, missing country, missing
    coords all degrade to the next available signal. When *no* signal
    resolves, callers receive an empty payload — the plugin must then
    skip location injection entirely rather than guess a city.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._hass = hass
        self._entry = entry
        self._cached: dict | None = None
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._entry.options.get(CONF_LOCATION_BIAS, DEFAULT_LOCATION_BIAS))

    def _read_entity_coords(self) -> tuple[float | None, float | None]:
        """Return live coords from the configured location entity, if any."""
        entity_id = self._entry.options.get(CONF_LOCATION_ENTITY)
        if not entity_id or self._hass is None:
            return (None, None)
        state = self._hass.states.get(entity_id)
        if state is None:
            return (None, None)
        attrs = state.attributes or {}
        lat = attrs.get("latitude")
        lon = attrs.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return (float(lat), float(lon))
        return (None, None)

    def _read_config_coords(self) -> tuple[float | None, float | None]:
        if self._hass is None:
            return (None, None)
        cfg = self._hass.config
        lat = getattr(cfg, "latitude", None)
        lon = getattr(cfg, "longitude", None)
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return (float(lat), float(lon))
        return (None, None)

    async def async_resolve(self) -> dict:
        """Return a location dict — possibly empty, never None.

        Result keys (any may be absent):
          * ``city``, ``region``, ``country_name``, ``country_iso``
          * ``lat``, ``lon`` (always pair, or both absent)
          * ``timezone``
          * ``language`` (BCP-47 from hass.config)
          * ``source``: "entity" or "config" — None when nothing resolved.

        Cached after first call; cache is invalidated on entry reload
        because a new Orchestrator (and Provider) is built then.
        """
        if not self.enabled:
            return {}
        if self._cached is not None:
            return self._cached

        async with self._lock:
            if self._cached is not None:
                return self._cached

            payload: dict = {}
            cfg = self._hass.config if self._hass is not None else None

            # Coordinates: entity first, then hass.config.
            lat, lon = self._read_entity_coords()
            source = "entity" if lat is not None and lon is not None else None
            if lat is None or lon is None:
                lat, lon = self._read_config_coords()
                if lat is not None and lon is not None:
                    source = "config"

            if lat is not None and lon is not None:
                payload["lat"] = lat
                payload["lon"] = lon

            # Country / timezone / language from hass.config — always
            # available even when geocoding fails.
            if cfg is not None:
                country = str(getattr(cfg, "country", "") or "").strip()
                if country:
                    payload["country_iso"] = country.upper()
                tz = str(getattr(cfg, "time_zone", "") or "").strip()
                if tz:
                    payload["timezone"] = tz
                lang = str(getattr(cfg, "language", "") or "").strip()
                if lang:
                    payload["language"] = lang

            # Reverse geocode if we have coords. Failure is silent.
            if lat is not None and lon is not None and cfg is not None:
                try:
                    geo: GeocodeResult | None = await reverse_geocode(
                        lat=lat,
                        lon=lon,
                        language=payload.get("language"),
                        config_dir=cfg.config_dir,
                    )
                except Exception:  # noqa: BLE001
                    _LOGGER.debug(
                        "AI Plugin: reverse_geocode raised unexpectedly",
                        exc_info=True,
                    )
                    geo = None
                if geo is not None:
                    if geo.city:
                        payload["city"] = geo.city
                    if geo.region:
                        payload["region"] = geo.region
                    if geo.country_name:
                        payload["country_name"] = geo.country_name
                    if geo.country_iso and "country_iso" not in payload:
                        payload["country_iso"] = geo.country_iso

            if source is not None:
                payload["source"] = source

            self._cached = payload
            return payload
