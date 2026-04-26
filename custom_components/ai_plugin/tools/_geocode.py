"""Reverse-geocoding helper for the AI Plugin location bias.

Resolves ``(lat, lon)`` to a coarse place description (city / region /
country) using OpenStreetMap's free Nominatim service. Results are cached
on disk under the HA config dir so we issue one network call per unique
(lat, lon, language) per install — Nominatim's usage policy permits 1
req/s for low-volume use.

Failure modes are explicit:
  * No internet / DNS failure / timeout → returns ``None``; caller falls
    back to whatever fields are already known (``country`` from
    ``hass.config``, raw coords, etc.).
  * Cached result exists → returned without a network call.
  * Coordinates outside Earth or ``None`` → returns ``None`` immediately.

The cache file is human-readable JSON so users can inspect / wipe it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_USER_AGENT = "HomeAssistant-AI-Plugin/0.5 (https://github.com/bigbabol1/AI-plugin)"
_REQUEST_TIMEOUT = 8  # seconds
_CACHE_FILENAME = ".ai_plugin_geocache.json"

# Per-process rate gate so concurrent setups don't hammer Nominatim.
_LAST_REQUEST_AT = 0.0
_RATE_LOCK = asyncio.Lock()
_MIN_INTERVAL = 1.1  # seconds between live requests


@dataclass(frozen=True)
class GeocodeResult:
    """Coarse place fields returned by the reverse geocoder.

    Any field may be ``None`` — Nominatim does not always populate all
    address tags, especially for rural coordinates.
    """

    city: str | None
    region: str | None
    country_name: str | None
    country_iso: str | None  # ISO-3166 alpha-2, upper-case

    def best_label(self) -> str | None:
        """Return the most specific non-empty field for prompt rendering."""
        return self.city or self.region or self.country_name


def _cache_path(config_dir: str) -> str:
    return os.path.join(config_dir, _CACHE_FILENAME)


def _round_key(lat: float, lon: float, language: str | None) -> str:
    """Cache key — coords rounded to 4 decimals (~11 m), lang lowered."""
    return f"{round(lat, 4):.4f},{round(lon, 4):.4f}:{(language or '').lower()}"


def _load_cache(path: str) -> dict[str, dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        _LOGGER.debug("AI Plugin: geocache unreadable, starting fresh", exc_info=True)
    return {}


def _save_cache(path: str, data: dict[str, dict[str, Any]]) -> None:
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp, path)
    except OSError:
        _LOGGER.debug("AI Plugin: geocache write failed", exc_info=True)


def _parse_nominatim(payload: dict[str, Any]) -> GeocodeResult:
    """Pick the most useful address fields out of a Nominatim response."""
    addr = payload.get("address") or {}
    # Nominatim emits city/town/village/hamlet/municipality depending on
    # population class. Pick the first that exists.
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("hamlet")
        or addr.get("municipality")
        or addr.get("suburb")
        or addr.get("city_district")
    )
    region = (
        addr.get("state")
        or addr.get("region")
        or addr.get("state_district")
        or addr.get("county")
    )
    country_name = addr.get("country")
    country_iso = addr.get("country_code")
    if isinstance(country_iso, str) and country_iso:
        country_iso = country_iso.upper()
    else:
        country_iso = None
    return GeocodeResult(
        city=city if isinstance(city, str) and city else None,
        region=region if isinstance(region, str) and region else None,
        country_name=country_name if isinstance(country_name, str) and country_name else None,
        country_iso=country_iso,
    )


async def reverse_geocode(
    lat: float | None,
    lon: float | None,
    language: str | None,
    config_dir: str,
) -> GeocodeResult | None:
    """Resolve coordinates to a coarse place; cache disk-side.

    Returns ``None`` when the lookup fails or coordinates are missing.
    Never raises — callers should fall through to whatever location data
    is available without geocoding.
    """
    if lat is None or lon is None:
        return None
    if not (-90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0):
        return None

    key = _round_key(float(lat), float(lon), language)
    path = _cache_path(config_dir)

    loop = asyncio.get_running_loop()
    cache = await loop.run_in_executor(None, _load_cache, path)

    cached = cache.get(key)
    if cached is not None:
        return GeocodeResult(
            city=cached.get("city"),
            region=cached.get("region"),
            country_name=cached.get("country_name"),
            country_iso=cached.get("country_iso"),
        )

    # Live fetch — respect rate limit.
    global _LAST_REQUEST_AT
    async with _RATE_LOCK:
        gap = time.monotonic() - _LAST_REQUEST_AT
        if gap < _MIN_INTERVAL:
            await asyncio.sleep(_MIN_INTERVAL - gap)
        _LAST_REQUEST_AT = time.monotonic()

    params = {
        "lat": f"{lat:.5f}",
        "lon": f"{lon:.5f}",
        "format": "jsonv2",
        "zoom": "10",
        "addressdetails": "1",
    }
    headers = {"User-Agent": _USER_AGENT}
    if language:
        headers["Accept-Language"] = language

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                _NOMINATIM_URL,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    _LOGGER.debug(
                        "AI Plugin: nominatim returned HTTP %s for %s,%s",
                        resp.status, lat, lon,
                    )
                    return None
                payload = await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError):
        _LOGGER.debug("AI Plugin: nominatim request failed", exc_info=True)
        return None
    except Exception:  # noqa: BLE001
        _LOGGER.debug("AI Plugin: nominatim unexpected error", exc_info=True)
        return None

    if not isinstance(payload, dict):
        return None

    result = _parse_nominatim(payload)
    cache[key] = {
        "city": result.city,
        "region": result.region,
        "country_name": result.country_name,
        "country_iso": result.country_iso,
    }
    await loop.run_in_executor(None, _save_cache, path, cache)
    return result
