"""Tests for the reverse-geocode helper."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.ai_plugin.tools import _geocode
from custom_components.ai_plugin.tools._geocode import (
    GeocodeResult,
    _parse_nominatim,
    _round_key,
    reverse_geocode,
)


def _mock_resp(status: int, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _mock_session(resp: MagicMock) -> MagicMock:
    sess = MagicMock()
    sess.get = MagicMock(return_value=resp)
    sess.__aenter__ = AsyncMock(return_value=sess)
    sess.__aexit__ = AsyncMock(return_value=False)
    return sess


@pytest.fixture(autouse=True)
def _reset_rate_gate():
    """Reset the module-level rate gate so tests don't sleep."""
    _geocode._LAST_REQUEST_AT = 0.0
    yield


# ── _round_key / _parse_nominatim ─────────────────────────────────────────────


def test_round_key_stable_across_minor_jitter() -> None:
    a = _round_key(52.520123, 13.404500, "de")
    b = _round_key(52.520099, 13.404501, "DE")
    assert a == b


def test_parse_nominatim_picks_city_first() -> None:
    payload = {
        "address": {
            "city": "Berlin",
            "town": "Wedding",
            "state": "Berlin",
            "country": "Germany",
            "country_code": "de",
        }
    }
    res = _parse_nominatim(payload)
    assert res.city == "Berlin"
    assert res.region == "Berlin"
    assert res.country_name == "Germany"
    assert res.country_iso == "DE"


def test_parse_nominatim_falls_back_to_village() -> None:
    payload = {"address": {"village": "Hintertupfingen", "country_code": "AT"}}
    res = _parse_nominatim(payload)
    assert res.city == "Hintertupfingen"
    assert res.country_iso == "AT"
    assert res.region is None


def test_parse_nominatim_handles_empty_address() -> None:
    res = _parse_nominatim({"address": {}})
    assert res.city is None
    assert res.region is None
    assert res.country_iso is None


def test_geocode_result_best_label() -> None:
    assert GeocodeResult("Berlin", "BE", "Germany", "DE").best_label() == "Berlin"
    assert GeocodeResult(None, "BE", "Germany", "DE").best_label() == "BE"
    assert GeocodeResult(None, None, "Germany", "DE").best_label() == "Germany"
    assert GeocodeResult(None, None, None, None).best_label() is None


# ── reverse_geocode ───────────────────────────────────────────────────────────


async def test_reverse_geocode_missing_coords_returns_none(tmp_path) -> None:
    assert await reverse_geocode(None, 13.4, "de", str(tmp_path)) is None
    assert await reverse_geocode(52.5, None, "de", str(tmp_path)) is None


async def test_reverse_geocode_invalid_coords_returns_none(tmp_path) -> None:
    assert await reverse_geocode(91.0, 0.0, "de", str(tmp_path)) is None
    assert await reverse_geocode(0.0, 181.0, "de", str(tmp_path)) is None


async def test_reverse_geocode_happy_path_writes_cache(tmp_path) -> None:
    payload = {
        "address": {
            "city": "Berlin",
            "country": "Germany",
            "country_code": "de",
        }
    }
    resp = _mock_resp(200, json_data=payload)
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        res = await reverse_geocode(52.52, 13.405, "de", str(tmp_path))
    assert res is not None
    assert res.city == "Berlin"
    assert res.country_iso == "DE"

    cache_file = os.path.join(str(tmp_path), ".ai_plugin_geocache.json")
    assert os.path.exists(cache_file)
    with open(cache_file) as fh:
        data = json.load(fh)
    assert any("Berlin" == v.get("city") for v in data.values())


async def test_reverse_geocode_cache_hit_skips_network(tmp_path) -> None:
    cache_file = os.path.join(str(tmp_path), ".ai_plugin_geocache.json")
    key = _round_key(52.52, 13.405, "de")
    with open(cache_file, "w") as fh:
        json.dump({key: {
            "city": "Cached City",
            "region": None,
            "country_name": "Cached Country",
            "country_iso": "CC",
        }}, fh)

    sess = MagicMock()
    sess.get = MagicMock(side_effect=AssertionError("network must not be hit"))
    sess.__aenter__ = AsyncMock(return_value=sess)
    sess.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=sess):
        res = await reverse_geocode(52.52, 13.405, "de", str(tmp_path))
    assert res is not None
    assert res.city == "Cached City"
    assert res.country_iso == "CC"


async def test_reverse_geocode_non_200_returns_none(tmp_path) -> None:
    resp = _mock_resp(503)
    with patch("aiohttp.ClientSession", return_value=_mock_session(resp)):
        res = await reverse_geocode(52.52, 13.405, "de", str(tmp_path))
    assert res is None


async def test_reverse_geocode_network_error_returns_none(tmp_path) -> None:
    sess = MagicMock()
    sess.__aenter__ = AsyncMock(return_value=sess)
    sess.__aexit__ = AsyncMock(return_value=False)
    sess.get = MagicMock(side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError()))
    with patch("aiohttp.ClientSession", return_value=sess):
        res = await reverse_geocode(52.52, 13.405, "de", str(tmp_path))
    assert res is None


async def test_reverse_geocode_timeout_returns_none(tmp_path) -> None:
    sess = MagicMock()
    sess.__aenter__ = AsyncMock(return_value=sess)
    sess.__aexit__ = AsyncMock(return_value=False)
    sess.get = MagicMock(side_effect=asyncio.TimeoutError())
    with patch("aiohttp.ClientSession", return_value=sess):
        res = await reverse_geocode(52.52, 13.405, "de", str(tmp_path))
    assert res is None
