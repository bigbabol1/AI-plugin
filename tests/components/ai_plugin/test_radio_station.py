"""Radio stations asked for by name, and the answer to "Which station?"."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from .test_conversation import _baseline_orch  # noqa: F401  # isort: skip (stubs HA first)

from custom_components.ai_plugin.const import SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE
from custom_components.ai_plugin.orchestrator.station_answer import as_station_request
from custom_components.ai_plugin.tools.ha_local import HALocalToolRegistry
from custom_components.ai_plugin.tools.ha_local.media import (
    NO_STATION_NAMED_RESULT,
    asks_for_radio,
    best_station,
)

SPEAKER = "media_player.kitchen"
STATIONS = [
    {"name": "Country Roadtrip", "uri": "radiobrowser://radio/1"},
    {"name": "Hit Radio North", "uri": "radiobrowser://radio/2"},
    {"name": "Hit Radio North - Classics", "uri": "radiobrowser://radio/3"},
]


def _registry(stations=None, search_fails=False):
    hass = MagicMock()
    hass.services.has_service = MagicMock(return_value=True)
    hass.config_entries.async_entries = MagicMock(return_value=[SimpleNamespace(entry_id="ma")])
    hass.states.get = lambda _: SimpleNamespace(state="idle", attributes={})

    async def _call(domain, service, *args, **kwargs):
        if service == "search":
            if search_fails:
                raise RuntimeError("no search")
            return {"radio": STATIONS if stations is None else stations}
        return None

    hass.services.async_call = AsyncMock(side_effect=_call)
    registry = HALocalToolRegistry(hass)
    registry._music_area = MagicMock(return_value=SimpleNamespace(name="kitchen"))
    registry._music_player_in_area = MagicMock(return_value=SPEAKER)
    return registry


def _played(registry):
    calls = [
        c for c in registry._hass.services.async_call.await_args_list if c.args[1] == "play_media"
    ]
    return calls[-1].kwargs["service_data"] if calls else None


def _searches(registry):
    return [c for c in registry._hass.services.async_call.await_args_list if c.args[1] == "search"]


@pytest.mark.parametrize("query", ["North radio", "Radio North", "radio station North", "Sender North"])
def test_these_ask_for_a_station(query) -> None:
    assert asks_for_radio(query)


@pytest.mark.parametrize("query", ["Enya", "Hotel California", "workout playlist", "jazz"])
def test_these_do_not(query) -> None:
    assert not asks_for_radio(query)


def test_a_station_must_share_a_word_with_the_request() -> None:
    assert best_station("North radio", STATIONS)["uri"] == "radiobrowser://radio/2"
    assert best_station("South radio", STATIONS) is None


async def test_a_station_named_with_radio_plays_the_station_not_a_track() -> None:
    """Live 2026-11-18: "Play RTL radio." played a Tidal news jingle."""
    registry = _registry()
    result = await registry._play_music("North radio", "kitchen")
    assert result == f"OK — playing 'Hit Radio North' (radio) on {SPEAKER}."
    assert _searches(registry)[0].args[2]["media_type"] == ["radio"]
    assert _played(registry) == {
        "media_id": "radiobrowser://radio/2", "media_type": "radio", "radio_mode": False,
    }


async def test_media_type_radio_without_the_word_still_searches_stations() -> None:
    registry = _registry()
    await registry._play_music("Hit North", "kitchen", media_type="radio")
    assert _played(registry)["media_id"] == "radiobrowser://radio/2"


async def test_no_station_found_for_media_type_radio_plays_nothing() -> None:
    registry = _registry(stations=[])
    result = await registry._play_music("South", "kitchen", media_type="radio")
    assert result.startswith("[No radio station found for 'South'")
    assert _played(registry) is None


async def test_the_word_radio_alone_asks_which_station() -> None:
    registry = _registry()
    assert await registry._play_music("radio", "kitchen") == NO_STATION_NAMED_RESULT
    assert await registry._play_music("some radio", "kitchen", media_type="radio") == NO_STATION_NAMED_RESULT
    registry._hass.services.async_call.assert_not_awaited()


async def test_a_song_with_radio_in_its_name_still_plays_when_no_station_matches() -> None:
    registry = _registry(stations=[{"name": "Radio Paradise", "uri": "radiobrowser://radio/9"}])
    await registry._play_music("Radio Ga Ga", "kitchen")
    assert _played(registry) == {"media_id": "Radio Ga Ga", "media_type": "track", "radio_mode": True}


async def test_a_failed_station_search_plays_the_words_as_asked() -> None:
    registry = _registry(search_fails=True)
    await registry._play_music("North radio", "kitchen")
    assert _played(registry)["media_id"] == "North radio"


@pytest.mark.parametrize("prompt", [SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE])
def test_both_prompts_name_media_type_radio(prompt) -> None:
    assert "media_type: 'radio'" in prompt
    assert "When you asked which station" in prompt


# ── the answer to "Which station?" ───────────────────────────────────────────

ASKED = "I can certainly do that, but what kind of radio station or genre would you like to hear?"


@pytest.mark.parametrize(
    ("message", "previous", "lang", "expected"),
    [
        ("Had you Ainz?", ASKED, "en", "Play the radio station Had you Ainz."),
        ("North radio.", "Which station?", "en", "Play the radio station North radio."),
        ("Radio Nord", "Welchen Sender möchtest du hören?", "de", "Spiel den Radiosender Radio Nord."),
        ("Never mind.", ASKED, "en", "Never mind."),
        ("Nein, egal", "Welchen Sender?", "de", "Nein, egal"),
        ("Had you Ainz?", "What would you like to hear?", "en", "Had you Ainz?"),
        ("Turn on the kitchen light please now", ASKED, "en", "Turn on the kitchen light please now"),
        ("North radio.", "", "en", "North radio."),
    ],
)
def test_an_answer_to_which_station(message, previous, lang, expected) -> None:
    assert as_station_request(message, previous, lang) == expected


# ── a speaker group by name ──────────────────────────────────────────────────

GROUP = "media_player.everyone"


def _group_registry():
    registry = _registry()
    registry._music_area = MagicMock(return_value=None)
    registry._is_exposed = MagicMock(return_value=True)
    entries = {
        GROUP: SimpleNamespace(
            platform="music_assistant", name=None, original_name="Everyone",
            aliases=["all speakers"], disabled_by=None, hidden_by=None,
        ),
        "media_player.kitchen": SimpleNamespace(
            platform="music_assistant", name=None, original_name="Kitchen",
            aliases=[], disabled_by=None, hidden_by=None,
        ),
        "media_player.tv": SimpleNamespace(
            platform="cast", name=None, original_name="all speakers TV",
            aliases=[], disabled_by=None, hidden_by=None,
        ),
    }
    registry._hass.states.get = lambda eid: SimpleNamespace(
        state="idle", attributes={"friendly_name": entries[eid].original_name}
    ) if eid in entries else None
    return registry, entries


@pytest.mark.parametrize("area", ["all speakers", "Everyone", "Everyon"])
async def test_a_speaker_group_by_name_alias_or_sound(area, monkeypatch) -> None:
    """Live 2026-11-18: "Play on all speakers." got "I can only play music on
    one speaker at a time", and the group's name was heard as another word."""
    registry, entries = _group_registry()
    from custom_components.ai_plugin.tools.ha_local import media

    monkeypatch.setattr(
        media.er, "async_get", lambda _hass: SimpleNamespace(entities=entries), raising=False
    )
    result = await registry._play_music("Hit North", area)
    assert result == f"OK — playing 'Hit North' (track) on {GROUP}."


async def test_an_unknown_speaker_plays_nothing(monkeypatch) -> None:
    registry, entries = _group_registry()
    from custom_components.ai_plugin.tools.ha_local import media

    monkeypatch.setattr(
        media.er, "async_get", lambda _hass: SimpleNamespace(entities=entries), raising=False
    )
    result = await registry._play_music("Hit North", "garage")
    assert result == "Unknown area or speaker 'garage'. Try list_areas."
    assert _played(registry) is None


@pytest.mark.parametrize("prompt", [SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE])
def test_both_prompts_name_speaker_groups(prompt) -> None:
    assert "area: 'all speakers'" in prompt
