"""'Play some music' names nothing: never search for the word itself."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from .test_conversation import _baseline_orch  # noqa: F401  # isort: skip (stubs HA first)

from custom_components.ai_plugin.const import SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE
from custom_components.ai_plugin.tools.ha_local import HALocalToolRegistry
from custom_components.ai_plugin.tools.ha_local.media import (
    NOTHING_NAMED_RESULT,
    names_nothing_to_play,
    taste_genres,
)


@pytest.mark.parametrize(
    "query",
    [
        "music", "Music.", "some music", "good music", "music I like", "current music",
        "something", "something else", "something different", "anything", "some songs", "tunes",
        "Musik", "etwas Schönes", "irgendwas", "musique", "música", "muziek", "muzyka",
    ],
)
def test_these_name_nothing_to_play(query) -> None:
    assert names_nothing_to_play(query)


@pytest.mark.parametrize(
    "query",
    [
        "electronic", "pop music", "electronic music", "jazz", "Enya", "Hotel California",
        "workout playlist", "Music of the Night", "Music Sounds Better With You",
        "Something by the Beatles", "something stupid", "Anything Goes", "Die Musik von Bach",
    ],
)
def test_these_do(query) -> None:
    assert not names_nothing_to_play(query)


async def test_nothing_named_goes_back_to_the_model_and_plays_nothing() -> None:
    """Live 2026-11-01: "Play music." started the same country mix every time,
    for someone who had saved electronic and pop as their taste."""
    hass = MagicMock()
    hass.services.has_service = MagicMock(return_value=True)
    hass.services.async_call = AsyncMock()
    result = await HALocalToolRegistry(hass)._play_music("music", area="kitchen")
    assert result == NOTHING_NAMED_RESULT
    hass.services.async_call.assert_not_awaited()


@pytest.mark.parametrize("prompt", [SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE])
def test_both_prompts_say_where_the_choice_comes_from(prompt) -> None:
    assert "Never send just 'music'" in prompt
    assert "their taste in [USER FACTS]" in prompt


# ── a different playlist each time, shuffled ─────────────────────────────────

SPEAKER = "media_player.hobby"
PLAYLISTS = [
    {"name": "Electronic Chill ", "uri": "library://1"},
    {"name": "Electronic Focus", "uri": "library://2"},
    {"name": "Electronic Classics", "uri": "library://3"},
]


def _registry(playlists=None, shuffle=False, search_fails=False):
    hass = MagicMock()
    hass.services.has_service = MagicMock(return_value=True)
    hass.config_entries.async_entries = MagicMock(return_value=[SimpleNamespace(entry_id="ma")])
    hass.states.get = lambda _: SimpleNamespace(state="idle", attributes={"shuffle": shuffle})

    async def _call(domain, service, *args, **kwargs):
        if service == "search":
            if search_fails:
                raise RuntimeError("no search")
            return {"playlists": PLAYLISTS if playlists is None else playlists}
        return None

    hass.services.async_call = AsyncMock(side_effect=_call)
    registry = HALocalToolRegistry(hass)
    registry._music_area = MagicMock(return_value=SimpleNamespace(name="hobby"))
    registry._music_player_in_area = MagicMock(return_value=SPEAKER)
    return registry


def _calls(registry, service):
    return [
        c for c in registry._hass.services.async_call.await_args_list if c.args[1] == service
    ]


async def _taste(registry, query="electronic"):
    return await registry._play_music(query, "hobby", from_taste=True)


@pytest.mark.parametrize(
    ("query", "genres"),
    [
        ("electronic, pop", ["electronic", "pop"]),
        ("electronic music and pop music playlists", ["electronic", "pop"]),
        ("Jazz und Soul", ["Jazz", "Soul"]),
        ("electronic", ["electronic"]),
        ("music", []),
        ("", []),
    ],
)
def test_taste_genres(query, genres) -> None:
    assert taste_genres(query) == genres


async def test_a_playlist_from_the_taste_is_found_and_played() -> None:
    registry = _registry(playlists=PLAYLISTS[:1])
    result = await _taste(registry)
    assert result == f"OK — playing 'Electronic Chill' (playlist) on {SPEAKER}."
    search = _calls(registry, "search")[0]
    assert search.args[2] == {
        "config_entry_id": "ma", "name": "electronic", "media_type": ["playlist"], "limit": 6,
    }
    assert search.kwargs["return_response"] is True
    play = _calls(registry, "play_media")[0]
    assert play.kwargs["service_data"]["media_id"] == "library://1"
    assert play.kwargs["service_data"]["media_type"] == "playlist"


async def test_never_the_same_playlist_twice_in_a_row() -> None:
    """Always the first result, it was the same music every time."""
    registry = _registry()
    played = []
    for _ in range(30):
        await _taste(registry)
        played.append(_calls(registry, "play_media")[-1].kwargs["service_data"]["media_id"])
    assert all(a != b for a, b in zip(played, played[1:], strict=False))
    assert set(played) == {"library://1", "library://2", "library://3"}


async def test_every_genre_of_the_taste_gets_its_turn() -> None:
    registry = _registry()
    names = set()
    for _ in range(40):
        await _taste(registry, "electronic, pop")
        names.add(_calls(registry, "search")[-1].args[2]["name"])
    assert names == {"electronic", "pop"}


async def test_a_single_playlist_still_plays_a_second_time() -> None:
    registry = _registry(playlists=PLAYLISTS[:1])
    await _taste(registry)
    assert (await _taste(registry)).startswith("OK")


@pytest.mark.parametrize("kwargs", [{"playlists": []}, {"search_fails": True}])
async def test_no_playlist_found_plays_the_genre(kwargs) -> None:
    registry = _registry(**kwargs)
    assert (await _taste(registry)).startswith("OK")
    assert _calls(registry, "play_media")[0].kwargs["service_data"]["media_id"] == "electronic"


async def test_a_taste_of_nothing_is_sent_back() -> None:
    registry = _registry()
    assert await _taste(registry, "music") == NOTHING_NAMED_RESULT
    registry._hass.services.async_call.assert_not_awaited()


async def test_it_shuffles_before_it_plays() -> None:
    """The queue is shuffled as it loads, so the order matters."""
    registry = _registry()
    await _taste(registry)
    services = [c.args[1] for c in registry._hass.services.async_call.await_args_list]
    assert services.index("shuffle_set") < services.index("play_media")
    call = _calls(registry, "shuffle_set")[0]
    assert call.args[2] == {"shuffle": True}
    assert call.kwargs["target"] == {"entity_id": SPEAKER}


async def test_something_named_plays_in_order_again() -> None:
    registry = _registry()
    await _taste(registry)
    await registry._play_music("Abbey Road", "hobby", media_type="album")
    assert _calls(registry, "shuffle_set")[-1].args[2] == {"shuffle": False}
    assert _calls(registry, "play_media")[-1].kwargs["service_data"]["media_id"] == "Abbey Road"


async def test_it_shuffles_even_if_the_speaker_reports_it_already_does() -> None:
    """Live: a speaker reporting on went off as the playlist started."""
    registry = _registry(shuffle=True)
    await _taste(registry)
    assert _calls(registry, "shuffle_set")[0].args[2] == {"shuffle": True}


async def test_a_speaker_shuffled_on_purpose_stays_shuffled() -> None:
    registry = _registry(shuffle=True)
    await _taste(registry)
    await registry._play_music("Abbey Road", "hobby", media_type="album")
    assert [c.args[2] for c in _calls(registry, "shuffle_set")] == [{"shuffle": True}]


async def test_something_named_never_touches_shuffle_otherwise() -> None:
    registry = _registry()
    await registry._play_music("Enya", "hobby")
    assert _calls(registry, "shuffle_set") == []
    assert _calls(registry, "search") == []


async def test_a_failing_shuffle_still_plays() -> None:
    registry = _registry()
    real = registry._hass.services.async_call.side_effect

    async def _call(domain, service, *args, **kwargs):
        if service == "shuffle_set":
            raise RuntimeError("not supported")
        return await real(domain, service, *args, **kwargs)

    registry._hass.services.async_call.side_effect = _call
    assert (await _taste(registry)).startswith("OK")
    assert _calls(registry, "play_media")


@pytest.mark.parametrize(("value", "passed"), [(True, True), ("true", True), (False, False), (None, False)])
async def test_the_tool_passes_from_taste(value, passed) -> None:
    registry = _registry()
    registry._play_music = AsyncMock(return_value="OK")
    await registry.call_tool("play_music", {"query": "x", "from_taste": value})
    assert registry._play_music.await_args.kwargs["from_taste"] is passed
