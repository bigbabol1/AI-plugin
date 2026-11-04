"""Tests for the media-shortcut guards: question guard, ambiguous German
bare triggers, exposure filtering, and German article area suffixes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.ai_plugin import shortcuts
from custom_components.ai_plugin.shortcuts import async_try_media_shortcut
from custom_components.ai_plugin.shortcuts.registry import _AREA_SUFFIX_RE


def _make_hass(states: dict[str, str], areas: dict[str, str] | None = None):
    """Build hass + registries. ``states``: entity_id → player state.
    ``areas``: entity_id → area_id (optional)."""
    hass = MagicMock()
    ent_reg = MagicMock()
    ent_reg.entities = {
        eid: SimpleNamespace(
            entity_id=eid, area_id=(areas or {}).get(eid), device_id=None
        )
        for eid in states
    }
    dev_reg = MagicMock()
    hass.states.get.side_effect = lambda eid: (
        SimpleNamespace(state=states[eid]) if eid in states else None
    )
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass, ent_reg, dev_reg


@pytest.fixture
def patched(monkeypatch):
    def _start(hass, ent_reg, dev_reg, area_list=()):
        area_reg = MagicMock()
        area_reg.async_list_areas.return_value = list(area_list)
        monkeypatch.setattr(shortcuts.registry.er, "async_get", lambda h: ent_reg)
        monkeypatch.setattr(shortcuts.registry.dr, "async_get", lambda h: dev_reg)
        monkeypatch.setattr(shortcuts.registry.ar, "async_get", lambda h: area_reg)
    return _start


# ── question guard ────────────────────────────────────────────────────────────


async def test_question_with_trigger_word_not_consumed(patched) -> None:
    """'what does stop mean?' must not stop the music with an empty reply."""
    hass, e, d = _make_hass({"media_player.kitchen": "playing"})
    patched(hass, e, d)
    result = await async_try_media_shortcut(hass, "what does stop mean?", lang="en")
    assert result is None
    hass.services.async_call.assert_not_awaited()


async def test_question_word_without_question_mark_not_consumed(patched) -> None:
    hass, e, d = _make_hass({"media_player.kitchen": "playing"})
    patched(hass, e, d)
    result = await async_try_media_shortcut(hass, "when is the next bus", lang="en")
    assert result is None
    hass.services.async_call.assert_not_awaited()


async def test_plain_command_still_consumed(patched) -> None:
    hass, e, d = _make_hass({"media_player.kitchen": "playing"})
    patched(hass, e, d)
    assert await async_try_media_shortcut(hass, "pause the music", lang="en") == (
        True,
        "",
    )


# ── ambiguous German bare triggers ────────────────────────────────────────────


async def test_modal_particle_halt_not_consumed(patched) -> None:
    """'das ist halt so' is a statement, not a stop command."""
    hass, e, d = _make_hass({"media_player.kitchen": "playing"})
    patched(hass, e, d)
    result = await async_try_media_shortcut(hass, "das ist halt so", lang="de")
    assert result is None
    hass.services.async_call.assert_not_awaited()


async def test_bare_halt_still_stops(patched) -> None:
    hass, e, d = _make_hass({"media_player.kitchen": "playing"})
    patched(hass, e, d)
    assert await async_try_media_shortcut(hass, "halt", lang="de") == (True, "")
    args, kwargs = hass.services.async_call.await_args
    assert args[1] == "media_stop"


async def test_und_so_weiter_not_consumed(patched) -> None:
    hass, e, d = _make_hass({"media_player.kitchen": "paused"})
    patched(hass, e, d)
    result = await async_try_media_shortcut(
        hass, "und so weiter und so fort", lang="de"
    )
    assert result is None
    hass.services.async_call.assert_not_awaited()


async def test_mach_weiter_still_resumes(patched) -> None:
    hass, e, d = _make_hass({"media_player.kitchen": "paused"})
    patched(hass, e, d)
    assert await async_try_media_shortcut(hass, "mach weiter", lang="de") == (True, "")
    args, kwargs = hass.services.async_call.await_args
    assert args[1] == "media_play"


# ── exposure filter ───────────────────────────────────────────────────────────


async def test_unexposed_player_not_targeted(patched, monkeypatch) -> None:
    """A player hidden from the conversation assistant must not be touched;
    with no exposed candidates the shortcut falls through to the LLM."""
    hass, e, d = _make_hass({"media_player.bedroom_tv": "playing"})
    patched(hass, e, d)
    monkeypatch.setattr(shortcuts.registry, "async_should_expose", lambda h, a, eid: False)
    result = await async_try_media_shortcut(hass, "pause", lang="en")
    assert result is None
    hass.services.async_call.assert_not_awaited()


# ── German article area suffixes ──────────────────────────────────────────────


def test_area_suffix_regex_strips_german_articles() -> None:
    m = _AREA_SUFFIX_RE.search("pause die musik in der küche")
    assert m is not None
    assert m.group("area") == "küche"
    m = _AREA_SUFFIX_RE.search("stopp die musik in dem hobbyraum")
    assert m is not None
    assert m.group("area") == "hobbyraum"
    m = _AREA_SUFFIX_RE.search("pause the music in the kitchen")
    assert m is not None
    assert m.group("area") == "kitchen"


async def test_german_article_area_scopes_command(patched) -> None:
    """'pause die musik in der küche' pauses ONLY the kitchen player."""
    hass, e, d = _make_hass(
        {"media_player.kitchen": "playing", "media_player.living": "playing"},
        areas={"media_player.kitchen": "a_kit", "media_player.living": "a_liv"},
    )
    kitchen = SimpleNamespace(id="a_kit", name="Küche", aliases=set())
    patched(hass, e, d, area_list=[kitchen])
    result = await async_try_media_shortcut(
        hass, "pause die musik in der küche", lang="de"
    )
    assert result == (True, "")
    args, kwargs = hass.services.async_call.await_args
    assert kwargs.get("target") == {"entity_id": ["media_player.kitchen"]}


# ── a command with no room named means this room ─────────────────────────────

SAT = "sat-hobby"
HOBBY, BATH, CAST = "media_player.hobby", "media_player.bath", "media_player.cast"


def _home(monkeypatch, states: dict[str, tuple[str, str | None]]):
    """``states``: entity_id → (player state, media title). The hobby speaker is
    in the satellite's room; the others are not."""
    hass = MagicMock()
    areas = {HOBBY: "hobby", BATH: "bath", CAST: "flat"}
    ent_reg = MagicMock()
    ent_reg.entities = {
        eid: SimpleNamespace(entity_id=eid, area_id=areas[eid], device_id=None) for eid in states
    }
    dev_reg = MagicMock()
    dev_reg.async_get.side_effect = lambda device: (
        SimpleNamespace(area_id="hobby") if device == SAT else None
    )
    hass.states.get.side_effect = lambda eid: SimpleNamespace(
        state=states[eid][0], attributes={"media_title": states[eid][1]}
    )
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    monkeypatch.setattr(shortcuts.registry.er, "async_get", lambda h: ent_reg)
    monkeypatch.setattr(shortcuts.registry.dr, "async_get", lambda h: dev_reg)
    monkeypatch.setattr(shortcuts.registry.ar, "async_get", lambda h: MagicMock())
    monkeypatch.setattr(shortcuts.registry, "is_exposed", lambda *_: True)
    return hass


def _targets(hass) -> list[str]:
    return hass.services.async_call.await_args.kwargs["target"]["entity_id"]


async def test_continue_does_not_start_the_bathroom(monkeypatch) -> None:
    """Live 2026-11-01: "Continue playing." played a song left in the bathroom's
    queue, and a speaker with nothing to play failed the whole call."""
    hass = _home(monkeypatch, {
        HOBBY: ("paused", "Song A"), BATH: ("idle", "Song B"), CAST: ("idle", None),
    })
    result = await async_try_media_shortcut(hass, "Continue playing.", device_id=SAT)
    assert result == (True, "")
    assert _targets(hass) == [HOBBY]


async def test_continue_finds_what_is_paused_in_another_room(monkeypatch) -> None:
    hass = _home(monkeypatch, {HOBBY: ("idle", None), BATH: ("paused", "Song B")})
    assert await async_try_media_shortcut(hass, "Continue playing.", device_id=SAT) == (True, "")
    assert _targets(hass) == [BATH]


async def test_continue_picks_up_an_idle_queue_only_in_this_room(monkeypatch) -> None:
    hass = _home(monkeypatch, {HOBBY: ("idle", "Song A"), BATH: ("idle", "Song B")})
    assert await async_try_media_shortcut(hass, "Continue playing.", device_id=SAT) == (True, "")
    assert _targets(hass) == [HOBBY]


async def test_nothing_to_continue_goes_to_the_model(monkeypatch) -> None:
    hass = _home(monkeypatch, {HOBBY: ("idle", None), BATH: ("idle", "Song B")})
    assert await async_try_media_shortcut(hass, "Continue playing.", device_id=SAT) is None
    hass.services.async_call.assert_not_awaited()


async def test_without_a_satellite_only_what_is_paused_continues(monkeypatch) -> None:
    hass = _home(monkeypatch, {HOBBY: ("idle", "Song A"), BATH: ("paused", "Song B")})
    assert await async_try_media_shortcut(hass, "Continue playing.") == (True, "")
    assert _targets(hass) == [BATH]


@pytest.mark.parametrize(("words", "service"), [("Pause.", "media_pause"), ("Next song.", "media_next_track")])
async def test_other_commands_stay_in_this_room_too(monkeypatch, words, service) -> None:
    hass = _home(monkeypatch, {HOBBY: ("playing", "A"), BATH: ("playing", "B")})
    assert await async_try_media_shortcut(hass, words, device_id=SAT) == (True, "")
    assert hass.services.async_call.await_args.args[1] == service
    assert _targets(hass) == [HOBBY]


async def test_a_command_reaches_another_room_if_nothing_plays_here(monkeypatch) -> None:
    hass = _home(monkeypatch, {HOBBY: ("idle", None), BATH: ("playing", "B")})
    assert await async_try_media_shortcut(hass, "Pause.", device_id=SAT) == (True, "")
    assert _targets(hass) == [BATH]


async def test_the_tool_stays_in_this_room_as_well(monkeypatch) -> None:
    from custom_components.ai_plugin.tools.ha_local import HALocalToolRegistry, media

    hass = _home(monkeypatch, {HOBBY: ("paused", "A"), BATH: ("idle", "B")})
    ent_reg = shortcuts.registry.er.async_get(hass)
    monkeypatch.setattr(media.er, "async_get", lambda h: ent_reg)
    registry = HALocalToolRegistry(hass)
    registry._is_exposed = lambda _: True
    assert (await registry._media_command("resume", None, device_id=SAT)).startswith("OK")
    assert _targets(hass) == [HOBBY]


async def test_the_tool_says_when_there_is_nothing_to_continue(monkeypatch) -> None:
    from custom_components.ai_plugin.tools.ha_local import HALocalToolRegistry, media

    hass = _home(monkeypatch, {HOBBY: ("idle", None), BATH: ("idle", "B")})
    ent_reg = shortcuts.registry.er.async_get(hass)
    monkeypatch.setattr(media.er, "async_get", lambda h: ent_reg)
    registry = HALocalToolRegistry(hass)
    registry._is_exposed = lambda _: True
    result = await registry._media_command("resume", None, device_id=SAT)
    assert result.startswith("[Nothing is paused")


@pytest.mark.parametrize("command", ["volume_set", "mute", "unmute", "volume_up"])
async def test_volume_with_nothing_playing_means_this_room(monkeypatch, command) -> None:
    """HassSetVolume, gone from the tool list, worked on an idle speaker too."""
    from custom_components.ai_plugin.tools.ha_local import HALocalToolRegistry, media

    hass = _home(monkeypatch, {HOBBY: ("idle", None), BATH: ("idle", None)})
    ent_reg = shortcuts.registry.er.async_get(hass)
    dev_reg = shortcuts.registry.dr.async_get(hass)
    area_reg = MagicMock()
    area_reg.async_get_area.side_effect = lambda area_id: (
        SimpleNamespace(id=area_id) if area_id else None
    )
    monkeypatch.setattr(media.er, "async_get", lambda h: ent_reg)
    monkeypatch.setattr(media.dr, "async_get", lambda h: dev_reg)
    monkeypatch.setattr(media.ar, "async_get", lambda h: area_reg)
    monkeypatch.setattr(
        shortcuts.volume, "async_step_volume", AsyncMock(), raising=True
    )
    registry = HALocalToolRegistry(hass)
    registry._is_exposed = lambda _: True
    result = await registry._media_command(command, None, level=40, device_id=SAT)
    assert result.startswith("OK") and HOBBY in result and BATH not in result


async def test_volume_with_no_satellite_and_nothing_playing_says_so(monkeypatch) -> None:
    from custom_components.ai_plugin.tools.ha_local import HALocalToolRegistry, media

    hass = _home(monkeypatch, {HOBBY: ("idle", None)})
    ent_reg = shortcuts.registry.er.async_get(hass)
    area_reg = MagicMock()
    area_reg.async_get_area.return_value = None
    monkeypatch.setattr(media.er, "async_get", lambda h: ent_reg)
    monkeypatch.setattr(media.ar, "async_get", lambda h: area_reg)
    registry = HALocalToolRegistry(hass)
    registry._is_exposed = lambda _: True
    result = await registry._media_command("volume_set", None, level=40)
    assert result.startswith("Nothing is")
    hass.services.async_call.assert_not_awaited()
    hass.services.async_call.assert_not_awaited()


def _players(monkeypatch, players: dict[str, dict]):
    """``players``: entity_id to its name, aliases, device class and exposure."""
    from custom_components.ai_plugin.tools.ha_local import HALocalToolRegistry, media

    hass = MagicMock()
    ent_reg = MagicMock()
    ent_reg.entities = {
        eid: SimpleNamespace(
            entity_id=eid, area_id=None, device_id=None, name=p.get("name"),
            original_name=None, aliases=set(p.get("aliases", ())),
        )
        for eid, p in players.items()
    }
    hass.states.get.side_effect = lambda eid: SimpleNamespace(
        state="idle",
        attributes={
            "friendly_name": players[eid].get("name"),
            "device_class": players[eid].get("device_class", "speaker"),
        },
    )
    hass.services.async_call = AsyncMock()
    monkeypatch.setattr(media.er, "async_get", lambda h: ent_reg)
    registry = HALocalToolRegistry(hass)
    registry._is_exposed = lambda eid: players[eid].get("exposed", True)
    registry._resolve_area = lambda name: None
    return registry, hass


@pytest.mark.parametrize(
    "name, players, expected",
    [
        ("TV", {"media_player.a": {"name": "TV"}, "media_player.b": {"name": "Kitchen"}},
         ["media_player.a"]),
        ("Fernseher", {"media_player.a": {"name": "Wohnzimmer", "aliases": ["Fernseher"]}},
         ["media_player.a"]),
        ("TV", {"media_player.a": {"name": "Cast", "aliases": ["TV Chromecast"]}},
         ["media_player.a"]),
        ("TV", {"media_player.a": {"name": "Samsung Frame", "device_class": "tv"},
                "media_player.b": {"name": "Radio"}},
         ["media_player.a"]),
        ("TV", {"media_player.a": {"name": "TV"},
                "media_player.b": {"name": "Samsung Frame", "device_class": "tv"}},
         ["media_player.a"]),
    ],
)
async def test_a_named_player_is_the_target(monkeypatch, name, players, expected) -> None:
    registry, hass = _players(monkeypatch, players)
    assert (await registry._media_command("mute", None, name=name)).startswith("OK")
    assert _targets(hass) == expected


async def test_an_unknown_player_is_said_not_guessed(monkeypatch) -> None:
    """Playing music elsewhere must not get muted for "mute the TV"."""
    registry, hass = _players(monkeypatch, {"media_player.a": {"name": "Radio"}})
    assert await registry._media_command("mute", None, name="TV") == (
        "[No media player is called 'TV'.]"
    )
    hass.services.async_call.assert_not_awaited()


async def test_a_room_given_as_the_device_is_the_room(monkeypatch) -> None:
    """"Unmute the bedroom" came as name: 'bedroom' and was told no such player."""
    registry, hass = _players(monkeypatch, {"media_player.a": {"name": "Speaker"}})
    bedroom = SimpleNamespace(name="Bedroom", id="b")
    registry._resolve_area = lambda name: bedroom if name == "bedroom" else None
    registry._media_players_in_area = lambda area, exposed_only=True: ["media_player.a"]
    assert (await registry._media_command("unmute", None, name="bedroom")).startswith("OK")
    assert _targets(hass) == ["media_player.a"]


async def test_a_player_that_is_not_exposed_is_not_found(monkeypatch) -> None:
    registry, hass = _players(
        monkeypatch, {"media_player.a": {"name": "TV", "exposed": False}}
    )
    assert (await registry._media_command("mute", None, name="TV")).startswith("[No media player")
    hass.services.async_call.assert_not_awaited()


async def test_a_device_passed_as_a_room_still_finds_it(monkeypatch) -> None:
    registry, hass = _players(monkeypatch, {"media_player.a": {"name": "TV"}})
    assert (await registry._media_command("pause", "TV")).startswith("OK")
    assert _targets(hass) == ["media_player.a"]


async def test_an_unknown_room_is_still_an_unknown_room(monkeypatch) -> None:
    registry, hass = _players(monkeypatch, {"media_player.a": {"name": "TV"}})
    assert await registry._media_command("pause", "Garden") == (
        "Unknown area 'Garden'. Try list_areas."
    )
    hass.services.async_call.assert_not_awaited()
