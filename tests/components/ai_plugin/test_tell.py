"""Giving someone a message, and moving the music to another room."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .test_conversation import _voice_orch_with_intent  # isort: skip (stubs HA first)

from custom_components.ai_plugin import satellite_speech
from custom_components.ai_plugin.orchestrator import core
from custom_components.ai_plugin.orchestrator.grounding import _any_media_success
from custom_components.ai_plugin.providers import ChatResponse, ToolCall
from custom_components.ai_plugin.tools.ha_local import HALocalToolRegistry, media
from custom_components.ai_plugin.tools.tell import async_tell

PHONE = SimpleNamespace(service="mobile_app_anna", name="Anna")
PHONES = "custom_components.ai_plugin.shortcuts.phone.phones_of"
PEOPLE = "custom_components.ai_plugin.shortcuts.phone.people_with_phones"


def _hass():
    hass = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass


async def _tell(hass, sender=None, known=True, **args):
    return await async_tell(
        hass, args, language="de-DE",
        resolve_person=lambda _: "person.anna" if known else None,
        sender_name=sender,
    )


async def test_a_message_goes_to_the_persons_phone() -> None:
    hass = _hass()
    with patch(PHONES, return_value=[PHONE]):
        result = await _tell(hass, person="Anna", message="Dinner is ready")
    assert result == "Sent to Anna's phone: Dinner is ready."
    call = hass.services.async_call.await_args
    assert call.args[:2] == ("notify", "mobile_app_anna")
    assert call.args[2] == {"title": "Nachricht", "message": "Dinner is ready"}


async def test_a_message_says_who_it_is_from_when_that_is_known() -> None:
    hass = _hass()
    with patch(PHONES, return_value=[PHONE]):
        await _tell(hass, sender="Ben", person="Anna", message="x")
    assert hass.services.async_call.await_args.args[2]["title"] == "Nachricht von Ben"


@pytest.mark.parametrize(("known", "phones"), [(False, []), (True, [])])
async def test_no_phone_names_who_has_one(known, phones) -> None:
    hass = _hass()
    with patch(PHONES, return_value=phones), patch(PEOPLE, return_value={"person.ben": "Ben"}):
        result = await _tell(hass, known=known, person="Anna", message="x")
    assert result == "[Not sent: no phone known for Anna. People with a phone: Ben.]"
    hass.services.async_call.assert_not_awaited()


async def test_a_phone_that_does_not_take_it_is_reported() -> None:
    hass = _hass()
    hass.services.async_call.side_effect = RuntimeError("offline")
    with patch(PHONES, return_value=[PHONE]):
        result = await _tell(hass, person="Anna", message="x")
    assert result == "[Not sent: Anna's phone did not take it.]"


async def test_a_message_for_a_room_is_spoken_there() -> None:
    hass = _hass()
    with patch.object(satellite_speech, "satellite_in_area", return_value="sat-kitchen"), \
         patch.object(satellite_speech, "async_announce", AsyncMock(return_value="spoken")) as say:
        result = await _tell(hass, area="kitchen", message="Dinner is ready")
    assert result == "Said in kitchen: Dinner is ready."
    say.assert_awaited_once_with(hass, "sat-kitchen", "scheduled_say", "de", label="Dinner is ready")


async def test_a_room_without_a_satellite_says_so() -> None:
    with patch.object(satellite_speech, "satellite_in_area", return_value=None):
        result = await _tell(_hass(), area="attic", message="x")
    assert result == "[Not said: there is no satellite in attic. Tell the user.]"


async def test_a_satellite_that_cannot_play_says_so() -> None:
    with patch.object(satellite_speech, "satellite_in_area", return_value="s"), \
         patch.object(satellite_speech, "async_announce", AsyncMock(return_value=None)):
        result = await _tell(_hass(), area="kitchen", message="x")
    assert result.startswith("[Not said")


@pytest.mark.parametrize(
    ("args", "words"),
    [({"person": "Anna"}, "'message'"), ({"message": "x"}, "who or which room")],
)
async def test_what_is_missing_goes_back_to_the_model(args, words) -> None:
    hass = _hass()
    result = await _tell(hass, **args)
    assert result.startswith("[Not sent") and words in result
    hass.services.async_call.assert_not_awaited()


# ── moving the music ─────────────────────────────────────────────────────────


def _registry(playing: tuple[str, ...], *, service: bool = True) -> HALocalToolRegistry:
    hass = _hass()
    hass.services.has_service = MagicMock(return_value=service)
    players = ("media_player.kitchen", "media_player.bedroom", "media_player.other_cast")
    hass.states.get = lambda eid: SimpleNamespace(
        state="playing" if eid in playing else "idle", attributes={}
    )
    entities = {
        eid: SimpleNamespace(
            platform="cast" if eid.endswith("_cast") else "music_assistant"
        )
        for eid in players
    }
    registry = HALocalToolRegistry(hass)
    registry._music_area = MagicMock(return_value=SimpleNamespace(name="bedroom"))
    registry._music_player_in_area = MagicMock(return_value="media_player.bedroom")
    registry._entities = entities
    return registry


async def _move(registry, area="bedroom"):
    with patch.object(media.er, "async_get", return_value=SimpleNamespace(entities=registry._entities)):
        return await registry.move_music(area)


async def test_the_music_moves_to_the_room() -> None:
    registry = _registry(("media_player.kitchen",))
    result = await _move(registry)
    assert result.startswith("OK") and "media_player.kitchen" in result
    call = registry._hass.services.async_call.await_args
    assert call.args[:2] == ("music_assistant", "transfer_queue")
    assert call.kwargs["target"] == {"entity_id": "media_player.bedroom"}
    assert call.kwargs["service_data"] == {"source_player": "media_player.kitchen", "auto_play": True}


async def test_only_music_assistant_players_count_as_playing() -> None:
    result = await _move(_registry(("media_player.other_cast",)))
    assert result.startswith("[Nothing is playing")


@pytest.mark.parametrize(
    ("registry", "words"),
    [
        (_registry(()), "Nothing is playing"),
        (_registry(("media_player.bedroom",)), "already playing"),
        (_registry(("media_player.kitchen",), service=False), "needs Music Assistant"),
    ],
)
async def test_when_there_is_nothing_to_move(registry, words) -> None:
    result = await _move(registry)
    assert result.startswith("[") and words in result
    registry._hass.services.async_call.assert_not_awaited()


async def test_a_room_can_be_named_by_a_synonym() -> None:
    registry = _registry(("media_player.kitchen",))
    registry._music_area.return_value = None
    bedroom = SimpleNamespace(name="bedroom")
    with patch("custom_components.ai_plugin.shortcuts.registry._resolve_area", return_value=bedroom):
        result = await _move(registry, "Schlafzimmer")
    assert result.startswith("OK")
    registry._music_player_in_area.assert_called_with(bedroom, True)


async def test_a_room_it_does_not_know_or_has_no_speaker() -> None:
    registry = _registry(("media_player.kitchen",))
    registry._music_area.return_value = None
    with patch("custom_components.ai_plugin.shortcuts.registry._resolve_area", return_value=None):
        assert "Unknown area" in await _move(registry, "attic")
    registry._music_area.return_value = SimpleNamespace(name="attic")
    registry._music_player_in_area.return_value = None
    assert "No speaker in attic" in await _move(registry, "attic")


async def test_a_failing_move_says_so() -> None:
    registry = _registry(("media_player.kitchen",))
    registry._hass.services.async_call.side_effect = RuntimeError("boom")
    assert await _move(registry) == "[move_music failed: boom]"


def _moved(result: str) -> list[dict]:
    return [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "move_music"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "move_music", "content": result},
    ]


def test_a_move_is_silent_only_if_it_worked() -> None:
    """The music moving is the confirmation; a failure has to be heard."""
    assert _any_media_success(_moved("OK — moved"))
    assert not _any_media_success(_moved("[Nothing is playing]"))


async def test_the_orchestrator_reaches_both_tools() -> None:
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._hass = None
    orch._scheduler = MagicMock()
    orch._ha_local = MagicMock()
    orch._ha_local.tool_names = set()
    orch._ha_local.move_music = AsyncMock(return_value="OK — moved")
    assert await orch._dispatch_tool("move_music", {"area": "bedroom"}, None, device_id="s") == "OK — moved"
    orch._ha_local.move_music.assert_awaited_once_with("bedroom", "s")
    tell = AsyncMock(return_value="Sent")
    with patch.object(core, "async_tell", tell), patch.object(core, "person_for_user", return_value=None):
        assert await orch._dispatch_tool("tell", {"message": "x", "person": "A"}, None) == "Sent"
    assert orch._is_lookup_tool("tell") is False
    assert orch._is_lookup_tool("move_music") is False


async def test_a_message_is_confirmed_aloud() -> None:
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._hass = None
    orch._scheduler = MagicMock()
    orch._scheduler.async_handle_spoken = AsyncMock(return_value=None)
    orch._provider.async_chat = AsyncMock(side_effect=[
        ChatResponse(content=None, tool_calls=[
            ToolCall(id="c1", name="HassLightSet", arguments={}),
            ToolCall(id="c2", name="tell", arguments={"message": "x", "person": "A"}),
        ]),
        ChatResponse(content="Sent to Anna.", tool_calls=[]),
    ])
    with patch.object(core, "async_tell", AsyncMock(return_value="Sent")), \
         patch.object(core, "person_for_user", return_value=None), \
         patch.object(core, "_any_actuator_success", return_value=True):
        reply = await orch.async_process("Dim the lights and tell Anna dinner is ready", "c1", "en")
    assert reply == "Sent to Anna."
