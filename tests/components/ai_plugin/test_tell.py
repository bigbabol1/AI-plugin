"""Giving someone a message, and moving the music to another room."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .test_conversation import _voice_orch_with_intent  # isort: skip (stubs HA first)

from custom_components.ai_plugin import satellite_speech
from custom_components.ai_plugin.orchestrator import core
from custom_components.ai_plugin.orchestrator.prompts import _speaker_block
from custom_components.ai_plugin.i18n import L
from custom_components.ai_plugin.orchestrator.grounding import (
    _any_unobservable_action,
    _any_media_success,
    _claims_undone_action,
    _claims_unsent_message,
    claims_action_done,
)
from custom_components.ai_plugin.providers import ChatResponse, ToolCall
from custom_components.ai_plugin.tools.ha_local import HALocalToolRegistry, media
from custom_components.ai_plugin.tools.tell import (
    TOOL_SCHEMA,
    async_broadcast,
    async_tell,
    tell_schema,
)

PHONE = SimpleNamespace(service="mobile_app_anna", name="Anna", apple=True)
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


def _person_description(schema) -> str:
    return schema["function"]["parameters"]["properties"]["person"]["description"]


def test_the_tool_names_the_people_with_a_phone() -> None:
    """Nothing else told the model who lives here: "To Ben ??" got "I'm not
    sure who Ben is" (2026-11-09)."""
    text = _person_description(tell_schema(["Ben", "Anna"]))
    assert "People with a phone: Anna, Ben." in text
    assert "What should I tell Anna?" in text
    assert "People with a phone" not in _person_description(TOOL_SCHEMA)


def test_with_nobody_to_name_the_tool_stays_as_it_is() -> None:
    assert tell_schema([]) is TOOL_SCHEMA
    assert tell_schema(["", None]) is TOOL_SCHEMA


def test_the_orchestrator_sends_the_tool_with_the_names() -> None:
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._mcp = orch._web_search = orch._browse_url = orch._memory = orch._ha_local = None
    orch._prune_schemas = False
    orch._scheduler = MagicMock()
    orch._hass = MagicMock()
    with patch.object(core, "people_with_phones", return_value={"person.ben": "Ben"}):
        schemas = orch._tool_schemas_for("hi", False, None)
    (tell,) = [s for s in schemas if s["function"]["name"] == "tell"]
    assert "People with a phone: Ben." in _person_description(tell)
    orch._hass = None
    (tell,) = [s for s in orch._tool_schemas_for("hi", False, None) if s["function"]["name"] == "tell"]
    assert tell is TOOL_SCHEMA


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


# ── "here", and a playing group ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    [
        "Move the playback in the bathroom to this room.",
        "Move the music here.",
        "Bring the music over here please.",
        "Verschieb die Musik hierher.",
        "Die Musik aus dem Bad in dieses Zimmer.",
        "Pon la música aquí.",
        "Mets la musique ici.",
        "Zet de muziek hierheen.",
        "Przenieś muzykę do tego pokoju.",
        "Põe a música aqui.",
    ],
)
def test_the_music_is_moved_here(message: str) -> None:
    assert core.moves_here(message)


@pytest.mark.parametrize(
    "message",
    [
        "Move the music to the bedroom.",
        "Move the music from here.",
        "Move the music in this room to the kitchen.",
        "I'm here in the kitchen, move the music to the bedroom.",
        "Play it everywhere except here.",
        "Verschieb die Musik von hier in die Küche.",
        "",
    ],
)
def test_the_music_is_not_moved_here(message: str) -> None:
    assert not core.moves_here(message)


def _moving_orch():
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._hass = MagicMock()
    orch._scheduler = MagicMock()
    orch._ha_local = MagicMock()
    orch._ha_local.tool_names = set()
    orch._ha_local.move_music = AsyncMock(return_value="OK — moved")
    return orch


async def _dispatch_move(orch, area: str, message: str, room_id: str | None) -> None:
    areas = MagicMock()
    areas.async_get_area.return_value = SimpleNamespace(name="Küche")
    with patch.object(core, "_caller_area_id", return_value=room_id), \
         patch.object(core.ar, "async_get", return_value=areas):
        await orch._dispatch_tool(
            "move_music", {"area": area}, None, user_message=message, device_id="s"
        )


async def test_here_means_the_room_of_the_satellite_whatever_the_model_sent() -> None:
    orch = _moving_orch()
    await _dispatch_move(orch, "living room", "Move the music in the bathroom to this room.", "kuche")
    orch._ha_local.move_music.assert_awaited_once_with("Küche", "s")


async def test_a_room_named_as_the_destination_stays() -> None:
    orch = _moving_orch()
    await _dispatch_move(orch, "bedroom", "Move the music to the bedroom.", "kuche")
    orch._ha_local.move_music.assert_awaited_once_with("bedroom", "s")


async def test_a_satellite_without_a_room_keeps_the_room_sent() -> None:
    orch = _moving_orch()
    await _dispatch_move(orch, "bedroom", "Move the music here.", None)
    orch._ha_local.move_music.assert_awaited_once_with("bedroom", "s")


async def test_a_playing_group_moves_rather_than_its_member() -> None:
    registry = _registry(("media_player.big_speaker", "media_player.wohnzimmer"))
    registry._entities = {
        eid: SimpleNamespace(platform="music_assistant")
        for eid in ("media_player.big_speaker", "media_player.wohnzimmer", "media_player.bedroom")
    }
    states = {
        "media_player.big_speaker": SimpleNamespace(state="playing", attributes={"mass_player_type": "player"}),
        "media_player.wohnzimmer": SimpleNamespace(state="playing", attributes={"mass_player_type": "group"}),
        "media_player.bedroom": SimpleNamespace(state="idle", attributes={"mass_player_type": "player"}),
    }
    registry._hass.states.get = states.get
    result = await _move(registry)
    assert "from media_player.wohnzimmer to" in result
    call = registry._hass.services.async_call.await_args
    assert call.kwargs["service_data"]["source_player"] == "media_player.wohnzimmer"


# ── a reply claiming something was done ──────────────────────────────────────


@pytest.mark.parametrize(
    "reply",
    [
        "I have moved the music to the bathroom for you.",
        "I've removed eggs from the shopping list.",
        "Okay, I've turned off the lights.",
        "Ich habe die Musik ins Bad verschoben.",
        "He movido la música.",
        "J'ai déplacé la musique.",
        "Ik heb de muziek verplaatst.",
        "Przeniosłem muzykę.",
        "Movi a música.",
    ],
)
def test_replies_that_claim_an_action(reply: str) -> None:
    assert claims_action_done(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "I haven't moved anything.",
        "I have not moved the music.",
        "I couldn't carry that out. Try naming the device or the room.",
        "The lights are on.",
        "Would you like me to move the music?",
        "Ich habe nichts geändert.",
        "Ik heb niets gewijzigd.",
        "No he cambiado nada.",
        "Je n'ai rien changé.",
        "Não alterei nada.",
        "",
    ],
)
def test_replies_that_claim_no_action(reply: str) -> None:
    assert not claims_action_done(reply)


CLAIM = "I have moved the music to the bathroom for you."


def _ran(name: str, result: str) -> list[dict]:
    return [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": name}}]},
        {"role": "tool", "tool_call_id": "c1", "name": name, "content": result},
    ]


@pytest.mark.parametrize(
    ("message", "reply", "tool_msgs", "undone"),
    [
        ("Play the playback in the bathroom also here.", CLAIM, [], True),
        ("Move the music.", CLAIM, _ran("move_music", "[Nothing is playing]"), True),
        ("Move the music.", CLAIM, _ran("move_music", "OK — moved"), False),
        ("Remove eggs.", "I've removed eggs.", _ran("HassListRemoveItem", "Removed"), False),
        ("Move the music.", CLAIM, _ran("web_search", "some result"), False),
        ("Did you move the music?", CLAIM, [], False),
        ("Where is the music?", CLAIM, [], False),
        ("Move the music.", "Which room?", [], False),
    ],
)
def test_a_claim_needs_something_behind_it(message, reply, tool_msgs, undone) -> None:
    assert _claims_undone_action(message, reply, tool_msgs) is undone


async def test_a_false_claim_is_replaced_in_the_reply_and_in_history() -> None:
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._provider.async_chat = AsyncMock(return_value=ChatResponse(content=CLAIM, tool_calls=[]))
    reply = await orch.async_process("Bring the bathroom music along.", "conv-claim", "en")
    assert reply == L.template("err_action_failed", "en")
    assert orch._context_mgr.get_history("conv-claim")[-1]["content"] == reply


async def test_a_true_claim_stays_in_history() -> None:
    orch = _voice_orch_with_intent("HassListRemoveItem", "I've removed eggs.")
    await orch.async_process("Remove eggs from the list.", "conv-true", "en")
    assert orch._context_mgr.get_history("conv-true")[-1]["content"] == "I've removed eggs."


# ── a second room as well ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("Also play in the bathroom.", "en"),
        ("Play the playback in the bathroom also here.", "en"),
        ("Play it in the kitchen too.", "en"),
        ("Play it in the kitchen as well.", "en"),
        ("Play the music on multiple speakers.", "en"),
        ("Spiel das auch im Bad.", "de"),
        ("Auch in der Küche abspielen.", "de"),
        ("Spiel die Musik in zwei Räumen.", "de"),
        ("Pon la música también en la cocina.", "es"),
        ("Joue aussi dans la cuisine.", "fr"),
        ("Speel het ook in de keuken.", "nl"),
        ("Zagraj też w kuchni.", "pl"),
        ("Toca também na cozinha.", "pt"),
    ],
)
def test_asking_for_a_second_room(message: str, lang: str) -> None:
    assert core.asks_for_second_room(message, lang)


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("Play some jazz.", "en"),
        ("Also play some jazz.", "en"),
        ("Play the Beatles too.", "en"),
        ("Play jazz in the kitchen.", "en"),
        ("Play something on Spotify too.", "en"),
        ("Move the music to the bathroom.", "en"),
        ("Also turn off the lights in the kitchen.", "en"),
        ("Spiel etwas Jazz im Bad.", "de"),
        ("Mach auch im Bad das Licht aus.", "de"),
    ],
)
def test_not_asking_for_a_second_room(message: str, lang: str) -> None:
    assert not core.asks_for_second_room(message, lang)


def _second_room_orch(playing: list[str]):
    orch = _voice_orch_with_intent("move_music", "")
    orch._ha_local = MagicMock()
    orch._ha_local.tool_names = set()
    orch._ha_local.music_players_playing = MagicMock(return_value=playing)
    orch._ha_local.move_music = AsyncMock(return_value="OK — moved")
    orch._provider.async_chat = AsyncMock(
        return_value=ChatResponse(content="Which song?", tool_calls=[])
    )
    return orch


async def test_a_second_room_is_refused_and_nothing_moves() -> None:
    orch = _second_room_orch(["media_player.kitchen"])
    reply = await orch.async_process("Also play in the bathroom.", "conv-room", "en")
    assert reply == L.template("second_room_refused", "en")
    orch._provider.async_chat.assert_not_awaited()
    orch._ha_local.move_music.assert_not_awaited()
    assert orch._context_mgr.get_history("conv-room")[-1]["content"] == reply


async def test_with_nothing_playing_the_model_answers() -> None:
    orch = _second_room_orch([])
    reply = await orch.async_process("Also play in the bathroom.", "conv-room2", "en")
    assert reply != L.template("second_room_refused", "en")
    orch._provider.async_chat.assert_awaited()


async def test_a_broadcast_is_spoken_by_every_other_satellite() -> None:
    with patch.object(
        satellite_speech, "announcing_satellites", return_value=["bath", "kitchen"]
    ) as found, patch.object(
        satellite_speech, "async_announce", AsyncMock(return_value="spoken")
    ) as say:
        result = await async_broadcast(
            MagicMock(), {"message": "Where are you?"}, language="en-US", device_id="living"
        )
    assert result == "Said in every other room: Where are you?"
    assert found.call_args.kwargs == {"besides": "living"}
    assert [c.args[1] for c in say.await_args_list] == ["bath", "kitchen"]
    assert say.await_args_list[0].args[2:] == ("scheduled_say", "en")
    assert say.await_args_list[0].kwargs == {"label": "Where are you?"}


@pytest.mark.parametrize(
    ("devices", "spoken", "start"),
    [
        ([], [], "[Not said: there is no other"),
        (["a"], [None], "[Not said: no satellite could"),
        (["a", "b"], ["x", None], "Said on 1 of 2"),
    ],
)
async def test_a_broadcast_says_what_did_not_reach_anyone(devices, spoken, start) -> None:
    with patch.object(satellite_speech, "announcing_satellites", return_value=devices), \
         patch.object(satellite_speech, "async_announce", AsyncMock(side_effect=spoken)):
        result = await async_broadcast(MagicMock(), {"message": "x"}, language="en")
    assert result.startswith(start)


async def test_a_broadcast_needs_a_message() -> None:
    result = await async_broadcast(MagicMock(), {"message": " "}, language="en")
    assert result.startswith("[Not said")


def test_only_satellites_that_can_announce_hear_a_broadcast() -> None:
    def entry(entity_id, device, features=1, config="esphome"):
        return SimpleNamespace(
            entity_id=entity_id, device_id=device,
            supported_features=features, config_entry_id=config,
        )

    hass = MagicMock()
    hass.config_entries.async_get_entry.side_effect = lambda c: SimpleNamespace(domain=c)
    registry = MagicMock()
    registry.entities.values.return_value = [
        entry("assist_satellite.kitchen", "kitchen"),
        entry("assist_satellite.living", "living"),
        entry("assist_satellite.silent", "silent", features=0),
        entry("assist_satellite.phone", "phone", config="voip"),
        entry("media_player.kitchen", "kitchen"),
    ]
    component = SimpleNamespace(AssistSatelliteEntityFeature=SimpleNamespace(ANNOUNCE=1))
    with patch.dict(sys.modules, {"homeassistant.components.assist_satellite": component}), \
         patch("homeassistant.helpers.entity_registry.async_get", return_value=registry):
        assert satellite_speech.announcing_satellites(hass, besides="living") == ["kitchen"]


async def test_the_orchestrator_answers_a_broadcast_itself_and_confirms_it_aloud() -> None:
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._hass = MagicMock()
    broadcast = AsyncMock(return_value="Said in every other room: x")
    with patch.object(core, "async_broadcast", broadcast):
        result = await orch._dispatch_tool(
            "HassBroadcast", {"message": "x"}, None, device_id="living", language="en"
        )
    assert result == "Said in every other room: x"
    broadcast.assert_awaited_once_with(
        orch._hass, {"message": "x"}, language="en", device_id="living"
    )
    orch._mcp.call_tool.assert_not_called()
    msgs = [{"role": "assistant", "tool_calls": [{"function": {"name": "HassBroadcast"}}]}]
    assert _any_unobservable_action(msgs) is True


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("Broadcast.", "en"),
        ("Announce", "en"),
        ("Please make an announcement.", "en"),
        ("Durchsage.", "de"),
        ("Mach eine Durchsage.", "de"),
        ("Anuncia.", "es"),
        ("Annonce.", "fr"),
        ("Omroepen.", "nl"),
        ("Ogłoś.", "pl"),
        ("Anuncia.", "pt"),
    ],
)
def test_a_broadcast_with_nothing_to_say(message: str, lang: str) -> None:
    assert core.asks_for_empty_broadcast(message, lang)


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("Broadcast, where are you?", "en"),
        ("Announce that dinner is ready.", "en"),
        ("Durchsage: das Essen ist fertig.", "de"),
    ],
)
def test_a_broadcast_with_a_message_stays_with_the_model(message: str, lang: str) -> None:
    assert not core.asks_for_empty_broadcast(message, lang)


@pytest.mark.parametrize(
    ("message", "lang", "expected"),
    [
        ("Never mind.", "en", True),
        ("No, forget it.", "en", True),
        ("Vergiss es.", "de", True),
        ("Stop the music in the kitchen.", "en", False),
        ("I'm here.", "en", False),
    ],
)
def test_taking_a_broadcast_back(message: str, lang: str, expected: bool) -> None:
    assert core.cancels_broadcast(message, lang) is expected


def _broadcast_orch():
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._hass = MagicMock()
    orch._provider.async_chat = AsyncMock(
        return_value=ChatResponse(content="Welcome home!", tool_calls=[])
    )
    return orch


async def test_the_answer_to_what_to_announce_is_announced() -> None:
    orch = _broadcast_orch()
    send = AsyncMock(return_value="Said in every other room: I'm here.")
    with patch.object(core, "async_broadcast", send):
        first = await orch.async_process("Broadcast.", "conv-a", "en", device_id="bedroom")
        # A delayed follow-up arrives in a new conversation.
        second = await orch.async_process("I'm here.", "conv-b", "en", device_id="bedroom")
    assert first == L.template("broadcast_what", "en")
    assert second == L.template("broadcast_sent", "en")
    send.assert_awaited_once_with(
        orch._hass, {"message": "I'm here."}, language="en", device_id="bedroom"
    )
    orch._provider.async_chat.assert_not_awaited()


async def test_only_the_next_turn_is_the_message() -> None:
    orch = _broadcast_orch()
    send = AsyncMock(return_value="Said in every other room: x")
    with patch.object(core, "async_broadcast", send):
        await orch.async_process("Broadcast.", "conv-c", "en", device_id="bedroom")
        await orch.async_process("Dinner is ready.", "conv-c", "en", device_id="bedroom")
        third = await orch.async_process("Hello.", "conv-c", "en", device_id="bedroom")
    assert send.await_count == 1
    assert third == "Welcome home!"


async def test_another_satellite_does_not_answer_for_this_one() -> None:
    orch = _broadcast_orch()
    send = AsyncMock()
    with patch.object(core, "async_broadcast", send):
        await orch.async_process("Broadcast.", "conv-d", "en", device_id="bedroom")
        reply = await orch.async_process("Hello.", "conv-e", "en", device_id="kitchen")
    send.assert_not_awaited()
    assert reply == "Welcome home!"


async def test_an_answer_long_after_the_question_is_an_ordinary_turn() -> None:
    orch = _broadcast_orch()
    send = AsyncMock()
    await orch.async_process("Broadcast.", "conv-f", "en", device_id="bedroom")
    orch._broadcast_asked["bedroom"] -= core._BROADCAST_ANSWER_S + 1
    with patch.object(core, "async_broadcast", send):
        reply = await orch.async_process("Hello.", "conv-f", "en", device_id="bedroom")
    send.assert_not_awaited()
    assert reply == "Welcome home!"


async def test_never_mind_announces_nothing() -> None:
    orch = _broadcast_orch()
    send = AsyncMock()
    with patch.object(core, "async_broadcast", send):
        await orch.async_process("Broadcast.", "conv-g", "en", device_id="bedroom")
        reply = await orch.async_process("Never mind.", "conv-g", "en", device_id="bedroom")
    send.assert_not_awaited()
    assert reply == L.template("broadcast_cancelled", "en")


async def test_a_broadcast_that_reached_nobody_says_so() -> None:
    orch = _broadcast_orch()
    send = AsyncMock(return_value="[Not said: no satellite could play it. Tell the user.]")
    with patch.object(core, "async_broadcast", send):
        await orch.async_process("Broadcast.", "conv-h", "en", device_id="bedroom")
        reply = await orch.async_process("Hello.", "conv-h", "en", device_id="bedroom")
    assert reply == L.template("broadcast_failed", "en")


@pytest.mark.parametrize(
    "reply",
    ["I've broadcast that you are here.", "I have announced it.", "Ich habe es durchgesagt."],
)
def test_a_broadcast_nothing_backs_is_not_claimed(reply: str) -> None:
    assert _claims_undone_action("I'm here.", reply, [])


@pytest.mark.parametrize(
    ("message", "reply", "lang", "expected"),
    [
        ("Broadcast something for me.", "What would you like me to broadcast?", "en", True),
        ("Can you make an announcement?", "What would you like me to announce?", "en", True),
        ("Mach eine Durchsage für mich.", "Was soll ich für dich durchsagen?", "de", True),
        ("Turn on the broadcast.", "Which radio?", "en", False),
        ("Broadcast something.", "Done, I told them.", "en", False),
        ("Dim the lights.", "What should I announce?", "en", False),
    ],
)
def test_the_model_asking_what_to_announce(message, reply, lang, expected) -> None:
    assert core.asks_what_to_broadcast(message, reply, lang) is expected


async def test_the_answer_to_the_models_own_question_is_announced() -> None:
    orch = _broadcast_orch()
    orch._provider.async_chat = AsyncMock(
        return_value=ChatResponse(content="What would you like me to broadcast?", tool_calls=[])
    )
    send = AsyncMock(return_value="Said in every other room: Hello.")
    with patch.object(core, "async_broadcast", send):
        await orch.async_process("Broadcast something for me.", "conv-i", "en", device_id="bedroom")
        reply = await orch.async_process("Hello.", "conv-j", "en", device_id="bedroom")
    assert reply == L.template("broadcast_sent", "en")
    send.assert_awaited_once_with(
        orch._hass, {"message": "Hello."}, language="en", device_id="bedroom"
    )



# ── who a message is for, and answering "what should I say?" ─────────────────

@pytest.mark.parametrize(
    ("reply", "lang", "expected"),
    [
        ("What would you like me to say to Anna?", "en", "Anna"),
        ("What message would you like me to send to Anna?", "en", "Anna"),
        ("I'll send the message to Anna. What would you like me to say?", "en", "Anna"),
        ("Ich schicke Anna eine Nachricht. Was soll ich ihr schreiben?", "de", "Anna"),
        (
            "Oh, my mistake! Since you are Ben, that means Anna is the other person at "
            "home. What message would you like me to send to Anna?",
            "en",
            "Anna",
        ),
        ("Was möchtest du Anna sagen?", "de", "Anna"),
        ("Anna and Ben are home. What would you like me to say?", "en", None),
        ("Who would you like me to send the message to, Anna or Ben?", "en", None),
        ("Would you like me to send a message to Anna instead?", "en", None),
        ("What would you like me to say to Anna or Ben?", "en", None),
        ("What would you like me to say?", "en", None),
        ("What would you like me to say to Anna.", "en", None),
        ("What time should Anna leave?", "en", None),
    ],
)
def test_the_model_asking_what_to_say_to_someone(reply, lang, expected) -> None:
    assert core.person_asked_about(reply, lang, ["Anna", "Ben"]) == expected


@pytest.mark.parametrize(
    ("message", "lang", "expected"),
    [
        ("Not to Anna, the other person.", "en", True),
        ("No, not to Anna.", "en", True),
        ("The other one.", "en", True),
        ("Nicht an Anna, an die andere Person.", "de", True),
        ("Not tonight, I'm working.", "en", False),
        ("Dinner is ready.", "en", False),
        ("Das Essen ist fertig.", "de", False),
    ],
)
def test_correcting_who_a_message_is_for(message, lang, expected) -> None:
    assert core.corrects_message_target(message, lang) is expected


@pytest.mark.parametrize(
    ("message", "reply", "tool_msgs", "unsent"),
    [
        ("Hello, please make a dance.", "I've sent the message to Anna.", [], True),
        ("You did not.", "I apologize, I have now sent the message to Anna.", [], True),
        ("Das Essen ist fertig.", "Ich habe Anna die Nachricht geschickt.", [], True),
        ("Dinner is ready.", "I've sent that to Anna.", [], True),
        ("Is it sent", "I have sent it.", [], False),
        ("Hello.", "Your message has been sent.", [], True),
        ("Hello.", "I've sent it.", _ran("tell", "[Not sent: no phone known for Anna.]"), True),
        ("Hello.", "I've sent it.", _ran("tell", "Sent to Anna's phone: Hello."), False),
        ("She didn't get it.", "I have already sent the message to Anna's phone.", [], False),
        ("Did you send it?", "I have sent it.", [], False),
        ("Hello.", "Who should I send it to?", [], False),
    ],
)
def test_a_sent_message_needs_something_behind_it(message, reply, tool_msgs, unsent) -> None:
    assert _claims_unsent_message(message, reply, tool_msgs) is unsent


def test_a_login_says_who_is_speaking() -> None:
    block = _speaker_block(None, False, signed_in="Anna")
    assert "signed in on this device as Anna" in block


def test_a_recognised_voice_goes_before_the_login() -> None:
    assert "recognised by voice as Ben" in _speaker_block("Ben", True, signed_in="Anna")


def test_no_login_no_speaker_block() -> None:
    assert _speaker_block(None, False) == ""


def test_the_login_name_comes_from_the_person() -> None:
    orch = core.Orchestrator.__new__(core.Orchestrator)
    anna = SimpleNamespace(
        entity_id="person.anna", attributes={"user_id": "u1", "friendly_name": "Anna"}
    )
    orch._hass = MagicMock()
    orch._hass.states.async_all.return_value = [anna]
    orch._hass.states.get.return_value = anna
    assert orch._signed_in_name("u1") == "Anna"
    assert orch._signed_in_name("someone-else") is None
    assert orch._signed_in_name(None) is None


def _message_orch(asks: str = "What would you like me to say to Anna?"):
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._hass = MagicMock()
    orch._scheduler = MagicMock()
    orch._scheduler.async_handle_spoken = AsyncMock(return_value=None)
    orch._scheduler.jobs.return_value = []
    orch._resolve_person = lambda _: "person.anna"
    orch._provider.async_chat = AsyncMock(return_value=ChatResponse(content=asks, tool_calls=[]))
    return orch


async def _ask_then(orch, answer, *, send=None, device="phone"):
    send = send or AsyncMock(return_value="Sent to Anna's phone: x.")
    with patch.object(core, "async_tell", send), patch.object(
        core, "people_with_phones", return_value={"person.anna": "Anna", "person.ben": "Ben"}
    ):
        await orch.async_process("Send Anna a message.", "conv-m", "en", device_id="phone")
        return await orch.async_process(answer, "conv-m", "en", device_id=device), send


async def test_the_answer_to_what_to_say_is_sent_word_for_word() -> None:
    orch = _message_orch()
    reply, send = await _ask_then(orch, "I am getting hungry.")
    assert send.await_args.args[1] == {"message": "I am getting hungry.", "person": "Anna"}
    assert reply == L.template(
        "message_sent_to", "en", name="Anna", label="I am getting hungry."
    )
    assert orch._provider.async_chat.await_count == 1


async def test_never_mind_sends_no_message() -> None:
    reply, send = await _ask_then(_message_orch(), "Never mind.")
    assert reply == L.template("message_cancelled", "en")
    send.assert_not_awaited()


async def test_a_correction_goes_to_the_model_and_nothing_is_sent() -> None:
    orch = _message_orch()
    _, send = await _ask_then(orch, "Not to Anna, the other person.")
    send.assert_not_awaited()
    assert orch._provider.async_chat.await_count == 2


async def test_another_device_does_not_answer_for_this_one() -> None:
    _, send = await _ask_then(_message_orch(), "Hello.", device="other-phone")
    send.assert_not_awaited()


async def test_a_message_answer_long_after_the_question_is_an_ordinary_turn() -> None:
    orch = _message_orch()
    send = AsyncMock(return_value="Sent")
    with patch.object(core, "async_tell", send), patch.object(
        core, "people_with_phones", return_value={"person.anna": "Anna"}
    ):
        await orch.async_process("Send Anna a message.", "conv-l", "en", device_id="phone")
        orch._message_asked["phone"] = ("Anna", core.time.monotonic() - 61)
        await orch.async_process("Hello.", "conv-l", "en", device_id="phone")
    send.assert_not_awaited()


async def test_a_message_that_failed_says_so() -> None:
    reply, _ = await _ask_then(
        _message_orch(), "Hello.", send=AsyncMock(return_value="[Not sent: no phone known.]")
    )
    assert reply == L.template("message_not_sent", "en")


async def test_a_false_sent_claim_is_replaced_in_the_reply_and_in_history() -> None:
    orch = _message_orch("I've sent the message to Anna.")
    with patch.object(core, "people_with_phones", return_value={"person.anna": "Anna"}):
        reply = await orch.async_process("Hello there.", "conv-f", "en", device_id="phone")
    assert reply == L.template("message_not_sent", "en")
    assert orch._context_mgr.get_history("conv-f")[-1]["content"] == reply


async def test_android_gets_a_message_at_once_and_ios_is_left_as_it_is() -> None:
    android = SimpleNamespace(service="mobile_app_ben", name="Ben", apple=False)
    hass = _hass()
    with patch(PHONES, return_value=[android, PHONE]):
        await _tell(hass, message="Dinner is ready", person="Anna")
    sent = {call.args[1]: call.args[2] for call in hass.services.async_call.await_args_list}
    assert sent["mobile_app_ben"]["data"] == {"ttl": 0, "priority": "high"}
    assert "data" not in sent["mobile_app_anna"]


async def test_a_false_removal_claim_asks_which_event() -> None:
    """German request, English event name: the model said it was deleted."""
    orch = _voice_orch_with_intent("HassLightSet", "")
    orch._hass = None
    orch._scheduler = MagicMock()
    orch._scheduler.async_handle_spoken = AsyncMock(return_value=None)
    orch._scheduler.now = MagicMock()
    orch._provider.async_chat = AsyncMock(side_effect=[
        ChatResponse(content=None, tool_calls=[
            ToolCall(id="c1", name="remove_calendar_event", arguments={"summary": "Hausbesichtigung"}),
        ]),
        ChatResponse(content="Ich habe die Hausbesichtigung für heute gelöscht.", tool_calls=[]),
    ])
    with patch.object(core, "async_remove_event", AsyncMock(return_value="[Not removed: nothing called 'Hausbesichtigung'.]")):
        reply = await orch.async_process("Lösch die Hausbesichtigung heute.", "c-rm", "de")
    assert reply == L.template("calendar_not_removed", "de")
    assert reply.endswith("?")
