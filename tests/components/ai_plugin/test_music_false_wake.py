"""A false wake from music is dropped silently; people are not."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ai_plugin import conversation as conv
from custom_components.ai_plugin.conversation import AIPluginConversationEntity
from custom_components.ai_plugin.voice_id import VoiceIDManager
from custom_components.ai_plugin.voice_id.buffer import BYTES_PER_SECOND, AudioBufferStore


def _audio(seconds: float) -> bytes:
    return b"\0\0" * int(seconds * BYTES_PER_SECOND / 2)


def _manager(text: str | None = None, seconds: float = 0.0) -> VoiceIDManager:
    mgr = VoiceIDManager.__new__(VoiceIDManager)
    mgr.buffers = AudioBufferStore()
    mgr.should_identify = MagicMock(return_value=True)
    if text is not None:
        mgr.buffers.put(text, _audio(seconds))
    return mgr


@pytest.mark.parametrize(
    "text,seconds,expected",
    [
        # The real one: 2026-11-01 15:01:50, hobby room.
        ("Nothing.", 9.7, True),
        ("Nothing at all.", 12.0, True),
        ("Thank you.", 7.0, True),
        # The mic closed quickly, which music does not let it do.
        ("Nothing.", 3.6, False),
        # Real turns from the logs with the longest audio for their words.
        ("Turn down the volume.", 7.6, False),
        ("Put an alarm for 10:58.", 8.2, False),
        # A sentence of lyrics is not told apart from a person: it stays.
        ("I want to dance with somebody who loves me", 12.0, False),
    ],
)
def test_too_few_words_for_the_audio(text, seconds, expected) -> None:
    assert _manager(text, seconds).too_few_words_for_the_audio(text, "dev", None) is expected


def test_no_stored_audio_is_no_evidence() -> None:
    assert _manager().too_few_words_for_the_audio("Nothing.", "dev", None) is False


def test_a_recognised_user_is_not_measured() -> None:
    mgr = _manager("Nothing.", 9.7)
    mgr.should_identify = MagicMock(return_value=False)
    assert mgr.too_few_words_for_the_audio("Nothing.", "dev", "user") is False


def test_the_check_leaves_the_clip_for_identification() -> None:
    mgr = _manager("Nothing.", 9.7)
    assert mgr.too_few_words_for_the_audio("Nothing.", "dev", None) is True
    assert mgr.buffers.take_clip("Nothing.") is not None


async def _unknown(reason: str | None) -> bool:
    mgr = _manager("Nothing.", 9.7)
    decision = None if reason is None else SimpleNamespace(reason=reason, person_id=None)
    mgr._async_decide = AsyncMock(return_value=decision)
    return await mgr.async_voice_is_unknown("Nothing.", "dev")


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("below_threshold", True),
        ("too_close", False),  # two people too alike is still somebody
        ("match", False),
        ("no_profiles", False),
        (None, False),  # model loading or fingerprint failed: no evidence
    ],
)
async def test_voice_counts_as_unknown_only_when_measured(reason, expected) -> None:
    assert await _unknown(reason) is expected


def _entity(*, too_few=True, unknown=True):
    ent = AIPluginConversationEntity.__new__(AIPluginConversationEntity)
    ent.hass = MagicMock()
    voice_id = MagicMock()
    voice_id.too_few_words_for_the_audio = MagicMock(return_value=too_few)
    voice_id.async_voice_is_unknown = AsyncMock(return_value=unknown)
    ent._orchestrator = SimpleNamespace(_voice_id=voice_id)
    return ent, voice_id


def _turn(text="Nothing."):
    return SimpleNamespace(text=text, device_id="dev", language="en-US")


async def _decide(ent, turn=None, players=("media_player.hobby",)):
    with (
        patch.object(conv, "satellite_entity", return_value="assist_satellite.hobby"),
        patch.object(conv, "room_players", return_value=list(players)),
    ):
        return await ent._is_music_false_wake(turn or _turn(), None)


async def test_music_and_nobody_known_is_dropped() -> None:
    ent, _ = _entity()
    assert await _decide(ent) is True


async def test_no_music_in_the_room_keeps_the_turn() -> None:
    ent, voice_id = _entity()
    assert await _decide(ent, players=()) is False
    voice_id.async_voice_is_unknown.assert_not_awaited()


async def test_a_known_voice_keeps_the_turn() -> None:
    ent, _ = _entity(unknown=False)
    assert await _decide(ent) is False


async def test_a_command_keeps_the_turn_whoever_says_it() -> None:
    ent, voice_id = _entity()
    assert await _decide(ent, _turn("Stop.")) is False
    voice_id.async_voice_is_unknown.assert_not_awaited()


async def test_a_normal_turn_does_no_extra_work() -> None:
    ent, voice_id = _entity(too_few=False)
    with patch.object(conv, "room_players") as players:
        assert await ent._is_music_false_wake(_turn(), None) is False
    players.assert_not_called()
    voice_id.async_voice_is_unknown.assert_not_awaited()


async def test_without_voice_recognition_nothing_is_dropped() -> None:
    ent, _ = _entity()
    ent._orchestrator = SimpleNamespace(_voice_id=None)
    assert await ent._is_music_false_wake(_turn(), None) is False


# ── through the whole turn ──────────────────────


def _full_entity(options=None):
    from .test_conversation import _make_mock_entry

    ent = AIPluginConversationEntity.__new__(AIPluginConversationEntity)
    ent._entry = _make_mock_entry(options=options)
    ent._attr_unique_id = "t"
    ent._recent_replies = {}
    ent.hass = MagicMock()
    orch = MagicMock()
    orch.async_process = AsyncMock(return_value="Understood, I'm here if you need anything else.")
    orch.is_voice_device = MagicMock(return_value=True)
    orch._voice_id = MagicMock()
    orch._voice_id.wants_follow_up = MagicMock(return_value=None)
    orch._voice_id.too_few_words_for_the_audio = MagicMock(return_value=True)
    orch._voice_id.async_voice_is_unknown = AsyncMock(return_value=True)
    ent._orchestrator = orch
    return ent


def _voice_input(text="Nothing."):
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    return ConversationInput(
        text=text, context=Context(), conversation_id="c",
        device_id="sat1", language="en", agent_id=None,
    )


async def test_the_dropped_turn_says_nothing_and_ends_the_session() -> None:
    ent = _full_entity()
    with (
        patch.object(conv, "satellite_entity", return_value="assist_satellite.hobby"),
        patch.object(conv, "room_players", return_value=["media_player.hobby"]),
    ):
        result = await ent.async_process(_voice_input())
    assert result.response.speech["plain"]["speech"] == ""
    assert result.continue_conversation is False
    ent._orchestrator.async_process.assert_not_awaited()


async def test_the_echo_filter_switch_has_no_say_in_it() -> None:
    """Off is how a home with its own echo handling runs. The guard stayed
    dormant behind this switch after it shipped, and nobody could tell."""
    from custom_components.ai_plugin.const import CONF_BASE_URL, CONF_MODEL

    ent = _full_entity(options={
        CONF_BASE_URL: "http://x/v1", CONF_MODEL: "m", "self_echo_filter": False,
    })
    with (
        patch.object(conv, "satellite_entity", return_value="assist_satellite.hobby"),
        patch.object(conv, "room_players", return_value=["media_player.hobby"]),
    ):
        result = await ent.async_process(_voice_input())
    assert result.response.speech["plain"]["speech"] == ""
    ent._orchestrator.async_process.assert_not_awaited()
