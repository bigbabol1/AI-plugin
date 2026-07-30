"""Tests for the self-echo filter (v0.9.35)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from custom_components.ai_plugin.conversation import (
    _is_self_echo,
    AIPluginConversationEntity,
)

from .test_conversation import _make_mock_entry


# ── pure matcher ──────────────────────────────────────────────────────────────


def test_echo_detected_on_near_exact_repeat() -> None:
    reply = "The living room is twenty one degrees and the humidity is forty percent."
    # STT re-hears it with a couple of word errors — still caught.
    stt = "the living room is twenty one degrees and humidity is forty percent"
    assert _is_self_echo(stt, [reply]) is True


def test_short_turns_never_filtered() -> None:
    assert _is_self_echo("yes", ["yes please, turning it on now"]) is False
    assert _is_self_echo("turn it off", ["Okay, I turned it off for you."]) is False


def test_real_followup_not_filtered() -> None:
    reply = "It is sunny and twenty degrees in Berlin right now."
    assert _is_self_echo("what about tomorrow morning then", [reply]) is False


def test_no_recent_replies_means_no_echo() -> None:
    assert _is_self_echo("the living room is twenty one degrees", []) is False


# ── entity wiring ─────────────────────────────────────────────────────────────


def _entity(options=None):
    ent = AIPluginConversationEntity.__new__(AIPluginConversationEntity)
    ent._entry = _make_mock_entry(options=options)
    ent._attr_unique_id = "t"
    ent._recent_replies = {}
    orch = MagicMock()
    orch.async_process = AsyncMock(return_value="It is sunny and twenty degrees outside.")
    orch.is_voice_device = MagicMock(return_value=True)
    ent._orchestrator = orch
    return ent


def _voice_input(text, device_id="sat1", language="en"):
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    return ConversationInput(
        text=text, context=Context(), conversation_id="c",
        device_id=device_id, language=language, agent_id=None,
    )


async def test_first_turn_answered_and_recorded() -> None:
    ent = _entity()
    result = await ent.async_process(_voice_input("what's the weather"))
    assert "sunny" in result.response.speech["plain"]["speech"]
    assert ent._recent_replies_for("sat1")  # reply remembered for next turn


async def test_echoed_reply_is_dropped_not_processed() -> None:
    ent = _entity()
    await ent.async_process(_voice_input("what's the weather"))
    ent._orchestrator.async_process.reset_mock()

    # The satellite re-hears its own reply.
    echo = await ent.async_process(
        _voice_input("it is sunny and twenty degrees outside")
    )

    assert echo.response.speech["plain"]["speech"] == ""
    ent._orchestrator.async_process.assert_not_awaited()  # never ran as a command


async def test_filter_off_lets_echo_through() -> None:
    from custom_components.ai_plugin.const import CONF_BASE_URL, CONF_MODEL

    ent = _entity(options={
        CONF_BASE_URL: "http://x/v1", CONF_MODEL: "m", "self_echo_filter": False,
    })
    await ent.async_process(_voice_input("what's the weather"))
    ent._orchestrator.async_process.reset_mock()
    await ent.async_process(_voice_input("it is sunny and twenty degrees outside"))
    ent._orchestrator.async_process.assert_awaited_once()  # filter disabled → processed


def test_reordered_command_not_filtered() -> None:
    """A real command reusing the reply's words in a different order must
    pass — this was the bag-of-words false positive."""
    from custom_components.ai_plugin.conversation import _is_self_echo

    reply = "The living room light is on."
    assert _is_self_echo("turn the living room light on", [reply]) is False


def test_repeat_question_not_filtered() -> None:
    from custom_components.ai_plugin.conversation import _is_self_echo

    reply = "The living room light is on."
    assert _is_self_echo("is the living room light on", [reply]) is False


def test_truncated_echo_fragment_still_dropped() -> None:
    """A contiguous fragment of the reply (mic caught the tail) is echo."""
    from custom_components.ai_plugin.conversation import _is_self_echo

    reply = "The living room light is on and the blinds are closed."
    assert _is_self_echo("light is on and the blinds are closed", [reply]) is True


def test_echo_with_one_stt_error_still_dropped() -> None:
    from custom_components.ai_plugin.conversation import _is_self_echo

    reply = "The temperature in the bedroom is twenty one degrees right now."
    stt = "the temperature in the bedroom is twenty two degrees right now"
    assert _is_self_echo(stt, [reply]) is True


# ── v0.9.42: tail echo (the follow-up feedback loop) ──────────────────────────
# 'Listen for follow-up' re-arms the mic exactly as TTS ends, so it catches the
# reply's last words. Those fragments are below ECHO_MIN_TOKENS, so the bigram
# filter ignored them by design — they ran as commands, got answered, and the
# new reply's tail started the next round.


def test_tail_fragment_of_reply_is_echo() -> None:
    from custom_components.ai_plugin.conversation import _is_tail_echo

    reply = "I'll turn off all the lights in your home now!"
    assert _is_tail_echo("your home now", [(1.0, reply)]) is True
    assert _is_tail_echo("home now", [(1.0, reply)]) is True
    assert _is_tail_echo("now", [(1.0, reply)]) is True


def test_tail_fragment_de() -> None:
    from custom_components.ai_plugin.conversation import _is_tail_echo

    assert _is_tail_echo("sind aus", [(1.0, "OK, alle Lichter sind aus.")]) is True
    assert _is_tail_echo("aus", [(1.0, "OK, alle Lichter sind aus.")]) is True


def test_tail_echo_survives_one_stt_error() -> None:
    from custom_components.ai_plugin.conversation import _is_tail_echo

    reply = "The bedroom light is on."
    assert _is_tail_echo("bedroom light on", [(1.0, reply)]) is True


def test_real_short_commands_are_not_tail_echo() -> None:
    from custom_components.ai_plugin.conversation import _is_tail_echo

    reply = "The living room light is on."
    for text in ("stop", "louder", "turn it off", "and the fan", "thanks"):
        assert _is_tail_echo(text, [(1.0, reply)]) is False, text


def test_single_word_only_matches_the_last_word() -> None:
    from custom_components.ai_plugin.conversation import _is_tail_echo

    reply = "The bedroom light is on."
    assert _is_tail_echo("on", [(1.0, reply)]) is True       # the actual last word
    assert _is_tail_echo("bedroom", [(1.0, reply)]) is False  # mid-sentence word


def test_long_turns_left_to_the_bigram_filter() -> None:
    from custom_components.ai_plugin.conversation import _is_tail_echo

    reply = "The living room light is on and the blinds are closed."
    assert _is_tail_echo("the blinds are closed", [(1.0, reply)]) is False


def test_tail_echo_needs_a_recent_reply() -> None:
    from custom_components.ai_plugin.conversation import _is_tail_echo

    assert _is_tail_echo("home now", []) is False


async def test_tail_echo_ends_the_session() -> None:
    """The loop breaker: after an echo we stop offering a follow-up."""
    ent = _entity()
    ent._orchestrator.async_process = AsyncMock(
        return_value="I'll turn off all the lights in your home now!"
    )
    first = await ent.async_process(_voice_input("switch all lights off"))
    assert first.continue_conversation is True

    ent._orchestrator.async_process.reset_mock()
    echo = await ent.async_process(_voice_input("home now"))

    assert echo.response.speech["plain"]["speech"] == ""
    ent._orchestrator.async_process.assert_not_awaited()
    assert echo.continue_conversation is False


async def test_chain_cap_breaks_undetected_loops() -> None:
    """Even a garbled tail we don't recognise cannot loop forever."""
    ent = _entity()
    # Each reply is distinct prose, so neither echo rule ever fires.
    replies = iter([f"Reply number {n} about something else entirely." for n in range(9)])
    ent._orchestrator.async_process = AsyncMock(
        side_effect=lambda **kw: next(replies)
    )

    outcomes = [
        (await ent.async_process(_voice_input(f"question {n} please"))).continue_conversation
        for n in range(6)
    ]

    # Four back-to-back follow-ups are allowed; the fifth ends the session.
    assert outcomes[:4] == [True, True, True, True]
    assert outcomes[4] is False
    # Counter reset after ending, so a fresh chain starts clean.
    assert outcomes[5] is True


async def test_pause_between_turns_resets_the_chain(monkeypatch) -> None:
    ent = _entity()
    replies = iter([f"Reply number {n} about something else entirely." for n in range(9)])
    ent._orchestrator.async_process = AsyncMock(side_effect=lambda **kw: next(replies))

    clock = {"t": 1000.0}
    monkeypatch.setattr(
        "custom_components.ai_plugin.conversation.time.monotonic",
        lambda: clock["t"],
    )
    outcomes = []
    for n in range(6):
        clock["t"] += 60.0  # a normal pause between turns
        outcomes.append(
            (await ent.async_process(_voice_input(f"question {n} please"))).continue_conversation
        )

    assert all(outcomes), "paced turns must never hit the loop breaker"


def test_tail_echo_window_scales_with_playback_length() -> None:
    """The stored timestamp is from generation — playback still has to happen,
    so a long reply's tail can legitimately arrive many seconds later."""
    from custom_components.ai_plugin.conversation import _is_tail_echo

    short = "Lights off."
    long_reply = (
        "The living room is twenty one degrees, the kitchen is nineteen, "
        "and the bedroom is eighteen degrees with the window open."
    )
    # A short reply finished speaking long ago — 15s later it can't be echo.
    assert _is_tail_echo("off", [(15.0, short)]) is False
    assert _is_tail_echo("off", [(2.0, short)]) is True
    # A 21-word reply takes ~8s to speak, so its tail can still land at 12s.
    assert _is_tail_echo("window open", [(12.0, long_reply)]) is True
    # ...but not once well past any plausible playback.
    assert _is_tail_echo("window open", [(18.0, long_reply)]) is False


async def test_chain_cap_fires_on_slow_chatty_loops(monkeypatch) -> None:
    """The shape of the real loop: 60-word replies, ~17s between turns.

    Measured by wall clock those turns look unhurried, so a flat "back to
    back" window never fired. Measured against playback — 60 words take
    ~24s to speak — every one of them arrives mid-reply.
    """
    ent = _entity()
    chatty = " ".join(f"word{n}" for n in range(60))
    replies = iter([f"{chatty} ending {n}." for n in range(9)])
    ent._orchestrator.async_process = AsyncMock(side_effect=lambda **kw: next(replies))

    clock = {"t": 1000.0}
    monkeypatch.setattr(
        "custom_components.ai_plugin.conversation.time.monotonic",
        lambda: clock["t"],
    )
    outcomes = []
    for n in range(6):
        clock["t"] += 17.0
        outcomes.append(
            (await ent.async_process(_voice_input(f"unrelated question {n} here"))).continue_conversation
        )

    assert outcomes[:4] == [True, True, True, True]
    assert outcomes[4] is False


# ── v0.9.43: playback-overlap echo ────────────────────────────────────────────
# The tail STT rewrites into different words entirely ("…like Trumbull County!"
# → "tremble down"). Nothing lexical survives, but HA's state says the mic was
# open while a speaker in the room was still playing.


def _hass_with_speaker(state: str = "idle", seconds_ago: float = 2.0,
                       speaker_is_caller: bool = False):
    """hass whose living-room speaker is playing / just stopped."""
    from datetime import timedelta
    from types import SimpleNamespace

    from homeassistant.util import dt as dt_util

    speaker_device = "dev_sat" if speaker_is_caller else "dev_speaker"
    devices = {
        "dev_sat": SimpleNamespace(id="dev_sat", area_id="a_liv"),
        "dev_speaker": SimpleNamespace(id="dev_speaker", area_id="a_liv"),
    }
    entities = {
        "media_player.big_speaker": SimpleNamespace(
            entity_id="media_player.big_speaker", area_id=None,
            device_id=speaker_device,
        ),
    }
    hass = MagicMock()
    hass.states.get.return_value = SimpleNamespace(
        state=state,
        last_changed=dt_util.utcnow() - timedelta(seconds=seconds_ago),
    )
    return hass, devices, entities


def _entity_with_hass(monkeypatch, hass, devices, entities):
    from custom_components.ai_plugin import conversation as conv

    ent = _entity()
    ent.hass = hass
    dev_reg = MagicMock()
    dev_reg.async_get.side_effect = devices.get
    ent_reg = MagicMock()
    ent_reg.entities = entities
    monkeypatch.setattr(conv.dr, "async_get", lambda h: dev_reg)
    monkeypatch.setattr(conv.er, "async_get", lambda h: ent_reg)
    monkeypatch.setattr(conv.er, "async_entries_for_device", lambda r, d: [])
    return ent


async def _first_then(ent, second_text: str, first_reply: str, language: str = "en"):
    """Answer one real turn, then feed `second_text` as the follow-up."""
    ent._orchestrator.async_process = AsyncMock(return_value=first_reply)
    await ent.async_process(
        _voice_input("why is it so hot today", device_id="dev_sat", language=language)
    )
    ent._orchestrator.async_process.reset_mock()
    return await ent.async_process(
        _voice_input(second_text, device_id="dev_sat", language=language)
    )


REPLY = (
    "Today's extreme heat is caused by a high-pressure system from the south "
    "that has been keeping temperatures elevated for about five days and "
    "pushing us toward 95 degrees in some areas like Trumbull County!"
)


async def test_mangled_tail_over_playback_is_dropped(monkeypatch) -> None:
    """The verbatim failure: two words with no relation to what was said."""
    ent = _entity_with_hass(monkeypatch, *_hass_with_speaker(seconds_ago=2.0))

    result = await _first_then(ent, " tremble down.", REPLY)

    assert result.response.speech["plain"]["speech"] == ""
    ent._orchestrator.async_process.assert_not_awaited()
    assert result.continue_conversation is False


async def test_speaker_still_playing_also_counts(monkeypatch) -> None:
    """Playback that began with our reply still counts while it runs."""
    ent = _entity_with_hass(
        monkeypatch, *_hass_with_speaker(state="playing", seconds_ago=0.5)
    )

    result = await _first_then(ent, " tremble down.", REPLY)

    assert result.response.speech["plain"]["speech"] == ""


async def test_tv_playing_in_the_room_is_not_our_tts(monkeypatch) -> None:
    """A TV running for five minutes must not make every short turn echo."""
    ent = _entity_with_hass(
        monkeypatch, *_hass_with_speaker(state="playing", seconds_ago=300.0)
    )

    await _first_then(ent, "and the bedroom", REPLY)

    assert ent._orchestrator.async_process.await_count == 1


async def test_real_short_command_over_playback_is_kept(monkeypatch) -> None:
    """A recognisable command is never dropped, however the timing looks."""
    for text, lang in (("turn it off", "en"), ("next", "en"),
                       ("lights off", "en"), ("licht aus", "de"),
                       ("alle lichter aus", "de")):
        ent = _entity_with_hass(monkeypatch, *_hass_with_speaker())
        result = await _first_then(ent, text, REPLY, language=lang)
        assert ent._orchestrator.async_process.await_count == 1, text
        assert result.response.speech["plain"]["speech"], text


async def test_bare_answer_over_playback_is_kept(monkeypatch) -> None:
    """"yes" answers the question the agent just asked — never swallow it."""
    for text in ("yes", "no", "ja", "louder", "again"):
        ent = _entity_with_hass(monkeypatch, *_hass_with_speaker())
        result = await _first_then(ent, text, "Do you want me to cool the room?")
        assert ent._orchestrator.async_process.await_count == 1, text
        assert result.response.speech["plain"]["speech"], text


async def test_wake_word_chime_does_not_trigger_the_rule(monkeypatch) -> None:
    """The satellite's OWN player plays the chime — it must not count."""
    ent = _entity_with_hass(
        monkeypatch, *_hass_with_speaker(speaker_is_caller=True)
    )

    result = await _first_then(ent, " tremble down.", REPLY)

    assert ent._orchestrator.async_process.await_count == 1


async def test_no_recent_reply_means_no_playback_echo(monkeypatch) -> None:
    """Music playing in the room + a fresh wake-word turn is not echo."""
    ent = _entity_with_hass(monkeypatch, *_hass_with_speaker(state="playing"))

    # No prior turn at all → nothing of ours could be playing.
    ent._orchestrator.async_process = AsyncMock(return_value="Sure thing.")
    result = await ent.async_process(_voice_input(" tremble down.", device_id="dev_sat"))

    assert ent._orchestrator.async_process.await_count == 1
    assert result.response.speech["plain"]["speech"] == "Sure thing."


async def test_long_turn_over_playback_is_kept(monkeypatch) -> None:
    """Four words or more carry enough signal for the text matchers."""
    ent = _entity_with_hass(monkeypatch, *_hass_with_speaker())

    result = await _first_then(ent, "what about tomorrow morning", REPLY)

    assert ent._orchestrator.async_process.await_count == 1
    assert result.response.speech["plain"]["speech"]


async def test_recorded_playback_gap_is_covered(monkeypatch) -> None:
    """The measured gaps between speaker-idle and the turn: 2.9s, 3.1s, 3.9s.

    Recorded on 2026-07-30 15:22-15:26 UTC on the living-room satellite,
    whose TTS is mirrored to wohnzimmer_2 / big_speaker_2 / small_speaker_2.
    """
    for gap in (2.9, 3.1, 3.9):
        ent = _entity_with_hass(monkeypatch, *_hass_with_speaker(seconds_ago=gap))
        result = await _first_then(ent, " tremble down.", REPLY)
        assert result.response.speech["plain"]["speech"] == "", gap
        assert result.continue_conversation is False, gap


async def test_playback_gap_beyond_grace_is_not_echo(monkeypatch) -> None:
    """Long after the room went quiet, a short turn is just a short turn."""
    ent = _entity_with_hass(monkeypatch, *_hass_with_speaker(seconds_ago=9.0))

    await _first_then(ent, " tremble down.", REPLY)

    assert ent._orchestrator.async_process.await_count == 1


# ── v0.9.45: delayed follow-up (plugin reopens the mic itself) ────────────────
# The satellite reopens its microphone when its OWN playback ends, ~1s before
# mirrored speakers stop. Nothing in the conversation API moves that, so the
# plugin ends the turn and reopens the mic itself once the room is quiet.


def _entity_with_satellite(monkeypatch, delay: float = 3.0, features: int = 3):
    from types import SimpleNamespace

    from custom_components.ai_plugin import conversation as conv
    from custom_components.ai_plugin.const import (
        CONF_BASE_URL, CONF_FOLLOW_UP_DELAY, CONF_MODEL,
    )

    ent = _entity(options={
        CONF_BASE_URL: "http://x/v1", CONF_MODEL: "m",
        CONF_FOLLOW_UP_DELAY: delay,
    })
    hass = MagicMock()
    hass.states.get.return_value = SimpleNamespace(
        state="idle", attributes={"supported_features": features},
    )
    hass.services.async_call = AsyncMock()
    created: list = []
    hass.async_create_task = lambda coro: created.append(coro) or MagicMock()
    ent.hass = hass
    monkeypatch.setattr(
        conv.er, "async_entries_for_device",
        lambda reg, dev: [SimpleNamespace(entity_id="assist_satellite.sat")],
    )
    monkeypatch.setattr(conv.er, "async_get", lambda h: MagicMock())
    return ent, hass, created


async def test_delay_hands_the_rearm_to_the_plugin(monkeypatch) -> None:
    ent, hass, created = _entity_with_satellite(monkeypatch)

    result = await ent.async_process(_voice_input("what's the weather", device_id="d1"))

    # The satellite is told NOT to reopen; we scheduled it instead.
    assert result.continue_conversation is False
    assert len(created) == 1
    for coro in created:
        coro.close()


async def test_zero_delay_keeps_satellite_behaviour(monkeypatch) -> None:
    ent, hass, created = _entity_with_satellite(monkeypatch, delay=0.0)

    result = await ent.async_process(_voice_input("what's the weather", device_id="d1"))

    assert result.continue_conversation is True
    assert created == []


async def test_satellite_without_start_conversation_is_left_alone(monkeypatch) -> None:
    """supported_features without START_CONVERSATION → don't break follow-up."""
    ent, hass, created = _entity_with_satellite(monkeypatch, features=1)

    result = await ent.async_process(_voice_input("what's the weather", device_id="d1"))

    assert result.continue_conversation is True
    assert created == []


async def test_reopen_waits_for_quiet_then_starts_conversation(monkeypatch) -> None:
    from custom_components.ai_plugin import conversation as conv

    ent, hass, _ = _entity_with_satellite(monkeypatch)
    slept: list[float] = []

    async def _fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(conv.asyncio, "sleep", _fake_sleep)
    # Room already quiet, so the wait loop exits immediately.
    monkeypatch.setattr(conv, "_speaker_was_playing", lambda *a: False)

    await ent._reopen_after_quiet("d1", "assist_satellite.sat", "Two words", 3.0)

    assert 3.0 in slept, "the configured quiet gap must be honoured"
    hass.services.async_call.assert_awaited_once()
    args, kwargs = hass.services.async_call.await_args
    assert args[0] == "assist_satellite" and args[1] == "start_conversation"
    assert args[2]["entity_id"] == "assist_satellite.sat"
    assert args[2]["preannounce"] is False
    assert args[2]["start_message"] == ""


async def test_reopen_skipped_when_satellite_is_busy(monkeypatch) -> None:
    from types import SimpleNamespace

    from custom_components.ai_plugin import conversation as conv

    ent, hass, _ = _entity_with_satellite(monkeypatch)
    hass.states.get.return_value = SimpleNamespace(
        state="responding", attributes={"supported_features": 3}
    )

    async def _fake_sleep(seconds):
        return None

    monkeypatch.setattr(conv.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(conv, "_speaker_was_playing", lambda *a: False)

    await ent._reopen_after_quiet("d1", "assist_satellite.sat", "hi", 3.0)

    hass.services.async_call.assert_not_awaited()


async def test_delayed_follow_up_keeps_conversation_history(monkeypatch) -> None:
    """The reopened mic starts a NEW HA conversation — history must follow."""
    ent, hass, created = _entity_with_satellite(monkeypatch)

    await ent.async_process(_voice_input("what's the weather", device_id="d1"))
    first_id = ent._orchestrator.async_process.await_args.kwargs["conversation_id"]

    # HA mints a fresh conversation id for the reopened session.
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    ent._orchestrator.async_process = AsyncMock(return_value="Sure.")
    await ent.async_process(ConversationInput(
        text="and tomorrow", context=Context(), conversation_id="brand-new-id",
        device_id="d1", language="en", agent_id=None,
    ))
    second_id = ent._orchestrator.async_process.await_args.kwargs["conversation_id"]

    assert second_id == first_id, "follow-up lost the conversation it continues"
    for coro in created:
        coro.close()
