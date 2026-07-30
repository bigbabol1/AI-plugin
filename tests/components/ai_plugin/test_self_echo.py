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


def _voice_input(text, device_id="sat1"):
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    return ConversationInput(
        text=text, context=Context(), conversation_id="c",
        device_id=device_id, language="en", agent_id=None,
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
