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
