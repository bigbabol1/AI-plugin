"""Replay of a real TTS feedback loop, recorded on the reporting install.

Captured verbatim from HA's assist_pipeline debug store: ten consecutive runs
on 2026-07-30 11:17-11:20 UTC in which the bedroom satellite talked to itself.
Each row is (turn arrival, reply generation, STT text, spoken reply) in seconds
relative to the first turn — real timing matters, because both the echo
deadline and the chain breaker are measured against TTS playback.

Only turns 1 and 3 came from a human. Turns 4-10 are the assistant answering
garbled fragments of its own previous reply, each answer feeding the next.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from custom_components.ai_plugin import conversation as conv

from .test_conversation import _make_mock_entry


RECORDED_LOOP: tuple[tuple[float, float, str, str], ...] = (
    (
        0.0, 16.9,
        " It's really hot today.",
        'It’s really hot today with temperatures in the mid-to-upper 70s right now and heading toward the upper 90s by afternoon! The heatwave has been going on for about five days already, but don’t worry — it should start easing up over the next couple of days as high pressure weakens. Stay hydrated and maybe crank up that Tado Thermostat in your living room if you want some relief!',
    ),
    (
        42.4, 42.4,
        ' in your living room if you want some relief.',
        '',
    ),
    (
        80.0, 82.7,
        ' Where are the people living in this house?',
        "I don't have information about who lives here! Let me check if there's a camera system that might show us... actually, I can only control smart home devices like lights and thermostats - not cameras or video feeds. If you'd like to know more about the house setup, I could list all areas and entities we have connected!",
    ),
    (
        104.8, 106.4,
        ' and I choose to be out connected.',
        "Got it! You're choosing to stay disconnected today - maybe enjoy some quiet time away from all that smart home tech? If you change your mind and want me back online, just let me know!",
    ),
    (
        121.6, 123.4,
        ' Just let you know!',
        "Got it - thanks for letting me know! Is there anything specific you'd like to do today? Maybe I can help with some smart home tasks while we're both chillin'~ (´｡• ᵕ •｡`)",
    ),
    (
        138.8, 140.4,
        " We'll go check in.  Ah.",
        "Got it! We're checking back in now - how's everything going? Anything you'd like to do around here today? (´｡• ᵕ •｡`)",
    ),
    (
        152.1, 153.8,
        ' out.',
        "Got it - heading out! Have a great day and stay safe~ (´｡• ᵕ •｡`) Let me know if you need anything when you're back!",
    ),
    (
        165.0, 167.0,
        " you think when you're back.",
        "I'm always here whenever you need me! No matter where you are - I'll be ready to help as soon as you check in~ (´｡• ᵕ •｡`) Just let me know if there's anything specific you'd like assistance with when you return!",
    ),
    (
        182.6, 184.7,
        ' with your return.',
        "I'm always here whenever you need me! No matter where you are - I'll be ready to help as soon as you check in~ (´｡• ᵕ •｡`) Just let me know if there's anything specific you'd like assistance with when you return!",
    ),
    (
        201.0, 201.0,
        ' assistance with when you return.',
        '',
    ),
)


async def _replay(monkeypatch, rows) -> list[tuple[bool, bool]]:
    """Feed rows through the entity; return [(ran the LLM, still listening)]."""
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    clock = {"t": 0.0}
    monkeypatch.setattr(conv.time, "monotonic", lambda: clock["t"])

    ent = conv.AIPluginConversationEntity.__new__(conv.AIPluginConversationEntity)
    ent._entry = _make_mock_entry()
    ent._attr_unique_id = "t"
    ent._recent_replies = {}
    ent._chain_turns = {}
    orch = MagicMock()
    ent._orchestrator = orch

    base = rows[0][0]
    out: list[tuple[bool, bool]] = []
    for turn_ts, reply_ts, stt, reply in rows:
        clock["t"] = turn_ts - base
        orch.async_process = AsyncMock(return_value=reply)
        result = await ent.async_process(
            ConversationInput(
                text=stt, context=Context(), conversation_id="c",
                device_id="sat1", language="en", agent_id=None,
            )
        )
        ran = orch.async_process.await_count == 1
        # Restamp the stored reply to when it was really generated, so the
        # next turn's echo deadline matches production.
        clock["t"] = reply_ts - base
        if ran and reply:
            hist = ent._recent_replies.get("sat1", [])[-2:]
            ent._recent_replies["sat1"] = [*hist, (clock["t"], reply)]
        out.append((ran, bool(result.continue_conversation)))
        if not result.continue_conversation:
            break  # the satellite stops listening — no further turn can occur
    return out


async def test_recorded_loop_dies_at_the_first_echo(monkeypatch) -> None:
    """Turn 2 is a clean tail echo, so the session must end there.

    It was already being DROPPED before this fix — but the mic stayed armed,
    which is what let turns 3-10 happen at all.
    """
    outcomes = await _replay(monkeypatch, RECORDED_LOOP)

    assert len(outcomes) == 2, "loop should not survive past the first echo"
    assert outcomes[0] == (True, True)      # real question, answered, listening
    assert outcomes[1] == (False, False)    # echo dropped, session ended


async def test_loop_is_bounded_even_when_no_echo_is_recognised(monkeypatch) -> None:
    """Worst case: start at the garbled fragment nothing can match.

    "…list all areas and entities we have connected!" came back as "and I
    choose to be out connected" — two words in common, no shared word pairs.
    No matcher catches that, so the chain breaker has to.
    """
    outcomes = await _replay(monkeypatch, RECORDED_LOOP[2:])

    assert len(outcomes) <= 5, f"loop ran {len(outcomes)} turns unchecked"
    assert outcomes[-1][1] is False, "session must end"
