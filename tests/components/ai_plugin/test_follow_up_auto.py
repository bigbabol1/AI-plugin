"""'Listen for follow-up' modes: always / never / auto (v0.9.55).

In auto the model decides: a reply ending in a question keeps the
microphone open, a statement ends the turn. Every existing guard (close
phrase, feedback-loop device list, self-echo, chain cap) still overrides.
"""

from __future__ import annotations

import pytest

from custom_components.ai_plugin.const import (
    FOLLOW_UP_ALWAYS,
    FOLLOW_UP_AUTO,
    FOLLOW_UP_AUTO_RULE,
    FOLLOW_UP_NEVER,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
    ends_with_question,
    follow_up_mode,
)

from .test_conversation import (
    _baseline_orch,
    _entity_for_continue,
    _make_mock_entry,
    _voice_input,
)


# ── option normalisation ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ({}, FOLLOW_UP_ALWAYS),  # unset: unchanged default
        ({"continue_conversation": True}, FOLLOW_UP_ALWAYS),  # pre-0.9.55 bool
        ({"continue_conversation": False}, FOLLOW_UP_NEVER),  # pre-0.9.55 bool
        ({"continue_conversation": "auto"}, FOLLOW_UP_AUTO),
        ({"continue_conversation": "never"}, FOLLOW_UP_NEVER),
        ({"continue_conversation": "bogus"}, FOLLOW_UP_ALWAYS),
    ],
)
def test_follow_up_mode_maps_legacy_and_new_values(stored, expected) -> None:
    assert follow_up_mode(stored) == expected


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Which room do you mean?", True),
        ("Welches Licht meinst du?", True),
        ('Did you mean "kitchen"?"', True),
        ("Do you want the big one (the ceiling light?)", True),
        ("Είσαι σίγουρος;", True),  # Greek question mark
        ("你要哪个房间？", True),
        ("The kitchen light is on.", False),
        ("Is it on? Yes, it is on.", False),  # question mid-reply, statement last
        ("", False),
        (None, False),
    ],
)
def test_ends_with_question(reply, expected) -> None:
    assert ends_with_question(reply) is expected


# ── the decision on a voice turn ─────────────────────────────────────────────


async def test_auto_keeps_listening_after_a_question() -> None:
    entity = _entity_for_continue({"continue_conversation": "auto"},
                                  reply="Which room do you mean?")
    result = await entity.async_process(_voice_input("turn on the light"))
    assert result.continue_conversation is True


async def test_auto_ends_after_a_statement() -> None:
    entity = _entity_for_continue({"continue_conversation": "auto"},
                                  reply="The kitchen light is on.")
    result = await entity.async_process(_voice_input("kitchen light on"))
    assert result.continue_conversation is False


async def test_auto_ends_after_an_empty_reply() -> None:
    # Media actions confirm by playing, with an empty reply: nothing was
    # asked, so there is nothing to listen for.
    entity = _entity_for_continue({"continue_conversation": "auto"}, reply="")
    result = await entity.async_process(_voice_input("play some jazz"))
    assert result.continue_conversation is False


async def test_auto_question_still_ended_by_close_phrase() -> None:
    entity = _entity_for_continue({"continue_conversation": "auto"},
                                  reply="Anything else?")
    result = await entity.async_process(_voice_input("thanks jarvis"))
    assert result.continue_conversation is False


async def test_auto_question_still_ended_on_feedback_loop_device() -> None:
    entity = _entity_for_continue(
        {"continue_conversation": "auto", "feedback_loop_devices": ["satdev"]},
        reply="Which room do you mean?",
    )
    result = await entity.async_process(_voice_input("turn on the light"))
    assert result.continue_conversation is False


async def test_always_ignores_reply_shape() -> None:
    entity = _entity_for_continue({"continue_conversation": "always"},
                                  reply="The kitchen light is on.")
    result = await entity.async_process(_voice_input("kitchen light on"))
    assert result.continue_conversation is True


@pytest.mark.parametrize("stored", ["never", False])
async def test_never_ignores_a_question(stored) -> None:
    entity = _entity_for_continue({"continue_conversation": stored},
                                  reply="Which room do you mean?")
    result = await entity.async_process(_voice_input("turn on the light"))
    assert result.continue_conversation is False


async def test_legacy_true_still_always_listens() -> None:
    entity = _entity_for_continue({"continue_conversation": True},
                                  reply="The kitchen light is on.")
    result = await entity.async_process(_voice_input("kitchen light on"))
    assert result.continue_conversation is True


# ── the prompt rule that makes the signal mean something ─────────────────────


def _orch(mode):
    return _baseline_orch(_make_mock_entry(options={
        "base_url": "http://x/v1", "model": "m", "continue_conversation": mode,
    }))


async def test_auto_adds_rule_to_voice_prompt() -> None:
    prompt = await _orch("auto")._build_system_prompt(voice_mode=True)
    assert prompt == f"{SYSTEM_PROMPT_VOICE}\n\n{FOLLOW_UP_AUTO_RULE}"


async def test_auto_rule_not_in_text_prompt() -> None:
    prompt = await _orch("auto")._build_system_prompt(voice_mode=False)
    assert prompt == SYSTEM_PROMPT_DEFAULT


@pytest.mark.parametrize("mode", ["always", "never", True, False])
async def test_other_modes_leave_voice_prompt_untouched(mode) -> None:
    prompt = await _orch(mode)._build_system_prompt(voice_mode=True)
    assert prompt == SYSTEM_PROMPT_VOICE
