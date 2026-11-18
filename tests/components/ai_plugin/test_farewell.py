"""Saying you are leaving gets a short goodbye, not a saved-fact reply."""

from __future__ import annotations

import pytest

from custom_components.ai_plugin.shortcuts.farewell import says_leaving, try_farewell


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("I'm leaving.", "en"),
        ("I’m leaving now!", "en"),
        ("Okay, I'm off.", "en"),
        ("I am heading out", "en"),
        ("I've got to go", "en"),
        ("Ich gehe jetzt.", "de"),
        ("So, ich bin dann mal weg.", "de"),
        ("Ich muss los", "de"),
        ("Je m'en vais.", "fr"),
        ("Me voy ya.", "es"),
        ("Vou sair agora.", "pt"),
        ("Ik ga nu weg.", "nl"),
        ("Wychodzę.", "pl"),
    ],
)
def test_leaving_gets_a_goodbye(text: str, lang: str) -> None:
    assert says_leaving(text, lang)
    assert try_farewell(text, lang)


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("I'm leaving at five, remind me to take the keys.", "en"),
        ("I'm leaving the light on.", "en"),
        ("Turn off the lights, I'm leaving.", "en"),
        ("I'm off work tomorrow.", "en"),
        ("I live in Utah.", "en"),
        ("Ich gehe morgen einkaufen.", "de"),
        ("Ich bin müde.", "de"),
        ("", "en"),
    ],
)
def test_other_messages_go_on(text: str, lang: str) -> None:
    assert not says_leaving(text, lang)
    assert try_farewell(text, lang) is None


def test_the_goodbye_is_in_the_language_asked() -> None:
    assert try_farewell("I'm leaving.", "en") == "See you later."
    assert try_farewell("Ich gehe jetzt.", "de") == "Bis später."


def test_without_a_language_any_language_counts() -> None:
    assert says_leaving("Ich gehe jetzt.")
    assert says_leaving("I'm leaving.")


async def test_leaving_never_reaches_the_model() -> None:
    from unittest.mock import AsyncMock

    from .test_tool_switches import _off, _scheduled_orch, _text

    orch = _scheduled_orch(_off(), [])
    orch._provider.async_chat = AsyncMock(
        side_effect=[_text("I've noted that you're leaving.")]
    )
    reply = await orch.async_process("I'm leaving.", "c-leave", "en")
    assert reply == "See you later."
    orch._provider.async_chat.assert_not_awaited()


async def test_leaving_ends_the_conversation() -> None:
    from .test_conversation import _entity_for_continue, _voice_input

    entity = _entity_for_continue({"continue_conversation": "always"},
                                  reply="See you later.")
    result = await entity.async_process(_voice_input("I'm leaving."))
    assert result.continue_conversation is False


async def test_a_sentence_trigger_answers_before_the_goodbye() -> None:
    """An automation with "I'm leaving" as its trigger is the user's own
    answer: its reply wins, and neither the goodbye nor the model is used."""
    from unittest.mock import AsyncMock

    from .test_tool_switches import _off, _scheduled_orch

    orch = _scheduled_orch(_off(), [])
    orch._provider.async_chat = AsyncMock()
    trigger = AsyncMock(return_value="Lights are off. Bye.")
    reply = await orch.async_process(
        "I'm leaving.", "c-trig", "en", sentence_trigger=trigger
    )
    assert reply == "Lights are off. Bye."
    trigger.assert_awaited_once()
    orch._provider.async_chat.assert_not_awaited()


async def test_no_sentence_trigger_match_goes_on_as_before() -> None:
    from unittest.mock import AsyncMock

    from .test_tool_switches import _off, _scheduled_orch

    orch = _scheduled_orch(_off(), [])
    trigger = AsyncMock(return_value=None)
    reply = await orch.async_process(
        "I'm leaving.", "c-trig2", "en", sentence_trigger=trigger
    )
    assert reply == "See you later."
    trigger.assert_awaited_once()


async def test_the_entity_hands_home_assistants_sentence_triggers_over() -> None:
    from unittest.mock import AsyncMock, patch

    from homeassistant.components import conversation as ha_conversation

    from .test_conversation import _entity_for_continue, _voice_input

    entity = _entity_for_continue({"continue_conversation": "always"}, reply="x")
    entity.hass = object()
    handle = AsyncMock(return_value="Bye.")
    with patch.object(ha_conversation, "async_handle_sentence_triggers", handle, create=True):
        await entity.async_process(_voice_input("I'm leaving."))
        trigger = entity._orchestrator.async_process.await_args.kwargs["sentence_trigger"]
        assert await trigger() == "Bye."
    handle.assert_awaited_once()

