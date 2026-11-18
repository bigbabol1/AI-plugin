"""A recipient's name misheard by speech-to-text is read as the name."""

from __future__ import annotations

import pytest

from custom_components.ai_plugin.orchestrator.heard_name import with_heard_name

NAMES = ["Darren", "Lisa"]


@pytest.mark.parametrize(
    ("message", "previous", "expected"),
    [
        ("Message darn that he can't eat chili.", "", "Message Darren that he can't eat chili."),
        ("Tell daring dinner is ready.", "", "Tell Darren dinner is ready."),
        ("Send a message to leesa that I'm late.", "", "Send a message to Lisa that I'm late."),
        ("Darn.", "Who should I tell that he can't eat chili?", "Darren."),
        ("The person darn.", "Who should I send it to?", "The person Darren."),
    ],
)
def test_a_misheard_recipient_becomes_the_name(message, previous, expected) -> None:
    assert with_heard_name(message, previous, NAMES, "en") == expected


@pytest.mark.parametrize(
    ("message", "previous"),
    [
        ("Message Lisa that dinner is ready.", ""),  # already the name
        ("Tell me a joke.", ""),
        ("Tell everyone dinner is ready.", ""),
        ("They aren't home yet.", ""),
        ("Darn.", "What should I tell Lisa?"),  # not asked who
        ("Darn, you are not listening to me at all?", "Who should I tell?"),  # no short answer
        ("Darn.", ""),
    ],
)
def test_other_words_stay(message, previous) -> None:
    assert with_heard_name(message, previous, NAMES, "en") == message


def test_german() -> None:
    assert with_heard_name("Schreib daren dass das Essen fertig ist.", "", NAMES, "de") == (
        "Schreib Darren dass das Essen fertig ist."
    )
    assert with_heard_name("Daren.", "Wem soll ich das schicken?", NAMES, "de") == "Darren."


def test_two_close_names_are_left_alone() -> None:
    assert with_heard_name("Message lina that I'm late.", "", ["Lena", "Lisa"], "en") == (
        "Message lina that I'm late."
    )


async def test_the_model_reads_the_corrected_name() -> None:
    from unittest.mock import AsyncMock, patch

    import custom_components.ai_plugin.orchestrator.core as core

    from .test_tool_switches import _off, _scheduled_orch, _text

    from unittest.mock import MagicMock

    orch = _scheduled_orch(_off(), [])
    orch._hass = MagicMock()
    orch._provider.async_chat = AsyncMock(side_effect=[_text("Who should I tell?")])
    with patch.object(core, "people_with_phones", return_value={"person.darren": "Darren"}):
        await orch.async_process("Message darn that dinner is ready.", "c-heard", "en")
    sent = orch._provider.async_chat.await_args.args[0]
    users = [m["content"] for m in sent if m.get("role") == "user"]
    assert users[-1] == "Message Darren that dinner is ready."
