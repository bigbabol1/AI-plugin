"""When does a question force a web search the model did not run?"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.ai_plugin.orchestrator.online import _is_online_query

from .test_conversation import _baseline_orch


@pytest.mark.parametrize(
    "text",
    [
        # Every forced search logged from 2026-10-22 to 10-28.
        "Man, today is cold, what do you think should I wear?",
        "What should I have for lunch today? What do you think?",
        "Do you think I should, uh, I can wear a tank top today when going out?",
        "Do you have anything scheduled for today?",
        "Habe ich heute Termine?",
        "Was meinst du, soll ich heute joggen gehen?",
    ],
)
def test_own_plans_and_advice_are_not_web_questions(text) -> None:
    assert not _is_online_query(text)


@pytest.mark.parametrize(
    "text",
    [
        "What's happening in Berlin today?",
        "Tell me the news today",
        "Who won the match yesterday?",
        "What happened in the US last night?",
        "Was ist heute in Berlin los?",
    ],
)
def test_current_events_still_are(text) -> None:
    assert _is_online_query(text)


@pytest.mark.parametrize(
    "text",
    [
        "What's the weather today?",
        "Is it raining right now?",
        "Wie ist heute das Wetter?",
        "Regnet es heute?",
    ],
)
def test_weather_with_no_place_stays_at_home(text) -> None:
    """2026-11-03: asked twice, the model answered from its earlier reading and
    the forced search replaced that with another place's weather."""
    assert not _is_online_query(text)


@pytest.mark.parametrize(
    "text",
    [
        "What's the weather in Tokyo today?",
        "Wie ist heute das Wetter in Hamburg?",
        "Is it raining in London right now?",
        "What is the market forecast today?",
    ],
)
def test_weather_elsewhere_and_other_forecasts_still_search(text) -> None:
    assert _is_online_query(text)


@pytest.mark.parametrize(
    "text",
    [
        "What day is today?",
        "What day is it today?",
        "Which day is today?",
        "What's the date today?",
        "What is today's date?",
        "What time is it right now?",
        "Welcher Tag ist heute?",
        "Was für ein Tag ist heute?",
        "Der wievielte ist heute?",
        "Welches Datum haben wir heute?",
    ],
)
def test_the_day_or_date_needs_no_search(text) -> None:
    assert not _is_online_query(text)


@pytest.mark.parametrize(
    "text",
    [
        "What day is the match today?",
        "What time is the game tonight?",
        "What time does the store close today?",
        "Welcher Tag ist heute in Berlin ein Feiertag?",
    ],
)
def test_a_question_about_something_else_today_still_searches(text) -> None:
    assert _is_online_query(text)


@pytest.mark.parametrize(
    "text",
    ["Enough for today.", "Genug für heute.", "It's cold today.", "That's it for this week."],
)
def test_a_time_word_alone_asks_for_nothing(text) -> None:
    assert not _is_online_query(text)


@pytest.mark.parametrize(
    "text",
    ["Any news today", "Top headlines today", "Sag mir die Nachrichten von heute",
     "Is the Berlin marathon today?", "Wann spielt der BVB heute?"],
)
def test_a_request_with_a_time_word_still_searches(text) -> None:
    assert _is_online_query(text)


def _orch_answering(tool_rounds):
    """Orchestrator whose model makes ``tool_rounds`` and then answers."""
    orch = _baseline_orch()
    orch._hass = MagicMock()
    orch._web_search = SimpleNamespace(async_search=AsyncMock(return_value="Brooklyn events"))
    replies = [
        SimpleNamespace(
            has_tool_calls=True,
            tool_calls=[SimpleNamespace(
                name=name, arguments={},
                to_assistant_message=lambda n=name: {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"function": {"name": n, "arguments": "{}"}}],
                },
                to_tool_result_message=lambda r: {"role": "tool", "content": r},
            )],
            reply_text=lambda: "",
        )
        for name in tool_rounds
    ]
    replies += [SimpleNamespace(has_tool_calls=False, tool_calls=[], reply_text=lambda: "Answer.")] * 3
    orch._provider = MagicMock()
    orch._provider.async_chat = AsyncMock(side_effect=replies)
    orch._dispatch_tool = AsyncMock(return_value="1. butter")
    orch._tool_schemas_for = lambda *a: [
        {"type": "function", "function": {"name": "get_items", "parameters": {}}},
        {"type": "function", "function": {"name": "web_search", "parameters": {}}},
    ]
    return orch


async def test_no_forced_search_after_the_model_read_the_house() -> None:
    """2026-10-28: the model read the to-do list, then a forced search of the
    raw question replaced its answer with events in Brooklyn."""
    orch = _orch_answering(["get_items"])
    await orch.async_process("What is going on right now?", "c1", "en")
    orch._web_search.async_search.assert_not_awaited()


async def test_forced_search_when_the_model_used_no_tool() -> None:
    orch = _orch_answering([])
    await orch.async_process("What is going on right now?", "c2", "en")
    orch._web_search.async_search.assert_awaited_once()


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("Wake me.", "en"),
        ("Wake me up.", "en"),
        ("Please set an alarm", "en"),
        ("Weck mich.", "de"),
        ("Stell einen Wecker.", "de"),
        ("Réveille-moi.", "fr"),
        ("Despiértame.", "es"),
        ("Maak me wakker.", "nl"),
        ("Obudź mnie.", "pl"),
        ("Acorda-me.", "pt"),
    ],
)
def test_an_alarm_request_with_no_time(message, lang) -> None:
    from custom_components.ai_plugin.scheduler.tools import _names_no_alarm_time

    assert _names_no_alarm_time(message, lang)


@pytest.mark.parametrize(
    ("message", "lang"),
    [
        ("Wake me up at 7.", "en"),
        ("Wake me at seven.", "en"),
        ("Wake me in 10 minutes.", "en"),
        ("Set an alarm for half past six.", "en"),
        ("7.", "en"),
        ("Seven.", "en"),
        ("Weck mich um 7.", "de"),
        ("Weck mich morgen früh.", "de"),
    ],
)
def test_an_alarm_request_that_names_a_time_passes(message, lang) -> None:
    from custom_components.ai_plugin.scheduler.tools import _names_no_alarm_time

    assert not _names_no_alarm_time(message, lang)


@pytest.mark.parametrize(
    "text",
    [
        "What were your thoughts again about Portugal?",
        "What did I tell you about Portugal?",
        "Did we talk about Japan?",
        "Do you remember what I said about Berlin?",
        "Was habe ich dir über Italien erzählt?",
        "Erinnerst du dich an Spanien?",
        "Was war deine Meinung zu Portugal?",
    ],
)
def test_a_question_about_an_earlier_conversation_forces_no_search(text: str) -> None:
    """The diary and the saved facts answer these. A forced search for "What
    were your thoughts again about Portugal?" brought back a stranger's travel
    review, read out as the user's own trip (2026-11-16)."""
    assert not _is_online_query(text)


@pytest.mark.parametrize(
    "text",
    [
        "What happened in Portugal this week?",
        "What is going on in Berlin today?",
        "Who won the election in Brazil yesterday?",
    ],
)
def test_news_about_a_place_still_forces_a_search(text: str) -> None:
    assert _is_online_query(text)


_FESTIVAL_NOTE = (
    "The user asked about events in Berlin this weekend. The assistant "
    "mentioned the International Short Film Festival and the Italian Film Festival."
)


def _orch_repeating_a_note(note: str, reply: str):
    orch = _orch_answering([])
    orch._diary_recall_block = AsyncMock(return_value=("", [note]))
    orch._provider.async_chat = AsyncMock(return_value=SimpleNamespace(
        has_tool_calls=False, tool_calls=[], reply_text=lambda: reply,
    ))
    return orch


async def test_an_old_note_does_not_answer_what_is_on_today() -> None:
    """2026-11-22: "Is there any celebrations happening in Berlin today?" was
    answered from a 13 November note on that weekend's festivals, no search."""
    orch = _orch_repeating_a_note(
        _FESTIVAL_NOTE,
        "The International Short Film Festival and the Italian Film Festival are running.",
    )
    await orch.async_process("Is there any celebrations happening in Berlin today?", "c3", "en")
    orch._web_search.async_search.assert_awaited_once()


async def test_a_note_still_answers_a_question_about_the_user() -> None:
    """2026-11-10: "Which film did I pick last weekend?" came from the note."""
    orch = _orch_repeating_a_note(
        "Recommended Arrival and Dune; he chose Arrival.", "You chose Arrival last weekend.",
    )
    await orch.async_process("Which film did I pick last weekend?", "c4", "en")
    orch._web_search.async_search.assert_not_awaited()


@pytest.mark.parametrize(("text", "expected"), [
    ("Which film did I pick last weekend?", True),
    ("What did we plan for the weekend?", True),
    ("Was habe ich letzte Woche gekocht?", True),
    ("Is there any celebrations happening in Berlin today?", False),
    ("What is the news in Germany today?", False),
    ("Was passiert heute in Berlin?", False),
])
def test_who_the_question_is_about(text, expected) -> None:
    from custom_components.ai_plugin.orchestrator.online import _speaks_of_the_user
    assert _speaks_of_the_user(text) is expected
