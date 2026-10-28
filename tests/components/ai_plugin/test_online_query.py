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
