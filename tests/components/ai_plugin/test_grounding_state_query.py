"""Which questions must be answered from list_entities."""

from __future__ import annotations

import pytest

from custom_components.ai_plugin.orchestrator.grounding import _is_state_set_query


@pytest.mark.parametrize(
    "text",
    [
        "Are any lights on?",
        "Which lights are on in the kitchen?",
        "which windows are open",
        "What's playing?",
        "Is anything running right now?",
        "Which lights are on upstairs?",
        "Welche Lichter sind an?",
        "Sind im Wohnzimmer Lichter an?",
        "Welche Fenster sind gerade offen?",
        "Welche Fenster sind auf im Bad?",
    ],
)
def test_state_questions_are_recognised(text: str) -> None:
    assert _is_state_set_query(text)


@pytest.mark.parametrize(
    "text",
    [
        "What's today on the news?",
        "What is on my calendar today?",
        "What's on TV tonight?",
        "Which team won on Sunday?",
        "Tell me some fun facts about Berlin.",
    ],
)
def test_other_questions_are_not(text: str) -> None:
    assert not _is_state_set_query(text)


def test_list_writes_count_as_acting_and_stay_out_of_retries() -> None:
    """A to-do add is a change to the house: a grounding retry must not
    offer it, and a turn that made one must not be retried into a second
    item."""
    from custom_components.ai_plugin.orchestrator.grounding import _any_actuator_call
    from custom_components.ai_plugin.orchestrator.toolcalls import _readonly_schemas

    schemas = [
        {"type": "function", "function": {"name": n}}
        for n in ("HassListAddItem", "get_items", "search_entities")
    ]
    assert [s["function"]["name"] for s in _readonly_schemas(schemas)] == [
        "get_items", "search_entities",
    ]
    msgs = [{"role": "assistant", "tool_calls": [
        {"function": {"name": "HassListAddItem", "arguments": "{}"}}
    ]}]
    assert _any_actuator_call(msgs) is True
