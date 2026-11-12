"""Conversation diary: notes per person, recall, sharing, expiry, the writer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from .test_conversation import _baseline_orch  # noqa: F401  # isort: skip (stubs HA first)

from custom_components.ai_plugin.diary import (
    BLOCK_MARKER,
    DiaryManager,
    format_block,
)
from custom_components.ai_plugin.diary import writer
from custom_components.ai_plugin.orchestrator import core

NOW = datetime(2026, 11, 10, 18, 44, tzinfo=timezone.utc)
STIR_FRY = "Anna asked for a dinner idea; suggested a spicy chicken stir-fry: slice the chicken, fry ginger and garlic in a wok."
LENTILS = "Ben wollte ein Rezept für Linsensuppe: rote Linsen, Karotten, 25 Minuten köcheln."
BOOTS = "Someone asked where to buy hiking boots; suggested the outdoor shop on Hauptstrasse."


class _Store:
    def __init__(self):
        self.saved = None

    async def async_save(self, data):
        self.saved = data

    async def async_remove(self):
        self.saved = "removed"


async def _diary() -> DiaryManager:
    diary = DiaryManager(MagicMock(), "e1")
    diary._store = _Store()
    await diary.async_add([STIR_FRY], owner="person.anna", owner_name="Anna", language="en", when=NOW)
    await diary.async_add([LENTILS], owner="person.ben", owner_name="Ben", language="de", when=NOW - timedelta(days=2))
    await diary.async_add([BOOTS], owner=None, owner_name=None, language="en", when=NOW - timedelta(days=40))
    return diary


async def test_notes_carry_whose_conversation_they_were() -> None:
    diary = await _diary()
    owners = [(e["owner"], e["owner_name"]) for e in diary.entries()]
    assert owners == [("person.anna", "Anna"), ("person.ben", "Ben"), (None, None)]
    assert diary._store.saved["entries"][0]["text"] == STIR_FRY


async def test_hidden_shows_own_and_unrecognised_notes_only() -> None:
    diary = await _diary()
    texts = lambda viewer, share: {e["text"] for e in diary.visible(viewer, share)}  # noqa: E731
    assert texts("person.anna", False) == {STIR_FRY, BOOTS}
    assert texts("person.ben", False) == {LENTILS, BOOTS}
    assert texts(None, False) == {BOOTS}
    assert texts("person.anna", True) == {STIR_FRY, LENTILS, BOOTS}
    assert texts(None, True) == {STIR_FRY, LENTILS, BOOTS}


async def test_hidden_recall_never_returns_another_persons_note() -> None:
    diary = await _diary()
    assert [e["text"] for e in diary.recall("Wie lange kocht die Linsensuppe?", "person.anna", False)] == []
    assert [e["text"] for e in diary.recall("Wie lange kocht die Linsensuppe?", "person.ben", False)] == [LENTILS]
    assert LENTILS in [e["text"] for e in diary.recall("Wie lange kocht die Linsensuppe?", "person.anna", True)]


async def test_without_the_model_notes_are_found_by_their_words() -> None:
    diary = await _diary()
    assert diary._encoder is None
    assert [e["text"] for e in diary.recall("How do I start the stir-fry with ginger?", "person.anna", False)] == [STIR_FRY]
    assert diary.recall("Turn on the kitchen lights", "person.anna", False) == []


async def test_with_the_model_notes_are_ranked_by_meaning() -> None:
    diary = await _diary()
    vectors = {
        STIR_FRY: np.array([1.0, 0.0]), LENTILS: np.array([0.0, 1.0]), BOOTS: np.array([0.6, 0.8]),
    }
    diary._encoder = MagicMock()
    diary._encoder.embed = lambda text: vectors.get(text, np.array([0.8, 0.6]))
    diary._vectors = {e["id"]: vectors[e["text"]] for e in diary.entries()}
    assert [e["text"] for e in diary.recall("the dinner you suggested", "person.anna", True)] == [BOOTS, STIR_FRY, LENTILS]
    diary._encoder.embed = lambda text: None
    assert diary.recall("你好", "person.anna", True) == []



async def test_old_notes_expire_and_forgetting_takes_only_ones_own() -> None:
    diary = await _diary()
    assert await diary.async_prune(30, now=NOW) == 1
    assert BOOTS not in [e["text"] for e in diary.entries()]
    assert await diary.async_forget("person.anna") == 1
    assert [e["text"] for e in diary.entries()] == [LENTILS]
    assert await diary.async_forget("person.anna") == 0


def test_the_block_says_whose_conversation_each_note_was() -> None:
    entries = [
        {"text": LENTILS, "owner_name": "Ben", "time": (NOW - timedelta(days=2)).isoformat()},
        {"text": BOOTS, "owner_name": None, "time": (NOW - timedelta(days=5)).isoformat()},
        {"text": STIR_FRY, "owner_name": "Anna", "time": NOW.isoformat()},
    ]
    block = format_block(entries, ZoneInfo("Europe/Berlin"), "someone in the home")
    lines = block.splitlines()
    assert lines[0] == BLOCK_MARKER
    assert "never mention them" in lines[1] and "do not search the web" in lines[1]
    assert lines[2].startswith("- Conversation with Anna, Tue 10 Nov 19:44: Anna asked")
    assert lines[3].startswith("- Conversation with Ben, Sun 08 Nov 19:44: ")
    assert lines[4].startswith("- Conversation with someone in the home, Thu 05 Nov 19:44: ")
    assert format_block([], ZoneInfo("Europe/Berlin"), "x") == ""


def test_the_writer_reads_only_what_was_said() -> None:
    turns = [
        {"role": "system", "content": "[EARLIER CONVERSATIONS]\n- old note"},
        {"role": "user", "content": "What should I cook tonight?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "results"},
        {"role": "assistant", "content": "A spicy chicken stir-fry."},
    ]
    prompt = writer.build_prompt(turns, "Anna", ["old note"])
    assert prompt[1]["content"] == "Conversation:\nPerson: What should I cook tonight?\nAssistant: A spicy chicken stir-fry."
    assert "about Anna in the third person" in prompt[0]["content"]
    assert prompt[0]["content"].endswith("Earlier notes:\n- old note")
    assert '{"notes": []}' in prompt[0]["content"]
    assert writer.build_prompt([{"role": "system", "content": "x"}], None, []) is None


@pytest.mark.parametrize(("reply", "notes"), [
    ('{"notes": ["One.", "Two."]}', ["One.", "Two."]),
    ('```json\n{"notes": ["One."]}\n```', ["One."]),
    ('Here you go: {"notes": []}', []),
    ('{"notes": ["a", "b", "c", "d"]}', ["a", "b", "c"]),
    ("not json", []),
    ('{"notes": "x"}', []),
    (None, []),
])
def test_the_writers_reply_is_read_leniently(reply, notes) -> None:
    assert writer.parse_notes(reply) == notes


@pytest.mark.parametrize(("message", "lang", "expected"), [
    ("Forget our conversations.", "en", True),
    ("Please delete our chats", "en", True),
    ("Vergiss unsere Gespräche.", "de", True),
    ("Lösche meine Unterhaltungen", "de", True),
    ("Oublie nos conversations.", "fr", True),
    ("Vergeet onze gesprekken.", "nl", True),
    ("Olvida nuestras conversaciones.", "es", True),
    ("Esquece as nossas conversas.", "pt", True),
    ("Zapomnij nasze rozmowy.", "pl", True),
    ("Forget that I like tea.", "en", False),
    ("Vergiss, dass ich Tee mag.", "de", False),
    ("What did we talk about?", "en", False),
])
def test_asking_to_forget_the_diary(message, lang, expected) -> None:
    assert core.asks_to_forget_diary(message, lang) is expected


@pytest.mark.parametrize(("reply", "notes", "expected"), [
    ("You chose Arrival last weekend.", ["Recommended Arrival and Dune; he chose Arrival."], True),
    ("Here is the news from Berlin.", ["Recommended Arrival and Dune; he chose Arrival."], False),
    ("You chose Arrival.", [], False),
    ("", ["x"], False),
])
def test_a_reply_from_a_note_needs_no_web_search(reply, notes, expected) -> None:
    assert core._answers_from_notes(reply, notes) is expected


def _diary_orch(diary):
    orch = _baseline_orch()
    orch._hass = None
    orch._diary = diary
    orch._diary_share = False
    orch._diary_convs = {}
    orch._conv_locks = {}
    orch._diary_pruned = 0.0
    orch._diary_days = 30
    return orch


async def test_notes_reach_the_prompt_once_and_not_for_commands() -> None:
    diary = await _diary()
    orch = _diary_orch(diary)
    orch._context_mgr.newest_block_is = AsyncMock(return_value=False)
    block, texts = await orch._diary_recall_block("How do I start the stir-fry with ginger?", "c1", "person.anna", "sys", 0)
    assert block.startswith(BLOCK_MARKER) and texts == [STIR_FRY]
    assert await orch._diary_recall_block("Turn off the lights in the kitchen.", "c1", "person.anna", "sys", 0) == ("", [])
    assert await orch._diary_recall_block("Thank you.", "c1", "person.anna", "sys", 0) == ("", [])
    orch._context_mgr.newest_block_is = AsyncMock(return_value=True)
    assert await orch._diary_recall_block("How do I start the stir-fry with ginger?", "c1", "person.anna", "sys", 0) == ("", [STIR_FRY])
    assert orch._diary_convs["c1"]["known"] == {STIR_FRY}


async def test_a_quiet_conversation_becomes_notes_with_its_owner() -> None:
    diary = await _diary()
    orch = _diary_orch(diary)
    orch._provider = MagicMock()
    orch._provider.async_complete = AsyncMock(return_value='{"notes": ["Anna planned a trip to Lisbon in March."]}')
    orch._diary_note("c2", "user", "Let's plan Lisbon in March.", "person.anna", "en-US")
    orch._diary_note("c2", "assistant", "Stay in Alfama, flights about 180 euros.")
    await orch._async_diary_tick()
    orch._provider.async_complete.assert_not_awaited()  # not quiet yet
    orch._diary_convs["c2"]["last"] -= core.DIARY_QUIET_S + 1
    orch._hass = MagicMock()
    orch._hass.states.get = MagicMock(return_value=MagicMock(attributes={"friendly_name": "Anna"}))
    await orch._async_diary_tick()
    newest = diary.entries()[-1]
    assert (newest["owner"], newest["owner_name"], newest["text"]) == ("person.anna", "Anna", "Anna planned a trip to Lisbon in March.")
    assert "c2" not in orch._diary_convs
    prompt = orch._provider.async_complete.await_args.args[0]
    assert "Person: Let's plan Lisbon in March." in prompt[1]["content"]


async def test_a_conversation_of_two_people_belongs_to_nobody() -> None:
    diary = await _diary()
    orch = _diary_orch(diary)
    orch._provider = MagicMock()
    orch._provider.async_complete = AsyncMock(return_value='{"notes": ["A note."]}')
    orch._diary_note("c3", "user", "First question here.", "person.anna", "en")
    orch._diary_note("c3", "user", "Second question here.", "person.ben", "en")
    orch._diary_convs["c3"]["last"] = 0.0
    await orch._async_diary_tick()
    assert diary.entries()[-1]["owner"] is None


async def test_no_notes_are_written_while_a_turn_runs() -> None:
    diary = await _diary()
    orch = _diary_orch(diary)
    orch._provider = MagicMock()
    orch._provider.async_complete = AsyncMock(return_value='{"notes": ["x"]}')
    orch._diary_note("c4", "user", "Some question here.", "person.anna", "en")
    orch._diary_convs["c4"]["last"] = 0.0
    lock = orch._get_conv_lock("other")
    async with lock:
        await orch._async_diary_tick()
    orch._provider.async_complete.assert_not_awaited()


async def test_forgetting_by_voice_clears_the_speakers_notes() -> None:
    diary = await _diary()
    orch = _diary_orch(diary)
    with patch.object(core.Orchestrator, "_speaking_person", return_value="person.anna"):
        reply = await orch.async_process("Forget our conversations.", "c5", "en")
    from custom_components.ai_plugin.i18n import L

    assert reply == L.template("diary_forgotten", "en")
    assert all(e["owner"] != "person.anna" for e in diary.entries())
