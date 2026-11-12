"""Turning a finished conversation into diary notes, with the same model."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

_LOGGER = logging.getLogger(__name__)

MAX_NOTES = 3
_MAX_TURN_CHARS = 1500

LANGUAGES = {
    "en": "English", "de": "German", "fr": "French", "nl": "Dutch",
    "es": "Spanish", "pt": "Portuguese", "pl": "Polish",
}

# "In the language of the conversation" was not enough: an English recipe chat
# came out as a German note, after German earlier notes (replay, 2026-11-10).
# "The details given" got "received the recipe with ginger and garlic" without
# a single step, so the prompt asks for the details themselves.
WRITER_PROMPT = (
    "You keep the diary of a home voice assistant. Read the conversation below "
    "and write the notes worth remembering later: what the person asked about or "
    "planned, and the details the assistant gave them, written out so the note "
    "alone answers a later question: the steps in order, amounts, names, places, "
    "times, prices and decisions, not just that they were given. One note per "
    "topic, at most three, each at most three sentences, in {language}, about "
    "{who} in the third person. Leave out device commands "
    "(lights, music, volume, heating, timers, alarms, reminders, messages, "
    "shopping list), greetings, thanks and small talk, and anything already in "
    "the earlier notes. Answer only with JSON of this shape: "
    '{{"notes": ["first note", "second note"]}}. When nothing is worth keeping, '
    'answer {{"notes": []}}.'
)


def conversation_text(messages: list[dict[str, Any]]) -> str:
    """The user and assistant turns as plain lines; system blocks and tool
    traffic stay out, so recalled notes are not written down again."""
    lines = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        label = "Person" if role == "user" else "Assistant"
        lines.append(f"{label}: {content[:_MAX_TURN_CHARS]}")
    return "\n".join(lines)


def build_prompt(
    messages: list[dict[str, Any]],
    who: str | None,
    known: list[str],
    language: str = "en",
) -> list[dict[str, str]] | None:
    text = conversation_text(messages)
    if not text:
        return None
    system = WRITER_PROMPT.format(
        who=who or "the person", language=LANGUAGES.get(language, "English")
    )
    if known:
        system += "\n\nEarlier notes:\n" + "\n".join(f"- {k}" for k in known)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Conversation:\n" + text},
    ]


def parse_notes(reply: str | None) -> list[str]:
    """Notes from the reply; tolerant of code fences and stray text around it."""
    if not reply:
        return []
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return []
    notes = data.get("notes") if isinstance(data, dict) else None
    if not isinstance(notes, list):
        return []
    return [str(n).strip() for n in notes if isinstance(n, str) and n.strip()][:MAX_NOTES]
