"""Tests covering the system prompt's i18n behaviour. v0.9.0 reduced the
surface to the negative assertion below; per-language hint blocks were
moved to YAML files under custom_components/ai_plugin/i18n/."""
from __future__ import annotations

import re

from custom_components.ai_plugin.const import (
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
)


def test_default_prompt_strips_german_fragments():
    base = SYSTEM_PROMPT_DEFAULT
    fragments = (
        "spiel jazz",
        "alle Lichter aus",
        "welche Lichter sind an",
        "sind Lichter an",
        "sind Fenster offen",
        "wetter in",
        "wetter draußen",
        "wetter draussen",
        "merk dir",
        "weiter spielen",
        "ganzes haus",
    )
    for frag in fragments:
        pattern = r"\b" + re.escape(frag) + r"\b"
        assert not re.search(pattern, base, re.IGNORECASE), (
            f"German fragment {frag!r} still present in SYSTEM_PROMPT_DEFAULT"
        )


def test_voice_prompt_strips_german_fragments():
    base = SYSTEM_PROMPT_VOICE
    fragments = (
        "spiel jazz",
        "welche Lichter sind an",
        "sind Lichter an",
        "wetter in",
        "weiter spielen",
    )
    for frag in fragments:
        pattern = r"\b" + re.escape(frag) + r"\b"
        assert not re.search(pattern, base, re.IGNORECASE), (
            f"German fragment {frag!r} still present in SYSTEM_PROMPT_VOICE"
        )


def test_default_prompt_keeps_english_examples():
    # Sanity: stripping non-EN must not delete the English routing examples.
    base = SYSTEM_PROMPT_DEFAULT
    assert "set_area_state" in base
    assert "list_entities" in base
    assert "play_music" in base
    assert "media_command" in base


def test_no_prompt_hints_in_built_system_prompt():
    """v0.9.0: per-language hint blocks must not appear in the rendered
    system prompt anymore."""
    from custom_components.ai_plugin.const import SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE
    for prompt in (SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE):
        assert "[GERMAN TRIGGER HINTS]" not in prompt
        assert "[FRENCH TRIGGER HINTS]" not in prompt
        assert "[SPANISH TRIGGER HINTS]" not in prompt
        assert "[PORTUGUESE TRIGGER HINTS]" not in prompt
        assert "[POLISH TRIGGER HINTS]" not in prompt
