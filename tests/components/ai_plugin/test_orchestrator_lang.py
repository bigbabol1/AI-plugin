"""Verify orchestrator passes HA pipeline language down to shortcuts."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub aiohttp before importing orchestrator so the import chain
# (orchestrator → openai_compat → aiohttp) succeeds in environments
# where aiohttp is not installed.
if "aiohttp" not in sys.modules:
    _aiohttp_stub = types.ModuleType("aiohttp")
    _aiohttp_stub.ClientSession = MagicMock  # type: ignore[attr-defined]
    _aiohttp_stub.ClientError = Exception  # type: ignore[attr-defined]
    _aiohttp_stub.ClientResponseError = Exception  # type: ignore[attr-defined]
    _aiohttp_stub.ClientConnectorError = Exception  # type: ignore[attr-defined]
    sys.modules["aiohttp"] = _aiohttp_stub

from custom_components.ai_plugin.orchestrator import _strip_narration


def test_strip_narration_de_keyword_match():
    text = "Ich prüfe die Luftfeuchtigkeit."
    assert _strip_narration(text, lang="de") == ""


def test_strip_narration_fr_keyword_match():
    text = "Je vérifie la météo."
    assert _strip_narration(text, lang="fr") == ""


def test_strip_narration_preserves_real_content():
    text = "It is 11:46 AM."
    assert _strip_narration(text, lang="en") == text


def test_strip_narration_unknown_lang_falls_back_to_en():
    text = "I'm checking the weather."
    # zz is unknown; L falls back to en, which strips this English narration.
    assert _strip_narration(text, lang="zz") == ""


def test_strip_narration_keeps_answer_sharing_a_line() -> None:
    """A narration sentence and the answer on ONE line: only the
    narration sentence dies."""
    from custom_components.ai_plugin.orchestrator import _strip_narration

    out = _strip_narration("I'm checking the temperature. It's 21 degrees.", "en")
    assert out == "It's 21 degrees."


def test_strip_narration_keeps_digit_bearing_narration_sentence() -> None:
    """Narration phrase + concrete data in the same sentence: the answer
    outranks the style rule."""
    from custom_components.ai_plugin.orchestrator import _strip_narration

    out = _strip_narration(
        "I'm looking at the sensor — it reads 21 degrees.", "en"
    )
    assert "21 degrees" in out


def test_strip_narration_pure_narration_still_blanked() -> None:
    from custom_components.ai_plugin.orchestrator import _strip_narration

    assert _strip_narration("I'm checking the temperature for you.", "en") == ""


def test_strip_narration_german_sentence_granular() -> None:
    from custom_components.ai_plugin.orchestrator import _strip_narration

    out = _strip_narration("Ich schaue nach. Im Schlafzimmer sind es 19 Grad.", "de")
    assert out == "Im Schlafzimmer sind es 19 Grad."
