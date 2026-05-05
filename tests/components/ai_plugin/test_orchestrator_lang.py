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
