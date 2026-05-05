"""Tests for the L façade — fallback policy and format substitution."""
from __future__ import annotations

import pytest

from custom_components.ai_plugin.i18n import L, LOCALIZATIONS


def test_label_returns_lang_value():
    # en is the only lang loaded right now; assert it returns its own value.
    assert L.label("temperature", "en") == "temperature"


def test_label_unknown_lang_falls_back_to_en():
    assert L.label("temperature", "zz") == "temperature"


def test_label_unknown_key_returns_literal():
    assert L.label("nonexistent_key", "en") == "nonexistent_key"


def test_template_format_substitutes_placeholders():
    out = L.template("attr_state", "en", label="temperature", val=25.0, unit="°C")
    assert out == "The temperature is 25.0°C."


def test_template_unknown_lang_uses_en_template():
    out = L.template("sun_set_at", "zz", time="20:39")
    assert out == "Sunset is at 20:39."


def test_keyword_re_returns_compiled_pattern():
    pat = L.keyword_re("sun_set", "en")
    assert pat is not None
    assert pat.search("when is sunset today") is not None


def test_keyword_re_word_boundary_no_substring_match():
    pat = L.keyword_re("sun_set", "en")
    assert pat is not None
    # 'sunset' alone is in the keyword list, but a longer phrase that
    # contains 'sunset' as a substring of a different word should not match.
    # 'subsunsetter' is a non-word; the word-boundary should reject it.
    assert pat.search("subsunsetter") is None


def test_pattern_list_empty_for_unset_key():
    assert L.pattern_list("narration_full", "en") == []
