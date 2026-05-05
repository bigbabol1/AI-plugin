"""Tests for the i18n loader, schema and façade."""
from __future__ import annotations

import pytest

from custom_components.ai_plugin.i18n import L, LOCALIZATIONS, SUPPORTED_LANGS


def test_loader_loads_english_baseline():
    assert "en" in LOCALIZATIONS
    en = LOCALIZATIONS["en"]
    assert en.code == "en"
    assert en.labels["temperature"] == "temperature"
    assert "{label}" in en.templates["attr_state"]
    assert en.keyword_re["narration"] is not None


def test_supported_langs_contains_en():
    assert "en" in SUPPORTED_LANGS
