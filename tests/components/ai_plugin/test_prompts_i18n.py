"""Tests for multilingual prompt hint resolution."""
from __future__ import annotations

from types import SimpleNamespace

from custom_components.ai_plugin.const import (
    SUPPORTED_TRIGGER_LANGUAGES,
    default_trigger_langs,
)


def _hass_with_lang(lang: str | None) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(language=lang))


def test_supported_languages_is_canonical():
    assert SUPPORTED_TRIGGER_LANGUAGES == ["de", "fr", "es", "pt", "pl"]


def test_default_trigger_langs_de_de():
    assert default_trigger_langs(_hass_with_lang("de-DE")) == ["de"]


def test_default_trigger_langs_pl_pl():
    assert default_trigger_langs(_hass_with_lang("pl-PL")) == ["pl"]


def test_default_trigger_langs_lowercases_region():
    # `.lower()` should normalise mixed-case locale strings.
    assert default_trigger_langs(_hass_with_lang("DE-de")) == ["de"]


def test_default_trigger_langs_en_us_is_empty():
    assert default_trigger_langs(_hass_with_lang("en-US")) == []


def test_default_trigger_langs_unsupported_is_empty():
    assert default_trigger_langs(_hass_with_lang("zh-CN")) == []


def test_default_trigger_langs_none_is_empty():
    assert default_trigger_langs(_hass_with_lang(None)) == []


def test_default_trigger_langs_empty_string_is_empty():
    assert default_trigger_langs(_hass_with_lang("")) == []
