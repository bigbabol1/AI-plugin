"""Tests for multilingual prompt hint resolution."""
from __future__ import annotations

from types import SimpleNamespace

from custom_components.ai_plugin.const import (
    PROMPT_HINTS_I18N,
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


def test_prompt_hints_has_all_supported_languages():
    assert set(PROMPT_HINTS_I18N) == set(SUPPORTED_TRIGGER_LANGUAGES)


def test_prompt_hints_each_lang_has_default_and_voice():
    for lang in SUPPORTED_TRIGGER_LANGUAGES:
        assert "default" in PROMPT_HINTS_I18N[lang], f"{lang} missing default"
        assert "voice" in PROMPT_HINTS_I18N[lang], f"{lang} missing voice"
        assert PROMPT_HINTS_I18N[lang]["default"].strip(), f"{lang} default empty"
        assert PROMPT_HINTS_I18N[lang]["voice"].strip(), f"{lang} voice empty"


def test_german_block_mentions_play_music_and_media_command():
    de_default = PROMPT_HINTS_I18N["de"]["default"]
    assert "play_music" in de_default
    assert "media_command" in de_default
    assert "set_area_state" in de_default
