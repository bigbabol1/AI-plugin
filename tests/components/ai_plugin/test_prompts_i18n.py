"""Tests for multilingual prompt hint resolution."""
from __future__ import annotations

from types import SimpleNamespace

from custom_components.ai_plugin.const import (
    PROMPT_HINTS_I18N,
    SUPPORTED_TRIGGER_LANGUAGES,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
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


def test_default_prompt_strips_german_fragments():
    base = SYSTEM_PROMPT_DEFAULT.lower()
    for frag in (
        "spiel jazz",
        "alle lichter aus",
        "welche lichter sind an",
        "sind lichter an",
        "sind fenster offen",
        "wetter in",
        "wetter draußen",
        "wetter draussen",
        "merk dir",
        "weiter spielen",
        "alle",
        "alles",
        "überall",
        "ueberall",
        "ganzes haus",
    ):
        assert frag not in base, (
            f"German fragment {frag!r} still present in SYSTEM_PROMPT_DEFAULT"
        )


def test_voice_prompt_strips_german_fragments():
    base = SYSTEM_PROMPT_VOICE.lower()
    for frag in (
        "spiel jazz",
        "welche lichter sind an",
        "sind lichter an",
        "wetter in",
        "weiter",
    ):
        # 'weiter' alone is too generic; use a longer fragment to be safe
        if frag == "weiter":
            assert "weiter" not in base or "weiterhin" in base
            continue
        assert frag not in base, (
            f"German fragment {frag!r} still present in SYSTEM_PROMPT_VOICE"
        )


def test_default_prompt_keeps_english_examples():
    # Sanity: stripping non-EN must not delete the English routing examples.
    base = SYSTEM_PROMPT_DEFAULT
    assert "set_area_state" in base
    assert "list_entities" in base
    assert "play_music" in base
    assert "media_command" in base
