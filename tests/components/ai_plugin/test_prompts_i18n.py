"""Tests for multilingual prompt hint resolution."""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

from custom_components.ai_plugin.const import (
    CONF_TRIGGER_LANGUAGES,
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


class _FakeOrchestrator:
    """Minimal stand-in for Orchestrator that exposes _build_system_prompt
    behavior with a controllable options dict and language."""

    def __init__(self, options: dict, hass_lang: str | None):
        from custom_components.ai_plugin.orchestrator import Orchestrator
        self._build_system_prompt = Orchestrator._build_system_prompt.__get__(self)
        # _build_system_prompt delegates to two sibling helpers; bind the
        # real implementations so the stubbed _location / _memory below
        # are actually consulted. _build_user_facts_block short-circuits
        # on self._memory is None; _build_location_block awaits
        # self._location.async_resolve() and returns "" when it yields None.
        self._build_location_block = (
            Orchestrator._build_location_block.__get__(self)
        )
        self._build_user_facts_block = (
            Orchestrator._build_user_facts_block.__get__(self)
        )
        self._entry = SimpleNamespace(options=options)
        self._hass = SimpleNamespace(config=SimpleNamespace(language=hass_lang))
        # Stub the LocationProvider context-block builder used inside
        # _build_system_prompt; tests focus on language-hint plumbing only.

        async def _no_location():
            return None

        self._location = SimpleNamespace(async_resolve=_no_location)
        self._memory = None  # short-circuits _build_user_facts_block


async def _run_build(opts, hass_lang, voice_mode):
    fake = _FakeOrchestrator(opts, hass_lang)
    return await fake._build_system_prompt(voice_mode=voice_mode, user_id=None)


def test_build_prompt_no_langs_omits_all_hint_blocks():
    prompt = asyncio.run(_run_build({CONF_TRIGGER_LANGUAGES: []}, "en-US", False))
    assert "GERMAN TRIGGER HINTS" not in prompt
    assert "FRENCH TRIGGER HINTS" not in prompt
    assert "POLISH TRIGGER HINTS" not in prompt


def test_build_prompt_with_de_appends_german_default_block():
    prompt = asyncio.run(_run_build({CONF_TRIGGER_LANGUAGES: ["de"]}, "en-US", False))
    assert PROMPT_HINTS_I18N["de"]["default"] in prompt


def test_build_prompt_voice_mode_uses_voice_variant():
    prompt = asyncio.run(_run_build({CONF_TRIGGER_LANGUAGES: ["de"]}, "en-US", True))
    assert PROMPT_HINTS_I18N["de"]["voice"] in prompt
    assert PROMPT_HINTS_I18N["de"]["default"] not in prompt


def test_build_prompt_with_two_langs_appends_both_in_order():
    prompt = asyncio.run(
        _run_build({CONF_TRIGGER_LANGUAGES: ["pl", "fr"]}, "en-US", False)
    )
    assert PROMPT_HINTS_I18N["pl"]["default"] in prompt
    assert PROMPT_HINTS_I18N["fr"]["default"] in prompt
    # Order matches input list.
    assert prompt.index(PROMPT_HINTS_I18N["pl"]["default"]) < prompt.index(
        PROMPT_HINTS_I18N["fr"]["default"]
    )


def test_build_prompt_unknown_lang_silently_skipped():
    prompt = asyncio.run(_run_build({CONF_TRIGGER_LANGUAGES: ["xx"]}, "en-US", False))
    # No crash. No xx block (none exists). No DE block fallback.
    assert "GERMAN TRIGGER HINTS" not in prompt


def test_build_prompt_missing_option_falls_through_to_auto_detect():
    # No CONF_TRIGGER_LANGUAGES key, German HA → DE block auto-injected.
    prompt = asyncio.run(_run_build({}, "de-DE", False))
    assert PROMPT_HINTS_I18N["de"]["default"] in prompt
