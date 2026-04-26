"""Tests for the multi-lingual locality token matcher."""

from __future__ import annotations

from custom_components.ai_plugin.tools._locality_tokens import (
    TOKENS,
    matches_locality,
)


def test_empty_text_no_match() -> None:
    assert matches_locality("", "en") is False
    assert matches_locality("", None) is False


def test_english_phrase_matches() -> None:
    assert matches_locality("any events near me tonight?", "en") is True
    assert matches_locality("show local events", "en") is True
    assert matches_locality("things to do nearby", "en") is True


def test_german_phrase_matches() -> None:
    assert matches_locality("veranstaltungen in meiner nähe", "de") is True
    assert matches_locality("wetter hier", "de-DE") is True


def test_japanese_phrase_matches_without_word_boundary() -> None:
    """CJK has no word boundary — substring match must still trigger."""
    assert matches_locality("近くのレストラン", "ja") is True
    assert matches_locality("地元のイベント", "ja-JP") is True


def test_chinese_phrase_matches() -> None:
    assert matches_locality("附近的餐厅", "zh") is True
    assert matches_locality("我附近的活动", "zh-CN") is True


def test_global_query_no_match() -> None:
    assert matches_locality("price of bitcoin", "en") is False
    assert matches_locality("how does photosynthesis work", "en") is False
    assert matches_locality("weather in tokyo", "en") is False


def test_unsupported_language_falls_back_to_all_tables() -> None:
    """When language is unknown, scan all tables — EN tokens should still match."""
    assert matches_locality("events near me", "xx") is True
    assert matches_locality("events near me", None) is True


def test_bcp47_primary_subtag_extracted() -> None:
    assert matches_locality("wetter hier", "de-AT") is True
    assert matches_locality("wetter hier", "DE-de") is True


def test_case_insensitive() -> None:
    assert matches_locality("EVENTS NEAR ME", "en") is True
    assert matches_locality("Wetter Hier", "de") is True


def test_all_token_tables_compile() -> None:
    """Every entry in TOKENS produces matching results — sanity for new langs."""
    for lang, tokens in TOKENS.items():
        for token in tokens:
            assert matches_locality(token, lang) is True, f"{lang}: {token}"
