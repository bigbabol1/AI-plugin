"""Tests for the i18n loader, schema and façade."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from custom_components.ai_plugin.i18n import L, LOCALIZATIONS, SUPPORTED_LANGS
from custom_components.ai_plugin.i18n._loader import LangData, _load_one, _check_against_reference
from custom_components.ai_plugin.i18n._schema import LocalizationError


def test_loader_loads_english_baseline():
    assert "en" in LOCALIZATIONS
    en = LOCALIZATIONS["en"]
    assert en.code == "en"
    assert en.labels["temperature"] == "temperature"
    assert "{label}" in en.templates["attr_state"]
    assert en.keyword_re["narration"] is not None


def test_supported_langs_contains_en():
    assert "en" in SUPPORTED_LANGS


def test_filename_mismatch_raises(tmp_path: Path):
    bad = tmp_path / "fr.yaml"
    bad.write_text(
        "meta:\n"
        "  code: \"de\"\n"      # mismatch with filename
        "  name: \"X\"\n"
        "labels: {temperature: \"x\"}\n"
        "templates:\n"
        "  attr_state: \"x\"\n"
        "keywords:\n"
        "  narration: [\"x\"]\n"
        "patterns:\n"
        "  narration_full: []\n",
        encoding="utf-8",
    )
    with pytest.raises(LocalizationError, match="must equal filename stem"):
        _load_one(bad)


def test_invalid_yaml_raises(tmp_path: Path):
    bad = tmp_path / "de.yaml"
    bad.write_text("meta: : :\n", encoding="utf-8")
    with pytest.raises(LocalizationError, match="malformed YAML"):
        _load_one(bad)


def test_invalid_regex_raises(tmp_path: Path):
    bad = tmp_path / "de.yaml"
    bad.write_text(
        "meta:\n"
        "  code: \"de\"\n"
        "  name: \"German\"\n"
        "labels: {temperature: \"Temperatur\"}\n"
        "templates:\n"
        "  attr_state: \"x\"\n"
        "keywords:\n"
        "  narration: [\"x\"]\n"
        "patterns:\n"
        "  narration_full: [\"[unclosed\"]\n",
        encoding="utf-8",
    )
    with pytest.raises(LocalizationError, match="invalid regex"):
        _load_one(bad)


def test_reference_completeness_warns_for_missing_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    en = LangData(
        code="en",
        labels={"temperature": "temperature", "humidity": "humidity"},
        templates={"attr_state": "x"},
        keywords={"narration": ["i'm checking"]},
        keyword_re={"narration": __import__("re").compile("x")},
        pattern_re={},
    )
    incomplete = LangData(
        code="de",
        labels={"temperature": "Temperatur"},   # humidity missing
        templates={"attr_state": "x"},
        keywords={"narration": ["ich prüfe"]},
        keyword_re={"narration": __import__("re").compile("x")},
        pattern_re={},
    )
    with caplog.at_level(logging.WARNING):
        _check_against_reference(incomplete, en, tmp_path / "de.yaml")
    assert any("humidity" in rec.message for rec in caplog.records)


def test_reference_completeness_no_warning_when_complete(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    en = LangData(
        code="en", labels={"x": "x"}, templates={"x": "x"},
        keywords={"x": ["x"]}, keyword_re={}, pattern_re={},
    )
    complete = LangData(
        code="de", labels={"x": "X"}, templates={"x": "X"},
        keywords={"x": ["x"]}, keyword_re={}, pattern_re={},
    )
    with caplog.at_level(logging.WARNING):
        _check_against_reference(complete, en, tmp_path / "de.yaml")
    assert not any("missing" in rec.message for rec in caplog.records)


def test_de_yaml_loads():
    assert "de" in LOCALIZATIONS
    de = LOCALIZATIONS["de"]
    assert de.labels["temperature"] == "Temperatur"
    assert "{label}" in de.templates["attr_state"]
    assert de.keyword_re["sun_set"].search("wann geht die sonne unter") is not None


def test_fr_yaml_loads():
    assert "fr" in LOCALIZATIONS
    fr = LOCALIZATIONS["fr"]
    assert fr.labels["temperature"] == "température"
    assert fr.keyword_re["narration"].search("je vérifie la météo") is not None


def test_de_template_format():
    assert L.template("sun_set_at", "de", time="20:39") == "Sonnenuntergang ist um 20:39."


def test_fr_template_format():
    assert L.template("sun_set_at", "fr", time="20:39") == "Le soleil se couche à 20:39."
