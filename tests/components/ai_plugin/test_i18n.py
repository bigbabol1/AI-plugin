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


def test_nl_yaml_loads():
    assert "nl" in LOCALIZATIONS
    nl = LOCALIZATIONS["nl"]
    assert nl.labels["temperature"] == "Temperatuur"
    assert "{label}" in nl.templates["attr_state"]
    assert nl.keyword_re["sun_set"].search("wanneer gaat de zon onder") is not None


def test_de_template_format():
    assert L.template("sun_set_at", "de", time="20:39") == "Sonnenuntergang ist um 20:39."


def test_fr_template_format():
    assert L.template("sun_set_at", "fr", time="20:39") == "Le soleil se couche à 20:39."


def test_all_shipped_languages_load():
    expected = {"en", "de", "fr", "es", "pt", "pl", "nl"}
    assert expected.issubset(set(LOCALIZATIONS.keys()))


@pytest.mark.parametrize("lang", ["de", "fr", "es", "pt", "pl", "nl"])
def test_each_lang_has_temperature_label(lang: str):
    assert "temperature" in LOCALIZATIONS[lang].labels


@pytest.mark.parametrize("lang", ["de", "fr", "es", "pt", "pl", "nl"])
def test_each_lang_compiles_sun_set_keyword(lang: str):
    pat = LOCALIZATIONS[lang].keyword_re["sun_set"]
    assert pat is not None


@pytest.mark.parametrize("lang", ["de", "fr", "es", "pt", "pl", "nl"])
def test_each_lang_matches_en_key_set(lang: str):
    """en.yaml is canonical in both directions.

    The loader only warns about keys a language is missing; a key a language
    defines and en.yaml does not is dead weight translators keep maintaining,
    which is how fallback_no_sensor and area_prefixes survived unused.
    """
    en, other = LOCALIZATIONS["en"], LOCALIZATIONS[lang]
    for section in ("labels", "templates", "keywords"):
        assert set(getattr(other, section)) == set(getattr(en, section)), section


# ── v0.9.26: localized failure strings ────────────────────────────────────────


def test_nl_template_format():
    assert L.template("sun_set_at", "nl", time="20:39") == "De zon gaat onder om 20:39."
    # Article-free readout: correct for de-words and het-words alike.
    assert L.template("attr_state", "nl", label="Vermogen", val="200", unit=" W") == (
        "Vermogen is 200 W."
    )


@pytest.mark.parametrize(
    ("key", "message", "name"),
    [
        ("action_on", "doe de keukenlamp aan", "keukenlamp"),
        ("action_on", "zet het licht aan", "licht"),
        ("action_on", "sfeerlamp aan", "sfeerlamp"),
        ("action_off", "doe de keukenlamp uit", "keukenlamp"),
        ("action_off", "zet de ventilator af", "ventilator"),
        ("action_open", "open de rolluiken", "rolluiken"),
        ("action_open", "doe de gordijnen open", "gordijnen"),
        ("action_close", "sluit de rolluiken", "rolluiken"),
        ("action_close", "doe de gordijnen dicht", "gordijnen"),
    ],
)
def test_nl_action_patterns_capture_the_device(key: str, message: str, name: str):
    hits = [m.group("name") for rx in L.pattern_list(key, "nl") if (m := rx.match(message))]
    assert hits and hits[0] == name


def test_error_templates_present_and_localized() -> None:
    from custom_components.ai_plugin.i18n import L

    assert L.template("err_no_answer", "en").startswith("I couldn't")
    assert "Antwort" in L.template("err_no_answer", "de")
    assert L.template("err_process", "de").startswith("Entschuldigung")
    for key in ("err_no_answer", "err_process", "note_tool_limit"):
        # Unknown language must fall back to English, never raise.
        assert L.template(key, "xx") == L.template(key, "en")
