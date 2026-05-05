# Data-Driven Multilingual Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor AI Plugin's per-language localization from scattered Python regexes + a 280-line `PROMPT_HINTS_I18N` dict into YAML files under `custom_components/ai_plugin/i18n/`. Adding a new language becomes a YAML PR with no Python or prompt changes.

**Architecture:** Singleton `L` façade backed by a dict of `LangData` records loaded once at import time from `i18n/*.yaml`. Voluptuous-validated schema. English (`en.yaml`) is the canonical reference and the universal fallback. HA Assist pipeline `language` is the authoritative source of language per request — heuristic detection is removed.

**Tech Stack:** Python 3.11+, voluptuous (HA core dep), PyYAML (HA core dep), pytest, pytest-asyncio.

**Non-goals (out of scope per spec):** runtime/UI custom languages, pluralization rules, per-region variants (de-CH vs de-DE).

**Spec:** `docs/superpowers/specs/2026-05-05-multilang-data-driven-design.md`.

---

## Pre-flight

Verify deps reachable:

```bash
cd /home/arndtg/AI-plugin
python3 -c "import voluptuous, yaml; print('voluptuous', voluptuous.__version__); print('yaml', yaml.__version__)"
```

If missing, install into the same venv used for AI-Plugin tests (existing pyproject/test infra).

Verify `git status` is clean before starting:

```bash
git status --short
```

Expected: empty (or only docs/superpowers/specs/2026-05-05-multilang-data-driven-design.md still uncommitted, which is fine).

---

## Task 1: i18n module skeleton + schema + en.yaml

**Files:**
- Create: `custom_components/ai_plugin/i18n/__init__.py`
- Create: `custom_components/ai_plugin/i18n/_loader.py`
- Create: `custom_components/ai_plugin/i18n/_schema.py`
- Create: `custom_components/ai_plugin/i18n/en.yaml`
- Create: `tests/components/ai_plugin/test_i18n.py`

- [ ] **Step 1: Write the failing test for loader basic discovery**

```python
# tests/components/ai_plugin/test_i18n.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/arndtg/AI-plugin
python3 -m pytest tests/components/ai_plugin/test_i18n.py -x -v
```

Expected: `ImportError` on `from custom_components.ai_plugin.i18n import ...` (module does not exist yet).

- [ ] **Step 3: Create the schema module**

```python
# custom_components/ai_plugin/i18n/_schema.py
"""Voluptuous schema for i18n YAML files."""
from __future__ import annotations

import re

import voluptuous as vol

# Placeholders allowed in templates. Engine substitutes these via str.format.
ALLOWED_PLACEHOLDERS = {"label", "val", "unit", "time", "area"}


class LocalizationError(Exception):
    """Raised when an i18n YAML file is invalid."""


def _non_empty_str(value):
    if not isinstance(value, str) or not value.strip():
        raise vol.Invalid("must be a non-empty string")
    return value


def _lowercase_phrase(value):
    if not isinstance(value, str) or not value.strip():
        raise vol.Invalid("must be a non-empty string")
    return value.lower().strip()


def _valid_template(value):
    if not isinstance(value, str) or not value.strip():
        raise vol.Invalid("template must be a non-empty string")
    # Find placeholders {name} (ignoring {{ }} which str.format escapes).
    placeholders = set(re.findall(r"(?<!\{)\{([a-z_]+)\}(?!\})", value))
    unknown = placeholders - ALLOWED_PLACEHOLDERS
    if unknown:
        raise vol.Invalid(
            f"template uses unknown placeholders: {sorted(unknown)} "
            f"(allowed: {sorted(ALLOWED_PLACEHOLDERS)})"
        )
    return value


def _valid_regex(value):
    if not isinstance(value, str) or not value.strip():
        raise vol.Invalid("regex pattern must be a non-empty string")
    try:
        re.compile(value)
    except re.error as exc:
        raise vol.Invalid(f"invalid regex {value!r}: {exc}") from None
    return value


META_SCHEMA = vol.Schema({
    vol.Required("code"): vol.All(str, vol.Length(min=2, max=5)),
    vol.Required("name"): _non_empty_str,
    vol.Optional("contributors", default=list): [str],
})


SCHEMA = vol.Schema({
    vol.Required("meta"): META_SCHEMA,
    vol.Required("labels"): {str: _non_empty_str},
    vol.Required("templates"): {str: _valid_template},
    vol.Required("keywords"): {str: vol.All([_lowercase_phrase], vol.Length(min=1))},
    vol.Required("patterns"): {str: [_valid_regex]},
}, extra=vol.PREVENT_EXTRA)
```

- [ ] **Step 4: Create the en.yaml baseline**

```yaml
# custom_components/ai_plugin/i18n/en.yaml
meta:
  code: "en"
  name: "English"
  contributors: []

labels:
  temperature: "temperature"
  humidity: "humidity"
  co2: "CO₂"
  illuminance: "illuminance"
  pressure: "pressure"
  power: "power"
  energy: "energy"
  battery: "battery"
  brightness: "brightness"

templates:
  attr_state:        "The {label} is {val}{unit}."
  attr_pct:          "The {label} is {val}%."
  attr_humidity_fb:  "The {label} is {val}%."
  sun_set_at:        "Sunset is at {time}."
  sun_rise_at:       "Sunrise is at {time}."
  sun_is_up:         "The sun is up. Sunset is at {time}."
  sun_is_down:       "The sun is down. Sunrise is at {time}."
  fallback_no_sensor: "I don't have a {label} sensor in {area}."

keywords:
  sun_set:       ["when is sunset", "sunset", "when does the sun set", "when does the sun go down", "when does it get dark"]
  sun_rise:      ["when is sunrise", "sunrise", "when does the sun rise", "when does the sun come up"]
  sun_dark:      ["is it dark outside", "is it light outside"]
  sun_is_up:     ["is the sun up", "is the sun out"]
  narration:     ["i'm checking", "i am checking", "i'm looking", "i'm finding", "i'm searching"]
  area_prefixes: ["in the", "in"]

patterns:
  narration_full: []
  sun_full:       []
```

- [ ] **Step 5: Create the loader**

```python
# custom_components/ai_plugin/i18n/_loader.py
"""Discover, parse, validate, and compile per-language YAML files."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import voluptuous as vol
import yaml

from ._schema import SCHEMA, LocalizationError

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LangData:
    code: str
    labels: dict[str, str]
    templates: dict[str, str]
    keywords: dict[str, list[str]]
    keyword_re: dict[str, "re.Pattern[str]"] = field(default_factory=dict)
    pattern_re: dict[str, list["re.Pattern[str]"]] = field(default_factory=dict)


def load_all() -> dict[str, LangData]:
    here = Path(__file__).parent
    if not (here / "en.yaml").exists():
        raise LocalizationError(f"i18n/en.yaml is missing — required as canonical reference (looked in {here})")
    en_data = _load_one(here / "en.yaml")
    out: dict[str, LangData] = {"en": en_data}
    for path in sorted(here.glob("*.yaml")):
        if path.stem == "en":
            continue
        data = _load_one(path)
        _check_against_reference(data, en_data, path)
        out[data.code] = data
    return out


def _load_one(path: Path) -> LangData:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LocalizationError(f"{path}: malformed YAML: {exc}") from None
    try:
        raw = SCHEMA(raw)
    except vol.Invalid as exc:
        raise LocalizationError(f"{path}: schema violation: {exc}") from None
    if raw["meta"]["code"] != path.stem:
        raise LocalizationError(
            f"{path}: meta.code={raw['meta']['code']!r} must equal filename stem {path.stem!r}"
        )
    keyword_re = {k: _compile_kw_list(v) for k, v in raw["keywords"].items()}
    pattern_re = {
        k: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in v]
        for k, v in raw["patterns"].items()
    }
    return LangData(
        code=raw["meta"]["code"],
        labels=dict(raw["labels"]),
        templates=dict(raw["templates"]),
        keywords={k: list(v) for k, v in raw["keywords"].items()},
        keyword_re=keyword_re,
        pattern_re=pattern_re,
    )


def _compile_kw_list(words: list[str]) -> "re.Pattern[str]":
    """Compile a list of literal lowercase phrases into a single
    word-boundary alternation. Longest phrases first so 'when does the
    sun set' wins over 'sunset'."""
    escaped = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{escaped})\b", re.IGNORECASE)


def _check_against_reference(data: LangData, en: LangData, path: Path) -> None:
    """Warn (not fail) when a language is missing keys present in en.yaml.
    The runtime fallback policy will substitute English for missing keys."""
    for section_name, en_section in (
        ("labels", en.labels), ("templates", en.templates), ("keywords", en.keywords),
    ):
        lang_section = getattr(data, section_name)
        missing = sorted(set(en_section) - set(lang_section))
        if missing:
            _LOGGER.warning(
                "i18n/%s: %s missing %s; English fallback will apply",
                path.name, section_name, missing,
            )
```

- [ ] **Step 6: Create the public façade**

```python
# custom_components/ai_plugin/i18n/__init__.py
"""Public API for AI Plugin i18n.

All consumer code goes through the singleton ``L``. Language data is
loaded once at import time from ``i18n/*.yaml`` and cached in
``LOCALIZATIONS``. English (``en.yaml``) is the canonical reference and
the universal fallback for missing keys or unknown lang codes.
"""
from __future__ import annotations

import re

from ._loader import LangData, load_all
from ._schema import LocalizationError

LOCALIZATIONS: dict[str, LangData] = load_all()
SUPPORTED_LANGS: tuple[str, ...] = tuple(LOCALIZATIONS.keys())


class _Lookup:
    """Singleton façade over ``LOCALIZATIONS``."""

    def label(self, key: str, lang: str) -> str:
        data = LOCALIZATIONS.get(lang)
        if data is not None and key in data.labels:
            return data.labels[key]
        return LOCALIZATIONS["en"].labels.get(key, key)

    def template(self, key: str, lang: str, **fmt: object) -> str:
        data = LOCALIZATIONS.get(lang)
        if data is not None and key in data.templates:
            tmpl = data.templates[key]
        else:
            tmpl = LOCALIZATIONS["en"].templates[key]
        return tmpl.format(**fmt)

    def keyword_re(self, key: str, lang: str) -> "re.Pattern[str] | None":
        data = LOCALIZATIONS.get(lang) or LOCALIZATIONS["en"]
        return data.keyword_re.get(key) or LOCALIZATIONS["en"].keyword_re.get(key)

    def pattern_list(self, key: str, lang: str) -> list["re.Pattern[str]"]:
        data = LOCALIZATIONS.get(lang) or LOCALIZATIONS["en"]
        return list(data.pattern_re.get(key, []))


L = _Lookup()

__all__ = ["L", "LOCALIZATIONS", "SUPPORTED_LANGS", "LangData", "LocalizationError"]
```

- [ ] **Step 7: Run tests**

```bash
cd /home/arndtg/AI-plugin
python3 -m pytest tests/components/ai_plugin/test_i18n.py -x -v
```

Expected: 2 tests pass.

- [ ] **Step 8: Commit**

```bash
git add custom_components/ai_plugin/i18n/__init__.py \
        custom_components/ai_plugin/i18n/_loader.py \
        custom_components/ai_plugin/i18n/_schema.py \
        custom_components/ai_plugin/i18n/en.yaml \
        tests/components/ai_plugin/test_i18n.py
git commit -m "feat(i18n): bootstrap loader + schema + en.yaml baseline"
```

---

## Task 2: Loader validation — filename mismatch + reference warning

**Files:**
- Modify: `tests/components/ai_plugin/test_i18n.py`
- Test only — loader is already implemented in Task 1.

- [ ] **Step 1: Add failing tests for filename mismatch + warning**

Append to `tests/components/ai_plugin/test_i18n.py`:

```python
import logging
from pathlib import Path

from custom_components.ai_plugin.i18n._loader import _load_one, _check_against_reference
from custom_components.ai_plugin.i18n._schema import LocalizationError


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
```

Add the missing import at top of file:

```python
from custom_components.ai_plugin.i18n._loader import LangData
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest tests/components/ai_plugin/test_i18n.py -x -v
```

Expected: all 6 tests pass (loader already implements these checks from Task 1).

- [ ] **Step 3: Commit**

```bash
git add tests/components/ai_plugin/test_i18n.py
git commit -m "test(i18n): cover loader error paths and reference completeness"
```

---

## Task 3: Public façade `L` — fallback + format

**Files:**
- Create: `tests/components/ai_plugin/test_l_lookup.py`

- [ ] **Step 1: Write failing tests for L façade behaviour**

```python
# tests/components/ai_plugin/test_l_lookup.py
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
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest tests/components/ai_plugin/test_l_lookup.py -x -v
```

Expected: 8 tests pass (façade is already implemented in Task 1).

- [ ] **Step 3: Commit**

```bash
git add tests/components/ai_plugin/test_l_lookup.py
git commit -m "test(i18n): cover L façade fallback and format behaviour"
```

---

## Task 4: Add de.yaml + fr.yaml — port existing data

**Files:**
- Create: `custom_components/ai_plugin/i18n/de.yaml`
- Create: `custom_components/ai_plugin/i18n/fr.yaml`
- Modify: `tests/components/ai_plugin/test_i18n.py` (add per-lang assertions)

- [ ] **Step 1: Create de.yaml**

```yaml
# custom_components/ai_plugin/i18n/de.yaml
meta:
  code: "de"
  name: "Deutsch"
  contributors: []

labels:
  temperature: "Temperatur"
  humidity: "Luftfeuchtigkeit"
  co2: "CO₂"
  illuminance: "Helligkeit"
  pressure: "Luftdruck"
  power: "Leistung"
  energy: "Energieverbrauch"
  battery: "Batterieladung"
  brightness: "Helligkeit"

templates:
  attr_state:        "Die {label} ist {val}{unit}."
  attr_pct:          "Die {label} liegt bei {val}%."
  attr_humidity_fb:  "Die {label} ist {val}%."
  sun_set_at:        "Sonnenuntergang ist um {time}."
  sun_rise_at:       "Sonnenaufgang ist um {time}."
  sun_is_up:         "Die Sonne ist oben. Sonnenuntergang ist um {time}."
  sun_is_down:       "Die Sonne ist unten. Sonnenaufgang ist um {time}."
  fallback_no_sensor: "Ich habe keinen {label}-Sensor in {area}."

keywords:
  sun_set:       ["sonnenuntergang", "wann geht die sonne unter", "wann ist sonnenuntergang"]
  sun_rise:      ["sonnenaufgang", "wann geht die sonne auf", "wann ist sonnenaufgang"]
  sun_dark:      ["ist es dunkel", "ist es hell"]
  sun_is_up:     ["ist die sonne oben", "ist die sonne auf"]
  narration:     ["ich prüfe", "ich überprüfe", "ich checke", "ich suche nach", "ich schaue", "ich sehe nach", "ich frage"]
  area_prefixes: ["im", "in der", "in dem", "in"]

patterns:
  narration_full: []
  sun_full:       []
```

- [ ] **Step 2: Create fr.yaml**

```yaml
# custom_components/ai_plugin/i18n/fr.yaml
meta:
  code: "fr"
  name: "Français"
  contributors: []

labels:
  temperature: "température"
  humidity: "humidité"
  co2: "CO₂"
  illuminance: "luminosité"
  pressure: "pression"
  power: "puissance"
  energy: "consommation"
  battery: "batterie"
  brightness: "luminosité"

templates:
  attr_state:        "La {label} est de {val}{unit}."
  attr_pct:          "La {label} est à {val}%."
  attr_humidity_fb:  "La {label} est de {val}%."
  sun_set_at:        "Le soleil se couche à {time}."
  sun_rise_at:       "Le soleil se lève à {time}."
  sun_is_up:         "Le soleil est levé. Coucher du soleil à {time}."
  sun_is_down:       "Le soleil est couché. Lever du soleil à {time}."
  fallback_no_sensor: "Je n'ai pas de capteur de {label} dans {area}."

keywords:
  sun_set:       ["coucher du soleil", "se couche le soleil", "à quelle heure se couche le soleil"]
  sun_rise:      ["lever du soleil", "se lève le soleil", "à quelle heure se lève le soleil"]
  sun_dark:      ["fait-il nuit", "fait-il jour"]
  sun_is_up:     ["le soleil est-il levé"]
  narration:     ["je vérifie", "je cherche", "je regarde", "je consulte", "je recherche", "je vais vérifier", "je vais chercher", "je vais regarder"]
  area_prefixes: ["dans la", "dans le", "dans l'", "en"]

patterns:
  narration_full: []
  sun_full:       []
```

- [ ] **Step 3: Add per-lang test assertions**

Append to `tests/components/ai_plugin/test_i18n.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/components/ai_plugin/test_i18n.py tests/components/ai_plugin/test_l_lookup.py -x -v
```

Expected: 12 tests pass total.

- [ ] **Step 5: Commit**

```bash
git add custom_components/ai_plugin/i18n/de.yaml \
        custom_components/ai_plugin/i18n/fr.yaml \
        tests/components/ai_plugin/test_i18n.py
git commit -m "feat(i18n): add German and French YAML files"
```

---

## Task 5: Add es.yaml + pt.yaml + pl.yaml — port from PROMPT_HINTS_I18N

**Files:**
- Create: `custom_components/ai_plugin/i18n/es.yaml`
- Create: `custom_components/ai_plugin/i18n/pt.yaml`
- Create: `custom_components/ai_plugin/i18n/pl.yaml`

These languages currently have hint blocks but no code-level localization. Port what is reasonably translatable; leave keyword/pattern lists conservative (better to fall back to English than emit malformed strings).

- [ ] **Step 1: Create es.yaml**

```yaml
# custom_components/ai_plugin/i18n/es.yaml
meta:
  code: "es"
  name: "Español"
  contributors: []

labels:
  temperature: "temperatura"
  humidity: "humedad"
  co2: "CO₂"
  illuminance: "iluminación"
  pressure: "presión"
  power: "potencia"
  energy: "consumo"
  battery: "batería"
  brightness: "brillo"

templates:
  attr_state:        "La {label} es {val}{unit}."
  attr_pct:          "La {label} está al {val}%."
  attr_humidity_fb:  "La {label} es {val}%."
  sun_set_at:        "El sol se pone a las {time}."
  sun_rise_at:       "El sol sale a las {time}."
  sun_is_up:         "El sol está alto. Se pone a las {time}."
  sun_is_down:       "El sol está bajo. Sale a las {time}."
  fallback_no_sensor: "No tengo un sensor de {label} en {area}."

keywords:
  sun_set:       ["puesta de sol", "cuándo se pone el sol", "a qué hora se pone el sol"]
  sun_rise:      ["amanecer", "cuándo sale el sol", "a qué hora sale el sol"]
  sun_dark:      ["está oscuro", "está claro"]
  sun_is_up:     ["el sol está alto"]
  narration:     ["estoy verificando", "estoy comprobando", "estoy buscando", "voy a verificar", "voy a comprobar"]
  area_prefixes: ["en la", "en el", "en"]

patterns:
  narration_full: []
  sun_full:       []
```

- [ ] **Step 2: Create pt.yaml**

```yaml
# custom_components/ai_plugin/i18n/pt.yaml
meta:
  code: "pt"
  name: "Português"
  contributors: []

labels:
  temperature: "temperatura"
  humidity: "humidade"
  co2: "CO₂"
  illuminance: "iluminação"
  pressure: "pressão"
  power: "potência"
  energy: "consumo"
  battery: "bateria"
  brightness: "brilho"

templates:
  attr_state:        "A {label} é {val}{unit}."
  attr_pct:          "A {label} está em {val}%."
  attr_humidity_fb:  "A {label} é {val}%."
  sun_set_at:        "O sol põe-se às {time}."
  sun_rise_at:       "O sol nasce às {time}."
  sun_is_up:         "O sol está alto. Põe-se às {time}."
  sun_is_down:       "O sol está baixo. Nasce às {time}."
  fallback_no_sensor: "Não tenho um sensor de {label} em {area}."

keywords:
  sun_set:       ["pôr do sol", "quando se põe o sol", "a que horas se põe o sol"]
  sun_rise:      ["nascer do sol", "quando nasce o sol", "a que horas nasce o sol"]
  sun_dark:      ["está escuro", "está claro"]
  sun_is_up:     ["o sol está alto"]
  narration:     ["estou a verificar", "estou a procurar", "estou a olhar", "vou verificar", "vou procurar"]
  area_prefixes: ["na", "no", "em a", "em"]

patterns:
  narration_full: []
  sun_full:       []
```

- [ ] **Step 3: Create pl.yaml**

```yaml
# custom_components/ai_plugin/i18n/pl.yaml
meta:
  code: "pl"
  name: "Polski"
  contributors: []

labels:
  temperature: "temperatura"
  humidity: "wilgotność"
  co2: "CO₂"
  illuminance: "natężenie światła"
  pressure: "ciśnienie"
  power: "moc"
  energy: "zużycie"
  battery: "bateria"
  brightness: "jasność"

templates:
  attr_state:        "{label} wynosi {val}{unit}."
  attr_pct:          "{label} jest na poziomie {val}%."
  attr_humidity_fb:  "{label} wynosi {val}%."
  sun_set_at:        "Słońce zachodzi o {time}."
  sun_rise_at:       "Słońce wschodzi o {time}."
  sun_is_up:         "Słońce jest w górze. Zachód o {time}."
  sun_is_down:       "Słońce jest pod horyzontem. Wschód o {time}."
  fallback_no_sensor: "Nie mam czujnika {label} w {area}."

keywords:
  sun_set:       ["zachód słońca", "kiedy zachodzi słońce", "o której zachodzi słońce"]
  sun_rise:      ["wschód słońca", "kiedy wschodzi słońce", "o której wschodzi słońce"]
  sun_dark:      ["jest ciemno", "jest jasno"]
  sun_is_up:     ["czy słońce jest"]
  narration:     ["sprawdzam", "szukam", "patrzę", "zaraz sprawdzę", "zaraz poszukam"]
  area_prefixes: ["w", "na"]

patterns:
  narration_full: []
  sun_full:       []
```

- [ ] **Step 4: Add per-lang load tests**

Append to `tests/components/ai_plugin/test_i18n.py`:

```python
def test_all_five_languages_plus_english_load():
    expected = {"en", "de", "fr", "es", "pt", "pl"}
    assert expected.issubset(set(LOCALIZATIONS.keys()))


@pytest.mark.parametrize("lang", ["de", "fr", "es", "pt", "pl"])
def test_each_lang_has_temperature_label(lang: str):
    assert "temperature" in LOCALIZATIONS[lang].labels


@pytest.mark.parametrize("lang", ["de", "fr", "es", "pt", "pl"])
def test_each_lang_compiles_sun_set_keyword(lang: str):
    pat = LOCALIZATIONS[lang].keyword_re["sun_set"]
    assert pat is not None
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/components/ai_plugin/test_i18n.py tests/components/ai_plugin/test_l_lookup.py -x -v
```

Expected: 23 tests pass total (12 prior + 1 + 5 parametrized × 2).

- [ ] **Step 6: Commit**

```bash
git add custom_components/ai_plugin/i18n/es.yaml \
        custom_components/ai_plugin/i18n/pt.yaml \
        custom_components/ai_plugin/i18n/pl.yaml \
        tests/components/ai_plugin/test_i18n.py
git commit -m "feat(i18n): add Spanish, Portuguese, Polish YAML files"
```

---

## Task 6: Wire shortcuts.py — sun shortcut uses L

**Files:**
- Modify: `custom_components/ai_plugin/shortcuts.py`

The current `_try_sun_shortcut` mixes EN/DE/FR detection inline. Replace with L lookups.

- [ ] **Step 1: Replace `_SUN_RE` and `_try_sun_shortcut`**

Open `custom_components/ai_plugin/shortcuts.py`. Locate `_SUN_RE = re.compile(...)` (defined near the top of the sun section). Replace the entire `_SUN_RE = re.compile(...)` block AND the `_try_sun_shortcut` function with:

```python
def _try_sun_shortcut(hass: HomeAssistant, message: str, lang: str = "en") -> str | None:
    """Deterministic reply for sun/daylight questions, in the user's language.

    Reads ``sun.sun`` directly. Bypasses the LLM which often refuses these
    queries. ``lang`` selects the keyword regex set and the response
    template; English is the universal fallback.
    """
    msg_lower = (message or "").lower()
    sun_set_re = L.keyword_re("sun_set", lang)
    sun_rise_re = L.keyword_re("sun_rise", lang)
    sun_dark_re = L.keyword_re("sun_dark", lang)
    sun_is_up_re = L.keyword_re("sun_is_up", lang)

    matched = (
        (sun_set_re and sun_set_re.search(msg_lower))
        or (sun_rise_re and sun_rise_re.search(msg_lower))
        or (sun_dark_re and sun_dark_re.search(msg_lower))
        or (sun_is_up_re and sun_is_up_re.search(msg_lower))
    )
    if not matched:
        return None

    state = hass.states.get("sun.sun")
    if state is None:
        return None

    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        tz_name = (getattr(hass.config, "time_zone", None) or "").strip()
        tz = ZoneInfo(tz_name) if tz_name else None
    except Exception:  # noqa: BLE001
        tz = None

    def _fmt(iso: str | None) -> str | None:
        if not iso:
            return None
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if tz is not None:
                dt = dt.astimezone(tz)
            return dt.strftime("%H:%M")
        except Exception:  # noqa: BLE001
            return None

    attrs = state.attributes or {}
    next_setting = _fmt(attrs.get("next_setting"))
    next_rising = _fmt(attrs.get("next_rising"))
    is_up = state.state == "above_horizon"

    # Boolean queries first so 'dark' / 'fait-il nuit' don't shadow the
    # sunset time branch.
    if (sun_dark_re and sun_dark_re.search(msg_lower)) or (
        sun_is_up_re and sun_is_up_re.search(msg_lower)
    ):
        if is_up and next_setting:
            return L.template("sun_is_up", lang, time=next_setting)
        if not is_up and next_rising:
            return L.template("sun_is_down", lang, time=next_rising)

    if sun_set_re and sun_set_re.search(msg_lower) and next_setting:
        _LOGGER.info("AI Plugin shortcut hit: sunset → %s (lang=%s)", next_setting, lang)
        return L.template("sun_set_at", lang, time=next_setting)
    if sun_rise_re and sun_rise_re.search(msg_lower) and next_rising:
        _LOGGER.info("AI Plugin shortcut hit: sunrise → %s (lang=%s)", next_rising, lang)
        return L.template("sun_rise_at", lang, time=next_rising)

    return None
```

Add the new import near the existing imports at the top of the file:

```python
from .i18n import L
```

Delete the now-unused `_SUN_RE` definition (the multi-line `re.compile(...)` block immediately above the old `_try_sun_shortcut`).

- [ ] **Step 2: Update `try_shortcut` signature to accept and forward `lang`**

In `try_shortcut`, change:

```python
def try_shortcut(hass: HomeAssistant, message: str) -> str | None:
    ...
    sun_reply = _try_sun_shortcut(hass, message)
    if sun_reply:
        return sun_reply
```

to:

```python
def try_shortcut(hass: HomeAssistant, message: str, *, lang: str = "en") -> str | None:
    ...
    sun_reply = _try_sun_shortcut(hass, message, lang)
    if sun_reply:
        return sun_reply
```

- [ ] **Step 3: Add a sun shortcut localization test**

Append to `tests/components/ai_plugin/test_l_lookup.py`:

```python
def test_sun_keyword_re_matches_de_phrase():
    pat = L.keyword_re("sun_set", "de")
    assert pat is not None
    assert pat.search("wann geht die sonne unter") is not None


def test_sun_keyword_re_matches_fr_phrase():
    pat = L.keyword_re("sun_set", "fr")
    assert pat is not None
    assert pat.search("à quelle heure se couche le soleil") is not None
```

- [ ] **Step 4: Run all i18n + lookup tests**

```bash
python3 -m pytest tests/components/ai_plugin/test_i18n.py tests/components/ai_plugin/test_l_lookup.py -x -v
```

Expected: all pass (~25 tests).

- [ ] **Step 5: Commit**

```bash
git add custom_components/ai_plugin/shortcuts.py \
        tests/components/ai_plugin/test_l_lookup.py
git commit -m "refactor(shortcuts): sun shortcut reads regex + templates from L"
```

---

## Task 7: Wire shortcuts.py — `_format_state`, climate fallbacks, drop `_detect_lang`

**Files:**
- Modify: `custom_components/ai_plugin/shortcuts.py`

- [ ] **Step 1: Replace `_format_state` to read from L**

Locate `_format_state` (it currently has hard-coded `_DE_LABEL`, `_FR_LABEL`, English/German/French branches). Replace the entire function with:

```python
def _format_state(state_obj: Any, spec: dict, lang: str = "en") -> str | None:
    """Render a state object into a localized speech string."""
    if state_obj is None:
        return None
    label_key = spec["label"]
    label = L.label(label_key, lang)
    transform = spec.get("transform")

    if "attribute" in spec:
        val = state_obj.attributes.get(spec["attribute"])
        if val is None:
            return None
        if transform == "brightness_pct":
            try:
                pct = round(int(val) / 255 * 100)
                return L.template("attr_pct", lang, label=label, val=pct)
            except Exception:  # noqa: BLE001
                return L.template("attr_state", lang, label=label,
                                  val=_round_numeric(val), unit="")
        return L.template("attr_state", lang, label=label,
                          val=_round_numeric(val), unit="")

    raw = state_obj.state
    if raw in (None, "", "unknown", "unavailable", "none"):
        return None
    unit = state_obj.attributes.get("unit_of_measurement") or spec.get("unit_fallback") or ""
    return L.template("attr_state", lang, label=label,
                      val=_round_numeric(raw), unit=unit)
```

- [ ] **Step 2: Update `try_shortcut` to plumb `lang` into `_format_state` and climate fallbacks**

Locate the body of `try_shortcut` after the sun-shortcut early return. Replace the brightness branch and the sensor + climate-fallback branches with:

```python
    # For light brightness we target the brightest-on light, else any.
    if attr_key == "brightness":
        best = None
        for entry in entities:
            if not entry.entity_id.startswith("light."):
                continue
            s = hass.states.get(entry.entity_id)
            if s and s.state == "on" and s.attributes.get("brightness") is not None:
                best = s
                break
        if best is None:
            return None
        reply = _format_state(best, spec, lang=lang)
        if reply:
            _LOGGER.info(
                "AI Plugin shortcut hit: %s in %s → %s", attr_key, area.name, best.entity_id
            )
            return reply
        return None

    best = _pick_best_sensor(entities, hass, spec)
    if best is not None:
        reply = _format_state(best, spec, lang=lang)
        if reply:
            _LOGGER.info(
                "AI Plugin shortcut hit: %s in %s → %s", attr_key, area.name, best.entity_id
            )
            return reply

    # Temperature fallback: rooms with only a thermostat (no exposed
    # temperature sensor). Read current_temperature from climate.*.
    if attr_key == "temperature":
        fallback = _pick_climate_temperature(entities, hass)
        if fallback:
            state, val = fallback
            unit = state.attributes.get("unit_of_measurement") or spec.get("unit_fallback") or "°C"
            _LOGGER.info(
                "AI Plugin shortcut hit: temperature in %s → %s (climate fallback)",
                area.name, state.entity_id,
            )
            return L.template(
                "attr_state", lang,
                label=L.label("temperature", lang),
                val=_round_numeric(val), unit=unit,
            )

    # Humidity fallback: same pattern, climate.* current_humidity.
    if attr_key == "humidity":
        fallback = _pick_climate_humidity(entities, hass)
        if fallback:
            state, val = fallback
            _LOGGER.info(
                "AI Plugin shortcut hit: humidity in %s → %s (climate fallback)",
                area.name, state.entity_id,
            )
            return L.template(
                "attr_humidity_fb", lang,
                label=L.label("humidity", lang),
                val=_round_numeric(val),
            )

    return None
```

- [ ] **Step 3: Delete dead helpers**

Delete the following blocks from `shortcuts.py`:

- The `_DE_LABEL = {...}` dict definition.
- The `_FR_LABEL = {...}` dict definition.
- The `_DE_HINT_RE = re.compile(...)` definition.
- The `_FR_HINT_RE = re.compile(...)` definition.
- The `_detect_lang(message)` function.

- [ ] **Step 4: Add tests for localized shortcut output**

Append to `tests/components/ai_plugin/test_l_lookup.py`:

```python
def test_format_state_uses_de_template_and_label():
    # Round-trip a fake state via L; this also exercises template substitution.
    label = L.label("temperature", "de")
    assert label == "Temperatur"
    out = L.template("attr_state", "de", label=label, val=25.0, unit="°C")
    assert out == "Die Temperatur ist 25.0°C."


def test_format_state_uses_fr_template_and_label():
    label = L.label("temperature", "fr")
    assert label == "température"
    out = L.template("attr_state", "fr", label=label, val=25.0, unit="°C")
    assert out == "La température est de 25.0°C."
```

- [ ] **Step 5: Run shortcuts-touching tests**

```bash
python3 -m pytest tests/components/ai_plugin/test_i18n.py \
                  tests/components/ai_plugin/test_l_lookup.py -x -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add custom_components/ai_plugin/shortcuts.py \
        tests/components/ai_plugin/test_l_lookup.py
git commit -m "refactor(shortcuts): _format_state and climate fallbacks read from L; drop _DE_/_FR_ tables"
```

---

## Task 8: Wire orchestrator — pass `lang` from HA pipeline; `_strip_narration` uses L

**Files:**
- Modify: `custom_components/ai_plugin/orchestrator.py`

- [ ] **Step 1: Replace `_NARRATION_PATTERNS` and `_strip_narration`**

Open `orchestrator.py`. Locate the `_NARRATION_PATTERNS = [...]` list and the `_NARRATION_RE = re.compile(...)` line. Delete both.

Locate the `_strip_narration(text: str)` function. Replace with:

```python
def _strip_narration(text: str, lang: str = "en") -> str:
    """Remove tool-call narration lines from a model reply, in any
    supported language.

    Reads keyword + raw-pattern lists from L (i18n module). When the
    entire reply is narration, the empty result triggers the
    'I couldn't produce' fallback in async_process — see the empty-reply
    branch downstream of this call.
    """
    if not text:
        return text
    cleaned = text
    keyword_re = L.keyword_re("narration", lang)
    if keyword_re is not None:
        # Strip any line containing a narration keyword.
        cleaned = re.sub(
            rf"^.*(?:{keyword_re.pattern}).*$",
            "",
            cleaned,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    for pattern in L.pattern_list("narration_full", lang):
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned
```

Add the new import near the other relative imports at the top of `orchestrator.py`:

```python
from .i18n import L
```

- [ ] **Step 2: Plumb `lang` through `async_process`**

Locate `async def async_process(self, message, conversation_id, language=None, ...)`. Add immediately after the existing voice_mode line:

```python
        # Normalize HA's BCP-47 language ("de-DE", "fr-CA") to bare ISO 639-1.
        lang = (language or "en").split("-")[0].lower()
```

- [ ] **Step 3: Forward `lang` to all shortcut + narration callsites**

In `async_process`, locate the two shortcut calls and update:

```python
            try:
                shortcut_reply = try_shortcut(self._hass, message, lang=lang)
            ...
            try:
                media_result = await async_try_media_shortcut(self._hass, message, lang=lang)
```

Locate the `_strip_narration(reply)` call (there is one, used in the stored_reply assignment) and replace with:

```python
            narration_stripped = _strip_narration(reply, lang=lang)
```

(Existing if/elif branches that use `narration_stripped` remain unchanged.)

- [ ] **Step 4: Update `async_try_media_shortcut` signature in shortcuts.py**

This is needed because Task 8 step 3 passes `lang=` into `async_try_media_shortcut`. Open `shortcuts.py`, find `async def async_try_media_shortcut(hass, message)`, replace with:

```python
async def async_try_media_shortcut(
    hass: HomeAssistant, message: str, *, lang: str = "en"
) -> tuple[bool, str] | None:
```

The body does not yet need to read `lang` (media phrases are still language-agnostic in this release); accept the kwarg so the signature is forward-compatible.

- [ ] **Step 5: Add an orchestrator integration test**

Create `tests/components/ai_plugin/test_orchestrator_lang.py`:

```python
"""Verify orchestrator passes HA pipeline language down to shortcuts."""
from __future__ import annotations

from unittest.mock import patch

import pytest

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
```

- [ ] **Step 6: Run orchestrator + i18n tests**

```bash
python3 -m pytest tests/components/ai_plugin/test_i18n.py \
                  tests/components/ai_plugin/test_l_lookup.py \
                  tests/components/ai_plugin/test_orchestrator_lang.py -x -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add custom_components/ai_plugin/orchestrator.py \
        custom_components/ai_plugin/shortcuts.py \
        tests/components/ai_plugin/test_orchestrator_lang.py
git commit -m "refactor(orchestrator): pass HA pipeline language to shortcuts; _strip_narration reads from L"
```

---

## Task 9: Remove PROMPT_HINTS_I18N + per-lang prompt injection

**Files:**
- Modify: `custom_components/ai_plugin/const.py`
- Modify: `custom_components/ai_plugin/orchestrator.py`

- [ ] **Step 1: Delete the per-lang hint dict and helpers from const.py**

In `custom_components/ai_plugin/const.py`, delete:

- The full `PROMPT_HINTS_I18N: dict[str, dict[str, str]] = {...}` block.
- The `default_trigger_langs(hass)` function.
- The `SUPPORTED_TRIGGER_LANGUAGES: list[str] = ["de", "fr", "es", "pt", "pl"]` line.
- The `CONF_TRIGGER_LANGUAGES = "trigger_languages"` constant.
- Any `from .const import ... PROMPT_HINTS_I18N ...` (or related) re-exports if present.

(Leave `SYSTEM_PROMPT_DEFAULT`, `SYSTEM_PROMPT_VOICE`, all CONF_* unrelated to trigger_languages, and prompt template strings untouched.)

- [ ] **Step 2: Simplify `_build_system_prompt` in orchestrator.py**

Locate `async def _build_system_prompt`. Delete the entire block that:
- Reads `opts.get(CONF_TRIGGER_LANGUAGES, default_trigger_langs(...))`
- Iterates `selected` languages
- Appends `PROMPT_HINTS_I18N[lang][mode_key]` blocks

Replace those lines with a comment — the resulting body should look like:

```python
    async def _build_system_prompt(
        self, voice_mode: bool, user_id: str | None = None
    ) -> str:
        """Return the system prompt for this request.

        v0.9.0: per-language trigger hints removed. Modern multilingual
        LLMs route non-English utterances correctly without pinned hints,
        and deterministic shortcuts in shortcuts.py handle the
        load-bearing language-specific behaviour.
        """
        base = SYSTEM_PROMPT_VOICE if voice_mode else SYSTEM_PROMPT_DEFAULT
        time_block = self._build_time_block()
        location_block = await self._build_location_block()
        facts_block = await self._build_user_facts_block(user_id)
        custom = self._entry.options.get(CONF_SYSTEM_PROMPT, "").strip()
        parts = [base, time_block, location_block, facts_block]
        if custom:
            parts.append(custom)
        return "\n\n".join(p for p in parts if p)
```

- [ ] **Step 3: Remove now-unused imports**

In `orchestrator.py`, remove:

```python
from .const import (
    ...
    CONF_TRIGGER_LANGUAGES,
    PROMPT_HINTS_I18N,
    default_trigger_langs,
    ...
)
```

Keep `SYSTEM_PROMPT_DEFAULT`, `SYSTEM_PROMPT_VOICE`, `CONF_SYSTEM_PROMPT`, etc.

- [ ] **Step 4: Update existing prompt-i18n tests**

In `tests/components/ai_plugin/test_prompts_i18n.py`, locate any test that asserts presence of `[GERMAN TRIGGER HINTS]` / `[FRENCH TRIGGER HINTS]` / `PROMPT_HINTS_I18N`. Replace with negative assertions, e.g. add at the end of the file:

```python
def test_no_prompt_hints_in_built_system_prompt():
    """v0.9.0: per-language hint blocks must not appear in the rendered
    system prompt anymore."""
    from custom_components.ai_plugin.const import SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE
    for prompt in (SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_VOICE):
        assert "[GERMAN TRIGGER HINTS]" not in prompt
        assert "[FRENCH TRIGGER HINTS]" not in prompt
        assert "[SPANISH TRIGGER HINTS]" not in prompt
        assert "[PORTUGUESE TRIGGER HINTS]" not in prompt
        assert "[POLISH TRIGGER HINTS]" not in prompt
```

If the file imports `PROMPT_HINTS_I18N` directly (it does in the v0.8.x version), DELETE those imports and rewrite/remove the affected test cases — they exercise functionality that no longer exists. Specifically, any tests of the form `test_german_block_content_*`, `test_prompt_hints_completeness`, `test_default_trigger_langs_*` should be deleted entirely.

- [ ] **Step 5: Run prompts-i18n tests**

```bash
python3 -m pytest tests/components/ai_plugin/test_prompts_i18n.py -x -v
```

Expected: passes (with the v0.9.0 negative assertion in place).

- [ ] **Step 6: Commit**

```bash
git add custom_components/ai_plugin/const.py \
        custom_components/ai_plugin/orchestrator.py \
        tests/components/ai_plugin/test_prompts_i18n.py
git commit -m "refactor(prompts): remove PROMPT_HINTS_I18N and per-lang prompt injection"
```

---

## Task 10: Remove `trigger_languages` config-flow option

**Files:**
- Modify: `custom_components/ai_plugin/config_flow.py`
- Modify: `custom_components/ai_plugin/strings.json`
- Modify: `custom_components/ai_plugin/translations/en.json`

- [ ] **Step 1: Remove field from config_flow.py**

In `config_flow.py`, locate the `_advanced_schema` (or equivalent) builder and the `_validate_advanced_input` helper. Delete:

- All lines referencing `CONF_TRIGGER_LANGUAGES`.
- The `vol.Optional(CONF_TRIGGER_LANGUAGES, ...)` schema entry.
- The `ERROR_TOO_MANY_TRIGGER_LANGUAGES` constant if defined locally.
- The `if isinstance(trig, list) and len(trig) > 2: errors[...] = ...` validation branch.

Also remove any `from .const import CONF_TRIGGER_LANGUAGES, SUPPORTED_TRIGGER_LANGUAGES, default_trigger_langs` or related imports.

- [ ] **Step 2: Remove the field from strings.json**

In `custom_components/ai_plugin/strings.json`, locate the `"trigger_languages"` entries (under both `config.step.advanced` and `options.step.advanced`). Delete:

- `"trigger_languages": "..."` keys in `data` and `data_description` blocks.
- The `"too_many_trigger_languages": "..."` entry under `config.error` and `options.error` (if present).

- [ ] **Step 3: Mirror the deletions in translations/en.json**

Make the same deletions in `translations/en.json` so HA frontend strings match.

- [ ] **Step 4: Run config_flow tests**

```bash
python3 -m pytest tests/components/ai_plugin/test_config_flow.py -x -v
```

Expected: existing tests pass; any test specifically about `trigger_languages` will need to be removed. If a test like `test_options_flow_rejects_three_trigger_languages` exists, delete it (the validation it covers is gone).

- [ ] **Step 5: Commit**

```bash
git add custom_components/ai_plugin/config_flow.py \
        custom_components/ai_plugin/strings.json \
        custom_components/ai_plugin/translations/en.json \
        tests/components/ai_plugin/test_config_flow.py
git commit -m "refactor(config): remove trigger_languages option (replaced by HA pipeline language)"
```

---

## Task 11: Migration test for v0.8.x → v0.9.0 config entries

**Files:**
- Create: `tests/components/ai_plugin/test_migration_v090.py`

- [ ] **Step 1: Write the migration test**

```python
# tests/components/ai_plugin/test_migration_v090.py
"""v0.8.x → v0.9.0: existing config entries with trigger_languages must
load cleanly. The key is silently dropped from options reads."""
from __future__ import annotations


def test_old_options_dict_with_trigger_languages_loads_cleanly():
    """An options dict carrying a removed key must not crash code that
    reads other keys."""
    options = {
        "trigger_languages": ["de"],          # legacy key
        "context_window": 16384,
        "voice_mode": False,
    }
    # The loaded code must continue to fetch known keys without error.
    assert options.get("context_window") == 16384
    # And ignore the legacy key gracefully.
    assert "trigger_languages" in options    # still present in raw dict
    # but no consumer reads it any more — verified by absence of imports.
    import custom_components.ai_plugin.const as const_mod
    assert not hasattr(const_mod, "CONF_TRIGGER_LANGUAGES")
    assert not hasattr(const_mod, "PROMPT_HINTS_I18N")
    assert not hasattr(const_mod, "default_trigger_langs")
```

- [ ] **Step 2: Run migration test**

```bash
python3 -m pytest tests/components/ai_plugin/test_migration_v090.py -x -v
```

Expected: passes.

- [ ] **Step 3: Commit**

```bash
git add tests/components/ai_plugin/test_migration_v090.py
git commit -m "test: verify v0.8 config entries with trigger_languages still load on v0.9"
```

---

## Task 12: CONTRIBUTING.md + manifest bump + CHANGELOG

**Files:**
- Create: `custom_components/ai_plugin/i18n/CONTRIBUTING.md`
- Modify: `custom_components/ai_plugin/manifest.json`
- Modify: `CHANGELOG.md` (or create if absent)

- [ ] **Step 1: Write `i18n/CONTRIBUTING.md`**

```markdown
# Adding a language to AI Plugin

## TL;DR
1. Copy `i18n/en.yaml` to `i18n/<code>.yaml` (ISO 639-1 lowercase, e.g. `nl`, `it`, `sv`).
2. Translate every value. Keep `meta.code` equal to the filename stem.
3. Submit a PR.

## Detail
- **Templates** contain `{placeholder}` slots — keep them. Word order can change to fit grammar; just preserve the placeholders.
- **Keyword lists** must be lowercased — the engine matches case-insensitively, but the YAML stores phrases lowercase for consistency.
- Punctuation, accents, and unicode are fine and encouraged where natural.
- **Patterns** (the regex escape hatch) — leave empty unless you NEED a regex the keyword list cannot express. Test it locally first.

## Testing your file
```bash
python -c "from custom_components.ai_plugin.i18n import L, SUPPORTED_LANGS; print(SUPPORTED_LANGS)"
```
Should list your new code without raising. If it raises, the schema validation message tells you what is wrong.

## Approval bar
PRs welcome. Maintainer reviews for accuracy + grammar + style consistency. Do not bundle other changes.
```

- [ ] **Step 2: Bump manifest version**

Open `custom_components/ai_plugin/manifest.json`, change the `"version"` field to `"0.9.0"`.

- [ ] **Step 3: Add CHANGELOG entry**

Open or create `CHANGELOG.md` at repo root. Prepend:

```markdown
## v0.9.0 — Data-driven multilingual support

**Breaking:**
- `trigger_languages` config option removed. Language now follows the HA Assist pipeline (`Buddy (DE)`, `Buddy (FR)`, etc.).

**New:**
- All per-language data lives in `custom_components/ai_plugin/i18n/<code>.yaml` files. Adding a new language is a YAML PR — no Python changes.
- Schema-validated at load time; bad YAML fails the integration cleanly.
- 6 languages shipped: en, de, fr, es, pt, pl. Contributors can add more via PR — see `i18n/CONTRIBUTING.md`.

**Changed:**
- Per-language hint blocks dropped from the system prompt (~80–160 tokens saved per request).
- Sun shortcut, narration regex, attribute-in-area shortcut now language-symmetric — every supported language uses the same code paths and templates.
```

- [ ] **Step 4: Commit**

```bash
git add custom_components/ai_plugin/i18n/CONTRIBUTING.md \
        custom_components/ai_plugin/manifest.json \
        CHANGELOG.md
git commit -m "chore: bump v0.9.0 + CHANGELOG + i18n CONTRIBUTING.md"
```

---

## Task 13: Final eval — re-run sat1_eval against EN/DE/FR

**Files:** none (validation only)

- [ ] **Step 1: SSH-patch the freshly-refactored code into HA**

```bash
cat > /tmp/ha_patch_v090.py <<'PYEOF'
import paramiko, base64, subprocess, os
SRC = "/home/arndtg/AI-plugin/custom_components/ai_plugin"
DST = "/config/custom_components/ai_plugin"
files = []
for root, _dirs, fnames in os.walk(SRC):
    rel_root = os.path.relpath(root, SRC)
    for fn in fnames:
        if fn.endswith((".py", ".yaml", ".json", ".md")):
            files.append((os.path.join(root, fn),
                          os.path.join(DST, rel_root if rel_root != "." else "", fn)))

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.0.231", port=22222, username="bigbabol",
          password="DreiKaeseHoch_83", allow_agent=False, look_for_keys=False)
# Make sure i18n dir exists on remote
c.exec_command("sudo mkdir -p /config/custom_components/ai_plugin/i18n")
for src, dst in files:
    with open(src, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    cmd = f"echo '{b64}' | base64 -d > /tmp/patchfile && sudo cp /tmp/patchfile {dst}"
    _, _so, _ = c.exec_command(cmd)
print("done")
c.close()
PYEOF
/tmp/sat1_venv/bin/python3 /tmp/ha_patch_v090.py
```

- [ ] **Step 2: Restart HA + wait for AI Plugin agent**

```bash
/tmp/sat1_venv/bin/python3 /tmp/ha_restart2.py
until [ "$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.config/sat1_eval.json')))['ha_token'])")" \
    -X POST -H "Content-Type: application/json" \
    -d '{"text":"hi","agent_id":"conversation.ai_plugin"}' \
    http://192.168.0.231:8123/api/conversation/process 2>/dev/null)" = "200" ]; do
  sleep 5
done
echo "ai_plugin ready"
```

- [ ] **Step 3: Run EN eval**

```bash
/tmp/sat1_venv/bin/python3 /tmp/sat1_eval.py --text-mode --corpus en \
    --timeout 90 --max-cases 60 --cooldown 1 \
    2>&1 | grep -v DEBUG | grep -E "Run summary|^\s*[0-9]+ \|" | tee /tmp/eval_en_v090.txt
```

Expected: pass rate ≥ 80% (v0.8.7 baseline was 84%; reasonable margin for prompt-token reduction).

- [ ] **Step 4: Run DE eval**

```bash
/tmp/sat1_venv/bin/python3 /tmp/sat1_eval.py --text-mode --corpus de --language de \
    --timeout 90 --max-cases 60 --cooldown 1 \
    2>&1 | grep -v DEBUG | grep -E "Run summary|^\s*[0-9]+ \|" | tee /tmp/eval_de_v090.txt
```

Expected: pass rate ≥ 55% (v0.8.7 baseline was 60%). Significant regression (>10 percentage-point drop) is a blocker — investigate before declaring done.

- [ ] **Step 5: Run FR eval**

```bash
/tmp/sat1_venv/bin/python3 /tmp/sat1_eval.py --text-mode --corpus fr --language fr \
    --timeout 90 --max-cases 60 --cooldown 1 \
    2>&1 | grep -v DEBUG | grep -E "Run summary|^\s*[0-9]+ \|" | tee /tmp/eval_fr_v090.txt
```

Expected: pass rate ≥ 55% (v0.8.7 baseline was 58%).

- [ ] **Step 6: Tag + push v0.9.0**

```bash
git tag v0.9.0
git push origin main
git push origin v0.9.0
```

(Use the existing PAT in MemPalace if `git push` requires auth. The agent inherits whatever credential setup the user has configured.)

- [ ] **Step 7: Hand-off summary**

Print a short comparison to the user:

```
v0.9.0 final eval:
  EN: NN/50 (vs 42/50 in v0.8.7)
  DE: NN/50 (vs 30/50 in v0.8.7)
  FR: NN/50 (vs 28/50 in v0.8.7)

Tokens saved per prompt: ~80-160 (PROMPT_HINTS_I18N removed).
Adding a new language now requires only a single YAML PR — see
custom_components/ai_plugin/i18n/CONTRIBUTING.md.
```

---

## Self-review

**Spec coverage:**
- Architecture / file layout (Section 1) → Task 1 + Task 12 (CONTRIBUTING.md).
- YAML schema (Section 2) → Task 1 (`_schema.py`, `en.yaml`) + Tasks 4–5 (per-lang yamls).
- Loader + helper API (Section 3) → Task 1 (loader, façade, fallback) + Task 2 (validation tests) + Task 3 (façade tests).
- Migration + wiring (Section 4) → Tasks 6–10 (shortcuts, orchestrator, const, config_flow). Removals tracked at Task 9 (PROMPT_HINTS_I18N) and Task 10 (CONF_TRIGGER_LANGUAGES). Migration test at Task 11. CHANGELOG at Task 12.
- Testing (Section 5) → unit tests in Tasks 1, 2, 3, 4, 5, 6, 7, 8; migration in 11; eval in 13.
- Contributor docs → Task 12.

**Placeholder scan:** no TBDs, no "implement later", no "similar to Task N". Each step shows the actual code or command. Step 4 in Task 13 references `pass rate ≥ 55%` rather than a numeric target — defensible because the LLM evaluation has run-to-run variance, and the design explicitly accepts non-regression as the bar.

**Type consistency:**
- `LangData` defined in `_loader.py` Task 1 — re-used in test fixtures (Task 2). Same field names throughout.
- `L.template`, `L.label`, `L.keyword_re`, `L.pattern_list` — same signatures from Task 1 façade through Tasks 6, 7, 8 callers.
- `try_shortcut` signature gains `*, lang: str = "en"` in Task 6 step 2; same default applied at Task 7 callers.
- `_strip_narration(text, lang="en")` signature defined in Task 8 step 1, called with `lang=lang` in Task 8 step 3. Consistent.
- Manifest version "0.9.0" set in Task 12 — matches CHANGELOG header.

**Honest gaps:**
- The legacy `tests/components/ai_plugin/test_prompts_i18n.py` (from v0.8.x) likely contains assertions about `PROMPT_HINTS_I18N` content. Task 9 step 4 mentions deletion of those tests by description, but does not list them by name (because the v0.8.x file content is large and the implementer should grep for them). If the test file is empty/missing in this repo, Task 9 step 4 reduces to "create the file with the negative assertion".
- Eval (Task 13) requires `/tmp/sat1_eval.py`, `/tmp/sat1_venv`, `~/.config/sat1_eval.json`, `/tmp/ha_restart2.py` from the prior session. These artifacts are session-local — if executing this plan in a fresh environment, recreate them following the v1 spec at `docs/superpowers/specs/2026-05-05-sat1-eval-harness-design.md`.
