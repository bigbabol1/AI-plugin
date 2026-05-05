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
