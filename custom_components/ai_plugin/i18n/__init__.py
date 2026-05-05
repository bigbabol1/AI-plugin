"""Public API for AI Plugin i18n.

All consumer code goes through the singleton ``L``. Language data is
loaded once at import time from ``i18n/*.yaml`` and cached in
``LOCALIZATIONS``. English (``en.yaml``) is the canonical reference and
the universal fallback for missing keys or unknown lang codes.
"""
from __future__ import annotations

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
            tmpl = LOCALIZATIONS["en"].templates.get(key)
            if tmpl is None:
                raise LocalizationError(
                    f"template {key!r} not found in any language (including en); "
                    f"add it to en.yaml as the canonical reference"
                )
        return tmpl.format(**fmt)

    def keyword_re(self, key: str, lang: str) -> "re.Pattern[str] | None":
        data = LOCALIZATIONS.get(lang) or LOCALIZATIONS["en"]
        return data.keyword_re.get(key) or LOCALIZATIONS["en"].keyword_re.get(key)

    def pattern_list(self, key: str, lang: str) -> list["re.Pattern[str]"]:
        data = LOCALIZATIONS.get(lang) or LOCALIZATIONS["en"]
        return list(data.pattern_re.get(key, []))


L = _Lookup()

__all__ = ["L", "LOCALIZATIONS", "SUPPORTED_LANGS", "LangData", "LocalizationError"]
