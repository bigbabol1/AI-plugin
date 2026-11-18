"""A short goodbye when the user says they are leaving.

Left to the model, "I'm leaving." got "I've noted that you're leaving.", and
the check for an unbacked saved claim then answered "I haven't saved that"
(2026-11-16). Leaving is no fact to keep and needs no tool.
"""

from __future__ import annotations

import re

from ..i18n import L, SUPPORTED_LANGS

_EDGE_RE = re.compile(r"^[\s,.!?…]+|[\s,.!?…]+$")
_SPACE_RE = re.compile(r"[\s,]+")


def _normalized(text: str | None) -> str:
    return _SPACE_RE.sub(" ", _EDGE_RE.sub("", (text or "").lower())).strip()


def says_leaving(text: str | None, lang: str | None = None) -> bool:
    """True when the whole message says the user is leaving, in ``lang`` or,
    with no language given, in any language we have."""
    norm = _normalized(text)
    if not norm:
        return False
    codes = [lang] if lang in SUPPORTED_LANGS else SUPPORTED_LANGS
    return any(rx.search(norm) for code in codes for rx in L.pattern_list("leaving", code))


def try_farewell(message: str, lang: str = "en") -> str | None:
    """The goodbye for a leaving message, else None."""
    if not says_leaving(message, lang):
        return None
    return L.template("farewell", lang)
