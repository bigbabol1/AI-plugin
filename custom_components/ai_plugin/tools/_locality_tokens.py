"""Multi-lingual locality keyword tables for web_search location-bias.

Used as a defensive fallback when the LLM omits the ``near_user`` parameter
on web_search. A short, hand-curated token list per BCP-47 language prefix
keeps the dependency surface zero and avoids false positives in unsupported
languages (the regex simply won't match → no injection → no harm).

The list is intentionally narrow: only words that strongly imply *the user's
own area* in a search query. Generic words like "weather" or "events" alone
do not qualify — the LLM must add a locality token, or set ``near_user``.

Extending: append to ``TOKENS`` with the BCP-47 prefix as key. Lower-case.
Word boundaries are added automatically. Multi-word phrases are matched
verbatim (case-insensitive).
"""

from __future__ import annotations

import re

# Keep entries short and unambiguous. Generic weather/event words live here
# only when paired with locality intent (e.g. "weather here" rather than
# just "weather"). A bare "weather" should NOT trigger the fallback — the
# LLM should set ``near_user``; otherwise we would inject a city into every
# weather query, which is wrong for "weather in Tokyo".
TOKENS: dict[str, tuple[str, ...]] = {
    "en": (
        "near me", "nearby", "near here", "around here", "around me",
        "in my area", "in my city", "in my town", "in my region",
        "close to me", "close by", "local events", "local news",
        "weather here", "weather near me", "events near me",
        "restaurants near me", "things to do nearby",
    ),
    "de": (
        "in der nähe", "in meiner nähe", "bei mir", "hier in der nähe",
        "in meiner stadt", "in meiner gegend", "in meiner umgebung",
        "vor ort", "lokale veranstaltungen", "veranstaltungen hier",
        "wetter hier", "ereignisse in der nähe",
    ),
    "es": (
        "cerca de mí", "cerca de aquí", "por aquí", "en mi zona",
        "en mi ciudad", "en mi región", "eventos locales",
        "tiempo aquí", "restaurantes cercanos",
    ),
    "fr": (
        "près de moi", "près d'ici", "à proximité", "dans ma région",
        "dans ma ville", "événements locaux", "météo ici",
        "restaurants à proximité",
    ),
    "it": (
        "vicino a me", "qui vicino", "nelle vicinanze", "nella mia zona",
        "nella mia città", "eventi locali", "tempo qui",
        "ristoranti vicini",
    ),
    "nl": (
        "bij mij in de buurt", "in de buurt", "hier in de buurt",
        "in mijn stad", "in mijn regio", "lokale evenementen",
        "weer hier",
    ),
    "pt": (
        "perto de mim", "aqui perto", "na minha cidade",
        "na minha região", "eventos locais", "tempo aqui",
    ),
    "pl": (
        "w pobliżu", "blisko mnie", "tutaj w okolicy",
        "w moim mieście", "wydarzenia lokalne", "pogoda tutaj",
    ),
    "sv": (
        "nära mig", "i närheten", "här i närheten", "lokala evenemang",
        "vädret här",
    ),
    "da": (
        "tæt på mig", "i nærheden", "her i nærheden", "lokale begivenheder",
        "vejret her",
    ),
    "no": (
        "i nærheten", "nær meg", "lokale arrangementer", "været her",
    ),
    "fi": (
        "lähelläni", "tässä lähellä", "lähelläni", "paikalliset tapahtumat",
        "sää täällä",
    ),
    "cs": (
        "blízko mě", "v okolí", "tady v okolí", "místní akce",
        "počasí tady",
    ),
    "tr": (
        "yakınımda", "buralarda", "şehrimde", "yerel etkinlikler",
        "buradaki hava",
    ),
    "ja": (
        "近くの", "ここの", "近所の", "地元の", "私の地域",
    ),
    "zh": (
        "附近的", "我附近", "我家附近", "本地的", "这里的",
    ),
}


def _compile(tokens: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a case-insensitive alternation with word boundaries.

    Word boundaries (\\b) work for ASCII. CJK tokens have no word boundary
    concept, so we drop \\b for those — substring match is fine because the
    tokens are themselves precise enough not to cause collisions.
    """
    parts: list[str] = []
    for token in tokens:
        escaped = re.escape(token)
        # Drop word boundaries when token contains non-ASCII letters that
        # the regex \\b heuristic can't anchor (CJK, etc.).
        if any(ord(c) > 127 for c in token if c.isalpha()):
            parts.append(escaped)
        else:
            parts.append(rf"\b{escaped}\b")
    return re.compile("|".join(parts), re.IGNORECASE)


_COMPILED: dict[str, re.Pattern[str]] = {
    lang: _compile(toks) for lang, toks in TOKENS.items()
}


def matches_locality(text: str, language: str | None = None) -> bool:
    """Return True if *text* contains a locality token for *language*.

    The language is mapped to its BCP-47 primary subtag (e.g. ``de-DE`` →
    ``de``). When no language is given, or the language is unsupported,
    falls back to scanning all token tables — safe because tokens are
    short and unambiguous, and a false positive merely results in a
    location injection that the caller already gates on a resolved
    location.
    """
    if not text:
        return False
    candidates: list[re.Pattern[str]]
    primary = (language or "").split("-", 1)[0].lower()
    if primary and primary in _COMPILED:
        candidates = [_COMPILED[primary]]
    else:
        candidates = list(_COMPILED.values())
    return any(p.search(text) for p in candidates)
