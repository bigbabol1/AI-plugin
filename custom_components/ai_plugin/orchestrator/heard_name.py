"""Household names that speech recognition turned into other words.

Cloud speech-to-text has no list of the home's names, so a name comes back
as "aren't" or "aunt". Where a message's recipient is named ("Message aren't
that he can't eat chili.") or in a short answer to "Who should I tell?", a
word that sounds like one person with a phone is taken as that name before
the model reads it (2026-11-17: the model asked who, then could not place
"Aren't." three times).
"""

from __future__ import annotations

import difflib
import re

from ..i18n import L
from ..voice_id import NAME_MATCH_LEAD, NAME_MATCH_MIN

_WORD_RE = re.compile(r"[\w'’]+")
# "Aren't." or "The person aren't.": an answer, not a new request.
_MAX_ANSWER_WORDS = 3


def _closest(word: str, names: list[str]) -> str | None:
    wanted = word.casefold().replace("’", "'")
    if any(wanted == n.casefold() for n in names):
        return None
    ranked = sorted(
        ((difflib.SequenceMatcher(None, wanted, n.casefold()).ratio(), n) for n in names),
        reverse=True,
    )
    if not ranked or ranked[0][0] < NAME_MATCH_MIN:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < NAME_MATCH_LEAD:
        return None
    return ranked[0][1]


def with_heard_name(message: str, previous_reply: str, names, lang: str) -> str:
    """``message`` with a misheard recipient replaced by the name it sounds
    like, or unchanged."""
    names = [n for n in names if n]
    if not message or not names:
        return message
    for rx in L.pattern_list("message_target", lang):
        match = rx.search(message)
        if match:
            name = _closest(match.group("name"), names)
            if name is None:
                return message
            start, end = match.span("name")
            return message[:start] + name + message[end:]
    words = _WORD_RE.findall(message)
    if not words or len(words) > _MAX_ANSWER_WORDS:
        return message
    if not any(rx.search((previous_reply or "").strip()) for rx in L.pattern_list("asks_who", lang)):
        return message
    name = _closest(words[-1], names)
    if name is None:
        return message
    start = message.rfind(words[-1])
    return message[:start] + name + message[start + len(words[-1]):]
