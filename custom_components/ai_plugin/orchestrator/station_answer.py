"""A short answer to "Which station?" is the station's name.

Asked which station to play, the answer came back from speech recognition
as "Had you Ainz?" (probably "Radio Eins"), and the model answered it as a
memory question: "I don't have any information about Ainz in my records"
(2026-11-18). A short answer to that question becomes a request to play the
station, so it reaches the station search and at worst comes back as "no
station found".
"""

from __future__ import annotations

import re

from ..i18n import L

_WORD_RE = re.compile(r"[\w'’]+")
# "Radio Eins." or "RTL, please": an answer, not a new request.
_MAX_ANSWER_WORDS = 4


def as_station_request(message: str, previous_reply: str, lang: str) -> str:
    """``message`` as a request to play the station it names, or unchanged."""
    answer = (message or "").strip()
    words = _WORD_RE.findall(answer)
    if not words or len(words) > _MAX_ANSWER_WORDS:
        return message
    previous = (previous_reply or "").strip()
    if not any(rx.search(previous) for rx in L.pattern_list("asks_station", lang)):
        return message
    bare = answer.rstrip(".!?").strip()
    if any(rx.search(bare) for rx in L.pattern_list("broadcast_cancel", lang)):
        return message
    return L.template("play_station", lang, name=bare)
