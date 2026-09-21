"""Does this question need the web, or can the house answer it?

The distinction decides whether web_search is offered at all: a question
about a sensor, an area or the state of the home is answerable locally
and must not be sent out, while a temporal or place-bound question
("what's on at the cinema tonight") cannot be answered from HA state at
any prompt length.
"""

from __future__ import annotations

import re

from .grounding import _any_tool_call

# Online-query grounding verifier.
#
# Rather than enumerating every possible trigger phrase (a losing battle),
# detect two broad signal classes:
#
# 1. EXPLICIT TEMPORAL ANCHOR — the message pins itself to a time window
#    the model cannot know: "this weekend", "last week", "right now", etc.
#    Any relative or recent-absolute time reference is a safe trigger.
#
# 2. REAL-WORLD PLACE QUERY — "in <City>" / "in <Country>" combined with
#    a question word or event verb. The model's training snapshot of any
#    given city's current events is always stale.
#
# Both classes are still excluded when the query is clearly about an HA
# sensor (temperature, humidity, thermostat) to avoid false positives.

_TEMPORAL_ANCHOR_RE = re.compile(
    r"\b("
    r"today|tonight|yesterday|last\s+night|this\s+morning|this\s+evening|"
    r"this\s+week(?:end)?|last\s+week(?:end)?|this\s+month|last\s+month|"
    r"this\s+year|last\s+year|right\s+now|at\s+the\s+moment|currently|"
    r"just\s+(?:now|happened|announced)|recently|"
    # German
    r"heute|gestern|heute\s+(?:nacht|abend|morgen)|"
    r"diese[sn]?\s+wochenende?|letzte[sn]?\s+wochenende?|"
    r"diese\s+woche|letzte\s+woche|diesen\s+monat|letzten\s+monat|"
    r"gerade\s+jetzt|im\s+moment|zur(?:zeit|zeit)|gerade\s+eben"
    r")\b",
    re.IGNORECASE,
)

# "in Berlin", "in Germany", "in New York" — proper noun after preposition
_PLACE_QUERY_RE = re.compile(
    r"\b(?:in|at|near|around|from|about)\s+[A-ZÄÖÜ][a-zäöüß]"
    r"(?:[a-zäöüß\-]*\s+[A-ZÄÖÜ][a-zäöüß][a-zäöüß]*)?",
)

_HA_SENSOR_RE = re.compile(
    r"\b(?:temperature|humidity|sensor|thermostat|climate|heizung|"
    r"temperatur|luftfeuchtigkeit|co2|pm2|lux|pressure|druck)\b",
    re.IGNORECASE,
)

# Verbs / question words that, combined with a place, signal a real-world query
_EVENT_VERB_RE = re.compile(
    r"\b(?:happen(?:ed|ing)?|going\s+on|event|news|what|who|when|where|"
    r"passier|geschah|los\s+(?:in|war)|was\s+(?:ist|war))\b",
    re.IGNORECASE,
)

# Meta-conversation patterns: user is talking ABOUT the previous reply,
# not asking a fresh question of the world. These should never trigger a
# web search even if a temporal token like "just now" appears in them.
_META_CONVO_RE = re.compile(
    r"\b(?:"
    # English
    r"you\s+(?:said|told|mentioned|wrote|just\s+(?:said|told|wrote))|"
    r"(?:tell|say|repeat|show)\s+(?:me\s+)?(?:that|it|this)\s+again|"
    r"what\s+did\s+you\s+(?:say|tell|mean)|"
    r"repeat\s+(?:that|it|this|yourself)|"
    r"already\s+(?:said|told|mentioned)|"
    # German
    r"du\s+(?:hast|sagtest)\s+(?:gerade|eben|vorhin)|"
    r"sagst?\s+(?:das|es)\s+(?:nochmal|noch\s+einmal)|"
    r"wiederhol(?:e|st)|"
    r"hast\s+du\s+gerade"
    r")\b",
    re.IGNORECASE,
)

def _is_online_query(text: str) -> bool:
    """Return True when the message almost certainly needs live web data.

    Triggers on:
    - Any explicit temporal anchor (this weekend, yesterday, right now…)
    - A named place combined with an event/question verb (what happened in Berlin)

    Excluded when:
    - Query is clearly about a local HA sensor.
    - Query is meta-conversation about the previous reply ("you said that
      just now"). The "just now" inside _TEMPORAL_ANCHOR_RE would otherwise
      cause a useless web_search on chat-about-chat messages.
    """
    if not text:
        return False
    if _HA_SENSOR_RE.search(text):
        return False
    if _META_CONVO_RE.search(text):
        return False
    if _TEMPORAL_ANCHOR_RE.search(text):
        return True
    if _PLACE_QUERY_RE.search(text) and _EVENT_VERB_RE.search(text):
        return True
    return False

# Short follow-up pronouns/question words that reference prior context.
_FOLLOWUP_RE = re.compile(
    r"^\s*(?:"
    r"when\??|where\??|who\??|what\??|how\??|why\??|"
    r"wann\??|wo\??|wer\??|was\??|wie\??|warum\??|"
    r"(?:when|where|what|how|why|wann|wo|wer|was|wie)\s+.{1,60}"
    r"|(?:and\s+)?(?:the|that|this|it|those|these|das|die|den|dieser?)\b.{0,80}"
    r")\s*\??$",
    re.IGNORECASE,
)

def _is_online_followup(text: str, tool_msgs: list[dict]) -> bool:
    """True when the current message is a short follow-up to a prior web search.

    Catches 'when exactly did that happen?', 'and the evacuation?', 'who was
    involved?' etc. after the model already retrieved online data in this turn
    or when recent tool messages contain a web_search result.
    """
    if not text or len(text.strip()) > 120:
        return False
    if _HA_SENSOR_RE.search(text):
        return False
    if not _FOLLOWUP_RE.search(text):
        return False
    # Only treat as online follow-up when there is evidence of a prior
    # web search in the current conversation turn's tool messages.
    return _any_tool_call(tool_msgs, "web_search")
