"""Cleaning up what the model wrote before it is spoken or shown.

Narration ("I'm checking the temperature…"), emoji and kaomoji, markdown
a TTS engine reads out as punctuation, and the filler sentence models
like to close on. All of it is post-processing on the reply text — none
of it changes what the model did.
"""

from __future__ import annotations

import logging
import re

from ..i18n import L

_LOGGER = logging.getLogger(__name__)

def _strip_narration(text: str, lang: str = "en") -> str:
    """Remove tool-call narration from a model reply, in any supported
    language.

    Reads keyword + raw-pattern lists from L (i18n module). Sentence-
    granular, not line-granular: voice replies are single-line, so killing
    the whole line on a keyword hit destroys answers that share a sentence
    boundary with the narration ("I'm checking. It's 21 degrees." lost
    both). A sentence containing a digit is always kept — narration
    phrases co-occurring with concrete data ("I'm looking at the sensor —
    it reads 21 degrees") mean the model answered while narrating, and
    the answer outranks the style rule. When the entire reply is
    narration, the empty result triggers the 'I couldn't produce'
    fallback in async_process — see the empty-reply branch downstream.
    """
    if not text:
        return text
    cleaned = text
    keyword_re = L.keyword_re("narration", lang)
    if keyword_re is not None:
        out_lines = []
        for line in cleaned.splitlines():
            parts = _SENTENCE_SPLIT_RE.split(line)
            kept = [
                p for p in parts
                if not (keyword_re.search(p) and not re.search(r"\d", p))
            ]
            out_lines.append(" ".join(p for p in kept if p).strip())
        cleaned = "\n".join(out_lines)
    for pattern in L.pattern_list("narration_full", lang):
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

# Strip emoji and pictographs. TTS either mispronounces or reads them
# ("smiling face with sunglasses"), and chatty models like qwen3 love
# sprinkling them into greetings. Unicode ranges cover emoji blocks,
# misc symbols, dingbats, transport, and common decorative chars.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # symbols & pictographs, emoticons, transport, extended-A/B
    "\U00002600-\U000027BF"   # misc symbols, dingbats
    "\U0001F000-\U0001F2FF"   # mahjong, domino, playing cards, enclosed alphanumerics
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "]",
    flags=re.UNICODE,
)

# Kaomoji ("(╯°□°）╯︵ ┻━┻", "(・_・)") are built from box-drawing, geometric,
# and CJK punctuation glyphs that sit outside the emoji ranges above, so they
# slip through and TTS reads them as garbage. Two-stage strip: bracketed
# groups containing such glyphs go first (this also removes enclosed ° and _
# that belong to the face), then any leftover glyphs of those classes.
# The degree sign itself is NOT a trigger — "21°C" must survive.
# ಠ/ಥ are Kannada letters that live almost exclusively in kaomoji when they
# appear in the plugin's shipped languages (en/de/fr/es/pt/pl).
_KAOMOJI_TRIGGER = "─-◿︰-﹏・･ಠಥ"

_KAOMOJI_GROUP_RE = re.compile(
    rf"[(（\[][^()（）\[\]]*[{_KAOMOJI_TRIGGER}][^()（）\[\]]*[)）\]]"
)

_KAOMOJI_CHAR_RE = re.compile(r"[─-◿︰-﹏]+|[ಠಥ]\s*_\s*[ಠಥ]")

def _strip_emoji(text: str) -> str:
    """Remove emoji/pictographs/kaomoji. Safe for TTS and voice output."""
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = _KAOMOJI_GROUP_RE.sub("", cleaned)
    cleaned = _KAOMOJI_CHAR_RE.sub("", cleaned)
    # Collapse residual double-spaces introduced by stripped glyphs.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()

# Trailing "check X for details" / "you can find more at Y" filler sentences.
# Models love padding factual answers with redirects to external sites; the
# system prompt forbids it but small local models still emit them. Strip them
# out post-hoc to keep replies tight.
_FILLER_CLOSING_RE = re.compile(
    r"(?:^|(?<=[.!?]\s))"
    r"(?:"
    # English patterns
    r"check\s+(?:the\s+|local\s+|specific\s+|out\s+)?"
    r"(?:event\s+)?(?:calendars?|listings?|details?|websites?|sites?|news\s+links?|providers?)"
    r"[^\n]*"
    r"|(?:specific\s+|detailed?\s+|more\s+|further\s+|additional\s+)?"
    r"(?:details?|information|info)\s+(?:can\s+be\s+found|are\s+available|is\s+available)"
    r"[^\n]*"
    r"|(?:you\s+(?:can|may|might|could|should)(?:\s+want\s+to|\s+wish\s+to|\s+like\s+to)?\s+"
    r"(?:check|find|visit|see|refer|consult|browse|explore|consider|look))"
    r"[^\n]*"
    r"|(?:consider\s+(?:checking|visiting|consulting|looking))[^\n]*"
    r"|(?:it(?:'s|\s+is)\s+(?:advisable|recommended|best)\s+to\s+(?:check|visit|consult))[^\n]*"
    r"|for\s+(?:[\w'-]+\s+){0,8}?"
    r"(?:information|info|details?|coverage|listings?|updates?|news)"
    r"[^\n]*"
    r"|(?:visit|refer\s+to|consult|see|check\s+out)\s+(?:the\s+)?"
    r"(?:website|site|page|link|url|provided\s+links?)[^\n]*"
    # German patterns
    r"|weitere\s+informationen[^\n]*"
    r"|f[uü]r\s+(?:weitere|spezifische|detaillierte?|mehr)\s+"
    r"(?:informationen|details?|infos?)[^\n]*"
    r"|schauen\s+sie[^\n]*"
    r"|besuchen\s+sie[^\n]*"
    r"|(?:einzelheiten|details?)\s+(?:finden\s+sie|sind\s+verf[uü]gbar)[^\n]*"
    r")\s*$",
    re.IGNORECASE,
)

# Markdown / formatting tokens that read awfully via TTS or that some
# announce-type integrations parse oddly when present in a single utterance.
_MD_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")

_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")

_MD_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+", re.MULTILINE)

_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)

def _flatten_for_voice(text: str) -> str:
    """Strip markdown + collapse newlines for voice TTS.

    Multi-line replies with bullets or **bold** read poorly via TTS
    ("asterisk asterisk Title asterisk asterisk") and may interact badly
    with announce-style media routing. Voice-mode replies should be
    plain flowing prose. Idempotent.
    """
    if not text:
        return text
    cleaned = _MD_BOLD_RE.sub(r"\1", text)
    cleaned = _MD_ITALIC_RE.sub(r"\1", cleaned)
    cleaned = _MD_HEADING_RE.sub("", cleaned)
    cleaned = _MD_BULLET_RE.sub("", cleaned)
    # Collapse newlines into sentence breaks
    cleaned = re.sub(r"\n+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()

def _strip_filler_closing(text: str) -> str:
    """Remove trailing 'check X for details' filler sentences.

    Iterative: a reply may contain two filler sentences in a row.
    Stops after a fixed number of passes so a pathological input
    cannot loop.
    """
    if not text:
        return text
    cleaned = text.strip()
    for _ in range(3):
        new = _FILLER_CLOSING_RE.sub("", cleaned).strip()
        if new == cleaned:
            break
        cleaned = new
    return cleaned


# Sentence boundary for the streaming gate. The trailing sentence is always
# held back so the filler-closing strip can still act on it.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
