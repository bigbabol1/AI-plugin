"""Is this short turn a real command, or the tail of our own TTS?

Used by the echo filter as a safety net: anything the deterministic
layers recognise is never dropped as echo.
"""

from __future__ import annotations

import re

from .actions import _match_action_intent
from .media import _detect_media_command
from .sensors import _detect_attribute  # noqa: F401 — kept for callers
from .sweep import _parse_sweep_noun

# Short turns we must never swallow as echo: anything the deterministic
# layers would recognise as a command, plus bare answers to a question the
# agent just asked. Everything else in a one-to-three word turn carries no
# recoverable meaning, which is exactly what a mis-heard TTS tail looks like.
_ANSWER_WORDS_RE = re.compile(
    r"^(?:"
    r"yes|yeah|yep|yup|sure|please|ok|okay|no|nope|not\s+now|cancel|"
    r"louder|quieter|again|repeat|"
    r"ja|jawohl|klar|bitte|nein|ne|nö|abbrechen|lauter|leiser|nochmal|"
    r"oui|non|d'accord|plus\s+fort|encore|"
    r"s[ií]|no|vale|m[aá]s\s+alto|otra\s+vez|"
    r"sim|n[aã]o|est[aá]\s+bem|mais\s+alto|de\s+novo|"
    r"tak|nie|dobrze|g[lł]o[sś]niej|jeszcze\s+raz"
    r")$",
    re.IGNORECASE,
)

def looks_like_command(message: str, lang: str = "en") -> bool:
    """True when a short turn is something the user could really have said.

    Used by the echo filter as a safety net before dropping a turn it
    suspects is the tail of the agent's own TTS: a recognisable command
    ("turn it off", "next"), or a bare answer ("yes", "louder"), is never
    treated as echo no matter how the timing looks.
    """
    msg = (message or "").strip().rstrip(".?!,").lower()
    if not msg:
        return False
    if _ANSWER_WORDS_RE.match(msg):
        return True
    if _match_action_intent(msg, lang) is not None:
        return True
    if _detect_media_command(msg) is not None:
        return True
    if _parse_sweep_noun(msg) is not None:
        return True
    return False
