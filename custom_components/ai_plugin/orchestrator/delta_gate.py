"""Sentence-safe streaming: hold a delta back until it is speakable.

A voice pipeline that starts talking on every token stutters; one that
waits for the whole reply loses the point of streaming. The gate
releases on sentence boundaries.
"""

from __future__ import annotations

import logging

from .textproc import (
    _SENTENCE_SPLIT_RE,
    _flatten_for_voice,
    _strip_emoji,
    _strip_narration,
)

_LOGGER = logging.getLogger(__name__)


class _DeltaGate:
    """Sentence-buffered forwarder between provider deltas and HA's chat log.

    Streamed speech cannot be recalled, so the gate only forwards sentences
    that survive the same sanitation the final reply gets, holds back the
    trailing sentence, and shuts permanently the moment the turn stops being
    stream-safe (actuator call, get_entity miss, thinking leak). A gate with
    no listener is inert — all methods are cheap no-ops.
    """

    def __init__(
        self,
        listener,  # Callable[[str], None] | None
        lang: str = "en",
        voice_mode: bool = False,
    ) -> None:
        self._listener = listener
        self._lang = lang
        self._voice = voice_mode
        self._buf = ""
        self._closed = False
        self._forwarded: list[str] = []

    @property
    def active(self) -> bool:
        return self._listener is not None and not self._closed

    @property
    def forwarded_text(self) -> str:
        return " ".join(self._forwarded)

    def close(self) -> None:
        """Stop live forwarding for the rest of the turn (safety trigger hit).

        The listener is retained: once sentences have been spoken, the
        delta stream is the pipeline's audio channel, so flush_final must
        still be able to deliver the final authoritative reply through it.
        Dropping the listener here would strand the user with a dangling
        preamble ("Sure.") and no answer.
        """
        self._closed = True
        self._buf = ""

    def new_response(self) -> None:
        """Discard the unemitted tail between LLM calls in the tool loop.

        Prose preceding a tool call (usually narration) must never be
        spoken — it stays in the buffer thanks to the hold-back rule and
        is dropped here.
        """
        self._buf = ""

    def feed(self, fragment: str) -> None:
        """Provider callback: accumulate and emit completed sentences."""
        if self._listener is None or self._closed or not fragment:
            return
        self._buf += fragment
        if "<think" in self._buf:
            self.close()
            return
        parts = _SENTENCE_SPLIT_RE.split(self._buf)
        if len(parts) <= 1:
            return
        *complete, self._buf = parts
        for sentence in complete:
            self._emit(sentence)

    def _emit(self, sentence: str) -> None:
        s = _strip_narration(sentence, lang=self._lang)
        s = _strip_emoji(s)
        if self._voice:
            s = _flatten_for_voice(s)
        s = s.strip()
        if not s:
            return
        self._forwarded.append(s)
        try:
            self._listener(s + " ")
        except Exception:  # noqa: BLE001
            _LOGGER.debug("AI Plugin: delta listener raised", exc_info=True)

    def flush_final(self, final_text: str) -> None:
        """Send whatever part of the finished reply was not yet streamed.

        When nothing streamed, the whole reply goes out in one delta (chat
        log stays the single source of what was said) — except on a closed
        gate, where the plain speech field is authoritative and streaming
        would bypass the suppression that closed the gate. Once sentences
        HAVE been streamed, the stream is what the pipeline speaks: a final
        reply that no longer extends the streamed prefix (a verifier or a
        later tool round rewrote it) is delivered in full — repeating a
        short preamble beats losing the answer.
        """
        if self._listener is None:
            return
        listener = self._listener
        self._listener = None
        final_text = (final_text or "").strip()
        if not self._forwarded:
            if final_text and not self._closed:
                try:
                    listener(final_text)
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("AI Plugin: delta listener raised", exc_info=True)
            return
        norm_fwd = " ".join(self.forwarded_text.split())
        norm_final = " ".join(final_text.split())
        if norm_final.startswith(norm_fwd):
            rest = norm_final[len(norm_fwd):].strip()
            if rest:
                try:
                    listener(rest)
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("AI Plugin: delta listener raised", exc_info=True)
        elif norm_final and norm_final not in norm_fwd:
            _LOGGER.debug(
                "AI Plugin: streamed prefix diverged from final reply — "
                "delivering the full reply through the stream"
            )
            try:
                listener(final_text)
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin: delta listener raised", exc_info=True)
