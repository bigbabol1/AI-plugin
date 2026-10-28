"""Short-lived audio buffers that link a transcript back to its audio.

The STT pass-through sees the audio but not the satellite; the
conversation agent sees the satellite but not the audio. The text the
agent receives is exactly the transcript the STT engine returned, so the
buffer is keyed by that transcript. Entries expire after ``TTL_SECONDS``.
An entry may carry the fingerprint the STT pass-through already started
while the user was speaking; whoever takes the entry awaits it.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

TTL_SECONDS = 30.0
MAX_ENTRIES = 8
# 15 s of 16 kHz 16-bit mono. Speech beyond 4 s is never analysed, but the
# raw clip includes the silence before end-of-speech fires.
MAX_PCM_BYTES = 15 * 16000 * 2

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def normalize_transcript(text: str) -> str:
    """Case- and punctuation-insensitive key for a transcript."""
    return _NON_WORD.sub(" ", (text or "").casefold()).strip()


@dataclass
class _Entry:
    pcm: bytes
    created: float
    ambiguous: bool = False
    fingerprint: asyncio.Future | None = None


class AudioBufferStore:
    """Transcript → PCM, with expiry and a duplicate-transcript guard."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[str, _Entry] = {}

    def put(
        self, transcript: str, pcm: bytes, fingerprint: asyncio.Future | None = None
    ) -> None:
        key = normalize_transcript(transcript)
        if not key or not pcm:
            return
        self._expire()
        if key in self._entries:
            # Two live clips with the same words (two rooms saying "stop" at
            # once): we cannot tell which audio belongs to which turn, so
            # neither is used. Unknown beats a confident wrong answer.
            self._entries[key].ambiguous = True
            return
        while len(self._entries) >= MAX_ENTRIES:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = _Entry(
            pcm=pcm[:MAX_PCM_BYTES], created=self._clock(), fingerprint=fingerprint
        )

    def take_clip(self, transcript: str) -> tuple[bytes, asyncio.Future | None] | None:
        """Remove and return (audio, early fingerprint or None), or None if unusable."""
        self._expire()
        entry = self._entries.pop(normalize_transcript(transcript), None)
        if entry is None or entry.ambiguous:
            return None
        return entry.pcm, entry.fingerprint

    def take(self, transcript: str) -> bytes | None:
        """Remove and return the audio for ``transcript`` (None if unusable)."""
        clip = self.take_clip(transcript)
        return clip[0] if clip else None

    def __len__(self) -> int:
        self._expire()
        return len(self._entries)

    def _expire(self) -> None:
        cutoff = self._clock() - TTL_SECONDS
        for key in [k for k, e in self._entries.items() if e.created < cutoff]:
            del self._entries[key]
