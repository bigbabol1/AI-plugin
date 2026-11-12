"""Conversation diary: what was talked about, kept across devices, restarts and days.

Home Assistant keeps a conversation for a few minutes and only on the device it
started on. "About the chicken stir fry you just suggested, tell me again how
to start" reached another satellite 22 minutes after the recipe and got "I
don't have any record" (2026-11-10). After a conversation goes quiet, the
model writes short notes about it; on later turns the notes closest to what is
said go into the prompt, and the model decides whether one fits. Replayed on
the voice model, 0 of 50 unrelated turns brought up an attached note and tool
calls were unchanged, while follow-ups answered from the right note 99 of 106
times when it was attached (2026-11-10).

Every note carries the person it came from. With sharing off, a person is
shown their own notes and those of unrecognised voices; with it on, everyone's.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import re
import time
from typing import Any
import uuid

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
BLOCK_MARKER = "[EARLIER CONVERSATIONS]"
RECALL_COUNT = 3
_MAX_NOTE_CHARS = 600
# A note below this similarity to what is said is left out. The model judges
# the rest; this only drops notes that share nothing with the turn. Unrelated
# turns score about 0.03-0.04; "How do I start the dish you suggested earlier?"
# scored 0.18 against the German note it meant (2026-11-10).
_MIN_SIMILARITY = 0.1
# No weighting by age. Tried 2026-11-12 (half-life 7 days, floor 0.5): the
# offline ranking improved, but on the voice model vague follow-ups ("the
# dinner you suggested" after two cooking chats) were answered from the right
# note 23 of 36 times against 27 without, because brand-new notes on other
# topics took slots from older ones that fit. The block already lists notes
# newest first.

BLOCK_HEAD = (
    BLOCK_MARKER + "\n"
    "Notes from earlier conversations in this home, newest first. They may have "
    "nothing to do with what is said now. When the user refers back to one or "
    "asks something a note answers, answer from that note and do not search the "
    "web for it. Otherwise ignore them and never mention them."
)


def _words(text: str) -> set[str]:
    """Longer words, cut to a stem-like prefix: "Linsensuppe" meets "Linsen"."""
    return {w[:6] for w in re.findall(r"\w+", text.lower()) if len(w) > 3}


class DiaryManager:
    """Notes per home, each with its owner. Created by async_setup_entry."""

    def __init__(self, hass, entry_id: str) -> None:
        self._hass = hass
        self._entry_id = entry_id
        self._store = None
        self._entries: list[dict[str, Any]] = []
        self._vectors: dict[str, Any] = {}
        self._encoder = None
        self._model_lock = asyncio.Lock()
        self.last_error: str | None = None

    # ── storage ──────────────────────────────────────────────────────────────

    async def async_load(self) -> None:
        from homeassistant.helpers.storage import Store  # noqa: PLC0415

        self._store = Store(self._hass, STORAGE_VERSION, f"ai_plugin_diary.{self._entry_id}")
        data = await self._store.async_load() or {}
        self._entries = list(data.get("entries") or [])

    async def _async_save(self) -> None:
        if self._store is not None:
            await self._store.async_save({"entries": self._entries})

    async def async_remove_storage(self) -> None:
        if self._store is not None:
            await self._store.async_remove()

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    async def async_prune(self, days: float, now: datetime | None = None) -> int:
        """Drop notes older than ``days``; returns how many went."""
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
        keep = [e for e in self._entries if _when(e) >= cutoff]
        gone = len(self._entries) - len(keep)
        if gone:
            for e in self._entries:
                if _when(e) < cutoff:
                    self._vectors.pop(e["id"], None)
            self._entries = keep
            await self._async_save()
        return gone

    async def async_add(
        self,
        notes: list[str],
        *,
        owner: str | None,
        owner_name: str | None,
        language: str,
        when: datetime,
    ) -> list[dict[str, Any]]:
        added = []
        for text in notes:
            text = re.sub(r"\s+", " ", str(text or "")).strip()[:_MAX_NOTE_CHARS]
            if not text:
                continue
            entry = {
                "id": uuid.uuid4().hex,
                "owner": owner,
                "owner_name": owner_name,
                "language": language,
                "time": when.astimezone(timezone.utc).isoformat(),
                "text": text,
            }
            self._entries.append(entry)
            added.append(entry)
        if added:
            await self._async_save()
            if self._encoder is not None:
                for entry in added:
                    self._vectors[entry["id"]] = self._encoder.embed(entry["text"])
        return added

    async def async_forget(self, owner: str | None) -> int:
        """Remove one person's notes (unrecognised voices: owner None)."""
        keep = [e for e in self._entries if e.get("owner") != owner]
        gone = len(self._entries) - len(keep)
        if gone:
            for e in self._entries:
                if e.get("owner") == owner:
                    self._vectors.pop(e["id"], None)
            self._entries = keep
            await self._async_save()
        return gone

    # ── model ────────────────────────────────────────────────────────────────

    def _model_dir(self) -> Path:
        return Path(self._hass.config.path(".storage", "ai_plugin_memory_model"))

    async def async_ensure_model(self) -> bool:
        """Download (first time) and load the model. False until it works."""
        if self._encoder is not None:
            return True
        async with self._model_lock:
            if self._encoder is not None:
                return True
            try:
                from homeassistant.helpers.aiohttp_client import (  # noqa: PLC0415
                    async_get_clientsession,
                )

                from .model import Encoder, async_download  # noqa: PLC0415

                await async_download(
                    async_get_clientsession(self._hass), self._model_dir(), self._hass
                )
                started = time.monotonic()
                encoder = await self._hass.async_add_executor_job(Encoder.load, self._model_dir())
                vectors = await self._hass.async_add_executor_job(
                    lambda: {e["id"]: encoder.embed(e["text"]) for e in self._entries}
                )
                self._encoder, self._vectors = encoder, vectors
                self.last_error = None
                _LOGGER.info(
                    "AI Plugin diary: model loaded in %.1f s (%d notes)",
                    time.monotonic() - started, len(self._entries),
                )
            except Exception as exc:  # noqa: BLE001 — offline, corrupt, numpy missing
                self.last_error = str(exc)
                _LOGGER.warning("AI Plugin diary: search model unavailable, using words: %s", exc)
                return False
        return True

    # ── recall ───────────────────────────────────────────────────────────────

    def visible(self, viewer: str | None, share: bool) -> list[dict[str, Any]]:
        """The notes this speaker may be shown."""
        if share:
            return list(self._entries)
        return [e for e in self._entries if e.get("owner") in (viewer, None)]

    def recall(
        self, text: str, viewer: str | None, share: bool, count: int = RECALL_COUNT
    ) -> list[dict[str, Any]]:
        """The notes closest to ``text``, best first; words alone without the model."""
        pool = self.visible(viewer, share)
        if not pool or not text.strip():
            return []
        if self._encoder is not None:
            query = self._encoder.embed(text)
            if query is None:
                return []
            scored = []
            for e in pool:
                vec = self._vectors.get(e["id"])
                if vec is None:
                    continue
                score = float(vec @ query)
                if score >= _MIN_SIMILARITY:
                    scored.append((score, e))
        else:
            asked = _words(text)
            scored = [
                (len(asked & _words(e["text"])) / (len(asked) or 1), e) for e in pool
            ]
            scored = [(s, e) for s, e in scored if s > 0]
        scored.sort(key=lambda pair: -pair[0])
        return [e for _, e in scored[:count]]


def _when(entry: dict[str, Any]) -> datetime:
    try:
        return datetime.fromisoformat(entry["time"])
    except (KeyError, TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)


def format_block(entries: list[dict[str, Any]], tz, household: str) -> str:
    """[EARLIER CONVERSATIONS] with whose conversation each note came from.

    Labelled so the model never presents someone else's conversation as the
    speaker's own; newest first, as the head says."""
    if not entries:
        return ""
    lines = []
    for e in sorted(entries, key=_when, reverse=True):
        when = _when(e).astimezone(tz)
        who = e.get("owner_name") or household
        lines.append(f"- Conversation with {who}, {when.strftime('%a %d %b %H:%M')}: {e['text']}")
    return BLOCK_HEAD + "\n" + "\n".join(lines)
