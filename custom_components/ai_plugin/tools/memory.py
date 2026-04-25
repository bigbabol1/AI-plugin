"""Native persistent memory tool for AI Plugin.

Stores user-defined facts in a JSON file inside the HA config directory.
No external dependencies — works out of the box.

Storage: <config_dir>/.ai_plugin_memory_<user_id>.json  (per HA user)
         <config_dir>/.ai_plugin_memory_anonymous.json  (unauthenticated)
Format:  {"facts": ["fact 1", "fact 2", ...]}

Tools exposed to the LLM:
  remember(fact)  — add a fact
  recall()        — list all stored facts
  forget(fact)    — remove a fact (case-insensitive substring match)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

_LOGGER = logging.getLogger(__name__)
_FILE_PREFIX = ".ai_plugin_memory_"


def _memory_path(config_dir: str, user_id: str | None) -> str:
    """Return the memory file path for the given user (or anonymous)."""
    suffix = user_id if user_id else "anonymous"
    return os.path.join(config_dir, f"{_FILE_PREFIX}{suffix}.json")


_STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are",
    "was", "were", "be", "been", "do", "does", "did", "have", "has", "had",
    "i", "my", "me", "you", "your", "he", "she", "it", "we", "they", "them",
    "this", "that", "these", "those", "for", "with", "from", "as", "by",
    "user", "fact", "user's", "always", "forget", "remember", "please",
    "what", "who", "where", "when", "why", "how",
}
_STOPWORDS_DE = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "und", "oder", "ist", "sind", "war", "waren", "sein", "habe", "hat", "hatte",
    "ich", "mein", "meine", "meinem", "du", "dein", "deine", "er", "sie", "es",
    "wir", "ihr", "diese", "dieser", "dieses", "von", "zu", "an", "auf",
    "mit", "für", "durch", "trinke", "esse", "mag", "name", "frau", "mann",
    "vergiss", "merk", "dir", "dass", "bitte", "nicht", "kein", "keine",
    "wie", "wer", "wo", "wann", "warum",
}
_STOPWORDS = _STOPWORDS_EN | _STOPWORDS_DE


def _detect_lang(text: str) -> str:
    """Crude lang detect from stopword presence. Returns 'de'|'en'|'unknown'."""
    import re  # noqa: PLC0415
    toks = {t for t in re.findall(r"\w+", text.lower()) if len(t) >= 2}
    de_hits = len(toks & _STOPWORDS_DE)
    en_hits = len(toks & _STOPWORDS_EN)
    if de_hits > en_hits and de_hits >= 1:
        return "de"
    if en_hits > de_hits and en_hits >= 1:
        return "en"
    return "unknown"


def _tokens(text: str) -> set[str]:
    import re  # noqa: PLC0415
    out = set()
    for tok in re.findall(r"\w+", text.lower()):
        if len(tok) >= 3 and tok not in _STOPWORDS:
            out.add(tok)
    return out


def _fuzzy_overlap(needle: str, haystack: str) -> bool:
    """True if needle and haystack share a non-trivial token, OR substring matches.

    Used as a sanity check when forget(index=N, fact='...') is called: prevents
    the LLM from removing an arbitrary fact when the user-supplied keyword has
    nothing to do with the indexed entry (e.g. user says 'forget Ferrari' and
    LLM picks index=1 which is 'KL is short for Kuala Lumpur' — no overlap).
    """
    if not needle.strip():
        return True
    if needle.lower() in haystack.lower():
        return True
    return bool(_tokens(needle) & _tokens(haystack))

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a fact to persistent memory so it can be recalled in future "
                "conversations. Use this when the user tells you something they want "
                "you to remember (e.g. preferences, names, routines)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember, written as a short statement.",
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Retrieve all facts stored in persistent memory. Call this at the start "
                "of a conversation or whenever you need to remember what the user has "
                "told you in the past."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": (
                "Remove a previously stored fact from persistent memory. Prefer "
                "passing 'index' (the 1-based position from the [USER FACTS] block "
                "or recall output) — this works regardless of language and avoids "
                "substring mismatches. 'fact' is a fallback for substring match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": (
                            "1-based index of the fact to remove, as shown in the "
                            "[USER FACTS] block or recall output. Preferred."
                        ),
                    },
                    "fact": {
                        "type": "string",
                        "description": (
                            "A short keyword from the fact you intend to remove "
                            "(in any language). When 'index' is set, this is "
                            "used as a sanity check — it must overlap the stored "
                            "text at index N or the call is rejected. When "
                            "'index' is unset, used as a substring match."
                        ),
                    },
                },
                "required": ["fact"],
            },
        },
    },
]

TOOL_NAMES = {"remember", "recall", "forget"}


class MemoryTool:
    """Reads and writes per-user persistent memory files."""

    def __init__(self, config_dir: str) -> None:
        self._config_dir = config_dir

    # ── public ───────────────────────────────────────────────────────────────

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        user_id: str | None = None,
        user_message: str = "",
    ) -> str:
        """Dispatch to the correct memory operation. Never raises."""
        path = _memory_path(self._config_dir, user_id)
        try:
            if name == "remember":
                return await self._remember(arguments.get("fact", "").strip(), path)
            if name == "recall":
                return await self._recall(path)
            if name == "forget":
                index = arguments.get("index")
                fact = (arguments.get("fact") or "").strip()
                return await self._forget(fact, path, index, user_message)
            return f"[Unknown memory tool: {name!r}]"
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("AI Plugin memory tool %r failed: %s", name, exc)
            return f"[Memory error: {exc}]"

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: str) -> list[str]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return list(data.get("facts", []))
        except FileNotFoundError:
            return []
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("AI Plugin memory: could not read %s: %s", path, exc)
            return []

    @staticmethod
    def _save(path: str, facts: list[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"facts": facts}, f, ensure_ascii=False, indent=2)

    async def _remember(self, fact: str, path: str) -> str:
        if not fact:
            return "Nothing to remember — fact was empty."
        import asyncio  # noqa: PLC0415
        loop = asyncio.get_running_loop()
        facts = await loop.run_in_executor(None, self._load, path)
        if fact.lower() not in (f.lower() for f in facts):
            facts.append(fact)
            await loop.run_in_executor(None, self._save, path, facts)
        return f"Remembered: {fact}"

    async def _recall(self, path: str) -> str:
        import asyncio  # noqa: PLC0415
        facts = await asyncio.get_running_loop().run_in_executor(None, self._load, path)
        if not facts:
            return "No facts stored yet."
        numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
        return f"Stored memories:\n{numbered}"

    async def _forget(
        self,
        fact: str,
        path: str,
        index: Any = None,
        user_message: str = "",
    ) -> str:
        import asyncio  # noqa: PLC0415
        loop = asyncio.get_running_loop()
        facts = await loop.run_in_executor(None, self._load, path)
        _LOGGER.info(
            "AI Plugin memory forget called: index=%r fact=%r user_msg=%r stored=%d",
            index, fact, user_message, len(facts),
        )

        # Anti-hallucination guard: when fact is supplied, it must trace back
        # to the user's actual request (substring/token overlap with
        # user_message). This catches the LLM lifting a keyword out of the
        # [USER FACTS] block after the user asked about something unrelated
        # (e.g. user says "forget Ferrari", LLM passes fact="KL" to satisfy a
        # naive substring check). Cross-language forget keeps working: the
        # LLM can pass the user's keyword as-is (e.g. fact="Kaffee" on a
        # message asking to forget coffee) and the index points to the
        # correct stored fact even when stored in another language.
        if user_message and fact and not _fuzzy_overlap(fact, user_message):
            _LOGGER.info(
                "AI Plugin memory forget refused: fact %r not in user message %r",
                fact, user_message,
            )
            return (
                f"Refusing to forget '{fact}' — that keyword did not appear "
                "in the user's request. Tell the user that fact is not stored."
            )

        if index is not None:
            try:
                idx = int(index)
            except (TypeError, ValueError):
                return f"Invalid index '{index}'. Use a 1-based number."
            if idx < 1 or idx > len(facts):
                return (
                    f"No fact at index {idx}. Stored facts are 1..{len(facts)}; "
                    "call recall to see the current list."
                )
            target = facts[idx - 1]
            # Require LLM to supply 'fact' arg so the user_message guard above
            # has something to verify. Without it, the model can pass an
            # arbitrary index based on a hallucinated user request.
            if not fact and user_message:
                _LOGGER.info(
                    "AI Plugin memory forget refused: index %d given without fact arg",
                    idx,
                )
                return (
                    "Refusing to forget without a 'fact' keyword. Pass "
                    "fact='<keyword copied verbatim from the user's message>' "
                    "alongside index so the call can be validated."
                )
            # When user msg and stored fact are in the same detected language,
            # require fact (which we already verified came from the user msg)
            # to also overlap the stored entry. Cross-language pairs skip this
            # check because tokens won't match (e.g. "Kaffee" vs "coffee").
            if user_message:
                lang_user = _detect_lang(user_message)
                lang_stored = _detect_lang(target)
                if (
                    lang_user != "unknown"
                    and lang_stored != "unknown"
                    and lang_user == lang_stored
                    and not _fuzzy_overlap(fact, target)
                ):
                    _LOGGER.info(
                        "AI Plugin memory forget refused: fact %r does not match stored %r (same lang %s)",
                        fact, target, lang_user,
                    )
                    return (
                        f"Refusing to forget index {idx} ({target!r}) — the "
                        f"keyword '{fact}' from the user's request does not "
                        "appear in that stored fact. Tell the user that fact "
                        "is not stored."
                    )
            removed_fact = facts.pop(idx - 1)
            await loop.run_in_executor(None, self._save, path, facts)
            return f"Forgotten: {removed_fact}"

        if not fact:
            return "Nothing to forget — provide an index or fact substring."
        needle = fact.lower()
        original_len = len(facts)
        facts = [f for f in facts if needle not in f.lower()]
        removed = original_len - len(facts)
        if removed:
            await loop.run_in_executor(None, self._save, path, facts)
            return f"Forgotten {removed} fact(s) matching '{fact}'."
        return f"No stored fact matched '{fact}'."
