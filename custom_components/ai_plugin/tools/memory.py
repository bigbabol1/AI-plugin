"""Native persistent memory tool for AI Plugin.

Stores user-defined facts in a JSON file inside the HA config directory.
No external dependencies — works out of the box.

Storage: <config_dir>/.ai_plugin_memory.json
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
_FILE_NAME = ".ai_plugin_memory.json"

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
                "Remove a previously stored fact from persistent memory. Use when the "
                "user says something is no longer true or asks you to forget something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": (
                            "The fact to remove. A case-insensitive substring match is "
                            "used, so you don't need an exact copy of the stored text."
                        ),
                    }
                },
                "required": ["fact"],
            },
        },
    },
]

TOOL_NAMES = {"remember", "recall", "forget"}


class MemoryTool:
    """Reads and writes the persistent memory file."""

    def __init__(self, config_dir: str) -> None:
        self._path = os.path.join(config_dir, _FILE_NAME)

    # ── public ───────────────────────────────────────────────────────────────

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch to the correct memory operation. Never raises."""
        try:
            if name == "remember":
                return await self._remember(arguments.get("fact", "").strip())
            if name == "recall":
                return await self._recall()
            if name == "forget":
                return await self._forget(arguments.get("fact", "").strip())
            return f"[Unknown memory tool: {name!r}]"
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("AI Plugin memory tool %r failed: %s", name, exc)
            return f"[Memory error: {exc}]"

    # ── private ───────────────────────────────────────────────────────────────

    def _load(self) -> list[str]:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            return list(data.get("facts", []))
        except FileNotFoundError:
            return []
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("AI Plugin memory: could not read %s: %s", self._path, exc)
            return []

    def _save(self, facts: list[str]) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"facts": facts}, f, ensure_ascii=False, indent=2)

    async def _remember(self, fact: str) -> str:
        if not fact:
            return "Nothing to remember — fact was empty."
        import asyncio  # noqa: PLC0415
        loop = asyncio.get_running_loop()
        facts = await loop.run_in_executor(None, self._load)
        if fact.lower() not in (f.lower() for f in facts):
            facts.append(fact)
            await loop.run_in_executor(None, self._save, facts)
        return f"Remembered: {fact}"

    async def _recall(self) -> str:
        import asyncio  # noqa: PLC0415
        facts = await asyncio.get_running_loop().run_in_executor(None, self._load)
        if not facts:
            return "No facts stored yet."
        numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
        return f"Stored memories:\n{numbered}"

    async def _forget(self, fact: str) -> str:
        if not fact:
            return "Nothing to forget — fact was empty."
        import asyncio  # noqa: PLC0415
        loop = asyncio.get_running_loop()
        facts = await loop.run_in_executor(None, self._load)
        needle = fact.lower()
        original_len = len(facts)
        facts = [f for f in facts if needle not in f.lower()]
        removed = original_len - len(facts)
        if removed:
            await loop.run_in_executor(None, self._save, facts)
            return f"Forgotten {removed} fact(s) matching '{fact}'."
        return f"No stored fact matched '{fact}'."
