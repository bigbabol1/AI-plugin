"""Context Manager — sliding-window token budget + summarization.

Root cause of local-model memory failure:
  system_prompt ≈ 2 000 tokens
  tool_schemas  ≈ 2 000 tokens  (Week 3: real value from MCP client)
  ─────────────────────────────
  On an 8 K model only ~4 000 tokens remain for history — and the
  sliding-window fix below keeps conversation history within that
  remaining budget, summarizing older turns when it gets tight.

Budget model
────────────
  available = max_tokens - system_tokens - tool_budget

  soft_limit = available × 0.65   → trigger summarization
  hard_limit = available × 0.85   → emergency truncation (last resort)

Token estimation
────────────────
  estimate_tokens(text) = max(1, len(text) // 4)
  (chars-to-tokens heuristic, ±20%).  The 65% soft limit provides
  headroom for the estimation error so we never over-run the model.

Concurrency
───────────
  Per-conversation asyncio.Lock — not a global lock — so concurrent
  users never block each other.

Summarization
─────────────
  When history_tokens > soft_limit we keep the most recent
  KEEP_RECENT_TURNS (= 4) full turns intact and ask the LLM to
  condense the rest into a single summary message.  If summarization
  fails (network error, model timeout) we log a warning and continue
  — the hard-limit truncation below is the safety net.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .providers.base import AbstractProvider

_LOGGER = logging.getLogger(__name__)

# Number of most-recent full turns (user + assistant pairs) to keep
# verbatim when summarizing older history.
_KEEP_RECENT_TURNS = 4


class ContextManager:
    """Manages per-conversation message history with token-aware trimming."""

    def __init__(
        self,
        max_tokens: int,
        tool_token_budget: int = 2000,
    ) -> None:
        """Initialise the context manager.

        Args:
            max_tokens: Total model context window size in tokens
                        (from CONF_CONTEXT_WINDOW, default 8192).
            tool_token_budget: Tokens reserved for tool schemas.
                               Default 2000; Week 3 MCP client replaces
                               this with the real schema token count.
        """
        self._max_tokens = max_tokens
        self._tool_budget = tool_token_budget
        self._history: dict[str, list[dict[str, str]]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # ── public helpers ────────────────────────────────────────────────────────

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate: 4 chars ≈ 1 token (±20%)."""
        return max(1, len(text) // 4)

    def get_history(self, conv_id: str) -> list[dict[str, str]]:
        """Return a snapshot of the raw history for a conversation (tests/debug)."""
        return list(self._history.get(conv_id, []))

    # ── per-conversation lock ─────────────────────────────────────────────────

    def _get_lock(self, conv_id: str) -> asyncio.Lock:
        if conv_id not in self._locks:
            self._locks[conv_id] = asyncio.Lock()
        return self._locks[conv_id]

    # ── private token math ────────────────────────────────────────────────────

    def _estimate_messages_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            content = m.get("content")
            if content:
                total += self.estimate_tokens(str(content))
            elif "tool_calls" in m:
                total += self.estimate_tokens(str(m["tool_calls"]))
            else:
                total += 1
        return total

    def _budget(self, system_prompt: str) -> tuple[int, int]:
        """Return (soft_limit, hard_limit) for history tokens."""
        system_tokens = self.estimate_tokens(system_prompt)
        available = max(0, self._max_tokens - system_tokens - self._tool_budget)
        return int(available * 0.65), int(available * 0.85)

    def budget_report(
        self,
        system_prompt: str,
        messages: list[dict],
        tool_schemas_tokens: int,
    ) -> dict:
        """Snapshot of token pressure for a single turn.

        Returns a plain dict so the caller can log or act without coupling to
        this class's internals.  Keys: system, tools, history, total, cap, pct.
        pct is history/available (0.0–1.0+); cap is self._max_tokens.
        """
        system_tokens = self.estimate_tokens(system_prompt)
        # messages list includes the system message; skip it for history count.
        history_msgs = [m for m in messages if m.get("role") != "system"]
        history_tokens = self._estimate_messages_tokens(history_msgs)
        total = system_tokens + tool_schemas_tokens + history_tokens
        available = max(1, self._max_tokens - system_tokens - tool_schemas_tokens)
        return {
            "system": system_tokens,
            "tools": tool_schemas_tokens,
            "history": history_tokens,
            "total": total,
            "cap": self._max_tokens,
            "pct": history_tokens / available,
        }

    # ── public API ────────────────────────────────────────────────────────────

    async def add_turn(self, conv_id: str, role: str, content: str) -> None:
        """Append a single text message to the conversation history."""
        async with self._get_lock(conv_id):
            self._history.setdefault(conv_id, []).append(
                {"role": role, "content": content}
            )

    async def add_raw_message(self, conv_id: str, message: dict) -> None:
        """Append a pre-formatted message dict (e.g. tool call or tool result)."""
        async with self._get_lock(conv_id):
            self._history.setdefault(conv_id, []).append(message)

    async def get_messages(
        self,
        conv_id: str,
        system_prompt: str,
    ) -> list[dict[str, str]]:
        """Return messages ready to send to the LLM.

        Format: [system_message, *history], hard-truncated so the total
        never exceeds 85 % of the available budget.
        """
        _, hard_limit = self._budget(system_prompt)

        async with self._get_lock(conv_id):
            history = list(self._history.get(conv_id, []))

        # Hard limit: drop oldest messages until we fit.
        # Drop from the front until the first user message so we never start
        # with an orphaned tool or assistant message.
        while history and self._estimate_messages_tokens(history) > hard_limit:
            history = history[1:]
        while history and history[0].get("role") != "user":
            history = history[1:]

        return [{"role": "system", "content": system_prompt}, *history]

    async def summarize_if_needed(
        self,
        conv_id: str,
        system_prompt: str,
        provider: AbstractProvider,
    ) -> None:
        """Summarize old turns when history approaches the soft token limit.

        Keeps the _KEEP_RECENT_TURNS most-recent full turns verbatim and
        replaces older turns with a single summary message.  Logs INFO
        when triggered so users can correlate slower responses.

        If the LLM call for summarization fails, we log a warning and
        return without mutating history — the hard limit in get_messages
        acts as the fallback.
        """
        soft_limit, _ = self._budget(system_prompt)

        async with self._get_lock(conv_id):
            history = list(self._history.get(conv_id, []))

        history_tokens = self._estimate_messages_tokens(history)
        if history_tokens <= soft_limit:
            return  # Nothing to do.

        keep_count = _KEEP_RECENT_TURNS * 2  # user+assistant pairs → messages
        if len(history) <= keep_count:
            return  # Not enough history to split; hard limit will handle it.

        to_summarize_raw = history[:-keep_count]
        to_keep = history[-keep_count:]
        # Summarizer uses async_complete (no tool support) — filter to text-only.
        to_summarize = [
            m for m in to_summarize_raw
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        ]

        _LOGGER.info(
            "AI Plugin: summarizing %d messages for conv_id=%s "
            "(history_tokens=%d > soft_limit=%d)",
            len(to_summarize),
            conv_id,
            history_tokens,
            soft_limit,
        )

        summary_text = await self._call_summarize(to_summarize, provider)
        if summary_text is None:
            return  # Summarization failed; hard limit is the safety net.

        summary_msg: dict[str, str] = {
            "role": "assistant",
            "content": f"Prior conversation summary: {summary_text}",
        }

        async with self._get_lock(conv_id):
            # Guard: only replace if history hasn't shrunk under us.
            current = self._history.get(conv_id, [])
            if len(current) >= len(to_keep):
                self._history[conv_id] = [summary_msg, *to_keep]

    async def remove_last_turn(self, conv_id: str) -> None:
        """Remove the last message from history (rollback on provider error).

        Called by Orchestrator when the LLM call fails after a user turn was
        already appended, to avoid leaving orphaned user turns in history.
        """
        async with self._get_lock(conv_id):
            history = self._history.get(conv_id)
            if history:
                history.pop()

    async def clear(self, conv_id: str) -> None:
        """Clear all history for a conversation (e.g. user requests reset)."""
        async with self._get_lock(conv_id):
            self._history.pop(conv_id, None)

    # ── private helpers ───────────────────────────────────────────────────────

    async def _call_summarize(
        self,
        messages: list[dict[str, str]],
        provider: AbstractProvider,
    ) -> str | None:
        """Ask the provider to summarize a list of messages.

        Returns the summary string, or None on failure.
        """
        prompt = [
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation history into a concise "
                    "paragraph. Include key facts, decisions, and context that "
                    "would be needed to continue the conversation naturally."
                ),
            },
            *messages,
            {
                "role": "user",
                "content": "Please summarize the conversation above.",
            },
        ]
        try:
            return await provider.async_complete(prompt)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "AI Plugin: summarization failed for conversation, continuing without: %s",
                exc,
            )
            return None
