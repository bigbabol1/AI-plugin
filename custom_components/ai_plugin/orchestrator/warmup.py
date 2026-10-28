"""Keep the voice prompt read in by the model between turns.

Every voice turn starts with the same ~10k tokens: the static system
prompt and the tool schemas. Ollama keeps what it last read and only
reads what is new, so a turn normally starts in well under a second. It
has to read everything again (7-8 s on the voice model) whenever that
head changed since the last turn: after a restart, when an MCP server
connects late or reconnects with other tools, or after something else
used the model with a different prompt. Here a background check notices
such a change while nobody is talking and has the model read the new
head in ahead of the next voice turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

_LOGGER = logging.getLogger(__name__)

# How often the head is checked. A change must be seen on two checks in a
# row before it is read in, so MCP servers that are still connecting after
# a restart do not cause one read-in each.
WARM_CHECK_S = 30.0
# No read-in within this long after a turn: a follow-up may still come,
# and it must not queue behind a read-in.
WARM_IDLE_S = 60.0


def _head_key(system_prompt: str, tools: list[dict]) -> str:
    return json.dumps([system_prompt, tools], sort_keys=True, default=str)


def _tool_name(schema: dict) -> str:
    return str((schema.get("function") or {}).get("name") or "?")


def head_change(old: tuple | None, new: tuple) -> str:
    """Say what differs between two (system prompt, tools) heads."""
    if old is None:
        return "first since start"
    parts = []
    if old[0] != new[0]:
        parts.append("system prompt changed")
    before = {_tool_name(t): json.dumps(t, sort_keys=True, default=str) for t in old[1]}
    after = {_tool_name(t): json.dumps(t, sort_keys=True, default=str) for t in new[1]}
    if added := sorted(after.keys() - before.keys()):
        parts.append(f"{len(added)} tools added ({', '.join(added[:6])})")
    if removed := sorted(before.keys() - after.keys()):
        parts.append(f"{len(removed)} tools removed ({', '.join(removed[:6])})")
    if changed := sorted(n for n in after.keys() & before.keys() if after[n] != before[n]):
        parts.append(f"tools changed ({', '.join(changed[:6])})")
    return "; ".join(parts) or "tool order changed"


class PromptWarmupMixin:
    """Read-in of the voice prompt head. Mixed into Orchestrator."""

    # (key, system prompt, tools) of the head the model last read for a
    # voice turn, or None when that is unknown or another prompt came since.
    _warm_head: tuple | None = None
    _last_llm_at: float = 0.0
    _voice_turn_seen: bool = False

    def async_start_warmup(self) -> None:
        """Start the background check; it stops with async_close."""
        if self._hass is None or self._prune_schemas:
            # Pruned schemas differ per message: there is no stable head.
            return
        task = self._hass.async_create_background_task(
            self._keep_voice_prompt_warm(), "ai_plugin voice prompt read-in"
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _note_prompt_head(
        self, system_prompt: str, tools: list[dict], voice_mode: bool
    ) -> None:
        """Record the head a turn is about to send to the model."""
        self._last_llm_at = time.monotonic()
        if not voice_mode:
            # The model now holds the text prompt, not the voice one.
            self._warm_head = None
            return
        self._voice_turn_seen = True
        key = _head_key(system_prompt, tools)
        warm = self._warm_head
        if warm is None or warm[0] != key:
            _LOGGER.debug(
                "AI Plugin: voice prompt not read in yet (%s) — this turn reads it all",
                head_change(warm[1:] if warm else None, (system_prompt, tools)),
            )
        self._warm_head = (key, system_prompt, tools)

    def _voice_in_use(self) -> bool:
        """Whether voice turns can reach this agent: one came since start, or
        an Assist pipeline with speech-to-text uses it. Without either, a
        read-in only pushes the text chat's prompt out of the model."""
        if self._voice_turn_seen:
            return True
        try:
            from ..voice_id.pipelines import _pipelines, agent_entity_ids  # noqa: PLC0415

            agents = agent_entity_ids(self._hass, self._entry)
            return any(
                p.stt_engine and p.conversation_engine in agents
                for p in _pipelines(self._hass)
            )
        except Exception:  # noqa: BLE001 — unknown: keep reading in, as before
            _LOGGER.debug("AI Plugin: pipeline check for the read-in failed", exc_info=True)
            return True

    async def _voice_prompt_head(self) -> tuple[str, list[dict]]:
        system_prompt = await self._build_system_prompt(True)
        # Any device id: satellite turns always carry one.
        return system_prompt, self._tool_schemas_for("", True, "read-in")

    async def _keep_voice_prompt_warm(self) -> None:
        pending: str | None = None
        while True:
            await asyncio.sleep(WARM_CHECK_S)
            if time.monotonic() - self._last_llm_at < WARM_IDLE_S:
                continue
            if not self._voice_in_use():
                continue
            try:
                system_prompt, tools = await self._voice_prompt_head()
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin: voice prompt head failed", exc_info=True)
                continue
            key = _head_key(system_prompt, tools)
            warm = self._warm_head
            if warm is not None and warm[0] == key:
                pending = None
                continue
            if pending != key:
                pending = key
                continue
            started = time.monotonic()
            try:
                done = await self._provider.async_warm(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "."},
                    ],
                    tools,
                )
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin: voice prompt read-in failed", exc_info=True)
                continue
            if done is None:
                return  # this server keeps no prompt between requests
            if self._last_llm_at > started:
                continue  # a turn ran meanwhile and recorded its own head
            _LOGGER.debug(
                "AI Plugin: read in the voice prompt ahead of the next turn in %.1f s (%s)",
                time.monotonic() - started,
                head_change(warm[1:] if warm else None, (system_prompt, tools)),
            )
            self._warm_head = (key, system_prompt, tools)
            pending = None
