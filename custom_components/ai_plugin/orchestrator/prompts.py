"""The blocks that make up the system prompt for one turn.

Kept apart from the turn logic because they are pure assembly: what the
model is told about the time, the house, the user's stored facts and
where it all is. The mixin exists so these read as methods on the
Orchestrator, which is how they are called.
"""

from __future__ import annotations

import asyncio
import logging

from ..const import (
    CONF_SYSTEM_PROMPT,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
)

_LOGGER = logging.getLogger(__name__)


class PromptBlocksMixin:
    """System prompt assembly. Mixed into Orchestrator."""

    async def _build_system_prompt(self, voice_mode: bool) -> str:
        """Return the STATIC system prompt for this request.

        Only content that is byte-stable across turns belongs here (base
        prompt, custom prompt, home location — cached after first resolve).
        Volatile blocks ([CURRENT TIME], [USER FACTS], [LAST ACTION]) go
        into a late system message via _build_volatile_block instead: a
        stable prompt head lets Ollama reuse its prompt prefix cache, so
        each turn only pre-fills the new tokens instead of the whole
        system prompt + tool schemas + history.

        v0.9.0: per-language trigger hints removed. Modern multilingual
        LLMs route non-English utterances correctly without pinned hints,
        and deterministic shortcuts in shortcuts.py handle the
        load-bearing language-specific behaviour.
        """
        base = SYSTEM_PROMPT_VOICE if voice_mode else SYSTEM_PROMPT_DEFAULT
        location_block = await self._build_location_block()
        custom = self._entry.options.get(CONF_SYSTEM_PROMPT, "").strip()
        parts = [base, location_block]
        if custom:
            parts.append(custom)
        return "\n\n".join(p for p in parts if p)

    async def _build_volatile_block(
        self, user_id: str | None, conversation_id: str
    ) -> str:
        """Per-turn context ([CURRENT TIME], [USER FACTS], [LAST ACTION]).

        Injected as a system message directly before the newest user turn
        so the static prompt head above stays cache-stable.
        """
        parts = [
            self._build_time_block(),
            await self._build_user_facts_block(user_id),
        ]
        last = self._last_entities.get(conversation_id)
        if last:
            parts.append(f"[LAST ACTION]\n{last}")
        return "\n\n".join(p for p in parts if p)

    def _build_time_block(self) -> str:
        """Return a fresh [CURRENT TIME] block in the user's local timezone.

        Computed per-request so small models that ignore tools still answer
        time questions correctly. Without this block the LLM hallucinates
        clock readings.
        """
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo
            cfg = self._hass.config if self._hass is not None else None
            tz_name = (getattr(cfg, "time_zone", None) or "").strip() if cfg else ""
            now = datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.now().astimezone()
            stamp = now.strftime("%A, %Y-%m-%d %H:%M %Z")
            return (
                f"[CURRENT TIME]\n{stamp}\n"
                "Use this for any question about the time, date, day-of-week, "
                "or 'how long until X'. Do NOT guess the time."
            )
        except Exception:  # noqa: BLE001
            _LOGGER.debug("AI Plugin: time block build failed", exc_info=True)
            return ""

    async def _build_user_facts_block(self, user_id: str | None) -> str:
        """Render stored user facts as a numbered prompt block.

        Auto-injected on every turn so small LLMs that fail to call the
        recall tool can still answer questions about previously-saved
        facts (name, preferences, etc.). The numbered list also doubles
        as the index reference for the two-step forget(index=N) flow.
        File I/O runs in the executor to keep the event loop unblocked.
        """
        if self._memory is None:
            return ""
        try:
            from ..tools.memory import _memory_path  # noqa: PLC0415
            path = _memory_path(self._memory._config_dir, user_id)
            facts = await asyncio.get_running_loop().run_in_executor(
                None, self._memory._load, path
            )
        except Exception:  # noqa: BLE001
            _LOGGER.debug("AI Plugin: could not read user facts", exc_info=True)
            return ""
        if not facts:
            return ""
        numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
        return (
            "[USER FACTS]\n"
            "Facts previously saved by the user. Use them when answering "
            "questions about who they are or their preferences. To remove "
            "a fact, call forget(index=N, fact='<keyword>') with the matching "
            "number. Never call forget for a fact that is not listed here.\n"
            f"{numbered}"
        )

    async def _build_location_block(self) -> str:
        """Render the resolved home location as a prompt block.

        The block is informational only — it does NOT instruct the model
        to inject the city into queries; that is handled deterministically
        in ``web_search.async_search`` based on the ``near_user`` flag and
        a locality regex. The block tells the model where the user lives
        so it can answer "where am I?" accurately.

        Renders only fields that resolved. When nothing resolves (no
        coords, no country, no entity, network offline) the block is
        omitted entirely — better silence than fabricated geography.
        """
        try:
            data = await self._location.async_resolve()
        except Exception:  # noqa: BLE001
            _LOGGER.debug("AI Plugin: location resolve failed", exc_info=True)
            return ""

        if not data:
            return ""

        parts: list[str] = []
        if data.get("city"):
            parts.append(f"city={data['city']}")
        if data.get("region"):
            parts.append(f"region={data['region']}")
        if data.get("country_name"):
            parts.append(f"country={data['country_name']}")
        elif data.get("country_iso"):
            parts.append(f"country={data['country_iso']}")
        if "lat" in data and "lon" in data and "city" not in data:
            parts.append(f"coords={data['lat']:.4f},{data['lon']:.4f}")
        if data.get("timezone"):
            parts.append(f"timezone={data['timezone']}")

        if not parts:
            return ""

        joined = ", ".join(parts)
        return (
            "[HOME LOCATION]\n"
            f"- {joined}\n"
            "- For questions about the user's immediate area, set "
            "near_user=true on web_search. The plugin will scope the "
            "query to this area; do not invent a city name."
        )
