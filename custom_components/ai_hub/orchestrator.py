"""Orchestrator: the core message-processing engine for AI Hub.

Week 1 scope: receive message → manage in-memory history → call LLM →
return reply. Context management (summarization) and tool routing ship
in Weeks 2 and 3.

┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator.process(message, conv_id, language, device_id)    │
│                                                                 │
│  1. Determine system prompt (voice mode vs standard)            │
│  2. Append user message to in-memory history (asyncio.Lock)     │
│  3. Build message list: [system] + history                      │
│  4. Call provider.async_complete(messages)                      │
│  5. Append assistant reply to history                           │
│  6. Return reply text                                           │
│                                                                 │
│  Error: OrchestratorError propagates to conversation.py which   │
│  catches it and returns a graceful ConversationResult.          │
└─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_RESPONSE_TIMEOUT,
    CONF_SYSTEM_PROMPT,
    CONF_VOICE_MODE,
    DEFAULT_BASE_URL,
    DEFAULT_RESPONSE_TIMEOUT,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
)
from .exceptions import OrchestratorError
from .providers.openai_compat import OpenAICompatProvider

if TYPE_CHECKING:
    from .providers.base import AbstractProvider

_LOGGER = logging.getLogger(__name__)


class Orchestrator:
    """Processes messages: history → system prompt → LLM → reply.

    One Orchestrator instance per config entry. Created at platform
    setup; replaced on options change via full entry reload.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self._hass = hass
        self._entry = entry
        self._provider: AbstractProvider = self._build_provider()
        # In-memory history: conv_id → list of message dicts
        # Week 2: replaced by ContextManager with sliding window + summarization
        self._history: dict[str, list[dict[str, str]]] = {}
        self._lock = asyncio.Lock()

    def _build_provider(self) -> AbstractProvider:
        """Build provider from current entry options."""
        opts = self._entry.options
        return OpenAICompatProvider(
            base_url=opts.get(CONF_BASE_URL, DEFAULT_BASE_URL),
            model=opts.get(CONF_MODEL, ""),
            api_key=opts.get(CONF_API_KEY) or None,
            timeout=opts.get(CONF_RESPONSE_TIMEOUT, DEFAULT_RESPONSE_TIMEOUT),
        )

    def _build_system_prompt(self, voice_mode: bool) -> str:
        """Return the system prompt for this request.

        Priority:
        1. User-configured custom instructions (if non-empty)
        2. Voice-mode compact prompt (if voice_mode=True)
        3. Default standard prompt
        """
        custom = self._entry.options.get(CONF_SYSTEM_PROMPT, "").strip()
        if custom:
            return custom
        if voice_mode:
            return SYSTEM_PROMPT_VOICE
        return SYSTEM_PROMPT_DEFAULT

    async def async_process(
        self,
        message: str,
        conversation_id: str,
        language: str,
        device_id: str | None = None,
    ) -> str:
        """Process a user message and return the assistant's reply.

        Args:
            message: The user's text input.
            conversation_id: Stable ID for this conversation thread.
            language: BCP-47 language code (passed through from HA Assist).
            device_id: Optional device ID (used for voice mode detection
                       in future; currently unused in Week 1).

        Returns:
            The assistant's reply text.

        Raises:
            OrchestratorError: on provider failure.
        """
        # Heuristic: if a device_id is present the request came from a voice
        # assistant pipeline (Assist mic → media player → device).  We also
        # honour the explicit CONF_VOICE_MODE config toggle as a fallback for
        # setups where the device_id is unavailable but the user always wants
        # the compact voice prompt.  Note: HA does not expose a dedicated
        # is_voice flag on ConversationInput (Context is not a dict), so
        # device_id presence is the closest reliable signal available.
        voice_mode: bool = (device_id is not None) or bool(
            self._entry.options.get(CONF_VOICE_MODE, False)
        )
        system_prompt = self._build_system_prompt(voice_mode)

        async with self._lock:
            history = self._history.setdefault(conversation_id, [])
            history.append({"role": "user", "content": message})
            # Snapshot for the LLM call (outside the lock)
            messages = [{"role": "system", "content": system_prompt}, *history]

        _LOGGER.debug(
            "Processing message for conv_id=%s lang=%s voice=%s history_len=%d",
            conversation_id,
            language,
            voice_mode,
            len(history),
        )

        reply = await self._provider.async_complete(messages)

        async with self._lock:
            # Re-fetch in case another coroutine touched history while we awaited
            conv_history = self._history.setdefault(conversation_id, [])
            # Only append if our user message is still the last one (guard against
            # duplicate calls on the same conv_id)
            if conv_history and conv_history[-1] == {"role": "user", "content": message}:
                conv_history.append({"role": "assistant", "content": reply})

        return reply

    async def async_close(self) -> None:
        """Close provider sessions. Called on integration unload."""
        await self._provider.async_close()
