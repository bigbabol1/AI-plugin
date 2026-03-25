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

import logging
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CONTEXT_WINDOW,
    CONF_MODEL,
    CONF_RESPONSE_TIMEOUT,
    CONF_SUMMARIZATION_ENABLED,
    CONF_SYSTEM_PROMPT,
    CONF_VOICE_MODE,
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_RESPONSE_TIMEOUT,
    DEFAULT_SUMMARIZATION_ENABLED,
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_VOICE,
)
from .context_manager import ContextManager
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
        opts = entry.options
        self._context_mgr = ContextManager(
            max_tokens=opts.get(CONF_CONTEXT_WINDOW, DEFAULT_CONTEXT_WINDOW),
            # Week 3: MCP client will supply actual schema token count here.
            tool_token_budget=2000,
        )
        self._summarization_enabled: bool = opts.get(
            CONF_SUMMARIZATION_ENABLED, DEFAULT_SUMMARIZATION_ENABLED
        )

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

        # 1. Add user turn to history.
        await self._context_mgr.add_turn(conversation_id, "user", message)

        # 2. Summarize old turns if we're approaching the soft token limit.
        if self._summarization_enabled:
            await self._context_mgr.summarize_if_needed(
                conversation_id, system_prompt, self._provider
            )

        # 3. Build the message list (system prompt + trimmed history).
        messages = await self._context_mgr.get_messages(conversation_id, system_prompt)

        _LOGGER.debug(
            "Processing message for conv_id=%s lang=%s voice=%s msg_count=%d",
            conversation_id,
            language,
            voice_mode,
            len(messages),
        )

        # 4. Call the LLM.
        reply = await self._provider.async_complete(messages)

        # 5. Append assistant reply to history.
        await self._context_mgr.add_turn(conversation_id, "assistant", reply)

        return reply

    async def async_close(self) -> None:
        """Close provider sessions. Called on integration unload."""
        await self._provider.async_close()
