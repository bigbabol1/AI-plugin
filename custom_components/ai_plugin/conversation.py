"""AI Plugin conversation entity — integrates with HA Assist pipeline.

This is the platform file for Platform.CONVERSATION. HA calls
async_setup_entry with async_add_entities, and we register one
AIPluginConversationEntity per config entry.

┌──────────────────────────────────────────────────────────────┐
│  HA Assist Pipeline                                           │
│       │ ConversationInput (text, conv_id, language, ...)     │
│       ▼                                                      │
│  AIPluginConversationEntity.async_process()                     │
│       │ delegates to Orchestrator.async_process()            │
│       ▼                                                      │
│  ConversationResult (IntentResponse with speech)             │
│                                                              │
│  Error path:                                                 │
│  OrchestratorError → caught here → graceful error speech    │
└──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationInput, ConversationResult
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util.ulid import ulid_now

from .const import DOMAIN
from .exceptions import OrchestratorError
from .orchestrator import Orchestrator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the AI Plugin conversation entity from a config entry."""
    entity = AIPluginConversationEntity(hass, config_entry)
    async_add_entities([entity])


class AIPluginConversationEntity(conversation.ConversationEntity):
    """Conversation entity that routes HA Assist through AI Plugin."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_languages = "*"

    @property
    def supported_languages(self) -> list[str] | str:
        """Return supported languages — '*' means all languages."""
        return "*"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._orchestrator = Orchestrator(hass, entry)
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
        )

    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process a user message from HA Assist and return a reply."""
        user_id: str | None = (
            getattr(user_input.context, "user_id", None) if user_input.context else None
        )
        # Prefer HA's session conversation_id. Fall back to a stable per-user ID
        # so context survives Assist panel close/reopen between commands.
        if user_input.conversation_id:
            conversation_id = user_input.conversation_id
        elif user_id:
            conversation_id = f"ai_plugin_user_{user_id}"
        else:
            conversation_id = f"ai_plugin_{self._entry.entry_id}"

        try:
            reply = await self._orchestrator.async_process(
                message=user_input.text,
                conversation_id=conversation_id,
                language=user_input.language,
                device_id=user_input.device_id,
                user_id=user_id,
            )
        except OrchestratorError as exc:
            _LOGGER.error("AI Plugin error processing message: %s", exc)
            reply = f"Sorry, I couldn't process that. ({exc})"

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(reply)
        return ConversationResult(
            response=intent_response,
            conversation_id=conversation_id,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Close provider sessions when the entity is removed."""
        await self._orchestrator.async_close()
