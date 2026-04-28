"""AI Plugin conversation entity — integrates with HA Assist pipeline."""

from __future__ import annotations

import logging
import re

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import ulid as ulid_util

from .const import (
    CONF_CONTINUE_CONVERSATION,
    CONVERSATION_CLOSE_PHRASES,
    DEFAULT_CONTINUE_CONVERSATION,
    DOMAIN,
)
from .exceptions import OrchestratorError
from .orchestrator import Orchestrator

_LOGGER = logging.getLogger(__name__)


# Collapse runs of non-word chars (punctuation, multiple spaces) into a single
# space so phrase matching survives STT trailing periods, commas, and so on.
# Unicode flag keeps umlauts (ü, ö, ä, ß) inside word characters.
_WORD_BREAK_RE = re.compile(r"\W+", re.UNICODE)


def _match_close_phrase(text: str | None) -> str | None:
    """Return the matched close phrase, or None.

    Matches a phrase if it appears as a word-bounded fragment in the STT
    text. Punctuation is normalised to whitespace before matching so that
    "Thanks Jarvis." and "thanks, Jarvis" both still match "thanks jarvis".
    Avoids spurious matches like "stop" inside "stopwatch".
    """
    if not text:
        return None
    normalized = f" {_WORD_BREAK_RE.sub(' ', text.lower()).strip()} "
    for phrase in CONVERSATION_CLOSE_PHRASES:
        norm_phrase = f" {_WORD_BREAK_RE.sub(' ', phrase.lower()).strip()} "
        if norm_phrase in normalized:
            return phrase
    return None


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
    _attr_supported_features = ConversationEntityFeature.CONTROL

    @property
    def supported_languages(self) -> list[str] | str:
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
        # Honour HA's ephemeral conversation_id so history resets when the
        # Assist panel is reopened or a new session begins. Follow-up turns
        # within the same session reuse the id HA echoes back. Long-term
        # memory lives in the remember/recall tool (.ai_plugin_memory_*.json),
        # not in conversation history.
        conversation_id = user_input.conversation_id or ulid_util.ulid_now()
        _LOGGER.info(
            "AI Plugin: conv_id=%s user_id=%s input=%r",
            conversation_id, user_id, user_input.text[:80],
        )

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
        continue_conversation = self._entry.options.get(
            CONF_CONTINUE_CONVERSATION, DEFAULT_CONTINUE_CONVERSATION
        )
        close_match = _match_close_phrase(user_input.text)
        if close_match is not None:
            _LOGGER.info(
                "AI Plugin: close phrase '%s' detected in %r — ending conversation",
                close_match, user_input.text[:80],
            )
            continue_conversation = False
        return ConversationResult(
            response=intent_response,
            conversation_id=conversation_id,
            continue_conversation=continue_conversation,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Close provider sessions when the entity is removed."""
        await self._orchestrator.async_close()
