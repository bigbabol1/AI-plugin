"""AI Plugin conversation entity — integrates with HA Assist pipeline."""

from __future__ import annotations

import asyncio
import logging
import re
import time

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
    CONF_FEEDBACK_LOOP_DEVICES,
    CONF_SELF_ECHO_FILTER,
    CONVERSATION_CLOSE_PHRASES,
    DEFAULT_CONTINUE_CONVERSATION,
    DEFAULT_SELF_ECHO_FILTER,
    DOMAIN,
    ECHO_MATCH_THRESHOLD,
    ECHO_MIN_TOKENS,
)
from .exceptions import OrchestratorError
from .i18n import L
from .orchestrator import Orchestrator

_LOGGER = logging.getLogger(__name__)


# Collapse runs of non-word chars (punctuation, multiple spaces) into a single
# space so phrase matching survives STT trailing periods, commas, and so on.
# Unicode flag keeps umlauts (ü, ö, ä, ß) inside word characters.
_WORD_BREAK_RE = re.compile(r"\W+", re.UNICODE)

# Self-echo filter window: replies older than this can't be the STT echo
# currently arriving, and at most this many replies are kept per device.
_ECHO_WINDOW_S = 30.0
_ECHO_HISTORY = 3


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


def _echo_tokens(text: str) -> list[str]:
    """Lowercase word tokens for echo comparison (Unicode-aware)."""
    return _WORD_BREAK_RE.sub(" ", (text or "").lower()).split()


def _is_self_echo(stt_text: str, recent_replies: list[str]) -> bool:
    """True when the STT input looks like the agent's own TTS fed back.

    A satellite whose reply audio plays through a separate speaker re-hears
    it and STT transcribes it as a fresh turn. We can't cancel that acoustic
    echo, but the agent knows what it just said — so we compare. The turn is
    echo when ECHO_MATCH_THRESHOLD of its tokens appear (in order) within a
    recent reply. STT is imperfect, so this is fuzzy containment, not
    equality. Short turns are never filtered (ECHO_MIN_TOKENS) so real
    follow-ups like "yes", "turn it off" always pass.
    """
    tokens = _echo_tokens(stt_text)
    if len(tokens) < ECHO_MIN_TOKENS:
        return False
    token_set = set(tokens)
    for reply in recent_replies:
        reply_set = set(_echo_tokens(reply))
        if not reply_set:
            continue
        overlap = len(token_set & reply_set) / len(token_set)
        if overlap >= ECHO_MATCH_THRESHOLD:
            return True
    return False


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
    # The assist pipeline only taps chat-log deltas (early TTS) when the
    # agent declares streaming support.
    _attr_supports_streaming = True

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
        # Per-device recent replies for the self-echo filter:
        # device_id -> list of (monotonic_ts, reply_text). Only replies
        # from the last _ECHO_WINDOW_S can be echo, so the list stays tiny.
        self._recent_replies: dict[str, list[tuple[float, str]]] = {}

    def _echo_store(self) -> dict[str, list[tuple[float, str]]]:
        store = getattr(self, "_recent_replies", None)
        if store is None:
            store = {}
            self._recent_replies = store
        return store

    def _remember_reply(self, device_id: str | None, reply: str) -> None:
        if not device_id or not reply.strip():
            return
        store = self._echo_store()
        now = time.monotonic()
        history = [
            (ts, txt)
            for ts, txt in store.get(device_id, [])
            if now - ts < _ECHO_WINDOW_S
        ]
        history.append((now, reply))
        store[device_id] = history[-_ECHO_HISTORY:]

    def _recent_replies_for(self, device_id: str | None) -> list[str]:
        if not device_id:
            return []
        now = time.monotonic()
        return [
            txt
            for ts, txt in self._echo_store().get(device_id, [])
            if now - ts < _ECHO_WINDOW_S
        ]

    async def _async_handle_message(
        self, user_input: ConversationInput, chat_log
    ) -> ConversationResult:
        """Process a user message from HA Assist and return a reply.

        Called by the base ConversationEntity.async_process inside HA's
        chat-session context. Streaming: orchestrator deltas flow through a
        queue into chat_log.async_add_delta_content_stream so the voice
        pipeline can start TTS before generation finishes. The consumer
        task starts lazily on the first delta — turns that never stream
        (shortcuts, suppressed actuations) add no empty chat-log entry.
        """
        user_id: str | None = (
            getattr(user_input.context, "user_id", None) if user_input.context else None
        )
        # Honour HA's session conversation_id so history resets when the
        # Assist panel is reopened or a new session begins. Long-term memory
        # lives in the remember/recall tool, not in conversation history.
        conversation_id = (
            getattr(chat_log, "conversation_id", None)
            or user_input.conversation_id
            or ulid_util.ulid_now()
        )
        _LOGGER.info(
            "AI Plugin: conv_id=%s user_id=%s input=%r",
            conversation_id, user_id, user_input.text[:80],
        )

        # Self-echo filter: if this "user" turn is really the satellite's
        # own TTS fed back through a separate speaker, drop it — do NOT run
        # it as a command, and speak nothing. We keep the session open
        # (normal follow-up rules below): the dropped turn produces no new
        # audio, so the loop dies on its own while a real follow-up still
        # lands. Recent replies are NOT cleared, so reverb/multi-fragment
        # echoes of the same reply are caught too.
        echo_filter = self._entry.options.get(
            CONF_SELF_ECHO_FILTER, DEFAULT_SELF_ECHO_FILTER
        )
        is_echo = echo_filter and _is_self_echo(
            user_input.text, self._recent_replies_for(user_input.device_id)
        )
        if is_echo:
            _LOGGER.info(
                "AI Plugin: dropped self-echo turn on device %s: %r",
                user_input.device_id, user_input.text[:80],
            )

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        consumer: asyncio.Task | None = None
        agent_id = getattr(self, "entity_id", None) or DOMAIN

        async def _deltas():
            yield {"role": "assistant"}
            while (item := await queue.get()) is not None:
                yield {"content": item}

        async def _consume() -> None:
            try:
                async for _ in chat_log.async_add_delta_content_stream(
                    agent_id, _deltas()
                ):
                    pass
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin: chat-log delta stream failed", exc_info=True)

        def _on_delta(text: str) -> None:
            nonlocal consumer
            queue.put_nowait(text)
            if consumer is None:
                consumer = asyncio.get_running_loop().create_task(_consume())

        stream_capable = hasattr(chat_log, "async_add_delta_content_stream")
        if is_echo:
            # Skip the LLM entirely; nothing to say, nothing to record.
            reply = ""
        else:
            try:
                reply = await self._orchestrator.async_process(
                    message=user_input.text,
                    conversation_id=conversation_id,
                    language=user_input.language,
                    device_id=user_input.device_id,
                    user_id=user_id,
                    on_delta=_on_delta if stream_capable else None,
                )
            except OrchestratorError as exc:
                _LOGGER.error("AI Plugin error processing message: %s", exc)
                _lang = (user_input.language or "en").split("-")[0].lower()
                reply = f"{L.template('err_process', _lang)} ({exc})"
            finally:
                if consumer is not None:
                    queue.put_nowait(None)
                    try:
                        await consumer
                    except Exception:  # noqa: BLE001
                        _LOGGER.debug(
                            "AI Plugin: delta consumer failed", exc_info=True
                        )

            # Record this reply so the next turn can recognise it as echo.
            self._remember_reply(user_input.device_id, reply)

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(reply)
        continue_conversation = self._entry.options.get(
            CONF_CONTINUE_CONVERSATION, DEFAULT_CONTINUE_CONVERSATION
        )
        # Satellites whose TTS is routed to a separate speaker (e.g. via
        # Mic to MediaPlayer) create an acoustic feedback loop: the mic
        # re-hears the spoken reply and re-recognises it as a new turn.
        # Until v0.9.31 this force-ended EVERY voice-satellite turn, which
        # silently disabled 'Listen for follow-up' on normal satellites
        # (own speaker + echo suppression, e.g. HA Voice PE). The guard is
        # now an explicit per-device list; everything else honours the
        # global toggle.
        feedback_devices = self._entry.options.get(CONF_FEEDBACK_LOOP_DEVICES) or []
        if user_input.device_id and user_input.device_id in feedback_devices:
            if continue_conversation:
                _LOGGER.debug(
                    "AI Plugin: device %s is in the feedback-loop guard list — "
                    "forcing continue_conversation=False",
                    user_input.device_id,
                )
            continue_conversation = False
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
