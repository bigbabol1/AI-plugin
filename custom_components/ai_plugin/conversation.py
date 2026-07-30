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

# Tail-echo window. The mic re-arms as TTS playback ends, so the fragment it
# catches is the reply's ENDING — but "when playback ends" depends on how long
# the reply is, and the timestamp we hold is from when it was GENERATED
# (synthesis and playback still have to happen). Allow a fixed slice for
# synthesis + STT finalisation plus the reply's own speaking time, capped so a
# stale reply can never justify dropping a turn. This rule fires below the
# bigram filter's minimum token count, where a fragment carries little
# evidence, so the bound stays as tight as playback physics allow.
_ECHO_TAIL_BASE_S = 5.0
_ECHO_TAIL_MAX_S = _ECHO_WINDOW_S
_TTS_WORDS_PER_S = 2.5
_ECHO_TAIL_TOKENS = 6

# Loop breaker. Garbled tails ("…areas and entities we have connected!" heard
# as "and I choose to be out connected") share too few word pairs with the
# reply for the bigram filter and are too long for the tail rule — no matcher
# can be trusted to catch them, so bound the chain instead. A turn is a chain
# link when it arrives while the previous reply could still have been playing;
# after this many consecutive links, stop offering a follow-up. A real
# dialogue that hits the cap just needs the wake word again.
_MAX_FOLLOW_UP_CHAIN = 4


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
    echo, but the agent knows what it just said — so we compare. The turn
    is echo when ECHO_MATCH_THRESHOLD of its word BIGRAMS appear in a
    recent reply. Order matters: an echo replays the reply's sequence,
    while a real command reusing the same vocabulary reorders it ("turn
    the living room light on" right after "The living room light is on"
    shares the words but not the adjacencies) — unordered overlap
    swallowed exactly those commands. STT is imperfect, so this is fuzzy
    containment, not equality. Short turns are never filtered
    (ECHO_MIN_TOKENS) so real follow-ups like "yes", "turn it off"
    always pass.
    """
    tokens = _echo_tokens(stt_text)
    if len(tokens) < ECHO_MIN_TOKENS:
        return False
    turn_bigrams = set(zip(tokens, tokens[1:]))
    if not turn_bigrams:
        return False
    for reply in recent_replies:
        r_tokens = _echo_tokens(reply)
        if len(r_tokens) < 2:
            continue
        reply_bigrams = set(zip(r_tokens, r_tokens[1:]))
        overlap = len(turn_bigrams & reply_bigrams) / len(turn_bigrams)
        if overlap >= ECHO_MATCH_THRESHOLD:
            return True
    return False


def _ordered_overlap(needle: list[str], hay: list[str]) -> int:
    """Length of the longest common subsequence (order-preserving) of tokens."""
    if not needle or not hay:
        return 0
    prev = [0] * (len(hay) + 1)
    for tok in needle:
        row = [0]
        for j, other in enumerate(hay):
            row.append(prev[j] + 1 if tok == other else max(prev[j + 1], row[j]))
        prev = row
    return prev[-1]


def _tail_echo_deadline(reply_tokens: int) -> float:
    """How long after generation a reply's tail can still be re-heard."""
    return min(
        _ECHO_TAIL_BASE_S + reply_tokens / _TTS_WORDS_PER_S, _ECHO_TAIL_MAX_S
    )


def _is_tail_echo(stt_text: str, recent_replies: list[tuple[float, str]]) -> bool:
    """True when a SHORT turn is the tail of a reply that just finished.

    The follow-up feedback loop nobody could kill: 'Listen for follow-up'
    re-arms the mic exactly as TTS playback ends, so it catches the last
    words of what was just spoken ("…all the lights in your home now!" →
    "home now"). Those fragments sit BELOW ECHO_MIN_TOKENS, so the bigram
    filter deliberately ignores them — they were run as fresh commands,
    answered, spoken, and the new reply's tail started the next round.

    ``recent_replies`` is [(age in seconds, reply text)]. Only the reply's
    last few words count, only while that reply could still be playing (or
    have just finished), and the fragment's words must appear IN ORDER (one
    may be lost to STT once the fragment is 3+ words). A single word is echo
    only when it is the reply's very last word, so one-word commands
    ("stop", "louder") still get through unless they are literally what was
    just said.
    """
    tokens = _echo_tokens(stt_text)
    if not tokens or len(tokens) >= ECHO_MIN_TOKENS:
        return False  # long turns are the bigram filter's job
    for age, reply in recent_replies:
        r_tokens = _echo_tokens(reply)
        if not r_tokens:
            continue
        if age > _tail_echo_deadline(len(r_tokens)):
            continue
        tail = r_tokens[-max(_ECHO_TAIL_TOKENS, 2 * len(tokens)):]
        if len(tokens) == 1:
            if tokens[0] == tail[-1]:
                return True
            continue
        needed = len(tokens) - (1 if len(tokens) >= 3 else 0)
        if _ordered_overlap(tokens, tail) >= needed:
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
        # Loop breaker state: device_id -> consecutive follow-up chain length.
        self._chain_turns: dict[str, int] = {}

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
        return [txt for _age, txt in self._recent_replies_with_age(device_id)]

    def _recent_replies_with_age(
        self, device_id: str | None
    ) -> list[tuple[float, str]]:
        """[(age in seconds, reply)] for the tail-echo check's own deadline."""
        if not device_id:
            return []
        now = time.monotonic()
        return [
            (now - ts, txt)
            for ts, txt in self._echo_store().get(device_id, [])
            if now - ts < _ECHO_WINDOW_S
        ]

    def _count_chain_turn(self, device_id: str | None) -> int:
        """Length of the follow-up chain this turn belongs to, itself included.

        A turn continues the chain when the previous reply could still have
        been playing as it arrived — the only window in which a TTS echo can
        exist. Anything later is a fresh interaction and resets the count, so
        only a genuine chain (a feedback loop, or a very fast dialogue) ever
        reaches the cap. Measuring against playback rather than a flat gap
        matters: the chatty replies that feed these loops take 20-30s to
        speak, so every link looked "slow" by wall clock.
        """
        if not device_id:
            return 1
        store = getattr(self, "_chain_turns", None)
        if store is None:
            store = {}
            self._chain_turns = store
        recent = self._recent_replies_with_age(device_id)
        linked = False
        if recent:
            age, reply = recent[-1]
            linked = age <= _tail_echo_deadline(len(_echo_tokens(reply)))
        count = store.get(device_id, 0) + 1 if linked else 1
        store[device_id] = count
        return count

    def _reset_chain_turns(self, device_id: str | None) -> None:
        store = getattr(self, "_chain_turns", None)
        if store and device_id:
            store.pop(device_id, None)

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
        is_echo = echo_filter and (
            _is_self_echo(
                user_input.text, self._recent_replies_for(user_input.device_id)
            )
            or _is_tail_echo(
                user_input.text,
                self._recent_replies_with_age(user_input.device_id),
            )
        )
        if is_echo:
            _LOGGER.info(
                "AI Plugin: dropped self-echo turn on device %s: %r",
                user_input.device_id, user_input.text[:80],
            )
        chain_len = self._count_chain_turn(user_input.device_id)

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
        if is_echo and continue_conversation:
            # Re-arming the mic after an echo just offers the reverb another
            # go — and a fragment we happen NOT to recognise restarts the
            # loop for real. End the session; the wake word is one word away.
            _LOGGER.info(
                "AI Plugin: self-echo on device %s — ending the session instead "
                "of listening again",
                user_input.device_id,
            )
            continue_conversation = False
        if continue_conversation and chain_len > _MAX_FOLLOW_UP_CHAIN:
            _LOGGER.warning(
                "AI Plugin: %d turns chained on device %s without a pause — "
                "ending the session to break a possible TTS feedback loop",
                chain_len, user_input.device_id,
            )
            continue_conversation = False
        if not continue_conversation:
            self._reset_chain_turns(user_input.device_id)
        return ConversationResult(
            response=intent_response,
            conversation_id=conversation_id,
            continue_conversation=continue_conversation,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Close provider sessions when the entity is removed."""
        await self._orchestrator.async_close()
