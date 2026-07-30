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
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    intent,
)
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util, ulid as ulid_util

from .const import (
    CONF_CONTINUE_CONVERSATION,
    CONF_FEEDBACK_LOOP_DEVICES,
    CONF_FOLLOW_UP_DELAY,
    CONF_SELF_ECHO_FILTER,
    CONVERSATION_CLOSE_PHRASES,
    DEFAULT_CONTINUE_CONVERSATION,
    DEFAULT_FOLLOW_UP_DELAY,
    DEFAULT_SELF_ECHO_FILTER,
    DOMAIN,
    ECHO_MATCH_THRESHOLD,
    ECHO_MIN_TOKENS,
)
from .exceptions import OrchestratorError
from .i18n import L
from .orchestrator import Orchestrator
from .shortcuts import looks_like_command

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

# Delayed follow-up. A satellite reopens its microphone the moment ITS OWN
# playback ends, which on a TTS-rerouted setup is a second before the room
# goes quiet — nothing in the conversation API can move that. So take the
# job away from it: end the turn (mic stays shut), wait for the speakers to
# actually stop, wait the configured quiet gap, then reopen the microphone
# with assist_satellite.start_conversation (empty message, no preannounce =
# silent listen). Requires the satellite to advertise START_CONVERSATION.
_SATELLITE_START_CONVERSATION_FEATURE = 2
_PLAYBACK_POLL_S = 0.5
_PLAYBACK_WAIT_CAP_S = 90.0
# A delayed follow-up arrives as a NEW HA conversation, so the history key is
# remapped back to the conversation it continues for this long.
_FOLLOW_UP_CONTINUITY_S = 120.0


# Playback-overlap echo. Satellites that mirror their TTS to other speakers
# re-arm the mic when their OWN playback ends — about a second before the
# room actually goes quiet — so the first thing recorded is the tail of the
# reply. Recorded on this install: satellite listening at 15:25:21, external
# speakers idle at 15:25:22, voice detected at 15:25:22, and "…like Trumbull
# County!" reached the agent as "tremble down". No text matcher can catch
# that — STT invented both words — but HA's own state says the microphone
# was open while a speaker was still playing.
# Grace covers capture + VAD silence + STT latency between the speaker going
# quiet and the turn reaching us. Measured on the three recorded echoes:
# 2.9s, 3.1s and 3.9s.
_PLAYBACK_ECHO_GRACE_S = 6.0
_PLAYBACK_START_SLACK_S = 1.0
_PLAYBACK_IDLE_STATES = frozenset({"idle", "paused", "standby", "off", "unavailable"})


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


def _speaker_was_playing(
    hass: HomeAssistant, device_id: str | None, reply_age: float
) -> bool:
    """True when a speaker in the caller's room was playing OUR reply.

    Only speakers OTHER than the calling device count: the satellite's own
    player is the one whose end re-arms the mic, and it also plays the
    wake-word chime, so including it would flag ordinary first turns.

    A player that is still playing only counts when it STARTED after we
    produced the reply (``reply_age`` seconds ago). Otherwise the living-room
    TV, or an hour-old music queue, would satisfy this for every follow-up in
    the room.
    """
    if not device_id or hass is None:
        return False
    try:
        dev_reg = dr.async_get(hass)
        ent_reg = er.async_get(hass)
        dev = dev_reg.async_get(device_id)
        area_id = dev.area_id if dev else None
        if not area_id:
            for entry in er.async_entries_for_device(ent_reg, device_id):
                if entry.area_id:
                    area_id = entry.area_id
                    break
        if not area_id:
            return False
        now = dt_util.utcnow()
        for eid, entry in ent_reg.entities.items():
            if not eid.startswith("media_player.") or entry.device_id == device_id:
                continue
            entry_area = entry.area_id
            if not entry_area and entry.device_id:
                other = dev_reg.async_get(entry.device_id)
                entry_area = other.area_id if other else None
            if entry_area != area_id:
                continue
            state = hass.states.get(eid)
            if state is None:
                continue
            since_change = (now - state.last_changed).total_seconds()
            if state.state == "playing":
                # Started after our reply → it is our TTS, not the TV.
                if since_change <= reply_age + _PLAYBACK_START_SLACK_S:
                    return True
                continue
            if (
                state.state in _PLAYBACK_IDLE_STATES
                and since_change <= _PLAYBACK_ECHO_GRACE_S
            ):
                return True
    except Exception:  # noqa: BLE001 — never let this block a real turn
        _LOGGER.debug("playback-overlap check failed", exc_info=True)
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
        # Delayed follow-up: pending reopen task and the conversation it
        # continues, per device.
        self._follow_up_tasks: dict[str, asyncio.Task] = {}
        self._follow_up_convs: dict[str, tuple[float, str]] = {}

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
        count = store.get(device_id, 0) + 1 if self._is_chain_link(device_id) else 1
        store[device_id] = count
        return count

    def _last_reply_age(self, device_id: str | None) -> float | None:
        """Age of our most recent reply on this device, if it could still play."""
        recent = self._recent_replies_with_age(device_id)
        if not recent:
            return None
        age, reply = recent[-1]
        if age > _tail_echo_deadline(len(_echo_tokens(reply))):
            return None
        return age

    def _is_chain_link(self, device_id: str | None) -> bool:
        """True when our previous reply could still have been playing."""
        return self._last_reply_age(device_id) is not None

    def _reset_chain_turns(self, device_id: str | None) -> None:
        store = getattr(self, "_chain_turns", None)
        if store and device_id:
            store.pop(device_id, None)

    def _satellite_for_device(self, device_id: str | None) -> str | None:
        """assist_satellite entity of the calling device, if it can be reopened."""
        hass = getattr(self, "hass", None)
        if not device_id or hass is None:
            return None
        try:
            for entry in er.async_entries_for_device(er.async_get(hass), device_id):
                if not entry.entity_id.startswith("assist_satellite."):
                    continue
                state = hass.states.get(entry.entity_id)
                features = (
                    state.attributes.get("supported_features", 0) if state else 0
                )
                if features & _SATELLITE_START_CONVERSATION_FEATURE:
                    return entry.entity_id
        except Exception:  # noqa: BLE001
            _LOGGER.debug("satellite lookup failed", exc_info=True)
        return None

    async def _reopen_after_quiet(
        self, device_id: str, satellite: str, reply: str, delay: float
    ) -> None:
        """Wait for our own TTS to finish, pause, then reopen the microphone."""
        hass = self.hass
        try:
            # Playback has not even started yet when we get here; hold until
            # the estimated end AND until every speaker in the room is quiet,
            # so a slow external speaker can't be cut short.
            spoken_for = len(_echo_tokens(reply)) / _TTS_WORDS_PER_S
            waited = 0.0
            await asyncio.sleep(min(spoken_for + 1.0, _PLAYBACK_WAIT_CAP_S))
            while waited < _PLAYBACK_WAIT_CAP_S:
                if not _speaker_was_playing(hass, device_id, _PLAYBACK_WAIT_CAP_S):
                    break
                await asyncio.sleep(_PLAYBACK_POLL_S)
                waited += _PLAYBACK_POLL_S
            await asyncio.sleep(delay)

            state = hass.states.get(satellite)
            if state is not None and state.state != "idle":
                _LOGGER.debug(
                    "AI Plugin: %s is %s — not reopening the microphone",
                    satellite, state.state,
                )
                return
            _LOGGER.info(
                "AI Plugin: reopening %s for a follow-up (%.1fs after the room "
                "went quiet)", satellite, delay,
            )
            await hass.services.async_call(
                "assist_satellite",
                "start_conversation",
                {
                    "entity_id": satellite,
                    "start_message": "",
                    "preannounce": False,
                },
                blocking=False,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _LOGGER.warning(
                "AI Plugin: could not reopen %s for a follow-up", satellite,
                exc_info=True,
            )

    def _schedule_follow_up(
        self, device_id: str, satellite: str, reply: str, delay: float,
        conversation_id: str,
    ) -> None:
        """Replace the satellite's own re-arm with our delayed one."""
        self._cancel_follow_up(device_id)
        task = self.hass.async_create_task(
            self._reopen_after_quiet(device_id, satellite, reply, delay)
        )
        self._follow_up_store()[device_id] = task
        self._follow_up_conv_store()[device_id] = (time.monotonic(), conversation_id)

    def _follow_up_store(self) -> dict[str, asyncio.Task]:
        store = getattr(self, "_follow_up_tasks", None)
        if store is None:
            store = {}
            self._follow_up_tasks = store
        return store

    def _follow_up_conv_store(self) -> dict[str, tuple[float, str]]:
        store = getattr(self, "_follow_up_convs", None)
        if store is None:
            store = {}
            self._follow_up_convs = store
        return store

    def _cancel_follow_up(self, device_id: str | None) -> None:
        task = self._follow_up_store().pop(device_id, None) if device_id else None
        if task is not None and not task.done():
            task.cancel()

    def _continued_conversation_id(
        self, device_id: str | None, conversation_id: str
    ) -> str:
        """Map a delayed follow-up back onto the conversation it continues.

        Reopening the microphone starts a fresh HA conversation, so without
        this the follow-up would lose every turn of context — "and in the
        bedroom?" would arrive with nothing to attach to.
        """
        if not device_id:
            return conversation_id
        pending = self._follow_up_conv_store().get(device_id)
        if not pending:
            return conversation_id
        started, prior = pending
        if time.monotonic() - started > _FOLLOW_UP_CONTINUITY_S:
            self._follow_up_conv_store().pop(device_id, None)
            return conversation_id
        if prior != conversation_id:
            _LOGGER.debug(
                "AI Plugin: delayed follow-up continues conversation %s", prior
            )
        return prior

    def _is_playback_overlap_echo(self, user_input: ConversationInput) -> bool:
        """True when a short, meaningless turn was recorded over our own TTS.

        Last line of defence, for tails the text matchers cannot see: STT
        substitutes words wholesale ("Trumbull County" → "tremble down"), so
        there is nothing left to compare. Four conditions have to hold at
        once — the turn is one to three words, our own reply could still have
        been playing, a speaker in the room was playing (or had just
        stopped), and the words are not something the user could plausibly
        have said. A recognisable command or a bare "yes" is never dropped
        here, however the timing looks.
        """
        text = user_input.text or ""
        tokens = _echo_tokens(text)
        if not tokens or len(tokens) >= ECHO_MIN_TOKENS:
            return False
        reply_age = self._last_reply_age(user_input.device_id)
        if reply_age is None:
            return False
        lang = (user_input.language or "en").split("-")[0].lower()
        if looks_like_command(text, lang):
            return False
        if not _speaker_was_playing(
            getattr(self, "hass", None), user_input.device_id, reply_age
        ):
            return False
        _LOGGER.info(
            "AI Plugin: %r arrived while a speaker in the room was still "
            "playing and means nothing on its own — treating as TTS echo",
            text[:60],
        )
        return True

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
        # A turn arrived, so any microphone reopen we still had pending for
        # this device is moot — the user (or an echo) got there first.
        self._cancel_follow_up(user_input.device_id)
        # A delayed follow-up arrives as a brand-new HA conversation; keep
        # writing history under the conversation it continues.
        history_id = self._continued_conversation_id(
            user_input.device_id, conversation_id
        )
        _LOGGER.info(
            "AI Plugin: conv_id=%s user_id=%s input=%r",
            history_id, user_id, user_input.text[:80],
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
            or self._is_playback_overlap_echo(user_input)
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
                    conversation_id=history_id,
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
            self._follow_up_conv_store().pop(user_input.device_id or "", None)
        else:
            # Delayed follow-up: take the re-arm away from the satellite so
            # the microphone opens only once the room is genuinely quiet.
            delay = float(
                self._entry.options.get(CONF_FOLLOW_UP_DELAY, DEFAULT_FOLLOW_UP_DELAY)
                or 0.0
            )
            satellite = (
                self._satellite_for_device(user_input.device_id)
                if delay > 0 and reply.strip()
                else None
            )
            if satellite:
                self._schedule_follow_up(
                    user_input.device_id, satellite, reply, delay, history_id
                )
                continue_conversation = False
        return ConversationResult(
            response=intent_response,
            conversation_id=conversation_id,
            continue_conversation=continue_conversation,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Close provider sessions when the entity is removed."""
        for device_id in list(getattr(self, "_follow_up_tasks", {})):
            self._cancel_follow_up(device_id)
        await self._orchestrator.async_close()
