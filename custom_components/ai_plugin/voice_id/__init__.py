"""Voice recognition: who is speaking on a voice turn.

One ``VoiceIDManager`` per config entry, kept in
``hass.data[DOMAIN][entry_id]["voice_id"]``. It owns the audio buffers the
STT pass-through fills, the enrolled fingerprints (HA Store, never audio),
the lazily loaded model and the enrollment sessions.

Fingerprints are computed only when the orchestrator asks: on turns that go
to the LLM, and on enrollment answers. Turns that HA or the plugin's own
shortcuts handle never touch the model; their audio just expires.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..const import (
    CONF_VOICE_ID_ENABLED,
    CONF_VOICE_ID_MARGIN,
    CONF_VOICE_ID_THRESHOLD,
    CONF_VOICE_ID_VOICE_ENROLL,
    DEFAULT_VOICE_ID_ENABLED,
    DEFAULT_VOICE_ID_VOICE_ENROLL,
    VOICE_ID_WAIT_SECONDS,
)
from .buffer import AudioBufferStore
from .enrollment import EnrollmentSessions, is_stop, prompts_for

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
MAX_SAMPLES_PER_MIC = 40
NAME_MATCH_MIN = 0.6
NAME_MATCH_LEAD = 0.2
# Opening the microphone after the first enrollment question was announced.
_SPEAKER_BUSY_STATES = {"playing", "buffering"}
_SPEAKER_POLL_S = 0.2
_SPEAKER_WAIT_CAP_S = 30.0
_LISTEN_SETTLE_S = 0.3
_ENROLL_REASKS = 2
_ANSWER_WAIT_CAP_S = 30.0
_MIC_START_GRACE_S = 3.0
_TURN_ARRIVAL_S = 2.0

VOICE_ENROLL_TOOL = "start_voice_enrollment"
VOICE_ENROLL_SCHEMA = {
    "type": "function",
    "function": {
        "name": VOICE_ENROLL_TOOL,
        "description": (
            "Start teaching the assistant to recognise the speaker's voice by asking "
            "them a few questions. Call it whenever the user wants you to learn, "
            "recognise or take samples of their voice. Call it straight away, also "
            "without a name: the tool says what to ask if it needs one. Its result "
            "is what to say; never claim it started unless the result says so."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "person": {
                    "type": "string",
                    "description": "The speaker's name, if they said it.",
                }
            },
        },
    },
}


@dataclass(frozen=True)
class Identity:
    """Who spoke a turn. ``person_id`` is a person entity id."""

    person_id: str
    name: str
    memory_key: str
    score: float


class VoiceIDManager:
    """Per-entry voice recognition state."""

    def __init__(self, hass, entry, store=None) -> None:
        self._hass = hass
        self._entry = entry
        self.buffers = AudioBufferStore()
        # Enrollment sessions outlive this manager: saving any option reloads
        # the entry, and a session armed a moment earlier must survive that.
        shared = getattr(hass, "data", None)
        if isinstance(shared, dict):
            self.sessions = shared.setdefault("ai_plugin_voice_sessions", {}).setdefault(
                entry.entry_id, EnrollmentSessions()
            )
        else:
            self.sessions = EnrollmentSessions()
        self._store = store
        self._data: dict[str, Any] = {"profiles": {}, "pipelines": {}}
        self._embedder = None
        self._model_lock = asyncio.Lock()
        # One fingerprint at a time: two rooms at once must not double the
        # memory peak on a small board.
        self._embed_lock = asyncio.Lock()
        self.last_error: str | None = None

    # ── settings ────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return bool(self._entry.options.get(CONF_VOICE_ID_ENABLED, DEFAULT_VOICE_ID_ENABLED))

    @property
    def voice_enroll_allowed(self) -> bool:
        return bool(
            self._entry.options.get(CONF_VOICE_ID_VOICE_ENROLL, DEFAULT_VOICE_ID_VOICE_ENROLL)
        )

    def _thresholds(self) -> tuple[float, float]:
        from .matching import DEFAULT_MARGIN, DEFAULT_THRESHOLD  # noqa: PLC0415

        opts = self._entry.options
        return (
            float(opts.get(CONF_VOICE_ID_THRESHOLD, DEFAULT_THRESHOLD)),
            float(opts.get(CONF_VOICE_ID_MARGIN, DEFAULT_MARGIN)),
        )

    # ── storage ─────────────────────────────────────────────────────────────

    async def async_load(self) -> None:
        if self._store is None:
            from homeassistant.helpers.storage import Store  # noqa: PLC0415

            self._store = Store(
                self._hass, STORAGE_VERSION, f"ai_plugin_voice.{self._entry.entry_id}"
            )
        data = await self._store.async_load()
        if isinstance(data, dict):
            self._data = {
                "profiles": dict(data.get("profiles") or {}),
                "pipelines": dict(data.get("pipelines") or {}),
            }

    async def async_remove_storage(self) -> None:
        if self._store is not None:
            await self._store.async_remove()

    async def _async_save(self) -> None:
        if self._store is not None:
            await self._store.async_save(self._data)

    @property
    def pipeline_originals(self) -> dict[str, str]:
        """pipeline_id → the STT engine it used before we switched it."""
        return self._data["pipelines"]

    async def async_set_pipeline_originals(self, mapping: dict[str, str]) -> None:
        self._data["pipelines"] = dict(mapping)
        await self._async_save()

    def has_profiles(self) -> bool:
        return any(self._data["profiles"].values())

    def profile_summary(self) -> dict[str, dict[str, int]]:
        """person_id → {mic_group: sample count}, for the settings screen."""
        out: dict[str, dict[str, int]] = {}
        for pid, samples in self._data["profiles"].items():
            counts: dict[str, int] = {}
            for s in samples:
                counts[s.get("mic_group", "")] = counts.get(s.get("mic_group", ""), 0) + 1
            if counts:
                out[pid] = counts
        return out

    async def async_delete_profile(self, person_id: str) -> None:
        if self._data["profiles"].pop(person_id, None) is not None:
            await self._async_save()

    async def _async_add_sample(self, person_id: str, vec, device_id: str) -> None:
        group = self.mic_group(device_id)
        samples = self._data["profiles"].setdefault(person_id, [])
        samples.append(
            {
                "vec": [round(float(v), 5) for v in vec],
                "device_id": device_id,
                "mic_group": group,
                "ts": int(time.time()),
            }
        )
        same = [s for s in samples if s.get("mic_group") == group]
        if len(same) > MAX_SAMPLES_PER_MIC:
            oldest = min(same, key=lambda s: s.get("ts", 0))
            samples.remove(oldest)
        await self._async_save()

    # ── HA lookups ──────────────────────────────────────────────────────────

    def mic_group(self, device_id: str | None) -> str:
        """Satellites of the same model share samples: same mic, same sound."""
        if not device_id:
            return ""
        try:
            from homeassistant.helpers import device_registry as dr  # noqa: PLC0415

            dev = dr.async_get(self._hass).async_get(device_id)
        except Exception:  # noqa: BLE001
            dev = None
        if dev is None:
            return device_id
        model = " ".join(str(p) for p in (dev.manufacturer, dev.model) if p)
        return model or device_id

    def persons(self) -> dict[str, dict[str, str | None]]:
        """person_id → {"name", "user_id"} from HA's person entities."""
        out: dict[str, dict[str, str | None]] = {}
        for state in self._hass.states.async_all("person"):
            out[state.entity_id] = {
                "name": state.attributes.get("friendly_name") or state.entity_id,
                "user_id": state.attributes.get("user_id"),
            }
        return out

    def memory_key(self, person_id: str) -> str:
        """Memory file key: the person's HA user, so app chat and voice share
        one memory; a person without an HA login gets its own key."""
        info = self.persons().get(person_id) or {}
        if info.get("user_id"):
            return str(info["user_id"])
        return "person_" + person_id.split(".", 1)[-1]

    def find_person(self, name: str) -> str | None:
        """Match a spoken name to exactly one person entity, else None.

        Speech-to-text mangles names ("Arndt Geafer" came back as "Aunt
        Gaither" and "aren't Jaffer"), so this is fuzzy: the best of the
        whole-name and per-word similarity must reach NAME_MATCH_MIN and lead
        the next person by NAME_MATCH_LEAD. Only HA's own people compete, and
        the session's first sentence says the name back, so a wrong pick is
        heard and can be stopped.
        """
        import difflib  # noqa: PLC0415

        wanted = (name or "").casefold().strip(" .!?,")
        if not wanted:
            return None
        exact = [pid for pid, p in self.persons().items() if str(p["name"]).casefold() == wanted]
        if len(exact) == 1:
            return exact[0]

        def _similarity(person_name: str) -> float:
            full = person_name.casefold()
            whole = difflib.SequenceMatcher(None, wanted, full).ratio()
            words = max(
                difflib.SequenceMatcher(None, w, x).ratio()
                for w in wanted.split()
                for x in full.split() or [full]
            )
            return max(whole, words)

        ranked = sorted(
            ((_similarity(str(p["name"])), pid) for pid, p in self.persons().items()),
            reverse=True,
        )
        if not ranked or ranked[0][0] < NAME_MATCH_MIN:
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < NAME_MATCH_LEAD:
            return None
        return ranked[0][1]

    # ── model ───────────────────────────────────────────────────────────────

    def _model_dir(self) -> Path:
        from .model import MODEL_DIR_NAME  # noqa: PLC0415

        return Path(self._hass.config.path(".storage", MODEL_DIR_NAME))

    async def async_ensure_model(self) -> bool:
        """Download (first time) and load the model. False if unavailable."""
        if self._embedder is not None:
            return True
        async with self._model_lock:
            if self._embedder is not None:
                return True
            try:
                from homeassistant.helpers.aiohttp_client import (  # noqa: PLC0415
                    async_get_clientsession,
                )

                from .model import Embedder, async_download  # noqa: PLC0415

                await async_download(
                    async_get_clientsession(self._hass), self._model_dir(), self._hass
                )
                started = time.monotonic()
                self._embedder = await self._hass.async_add_executor_job(
                    Embedder.load, self._model_dir()
                )
                self.last_error = None
                _LOGGER.info(
                    "AI Plugin voice ID: model loaded in %.1f s", time.monotonic() - started
                )
            except Exception as exc:  # noqa: BLE001 — numpy missing, offline, corrupt
                self.last_error = str(exc)
                _LOGGER.warning("AI Plugin voice ID unavailable: %s", exc)
                return False
        return True

    async def _async_embed(self, pcm: bytes):
        if not await self.async_ensure_model():
            return None, 0.0
        async with self._embed_lock:
            started = time.monotonic()
            result = await self._hass.async_add_executor_job(self._embedder.embed, pcm)
            _LOGGER.debug(
                "AI Plugin voice ID: fingerprint of %.1f s speech in %.0f ms",
                result[1], (time.monotonic() - started) * 1000,
            )
            return result

    def early_fingerprint_wanted(self) -> bool:
        """Whether the STT pass-through should start fingerprints itself.

        Only when a result could be used (someone is enrolled, or a session
        is collecting samples) and the model is already loaded: speech-to-
        text must never wait for a download or a model load.
        """
        return bool(
            self.enabled
            and self._embedder is not None
            and (self.has_profiles() or self.sessions.any_active())
        )

    def start_fingerprint(self, pcm: bytes) -> asyncio.Future:
        """Fingerprint ``pcm`` in the background; returns the running task."""
        task = asyncio.ensure_future(self._async_embed(pcm))
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        return task

    # ── identification (LLM turns only) ─────────────────────────────────────

    def should_identify(self, text: str, device_id: str | None, user_id: str | None) -> bool:
        return bool(
            self.enabled and device_id and user_id is None and self.has_profiles() and text
        )

    async def async_identify(self, text: str, device_id: str) -> Identity | None:
        """Fingerprint this turn's audio and match it. None = unknown."""
        from .matching import Sample, decide  # noqa: PLC0415

        clip = self.buffers.take_clip(text)
        if clip is None:
            _LOGGER.debug("AI Plugin voice ID: no audio for this turn")
            return None
        if self._embedder is None:
            # Never download or load inside a user's turn; setup starts that
            # in the background. Until it is ready, every turn is unknown.
            self._hass.async_create_task(self.async_ensure_model())
            _LOGGER.debug("AI Plugin voice ID: model not loaded yet")
            return None
        started = time.monotonic()
        pcm, task = clip
        if task is None:
            task = self.start_fingerprint(pcm)
        else:
            _LOGGER.debug("AI Plugin voice ID: using the fingerprint speech-to-text already started")
        # Not wait_for: cancelling would abandon an executor thread that is
        # still computing while the lock is released. The task finishes on
        # its own and its late result is dropped.
        done, _ = await asyncio.wait({task}, timeout=VOICE_ID_WAIT_SECONDS)
        if not done:
            _LOGGER.debug("AI Plugin voice ID: timed out after %.1f s", VOICE_ID_WAIT_SECONDS)
            return None
        try:
            vec, seconds = task.result()
        except Exception:  # noqa: BLE001 — a failed fingerprint is "unknown"
            _LOGGER.debug("AI Plugin voice ID: fingerprint failed", exc_info=True)
            return None
        if vec is None:
            _LOGGER.debug("AI Plugin voice ID: %.1f s of speech is too short", seconds)
            return None
        import numpy as np  # noqa: PLC0415

        profiles = {
            pid: [
                Sample(np.asarray(s["vec"], dtype=np.float32), s.get("mic_group", ""))
                for s in samples
            ]
            for pid, samples in self._data["profiles"].items()
            if samples
        }
        threshold, margin = self._thresholds()
        decision = decide(vec, profiles, self.mic_group(device_id), threshold, margin)
        ms = (time.monotonic() - started) * 1000
        _LOGGER.debug(
            "AI Plugin voice ID: %s (%s) score=%.2f margin=%.2f speech=%.1fs %.0fms",
            decision.person_id or "unknown", decision.reason,
            decision.score, decision.margin, seconds, ms,
        )
        if decision.person_id is None:
            return None
        info = self.persons().get(decision.person_id)
        if info is None:  # person deleted since enrollment
            return None
        return Identity(
            decision.person_id,
            str(info["name"]),
            self.memory_key(decision.person_id),
            decision.score,
        )

    # ── enrollment ──────────────────────────────────────────────────────────

    def _person_name(self, person_id: str) -> str:
        return str((self.persons().get(person_id) or {}).get("name") or person_id)

    def start_enrollment(self, device_id: str, person_id: str, lang: str) -> str:
        """Begin now (voice-started); returns the first thing to say."""
        return self.sessions.start(device_id, person_id, self._person_name(person_id), lang)

    def arm_enrollment(self, device_id: str, person_id: str, lang: str) -> None:
        """Begin with the next turn on that satellite (settings screen)."""
        self.sessions.arm(device_id, person_id, self._person_name(person_id), lang)
        _LOGGER.info(
            "AI Plugin voice ID: enrollment for %s armed on %s; waiting for the wake word",
            self._person_name(person_id), device_id,
        )

    def _mic2mp_pipeline(self, satellite: str):
        """The Mic to MediaPlayer instance routing ``satellite``'s speech, if any."""
        from ..satellite_speech import mic2mp_for  # noqa: PLC0415

        return mic2mp_for(self._hass, satellite)

    async def async_start_on_satellite(
        self, device_id: str, satellite: str, person_id: str, fallback_lang: str = "en"
    ) -> None:
        """Start enrollment from the settings screen: speak the introduction
        and first question on the satellite, then listen for the answer.

        Language and voice are those of the satellite's own Assist pipeline,
        so the questions sound like every other reply there.
        """
        pipeline = self._satellite_pipeline(device_id)
        lang = (
            pipeline.language.split("-")[0].lower()
            if pipeline is not None and pipeline.language
            else fallback_lang
        )
        text = self.start_enrollment(device_id, person_id, lang)
        _LOGGER.info(
            "AI Plugin voice ID: enrollment for %s started on %s (%s)",
            self._person_name(person_id), satellite, pipeline.name if pipeline else lang,
        )
        routed = self._mic2mp_pipeline(satellite)
        if routed is None:
            # The satellite renders this with its pipeline's voice, speaks it
            # and then listens on its own.
            await self._hass.services.async_call(
                "assist_satellite",
                "start_conversation",
                {"entity_id": satellite, "start_message": text},
                blocking=False,
            )
            return
        # Speaking takes as long as the sentence; the settings screen must not
        # wait for it.
        self._hass.async_create_task(
            self._async_announce_then_listen(
                device_id, satellite, text, pipeline,
                getattr(routed, "media_player_entity_id", None),
            )
        )

    def _satellite_pipeline(self, device_id: str):
        """The Assist pipeline this satellite runs, or None."""
        from ..satellite_speech import satellite_pipeline  # noqa: PLC0415

        return satellite_pipeline(self._hass, device_id)

    def _speech_url(self, text: str, pipeline) -> str | None:
        """Render ``text`` with the pipeline's TTS engine, language and voice."""
        from ..satellite_speech import speech_url  # noqa: PLC0415

        return speech_url(self._hass, text, pipeline)

    async def _async_announce_then_listen(
        self, device_id: str, satellite: str, text: str, pipeline, speaker: str | None
    ) -> None:
        """Speak on the routed speaker, open the satellite's microphone the
        moment it is quiet, and ask again if no answer arrives.

        Measured 2026-10-24 on a Music Assistant speaker: the announce call
        returns about 2 s after playback ends, and people answer straight
        away — the microphone opened 3 s after the question and heard nothing.
        So the speaker is watched directly while the call runs, and a missed
        answer gets the question once more (at most twice) instead of ending
        the enrollment.
        """
        session = self.sessions.get(device_id)
        retry = str(prompts_for(session.lang if session else "en").get("retry", ""))
        message = text
        for attempt in range(1 + _ENROLL_REASKS):
            data: dict[str, Any] = {"satellite_entity_id": satellite}
            url = self._speech_url(message, pipeline)
            if url:
                data["audio_url"] = url
            else:
                data["message"] = message
                if pipeline is not None and pipeline.tts_engine:
                    data["tts_entity_id"] = pipeline.tts_engine
                    if pipeline.tts_language:
                        data["language"] = pipeline.tts_language
            try:
                opened = await self._async_speak_and_open(
                    device_id, satellite, data, message, speaker
                )
            except Exception:  # noqa: BLE001
                _LOGGER.warning(
                    "AI Plugin voice ID: could not ask on %s; saying the wake word "
                    "continues the enrollment", satellite, exc_info=True,
                )
                return
            answered = await self._async_wait_for_answer(device_id, satellite, opened)
            session = self.sessions.get(device_id)
            if answered or session is None:
                return
            _LOGGER.info(
                "AI Plugin voice ID: no answer on %s (attempt %d); asking again",
                satellite, attempt + 1,
            )
            message = f"{retry} {self.sessions.current_question(session)}".strip()

    async def _async_speak_and_open(
        self, device_id: str, satellite: str, data: dict, message: str, speaker: str | None
    ) -> float:
        """Announce ``message`` (as ``data`` says), then open the microphone as
        soon as the speaker stops. Returns the monotonic time it opened."""
        announce = asyncio.ensure_future(
            self._hass.services.async_call("mic_to_mediaplayer", "announce", data, blocking=True)
        )

        def _busy() -> bool:
            state = self._hass.states.get(speaker) if speaker else None
            return state is not None and state.state in _SPEAKER_BUSY_STATES

        waited = 0.0
        started = False
        while waited < _SPEAKER_WAIT_CAP_S:
            if _busy():
                started = True
            elif started or announce.done():
                break
            await asyncio.sleep(_SPEAKER_POLL_S)
            waited += _SPEAKER_POLL_S
        if announce.done() and announce.exception() is not None:
            raise announce.exception()
        await asyncio.sleep(_LISTEN_SETTLE_S)
        entity = (
            self._hass.data.get("ai_plugin", {})
            .get(self._entry.entry_id, {})
            .get("conversation_entity")
        )
        if entity is not None:
            # If the microphone still catches the end of the question, the
            # echo filter drops it instead of storing the speaker's voice.
            entity._remember_reply(device_id, message)
        await self._hass.services.async_call(
            "assist_satellite",
            "start_conversation",
            {"entity_id": satellite, "start_message": "", "preannounce": False},
            blocking=False,
        )
        return time.monotonic()

    async def _async_wait_for_answer(self, device_id: str, satellite: str, opened: float) -> bool:
        """True once an enrollment turn arrived after ``opened``; False when
        the satellite went back to idle without one."""
        waited, was_active = 0.0, False
        while waited < _ANSWER_WAIT_CAP_S:
            session = self.sessions.get(device_id)
            if session is None or session.last_activity > opened:
                return True
            state = self._hass.states.get(satellite)
            if state is not None and state.state != "idle":
                was_active = True
            elif was_active or waited > _MIC_START_GRACE_S:
                # Listening ended. Give the turn a moment to reach us.
                await asyncio.sleep(_TURN_ARRIVAL_S)
                session = self.sessions.get(device_id)
                return session is None or session.last_activity > opened
            await asyncio.sleep(_SPEAKER_POLL_S)
            waited += _SPEAKER_POLL_S
        return False

    def enrollment_active(self, device_id: str | None) -> bool:
        """A session is collecting answers on this satellite.

        Sessions outlive entry reloads, so one started just before voice
        recognition was switched off would otherwise keep claiming the
        satellite's turns without any audio to sample.
        """
        return self.enabled and self.sessions.active(device_id)

    def wants_follow_up(self, device_id: str | None) -> bool:
        """Keep the microphone open while an enrollment session is running."""
        return self.enrollment_active(device_id)

    async def async_enrollment_turn(self, text: str, device_id: str | None) -> str | None:
        """Handle a turn on a satellite with a running session, else None.

        Runs before every shortcut: an answer like "turn it all off" must be
        recorded, never executed.
        """
        session = self.sessions.get(device_id)
        if session is None:
            return None
        if not session.asked:
            # Armed from the settings screen: this wake-up turn only starts
            # the questions; what was said is not a sample.
            self.buffers.take(text)
            _LOGGER.info("AI Plugin voice ID: enrollment for %s started on %s", session.name, device_id)
            return self.sessions.begin(session)
        if is_stop(text, session.lang):
            self.sessions.stop(session.device_id)
            self.buffers.take(text)
            return str(prompts_for(session.lang)["stopped"]).format(n=session.saved)
        clip = self.buffers.take_clip(text)
        vec = None
        if clip is not None:
            pcm, task = clip
            try:
                vec, _ = await (task if task is not None else self._async_embed(pcm))
            except Exception:  # noqa: BLE001
                _LOGGER.debug("AI Plugin voice ID: enrollment fingerprint failed", exc_info=True)
        if vec is not None:
            await self._async_add_sample(session.person_id, vec, session.device_id)
        reply, finished = self.sessions.answer(session, vec is not None)
        _LOGGER.info(
            "AI Plugin voice ID: enrollment answer for %s %s (%d saved%s)",
            session.name, "saved" if vec is not None else "too short",
            session.saved, ", finished" if finished else "",
        )
        return reply
