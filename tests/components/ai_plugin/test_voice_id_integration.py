"""Voice recognition wiring: manager, LLM-only trigger, enrollment, STT
pass-through, pipeline switching."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from custom_components.ai_plugin.voice_id import Identity, VoiceIDManager
from custom_components.ai_plugin.voice_id import enrollment as ve

from .test_conversation import _baseline_orch, _make_mock_entry

CORE = "custom_components.ai_plugin.orchestrator.core"


def _unit(*vals: float) -> np.ndarray:
    v = np.zeros(192, np.float32)
    v[: len(vals)] = vals
    return v / np.linalg.norm(v)


def _person(entity_id: str, name: str, user_id: str | None = None):
    return SimpleNamespace(
        entity_id=entity_id, attributes={"friendly_name": name, "user_id": user_id}
    )


def _hass(persons=()) -> MagicMock:
    hass = MagicMock()
    hass.states.async_all.side_effect = lambda domain=None: list(persons)

    async def _executor(fn, *args):
        return fn(*args)

    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    hass.async_create_task = MagicMock(side_effect=lambda coro: coro.close())
    return hass


class _FakeEmbedder:
    def __init__(self, vec) -> None:
        self.vec = vec
        self.calls = 0

    def embed(self, pcm: bytes):
        self.calls += 1
        return self.vec, 2.0


def _manager(persons=(), options=None, embedder=None) -> VoiceIDManager:
    entry = _make_mock_entry({"voice_id_enabled": True, **(options or {})})
    mgr = VoiceIDManager(_hass(persons), entry)
    mgr._embedder = embedder
    mgr.mic_group = lambda device_id: "respeaker"
    return mgr


ANNA = _person("person.anna", "Anna", user_id="u-anna")
BEN = _person("person.ben", "Ben")


async def _enroll(mgr: VoiceIDManager, person_id: str, vec, n: int = 3) -> None:
    for _ in range(n):
        await mgr._async_add_sample(person_id, vec, "sat1")


# ── manager ───────────────────────────────────────────────────────────────────


async def test_identify_names_the_enrolled_speaker() -> None:
    mgr = _manager([ANNA, BEN], embedder=_FakeEmbedder(_unit(1, 0.05)))
    await _enroll(mgr, "person.anna", _unit(1, 0))
    await _enroll(mgr, "person.ben", _unit(0, 1))
    mgr.buffers.put("what is on my list", b"\x00\x01" * 100)
    ident = await mgr.async_identify("What is on my list?", "sat1")
    assert ident == Identity("person.anna", "Anna", "u-anna", pytest.approx(ident.score))


async def test_identify_without_audio_is_unknown() -> None:
    mgr = _manager([ANNA], embedder=_FakeEmbedder(_unit(1)))
    await _enroll(mgr, "person.anna", _unit(1))
    assert await mgr.async_identify("no buffer for this", "sat1") is None


async def test_identify_never_loads_the_model_inside_a_turn() -> None:
    mgr = _manager([ANNA], embedder=None)
    await _enroll(mgr, "person.anna", _unit(1))
    mgr.buffers.put("hello", b"\x00\x01")
    assert await mgr.async_identify("hello", "sat1") is None
    mgr._hass.async_create_task.assert_called_once()  # background load scheduled


def test_memory_key_prefers_the_linked_user() -> None:
    mgr = _manager([ANNA, BEN])
    assert mgr.memory_key("person.anna") == "u-anna"
    assert mgr.memory_key("person.ben") == "person_ben"


def test_find_person_requires_a_single_match() -> None:
    mgr = _manager([ANNA, BEN, _person("person.anna_m", "Anna Maria")])
    assert mgr.find_person("anna") == "person.anna"
    assert mgr.find_person("Ben") == "person.ben"
    assert mgr.find_person("Carl") is None


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        # Real speech-to-text output for "Arndt Geafer", 2026-10-24.
        ("Aunt Gaither", "person.arndt"),
        ("aren't Jaffer", "person.arndt"),
        ("Lenna", "person.lena"),
        ("Anna", None),
    ],
)
def test_find_person_survives_mangled_names(heard, expected) -> None:
    mgr = _manager([_person("person.arndt", "Arndt Geafer"), _person("person.lena", "Lena Weber")])
    assert mgr.find_person(heard) == expected


async def test_samples_are_capped_per_microphone() -> None:
    mgr = _manager([ANNA])
    await _enroll(mgr, "person.anna", _unit(1), n=45)
    assert mgr.profile_summary() == {"person.anna": {"respeaker": 40}}


def test_should_identify_only_for_anonymous_voice_turns_with_profiles() -> None:
    mgr = _manager([ANNA])
    assert not mgr.should_identify("hi", "sat1", None)  # no profiles yet
    mgr._data["profiles"]["person.anna"] = [{"vec": [1.0], "mic_group": "x"}]
    assert mgr.should_identify("hi", "sat1", None)
    assert not mgr.should_identify("hi", "sat1", "u-anna")  # app/text user known
    assert not mgr.should_identify("hi", None, None)


# ── LLM-only trigger (the user's requirement) ─────────────────────────────────


def _voice_orch(voice):
    orch = _baseline_orch()
    orch._hass = MagicMock()
    orch._is_voice_device = lambda device_id: True
    orch._voice_id = voice
    provider = MagicMock()
    provider.async_complete = AsyncMock(return_value="It is sunny.")
    provider.async_chat = AsyncMock(return_value=SimpleNamespace(content="It is sunny.", tool_calls=[]))
    orch._provider = provider
    return orch


def _fake_voice(identity=None, profiles=True):
    voice = MagicMock()
    voice.enabled = True
    voice.voice_enroll_allowed = False
    voice.enrollment_active.return_value = False
    voice.should_identify.return_value = profiles
    voice.async_identify = AsyncMock(return_value=identity)
    return voice


def _no_shortcuts(**hit):
    return [
        patch(f"{CORE}.try_shortcut", return_value=hit.get("try_shortcut")),
        patch(f"{CORE}.async_try_media_shortcut", AsyncMock(return_value=hit.get("media"))),
        patch(f"{CORE}.async_try_domain_sweep_shortcut", AsyncMock(return_value=hit.get("sweep"))),
        patch(f"{CORE}.async_try_action_shortcut", AsyncMock(return_value=hit.get("action"))),
    ]


async def _run(orch, message="What's the weather?", **hit):
    patches = _no_shortcuts(**hit)
    for p in patches:
        p.start()
    try:
        return await orch.async_process(
            message=message, conversation_id="c1", language="en", device_id="sat1"
        )
    finally:
        for p in patches:
            p.stop()


async def test_shortcut_turn_never_fingerprints() -> None:
    voice = _fake_voice()
    orch = _voice_orch(voice)
    reply = await _run(orch, "temperature in the kitchen", try_shortcut="It is 21 degrees.")
    assert reply == "It is 21 degrees."
    voice.async_identify.assert_not_called()


async def test_action_shortcut_turn_never_fingerprints() -> None:
    voice = _fake_voice()
    orch = _voice_orch(voice)
    await _run(orch, "turn on the lamp", action=(True, ""))
    voice.async_identify.assert_not_called()


async def test_llm_turn_fingerprints_once() -> None:
    voice = _fake_voice()
    orch = _voice_orch(voice)
    reply = await _run(orch)
    assert reply
    voice.async_identify.assert_awaited_once_with("What's the weather?", "sat1")


async def test_no_profiles_means_no_fingerprint() -> None:
    voice = _fake_voice(profiles=False)
    orch = _voice_orch(voice)
    await _run(orch)
    voice.async_identify.assert_not_called()


async def test_text_turn_never_fingerprints() -> None:
    voice = _fake_voice()
    orch = _voice_orch(voice)
    orch._is_voice_device = lambda device_id: False
    await _run(orch)
    voice.async_identify.assert_not_called()


async def test_recognised_speaker_reaches_memory_and_prompt() -> None:
    voice = _fake_voice(Identity("person.anna", "Anna", "u-anna", 0.6))
    orch = _voice_orch(voice)
    seen = {}

    async def _volatile(user_id, conversation_id, speaker=None, speaker_checked=False):
        seen.update(user_id=user_id, speaker=speaker, checked=speaker_checked)
        return ""

    orch._build_volatile_block = _volatile
    await _run(orch)
    assert seen == {"user_id": "u-anna", "speaker": "Anna", "checked": True}


def _memory():
    memory = MagicMock()
    memory.call_tool = AsyncMock(return_value="Remembered: x")
    return memory


async def test_unrecognised_speaker_cannot_save_to_memory() -> None:
    """"I'm Art" from a voice nobody recognised was promised to be
    remembered (2026-10-26); facts are only stored for known speakers."""
    orch = _voice_orch(_fake_voice())
    await _run(orch)
    orch._memory = _memory()
    for tool in ("remember", "forget"):
        out = await orch._dispatch_tool(tool, {"fact": "My name is Art"}, device_id="sat1")
        assert out.startswith("[Not saved")
    orch._memory.call_tool.assert_not_called()
    await orch._dispatch_tool("recall", {}, device_id="sat1")
    orch._memory.call_tool.assert_awaited_once()


async def test_recognised_speaker_can_save_to_memory() -> None:
    orch = _voice_orch(_fake_voice(Identity("person.anna", "Anna", "u-anna", 0.6)))
    await _run(orch)
    orch._memory = _memory()
    assert await orch._dispatch_tool("remember", {"fact": "likes tea"}, "u-anna", device_id="sat1") == "Remembered: x"


async def test_turn_without_voice_check_can_save_to_memory() -> None:
    """Voice ID off, or no profiles yet: memory works as before."""
    orch = _voice_orch(_fake_voice(profiles=False))
    await _run(orch)
    orch._memory = _memory()
    assert await orch._dispatch_tool("remember", {"fact": "likes tea"}, device_id="sat1") == "Remembered: x"


async def test_failed_identification_still_answers() -> None:
    voice = _fake_voice()
    voice.async_identify = AsyncMock(side_effect=RuntimeError("boom"))
    orch = _voice_orch(voice)
    assert await _run(orch)


async def test_volatile_block_speaker_line() -> None:
    """Who speaks NOW must override a name from earlier turns: the model
    told the second speaker the first one's name twice although their voice
    was recognised."""
    orch = _baseline_orch()
    orch._hass = None
    block = await orch._build_volatile_block(None, "c1", speaker="Anna")
    assert "recognised by voice as Anna" in block and "earlier turns" in block
    unknown = await orch._build_volatile_block(None, "c1", speaker_checked=True)
    assert "could not tell who is speaking this time" in unknown
    assert "Do not assume" in unknown
    assert "Nothing can be saved to memory" in unknown
    # An enrolled speaker who scored just under the threshold was told their
    # voice was not registered yet; a guest must still be able to enroll.
    assert "Never say their voice is unknown, not registered" in unknown
    assert "learn their voice is still carried out" in unknown
    assert "[SPEAKER]" not in await orch._build_volatile_block(None, "c1")


def test_enrollment_without_a_name_uses_the_recognised_speaker() -> None:
    """'Let's do some voice samples' from someone already recognised adds
    samples for this microphone to their profile."""
    mgr = _manager([ANNA, BEN])
    orch = _voice_orch(mgr)
    orch._turn_speakers = {"sat1": "person.ben"}
    orch._enroll_replies = {}
    out = orch._start_voice_enrollment({}, "sat1", "en")
    assert "started" in out and "Ben" in out
    assert mgr.sessions.get("sat1").person_id == "person.ben"


def test_enrollment_without_a_name_or_speaker_asks() -> None:
    mgr = _manager([ANNA, BEN])
    orch = _voice_orch(mgr)
    orch._turn_speakers = {}
    out = orch._start_voice_enrollment({}, "sat1", "en")
    assert "NOT started" in out and not mgr.sessions.active("sat1")


async def test_started_enrollment_is_spoken_verbatim() -> None:
    """The tool loop ends with the first question itself, not a paraphrase
    (the model once said 'I've started the process' after a failed start)."""
    mgr = _manager([ANNA])
    orch = _voice_orch(mgr)
    orch._enroll_replies = {}
    orch._max_tool_iterations = 3
    call = SimpleNamespace(
        name="start_voice_enrollment", arguments={"person": "Anna"},
        to_assistant_message=lambda: {"role": "assistant", "content": None},
        to_tool_result_message=lambda r: {"role": "tool", "content": r},
    )
    orch._provider.async_chat = AsyncMock(
        return_value=SimpleNamespace(has_tool_calls=True, tool_calls=[call], reply_text=lambda: "")
    )
    reply, _ = await orch._tool_loop(
        [{"role": "user", "content": "learn my voice"}],
        [{"function": {"name": "start_voice_enrollment"}}],
        device_id="sat1", language="en",
    )
    assert reply == ve.prompts_for("en")["start"].format(name="Anna", n=ve.TARGET_SAMPLES) + " " + ve.prompts_for("en")["questions"][0]
    assert orch._provider.async_chat.await_count == 1


# ── enrollment through the orchestrator ───────────────────────────────────────


async def test_enrollment_answer_is_recorded_not_executed() -> None:
    mgr = _manager([ANNA], embedder=_FakeEmbedder(_unit(1)))
    mgr.start_enrollment("sat1", "person.anna", "en")
    mgr.buffers.put("turn off all the lights", b"\x00\x01" * 100)
    orch = _voice_orch(mgr)
    action = AsyncMock(return_value="OK, all lights are off.")
    with patch(f"{CORE}.async_try_domain_sweep_shortcut", action):
        reply = await orch.async_process(
            message="Turn off all the lights", conversation_id="c1",
            language="en", device_id="sat1",
        )
    action.assert_not_called()
    assert reply == ve.prompts_for("en")["questions"][1]
    assert mgr.profile_summary() == {"person.anna": {"respeaker": 1}}
    assert mgr.wants_follow_up("sat1")


async def test_enrollment_stop_word_ends_session() -> None:
    mgr = _manager([ANNA], embedder=_FakeEmbedder(_unit(1)))
    mgr.start_enrollment("sat1", "person.anna", "de")
    reply = await mgr.async_enrollment_turn("Stopp", "sat1")
    assert "beendet" in reply and not mgr.sessions.active("sat1")


async def test_enrollment_on_another_satellite_is_ignored() -> None:
    mgr = _manager([ANNA], embedder=_FakeEmbedder(_unit(1)))
    mgr.start_enrollment("sat1", "person.anna", "en")
    assert await mgr.async_enrollment_turn("hello", "sat2") is None


def test_enrollment_tool_rejects_unknown_person() -> None:
    mgr = _manager([ANNA, BEN])
    orch = _voice_orch(mgr)
    out = orch._start_voice_enrollment({"person": "Carl"}, "sat1", "en")
    assert "NOT started" in out and "Anna" in out
    assert not mgr.sessions.active("sat1")
    out = orch._start_voice_enrollment({"person": "anna"}, "sat1", "en")
    assert mgr.sessions.active("sat1") and "What did you do today?" in out


def test_enrollment_tool_respects_the_switch() -> None:
    mgr = _manager([ANNA], options={"voice_id_voice_enroll": False})
    orch = _voice_orch(mgr)
    assert "not available" in orch._start_voice_enrollment({"person": "Anna"}, "sat1", "en")


# ── STT pass-through ──────────────────────────────────────────────────────────


stt = sys.modules["homeassistant.components.stt"]


def _meta(**kw):
    base = dict(
        language="de-DE", format=stt.AudioFormats.WAV, codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16, sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )
    base.update(kw)
    return stt.SpeechMetadata(**base)


class _Engine:
    supported_languages = ["de-DE"]
    supported_formats = [stt.AudioFormats.WAV]
    supported_codecs = [stt.AudioCodecs.PCM]
    supported_bit_rates = [stt.AudioBitRates.BITRATE_16]
    supported_sample_rates = [stt.AudioSampleRates.SAMPLERATE_16000]
    supported_channels = [stt.AudioChannels.CHANNEL_MONO]

    def __init__(self, text="Wie spät ist es?", state=stt.SpeechResultState.SUCCESS):
        self.text, self.state, self.received = text, state, []

    async def async_process_audio_stream(self, metadata, stream):
        async for chunk in stream:
            self.received.append(chunk)
        return stt.SpeechResult(self.text, self.state)


async def _chunks():
    for i in range(3):
        yield bytes([i]) * 320


@pytest.fixture
def engine():
    eng = _Engine()
    stt.ENGINES["stt.home_assistant_cloud"] = eng
    yield eng
    stt.ENGINES.clear()


def _wrapper(mgr):
    from custom_components.ai_plugin.stt import VoiceIDSpeechToText

    return VoiceIDSpeechToText(mgr._hass, mgr._entry, mgr, "stt.home_assistant_cloud")


async def test_passthrough_forwards_audio_and_buffers_it(engine) -> None:
    mgr = _manager()
    ent = _wrapper(mgr)
    result = await ent.async_process_audio_stream(_meta(), _chunks())
    assert result.text == "Wie spät ist es?"
    assert engine.received == [bytes([i]) * 320 for i in range(3)]
    assert mgr.buffers.take("wie spät ist es") == b"".join(engine.received)
    assert ent.supported_languages == ["de-DE"]


async def test_passthrough_does_not_buffer_other_formats(engine) -> None:
    mgr = _manager()
    await _wrapper(mgr).async_process_audio_stream(_meta(codec=stt.AudioCodecs.OPUS), _chunks())
    assert len(mgr.buffers) == 0


async def test_passthrough_returns_engine_errors_untouched(engine) -> None:
    engine.state, engine.text = stt.SpeechResultState.ERROR, None
    mgr = _manager()
    result = await _wrapper(mgr).async_process_audio_stream(_meta(), _chunks())
    assert result.result == stt.SpeechResultState.ERROR
    assert len(mgr.buffers) == 0


async def test_passthrough_without_engine_reports_error() -> None:
    mgr = _manager()
    result = await _wrapper(mgr).async_process_audio_stream(_meta(), _chunks())
    assert result.result == stt.SpeechResultState.ERROR
    assert _wrapper(mgr).supported_languages == []


def _speech_chunks(silence_s: float, speech_s: float, tail_s: float = 0.0):
    """16 kHz PCM in 0.1 s chunks: silence, a loud tone, then silence."""
    t = np.arange(int(speech_s * 16000)) / 16000
    tone = (0.5 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()
    pcm = bytes(int(silence_s * 16000) * 2) + tone + bytes(int(tail_s * 16000) * 2)
    return [pcm[i : i + 3200] for i in range(0, len(pcm), 3200)]


async def _stream(chunks, seen):
    for chunk in chunks:
        seen.append(chunk)
        yield chunk


def _spy_starts(mgr):
    """Record how much audio had streamed in when each fingerprint started."""
    starts, real = [], mgr.start_fingerprint

    def _start(pcm):
        starts.append(len(pcm))
        return real(pcm)

    mgr.start_fingerprint = _start
    return starts


async def test_fingerprint_starts_while_still_speaking(engine) -> None:
    mgr = _manager([ANNA], embedder=_FakeEmbedder(_unit(1)))
    await _enroll(mgr, "person.anna", _unit(1))
    starts = _spy_starts(mgr)
    chunks, seen = _speech_chunks(0.5, 6.0, tail_s=0.8), []
    await _wrapper(mgr).async_process_audio_stream(_meta(), _stream(chunks, seen))
    total = sum(len(c) for c in chunks)
    # Started once, after 4 s of speech had arrived and well before the end.
    assert len(starts) == 1
    assert (0.5 + 4.0) * 32000 <= starts[0] <= (0.5 + 4.5) * 32000 < total
    pcm, task = mgr.buffers.take_clip("wie spät ist es")
    assert len(pcm) == total and task is not None
    assert (await task)[0] is not None


async def test_short_turn_fingerprints_at_end_of_speech(engine) -> None:
    mgr = _manager([ANNA], embedder=_FakeEmbedder(_unit(1)))
    await _enroll(mgr, "person.anna", _unit(1))
    starts = _spy_starts(mgr)
    chunks = _speech_chunks(0.3, 1.5, tail_s=0.7)
    await _wrapper(mgr).async_process_audio_stream(_meta(), _stream(chunks, []))
    assert starts == [sum(len(c) for c in chunks)]


async def test_no_early_fingerprint_without_profiles_or_model(engine) -> None:
    nobody = _manager([ANNA], embedder=_FakeEmbedder(_unit(1)))  # no one enrolled
    no_model = _manager([ANNA], embedder=None)
    await _enroll(no_model, "person.anna", _unit(1))
    for mgr in (nobody, no_model):
        starts = _spy_starts(mgr)
        await _wrapper(mgr).async_process_audio_stream(
            _meta(), _stream(_speech_chunks(0.3, 5.0), [])
        )
        assert starts == []
        assert mgr.buffers.take_clip("wie spät ist es")[1] is None


async def test_enrollment_session_alone_enables_early_fingerprint(engine) -> None:
    mgr = _manager([ANNA], embedder=_FakeEmbedder(_unit(1)))
    mgr.start_enrollment("sat1", "person.anna", "en")
    starts = _spy_starts(mgr)
    await _wrapper(mgr).async_process_audio_stream(_meta(), _stream(_speech_chunks(0.3, 1.0), []))
    assert len(starts) == 1


async def test_identify_and_enrollment_reuse_the_early_fingerprint() -> None:
    emb = _FakeEmbedder(_unit(1))
    mgr = _manager([ANNA], embedder=emb)
    await _enroll(mgr, "person.anna", _unit(1))
    mgr.buffers.put("who am i", b"\x00\x01" * 100, mgr.start_fingerprint(b"\x00\x01" * 100))
    assert (await mgr.async_identify("Who am I?", "sat1")).person_id == "person.anna"
    assert emb.calls == 1

    mgr.start_enrollment("sat1", "person.anna", "en")
    mgr.buffers.put("pasta", b"\x00\x01" * 100, mgr.start_fingerprint(b"\x00\x01" * 100))
    await mgr.async_enrollment_turn("Pasta.", "sat1")
    assert emb.calls == 2
    assert mgr.profile_summary() == {"person.anna": {"respeaker": 4}}


# ── pipeline switching ────────────────────────────────────────────────────────


ap = sys.modules["homeassistant.components.assist_pipeline"]


@pytest.fixture
def registry():
    entries = {
        "conversation.ai_plugin": SimpleNamespace(
            entity_id="conversation.ai_plugin", config_entry_id="test-entry-id",
            platform="ai_plugin", unique_id="conv",
        ),
        "stt.ai_plugin_voice_id": SimpleNamespace(
            entity_id="stt.ai_plugin_voice_id", config_entry_id="test-entry-id",
            platform="ai_plugin", unique_id="test-entry-id_voice_id_stt.home_assistant_cloud",
        ),
    }
    reg = MagicMock()
    reg.entities = entries
    with patch("homeassistant.helpers.entity_registry.async_get", return_value=reg):
        yield reg
    ap.PIPELINES.clear()


def _pipeline(pid, agent, engine):
    return SimpleNamespace(id=pid, name=pid, conversation_engine=agent, stt_engine=engine)


async def test_switch_and_restore_only_our_pipelines(registry) -> None:
    from custom_components.ai_plugin.voice_id import pipelines

    ours = _pipeline("buddy", "conversation.ai_plugin", "stt.home_assistant_cloud")
    other = _pipeline("ha", "conversation.home_assistant", "stt.home_assistant_cloud")
    ap.PIPELINES.extend([ours, other])
    mgr = _manager()
    assert pipelines.engines_to_wrap(mgr._hass, mgr._entry, mgr) == ["stt.home_assistant_cloud"]
    assert await pipelines.async_switch(mgr._hass, mgr._entry, mgr) == 1
    assert ours.stt_engine == "stt.ai_plugin_voice_id"
    assert other.stt_engine == "stt.home_assistant_cloud"
    assert mgr.pipeline_originals == {"buddy": "stt.home_assistant_cloud"}
    # Switching again changes nothing; the wrapped engine is still known.
    assert await pipelines.async_switch(mgr._hass, mgr._entry, mgr) == 0
    assert pipelines.engines_to_wrap(mgr._hass, mgr._entry, mgr) == ["stt.home_assistant_cloud"]
    assert await pipelines.async_restore(mgr._hass, mgr._entry, mgr) == 1
    assert ours.stt_engine == "stt.home_assistant_cloud"
    assert mgr.pipeline_originals == {}


async def test_restore_leaves_a_hand_changed_pipeline_alone(registry) -> None:
    from custom_components.ai_plugin.voice_id import pipelines

    ours = _pipeline("buddy", "conversation.ai_plugin", "stt.home_assistant_cloud")
    ap.PIPELINES.append(ours)
    mgr = _manager()
    await pipelines.async_switch(mgr._hass, mgr._entry, mgr)
    ours.stt_engine = "stt.faster_whisper"  # user picked another engine
    assert await pipelines.async_restore(mgr._hass, mgr._entry, mgr) == 0
    assert ours.stt_engine == "stt.faster_whisper"


# ── settings screens ──────────────────────────────────────────────────────────


def _load_steps_mixin():
    """Import voice_id_steps without the config_flow package, whose other
    modules need HA's real flow machinery (absent under the test stubs)."""
    import importlib.util
    from pathlib import Path

    import custom_components.ai_plugin as pkg

    helpers = sys.modules["homeassistant.helpers"]
    if not hasattr(helpers, "selector"):
        helpers.selector = MagicMock()
        sys.modules["homeassistant.helpers.selector"] = helpers.selector
    path = Path(pkg.__file__).parent / "config_flow" / "voice_id_steps.py"
    spec = importlib.util.spec_from_file_location(
        "custom_components.ai_plugin._steps.voice_id_steps", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VoiceIDStepsMixin


class _Flow:
    """VoiceIDStepsMixin on a stand-in flow that records what it shows."""

    def __init__(self, mgr) -> None:
        self.__class__ = type("F", (_Flow, _load_steps_mixin()), {})
        self.hass = mgr._hass
        self.hass.data = {"ai_plugin": {"test-entry-id": {"voice_id": mgr}}}
        self.hass.config.language = "en"
        self.hass.services.async_call = AsyncMock()
        self.hass.states.get = MagicMock(return_value=None)
        self._config_entry = mgr._entry
        self._options = dict(mgr._entry.options)

    def async_show_menu(self, **kw):
        return {"type": "menu", **kw}

    def async_show_form(self, **kw):
        return {"type": "form", **kw}

    def async_abort(self, reason, description_placeholders=None):
        return {"type": "abort", "reason": reason, "placeholders": description_placeholders}

    def async_create_entry(self, title, data):
        return {"type": "create_entry", "data": data}


@pytest.fixture
def satellites():
    reg = MagicMock()
    reg.entities = {
        "assist_satellite.kitchen": SimpleNamespace(
            entity_id="assist_satellite.kitchen", device_id="sat1",
            platform="esphome", config_entry_id="other", unique_id="sat",
        ),
        "select.kitchen_assistant": SimpleNamespace(
            entity_id="select.kitchen_assistant", device_id="sat1",
            platform="esphome", config_entry_id="other", unique_id="aa-pipeline",
        ),
    }
    with patch("homeassistant.helpers.entity_registry.async_get", return_value=reg):
        yield


async def test_enroll_screen_lets_a_plain_satellite_ask_itself(satellites) -> None:
    mgr = _manager([ANNA], embedder=_FakeEmbedder(_unit(1)))
    flow = _Flow(mgr)
    form = await flow.async_step_voice_id_enroll()
    assert form["type"] == "form"
    flow_mod = sys.modules["homeassistant.helpers"]
    person_lists = [
        c.kwargs["options"]
        for c in flow_mod.selector.SelectSelectorConfig.call_args_list
        if any(o["value"] == "person.anna" for o in c.kwargs.get("options", []))
    ]
    assert [o["label"] for o in person_lists[-1]] == ["Anna (not enrolled yet)"]
    result = await flow.async_step_voice_id_enroll({"person": "person.anna", "satellite": "sat1"})
    assert result["reason"] == "voice_id_enroll_started"
    domain, service, data = flow.hass.services.async_call.await_args.args[:3]
    assert (domain, service) == ("assist_satellite", "start_conversation")
    assert data["entity_id"] == "assist_satellite.kitchen"
    assert data["start_message"].endswith(ve.prompts_for("en")["questions"][0])
    # The next turn on that satellite is already the first answer.
    mgr.buffers.put("i went for a walk", b"\x00\x01" * 100)
    await mgr.async_enrollment_turn("I went for a walk", "sat1")
    assert mgr.profile_summary() == {"person.anna": {"respeaker": 1}}


def _routed_flow(answer_after_listen: bool):
    """Flow whose satellite is behind Mic to MediaPlayer; the speaker plays
    for two polls per announcement, and a listen either yields a turn or not."""
    mgr = _manager([ANNA])
    flow = _Flow(mgr)
    routed = SimpleNamespace(
        pipeline_id="buddy", media_player_entity_id="media_player.bedroom",
        matches_satellite_id=lambda sat: sat == "assist_satellite.kitchen",
    )
    flow.hass.data["mic_to_mediaplayer"] = {"e1": routed}
    flow.hass.services.has_service = MagicMock(return_value=True)
    entity = MagicMock()
    flow.hass.data["ai_plugin"]["test-entry-id"]["conversation_entity"] = entity
    background = []
    flow.hass.async_create_task = MagicMock(side_effect=background.append)
    speaker = []

    async def _call(domain, service, data, blocking=False):
        if service == "announce":
            speaker.extend(["playing", "playing"])
        elif service == "start_conversation" and answer_after_listen:
            import time as _t

            mgr.sessions.get("sat1").last_activity = _t.monotonic() + 5

    flow.hass.services.async_call = AsyncMock(side_effect=_call)

    def _state(eid):
        if eid == "media_player.bedroom":
            return SimpleNamespace(state=speaker.pop(0) if speaker else "idle", attributes={})
        if eid == "select.kitchen_assistant":
            return SimpleNamespace(state="Buddy (DE)", attributes={"options": ["preferred", "Buddy", "Buddy (DE)"]})
        return SimpleNamespace(state="idle", attributes={})

    flow.hass.states.get = MagicMock(side_effect=_state)
    return mgr, flow, entity, background


_FAST = {
    "_LISTEN_SETTLE_S": 0, "_SPEAKER_POLL_S": 0.001,
    "_MIC_START_GRACE_S": 0.01, "_TURN_ARRIVAL_S": 0,
}


async def _run_background(background):
    patches = [patch(f"custom_components.ai_plugin.voice_id.{k}", v) for k, v in _FAST.items()]
    for p in patches:
        p.start()
    try:
        await background[0]
    finally:
        for p in patches:
            p.stop()


_PIPELINES = [
    SimpleNamespace(id="buddy", name="Buddy", language="en", conversation_engine="conversation.ai_plugin", tts_engine="tts.cloud",
                    tts_language="en-US", tts_voice="LunaNeural"),
    SimpleNamespace(id="buddy_de", name="Buddy (DE)", language="de", conversation_engine="conversation.ai_plugin", tts_engine="tts.cloud",
                    tts_language="de-DE", tts_voice="MajaNeural"),
]


async def test_enroll_screen_speaks_through_mic_to_mediaplayer(satellites) -> None:
    """Replies there play on another media player; a satellite announcement
    would go to the speaker nobody hears (seen 2026-10-24). The question is
    rendered with the satellite's own pipeline (language AND voice — the
    first attempt read English text in a German default voice), the
    microphone opens as soon as the speaker is quiet, and the question is
    registered with the echo filter."""
    tts = sys.modules["homeassistant.components.tts"]
    tts.STREAMS.clear()
    mgr, flow, entity, background = _routed_flow(answer_after_listen=True)
    ap.PIPELINES.extend(_PIPELINES)
    try:
        result = await flow.async_step_voice_id_enroll({"person": "person.anna", "satellite": "sat1"})
        assert result["reason"] == "voice_id_enroll_started"
        flow.hass.services.async_call.assert_not_awaited()  # nothing blocks the screen
        await _run_background(background)
    finally:
        ap.PIPELINES.clear()
    assert mgr.sessions.get("sat1").lang == "de"
    stream = tts.STREAMS[-1]
    assert (stream.engine, stream.language, stream.options) == ("tts.cloud", "de-DE", {"voice": "MajaNeural"})
    assert stream.message.endswith(ve.prompts_for("de")["questions"][0])
    calls = [c.args[:3] for c in flow.hass.services.async_call.await_args_list]
    assert [c[:2] for c in calls] == [
        ("mic_to_mediaplayer", "announce"), ("assist_satellite", "start_conversation"),
    ]
    assert calls[0][2] == {"satellite_entity_id": "assist_satellite.kitchen", "audio_url": stream.url}
    assert calls[1][2] == {
        "entity_id": "assist_satellite.kitchen", "start_message": "", "preannounce": False,
    }
    entity._remember_reply.assert_called_once()
    assert entity._remember_reply.call_args.args == ("sat1", stream.message)


async def test_satellite_language_is_the_default(satellites) -> None:
    tts = sys.modules["homeassistant.components.tts"]
    tts.STREAMS.clear()
    mgr, flow, entity, background = _routed_flow(answer_after_listen=True)
    real_state = flow.hass.states.get.side_effect
    flow.hass.states.get = MagicMock(
        side_effect=lambda eid: SimpleNamespace(state="preferred", attributes={"options": ["preferred"]})
        if eid == "select.kitchen_assistant" else real_state(eid)
    )
    ap.PIPELINES.extend(_PIPELINES)
    try:
        await flow.async_step_voice_id_enroll({"person": "person.anna", "satellite": "sat1"})
        await _run_background(background)
    finally:
        ap.PIPELINES.clear()
    assert mgr.sessions.get("sat1").lang == "en"
    assert tts.STREAMS[-1].options == {"voice": "LunaNeural"}


async def test_missed_first_answer_is_asked_again(satellites) -> None:
    """Measured 2026-10-24: the answer came before the microphone opened and
    the session simply stalled. Now the question is repeated, at most twice."""
    tts = sys.modules["homeassistant.components.tts"]
    tts.STREAMS.clear()
    mgr, flow, entity, background = _routed_flow(answer_after_listen=False)
    ap.PIPELINES.extend(_PIPELINES)
    try:
        await flow.async_step_voice_id_enroll({"person": "person.anna", "satellite": "sat1"})
        await _run_background(background)
    finally:
        ap.PIPELINES.clear()
    announces = [c for c in flow.hass.services.async_call.await_args_list if c.args[1] == "announce"]
    question = ve.prompts_for("de")["questions"][0]
    assert len(announces) == 3
    assert tts.STREAMS[1].message == f"Entschuldige, das habe ich nicht verstanden. {question}"
    assert mgr.sessions.active("sat1")  # still waiting; the wake word continues it


async def test_enroll_screen_falls_back_to_the_wake_word(satellites) -> None:
    mgr = _manager([ANNA])
    flow = _Flow(mgr)
    flow.hass.services.async_call = AsyncMock(side_effect=RuntimeError("offline"))
    result = await flow.async_step_voice_id_enroll({"person": "person.anna", "satellite": "sat1"})
    assert result["reason"] == "voice_id_enroll_armed"
    assert mgr.sessions.get("sat1").asked == 0


async def test_enroll_screen_requires_the_feature(satellites) -> None:
    mgr = _manager([ANNA], options={"voice_id_enabled": False})
    assert (await _Flow(mgr).async_step_voice_id_enroll())["reason"] == "voice_id_disabled"


async def test_delete_screen_removes_fingerprints(satellites) -> None:
    mgr = _manager([ANNA])
    await _enroll(mgr, "person.anna", _unit(1))
    flow = _Flow(mgr)
    menu = await flow.async_step_voice_id()
    assert "Anna: 3 samples" in menu["description_placeholders"]["status"]
    result = await flow.async_step_voice_id_delete({"person": "person.anna"})
    assert result["reason"] == "voice_id_deleted" and mgr.profile_summary() == {}


async def test_settings_screen_saves_options(satellites) -> None:
    mgr = _manager([ANNA])
    flow = _Flow(mgr)
    assert (await flow.async_step_voice_id_settings())["type"] == "form"
    result = await flow.async_step_voice_id_settings(
        {"voice_id_enabled": False, "voice_id_voice_enroll": True,
         "voice_id_threshold": 0.45, "voice_id_margin": 0.1}
    )
    assert result["type"] == "create_entry" and result["data"]["voice_id_threshold"] == 0.45


def test_armed_session_survives_an_entry_reload() -> None:
    """Saving any option reloads the entry and builds a new manager."""
    hass = _hass([ANNA])
    hass.data = {}
    entry = _make_mock_entry({"voice_id_enabled": True})
    VoiceIDManager(hass, entry).arm_enrollment("sat1", "person.anna", "en")
    assert VoiceIDManager(hass, entry).sessions.active("sat1")


def test_session_stops_claiming_turns_once_the_feature_is_off() -> None:
    """Without the feature there is no audio to sample: the satellite's turns
    and microphone must go back to normal instead of "that was a bit short"."""
    hass = _hass([ANNA])
    hass.data = {}
    VoiceIDManager(hass, _make_mock_entry({"voice_id_enabled": True})).start_enrollment(
        "sat1", "person.anna", "en"
    )
    off = VoiceIDManager(hass, _make_mock_entry({"voice_id_enabled": False}))
    assert off.sessions.active("sat1")
    assert not off.enrollment_active("sat1")
    assert not off.wants_follow_up("sat1")
