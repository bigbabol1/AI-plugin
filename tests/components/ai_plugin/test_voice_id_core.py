"""Voice recognition core: features, numpy runner, buffers, matching, enrollment.

The full-model parity test runs the model files in the repository's
voice_model/ folder (AI_PLUGIN_VOICE_MODEL_DIR overrides the location). Its
reference embeddings were produced by sherpa-onnx (onnxruntime) on the same
synthetic signals.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from custom_components.ai_plugin.voice_id import buffer as vb
from custom_components.ai_plugin.voice_id import enrollment as ve
from custom_components.ai_plugin.voice_id import features as vf
from custom_components.ai_plugin.voice_id import matching as vm
from custom_components.ai_plugin.voice_id.runner import Graph

FIXTURES = Path(__file__).parent / "fixtures"


def _synth(seed: int, seconds: float) -> bytes:
    """Deterministic voiced test signal; must match the reference script."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(16000 * seconds)) / 16000
    f0 = 110 + 40 * seed + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / 16000
    voice = sum(np.sin(k * phase) / k for k in range(1, 12)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x = 0.2 * voice / np.abs(voice).max() + 0.01 * rng.standard_normal(t.size)
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


# ── features ──────────────────────────────────────────────────────────────────


def test_fbank_frames_and_normalisation() -> None:
    audio = vf.pcm16_to_float(_synth(1, 1.0))
    feats = vf.fbank(audio)
    # snip_edges framing: 1 + (16000 - 400) // 160 frames, no padding.
    assert feats.shape == (98, 80)
    assert feats.dtype == np.float32
    np.testing.assert_allclose(feats.mean(axis=0), 0.0, atol=1e-4)


def test_fbank_rejects_audio_shorter_than_a_frame() -> None:
    with pytest.raises(ValueError):
        vf.fbank(np.zeros(100, dtype=np.float32))


def test_trim_speech_cuts_silence_and_caps_length() -> None:
    speech = vf.pcm16_to_float(_synth(2, 8.0))
    padded = np.concatenate([np.zeros(16000, np.float32), speech, np.zeros(16000, np.float32)])
    trimmed = vf.trim_speech(padded)
    assert len(trimmed) == int(vf.MAX_SPEECH_SECONDS * 16000)
    assert np.abs(trimmed[:320]).max() > 0.01  # starts on speech, not silence


# ── runner ────────────────────────────────────────────────────────────────────


def _conv_graph(weight: np.ndarray, group: int, pads: list[int]) -> Graph:
    return Graph(
        {
            "output": "y",
            "nodes": [
                {
                    "op": "Conv",
                    "in": ["x", "w", "b"],
                    "out": ["y"],
                    "attrs": {
                        "pads": pads, "group": group, "dilations": [1],
                        "strides": [1], "kernel_shape": [weight.shape[2]],
                    },
                }
            ],
        },
        {"w": weight, "b": np.arange(weight.shape[0], dtype=np.float32)},
    )


def _reference_conv(x, w, b, group, pads):
    x = np.pad(x, ((0, 0), (0, 0), tuple(pads)))
    n, c, t = x.shape
    o, cg, k = w.shape
    out = np.zeros((n, o, t - k + 1), np.float64)
    per_group = o // group
    for oc in range(o):
        g = oc // per_group
        for ic in range(cg):
            for j in range(k):
                out[:, oc] += w[oc, ic, j] * x[:, g * cg + ic, j : j + t - k + 1]
    return out + b[None, :, None]


@pytest.mark.parametrize(("group", "k"), [(1, 1), (1, 3), (4, 5)])
def test_conv_matches_reference(group: int, k: int) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 4, 20)).astype(np.float32)
    w = rng.standard_normal((4, 4 // group, k)).astype(np.float32)
    pads = [k // 2, k // 2]
    y = _conv_graph(w, group, pads).run({"x": x})
    expected = _reference_conv(x, w, np.arange(4, dtype=np.float32), group, pads)
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)


def test_runner_frees_intermediates_and_handles_shape_ops() -> None:
    g = Graph(
        {
            "output": "out",
            "nodes": [
                {"op": "Shape", "in": ["x"], "out": ["s"], "attrs": {}},
                {"op": "Gather", "in": ["s", "i2"], "out": ["t"], "attrs": {"axis": 0}},
                {"op": "Unsqueeze", "in": ["t", "ax0"], "out": ["t1"], "attrs": {}},
                {"op": "ReduceSum", "in": ["x", "ax2"], "out": ["r"], "attrs": {"keepdims": 0}},
                {"op": "Cast", "in": ["t1"], "out": ["tf"], "attrs": {"to": 1}},
                {"op": "Div", "in": ["r", "tf"], "out": ["mean"], "attrs": {}},
                {"op": "Softmax", "in": ["mean"], "out": ["out"], "attrs": {"axis": 1}},
            ],
        },
        {
            "i2": np.array(2, np.int64),
            "ax0": np.array([0], np.int64),
            "ax2": np.array([2], np.int64),
        },
    )
    x = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    out = g.run({"x": x})
    mean = x.mean(axis=2)
    expected = np.exp(mean - mean.max()) / np.exp(mean - mean.max()).sum()
    np.testing.assert_allclose(out, expected, rtol=1e-6)
    # Every intermediate is scheduled for release after its last consumer.
    freed = {name for names in g._free_after for name in names}
    assert {"s", "t", "t1", "r", "tf", "mean"} <= freed


def test_unknown_op_raises() -> None:
    g = Graph({"output": "y", "nodes": [{"op": "Erf", "in": ["x"], "out": ["y"], "attrs": {}}]}, {})
    with pytest.raises(NotImplementedError):
        g.run({"x": np.zeros(2, np.float32)})


_MODEL_DIR = os.environ.get("AI_PLUGIN_VOICE_MODEL_DIR") or str(
    Path(__file__).resolve().parents[3] / "voice_model"
)


@pytest.mark.skipif(
    not (Path(_MODEL_DIR) / "titanet_small.json").is_file(), reason="model files not present"
)
def test_full_model_matches_onnxruntime_reference() -> None:
    from custom_components.ai_plugin.voice_id.model import Embedder

    embedder = Embedder.load(Path(_MODEL_DIR))
    refs = json.loads((FIXTURES / "voice_id_reference.json").read_text())
    for ref in refs:
        vec, seconds = embedder.embed(_synth(ref["seed"], ref["seconds"]))
        assert vec is not None and seconds > 1.0
        assert float(vec @ np.asarray(ref["embedding"], np.float32)) > 0.9999


# ── buffer ────────────────────────────────────────────────────────────────────


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_buffer_matches_transcript_ignoring_case_and_punctuation() -> None:
    store = vb.AudioBufferStore()
    store.put("What's the weather, today?", b"\x01\x02")
    assert store.take("what s the weather today") == b"\x01\x02"
    assert store.take("what s the weather today") is None  # taken once


def test_buffer_carries_the_early_fingerprint() -> None:
    store = vb.AudioBufferStore()
    marker = object()
    store.put("hello there", b"\x01\x02", marker)
    assert store.take_clip("Hello there!") == (b"\x01\x02", marker)
    store.put("hello there", b"\x01\x02")
    assert store.take_clip("hello there") == (b"\x01\x02", None)


def test_buffer_expires() -> None:
    clock = _Clock()
    store = vb.AudioBufferStore(clock)
    store.put("hello there", b"\x00\x00")
    clock.now += vb.TTL_SECONDS + 1
    assert store.take("hello there") is None


def test_buffer_drops_ambiguous_duplicates() -> None:
    """Two rooms saying the same words at once: neither clip is usable."""
    store = vb.AudioBufferStore()
    store.put("stop", b"\x01\x00")
    store.put("stop", b"\x02\x00")
    assert store.take("stop") is None


def test_buffer_is_bounded() -> None:
    store = vb.AudioBufferStore()
    for i in range(vb.MAX_ENTRIES + 3):
        store.put(f"turn {i}", b"\x00" * (vb.MAX_PCM_BYTES + 10))
    assert len(store) == vb.MAX_ENTRIES
    assert len(store.take(f"turn {vb.MAX_ENTRIES + 2}")) == vb.MAX_PCM_BYTES
    assert store.take("turn 0") is None


# ── matching ──────────────────────────────────────────────────────────────────


def _unit(*vals: float) -> np.ndarray:
    v = np.zeros(192, np.float32)
    v[: len(vals)] = vals
    return v / np.linalg.norm(v)


def _samples(vec: np.ndarray, group: str, n: int = 3) -> list[vm.Sample]:
    return [vm.Sample(vec, group) for _ in range(n)]


def test_decide_names_the_clear_winner() -> None:
    anna, ben = _unit(1, 0), _unit(0, 1)
    d = vm.decide(_unit(1, 0.1), {"person.anna": _samples(anna, "a"), "person.ben": _samples(ben, "a")}, "a")
    assert d.person_id == "person.anna" and d.reason == "match"


def test_decide_unknown_below_threshold() -> None:
    d = vm.decide(_unit(0, 0, 1), {"person.anna": _samples(_unit(1, 0), "a")}, "a")
    assert d.person_id is None and d.reason == "below_threshold"


def test_decide_unknown_when_two_people_are_too_close() -> None:
    probe = _unit(1, 1)
    d = vm.decide(
        probe,
        {"person.anna": _samples(_unit(1, 0.9), "a"), "person.ben": _samples(_unit(0.9, 1), "a")},
        "a",
    )
    assert d.person_id is None and d.reason == "too_close"


def test_decide_prefers_samples_from_the_same_microphone() -> None:
    """Anna's bathroom samples decide a bathroom clip, not her ReSpeaker ones."""
    probe = _unit(0, 1)
    anna = _samples(_unit(1, 0), "respeaker") + _samples(_unit(0.1, 1), "waveshare")
    assert vm.person_score(probe, anna, "waveshare") > 0.99
    assert vm.person_score(probe, anna, "respeaker") < vm.person_score(probe, anna, "waveshare")


def test_decide_without_profiles() -> None:
    assert vm.decide(_unit(1), {}, "a").reason == "no_profiles"


# ── enrollment ────────────────────────────────────────────────────────────────


def test_enrollment_session_runs_to_target() -> None:
    sessions = ve.EnrollmentSessions(_Clock())
    first = sessions.start("sat1", "person.anna", "Anna", "en")
    assert "Anna" in first and first.endswith("?")
    session = sessions.get("sat1")
    for i in range(ve.TARGET_SAMPLES - 1):
        reply, done = sessions.answer(session, True)
        assert not done
    reply, done = sessions.answer(session, True)
    assert done and str(ve.TARGET_SAMPLES) in reply
    assert not sessions.active("sat1")


def test_enrollment_short_answer_does_not_count() -> None:
    sessions = ve.EnrollmentSessions(_Clock())
    sessions.start("sat1", "person.anna", "Anna", "de")
    session = sessions.get("sat1")
    reply, done = sessions.answer(session, False)
    assert not done and session.saved == 0 and "kurz" in reply


def test_armed_session_waits_longer_for_the_wake_word() -> None:
    clock = _Clock()
    sessions = ve.EnrollmentSessions(clock)
    sessions.arm("sat1", "person.anna", "Anna", "en")
    clock.now += ve.IDLE_TIMEOUT_SECONDS + 1
    assert sessions.active("sat1")
    clock.now += ve.ARMED_TIMEOUT_SECONDS
    assert not sessions.active("sat1")


def test_enrollment_is_per_satellite_and_times_out() -> None:
    clock = _Clock()
    sessions = ve.EnrollmentSessions(clock)
    sessions.start("sat1", "person.anna", "Anna", "en")
    assert not sessions.active("sat2")
    clock.now += ve.IDLE_TIMEOUT_SECONDS + 1
    assert not sessions.active("sat1")


@pytest.mark.parametrize(
    ("text", "lang", "stop"),
    [("Stopp.", "de", True), ("cancel", "de", True), ("abbrechen bitte", "de", True),
     ("I stopped at the bakery and bought bread", "en", False), ("", "en", False)],
)
def test_stop_words(text: str, lang: str, stop: bool) -> None:
    assert ve.is_stop(text, lang) is stop


def test_every_language_has_enough_questions() -> None:
    for lang, p in ve._PROMPTS.items():
        assert len(p["questions"]) >= ve.TARGET_SAMPLES, lang
        p["start"].format(name="X", n=8)
        p["done"].format(name="X", n=8)
        p["stopped"].format(n=3)
