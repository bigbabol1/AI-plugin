"""Log-mel front end for the speaker-embedding model.

Reproduces sherpa-onnx's NeMo speaker path bit-for-bit (kaldi-native-fbank
with is_librosa): hann window, pre-emphasis 0.97 per frame, no DC removal,
snip_edges framing, 80 slaney-scale mel bins from 0 to 7600 Hz with slaney
normalisation, log with a 1.19e-7 floor, then per-feature mean/std
normalisation. Verified against sherpa-onnx embeddings at cosine 1.000000.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
_WIN = 400  # 25 ms
_HOP = 160  # 10 ms
_NFFT = 512
_NMEL = 80
_HIGH_HZ = SAMPLE_RATE / 2 - 400
_LOG_FLOOR = 1.1920929e-07

# Speech is trimmed to the span between the first and last voiced 20 ms
# frame: satellites keep recording until end-of-speech fires, so up to two
# thirds of a clip is silence. Analysis is capped at 4 s of speech: on 31 real
# satellite clips (2 speakers, leave-one-out) 4 s and 6 s gave identical
# decisions, 3 s lowered the closest margin from 0.15 to 0.10. 4 s costs about
# a third less time than 6 s, and once a user has said 4 s the fingerprint can
# start while they are still talking.
MAX_SPEECH_SECONDS = 4.0


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    log_part = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(f >= 1000.0, log_part, 3.0 * f / 200.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    log_part = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
    return np.where(m >= 15.0, log_part, 200.0 * m / 3.0)


def _mel_banks() -> np.ndarray:
    pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(_HIGH_HZ), _NMEL + 2))
    freqs = np.arange(_NFFT // 2 + 1) * SAMPLE_RATE / _NFFT
    banks = np.zeros((_NMEL, freqs.size))
    for b in range(_NMEL):
        left, centre, right = pts[b], pts[b + 1], pts[b + 2]
        rise = (freqs - left) / (centre - left)
        fall = (right - freqs) / (right - centre)
        banks[b] = np.maximum(0.0, np.minimum(rise, fall)) * (2.0 / (right - left))
    return banks


_BANKS = _mel_banks()
_WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_WIN) / _WIN)


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Little-endian 16-bit PCM bytes → float32 samples in [-1, 1)."""
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def trim_speech(audio: np.ndarray) -> np.ndarray:
    """Cut leading/trailing silence and cap at MAX_SPEECH_SECONDS."""
    hop = SAMPLE_RATE // 50
    frames = audio[: len(audio) // hop * hop].reshape(-1, hop)
    if frames.size:
        rms = np.sqrt((frames**2).mean(axis=1))
        voiced = np.flatnonzero(rms > max(float(rms.max()) * 0.1, 1e-3))
        if voiced.size:
            audio = audio[voiced[0] * hop : (voiced[-1] + 1) * hop]
    return audio[: int(MAX_SPEECH_SECONDS * SAMPLE_RATE)]


def fbank(audio: np.ndarray) -> np.ndarray:
    """Return normalised log-mel features, shape (frames, 80), float32.

    Raises ValueError when the audio is shorter than one 25 ms frame.
    """
    audio = np.asarray(audio, dtype=np.float64)
    n = 1 + (len(audio) - _WIN) // _HOP
    if n < 1:
        raise ValueError("audio shorter than one frame")
    idx = np.arange(_WIN)[None, :] + _HOP * np.arange(n)[:, None]
    frames = audio[idx]
    frames = np.concatenate(
        [frames[:, :1] * (1 - 0.97), frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1
    )
    spec = np.fft.rfft(frames * _WINDOW, n=_NFFT)
    mel = (spec.real**2 + spec.imag**2) @ _BANKS.T
    mel = np.log(np.maximum(mel, _LOG_FLOOR))
    mean = mel.mean(axis=0)
    std = np.sqrt(((mel - mean) ** 2).mean(axis=0))
    return ((mel - mean) / (std + 1e-5)).astype(np.float32)
