"""Speaker-embedding model: download once, verify, load, embed.

The model is NVIDIA TitaNet-small (Apache-2.0, via the NeMo toolkit),
converted from the sherpa-onnx ONNX export to graph.json + float16 weights
(classifier head removed) by scripts/convert_voice_model.py; see
voice_model/README.md. It is not part of the HACS download: the files are
fetched on first enable, checked against pinned SHA-256 digests, and kept
under <config>/.storage.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from .features import SAMPLE_RATE, fbank, pcm16_to_float, trim_speech
from .runner import Graph

_LOGGER = logging.getLogger(__name__)

MODEL_DIR_NAME = "ai_plugin_voice"
# The files live in the repository's voice_model/ folder (Apache-2.0, outside
# the HACS zip). Fetched from the main branch, not a release or commit: those
# can be deleted, and the pinned digests below reject any changed file.
# A new model version must use new file names.
MODEL_BASE_URL = "https://raw.githubusercontent.com/bigbabol1/AI-plugin/main/voice_model/"
MODEL_FILES: dict[str, str] = {
    "titanet_small.json": "e53db2e3ba694c645a106b34599d4623c2532481bfc7c268f3b1582c909df427",
    "titanet_small_f16.npz": "a44a1ba3492da72b3b0ead45290b040e93e932f654cd56147d6f1227e1716f14",
}
# Below this much speech the fingerprint is noise; such clips are skipped
# for identification and refused as enrollment samples.
MIN_SPEECH_SECONDS = 0.5


class ModelUnavailable(Exception):
    """The model files are missing or failed verification."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def missing_files(model_dir: Path) -> list[str]:
    """Names of model files that are absent or fail their digest."""
    bad = []
    for name, digest in MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            bad.append(name)
    return bad


async def async_download(session, model_dir: Path, hass) -> None:
    """Fetch every missing file; raises ModelUnavailable on any failure."""
    bad = await hass.async_add_executor_job(missing_files, model_dir)
    if not bad:
        return
    await hass.async_add_executor_job(lambda: model_dir.mkdir(parents=True, exist_ok=True))
    for name in bad:
        url = MODEL_BASE_URL + name
        _LOGGER.info("AI Plugin voice ID: downloading %s", url)
        try:
            async with session.get(url, timeout=300) as resp:
                resp.raise_for_status()
                data = await resp.read()
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailable(f"download of {name} failed: {exc}") from exc
        if hashlib.sha256(data).hexdigest() != MODEL_FILES[name]:
            raise ModelUnavailable(f"{name}: checksum mismatch")

        def _write(path: Path = model_dir / name, payload: bytes = data) -> None:
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(payload)
            os.replace(tmp, path)

        await hass.async_add_executor_job(_write)


class Embedder:
    """Loaded model. ``embed`` is CPU-bound: call it from an executor."""

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    @classmethod
    def load(cls, model_dir: Path) -> "Embedder":
        bad = missing_files(model_dir)
        if bad:
            raise ModelUnavailable(f"missing or corrupt: {', '.join(bad)}")
        return cls(
            Graph.load(
                str(model_dir / "titanet_small.json"),
                str(model_dir / "titanet_small_f16.npz"),
            )
        )

    def embed(self, pcm: bytes) -> tuple[np.ndarray | None, float]:
        """Return (unit-norm fingerprint, seconds of speech analysed).

        The fingerprint is None when there is too little speech.
        """
        audio = trim_speech(pcm16_to_float(pcm))
        seconds = len(audio) / SAMPLE_RATE
        if seconds < MIN_SPEECH_SECONDS:
            return None, seconds
        feats = fbank(audio)
        vec = self._graph.run(
            {
                "audio_signal": feats.T[None].astype(np.float32),
                "length": np.array([len(feats)], np.int64),
            }
        )[0]
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm == 0.0:
            return None, seconds
        return (vec / norm).astype(np.float32), seconds
