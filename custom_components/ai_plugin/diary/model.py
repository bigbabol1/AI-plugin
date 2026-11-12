"""Sentence embeddings for the conversation diary, in plain Python and numpy.

The model is potion-multilingual-128M (Model2Vec static embeddings, MIT),
cut down by scripts/convert_memory_model.py to the pieces written in Latin
script and stored as 4-bit values: 39.6 MB instead of 512 MB, with the same
recall on the diary test set. An embedding is the mean of the pieces of the
text, so one costs well under a millisecond and needs no extra library. The
file is fetched once from the repository's memory_model/ folder, checked
against its digest, and kept under <config>/.storage/ai_plugin_memory_model/.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import re
import unicodedata

import numpy as np

_LOGGER = logging.getLogger(__name__)

MODEL_BASE_URL = "https://raw.githubusercontent.com/bigbabol1/AI-plugin/main/memory_model/"
MODEL_FILE = "potion_latin_q4.npz"
MODEL_SHA256 = "e28b9ea3d72582e1bafe0dc1105100d4fb0bcb8b356f63566de7fb4cc6ba8997"
DIM = 256
# Longest piece tried at each position. Longer pieces exist but are rare whole
# phrases; capping keeps the search linear in the text.
_MAX_PIECE = 24

_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_PUNCT_RE = re.compile("([" + re.escape(_PUNCT) + "])")


class ModelUnavailable(Exception):
    """The model file is missing, corrupt or could not be fetched."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_ok(model_dir: Path) -> bool:
    path = model_dir / MODEL_FILE
    return path.is_file() and sha256_file(path) == MODEL_SHA256


async def async_download(session, model_dir: Path, hass) -> None:
    """Fetch the file when it is missing; raises ModelUnavailable on failure."""
    if await hass.async_add_executor_job(model_ok, model_dir):
        return
    url = MODEL_BASE_URL + MODEL_FILE
    _LOGGER.info("AI Plugin diary: downloading %s", url)
    try:
        async with session.get(url, timeout=600) as resp:
            resp.raise_for_status()
            data = await resp.read()
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailable(f"download failed: {exc}") from exc
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ModelUnavailable("checksum mismatch")

    def _write() -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / MODEL_FILE
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    await hass.async_add_executor_job(_write)


def normalize(text: str) -> str:
    """The tokenizer's own normalisation: NFKC, punctuation set apart."""
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RE.sub(r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


class Encoder:
    """Loaded model. ``embed`` is CPU-bound but fast (a dictionary walk)."""

    def __init__(self, pieces: list[str], scores: np.ndarray, packed: np.ndarray, scale: np.ndarray) -> None:
        self._index = {piece: i for i, piece in enumerate(pieces)}
        self._scores = scores.astype(np.float64).tolist()
        self._unk = min(self._scores) - 10.0
        self._packed = packed
        self._scale = scale.astype(np.float32)
        self._space = self._index.get("\u2581")

    @classmethod
    def load(cls, model_dir: Path) -> "Encoder":
        if not model_ok(model_dir):
            raise ModelUnavailable("missing or corrupt")
        with np.load(model_dir / MODEL_FILE) as data:
            pieces = data["pieces"].tobytes().decode("utf-8").split("\x00")
            return cls(pieces, data["scores"], data["packed"], data["scale"])

    def piece_ids(self, text: str) -> list[int]:
        """Unigram segmentation with the highest total score (Viterbi)."""
        s = "▁" + normalize(text).replace(" ", "▁")
        n = len(s)
        best = [float("-inf")] * (n + 1)
        best[0] = 0.0
        back: list[tuple[int, int]] = [(0, -1)] * (n + 1)
        index, scores = self._index, self._scores
        for i in range(n):
            here = best[i]
            if here == float("-inf"):
                continue
            hit = False
            for j in range(i + 1, min(n, i + _MAX_PIECE) + 1):
                k = index.get(s[i:j])
                if k is not None:
                    hit = True
                    if here + scores[k] > best[j]:
                        best[j] = here + scores[k]
                        back[j] = (i, k)
            # No piece starts here (a letter outside the kept scripts): skip
            # the character as unknown, which the embedding then leaves out.
            if (not hit or best[i + 1] == float("-inf")) and here + self._unk > best[i + 1]:
                best[i + 1] = here + self._unk
                back[i + 1] = (i, -1)
        ids, j = [], n
        while j > 0:
            i, k = back[j]
            if k >= 0:
                ids.append(k)
            j = i
        return ids[::-1]

    def embed(self, text: str) -> np.ndarray | None:
        """Unit-length embedding, or None when nothing in the text is known."""
        ids = self.piece_ids(text)
        # Only the word-start marker: "" or a script outside the kept ones.
        if not ids or all(i == self._space for i in ids):
            return None
        rows = self._packed[ids]
        low = (rows & 0x0F).astype(np.int8)
        high = (rows >> 4).astype(np.int8)
        low[low > 7] -= 16
        high[high > 7] -= 16
        values = np.empty((len(ids), DIM), dtype=np.float32)
        values[:, 0::2] = low
        values[:, 1::2] = high
        vec = (values * self._scale[ids, None]).mean(axis=0)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else None
