"""Convert potion-multilingual-128M into the conversation-memory model file.

Source: minishlab/potion-multilingual-128M on Hugging Face (Model2Vec static
embeddings, MIT licence): model.safetensors and tokenizer.json.

    pip install numpy safetensors
    python scripts/convert_memory_model.py <potion dir> memory_model/potion_latin_q4.npz

Keeps only the vocabulary pieces written in Latin script (plus digits and
punctuation): 274,057 of 500,353, which covers every language AI Plugin speaks.
Each embedding row is stored as signed 4-bit values with one float16 scale per
row. On the diary recall test set (60 follow-ups, English and German) this
keeps recall@3 at 0.92, the same as the full float32 model, at 35 MB instead of
512 MB. A new version needs a new file name in memory_model/ and a new digest
in memory/model.py.
"""

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_EXTRA = "€£°§–—’‘“”„«»…"


def latin_ok(piece: str) -> bool:
    for ch in piece:
        if ch == "▁" or ch.isdigit() or ch in _PUNCT or ch in _EXTRA:
            continue
        if not unicodedata.name(ch, "").startswith("LATIN"):
            return False
    return True


def main(src: str, out: str) -> None:
    src_dir = Path(src)
    vocab = json.loads((src_dir / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    emb = load_file(str(src_dir / "model.safetensors"))["embeddings"]
    # 0 and 1 are [PAD] and [UNK]; Model2Vec drops [UNK] from every sentence.
    keep = [i for i, (piece, _) in enumerate(vocab) if i > 1 and latin_ok(piece)]
    pieces = [vocab[i][0] for i in keep]
    assert not any("\x00" in p for p in pieces)
    scores = np.array([vocab[i][1] for i in keep], dtype=np.float32)
    rows = emb[keep].astype(np.float32)
    scale = np.abs(rows).max(axis=1) / 7.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(rows / scale[:, None]), -7, 7).astype(np.int8)
    nib = (q & 0x0F).astype(np.uint8)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    np.savez(
        out,
        pieces=np.frombuffer("\x00".join(pieces).encode("utf-8"), dtype=np.uint8),
        scores=scores,
        packed=packed,
        scale=scale.astype(np.float16),
    )
    print(f"{len(pieces)} pieces, {Path(out).stat().st_size / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
