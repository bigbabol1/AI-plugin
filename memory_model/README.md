# Conversation memory model

This file is the sentence-embedding model AI Plugin uses to find earlier
conversations in its diary. It is **not** part of the HACS download: the plugin
fetches it from this folder the first time the diary is on, checks it against
the SHA-256 digest pinned in `custom_components/ai_plugin/diary/model.py`, and
stores it under `<config>/.storage/ai_plugin_memory_model/`. Until it is there,
notes are found by their words.

| File | SHA-256 |
|---|---|
| `potion_latin_q4.npz` | `e28b9ea3d72582e1bafe0dc1105100d4fb0bcb8b356f63566de7fb4cc6ba8997` |

## Licence

**This folder is licensed under the MIT License (see `LICENSE`), not under the
PolyForm Noncommercial licence that covers the rest of this repository.**

The model is [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M)
by the Minish Lab (Model2Vec static embeddings), published under the MIT
License. `LICENSE` is the Model2Vec licence text.

## Changes made to the original

Produced by `scripts/convert_memory_model.py`:

- Only the vocabulary pieces written in Latin script (plus digits and
  punctuation) are kept: 274,057 of 500,353. Every language AI Plugin speaks
  is written in Latin script.
- Each embedding row is stored as signed 4-bit values with one float16 scale
  per row, instead of float32.
- The tokenizer's vocabulary and scores are stored next to the weights, so the
  plugin needs no tokenizer library.

On the plugin's diary recall test (60 follow-up questions in English and
German), recall@3 is 0.92, the same as the original model; the file is 39.6 MB
instead of 512 MB.
