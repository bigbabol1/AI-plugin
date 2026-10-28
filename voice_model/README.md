# Voice recognition model

These files are the speaker-embedding model used by the AI Plugin's optional
voice recognition. They are **not** part of the HACS download: the plugin
fetches them from this folder the first time voice recognition is switched
on, checks them against SHA-256 digests pinned in
`custom_components/ai_plugin/voice_id/model.py`, and stores them under
`<config>/.storage/ai_plugin_voice/`.

| File | SHA-256 |
|---|---|
| `titanet_small.json` | `e53db2e3ba694c645a106b34599d4623c2532481bfc7c268f3b1582c909df427` |
| `titanet_small_f16.npz` | `a44a1ba3492da72b3b0ead45290b040e93e932f654cd56147d6f1227e1716f14` |

## Licence

**This folder is licensed under the Apache License 2.0 (see `LICENSE`), not
under the PolyForm Noncommercial licence that covers the rest of this
repository.**

The model is NVIDIA TitaNet-small (`titanet_small` v1.19.0 on NVIDIA NGC),
whose model card places it under the licence of the NVIDIA NeMo toolkit,
Apache-2.0. Source of the weights: the ONNX export
`nemo_en_titanet_small.onnx` from the sherpa-onnx speaker-recognition model
release.

## Changes made to the original

Produced by `scripts/convert_voice_model.py`:

- The ONNX graph was converted to a JSON node list plus a NumPy `.npz`
  weight file, so it can run on NumPy alone inside Home Assistant.
- Only the nodes feeding the speaker-embedding output (`embs`) were kept;
  the speaker-classification head was removed.
- Weight tensors with more than 1000 values are stored as float16 and
  converted back to float32 when loaded.

Embeddings from these files match the original ONNX model (cosine 1.0000
on test signals and real recordings).
