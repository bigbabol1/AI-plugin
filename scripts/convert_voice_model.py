"""Convert the TitaNet-small ONNX export into the voice-ID model files.

Source: nemo_en_titanet_small.onnx from the sherpa-onnx speaker-recognition
model release (NVIDIA TitaNet-small, Apache-2.0 via the NeMo toolkit).

    pip install onnx numpy
    python scripts/convert_voice_model.py nemo_en_titanet_small.onnx out/titanet_small

writes out/titanet_small.json (graph) and out/titanet_small_f16.npz
(weights). Only the nodes feeding the ``embs`` output are kept, which drops
the 3.2M-parameter speaker-classification head; weight tensors larger than
1000 values are stored as float16 (identical decisions on real clips) and
upcast at load. The published copies are in voice_model/ (with the
Apache-2.0 licence and a note of these changes); a new version needs new
file names there and new MODEL_FILES digests in voice_id/model.py.
"""

import json
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main(src: str, out: str) -> None:
    graph = onnx.load(src).graph
    producers = {o: n for n in graph.node for o in n.output}
    needed, stack = set(), ["embs"]
    while stack:
        node = producers.get(stack.pop())
        if node is None or id(node) in needed:
            continue
        needed.add(id(node))
        stack.extend(i for i in node.input if i)

    initializers = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    weights, nodes, used = {}, [], set()
    for node in graph.node:
        if id(node) not in needed:
            continue
        attrs = {}
        for attr in node.attribute:
            value = helper.get_attribute_value(attr)
            if isinstance(value, onnx.TensorProto):
                key = f"__const_{len(weights)}"
                weights[key] = numpy_helper.to_array(value)
                value = {"tensor": key}
            elif isinstance(value, bytes):
                value = value.decode()
            elif isinstance(value, (list, tuple)):
                value = [v.decode() if isinstance(v, bytes) else v for v in value]
            attrs[attr.name] = value
        nodes.append({"op": node.op_type, "in": list(node.input), "out": list(node.output), "attrs": attrs})
        used.update(node.input)
    for name, value in initializers.items():
        if name in used:
            weights[name] = value

    stored = {
        k: v.astype(np.float16) if v.dtype == np.float32 and v.size > 1000 else v
        for k, v in weights.items()
    }
    np.savez(out + "_f16.npz", **stored)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump({"inputs": [i.name for i in graph.input], "output": "embs", "nodes": nodes}, f)
    print(f"{len(nodes)} nodes, {sum(v.size for v in weights.values())} parameters")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
