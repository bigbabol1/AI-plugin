"""Minimal numpy interpreter for the converted speaker-embedding graph.

Home Assistant's container (Alpine/musl) has no onnxruntime wheel, but it
ships numpy. The model's ONNX graph is converted offline (scripts/
convert_voice_model.py) into graph.json + weights.npz; this module executes
the 33 operator types that graph uses (ONNX opset 17 semantics). Outputs
match onnxruntime at cosine 1.000000.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

_CAST = {1: np.float32, 6: np.int32, 7: np.int64, 9: np.bool_, 11: np.float64}

_ELEMENTWISE = {
    "Add": np.add,
    "Sub": np.subtract,
    "Mul": np.multiply,
    "Pow": np.power,
    "Equal": np.equal,
    "Less": np.less,
    "Not": np.logical_not,
    "Relu": lambda v: np.maximum(v, 0),
    "Sigmoid": lambda v: 1 / (1 + np.exp(-v)),
    "Tanh": np.tanh,
    "Sqrt": np.sqrt,
}


def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None, attrs: dict) -> np.ndarray:
    pad_l, pad_r = attrs["pads"]
    group = attrs["group"]
    dil = attrs["dilations"][0]
    if attrs["strides"][0] != 1:
        raise NotImplementedError("only stride 1")
    k = w.shape[2]
    x = np.pad(x, ((0, 0), (0, 0), (pad_l, pad_r)))
    t_out = x.shape[2] - dil * (k - 1)
    if group == 1:
        y = sum(
            np.einsum("oc,nct->not", w[:, :, j], x[:, :, j * dil : j * dil + t_out], optimize=True)
            for j in range(k)
        )
    elif group == x.shape[1] == w.shape[0]:  # depthwise
        y = sum(w[None, :, 0, j, None] * x[:, :, j * dil : j * dil + t_out] for j in range(k))
    else:
        raise NotImplementedError(f"conv group={group}")
    if b is not None:
        y = y + b[None, :, None]
    return y.astype(np.float32, copy=False)


def _opt(inputs: list, i: int) -> Any:
    return inputs[i] if len(inputs) > i else None


class Graph:
    """A converted graph plus weights; ``run`` evaluates it on numpy arrays."""

    def __init__(self, graph: dict, weights: dict[str, np.ndarray]) -> None:
        self.nodes: list[dict] = graph["nodes"]
        self.output: str = graph["output"]
        # float16 storage halves the download; compute stays float32.
        self.weights = {
            k: v.astype(np.float32) if v.dtype == np.float16 else v
            for k, v in weights.items()
        }
        # Free each intermediate after its last consumer: the pooling stage
        # holds 3072- and 9216-channel tensors, and keeping every one alive
        # cost ~400 MB on a 13 s clip.
        last: dict[str, int] = {}
        for i, node in enumerate(self.nodes):
            for name in node["in"]:
                last[name] = i
        self._free_after: list[list[str]] = [[] for _ in self.nodes]
        for name, i in last.items():
            if name not in self.weights and name != self.output:
                self._free_after[i].append(name)

    @classmethod
    def load(cls, json_path: str, npz_path: str) -> "Graph":
        with open(json_path, encoding="utf-8") as f:
            graph = json.load(f)
        with np.load(npz_path) as z:
            weights = {k: z[k] for k in z.files}
        return cls(graph, weights)

    def run(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        env: dict[str, Any] = dict(self.weights)
        env.update(feed)
        for idx, node in enumerate(self.nodes):
            out = np.asarray(self._eval(node, env))
            if out.dtype == np.float64:
                out = out.astype(np.float32)
            env[node["out"][0]] = out
            for name in self._free_after[idx]:
                env.pop(name, None)
        return env[self.output]

    def _eval(self, node: dict, env: dict) -> Any:  # noqa: C901, PLR0911, PLR0912
        op, a = node["op"], node["attrs"]
        x = [env[i] if i else None for i in node["in"]]
        if op == "Constant":
            return self.weights[a["value"]["tensor"]]
        if op == "Conv":
            return _conv1d(x[0], x[1], _opt(x, 2), a)
        if op == "BatchNormalization":
            data, scale, bias, mean, var = x
            shape = (1, -1) + (1,) * (data.ndim - 2)
            out = (data - mean.reshape(shape)) / np.sqrt(
                var.reshape(shape) + a.get("epsilon", 1e-5)
            )
            return (out * scale.reshape(shape) + bias.reshape(shape)).astype(np.float32)
        if op == "Shape":
            return np.array(x[0].shape, np.int64)
        if op == "Gather":
            return np.take(x[0], x[1], axis=a.get("axis", 0))
        if op == "Unsqueeze":
            out = x[0]
            rank = x[0].ndim + len(x[1])
            for ax in sorted(int(v) % rank for v in x[1]):
                out = np.expand_dims(out, ax)
            return out
        if op == "Squeeze":
            axes = _opt(x, 1)
            return np.squeeze(x[0], axis=tuple(int(v) for v in axes)) if axes is not None else np.squeeze(x[0])
        if op == "Slice":
            data, starts, ends = x[0], x[1], x[2]
            axes = _opt(x, 3)
            steps = _opt(x, 4)
            axes = np.arange(len(starts)) if axes is None else axes
            steps = np.ones(len(starts), np.int64) if steps is None else steps
            sl = [slice(None)] * data.ndim
            for st, en, ax, sp in zip(starts, ends, axes, steps):
                sl[int(ax)] = slice(int(st), int(min(en, np.iinfo(np.int64).max)), int(sp))
            return data[tuple(sl)]
        if op == "Concat":
            return np.concatenate(x, axis=a["axis"])
        if op == "Reshape":
            shape = [
                x[0].shape[i] if (v == 0 and not a.get("allowzero")) else int(v)
                for i, v in enumerate(x[1])
            ]
            return x[0].reshape(shape)
        if op == "ConstantOfShape":
            value = self.weights[a["value"]["tensor"]]
            return np.full(tuple(int(s) for s in x[0]), value.reshape(-1)[0], dtype=value.dtype)
        if op == "Expand":
            return x[0] * np.ones(tuple(int(s) for s in x[1]), dtype=x[0].dtype)
        if op == "Tile":
            return np.tile(x[0], tuple(int(s) for s in x[1]))
        if op == "Range":
            return np.arange(x[0], x[1], x[2]).astype(np.result_type(x[0]))
        if op == "Cast":
            return x[0].astype(_CAST[a["to"]])
        if op == "Where":
            return np.where(x[0], x[1], x[2])
        if op == "ReduceSum":
            axes = _opt(x, 1)
            axes = tuple(int(v) for v in axes) if axes is not None else None
            return np.sum(x[0], axis=axes, keepdims=bool(a.get("keepdims", 1)))
        if op == "Transpose":
            return np.transpose(x[0], a["perm"])
        if op == "MatMul":
            return np.matmul(x[0], x[1])
        if op == "Softmax":
            axis = a.get("axis", -1)
            e = np.exp(x[0] - x[0].max(axis=axis, keepdims=True))
            return e / e.sum(axis=axis, keepdims=True)
        if op == "Clip":
            lo = _opt(x, 1)
            hi = _opt(x, 2)
            return np.clip(x[0], -np.inf if lo is None else lo, np.inf if hi is None else hi)
        if op == "Div":
            if np.issubdtype(x[0].dtype, np.integer):
                return x[0] // x[1]
            return x[0] / x[1]
        if op in _ELEMENTWISE:
            return _ELEMENTWISE[op](*x)
        raise NotImplementedError(f"ONNX op {op}")
