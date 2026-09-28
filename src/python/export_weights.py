"""Excerpt from the training pipeline (mlp_train.py) - exporting weights for the browser.
Author: Hadi Sarhangi Fard

Two things happen here:

1. Every dense layer's weight and bias tensors are cast to Float16 and
   Base64-encoded, so the 3D visualizer (see ../js/core/weights-decoder.js)
   can decode them back to Float32 with a small, fast download.
2. Each training "milestone" (for example "~50 images", "epoch 1",
   "50x dataset") gets its own snapshot file, so the app's training-progress
   slider can load a fully independent set of weights for any point in time.
"""
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LayerSnapshot:
    layer_index: int
    name: str
    activation: str
    weight: torch.Tensor
    bias: torch.Tensor


def capture_layer_snapshots(model: nn.Module, activations: list[str]) -> list[LayerSnapshot]:
    """Pull the current parameters out of every nn.Linear layer, in order."""
    dense_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    return [
        LayerSnapshot(
            layer_index=idx,
            name=f"dense_{idx}",
            activation=activation,
            weight=layer.weight.detach().cpu(),
            bias=layer.bias.detach().cpu(),
        )
        for idx, (layer, activation) in enumerate(zip(dense_layers, activations))
    ]


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Float32 tensor -> Float16 -> raw little-endian bytes -> Base64 string."""
    array = tensor.detach().cpu().to(torch.float16).numpy()
    little_endian = np.ascontiguousarray(array.astype("<f2", copy=False)).view("<u2")
    return base64.b64encode(little_endian.tobytes()).decode("ascii")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "snapshot"


def write_snapshot_file(snapshots: list[LayerSnapshot], directory: Path, order: int, identifier: str) -> Path:
    """One JSON file per timeline milestone: every layer's Float16/Base64 weights and biases."""
    path = directory / f"{order:03d}_{slugify(identifier)}.json"
    payload = {
        "version": 1,
        "dtype": "float16",
        "layers": [
            {
                "layer_index": s.layer_index,
                "name": s.name,
                "activation": s.activation,
                "weights": {"shape": list(s.weight.shape), "data": tensor_to_base64(s.weight)},
                "biases": {"shape": list(s.bias.shape), "data": tensor_to_base64(s.bias)},
            }
            for s in snapshots
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path


def export_model_metadata(output_path: Path, architecture: list[int], normalization: dict, timeline: list[dict]) -> None:
    """The small top-level JSON: architecture, normalization stats and the
    list of timeline entries, each pointing at one snapshot file on disk."""
    payload = {
        "version": 2,
        "dtype": "float16",
        "weights": {"storage": "per_snapshot_files", "format": "layer_array_v1", "precision": "float16"},
        "network": {"architecture": architecture, "normalization": normalization},
        "timeline": timeline,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
