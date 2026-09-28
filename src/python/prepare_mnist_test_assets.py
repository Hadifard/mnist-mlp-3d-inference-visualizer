"""Excerpt from the tooling pipeline (mnist_to_web.py) - MNIST test set for the browser.
Author: Hadi Sarhangi Fard

The "Draw here" panel's digit buttons (0-9) load a random real MNIST test
image. Rather than shipping 10,000 individual PNGs, the raw IDX test files
are converted once into two flat binary blobs (images and labels) plus a
small JSON manifest describing their shape - a fetch() of a manifest and
two .bin files is enough for the browser to reconstruct any test sample.
"""
import json
import struct
from pathlib import Path

MNIST_MAGIC_IMAGES = 2051
MNIST_MAGIC_LABELS = 2049


def load_idx(path: Path, expected_magic: int) -> tuple[tuple[int, ...], bytes]:
    """Parse an IDX file's header and return (shape, raw pixel/label bytes)."""
    raw = path.read_bytes()
    magic, = struct.unpack_from(">I", raw, 0)
    if magic != expected_magic:
        raise ValueError(f"{path} has magic {magic}, expected {expected_magic}.")

    num_dimensions = raw[3]
    offset = 4
    shape = []
    for _ in range(num_dimensions):
        size = struct.unpack_from(">I", raw, offset)[0]
        shape.append(size)
        offset += 4
    return tuple(shape), raw[offset:]


def write_web_assets(image_shape, image_bytes: bytes, label_bytes: bytes, output_dir: Path) -> None:
    """uint8 pixels/labels stay uint8: the browser normalises to [0, 1] itself
    (see feedforward-model.js), which keeps this export about 7.8 MB total
    for all 10,000 test images instead of a much larger float format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mnist-test-images-uint8.bin").write_bytes(image_bytes)
    (output_dir / "mnist-test-labels-uint8.bin").write_bytes(label_bytes)

    manifest = {
        "version": 1,
        "numSamples": image_shape[0],
        "imageShape": [image_shape[1], image_shape[2]],
        "image": {"file": "mnist-test-images-uint8.bin", "dtype": "uint8"},
        "labels": {"file": "mnist-test-labels-uint8.bin", "dtype": "uint8"},
    }
    (output_dir / "mnist-test-manifest.json").write_text(json.dumps(manifest, indent=2))
