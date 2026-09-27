"""Safetensors files whose tensors start on 64-byte boundaries.

The app maps weight files rather than reading them, and views each tensor in
place (whisper/MappedWeights.swift), so alignment is what makes a view a
plain offset instead of a misaligned read. `mx.save_safetensors` does not
pad, so every tensor lands wherever the header happens to end.

The format forbids gaps between tensors, so alignment comes from ordering:
the JSON header is padded with spaces to a multiple of 64, and tensors whose
byte sizes are multiples of 64 go first. The few that are not (odd-sized
blobs) go last, where they cannot push anything else off the boundary.
"""
import json
import os
import struct

import mlx.core as mx
import numpy as np

ALIGN = 64
DTYPES = {mx.float16: "F16", mx.float32: "F32", mx.uint32: "U32", mx.bfloat16: "BF16", mx.uint8: "U8"}


def save_aligned(path, tensors):
    items = sorted(tensors.items(), key=lambda kv: (kv[1].nbytes % ALIGN != 0, -kv[1].itemsize, kv[0]))
    header, offset = {}, 0
    for name, array in items:
        header[name] = {"dtype": DTYPES[array.dtype], "shape": list(array.shape),
                        "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    text = json.dumps(header, separators=(",", ":"))
    text += " " * ((-(8 + len(text))) % ALIGN)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text.encode())
        for _, array in items:
            f.write(np.array(array).tobytes())
    misaligned = [n for n, h in header.items() if (8 + len(text) + h["data_offsets"][0]) % ALIGN]
    print(f"{os.path.basename(path)}: {len(items)} tensors, {os.path.getsize(path):,} bytes, "
          f"{len(misaligned)} not 64-byte aligned")
