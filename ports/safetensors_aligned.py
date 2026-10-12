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


def save_aligned(path, tensors, metadata=None):
    """`metadata`, string to string, goes in the header's `__metadata__` —
    where a licence's "this file was changed" notice can travel with it."""
    items = sorted(tensors.items(), key=lambda kv: (kv[1].nbytes % ALIGN != 0, -kv[1].itemsize, kv[0]))
    header, offset = {}, 0
    for name, array in items:
        header[name] = {"dtype": DTYPES[array.dtype], "shape": list(array.shape),
                        "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    text = json.dumps({"__metadata__": metadata, **header} if metadata else header, separators=(",", ":"))
    text += " " * ((-(8 + len(text))) % ALIGN)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text.encode())
        for _, array in items:
            # NumPy has no bfloat16; its bits are written as they are.
            raw = mx.view(array, mx.uint16) if array.dtype == mx.bfloat16 else array
            f.write(np.array(raw).tobytes())
    misaligned = [n for n, h in header.items() if (8 + len(text) + h["data_offsets"][0]) % ALIGN]
    print(f"{os.path.basename(path)}: {len(items)} tensors, {os.path.getsize(path):,} bytes, "
          f"{len(misaligned)} not 64-byte aligned")


# MLX shapes are 32-bit, so MappedWeights.swift can't view a file over 2 GiB
# as one array: bigger models are written in parts.
PART_BYTES = 1_900_000_000


def save_parts(stem, tensors, metadata=None):
    """Writes `<stem>.part1.safetensors`... under PART_BYTES each, split on
    whole source tensors so a quantized triple stays together. Returns the
    number of parts."""
    groups = {}
    for name, array in tensors.items():
        base = name.rsplit(".", 1)[0] if name.endswith((".wq", ".scales", ".biases")) else name
        groups.setdefault(base, {})[name] = array
    parts, current, size = [], {}, 0
    for base in sorted(groups):
        group = groups[base]
        group_bytes = sum(a.nbytes for a in group.values())
        if current and size + group_bytes > PART_BYTES:
            parts.append(current)
            current, size = {}, 0
        current.update(group)
        size += group_bytes
    parts.append(current)
    for index, part in enumerate(parts, start=1):
        save_aligned(f"{stem}.part{index}.safetensors", part, metadata)
    return len(parts)
