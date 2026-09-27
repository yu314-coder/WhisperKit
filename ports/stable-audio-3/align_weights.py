"""Rewrites the Stable Audio 3 safetensors files with 64-byte-aligned tensors.

Input is release sa3-small-weights-v1 (MLX-format conversions of
https://huggingface.co/stabilityai/stable-audio-3-optimized). Tensors are
unchanged — same names, dtypes and values — only the layout moves, so the
app can map the files instead of reading them into memory.

    python align_weights.py <v1 dir> <out dir>
"""
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from safetensors_aligned import save_aligned  # noqa: E402

FILES = [
    "t5gemma_f16", "same_s_decoder_f32",
    "dit_sm-music_f16", "sa3_conditioner_sm-music",
    "dit_medium_f16.part1", "dit_medium_f16.part2", "sa3_conditioner_medium",
]

if __name__ == "__main__":
    source, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    for name in FILES:
        save_aligned(f"{out}/{name}.safetensors", dict(mx.load(f"{source}/{name}.safetensors")))
