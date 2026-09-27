"""Builds the ACE-Step 1.5 weight files WhisperKit downloads.

Source: https://huggingface.co/ACE-Step/Ace-Step1.5 (MIT) —
acestep-v15-turbo/, Qwen3-Embedding-0.6B/, and the VAE as converted for MLX
(ace_vae_f16.safetensors from release acestep-int8-v1, weight norm folded).

    python convert_weights.py <Ace-Step1.5 dir> <v1 vae file> <out dir>

Output, one file per stage so the app holds only what the running stage needs:

    ace_qwen_f16      text encoder (Qwen3-Embedding-0.6B), float16
    ace_cond_q8       condition encoder: lyric and timbre stacks, projection
    ace_decoder_q8    diffusion transformer only
    ace_vae_f16       VAE decoder only (the encoder half is not used)
    ace_silence_full  the ten-minute silence latent, frames-first

In the _q8 files, 2-D projections are int8 with group size 64 and everything
else is float16. The text encoder stays float16: Qwen3's hidden states carry
large outlier channels that int8 groups flatten, and that one stage accounted
for most of the drift from the reference — final latent cosine 0.960 with it
at int8, 0.983 without, against the float32 pipeline over eight steps.

Every tensor starts on a 64-byte boundary. The app maps these files rather
than reading them, and views each tensor in place, so alignment is what
makes a view a plain offset instead of a misaligned read. The safetensors
format forbids gaps between tensors, so alignment comes from ordering: the
header is padded with spaces to a multiple of 64, and tensors whose sizes
are multiples of 64 go first.
"""
import json
import os
import struct
import sys

import mlx.core as mx
import numpy as np
import torch

ALIGN = 64
DTYPES = {mx.float16: "F16", mx.float32: "F32", mx.uint32: "U32", mx.bfloat16: "BF16"}


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


def quantized(weights, keep=lambda k: True):
    out = {}
    for k, v in weights.items():
        if not keep(k):
            continue
        if v.ndim == 2 and min(v.shape) >= 64:
            wq, scales, biases = mx.quantize(v.astype(mx.float32), group_size=64, bits=8)
            out[k + ".wq"], out[k + ".scales"], out[k + ".biases"] = wq, scales, biases
        else:
            out[k] = v.astype(mx.float16)
    return out


def main(source, vae_file, out):
    os.makedirs(out, exist_ok=True)
    dit = dict(mx.load(f"{source}/acestep-v15-turbo/model.safetensors"))
    save_aligned(f"{out}/ace_cond_q8.safetensors", quantized(dit, lambda k: k.startswith("encoder.")))
    save_aligned(f"{out}/ace_decoder_q8.safetensors", quantized(dit, lambda k: k.startswith("decoder.")))
    del dit

    qwen = mx.load(f"{source}/Qwen3-Embedding-0.6B/model.safetensors")
    save_aligned(f"{out}/ace_qwen_f16.safetensors", {k: v.astype(mx.float16) for k, v in qwen.items()})
    del qwen

    # Decoder only. Upsampling weights are stored as (in, taps, out) under
    # ".taps" — the layout the app multiplies by — instead of MLX's
    # (out, taps, in) transposed-convolution layout, so it never has to
    # transpose a copy per decoding window.
    vae = {}
    for k, v in mx.load(vae_file).items():
        if not k.startswith("decoder."):
            continue
        if k.endswith("conv_t1.weight"):
            vae[k.replace(".weight", ".taps")] = mx.contiguous(v.transpose(2, 1, 0)).astype(mx.float16)
        else:
            vae[k] = v.astype(mx.float16)
    save_aligned(f"{out}/ace_vae_f16.safetensors", vae)

    silence = torch.load(f"{source}/acestep-v15-turbo/silence_latent.pt", weights_only=True)
    silence = silence.transpose(1, 2).float().numpy()  # (1, 15000, 64)
    save_aligned(f"{out}/ace_silence_full.safetensors", {"silence": mx.array(silence).astype(mx.float16)})


if __name__ == "__main__":
    main(*sys.argv[1:4])
