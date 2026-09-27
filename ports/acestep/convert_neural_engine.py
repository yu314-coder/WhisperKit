"""Builds ACE-Step's diffusion transformer for the Neural Engine.

    python convert_neural_engine.py <acestep-v15-turbo dir> <out dir> [chunks]

The Neural Engine runs Core ML only, in float16, on fixed shapes. So:

  * The 24 layers are cut into four programs of six. One program holding
    all 24 would be a single 4 GB weight file, over the 2 GB asset cap, and
    the Neural Engine compiles smaller programs far faster.
  * Each program holds one function per length bucket and per conditioning
    bucket, sharing one copy of the weights (a multifunction model, iOS 18).
    A clip is padded up to its bucket; padded positions are masked out of
    every attention, so the positions that are kept see exactly what they
    would unpadded.
  * RMS norms are computed on a pre-scaled input. Squaring an activation
    above ~256 overflows float16; mean((x/64)^2) cannot, and
    rsqrt(that)/64 is the same 1/rms.

Everything outside the layers — the timestep embeddings, the patch
projections in and out, the conditioning projection and the final norm —
is small and stays on the GPU in MLX, in ace_dit_outer_f16.

Output: ace_dit_ane_c{0..3}.mlmodelc, compiled, and ace_dit_outer_f16.
"""
import filecmp
import gc
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HEADS, KV_HEADS, HEAD_DIM, WIDTH = 16, 8, 128, 2048
LAYERS, PER_CHUNK = 24, 6
WINDOW = 128

BLOCK = 128  # the sliding window's radius, and its attention block

# Patched positions (two latent frames each, 12.5 a second), whole blocks.
# The app renders a clip 20% longer than asked, at least 8 s more, and cuts;
# 5,760 positions is 461 s, covering the 6:24 maximum. A clip is padded up
# to the next bucket, so padding costs at most the spacing between them.
LENGTH_BUCKETS = [384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3840, 4608, 5760]
# Conditioning tokens: caption (up to 256) + lyrics (up to 2048) + timbre
# (1), rounded up to 64. An instrumental or a short verse fits the first.
CONDITIONING_BUCKETS = [512, 2368]


def function_name(length, tokens):
    return f"p{length}_e{tokens}"


def rms(x, w, eps=1e-6):
    s = (x * (1.0 / 64)).pow(2).mean(-1, keepdim=True)
    return x * (torch.rsqrt(s + eps / 4096) * (1.0 / 64)) * w


def rotate(x, c, s):
    return x * c + torch.cat([-x[..., HEAD_DIM // 2:], x[..., :HEAD_DIM // 2]], -1) * s


class Chunk(nn.Module):
    """Layers [first, first + count) of the decoder.

    Inputs: hidden (1,P,2048); encoder (1,E,2048), already through
    condition_embedder; encoder_mask (1,1,1,E) additive; temb (1,6,2048),
    the timestep modulation; cos, sin (1,1,P,128); local_mask
    (1,P/128,128,384) additive, the sliding window and padding by block (see
    `sliding`); full_mask (1,1,1,P) additive, padding only.
    """

    def __init__(self, weights, first, count, length, tokens):
        super().__init__()
        self.first, self.count, self.P, self.E = first, count, length, tokens
        for k, v in weights.items():
            self.register_buffer(k.replace(".", "_"), v)

    def g(self, i, k):
        return getattr(self, f"layers_{i}_{k}".replace(".", "_"))

    def attention(self, i, pre, x, memory, length, cos, sin, mask, sliding=False):
        q = (x @ self.g(i, f"{pre}.q_proj.weight").T).view(1, -1, HEADS, HEAD_DIM)
        k = (memory @ self.g(i, f"{pre}.k_proj.weight").T).view(1, -1, KV_HEADS, HEAD_DIM)
        v = (memory @ self.g(i, f"{pre}.v_proj.weight").T).view(1, -1, KV_HEADS, HEAD_DIM)
        q = rms(q, self.g(i, f"{pre}.q_norm.weight")).transpose(1, 2)
        k = rms(k, self.g(i, f"{pre}.k_norm.weight")).transpose(1, 2)
        v = v.transpose(1, 2)
        if cos is not None:
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        group = HEADS // KV_HEADS
        k = k[:, :, None].expand(-1, -1, group, -1, -1).reshape(1, HEADS, -1, HEAD_DIM)
        v = v[:, :, None].expand(-1, -1, group, -1, -1).reshape(1, HEADS, -1, HEAD_DIM)
        if sliding:
            out = self.sliding(q, k, v, mask)
        else:
            scores = (q @ k.transpose(-1, -2)) * HEAD_DIM ** -0.5 + mask
            out = scores.softmax(-1) @ v
        out = out.transpose(1, 2).reshape(1, -1, HEADS * HEAD_DIM)
        return out @ self.g(i, f"{pre}.o_proj.weight").T

    def sliding(self, q, k, v, mask):
        """Attention within +-128 positions, a block of 128 queries at a time
        against its own block and both neighbours — 384 keys, not all of
        them. At the 6:24 maximum that is 15 times less work for half the
        layers. `mask` is (1, blocks, 128, 384): the window and padding."""
        qb = q.reshape(HEADS, -1, BLOCK, HEAD_DIM)
        zero = torch.zeros(HEADS, 1, BLOCK, HEAD_DIM)
        kb = torch.cat([zero, k.reshape(HEADS, -1, BLOCK, HEAD_DIM), zero], 1)
        vb = torch.cat([zero, v.reshape(HEADS, -1, BLOCK, HEAD_DIM), zero], 1)
        kn = torch.cat([kb[:, :-2], kb[:, 1:-1], kb[:, 2:]], 2)
        vn = torch.cat([vb[:, :-2], vb[:, 1:-1], vb[:, 2:]], 2)
        scores = (qb @ kn.transpose(-1, -2)) * HEAD_DIM ** -0.5 + mask[0]
        out = scores.softmax(-1) @ vn
        return out.reshape(1, HEADS, -1, HEAD_DIM)

    def forward(self, h, encoder, encoder_mask, temb, cos, sin, local_mask, full_mask):
        for i in range(self.first, self.first + self.count):
            mod = self.g(i, "scale_shift_table") + temb
            shift, scale, gate, mlp_shift, mlp_scale, mlp_gate = [mod[:, j:j + 1] for j in range(6)]
            n = rms(h, self.g(i, "self_attn_norm.weight")) * (1 + scale) + shift
            sliding = i % 2 == 0
            h = h + self.attention(i, "self_attn", n, n, self.P, cos, sin,
                                   local_mask if sliding else full_mask, sliding) * gate
            n = rms(h, self.g(i, "cross_attn_norm.weight"))
            h = h + self.attention(i, "cross_attn", n, encoder, self.E, None, None, encoder_mask)
            n = rms(h, self.g(i, "mlp_norm.weight")) * (1 + mlp_scale) + mlp_shift
            gated = n @ self.g(i, "mlp.gate_proj.weight").T
            up = n @ self.g(i, "mlp.up_proj.weight").T
            h = h + ((gated * torch.sigmoid(gated) * up) @ self.g(i, "mlp.down_proj.weight").T) * mlp_gate
        return h


NAMES = ["hidden", "encoder", "encoder_mask", "temb", "cos", "sin", "local_mask", "full_mask"]


def load_layers(checkpoint, first, count):
    weights = {}
    with safe_open(checkpoint, "pt") as f:
        for k in f.keys():
            parts = k.split(".")
            if k.startswith("decoder.layers.") and first <= int(parts[2]) < first + count:
                weights["layers." + ".".join(parts[2:])] = f.get_tensor(k).float()
    return weights


def example_inputs(length, tokens, valid=None, valid_tokens=None, seed=0):
    """Random activations with realistic masks, `valid` of `length` real."""
    valid = valid or length
    valid_tokens = valid_tokens or tokens * 3 // 4
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(1, length, WIDTH, generator=g)
    encoder = torch.randn(1, tokens, WIDTH, generator=g)
    encoder_mask = torch.zeros(1, 1, 1, tokens)
    encoder_mask[..., valid_tokens:] = -1e4
    temb = torch.randn(1, 6, WIDTH, generator=g) * 0.1
    pos = torch.arange(length).float()
    inverse = 1.0 / (1e6 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    angle = pos[:, None] * inverse[None]
    cos = torch.cat([angle.cos(), angle.cos()], -1)[None, None]
    sin = torch.cat([angle.sin(), angle.sin()], -1)[None, None]
    i = torch.arange(length)
    full = torch.where(i < valid, 0.0, -1e4)[None, None, None]
    return [h, encoder, encoder_mask, temb, cos, sin, sliding_mask(length, valid), full]


def sliding_mask(length, valid):
    """The block layout's mask: query q of block b sees key slot s, which is
    position (b - 1) * 128 + s, when that is a real position within 128."""
    blocks = length // BLOCK
    b = torch.arange(blocks)[:, None, None]
    q = torch.arange(BLOCK)[None, :, None]
    s = torch.arange(3 * BLOCK)[None, None, :]
    query = b * BLOCK + q
    key = (b - 1) * BLOCK + s
    keep = ((query - key).abs() <= WINDOW) & (key >= 0) & (key < valid)
    return torch.where(keep, 0.0, -1e4)[None]


def dense_sliding_mask(length, valid):
    i = torch.arange(length)
    keep = ((i[:, None] - i[None]).abs() <= WINDOW) & (i[None] < valid)
    return torch.where(keep, 0.0, -1e4)[None, None]


def trace(weights, first):
    """Traced once, at the smallest shapes: nothing in the graph depends on
    the length, so each bucket is converted from the input shapes alone."""
    model = Chunk(weights, first, PER_CHUNK, LENGTH_BUCKETS[0], CONDITIONING_BUCKETS[0]).eval()
    with torch.no_grad():
        return torch.jit.trace(model, tuple(example_inputs(LENGTH_BUCKETS[0], CONDITIONING_BUCKETS[0])))


def input_shapes(length, tokens):
    return {"hidden": (1, length, WIDTH), "encoder": (1, tokens, WIDTH), "encoder_mask": (1, 1, 1, tokens),
            "temb": (1, 6, WIDTH), "cos": (1, 1, length, HEAD_DIM), "sin": (1, 1, length, HEAD_DIM),
            "local_mask": (1, length // BLOCK, BLOCK, 3 * BLOCK), "full_mask": (1, 1, 1, length)}


def convert_function(traced, length, tokens, path):
    import coremltools as ct
    shapes = input_shapes(length, tokens)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=shapes[n], dtype=np.float16) for n in NAMES],
        outputs=[ct.TensorType(name="out", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        skip_model_load=True)
    ml.save(path)


def build_chunk(checkpoint, chunk, out, lengths=LENGTH_BUCKETS, tokens=CONDITIONING_BUCKETS):
    import coremltools as ct
    first = chunk * PER_CHUNK
    traced = trace(load_layers(checkpoint, first, PER_CHUNK), first)
    gc.collect()
    # A fixed folder with a marker per finished function, so an interrupted
    # build picks up where it stopped.
    work = f"{out}/work_c{chunk}"
    os.makedirs(work, exist_ok=True)
    for length in lengths:
        for count in tokens:
            name = function_name(length, count)
            path = f"{work}/{name}.mlpackage"
            if not os.path.exists(path + ".done"):
                shutil.rmtree(path, ignore_errors=True)
                convert_function(traced, length, count, path)
                open(path + ".done", "w").close()
                gc.collect()
            print(f"chunk {chunk}: {name}", flush=True)
    package = f"{out}/ace_dit_ane_c{chunk}.mlpackage"
    merge(work, [function_name(l, t) for l in lengths for t in tokens], package)
    compiled = f"{out}/ace_dit_ane_c{chunk}.mlmodelc"
    shutil.rmtree(compiled, ignore_errors=True)
    subprocess.run(["xcrun", "coremlcompiler", "compile", package, out], check=True,
                   stdout=subprocess.DEVNULL)
    print(f"chunk {chunk}: compiled", flush=True)


def merge(work, names, package):
    """One multifunction model from single-function ones, sharing one weight
    file.

    `ct.utils.save_multifunction` loads every function's weights to find
    the shared ones — 22 GB for 30 functions of a 755 MB chunk, which sent
    a 24 GB Mac 48 GB into swap. Every function here was converted from the
    same trace, so their weight files are byte-identical and every constant
    sits at the same offset; the functions' program descriptions can simply
    be combined around one copy. Checked against save_multifunction's own
    output for two functions: identical specification.
    """
    import coremltools as ct
    first = ct.utils.load_spec(f"{work}/{names[0]}.mlpackage")
    weights = f"{work}/{names[0]}.mlpackage/Data/com.apple.CoreML/weights/weight.bin"
    merged = ct.proto.Model_pb2.Model()
    merged.CopyFrom(first)
    del merged.description.input[:]
    del merged.description.output[:]
    del merged.description.functions[:]
    merged.mlProgram.functions.clear()
    for name in names:
        path = f"{work}/{name}.mlpackage"
        assert filecmp.cmp(f"{path}/Data/com.apple.CoreML/weights/weight.bin", weights, shallow=False), name
        spec = ct.utils.load_spec(path)
        function = merged.description.functions.add()
        function.name = name
        function.input.extend(spec.description.input)
        function.output.extend(spec.description.output)
        merged.mlProgram.functions[name].CopyFrom(spec.mlProgram.functions["main"])
    merged.description.defaultFunctionName = names[0]
    shutil.rmtree(package, ignore_errors=True)
    ct.models.MLModel(merged, weights_dir=os.path.dirname(weights), skip_model_load=True).save(package)


def save_outer(checkpoint, out):
    """Needs MLX, unlike the rest; run with --outer."""
    import mlx.core as mx
    from safetensors_aligned import save_aligned
    outer = {}
    with safe_open(checkpoint, "pt") as f:
        for k in f.keys():
            if k.startswith("decoder.") and not k.startswith("decoder.layers."):
                outer[k] = mx.array(f.get_tensor(k).float().numpy()).astype(mx.float16)
    save_aligned(f"{out}/ace_dit_outer_f16.safetensors", outer)


if __name__ == "__main__":
    source, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    checkpoint = f"{source}/model.safetensors"
    if len(sys.argv) > 3 and sys.argv[3] == "--outer":
        save_outer(checkpoint, out)
    else:
        chunks = [int(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 else range(LAYERS // PER_CHUNK)
        for chunk in chunks:
            build_chunk(checkpoint, chunk, out)
