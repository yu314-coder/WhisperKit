"""Builds the files the larger ACE-Step 1.5 versions add to the base set.

Sources (MIT): https://huggingface.co/ACE-Step/acestep-v15-xl-turbo and
https://huggingface.co/ACE-Step/acestep-5Hz-lm-4B.

    python convert_large.py <xl-turbo dir> <lm-4B dir> <out dir>

Output:

    ace_xl_decoder_q8.partN   the 4B diffusion transformer, int8, in parts
    ace_xl_cond_q8            the condition-encoder tensors XL retrained
    ace_lm4b_q8.partN         the 4B planner, int8, in parts

Parts, because a release asset is capped at 2 GB. The app merges them.

XL keeps the 2B's condition encoder layers as they are: of 201 non-decoder
tensors, 192 are bit-identical to acestep-v15-turbo's. Only the two input
embeddings, their norms, the timbre token and the text projection were
retrained, so those are all ace_xl_cond_q8 holds; the app lays them over
ace_cond_q8. The detokenizer, FSQ quantizer and silence latent are identical
and shared as they are.
"""
import glob
import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from convert_weights import quantized  # noqa: E402
from safetensors_aligned import save_aligned  # noqa: E402

PART_BYTES = 1_900_000_000


def quantize_files(directory, keep=lambda k: True):
    """Quantizes one source file at a time, so a 20 GB float32 checkpoint
    never has to be in memory at once."""
    out = {}
    for path in sorted(glob.glob(f"{directory}/*.safetensors")):
        part = quantized(mx.load(path), keep)
        mx.eval(part)
        out.update(part)
    return out


def save_parts(stem, tensors):
    """Splits on whole source tensors, so a packed triple stays together."""
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
        save_aligned(f"{stem}.part{index}.safetensors", part)
    return len(parts)


def main(xl, lm4b, out, base_turbo=None):
    os.makedirs(out, exist_ok=True)
    decoder = quantize_files(xl, lambda k: k.startswith("decoder."))
    print("decoder parts:", save_parts(f"{out}/ace_xl_decoder_q8", decoder))
    del decoder
    dit = {}
    for path in sorted(glob.glob(f"{xl}/*.safetensors")):
        dit.update({k: v for k, v in mx.load(path).items() if k.startswith("encoder.")})

    retrained = [
        "encoder.lyric_encoder.embed_tokens.bias", "encoder.lyric_encoder.embed_tokens.weight",
        "encoder.lyric_encoder.norm.weight", "encoder.text_projector.weight",
        "encoder.timbre_encoder.embed_tokens.bias", "encoder.timbre_encoder.embed_tokens.weight",
        "encoder.timbre_encoder.norm.weight", "encoder.timbre_encoder.special_token",
    ]
    if base_turbo:
        # Confirms the list: everything else under encoder. must match.
        base = mx.load(f"{base_turbo}/model.safetensors")
        changed = sorted(k for k in dit if k.startswith("encoder.")
                         and not mx.array_equal(dit[k].astype(mx.float32), base[k].astype(mx.float32)))
        assert changed == sorted(retrained), changed
    save_aligned(f"{out}/ace_xl_cond_q8.safetensors", quantized({k: dit[k] for k in retrained}))
    del dit

    # The 4B checkpoint names its tensors "model.…"; the 1.7B's, which the
    # app reads, have no prefix.
    planner = {k.removeprefix("model."): v for k, v in quantize_files(lm4b).items()}
    print("planner parts:", save_parts(f"{out}/ace_lm4b_q8", planner))


if __name__ == "__main__":
    main(*sys.argv[1:5])
