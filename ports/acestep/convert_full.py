"""Full-precision versions of the stages that were int8.

    python convert_full.py <acestep-v15-turbo dir> <acestep-5Hz-lm-1.7B dir> \
                           <acestep-v15-xl-turbo dir> <out dir>

Output:

    ace_lm_bf16.part{1,2}   the 1.7B planner as published, bfloat16
    ace_cond_f16            condition encoder, float16
    ace_hints_f16           FSQ codebook (float32) and detokenizer, float16
    ace_xl_cond_f16         the encoder tensors XL retrained, float16

The planner runs in bfloat16, as upstream does: Qwen3's activations
overflow float16. The two transformers are float16 already where they fit —
the 2B on the Neural Engine — and XL's stays int8 (ace_xl_decoder_q8): at
float16 it is 8.1 GB, more than an 8 GB device holds.
"""
import glob
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from convert_large import save_parts  # noqa: E402
from convert_weights import fsq_codebook  # noqa: E402
from safetensors_aligned import save_aligned  # noqa: E402

RETRAINED = [
    "encoder.lyric_encoder.embed_tokens.bias", "encoder.lyric_encoder.embed_tokens.weight",
    "encoder.lyric_encoder.norm.weight", "encoder.text_projector.weight",
    "encoder.timbre_encoder.embed_tokens.bias", "encoder.timbre_encoder.embed_tokens.weight",
    "encoder.timbre_encoder.norm.weight", "encoder.timbre_encoder.special_token",
]


def main(turbo, lm, xl, out):
    os.makedirs(out, exist_ok=True)
    print("planner parts:", save_parts(f"{out}/ace_lm_bf16", dict(mx.load(f"{lm}/model.safetensors"))))

    dit = mx.load(f"{turbo}/model.safetensors")
    save_aligned(f"{out}/ace_cond_f16.safetensors",
                 {k: v.astype(mx.float16) for k, v in dit.items() if k.startswith("encoder.")})
    hints = {k: v.astype(mx.float16) for k, v in dit.items()
             if k.startswith("detokenizer.") or k.startswith("tokenizer.quantizer.project_out")}
    hints["fsq_codebook"] = fsq_codebook(os.path.dirname(turbo.rstrip("/")))
    save_aligned(f"{out}/ace_hints_f16.safetensors", hints)
    del dit

    retrained = {}
    for path in sorted(glob.glob(f"{xl}/*.safetensors")):
        retrained.update({k: v.astype(mx.float16) for k, v in mx.load(path).items() if k in RETRAINED})
    assert sorted(retrained) == sorted(RETRAINED)
    save_aligned(f"{out}/ace_xl_cond_f16.safetensors", retrained)


if __name__ == "__main__":
    main(*sys.argv[1:5])
