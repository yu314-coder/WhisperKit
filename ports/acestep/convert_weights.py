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
    ace_lm_q8         the 1.7B planner (acestep-5Hz-lm-1.7B), int8
    ace_hints_q8      FSQ codebook and detokenizer: planner tokens to the
                      25 Hz guide the diffusion transformer follows

In the _q8 files, 2-D projections are int8 with group size 64 and everything
else is float16. The text encoder stays float16: Qwen3's hidden states carry
large outlier channels that int8 groups flatten, and that one stage accounted
for most of the drift from the reference — final latent cosine 0.960 with it
at int8, 0.983 without, against the float32 pipeline over eight steps.

Every tensor starts on a 64-byte boundary; see ../safetensors_aligned.py.
"""
import os
import sys

import mlx.core as mx
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from safetensors_aligned import save_aligned  # noqa: E402

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


def fsq_codebook(source):
    """Every FSQ code as the 6-dim vector the quantizer sums before its
    output projection — read from the reference implementation rather than
    re-derived, so level offsets and scaling are exactly its own."""
    sys.path.insert(0, f"{source}/acestep-v15-turbo")
    from configuration_acestep_v15 import AceStepConfig
    from vector_quantize_pytorch import ResidualFSQ
    config = AceStepConfig.from_pretrained(f"{source}/acestep-v15-turbo")
    fsq = ResidualFSQ(dim=config.fsq_dim, levels=config.fsq_input_levels,
                      num_quantizers=config.fsq_input_num_quantizers)
    with torch.no_grad():
        codes = fsq.get_codes_from_indices(torch.arange(64000).view(1, 64000, 1))[0, 0]
    return mx.array(codes.numpy())


def main(source, vae_file, out):
    os.makedirs(out, exist_ok=True)
    dit = dict(mx.load(f"{source}/acestep-v15-turbo/model.safetensors"))
    save_aligned(f"{out}/ace_cond_q8.safetensors", quantized(dit, lambda k: k.startswith("encoder.")))
    save_aligned(f"{out}/ace_decoder_q8.safetensors", quantized(dit, lambda k: k.startswith("decoder.")))
    hints = quantized(dit, lambda k: k.startswith("detokenizer.") or k.startswith("tokenizer.quantizer.project_out"))
    hints["fsq_codebook"] = fsq_codebook(source)
    save_aligned(f"{out}/ace_hints_q8.safetensors", hints)
    del dit

    planner = mx.load(f"{source}/acestep-5Hz-lm-1.7B/model.safetensors")
    save_aligned(f"{out}/ace_lm_q8.safetensors", quantized(planner))
    del planner

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
