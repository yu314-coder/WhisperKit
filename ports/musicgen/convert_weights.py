"""Convert the HuggingFace MusicGen checkpoint into MLX-ready safetensors.

Three transformations are needed:
  * weight_norm is folded (w = g * v / ||v||), because MLX has no such wrapper;
  * PyTorch conv1d weights are (out, in, k) and MLX wants (out, k, in);
  * transposed convs are (in, out, k) in PyTorch, so they permute differently.
LSTM weights are left exactly as they are and unpacked on the Swift side.
"""
import re, sys
import torch, mlx.core as mx
from safetensors.torch import load_file

SRC = sys.argv[1]; DST = sys.argv[2]
sd = load_file(SRC)

# Decoder upsampling stages use ConvTranspose1d; everything else is conv1d.
TRANSPOSED = {f"audio_encoder.decoder.layers.{i}.conv" for i in (3, 6, 9, 12)}

def fold_weight_norm(sd):
    out, drop = {}, set()
    for k in list(sd):
        if k.endswith(".weight_g"):
            base = k[:-len(".weight_g")]
            v = sd.get(base + ".weight_v")
            if v is None: continue
            g = sd[k]
            dims = list(range(1, v.dim()))
            norm = torch.norm(v.float(), p=2, dim=dims, keepdim=True)
            out[base + ".weight"] = (g.float() * v.float() / (norm + 1e-12))
            drop.add(k); drop.add(base + ".weight_v")
    for k, v in sd.items():
        if k not in drop and not k.endswith(".weight_v"):
            out.setdefault(k, v)
    return out

sd = fold_weight_norm(sd)

converted, conv_count, tr_count = {}, 0, 0
for k, v in sd.items():
    t = v.float()
    if k.endswith(".weight") and t.dim() == 3:
        base = k[:-len(".weight")]
        if base in TRANSPOSED:
            t = t.permute(1, 2, 0).contiguous()   # (in,out,k) -> (out,k,in)
            tr_count += 1
        else:
            t = t.permute(0, 2, 1).contiguous()   # (out,in,k) -> (out,k,in)
            conv_count += 1
    # Half precision for the transformers; the codec stays fp32 because its
    # output is audio samples rather than logits.
    half = k.startswith(("text_encoder", "decoder", "enc_to_dec_proj"))
    arr = mx.array(t.numpy())
    converted[k] = arr.astype(mx.float16) if half else arr.astype(mx.float32)

mx.save_safetensors(DST, converted)
print(f"{len(converted)} tensors  ({conv_count} conv, {tr_count} transposed-conv, weight_norm folded)")
