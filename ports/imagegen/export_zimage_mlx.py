"""Z-Image-Turbo (Tongyi-MAI, Apache 2.0) for MLX on the GPU.

    python export_zimage_mlx.py <diffusers repo dir> <abliterated encoder .gguf> <out dir>

Writes, 64-byte aligned for MappedWeights.swift, in parts under 2 GB:

- zimage_text_q8.partN.safetensors — the prompt encoder, Qwen3-4B: the
  **abliterated** build (BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1,
  Apache 2.0, from its F16 GGUF), which has its refusal directions removed
  so prompts aren't steered away from (KL 0.0004 against the original).
  Only its first 35 layers: Z-Image reads the second-to-last layer's states.
  8-bit in groups of 64 (scales float16), norms float32; float32 activations.
- zimage_transformer_q8.partN.safetensors — the 6B single-stream
  transformer, 8-bit (scales bfloat16), everything else bfloat16.
- zimage_vae.safetensors — the FLUX.1-style VAE decoder (16 channels),
  float16, convolutions in MLX's (out, kh, kw, in) layout.
- zimage_vocab.json, zimage_merges.txt — Qwen's tokenizer.
"""
import glob, os, shutil, sys
import numpy as np
import mlx.core as mx
from gguf import GGUFReader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safetensors_aligned import save_aligned, save_parts  # noqa: E402

repo, encoder_gguf, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
TEXT_LAYERS = 35


def quantized(tensors, scale_dtype, other_dtype):
    """Consumes `tensors`, so each float32 original is freed once quantized:
    the transformer is 24.6 GB of float32."""
    result = {}
    for name in list(tensors):
        value = tensors.pop(name)
        if value.ndim == 2 and value.shape[1] % 64 == 0:
            q, scales, biases = mx.quantize(value.astype(mx.float32), group_size=64, bits=8)
            result[name + '.wq'] = q
            result[name + '.scales'] = scales.astype(scale_dtype)
            result[name + '.biases'] = biases.astype(scale_dtype)
        else:
            result[name] = value.astype(other_dtype)
        mx.eval([v for k, v in result.items() if k.startswith(name)])
    return result


# The abliterated text encoder, GGUF names to Hugging Face's -----------------------
GGUF_NAMES = {
    'attn_norm': 'input_layernorm', 'ffn_norm': 'post_attention_layernorm',
    'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj', 'attn_v': 'self_attn.v_proj',
    'attn_output': 'self_attn.o_proj', 'attn_q_norm': 'self_attn.q_norm', 'attn_k_norm': 'self_attn.k_norm',
    'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj', 'ffn_down': 'mlp.down_proj',
}
text = {}
for tensor in GGUFReader(encoder_gguf).tensors:
    if tensor.name == 'token_embd.weight':
        name = 'embed_tokens.weight'
    elif tensor.name.startswith('blk.'):
        _, layer, kind, _ = tensor.name.split('.')
        if int(layer) >= TEXT_LAYERS:
            continue
        name = f'layers.{layer}.{GGUF_NAMES[kind]}.weight'
    else:
        continue   # output_norm: past the layer Z-Image reads
    text[name] = mx.array(np.asarray(tensor.data))
print('text parts:', save_parts(f'{out}/zimage_text_q8', quantized(text, mx.float16, mx.float32)))
del text

# Transformer ------------------------------------------------------------------------
raw = {}
for path in sorted(glob.glob(f'{repo}/transformer/*.safetensors')):
    raw.update(mx.load(path))
print('transformer parts:', save_parts(f'{out}/zimage_transformer_q8', quantized(raw, mx.bfloat16, mx.bfloat16)))
del raw

# VAE decoder --------------------------------------------------------------------------
vae = {}
for name, value in mx.load(glob.glob(f'{repo}/vae/*.safetensors')[0]).items():
    if not name.startswith('decoder.'):
        continue
    if value.ndim == 4:
        value = value.transpose(0, 2, 3, 1)
    vae[name] = value.astype(mx.float16)
save_aligned(f'{out}/zimage_vae.safetensors', vae)

shutil.copyfile(f'{repo}/tokenizer/vocab.json', f'{out}/zimage_vocab.json')
shutil.copyfile(f'{repo}/tokenizer/merges.txt', f'{out}/zimage_merges.txt')
print('tokenizer copied')
