"""FLUX.2 [klein] 4B (Black Forest Labs, Apache 2.0) for MLX on the GPU.

    python export_flux_mlx.py <diffusers repo dir> <out dir>

Writes, all 64-byte aligned for MappedWeights.swift, in parts under 2 GB:

- flux_text_q8.partN.safetensors — Qwen3-4B (Apache 2.0), the prompt
  encoder: only its first 27 layers, since FLUX reads the hidden states after
  layers 9, 18 and 27. Matrices and the embedding table 8-bit in groups of 64
  (scales float16), norms float32; it runs with float32 activations.
- flux_transformer_q8.partN.safetensors — the 3.9B transformer, 8-bit
  (scales bfloat16, the activations' type), norms bfloat16. FLUX overflows
  float16 from its second step, so it runs in bfloat16.
- flux_vae.safetensors — the VAE decoder, its post-quant convolution and the
  latent batch-norm statistics; float16 (statistics float32), convolution
  weights in MLX's (out, kh, kw, in) layout.
- flux_vocab.json, flux_merges.txt — Qwen's tokenizer, as published.

Precision, end to end against the bfloat16 PyTorch pipeline (1024 px,
PSNR of the picture): 8-bit text and transformer 29.9 / 20.2 dB, visually
the same picture; a 4-bit transformer 19.5 / 15.7 dB and a 4-bit text
encoder 21.3 / 16.6 dB, each visibly changing the scene.
"""
import glob, os, shutil, sys
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safetensors_aligned import save_aligned, save_parts  # noqa: E402

repo, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
TEXT_LAYERS = 27


def quantized(tensors, scale_dtype, other_dtype):
    result = {}
    for name, value in tensors.items():
        if value.ndim == 2 and value.shape[1] % 64 == 0:
            q, scales, biases = mx.quantize(value.astype(mx.float32), group_size=64, bits=8)
            result[name + '.wq'] = q
            result[name + '.scales'] = scales.astype(scale_dtype)
            result[name + '.biases'] = biases.astype(scale_dtype)
        else:
            result[name] = value.astype(other_dtype)
        mx.eval([v for k, v in result.items() if k.startswith(name)])
    return result


# Qwen3, first 27 layers ---------------------------------------------------------
raw = {}
for path in sorted(glob.glob(f'{repo}/text_encoder/*.safetensors')):
    raw.update(mx.load(path))
text = {}
for name, value in raw.items():
    name = name.replace('model.', '', 1)
    if name.startswith('layers.') and int(name.split('.')[1]) >= TEXT_LAYERS:
        continue
    if name in ('norm.weight', 'lm_head.weight'):
        continue
    text[name] = value
del raw
print('text parts:', save_parts(f'{out}/flux_text_q8', quantized(text, mx.float16, mx.float32)))
del text

# Transformer ----------------------------------------------------------------------
raw = mx.load(f'{repo}/transformer/diffusion_pytorch_model.safetensors')
print('transformer parts:', save_parts(f'{out}/flux_transformer_q8', quantized(raw, mx.bfloat16, mx.bfloat16)))
del raw

# VAE decoder ------------------------------------------------------------------------
vae = {}
for name, value in mx.load(f'{repo}/vae/diffusion_pytorch_model.safetensors').items():
    if name.startswith(('encoder.', 'quant_conv.')) or name == 'bn.num_batches_tracked':
        continue
    if value.ndim == 4:
        value = value.transpose(0, 2, 3, 1)
    vae[name] = value.astype(mx.float32 if name.startswith('bn.') else mx.float16)
save_aligned(f'{out}/flux_vae.safetensors', vae)

# Tokenizer ----------------------------------------------------------------------------
shutil.copyfile(f'{repo}/tokenizer/vocab.json', f'{out}/flux_vocab.json')
shutil.copyfile(f'{repo}/tokenizer/merges.txt', f'{out}/flux_merges.txt')
print('tokenizer copied')
