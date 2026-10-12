"""Qwen-Image 2.1 Turbo (Qwen, Qwen RESEARCH LICENSE) for MLX on the GPU.

    python export_qwenimage_mlx.py <Qwen-Image-2.1-Turbo repo dir> <abliterated encoder repo dir> <out dir>

Writes, 64-byte aligned for MappedWeights.swift, in parts under 2 GB:

- qwenimage_text_q8.partN.safetensors — the prompt encoder, Qwen3-VL-8B's
  language model: the **abliterated** build
  (pottokao/Qwen-Image-2.1-Text-Encoder-Heretic, Apache 2.0), a drop-in
  for the stock encoder with its refusal directions removed (KL 0.022).
  Text path only — no vision tower, no final norm, no LM head: Qwen-Image
  reads the last layer's states before the norm. 8-bit in groups of 64
  (scales float16), norms float32; float32 activations.
- qwenimage_transformer_q8.partN.safetensors — the 7B single-stream
  transformer, 8-bit (scales bfloat16), everything else bfloat16.
- qwenimage_vae.safetensors — the VAE decoder, bfloat16, convolutions in
  MLX's (out, kh, kw, in) layout; the one-frame decode never touches the
  temporal upsamplers, so they're left out.
- qwenimage_vocab.json, qwenimage_merges.txt — Qwen's tokenizer.
- qwenimage_LICENSE.txt, qwenimage_NOTICE.txt — the Qwen RESEARCH LICENSE
  AGREEMENT, which asks for both to go with every copy, and for changed
  files to say so: each weight file also carries that notice in its header.
"""
import glob, json, os, shutil, sys
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safetensors_aligned import save_aligned, save_parts  # noqa: E402

repo, encoder, out = sys.argv[1], sys.argv[2], sys.argv[3]
# Optional fourth argument: which pieces, e.g. "vae,tokenizer".
pieces = sys.argv[4].split(',') if len(sys.argv) > 4 else ['text', 'transformer', 'vae', 'tokenizer', 'licence']
os.makedirs(out, exist_ok=True)

ATTRIBUTION = ('Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 '
               'Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.')
CHANGED = {
    'text': 'Changed by Yu Yao-Hsing for the WhisperKit app: Qwen3-VL-8B-Instruct\'s language model '
            '(abliterated build pottokao/Qwen-Image-2.1-Text-Encoder-Heretic, Apache 2.0) without its '
            'vision tower, final norm and LM head, quantized to 8 bits for MLX.',
    'transformer': 'Changed by Yu Yao-Hsing for the WhisperKit app: Qwen-Image 2.1 Turbo\'s transformer, '
                   'quantized to 8 bits for MLX. ' + ATTRIBUTION,
    'vae': 'Changed by Yu Yao-Hsing for the WhisperKit app: Qwen-Image 2.1\'s VAE decoder only, '
           'in bfloat16, convolutions transposed for MLX. ' + ATTRIBUTION,
}


def quantized(tensors, scale_dtype, other_dtype, keep=()):
    """Consumes `tensors`, so each original is freed once quantized."""
    result = {}
    for name in list(tensors):
        value = tensors.pop(name)
        if value.ndim == 2 and value.shape[1] % 64 == 0 and name not in keep:
            q, scales, biases = mx.quantize(value.astype(mx.float32), group_size=64, bits=8)
            result[name + '.wq'] = q
            result[name + '.scales'] = scales.astype(scale_dtype)
            result[name + '.biases'] = biases.astype(scale_dtype)
        else:
            result[name] = value.astype(other_dtype)
        mx.eval([v for k, v in result.items() if k.startswith(name)])
    return result


# The abliterated text encoder: the language model's layers and embeddings -------------
PREFIX = 'model.language_model.'
if 'text' in pieces:
    text = {}
    for path in sorted(glob.glob(f'{encoder}/model-0*.safetensors')):
        for name, value in mx.load(path).items():
            if name.startswith(PREFIX) and not name.startswith(PREFIX + 'norm.'):
                text[name[len(PREFIX):]] = value
    print('text tensors:', len(text))
    print('text parts:', save_parts(f'{out}/qwenimage_text_q8', quantized(text, mx.float16, mx.float32),
                                      {'notice': CHANGED['text']}))
    del text

# Transformer --------------------------------------------------------------------------
if 'transformer' in pieces:
    raw = {}
    for path in sorted(glob.glob(f'{repo}/transformer/*.safetensors')):
        raw.update(mx.load(path))
    print('transformer parts:', save_parts(f'{out}/qwenimage_transformer_q8',
                                           quantized(raw, mx.bfloat16, mx.bfloat16, keep=('img_in.weight', 'proj_out.weight')),
                                           {'notice': CHANGED['transformer']}))
    del raw

# VAE decoder --------------------------------------------------------------------------
if 'vae' in pieces:
    vae = {}
    for name, value in mx.load(glob.glob(f'{repo}/vae/*.safetensors')[0]).items():
        if not name.startswith(('decoder.', 'post_quant_conv.')) or 'time_conv' in name:
            continue
        if name.endswith('.gamma'):
            value = value.reshape(-1)
        elif value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        vae[name] = value.astype(mx.bfloat16)
    save_aligned(f'{out}/qwenimage_vae.safetensors', vae, {'notice': CHANGED['vae']})

# Tokenizer: the byte-level BPE's vocabulary and merges, as Qwen's other repos ship them.
if 'tokenizer' in pieces:
    model = json.load(open(f'{repo}/processor/tokenizer.json'))['model']
    with open(f'{out}/qwenimage_vocab.json', 'w') as f:
        json.dump(model['vocab'], f, ensure_ascii=False)
    with open(f'{out}/qwenimage_merges.txt', 'w') as f:
        f.write('#version: 0.2\n')
        for merge in model['merges']:
            f.write((merge if isinstance(merge, str) else ' '.join(merge)) + '\n')
    print('tokenizer written')

if 'licence' in pieces:
    shutil.copyfile(f'{repo}/LICENSE', f'{out}/qwenimage_LICENSE.txt')
    with open(f'{out}/qwenimage_NOTICE.txt', 'w') as f:
        f.write(ATTRIBUTION + '\n\n'
                'These files are derived from Qwen-Image 2.1 Turbo (https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) '
                'and changed by Yu Yao-Hsing for the WhisperKit app:\n'
                '- ' + CHANGED['transformer'].split(' ' + ATTRIBUTION)[0] + '\n'
                '- ' + CHANGED['vae'].split(' ' + ATTRIBUTION)[0] + '\n\n'
                'The prompt reader is ' + CHANGED['text'].split(': ', 1)[1] + '\n\n'
                'Its own notice follows.\n\n' + open(f'{encoder}/NOTICE').read())
    print('licence and notice written')
