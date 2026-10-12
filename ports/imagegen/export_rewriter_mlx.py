"""The prompt rewriter for the Image tab: Qwen-Image 2.1's PE-T2I (a fine-tuned
Qwen3.5 9B, Qwen RESEARCH LICENSE), the abliterated build
pottokao/Qwen-Image-2.1-PE-T2I-Heretic, for MLX on the GPU.

First quantize it with mlx-lm, whose loader puts the checkpoint in MLX's
layout (the depthwise convolution's axes; RMS norm weights stored as
weight − 1 made whole), then repack:

    python -m mlx_lm convert --hf-path <Heretic dir> --mlx-path <mlx dir> -q --q-bits 4 --q-group-size 64
    python export_rewriter_mlx.py <mlx dir> <Heretic dir> <out dir>

Writes, 64-byte aligned for MappedWeights.swift:

- rewriter_q4.partN.safetensors — the language model (no vision tower):
  matrices 4-bit in groups of 64 as `<name>.weight.wq/.scales/.biases`
  (scales bfloat16), the rest bfloat16 but the gated delta rule's A_log,
  float32. Each file's header says how it was changed, as the licence asks.
- rewriter_vocab.json, rewriter_merges.txt, rewriter_specials.json — the
  tokenizer: Qwen3.5's own byte-level BPE (248k tokens) and its special
  tokens' ids.
- rewriter_system_prompt.txt — the instructions the fine-tune was trained
  under; the model is useless without them.
- rewriter_LICENSE.txt, rewriter_NOTICE.txt — the licence and its
  attribution notice.
"""
import glob, json, os, shutil, sys
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safetensors_aligned import save_parts  # noqa: E402

source, heretic, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

ATTRIBUTION = ('Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 '
               'Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.')
CHANGED = ('Changed by Yu Yao-Hsing for the WhisperKit app: Qwen-Image-2.1-PE-T2I (abliterated build '
           'pottokao/Qwen-Image-2.1-PE-T2I-Heretic), its language model only, quantized to 4 bits for MLX.')

raw = {}
for path in sorted(glob.glob(f'{source}/model*.safetensors')):
    raw.update(mx.load(path))
tensors = {}
for name, value in raw.items():
    if 'vision' in name:
        continue
    name = name.replace('language_model.model.', '').replace('language_model.', '')
    base, _, kind = name.rpartition('.')
    if kind in ('scales', 'biases'):
        tensors[f'{base}.weight.{kind}'] = value
    elif kind == 'weight' and value.dtype == mx.uint32:
        tensors[f'{base}.weight.wq'] = value
    elif kind == 'A_log':
        tensors[name] = value.astype(mx.float32)
    else:
        tensors[name] = value
print('tensors:', len(tensors))
print('parts:', save_parts(f'{out}/rewriter_q4', tensors, {'notice': CHANGED + ' ' + ATTRIBUTION}))

tokenizer = json.load(open(f'{heretic}/tokenizer.json'))
with open(f'{out}/rewriter_vocab.json', 'w') as f:
    json.dump(tokenizer['model']['vocab'], f, ensure_ascii=False)
with open(f'{out}/rewriter_merges.txt', 'w') as f:
    f.write('#version: 0.2\n')
    for merge in tokenizer['model']['merges']:
        f.write((merge if isinstance(merge, str) else ' '.join(merge)) + '\n')
with open(f'{out}/rewriter_specials.json', 'w') as f:
    json.dump({t['content']: t['id'] for t in tokenizer['added_tokens']}, f, ensure_ascii=False)
shutil.copyfile(f'{heretic}/system_prompt.txt', f'{out}/rewriter_system_prompt.txt')
shutil.copyfile(f'{heretic}/LICENSE', f'{out}/rewriter_LICENSE.txt')
with open(f'{out}/rewriter_NOTICE.txt', 'w') as f:
    f.write(ATTRIBUTION + '\n\n' + CHANGED + '\n')
print('tokenizer, system prompt, licence and notice written')
