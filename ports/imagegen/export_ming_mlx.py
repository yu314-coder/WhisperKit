"""Ming-Image-0.2-Design (inclusionAI, MIT) for MLX on the GPU.

    python export_ming_mlx.py <model dir> <out dir> [pieces]

<model dir> is a snapshot of inclusionAI/Ming-Image-0.2-Design. Writes, 64-byte
aligned for MappedWeights.swift, in parts under 2 GB:

- ming_text.partN.safetensors — the prompt reader, Ling-lite (BailingMoeV2,
  16.8B, 256 experts, 8 per token) without its vision tower, audio router and
  LM head. The experts, stacked per layer as (256, out, in), are 4-bit in
  groups of 64; attention, the dense first layer and the shared experts
  8-bit; the routers float32; norms and the learned query tokens bfloat16.
- ming_condition.partN.safetensors — what turns its states into the
  transformer's two captions: the connector (Qwen2 1.5B, run without a causal
  mask) and its projections, plus the direct projection of layers 5, 12 and
  20. 8-bit, the rest bfloat16.
- ming_transformer.partN.safetensors — the 6B diffusion transformer
  (Z-Image's design), 8-bit (scales bfloat16), the rest bfloat16.
- ming_vae.safetensors — the Qwen-Image (Wan 2.1) VAE decoder for one frame,
  RGBA out: each causal 3D convolution's last time slice, as MLX's
  (out, kh, kw, in); bfloat16.
- ming_vocab.json, ming_merges.txt — Ling's byte-level BPE.
"""
import glob, json, os, sys
import numpy as np
import mlx.core as mx
from safetensors import safe_open
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safetensors_aligned import save_aligned, save_parts  # noqa: E402

repo, out = sys.argv[1], sys.argv[2]
pieces = sys.argv[3].split(',') if len(sys.argv) > 3 else ['text', 'condition', 'transformer', 'vae', 'tokenizer']
os.makedirs(out, exist_ok=True)
NOTICE = {'notice': 'Ming-Image-0.2-Design, Copyright (c) Ant Group, MIT License. Converted by Yu Yao-Hsing '
                    'for the WhisperKit app: quantized and rearranged for MLX.'}


def reader(folder):
    """name -> loader, across a folder's safetensors shards, read as float32 MLX arrays."""
    handles = {path: safe_open(path, 'pt') for path in glob.glob(f'{folder}/*.safetensors')}
    where = {name: path for path, handle in handles.items() for name in handle.keys()}

    def get(name):
        return mx.array(handles[where[name]].get_tensor(name).float().numpy())
    return where, get


def q(result, name, value, bits, scale_dtype=mx.bfloat16):
    packed, scales, biases = mx.quantize(value, group_size=64, bits=bits)
    result[name + '.wq'], result[name + '.scales'], result[name + '.biases'] = \
        packed, scales.astype(scale_dtype), biases.astype(scale_dtype)
    mx.eval(packed, scales, biases)


if 'text' in pieces:
    names, get = reader(f'{repo}/mllm')
    config = json.load(open(f'{repo}/mllm/config.json'))['llm_config']
    text = {}
    q(text, 'embed', get('model.model.word_embeddings.weight'), 8)
    text['norm.weight'] = get('model.model.norm.weight').astype(mx.bfloat16)
    mlp = {}
    from safetensors.numpy import load_file  # noqa: F401  (float32 only)
    with safe_open(f'{repo}/mlp/model.safetensors', 'pt') as f:
        text['query_tokens'] = mx.array(f.get_tensor('query_tokens_dict.16x16').float().numpy()).astype(mx.bfloat16)
    for i in range(config['num_hidden_layers']):
        p = f'model.model.layers.{i}.'
        o = f'layers.{i}.'
        for norm in ['input_layernorm', 'post_attention_layernorm', 'attention.q_norm', 'attention.k_norm']:
            text[o + norm + '.weight'] = get(p + norm + '.weight').astype(mx.bfloat16)
        for linear in ['attention.query_key_value', 'attention.dense']:
            q(text, o + linear + '.weight', get(p + linear + '.weight'), 8)
        if i < config['first_k_dense_replace']:
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                q(text, o + f'mlp.{proj}.weight', get(p + f'mlp.{proj}.weight'), 8)
        else:
            for gate in ['gate', 'image_gate']:
                text[o + f'mlp.{gate}.weight'] = get(p + f'mlp.{gate}.weight')
                text[o + f'mlp.{gate}.expert_bias'] = get(p + f'mlp.{gate}.expert_bias')
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                q(text, o + f'mlp.shared_experts.{proj}.weight', get(p + f'mlp.shared_experts.{proj}.weight'), 8)
                stacked = mx.stack([get(p + f'mlp.experts.{e}.{proj}.weight') for e in range(config['num_experts'])])
                q(text, o + f'mlp.experts.{proj}', stacked, 4)
                del stacked
        print('text layer', i, flush=True)
    print('text parts:', save_parts(f'{out}/ming_text', text, NOTICE))
    del text

if 'condition' in pieces:
    names, get = reader(f'{repo}/connector')
    cond = {}
    for name in names:
        if name.startswith('model.embed_tokens') or name.startswith('lm_head'):
            continue
        value = get(name)
        short = 'connector.' + name[len('model.'):]
        if value.ndim == 2:
            q(cond, short, value, 8)
        else:
            cond[short] = value.astype(mx.bfloat16)
    with safe_open(f'{repo}/mlp/model.safetensors', 'pt') as f:
        for name in f.keys():
            if name.startswith('query_tokens'):
                continue
            value = mx.array(f.get_tensor(name).float().numpy())
            if value.ndim == 2:
                q(cond, name, value, 8)
            else:
                cond[name] = value.astype(mx.bfloat16)
    print('condition parts:', save_parts(f'{out}/ming_condition', cond, NOTICE))
    del cond

if 'transformer' in pieces:
    names, get = reader(f'{repo}/transformer')
    dit = {}
    for name in sorted(names):
        value = get(name)
        if value.ndim == 2 and value.shape[1] % 64 == 0:
            q(dit, name, value, 8)
        else:
            dit[name] = value.astype(mx.bfloat16)
    print('transformer parts:', save_parts(f'{out}/ming_transformer', dit, NOTICE))
    del dit

if 'vae' in pieces:
    names, get = reader(f'{repo}/vae')
    vae = {}
    for name in names:
        if not (name.startswith('decoder.') or name.startswith('post_quant_conv.')):
            continue
        value = get(name)
        if value.ndim == 5:          # causal (out, in, t, kh, kw): only the last slice sees one frame
            value = value[:, :, -1].transpose(0, 2, 3, 1)
        elif value.ndim == 4:        # Conv2d (out, in, kh, kw)
            value = value.transpose(0, 2, 3, 1)
        vae[name] = value.astype(mx.bfloat16)
    save_aligned(f'{out}/ming_vae.safetensors', vae, NOTICE)
    print('vae', len(vae))

if 'tokenizer' in pieces:
    tokenizer = json.load(open(f'{repo}/mllm/tokenizer.json'))
    json.dump(tokenizer['model']['vocab'], open(f'{out}/ming_vocab.json', 'w'), ensure_ascii=False)
    with open(f'{out}/ming_merges.txt', 'w') as f:
        f.write('#version: 0.2\n')
        for merge in tokenizer['model']['merges']:
            f.write((merge if isinstance(merge, str) else ' '.join(merge)) + '\n')
    added = {t['content']: t['id'] for t in tokenizer['added_tokens']}
    print('specials', {k: added[k] for k in ['<role>', '</role>', '<|role_end|>', '<|endoftext|>'] if k in added})
