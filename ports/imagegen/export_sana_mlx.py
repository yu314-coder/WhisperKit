"""SANA-Sprint's MLX side: what runs on the GPU rather than the Neural Engine.

    python export_sana_mlx.py <diffusers repo dir> <out dir> [bits]
    python export_sana_mlx.py --gemma-from <Gemma 2 2B-it checkpoint dir> <out dir> [bits]

Writes, all 64-byte aligned for MappedWeights.swift:

- sana_gemma_q<bits>.partN.safetensors — Gemma 2 2B, the prompt encoder. Its
  matrices and embedding table quantized to <bits> (default 8) in groups of
  64 as `<name>.wq/.scales/.biases`; norms float32. Through the whole
  pipeline 8 bits draws the reference picture (PSNR 41 / 30 dB against
  float32 Gemma's 51 / 39); 6 bits, or a 4-bit embedding table with 8-bit
  layers, changed faces visibly (28 / 23 dB). Gemma runs with float32
  activations: its residual stream overflows float16. In parts under 2 GB,
  since MLX can't view a bigger file as one array.
- sana_extras.safetensors — the caption projection and its RMSNorm
  (float16), and the timestep/guidance modulation for the fixed two-step
  schedule (float32): `modulation` (2, 6912) and `embedded` (2, 1152), one
  row per step. SANA-Sprint always samples at t = 1.5708 and 1.3 with
  guidance 4.5 (scaled by 0.1, as upstream does), so these never change.
- sana_decoder_head.safetensors — the DC-AE decoder's input convolution and
  its first three up blocks (32 to 128 px, the ones with linear attention),
  float16, convolution weights in MLX's (out, kh, kw, in) layout. The rest
  of the decoder is a Core ML program (convert_sana.py decoder_tail).
- gemma_bpe.bin — the tokenizer's tables (see write_tokenizer).

--gemma-from writes only the prompt encoder, as
sana_gemma_abliterated_q<bits>.partN.safetensors, from a Gemma2ForCausalLM
checkpoint of the same model: the app's is the abliterated build
(IlyaGusev/gemma-2-2b-it-abliterated, Gemma terms), which differs from
SANA's own only in the matrices abliteration orthogonalizes (cosine
0.9997+), with the refusal direction gone from what it writes into the
residual stream.
"""
import glob, json, math, os, struct, sys
import numpy as np
import torch
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safetensors_aligned import save_aligned, save_parts  # noqa: E402
from safetensors.torch import load_file
from diffusers import SanaTransformer2DModel

# --extras-only: just sana_extras.safetensors, for another size of SANA-Sprint
# (1.6B) that shares the 0.6B's Gemma, decoder and tokenizer byte for byte.
extras_only = '--extras-only' in sys.argv
gemma_from = '--gemma-from' in sys.argv
args = [a for a in sys.argv[1:] if a not in ('--extras-only', '--gemma-from')]
repo, out = args[0], args[1]
bits = int(args[2]) if len(args) > 2 else 8
os.makedirs(out, exist_ok=True)


def to_mx(t, dtype=mx.float16):
    return mx.array(t.float().numpy()).astype(dtype)


# Gemma -----------------------------------------------------------------------
if not extras_only:
    gemma = {}
    if gemma_from:
        # A causal LM: its model's weights under `model.`, and an LM head
        # (tied to the embeddings) that the encoder doesn't use.
        for path in sorted(glob.glob(f'{repo}/model-*.safetensors')):
            gemma.update({k[len('model.'):]: v for k, v in load_file(path).items() if k.startswith('model.')})
    else:
        for path in sorted(glob.glob(f'{repo}/text_encoder/*.safetensors')):
            gemma.update(load_file(path))
    tensors = {}
    for name, weight in gemma.items():
        if weight.ndim == 2:
            wq, scales, biases = mx.quantize(to_mx(weight, mx.float32), group_size=64, bits=bits)
            tensors[name + '.wq'], tensors[name + '.scales'], tensors[name + '.biases'] = \
                wq, scales.astype(mx.float16), biases.astype(mx.float16)
        else:
            tensors[name] = to_mx(weight, mx.float32)
    mx.eval(tensors)
    stem = 'sana_gemma_abliterated' if gemma_from else 'sana_gemma'
    print('Gemma parts:', save_parts(f'{out}/{stem}_q{bits}', tensors))
    del gemma, tensors
    if gemma_from:
        sys.exit(0)

# Caption projection and modulation ------------------------------------------
transformer = SanaTransformer2DModel.from_pretrained(repo, subfolder='transformer', torch_dtype=torch.float32).eval()
extras = {}
for name, weight in transformer.state_dict().items():
    if name.startswith('caption_projection.') or name.startswith('caption_norm.'):
        extras[name] = to_mx(weight)
with torch.no_grad():
    rows = []
    for t in (1.5708, 1.3):
        scm = torch.tensor([math.sin(t) / (math.cos(t) + math.sin(t))], dtype=torch.float32)
        rows.append(transformer.time_embed(scm, guidance=torch.tensor([4.5 * 0.1]), hidden_dtype=torch.float32))
extras['modulation'] = mx.array(torch.cat([r[0] for r in rows]).numpy())
extras['embedded'] = mx.array(torch.cat([r[1] for r in rows]).numpy())
save_aligned(f'{out}/sana_extras.safetensors', extras)

# DC-AE decoder, first three stages ------------------------------------------
if not extras_only:
    vae = load_file(glob.glob(f'{repo}/vae/*.safetensors')[0])
    head = {}
    for name, weight in vae.items():
        if not name.startswith('decoder.'):
            continue
        name = name[len('decoder.'):]
        if not name.startswith(('conv_in.', 'up_blocks.3.', 'up_blocks.4.', 'up_blocks.5.')):
            continue
        if weight.ndim == 4:              # conv (out, in, kh, kw) -> (out, kh, kw, in)
            weight = weight.permute(0, 2, 3, 1).contiguous()
        head[name] = to_mx(weight)
    save_aligned(f'{out}/sana_decoder_head.safetensors', head)


# Tokenizer --------------------------------------------------------------------
def write_tokenizer(path):
    """Gemma's tokenizer as integer tables, so the app needn't parse a 34 MB
    JSON of 580,604 merges. All little-endian u32:

        magic 'GBPE', single-character count S, merge count M, added count A
        S x (code point, id), sorted by code point
        256 x id of the byte-fallback tokens <0x00>...<0xFF> (<unk> if none)
        M x (left id, right id, merged id, rank), sorted by (left, right)
        A x (id, UTF-8 length), then the added tokens' UTF-8, concatenated

    Added tokens (newline runs, <start_of_turn>, HTML tags...) are matched in
    the raw text before anything else, longest first, as Hugging Face does.
    """
    data = json.load(open(glob.glob(f'{repo}/tokenizer/tokenizer.json')[0]))
    model = data['model']
    assert model['type'] == 'BPE' and model['byte_fallback'] and not model['ignore_merges']
    assert data['normalizer'] == {'type': 'Replace', 'pattern': {'String': ' '}, 'content': '▁'}
    vocab = model['vocab']
    singles = sorted((ord(token), i) for token, i in vocab.items() if len(token) == 1)
    # A few bytes have no fallback token (tab is an ordinary one); a character
    # needing them would be <unk>, as upstream.
    byte_ids = [vocab.get(f'<0x{b:02X}>', vocab[model['unk_token']]) for b in range(256)]
    merges = []
    for rank, pair in enumerate(model['merges']):
        left, right = pair if isinstance(pair, list) else pair.split(' ', 1)
        merges.append((vocab[left], vocab[right], vocab[left + right], rank))
    merges.sort()
    added = [(a['id'], a['content'].encode()) for a in data['added_tokens']]
    assert not any(a['lstrip'] or a['rstrip'] or a['single_word'] or a['normalized'] for a in data['added_tokens'])
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sIII', b'GBPE', len(singles), len(merges), len(added)))
        f.write(np.array(singles, dtype='<u4').tobytes())
        f.write(np.array(byte_ids, dtype='<u4').tobytes())
        f.write(np.array(merges, dtype='<u4').tobytes())
        f.write(np.array([(i, len(b)) for i, b in added], dtype='<u4').tobytes())
        f.write(b''.join(b for _, b in added))
    print(f'{os.path.basename(path)}: {len(singles)} characters, {len(merges)} merges, '
          f'{len(added)} added tokens, {os.path.getsize(path):,} bytes')


if __name__ == '__main__' and not extras_only:
    write_tokenizer(f'{out}/gemma_bpe.bin')
