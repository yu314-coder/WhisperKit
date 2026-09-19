# Model ports

Swift/MLX ports written for the Music tab, kept here because they are not all
shippable in the iOS app. Each was verified against the PyTorch reference by
comparing tensors stage by stage, not by listening.

## musicgen/ — MusicGen Small (Meta)

Complete and verified end to end: 187 of 188 greedy tokens identical to
PyTorch, audio cosine 0.9998. The single difference is an argmax near-tie
flipped by fp16 rounding.

| stage | max diff | cosine |
|---|---|---|
| T5-base encoder | 1.4e-06 | 1.0 |
| Decoder LM, 24 layers | 0.0038 | 0.99999994 |
| EnCodec decoder | 1.25e-06 | 1.0 |

**Not shipped.** Meta licenses the weights CC-BY-NC 4.0 — non-commercial only,
which does not fit an App Store app. The code is here; the weights are not
hosted or bundled.

Written from scratch because the reference MLX port runs T5 in PyTorch: the T5
relative-position bucket bias, a 2-layer LSTM (MLX has none), reflect padding
as a gather, transposed-conv overhang trimming, the 4-codebook delay pattern,
and classifier-free guidance.

## acestep/ — ACE-Step 1.5 (MIT)

Generating. Text prompt and optional lyrics, both verified stage by stage.

| stage | max diff | cosine |
|---|---|---|
| Qwen3 text embedder, 28 layers | 0.00084 | 1.0 |
| 1.7B planner, 28 layers | 0.00043 | 1.0 |
| DiT layer, sliding / full | 0.00035 / 0.0026 | 1.0 / 0.99999994 |
| **Full DiT forward, 24 layers** | **4.3e-05** | **1.0** |
| VAE decoder, 1920x upsample | 3.1e-06 | 0.9999999 |
| Lyric encoder, 8 layers | 2.1e-06 | 0.99999994 |
| Timbre encoder, 4 layers | 4.1e-06 | 1.0 |
| Packed conditioning | 4.1e-06 | 0.9999999 |

**Shipped, quantized.** As published it is 10.09 GB and already bfloat16, so
conversion alone cannot shrink it. Two changes make it fit: the 1.7B planner
is dropped (its hints are optional and the generation path passes none), and
the projections are quantized to int8. Download is 3.71 GB.

int8 is the floor, not a preference — velocity cosine against the fp32
reference: int8 0.9989, 6-bit 0.9826, 4-bit/group-32 0.8583, 4-bit/group-64
0.7465.

Measured generating 8 s of 48 kHz stereo on a Mac: **2.7 GB peak for a prompt,
3.2 GB with lyrics**, in 2-3 s. For comparison Stable Audio 3 Medium needs
3.5 GB and is killed on an 8 GB iPad, so the margin is real but thin.

Still missing: the FSQ tokenizer and detokenizer. They turn supplied audio
into the 64-wide acoustic latents the timbre encoder consumes, so reference
audio and cover generation are unavailable — the timbre encoder itself is
ported and verified, it simply has no way to be fed. The sliding window is
also untested: verification used 20 patches against a 128-wide window, so
sliding and full attention were identical.

## What kept going wrong

Every failure was a convention, never the mathematics, and none were visible
in the weights:

- **RoPE belongs to self-attention only.** True in T5, MusicGen and ACE-Step
  alike; applying it in cross-attention silently changes the output.
- **`to_kv` packs k, kDiff, v** in Stable Audio 3 Medium — not k, v, kDiff.
  The wrong order still decodes, as an undifferentiated wash.
- **Oobleck's Snake stores log-scale parameters.** Reading alpha and beta raw
  gave maxdiff 18,398 and cosine -0.044.
- **ACE-Step's timbre encoder ships a `special_token` it never uses.** The
  line prepending it is commented out in the reference; position zero of the
  projected input carries the summary. Prepending it scored cosine 0.85.

The method that caught all three: run the reference, save golden tensors, and
compare maxdiff and cosine per stage before moving on.
