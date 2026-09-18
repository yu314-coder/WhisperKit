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

Components verified; end-to-end assembly unfinished.

| stage | max diff | cosine |
|---|---|---|
| Qwen3 text embedder, 28 layers | 0.00084 | 1.0 |
| 1.7B planner, 28 layers | 0.00043 | 1.0 |
| DiT layer, sliding / full | 0.00035 / 0.0026 | 1.0 / 0.99999994 |
| **Full DiT forward, 24 layers** | **4.3e-05** | **1.0** |
| VAE decoder, 1920x upsample | 3.1e-06 | 0.9999999 |

**Cannot run on iOS.** The weights are 10.09 GB *already in bfloat16*, so
conversion cannot shrink them; the largest device on hand is 12 GB. This is a
macOS-phase model, where 24 GB makes it viable.

Still to do: the conditioning encoder (lyric and timbre branches — both reuse
the same Qwen3 block already verified here), and the sampler, which is an
8-step Euler integration of the velocity the DiT already produces correctly.
The sliding window is also untested: verification used 20 patches against a
128-wide window, so sliding and full attention were identical.

## What kept going wrong

Every failure was a convention, never the mathematics, and none were visible
in the weights:

- **RoPE belongs to self-attention only.** True in T5, MusicGen and ACE-Step
  alike; applying it in cross-attention silently changes the output.
- **`to_kv` packs k, kDiff, v** in Stable Audio 3 Medium — not k, v, kDiff.
  The wrong order still decodes, as an undifferentiated wash.
- **Oobleck's Snake stores log-scale parameters.** Reading alpha and beta raw
  gave maxdiff 18,398 and cosine -0.044.

The method that caught all three: run the reference, save golden tensors, and
compare maxdiff and cosine per stage before moving on.
