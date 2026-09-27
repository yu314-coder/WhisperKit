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

Ships in the app — the code is `whisper/ACE*.swift`; only the weight
converter lives here (`convert_weights.py`). Text prompt and lyrics in any of
its 50 vocal languages.

Verified against the official pipeline (`ace-step/ACE-Step-1.5`,
`generate_audio`), not only module by module — the module checks all passed
while the port was conditioning the model wrongly, because each module was
fed what the port fed it rather than what the official code does:

| check | max diff | cosine |
|---|---|---|
| Caption and lyric token ids, incl. 195 Ukrainian lyric tokens | identical | — |
| Text encoder (float32 weights) | 0.00026 | 1.0 |
| Packed conditioning (float32) | 0.00026 | 0.9999999 |
| **Final latent after 8 steps, 30 s (float32)** | 0.149 | **0.99998** |
| Final latent, shipped weights | 3.16 | 0.983 |
| Tiled decode (48 + 12 overlap) vs whole clip | 0 | 1.0 |

What the module checks missed, all fixed:

- **Prompt layout.** The caption goes in an instruction/caption/metadata
  template ending in `<|endoftext|>`, and lyrics in a language/lyric template.
  The tokenizer must read the literal `<|endoftext|>` as the special token.
- **Lyrics are table rows.** The lyric encoder takes `embed_tokens` lookups,
  not text-encoder output. An instrumental still sends "[Instrumental]" in
  language "unknown", never an empty slot.
- **Timbre is always present.** Without reference audio it is read from the
  first 750 frames of the silence latent.
- **Chunk mask is ones**, meaning "generate every frame". Zeros mean "keep".
- **Sliding window.** Even-numbered layers of every stack attend only within
  128 positions. Invisible below 130 positions, which is all the module
  checks used; a 30-second clip is 375, and timbre is always 750.
- **Silence latent** was trimmed to 60 s, which capped generation there.

**Memory.** Peak for a 30-second song on a Mac went from 3.58 GB to 0.74 GB,
same output and speed; the 6:24 maximum peaks at 1.6 GB.

- Weights are mapped (`MappedWeights.swift`), not read into memory, so they
  are clean file-backed pages rather than memory charged to the app. This
  alone took diffusion from 2.45 GB to 0.8 GB. Files are written with every
  tensor 64-byte aligned so each is a plain view.
- One file per stage, loaded and released in turn: text encoder, condition
  encoder, diffusion transformer, decoder. The audio tokenizer and
  detokenizer (210M parameters the text path never runs) and the VAE's
  encoder half are no longer shipped.
- Freed GPU memory returns to the system a few hundred milliseconds late;
  the generator waits for the footprint to settle between stages instead of
  stacking one stage on the last.
- Attention is MLX's fused kernel. Unfused, the 6:24 maximum needs a
  16 x 4,800 x 4,800 score matrix — 1.5 GB — per layer.
- MLX lowers strided transposed convolutions to an explicit unfold, a 1 GB
  temporary in the decoder's fourth block. With kernel = 2 x stride it is one
  matrix product and a neighbour shift instead (matches to 2.7e-06).
- The decoder runs in 48-frame windows with 12 frames of context either side,
  in float16, streaming to the WAV file.

**Precision.** int8 for the two transformers; below that the output degrades
(velocity cosine: int8 0.9989, 6-bit 0.9826, 4-bit 0.7465). The text encoder
stays float16: Qwen3's outlier channels made it the largest single source of
drift at int8 (final latent 0.960 with it quantized, 0.983 without).

Still missing: the FSQ tokenizer and detokenizer, which turn supplied audio
into acoustic latents — so reference audio and covers are unavailable.

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

The method that caught these: run the reference, save golden tensors, and
compare maxdiff and cosine per stage before moving on.
