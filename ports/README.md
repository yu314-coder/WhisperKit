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
converters live here (`convert_weights.py`, `convert_neural_engine.py`,
`convert_large.py`). Text prompt and lyrics in any of
its 50 vocal languages, with the 1.7B planner as upstream runs it.

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

**Planner** (`ACEPlanner.swift`). Without it the diffusion transformer decides
when a song is over, and at 81 seconds four of four runs went silent 5 to 24
seconds early. With it, four of four have sound to the last second.

| check | result |
|---|---|
| Chat-template prompts, both phases, token ids | identical |
| Reasoning block vs PyYAML, 13 captions (folding, quoting, Ukrainian, Chinese) | identical |
| Next-token logits, float32 / shipped int8 | cosine 1.0 / 0.99999, same argmax |
| Same, through the KV cache | cosine 1.0 / 0.99999 |
| Guidance pair batched vs separate | cosine 0.99997 |
| Tokens to 25 Hz guide (FSQ + detokenizer) | cosine 0.999996 |

**Following the prompt, and holding together.** Scored with CLAP
(laion/clap-htsat-unfused), 7 prompts × 2 seeds, 45 s each: how well each
clip matches its own prompt, how alike neighbouring 3 s windows are, how
many neighbours differ abruptly (similarity below 0.8), and how alike
windows 15 s or more apart are.

| setting | own prompt | neighbours | abrupt | 15 s+ apart |
|---|---|---|---|---|
| 2B, plan for 2 of 8 steps (build 21) | 0.570 | 0.932 | 4/196 | 0.900 |
| 2B, plan throughout (upstream default) | 0.438 | 0.934 | 6/196 | 0.892 |
| **XL 4B, plan throughout** | **0.496** | **0.942** | 6/196 | **0.911** |
| 2B with the 4B planner | 0.501 | 0.913 | 17/196 | 0.843 |

Build 21 followed the plan for two steps and let the transformer continue
as plain text-to-music, which matched the words best at 45 s but came apart
over longer songs: at 81 s, windows 15 s apart agreed at 0.892 against 0.965
with the plan throughout (0.690 with no plan). Build 22 follows the plan
throughout, as upstream's `audio_cover_strength` does by default, and offers
XL for closer prompt-following. The 4B planner matched prompts better but
broke songs up more, so both versions keep the 1.7B.

The transformer shapes a whole piece to whatever window it renders, ending
included, so the window is made 20% longer than what is kept and the cut is
faded; no run above left a silent tail.

**The planner's guidance was broken by an MLX bug (fixed in 1.2 (24)).**
MLX's fused rotary embedding (mlx-swift 0.31.6) mis-rotates every row but
the first when a batch has one position per row — max difference 3.7
against the same row alone, and against its neighbouring positions. The
planner writes each code as exactly that batch: the guidance pair. So the
unconditional row was scrambled on every step (next-code entropy 9.8 where
the row alone has 8.1), and guidance against that noise sharpened the
plan until it looped: 90% of a 1:52 plan repeated its first 4 or 16
seconds, against 12-32% for a line-for-line Python copy of upstream's
sampler on the original weights. Found by feeding the reference's codes
through the port and comparing step by step — conditional alone matched
(entropy 4.09 / 4.09), unconditional alone matched, the pair did not,
blocks of several codes did. `ACEQwen3.rotate` folds such a batch into the
heads. After the fix the guided distribution matches the reference at every
stage of the song (entropy 4.28 / 4.24 early, 0.65 / 0.68 late); on seven
prompts at 81 s, own-prompt CLAP 0.506 -> 0.551 and no plan repeats more
than a quarter of itself. Every earlier build with the planner had it.

**Full precision.** The planner now runs as published, bfloat16 (upstream's
precision; Qwen3 overflows float16), and the condition encoder and hints in
float16 (`convert_full.py`); only XL's transformer stays int8, because at
float16 it is 8.1 GB. With the fix, seven prompts at 81 s:

| | own prompt | margin over others | neighbours | abrupt | 15 s+ apart |
|---|---|---|---|---|---|
| int8 planner, before the fix | 0.506 | +0.035 | 0.903 | 16/182 | 0.839 |
| int8 planner, fixed | 0.551 | +0.062 | 0.908 | 15/182 | 0.858 |
| **full precision, fixed** | **0.594** | **+0.140** | **0.914** | **11/182** | **0.863** |

Planning costs about 1.3x the int8 time (54 s against 40 s for 1:52 on an
M4).

**Loops.** Followed throughout, the plan's own habits become the song's,
and on long pieces the planner loops. For a 1:52 underscore it copied a
4-second pattern for 34 seconds and then wrote one token 446 times; another
seed replayed a 16-second phrase to the end (91% and 90% of tokens equal to
the stretch 4 or 16 s earlier). Four of seven ordinary prompts at 81 s had
exact copies of 37-67 s. Upstream samples the same way (temperature 0.85,
top-p 0.9, guidance 2.0, no repetition penalty). Two changes:

- *A copy guard* (`ACEPlanner.copyPenalties`, a "don't repeat yourself"
  sampler): a token that would extend an exact copy of earlier tokens past
  12 (2.4 s) is penalised, steeply. Where the planner is not looping a token
  never repeats more than twice in a row, so long exact copies are the
  failure, not the style. On the seven prompts, longest copy 67 s -> 3.6 s,
  own-prompt CLAP 0.506 -> 0.524, neighbouring windows 0.903 -> 0.892. A 4 s
  allowance was as smooth as none but let near-loops back (0.70-0.90).
- *Timelines planned part by part.* "0:16-0:34 a second voice enters…" in a
  prompt becomes a part: its description leads its own caption, and it is
  written as the continuation of every earlier part. On the 1:52 prompt,
  repetition at the loop lag fell from 0.91/0.90 to 0.25/0.08 and distinct
  tokens rose from 31/103 to 139/385; the parts sounded less alike (CLAP
  between parts 0.83 -> 0.68-0.74) and nearer their own descriptions. The
  timeline is taken out of the caption the transformer reads, which keeps
  only 256 tokens and would otherwise lose the style to it.

**DCW**, upstream's default sampler correction for turbo models, is ported
(Haar, closed form; latent matches the repository sampler at 0.9999985).
Its strengths depend on Think: the library default is 0.05 low / 0.02 high,
but upstream's interface — and its web demo — uses 0.02 / 0.06 with the
planner on. Builds before 1.2 (28) used the former with the planner. On XL,
seven prompts at 81 s: abrupt changes 12 -> 7 of 182, windows 15 s apart
0.870 -> 0.883. Upstream's exact length handling (render only what is
asked, silence allowed) left 6 of 7 songs with a silent tail again.

Departures from upstream defaults, each deliberate:

- **The caption is the user's.** Upstream lets the planner rewrite it; at its
  sampling temperature "a restrained underscore with soft granular pads" came
  back as energetic synthwave, and the music followed. `use_cot_caption` is
  upstream's own switch for this.
- **No planned silence.** The planner learned from recordings that end in
  silence and writes 5-15 s of it at the end of a requested length. The
  silence token (35847, found by tokenizing the silence latent) is excluded.
- **A longer render**, cut and faded — see above.
- **Tempo, key and meter from the prompt** are written into the reasoning
  instead of sampled, as upstream does for its UI fields.
- **Phase 1 without the 2,300-line state machine:** field names are forced in
  order and values sampled, with the same multi-line caption rule.

Planning an 81-second song takes about 25 s on an M4 Mac; the guidance pair
runs as one batch of two, which costs the same as one row.

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

**Against the official pipeline, run on the same Mac** (`generate_music` at
its defaults: planner rewriting the caption, plan throughout, peak
normalisation) on the seven prompts at 81 s:

| | own prompt | margin | neighbours | abrupt | 15 s+ apart | silent tail |
|---|---|---|---|---|---|---|
| official, 2B | 0.556 | +0.061 | 0.877 | 30/182 | 0.716 | 6/7 |
| app, 2B (Neural Engine) | 0.594 | +0.140 | 0.914 | 11/182 | 0.863 | 0/7 |
| app, XL float16 | 0.608 | +0.125 | 0.916 | 12/182 | 0.870 | 0/7 |

**XL** was checked against the official XL (bfloat16, on the Mac GPU) on
the same inputs: packed conditioning cosine 0.999995; one velocity
prediction 0.99918 at float16, 0.99878 at int8. It ships at float16.

**Loudness.** The decoder's output runs past full scale on loud passages and
was written clipped (every song peaked at 0 dBFS; 5,843 clipped samples in
one 1:52 song). As upstream does by default, every song is now scaled to a
-1 dBFS peak before it is written.

**Held in memory (1.2 (30)).** Weights were always mapped, which kept them
out of the footprint but on an 8 GB iPad made XL at float16 (8.1 GB) page
in from storage at every step: 165 s for 30 seconds of music, the gauge
near empty. A stage's weights are now read into memory in one pass when
they fit what iOS allows (about 7.5 GB there, with the increased memory
limit entitlement), and mapped only when they do not. XL ships twice: int8
(4.7 GB, held in memory) and XL Full (float16). At 81 s on seven prompts
they score alike — 0.608 / 0.609 own prompt, 6 / 7 abrupt changes, 0.883 /
0.883 across 15 s — so int8 is the version for 8 GB devices. The 4B
planner, retested after the rope fix, still did worse with both
transformers (own prompt 0.573 with 2B, 0.589 with XL).

**Versions.** Both share every file but the transformer:

- *ACE-Step 1.5* runs the 2B transformer on the Neural Engine
  (`convert_neural_engine.py`, `whisper/ACENeuralTransformer.swift`): 24
  layers as four Core ML programs of six, float16, each with one function
  per length (15 sizes, up to 7:40 rendered) and conditioning size (2),
  sharing one copy of the weights. Padding is masked out of every attention,
  so kept positions see exactly what they would unpadded (checked: identical
  to the unpadded computation in float32). The sliding-window layers attend
  in blocks of 128 queries against 384 keys. RMS norms run on a pre-scaled
  input so squaring cannot overflow float16.
- *ACE-Step 1.5 XL* runs the 4B turbo transformer (32 layers, width 2,560)
  on the GPU at float16 (`convert_full.py`; the int8 set from
  `convert_large.py` is superseded). Of its 201 non-transformer tensors,
  192 are bit-identical to the 2B's; the 9 it retrained ship as a 5 MB file
  laid over the 2B condition encoder.

| 45 s song, M4 Mac | transformer, 8 steps | latent vs float32 transformer |
|---|---|---|
| 2B int8, GPU (build 21) | 11 s | 0.99954 |
| 2B float16, Neural Engine | 3.6 s | 0.99963 |

The first run at a new length waits while iOS compiles that length's
programs for the Neural Engine — two minutes on the Mac with its CPU busy,
five seconds once cached.

Still missing: the FSQ audio encoder side, which turns supplied audio into
tokens — so reference audio and covers are unavailable.

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
