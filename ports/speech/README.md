# Transcribe tab: the NVIDIA engines

Canary 1B v2 and Nemotron 3.5 ASR run beside Whisper (WhisperKit) in the Transcribe tab,
on the GPU with MLX. The model code is vendored from
[mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift) (MIT) into `whisper/SpeechModels/`,
minus its Hugging Face downloader; `whisper/SpeechEngine.swift` and `LiveTranscriber.swift` are the app's side.

## Weights

Published unchanged on the `speech-v1` release, each with its licence file:

| Engine | Source | Licence | Size |
|---|---|---|---|
| Canary 1B v2 | `Mediform/canary-1b-v2-mlx-q8` (8-bit MLX of `nvidia/canary-1b-v2`) | CC BY 4.0 | 1.1 GB |
| Nemotron 3.5 ASR | `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` (of `nvidia/nemotron-3.5-asr-streaming-0.6b`) | OpenMDW 1.1 | 0.76 GB |

## Benchmark (FLEURS, 100 clips a language, M4)

`bench.sh` runs the app's four Core ML Whisper builds through `whisperkit-cli` (Neural Engine),
`bench_mlx.py` / `bench_canary.py` the NVIDIA models through Python mlx-audio (GPU), and
`score.py` scores them: English WER with Whisper's English normalizer; Chinese and Japanese
CER with punctuation and spaces removed and Chinese folded to Simplified. Speed is audio
seconds per wall-clock second.

| Model | English WER | Chinese CER | Japanese CER | Speed |
|---|---|---|---|---|
| Whisper Small | 8.77 | 18.28 | 14.82 | 13–17× |
| Whisper Turbo (large-v3-turbo) | 6.55 | **13.41** | **11.13** | 10–13× |
| Whisper Large V3 | 5.91 | 15.12 | 14.33 | 3–4× |
| Whisper Large V3 Turbo (Argmax) | 5.77 | 15.12 | 14.07 | 4–5× |
| Nemotron 3.5 ASR 0.6B | 8.58 | 18.80 | 12.35 | 9–11× |
| Canary 1B v2 | **4.38** | — | — | 17× |

The vendored Swift code reproduces these (Canary 4.29; Nemotron 8.68 / 19.12 / 12.43). Large V3
loses Chinese and Japanese to Turbo by adding phrases at the ends of clips ("谢谢大家", a
YouTube outro).

## App side

- Files: cut at pauses (Canary 10–20 s pieces, Nemotron 45–60 s), lines timed — Canary's by
  sentence length within its piece, Nemotron's by its own alignment — and lines over ~14 s cut
  at commas. Returned as WhisperKit `TranscriptionResult`s, so saving, subtitles and playback
  are Whisper's.
- Live (Nemotron): microphone → 16 kHz → the streaming session every ~0.3 s at 560 ms chunks,
  restarted at a pause every 25–60 s (the session re-reads its whole buffer each step; the model
  looks only 4.5 s back). On the M4 the feed runs about 8× faster than real time.
