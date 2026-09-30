# WhisperKit

On-device speech transcription and music generation for iPhone and iPad. Nothing you record, type or create leaves the device — transcription runs on the Apple Neural Engine via [WhisperKit](https://github.com/argmaxinc/WhisperKit), and music is made by a port of [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) running on the Neural Engine and GPU.

> **Note:** this is the iOS/iPadOS app, released on the App Store as [WhisperKit](https://apps.apple.com/app/whisperkit/id6764759491). It is a *consumer* of the Argmax WhisperKit library, not that library itself.

## Transcribe

- **Record, import, share or link** — record in the app; import audio or video from Files, iCloud or Photos; share a file to the app from Voice Memos, Files or Mail; or paste a link to an audio or video file and the app downloads it (video platforms such as YouTube and Instagram are refused — see `MediaLink.swift`).
- **99 languages**, with auto-detection or an explicit choice; Chinese can be written in Simplified or Traditional characters.
- **Four models** — Small (217 MB), Turbo (646 MB), Large V3 (948 MB) and Large V3 Turbo (1.1 GB). Each downloads once, then works offline, on the GPU or the Neural Engine.
- **Library** — every transcript is saved with its audio and waveform, searchable by title, text or language; tap a line to play from it; speaker labels per segment.
- **Export** as TXT, SRT, VTT, Markdown or JSON.

## Make music (beta)

- **Describe the music** — style, mood, instruments, voice; say a length ("90 seconds", "2:30"), a tempo or a key and it is used.
- **Lyrics in either box** — type them in the lyrics box or right in the description (`lyric is "…"`, `歌词是「…」`, lines after a blank line). On devices with Apple Intelligence, Apple's on-device language model finds them (`LyricsFinder.swift`), and every line it returns must appear in the prompt letter for letter. Section tags are added for the model; the sung language follows the lyrics' script.
- **Fuller arrangement** (on by default) — the planner writes a full arrangement after your words, which stay first and unchanged.
- **Three versions** — ACE-Step 1.5 (2B, on the Neural Engine, ~9 GB download), XL (4B at int8, ~11 GB) and XL Full (4B at float16, ~15 GB). Weights come from this repository's [releases](https://github.com/yu314-coder/WhisperKit/releases).
- **History** — every song keeps its prompt, lyrics and settings; open one to play, share or use its prompt again.

## In the background

On iOS 26 and later, transcriptions, downloads and music keep running when you switch apps: each is a `BGContinuedProcessingTask` whose progress iOS shows (`BackgroundWork.swift`). Work that needs the GPU asks for background GPU time; where a device can't give it, `GPUGate` pauses the work at its next step instead of letting Metal refuse it, and it carries on when the app is back. Earlier iOS versions get the usual short background time and a Live Activity.

## Privacy

No accounts, no analytics, no ads, no servers. The app's only network requests are one-directional HTTPS downloads: transcription models from Hugging Face, music models from this repository's GitHub releases, and a file from a link the user pastes. See [PRIVACY.md](PRIVACY.md) and the [published policy](https://yu314-coder.github.io/privacy.html#whisper).

## Requirements

- iOS / iPadOS 18.5+ (background continued processing and Apple Intelligence lyric detection need iOS 26+)
- Xcode 26 or later (developed with Xcode 27)
- A device with a Neural Engine; music needs a recent iPhone or iPad with plenty of memory (XL warns below 7 GB)

## Building

```bash
git clone https://github.com/yu314-coder/WhisperKit.git
cd WhisperKit
open whisper.xcodeproj
```

Swift Package Manager resolves WhisperKit and mlx-swift on first build. mlx-swift ships a build plugin Xcode won't run untrusted, so command-line builds need `-skipPackagePluginValidation`. Set your own signing team in the `whisper` and `TranscriptionWidgetExtension` targets before running on a device. MLX doesn't run in the Simulator, so music generation needs a device.

## Layout

| Path | Role |
|---|---|
| `whisper/ContentView.swift` | Transcribe tab — recording, import, models, transcription |
| `whisper/MusicView.swift` | Music tab — prompt, lyrics, progress, result, history |
| `whisper/MusicEngine.swift` | Downloads music weights and runs a generation |
| `whisper/ACEPipeline.swift` | ACE-Step pipeline: planner → text and lyric encoders → transformer → VAE |
| `whisper/ACEPlanner.swift` | The 5 Hz language-model planner (song layout, captions) |
| `whisper/ACENeuralTransformer.swift` | The 2B transformer as Core ML programs on the Neural Engine |
| `whisper/MusicRequest.swift` | Sorts a prompt into description and lyrics |
| `whisper/LyricsFinder.swift` | On-device lyric detection with Foundation Models |
| `whisper/MediaLink.swift`, `LinkImportSheet.swift` | Transcribe a link |
| `whisper/BackgroundWork.swift` | Continued-processing tasks and the GPU gate |
| `whisper/HelpView.swift` | The "?" pages |
| `whisper/SavedTranscript.swift`, `SavedMusic.swift` | SwiftData models |
| `whisper/AudioConverter.swift` | Normalizes any input to 16 kHz mono PCM WAV |
| `whisper/TranscriptLibraryView.swift`, `TranscriptDetailView.swift` | Transcript library, playback, speakers, export |
| `TranscriptionWidget/` | Live Activity and Dynamic Island |
| `ports/` | Conversion scripts and notes for the music models (`ports/README.md`) |

### A note on `AudioConverter`

All input is converted to 16 kHz mono 16-bit PCM WAV before transcription. Beyond being what Whisper consumes internally, this sidesteps a real bug: under "Designed for iPad" on Mac, `ExtAudioFile`'s AAC decoder fails with `kAudioFileUnsupportedDataFormatError`. `AVAssetReader` uses a different decode path that works on both platforms.

## License

MIT — see [LICENSE](LICENSE). WhisperKit and ACE-Step 1.5 are MIT-licensed too.

## Author

Yu Yao-Hsing ([@yu314-coder](https://github.com/yu314-coder))
