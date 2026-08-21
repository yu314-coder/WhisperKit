# WhisperKit

On-device speech transcription for iPhone and iPad. Audio never leaves the device — transcription runs locally on the Apple Neural Engine via [WhisperKit](https://github.com/argmaxinc/WhisperKit).

> **Note:** this is the iOS/iPadOS app. It is a *consumer* of the Argmax WhisperKit library, not that library itself.

## What it does

- **Record, import, or pull from Photos** — capture audio in-app, import an audio/video file from Files or iCloud, or transcribe a video from your Photos library.
- **99 languages**, with auto-detection or an explicit language choice.
- **Four model sizes** — Base (~150 MB) through Large V3 Turbo (~950 MB). Each downloads once and then works entirely offline.
- **Library** — every transcript is saved to SwiftData with its audio, searchable by title, body text, or language.
- **Segment playback** — tap any segment to seek the audio to it; the active line highlights as it plays. Speaker labels are editable per segment.
- **Export** as TXT, SRT, VTT, Markdown, or JSON.
- **Live Activity** — progress on the Lock Screen and in the Dynamic Island while transcribing in the background.

## Privacy

No accounts, no analytics, no ads, no servers. The only network request the app makes is downloading model weights from Hugging Face's public model hub; no user data travels in either direction. See [PRIVACY.md](PRIVACY.md).

## Requirements

- iOS / iPadOS 18.5+
- Xcode 16+ (developed against Xcode 26)
- A device with a Neural Engine for usable performance

## Building

```bash
git clone https://github.com/yu314-coder/WhisperKit.git
cd WhisperKit
open whisper.xcodeproj
```

Swift Package Manager resolves WhisperKit on first build. Set your own signing team in the `whisper` and `TranscriptionWidgetExtension` targets before running on a device.

## Layout

| Path | Role |
|---|---|
| `whisper/ContentView.swift` | Main screen — recording, import, model management, transcription |
| `whisper/SavedTranscript.swift` | SwiftData models (`SavedTranscript`, `SavedSegment`) |
| `whisper/AudioConverter.swift` | Normalizes any input to 16 kHz mono PCM WAV |
| `whisper/AudioFiles.swift` | Saved-audio storage, addressed by relative path |
| `whisper/TranscriptLibraryView.swift` | Searchable transcript library |
| `whisper/TranscriptDetailView.swift` | Segment list, playback, speaker labels, export |
| `whisper/TranscriptPlayer.swift` | `AVAudioPlayer` wrapper publishing the playhead |
| `whisper/TranscriptExporter.swift` | TXT / SRT / VTT / Markdown / JSON writers |
| `TranscriptionWidget/` | Live Activity and Dynamic Island |

### A note on `AudioConverter`

All input is converted to 16 kHz mono 16-bit PCM WAV before transcription. Beyond being what Whisper consumes internally, this sidesteps a real bug: under "Designed for iPad" on Mac, `ExtAudioFile`'s AAC decoder fails with `kAudioFileUnsupportedDataFormatError`. `AVAssetReader` uses a different decode path that works on both platforms.

## License

No license has been chosen yet, so default copyright applies — all rights reserved.

## Author

Yu Yao-Hsing ([@yu314-coder](https://github.com/yu314-coder))
