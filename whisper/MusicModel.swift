import Foundation

/// The text-to-music models offered in the Music tab.
///
/// Unlike the Whisper lineup, these do not share one runtime. Stable Audio 3
/// runs as an MLX graph, Magenta RealTime 2 as Core ML on the Neural Engine,
/// and two of them have no Apple-silicon build published at all — their
/// weights have to be converted before they can be listed as anything but
/// unavailable. `availability` is what the picker reads, so a model whose
/// conversion is not finished says so instead of offering a button that fails.
enum MusicModel: String, CaseIterable, Identifiable {
    case aceStep15            = "ace-step-1.5-2b"
    case stableAudio3Medium   = "stable-audio-3-medium-1.4b"
    case stableAudio3Small    = "stable-audio-3-small-music-433m"
    case magentaRealtime2     = "magenta-realtime-2-small-230m"
    case musicGenSmall        = "musicgen-small-300m"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .aceStep15:          return "ACE-Step 1.5"
        case .stableAudio3Medium: return "Stable Audio 3 Medium"
        case .stableAudio3Small:  return "Stable Audio 3 Small-Music"
        case .magentaRealtime2:   return "Magenta RealTime 2 Small"
        case .musicGenSmall:      return "MusicGen Small"
        }
    }

    /// Parameter count, as the model is published.
    var parameterLabel: String {
        switch self {
        case .aceStep15:          return "2B"
        case .stableAudio3Medium: return "1.4B"
        case .stableAudio3Small:  return "433M"
        case .magentaRealtime2:   return "230M"
        case .musicGenSmall:      return "300M"
        }
    }

    /// Which runtime executes the model, once its weights exist locally.
    enum Engine {
        /// MLX graphs (`.npz`), executed on the GPU through MLX Swift.
        case mlx
        /// Core ML, executed on the Neural Engine.
        case coreML
    }

    var engine: Engine {
        switch self {
        case .magentaRealtime2: return .coreML
        default:                return .mlx
        }
    }

    /// Whether this model can run on this platform yet, and if not, why.
    enum Availability: Equatable {
        /// Implemented and runnable once the weights are downloaded.
        case ready
        /// Apple-silicon weights exist, but the pipeline is not wired up yet.
        case weightsPublished(note: String)
        /// No Apple-silicon build exists; the weights must be converted first.
        case needsConversion(note: String)
        /// Ported and working, but the weights may not be distributed here.
        case licenceRestricted(note: String)
        /// Runs, but not on a device with this much memory.
        case needsMoreMemory(note: String)
    }

    /// Physical memory this model needs, in gigabytes.
    ///
    /// Measured, not guessed: Medium peaks at 3.5 GB generating, which an 8 GB
    /// iPad Air cannot survive — iOS killed it mid-generation in testing. A
    /// 12 GB device has the headroom. Small peaks near 1.8 GB and runs
    /// anywhere.
    var minimumPhysicalMemoryGB: Double {
        switch self {
        case .stableAudio3Medium: return 10
        default:                  return 0
        }
    }

    static var physicalMemoryGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
    }

    var availability: Availability {
        switch self {
        case .stableAudio3Small:
            return .ready
        case .stableAudio3Medium:
            guard Self.physicalMemoryGB >= minimumPhysicalMemoryGB else {
                return .needsMoreMemory(
                    note: String(format: "Medium peaks around 3.5 GB while generating, which is more than iOS allows an app on a %.0f GB device — it is killed part-way. Small runs comfortably here.", Self.physicalMemoryGB))
            }
            return .ready
        case .magentaRealtime2:
            return .weightsPublished(
                note: "Core ML graphs are published, but text prompting needs MusicCoCa, which runs on a Mac today.")
        case .musicGenSmall:
            // Ported and verified against the reference (187 of 188 tokens
            // identical), but Meta releases the weights under CC-BY-NC 4.0 —
            // non-commercial only — so they are not shipped with this app.
            return .licenceRestricted(
                note: "Ported and working, but Meta licenses these weights for non-commercial use only (CC-BY-NC 4.0), so they are not distributed with this app.")
        case .aceStep15:
            // Measured on a Mac: 2.7 GB for a prompt alone, 3.2 GB with
            // lyrics. Medium needs 3.5 GB and an 8 GB iPad cannot survive it,
            // so this sits just under a line known to fail. Offered anyway
            // rather than fenced off — it may well fit where Medium does not
            // — but the note says plainly what the risk is.
            return .ready
        }
    }

    var isRunnable: Bool { availability == .ready }

    /// Only ACE-Step has a lyric encoder; the others take a prompt alone.
    var supportsLyrics: Bool { self == .aceStep15 }

    /// Shown under a runnable model when it is close to what the device can
    /// hold. ACE-Step peaks at 2.7 GB for a prompt and 3.2 GB with lyrics;
    /// Medium needs 3.5 GB and is killed on 8 GB hardware, so the margin here
    /// is real but thin.
    var memoryCaution: String? {
        guard self == .aceStep15, Self.physicalMemoryGB < 10 else { return nil }
        return String(format: "Peaks near 3 GB while generating. On this %.0f GB device that is close to the limit, so it may be stopped part-way — more likely with lyrics than without.", Self.physicalMemoryGB)
    }

    /// Where the app fetches weights from.
    ///
    /// Not Hugging Face directly: MLX reads `safetensors` and the published
    /// files are `npz`, so they are converted first and the conversions are
    /// hosted as a release. Stability's licence and a notice of exactly what
    /// was changed sit alongside them.
    private static let releaseRoot =
        "https://github.com/yu314-coder/WhisperKit/releases/download"

    /// Each model family has its own release, so they can be re-cut
    /// independently.
    var releaseTag: String {
        switch self {
        case .aceStep15: return "acestep-int8-v1"
        default:         return "sa3-small-weights-v1"
        }
    }

    /// Weights live in their own folder per family; switching models does not
    /// disturb another family's download.
    var weightsDirectory: URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let folder: String
        switch self {
        case .aceStep15: folder = "MusicModels/ace-step"
        default:         folder = "MusicModels/stable-audio-3"
        }
        return documents.appendingPathComponent(folder, isDirectory: true)
    }

    func downloadURL(for fileName: String) -> URL? {
        URL(string: "\(Self.releaseRoot)/\(releaseTag)/\(fileName)")
    }

    /// Files to fetch from `weightsRepo`, with their published sizes in MB.
    ///
    /// Stable Audio 3 is assembled from four separate graphs rather than one
    /// bundle: a shared text encoder, a diffusion transformer that differs per
    /// variant, and the SAME decoder that turns latents into audio.
    /// Exact byte counts, read from the hosted assets.
    ///
    /// Exact rather than rounded because these double as the "already
    /// downloaded" test. An earlier version compared against megabytes scaled
    /// by 900,000, which made the 792,862-byte conditioner look perpetually
    /// missing and re-downloaded it on every run.
    var weightFiles: [(path: String, bytes: Int64)] {
        // Text encoder, decoder and tokenizer are shared by both Stable Audio
        // variants, so switching between them only fetches a different DiT.
        let shared: [(String, Int64)] = [
            ("t5gemma_f16.safetensors", 567_416_533),
            ("same_s_decoder_f32.safetensors", 218_069_578),
            ("t5gemma_tokenizer.model", 4_241_003),
        ]
        switch self {
        case .stableAudio3Small:
            return (shared + [("dit_sm-music_f16.safetensors", 919_104_895),
                              ("sa3_conditioner_sm-music.safetensors", 792_862)])
                .map { (path: $0.0, bytes: $0.1) }
        case .stableAudio3Medium:
            // Two shards: one 2.9 GB file exceeds the 2 GB release-asset cap.
            return (shared + [("dit_medium_f16.part1.safetensors", 1_445_754_371),
                              ("dit_medium_f16.part2.safetensors", 1_461_439_235),
                              ("sa3_conditioner_medium.safetensors", 792_862)])
                .map { (path: $0.0, bytes: $0.1) }
        case .aceStep15:
            // int8 projections, fp16 elsewhere. The DiT is sharded because
            // 2.69 GB exceeds the 2 GB cap on a release asset.
            return [("ace_dit_q8.part1.safetensors", 1_265_350_530),
                    ("ace_dit_q8.part2.safetensors", 1_429_400_896),
                    ("ace_qwen_q8.safetensors", 670_383_070),
                    ("ace_vae_f16.safetensors", 337_352_796),
                    ("ace_silence.safetensors", 192_101),
                    ("ace_vocab.json", 2_776_833),
                    ("ace_merges.txt", 1_671_853)]
                .map { (path: $0.0, bytes: $0.1) }
        case .magentaRealtime2, .musicGenSmall:
            return []
        }
    }

    var downloadBytes: Int64 { weightFiles.reduce(0) { $0 + $1.bytes } }

    /// Download size for the picker. Unconverted models have no honest number
    /// to show, because nothing has been built to measure.
    var sizeLabel: String {
        let bytes = downloadBytes
        guard bytes > 0 else { return "—" }
        let mb = Double(bytes) / 1_000_000
        return mb >= 1000 ? String(format: "%.1f GB", mb / 1000) : String(format: "%.0f MB", mb)
    }

    var tagline: String {
        switch self {
        case .aceStep15:          return "Songs with vocals"
        case .stableAudio3Medium: return "Highest quality"
        case .stableAudio3Small:  return "Recommended"
        case .magentaRealtime2:   return "Live jamming"
        case .musicGenSmall:      return "Lightest"
        }
    }
}
