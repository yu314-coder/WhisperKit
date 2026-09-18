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
    }

    var availability: Availability {
        switch self {
        case .stableAudio3Small:
            return .ready
        case .stableAudio3Medium:
            return .ready
        case .magentaRealtime2:
            return .weightsPublished(
                note: "Core ML graphs are published, but text prompting needs MusicCoCa, which runs on a Mac today.")
        case .musicGenSmall:
            return .needsConversion(
                note: "An MLX port exists for macOS in Python. Needs porting to MLX Swift, with an EnCodec decoder.")
        case .aceStep15:
            return .needsConversion(
                note: "Desktop GPU builds only. Needs an MLX conversion and a Swift implementation of the planner and renderer.")
        }
    }

    var isRunnable: Bool { availability == .ready }

    /// Where the app fetches weights from.
    ///
    /// Not Hugging Face directly: MLX reads `safetensors` and the published
    /// files are `npz`, so they are converted first and the conversions are
    /// hosted as a release. Stability's licence and a notice of exactly what
    /// was changed sit alongside them.
    static let releaseBase =
        "https://github.com/yu314-coder/WhisperKit/releases/download/sa3-small-weights-v1"

    func downloadURL(for fileName: String) -> URL? {
        URL(string: "\(Self.releaseBase)/\(fileName)")
    }

    /// Files to fetch from `weightsRepo`, with their published sizes in MB.
    ///
    /// Stable Audio 3 is assembled from four separate graphs rather than one
    /// bundle: a shared text encoder, a diffusion transformer that differs per
    /// variant, and the SAME decoder that turns latents into audio.
    var weightFiles: [(path: String, megabytes: Int)] {
        // Text encoder, decoder and tokenizer are shared by both Stable Audio
        // variants, so switching between them only fetches a different DiT.
        let shared: [(String, Int)] = [
            ("t5gemma_f16.safetensors", 567),
            ("same_s_decoder_f32.safetensors", 218),
            ("t5gemma_tokenizer.model", 4),
        ]
        switch self {
        case .stableAudio3Small:
            return (shared + [("dit_sm-music_f16.safetensors", 919),
                              ("sa3_conditioner_sm-music.safetensors", 1)])
                .map { (path: $0.0, megabytes: $0.1) }
        case .stableAudio3Medium:
            // Two shards: one 2.9 GB file exceeds the 2 GB release-asset cap.
            return (shared + [("dit_medium_f16.part1.safetensors", 1454),
                              ("dit_medium_f16.part2.safetensors", 1454),
                              ("sa3_conditioner_medium.safetensors", 1)])
                .map { (path: $0.0, megabytes: $0.1) }
        case .magentaRealtime2, .aceStep15, .musicGenSmall:
            return []
        }
    }

    var downloadMegabytes: Int { weightFiles.reduce(0) { $0 + $1.megabytes } }

    /// Download size for the picker. Unconverted models have no honest number
    /// to show, because nothing has been built to measure.
    var sizeLabel: String {
        let mb = downloadMegabytes
        guard mb > 0 else { return "—" }
        return mb >= 1000 ? String(format: "%.1f GB", Double(mb) / 1000) : "\(mb) MB"
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
