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
            return .weightsPublished(
                note: "Uses the same pipeline as Small-Music with a larger DiT. Untested on iOS.")
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

    /// Hugging Face repository the weights come from, where one is published.
    var weightsRepo: String? {
        switch self {
        case .stableAudio3Small, .stableAudio3Medium:
            return "stabilityai/stable-audio-3-optimized"
        case .magentaRealtime2:
            return "mattmireles/magenta-realtime-2-iphone"
        case .aceStep15, .musicGenSmall:
            return nil
        }
    }

    /// Files to fetch from `weightsRepo`, with their published sizes in MB.
    ///
    /// Stable Audio 3 is assembled from four separate graphs rather than one
    /// bundle: a shared text encoder, a diffusion transformer that differs per
    /// variant, and the SAME decoder that turns latents into audio.
    var weightFiles: [(path: String, megabytes: Int)] {
        switch self {
        case .stableAudio3Small:
            return [("MLX/t5gemma_f16.npz", 567),
                    ("MLX/dit_sm-music_f16.npz", 919),
                    ("MLX/same_s_decoder_f32.npz", 218)]
        case .stableAudio3Medium:
            return [("MLX/t5gemma_f16.npz", 567),
                    ("MLX/dit_medium_f16.npz", 2910),
                    ("MLX/same_s_decoder_f32.npz", 218)]
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
