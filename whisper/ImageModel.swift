import Foundation

/// The Image tab's models, each running entirely on the device — listed
/// best pictures first, as the model menu shows them.
enum ImageModel: String, CaseIterable, Identifiable {
    case qwenImage   = "qwen-image-2.1-turbo"
    case zImage      = "z-image-turbo"
    case fluxKlein   = "flux2-klein-4b"
    case sanaSprintLarge = "sana-sprint-1.6b"
    case sanaSprint  = "sana-sprint-0.6b"
    case dreamShaper = "dreamshaper-8-hyper"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .dreamShaper: return "DreamShaper"
        case .sanaSprint:  return "SANA-Sprint"
        case .sanaSprintLarge: return "SANA-Sprint 1.6B"
        case .fluxKlein:   return "FLUX.2 klein"
        case .zImage:      return "Z-Image Turbo"
        case .qwenImage:   return "Qwen-Image 2.1 Turbo"
        }
    }

    var parameterLabel: String {
        switch self {
        case .dreamShaper: return "1B"
        case .sanaSprint:  return "0.6B"
        case .sanaSprintLarge: return "1.6B"
        case .fluxKlein:   return "4B"
        case .zImage:      return "6B"
        case .qwenImage:   return "7B"
        }
    }

    var tagline: String {
        switch self {
        case .dreamShaper: return "Fastest · 512 × 512 · Stable Diffusion 1.5, four steps"
        case .sanaSprint:  return "Sharp 1024 × 1024 in two steps"
        case .sanaSprintLarge: return "Better pictures than the 0.6B, still in seconds"
        case .fluxKlein:   return "Follows long prompts closely · several shapes"
        case .zImage:      return "Most lifelike · nothing steers your prompt away"
        case .qwenImage:   return "Finest detail and lettering · slowest · nothing steers your prompt away"
        }
    }

    var engineLabel: String {
        switch self {
        case .dreamShaper:           return "NEURAL ENGINE"
        case .sanaSprint, .sanaSprintLarge, .fluxKlein, .zImage, .qwenImage: return "GPU"
        }
    }

    /// Whether a run needs the GPU, and so background GPU time to carry on
    /// off screen.
    var usesGPU: Bool { self != .dreamShaper }

    /// The upstream licences, for the help page and the release notes.
    var licence: String {
        switch self {
        case .dreamShaper: return "DreamShaper 8 and Stable Diffusion 1.5: CreativeML OpenRAIL-M. Hyper-SD: CreativeML OpenRAIL++-M."
        case .sanaSprint, .sanaSprintLarge:
            return "SANA-Sprint: Apache 2.0. Its text encoder, Gemma 2: Gemma Terms of Use."
        case .fluxKlein:   return "FLUX.2 [klein] 4B and Qwen3: Apache 2.0."
        case .zImage:      return "Z-Image-Turbo (Tongyi-MAI) and its abliterated Qwen3 prompt reader: Apache 2.0."
        case .qwenImage:
            return "Qwen-Image 2.1 Turbo: Qwen RESEARCH LICENSE AGREEMENT — Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved. Its abliterated Qwen3-VL prompt reader: Apache 2.0."
        }
    }

    // MARK: - Weights

    /// SANA-Sprint 1.6B shares the 0.6B's folder: the two read the same
    /// Gemma 2, decoder and tokenizer, byte for byte, so whichever comes
    /// second downloads only its own transformer.
    var weightsDirectory: URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let folder = self == .sanaSprintLarge ? Self.sanaSprint.rawValue : rawValue
        return documents.appendingPathComponent("ImageModels/\(folder)", isDirectory: true)
    }

    /// Every file that any model keeps in this model's folder: what a
    /// download there must leave alone.
    var folderFiles: [WeightFile] {
        Self.allCases.filter { $0.weightsDirectory == weightsDirectory }.flatMap(\.weightFiles)
    }

    private static let release = "imagegen-v1"

    var weightFiles: [WeightFile] {
        switch self {
        case .dreamShaper: return Self.dreamShaperFiles
        case .sanaSprint:  return Self.sanaFiles
        case .sanaSprintLarge: return Self.sanaLargeFiles
        case .fluxKlein:   return Self.fluxFiles
        case .zImage:      return Self.zImageFiles
        case .qwenImage:   return Self.qwenImageFiles
        }
    }

    /// SD 1.5's three Core ML programs, one Apple Archive each, and CLIP's
    /// vocabulary. The UNet is palettized to 8 bits: 6 bits drew a
    /// different picture from the same noise (a shirt where the float16
    /// model drew a suit); 8 bits match it as closely as float16 matches
    /// PyTorch.
    static let dreamShaperFiles: [WeightFile] = [
        archive("sd_text_encoder.mlmodelc", 122973950, [
            ("analytics/coremldata.bin", 243),
            ("coremldata.bin", 760),
            ("metadata.json", 2743),
            ("model.mil", 178683),
            ("weights/weight.bin", 123243648),
        ]),
        archive("sd_unet.mlmodelc", 834376219, [
            ("analytics/coremldata.bin", 243),
            ("coremldata.bin", 1181),
            ("metadata.json", 3724),
            ("model.mil", 3137366),
            ("weights/weight.bin", 860203008),
        ]),
        archive("sd_vae_decoder.mlmodelc", 91978914, [
            ("analytics/coremldata.bin", 243),
            ("coremldata.bin", 685),
            ("metadata.json", 2438),
            ("model.mil", 194902),
            ("weights/weight.bin", 98993280),
        ]),
        WeightFile(path: "clip_vocab.json", bytes: 1059962, release: release),
        WeightFile(path: "clip_merges.txt", bytes: 524619, release: release),
    ]
    /// SANA-Sprint: Gemma 2 in two parts (8-bit), the caption projection
    /// and fixed-schedule modulation, the decoder's first half for MLX, the
    /// transformer (8-bit palettized) and the decoder's second half as Core
    /// ML programs, and the tokenizer's tables.
    static let sanaFiles: [WeightFile] = sanaShared + [
        WeightFile(path: "sana_extras.safetensors", bytes: 8034688, release: release),
        archive("sana_transformer.mlmodelc", 578926700, [
            ("analytics/coremldata.bin", 243),
            ("coremldata.bin", 615),
            ("metadata.json", 3699),
            ("model.mil", 1031631),
            ("weights/weight.bin", 579330688),
        ]),
    ]
    /// SANA-Sprint 1.6B: its own transformer and caption projection; the
    /// rest is the 0.6B's.
    static let sanaLargeFiles: [WeightFile] = sanaShared + [
        WeightFile(path: "sana16_extras.safetensors", bytes: 20496704, release: release),
        archive("sana16_transformer.mlmodelc", 1550705807, [
            ("analytics/coremldata.bin", 243),
            ("coremldata.bin", 619),
            ("metadata.json", 3788),
            ("model.mil", 857470),
            ("weights/weight.bin", 1560031104),
        ]),
    ]
    private static let sanaShared: [WeightFile] = [
        WeightFile(path: "gemma_bpe.bin", bytes: 9462292, release: release),
        WeightFile(path: "sana_gemma_q8.part1.safetensors", bytes: 1890696320, release: release),
        WeightFile(path: "sana_gemma_q8.part2.safetensors", bytes: 887825856, release: release),
        WeightFile(path: "sana_decoder_head.safetensors", bytes: 273609920, release: release),
        archive("sana_decoder_tail.mlmodelc", 41698134, [
            ("analytics/coremldata.bin", 243),
            ("coremldata.bin", 510),
            ("metadata.json", 2307),
            ("model.mil", 81918),
            ("weights/weight.bin", 44859456),
        ]),
    ]
    /// FLUX.2 [klein]: Qwen3's first 27 layers and the transformer, 8-bit,
    /// in parts under 2 GB; the VAE decoder; Qwen's tokenizer.
    static let fluxFiles: [WeightFile] = [
        WeightFile(path: "flux_vocab.json", bytes: 2776833, release: release),
        WeightFile(path: "flux_merges.txt", bytes: 1671853, release: release),
        WeightFile(path: "flux_text_q8.part1.safetensors", bytes: 1889803264, release: release),
        WeightFile(path: "flux_text_q8.part2.safetensors", bytes: 1419417600, release: release),
        WeightFile(path: "flux_transformer_q8.part1.safetensors", bytes: 1890531904, release: release),
        WeightFile(path: "flux_transformer_q8.part2.safetensors", bytes: 1875920128, release: release),
        WeightFile(path: "flux_transformer_q8.part3.safetensors", bytes: 351370048, release: release),
        WeightFile(path: "flux_vae.safetensors", bytes: 99257094, release: release),
    ]

    /// Z-Image-Turbo: the abliterated Qwen3 (35 layers) and the transformer,
    /// 8-bit, in parts under 2 GB; the VAE decoder; Qwen's tokenizer.
    static let zImageFiles: [WeightFile] = [
        WeightFile(path: "zimage_vocab.json", bytes: 2776833, release: release),
        WeightFile(path: "zimage_merges.txt", bytes: 1671853, release: release),
        WeightFile(path: "zimage_text_q8.part1.safetensors", bytes: 1889803264, release: release),
        WeightFile(path: "zimage_text_q8.part2.safetensors", bytes: 1874918912, release: release),
        WeightFile(path: "zimage_text_q8.part3.safetensors", bytes: 402559296, release: release),
        WeightFile(path: "zimage_transformer_q8.part1.safetensors", bytes: 1884330736, release: release),
        WeightFile(path: "zimage_transformer_q8.part2.safetensors", bytes: 1880718080, release: release),
        WeightFile(path: "zimage_transformer_q8.part3.safetensors", bytes: 1880718016, release: release),
        WeightFile(path: "zimage_transformer_q8.part4.safetensors", bytes: 894918128, release: release),
        WeightFile(path: "zimage_vae.safetensors", bytes: 99106182, release: release),
    ]

    /// Qwen-Image 2.1 Turbo: the abliterated Qwen3-VL's language model and
    /// the transformer, 8-bit, in parts under 2 GB; the VAE decoder; Qwen's
    /// tokenizer; and the licence and notice its terms ask to go with it.
    static let qwenImageFiles: [WeightFile] = [
        WeightFile(path: "qwenimage_vocab.json", bytes: 3080118, release: release),
        WeightFile(path: "qwenimage_merges.txt", bytes: 1671853, release: release),
        WeightFile(path: "qwenimage_text_q8.part1.safetensors", bytes: 1891441856, release: release),
        WeightFile(path: "qwenimage_text_q8.part2.safetensors", bytes: 1898776896, release: release),
        WeightFile(path: "qwenimage_text_q8.part3.safetensors", bytes: 1898776896, release: release),
        WeightFile(path: "qwenimage_text_q8.part4.safetensors", bytes: 1898793920, release: release),
        WeightFile(path: "qwenimage_text_q8.part5.safetensors", bytes: 454633024, release: release),
        WeightFile(path: "qwenimage_transformer_q8.part1.safetensors", bytes: 1856072832, release: release),
        WeightFile(path: "qwenimage_transformer_q8.part2.safetensors", bytes: 1853909312, release: release),
        WeightFile(path: "qwenimage_transformer_q8.part3.safetensors", bytes: 1853909312, release: release),
        WeightFile(path: "qwenimage_transformer_q8.part4.safetensors", bytes: 1853909184, release: release),
        WeightFile(path: "qwenimage_transformer_q8.part5.safetensors", bytes: 142616320, release: release),
        WeightFile(path: "qwenimage_vae.safetensors", bytes: 506155944, release: release),
        WeightFile(path: "qwenimage_LICENSE.txt", bytes: 7831, release: release),
        WeightFile(path: "qwenimage_NOTICE.txt", bytes: 1602, release: release),
    ]

    /// The filter every picture passes before it's shown (SafetyChecker),
    /// shared by all three models and downloaded with the first.
    static let safetyFiles: [WeightFile] = [
        archive("safety_checker.mlmodelc", 85588608, [
            ("analytics/coremldata.bin", 243),
            ("coremldata.bin", 524),
            ("metadata.json", 2259),
            ("model.mil", 150859),
            ("weights/weight.bin", 86130752),
        ]),
    ]

    static var safetyDirectory: URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documents.appendingPathComponent("ImageModels/picture-check", isDirectory: true)
    }

    private static func archive(_ folder: String, _ bytes: Int64, _ contents: [(String, Int64)]) -> WeightFile {
        WeightFile(path: "\(folder).aar", bytes: bytes, release: release,
                   contents: contents.map { (path: "\(folder)/\($0.0)", bytes: $0.1) })
    }

    var downloadBytes: Int64 { weightFiles.reduce(0) { $0 + $1.bytes } }

    var sizeLabel: String {
        let bytes = Double(downloadBytes)
        return bytes >= 1e9 ? String(format: "%.1f GB", bytes / 1e9) : String(format: "%.0f MB", bytes / 1e6)
    }

    /// A caution for this device, or nil — never a block.
    var memoryWarning: String? {
        let gigabytes = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        switch self {
        case .dreamShaper, .sanaSprint, .sanaSprintLarge:
            return nil
        case .fluxKlein, .zImage, .qwenImage:
            // Qwen-Image's transformer alone is 7.5 GB.
            guard gigabytes < (self == .qwenImage ? 11 : 7) else { return nil }
            return String(format: "%@ runs here. With %.0f GB on this device its weights are read from storage as it goes, so pictures take longer.", displayName, gigabytes.rounded())
        }
    }
}
