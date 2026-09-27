import Foundation

/// The Music tab's models: ACE-Step 1.5, in the versions that run here.
///
/// Every version shares the text encoder, condition encoder, planner, hints
/// and audio decoder, and all of them share one folder, so moving between
/// versions downloads only what differs: the diffusion transformer, and for
/// XL the few condition-encoder tensors it retrained.
enum MusicModel: String, CaseIterable, Identifiable {
    /// The raw value predates the versions; kept so a stored choice still
    /// resolves.
    case aceStep15   = "ace-step-1.5-2b"
    case aceStep15XL = "ace-step-1.5-xl-4b"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .aceStep15:   return "ACE-Step 1.5"
        case .aceStep15XL: return "ACE-Step 1.5 XL"
        }
    }

    /// The diffusion transformer's size, as published.
    var parameterLabel: String {
        switch self {
        case .aceStep15:   return "2B"
        case .aceStep15XL: return "4B"
        }
    }

    var tagline: String {
        switch self {
        case .aceStep15:   return "Recommended · runs on the Neural Engine"
        case .aceStep15XL: return "Richer sound · slower, needs 8 GB"
        }
    }

    /// Where the diffusion transformer runs.
    var engineLabel: String {
        switch self {
        case .aceStep15:   return "NEURAL ENGINE"
        case .aceStep15XL: return "GPU"
        }
    }

    /// Whether this model can run on this device, and if not, why.
    enum Availability: Equatable {
        case ready
        case needsMoreMemory(note: String)
    }

    /// XL maps 4.7 GB of transformer weights. They are file-backed and not
    /// charged to the app, but a 6 GB device has too little room for the
    /// pages the GPU keeps in use.
    static let xlMinimumMemoryGB = 7.5

    var availability: Availability {
        switch self {
        case .aceStep15:
            return .ready
        case .aceStep15XL:
            let gigabytes = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
            return gigabytes >= Self.xlMinimumMemoryGB
                ? .ready
                : .needsMoreMemory(note: String(format: "XL needs a device with 8 GB of memory; this one has %.0f GB. ACE-Step 1.5 runs here.", gigabytes))
        }
    }

    var isRunnable: Bool { availability == .ready }

    var supportsLyrics: Bool { true }

    // MARK: - Generator settings

    /// Points the generator at this version's files.
    func configure(_ generator: inout ACEGenerator) {
        switch self {
        case .aceStep15:
            generator.transformerEngine = .neuralEngine
        case .aceStep15XL:
            generator.transformerEngine = .gpu
            generator.transformerShape = .xl
            generator.transformerFiles = Self.parts("ace_xl_decoder_q8", 3)
            generator.conditionerFiles = [ACEGenerator.File.conditioner, "ace_xl_cond_f16.safetensors"]
            // Float16 would be 8.1 GB, more than an 8 GB device holds; the
            // XL transformer stays int8 (4.7 GB). Everything else is full
            // precision in both versions.
            // The 4B planner matched prompts more closely (CLAP 0.501 against
            // 0.438) but its songs broke apart more: 17 abrupt changes in 196
            // window pairs against 6. The 1.7B planner stays.
        }
    }

    private static func parts(_ stem: String, _ count: Int) -> [String] {
        (1 ... count).map { "\(stem).part\($0).safetensors" }
    }

    // MARK: - Weights

    private static let releaseRoot =
        "https://github.com/yu314-coder/WhisperKit/releases/download"

    /// All versions share one folder, so shared files are fetched once.
    var weightsDirectory: URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documents.appendingPathComponent("MusicModels/ace-step", isDirectory: true)
    }

    struct WeightFile {
        /// Relative to `weightsDirectory`; may name a subfolder.
        let path: String
        /// Exact, read from the hosted asset: it doubles as the "already
        /// downloaded" test.
        let bytes: Int64
        let release: String
    }

    /// The files every version reads: from release acestep-v2, and the
    /// full-precision planner, condition encoder and hints from acestep-v4.
    private static let shared: [WeightFile] = ([
        (ACEGenerator.File.textEncoder, 1_191_586_112),
        (ACEGenerator.File.decoder, 168_807_360),
        (ACEGenerator.File.silence, 1_920_128),
        (ACEGenerator.File.vocabulary, 2_776_833),
        (ACEGenerator.File.merges, 1_671_853),
    ] as [(String, Int64)]).map { WeightFile(path: $0.0, bytes: $0.1, release: "acestep-v2") }
        + ([
            (ACEGenerator.File.conditioner, 1_216_753_088),
            (ACEGenerator.File.hints, 211_591_744),
            (ACEGenerator.File.planner[0], 1_896_403_904),
            (ACEGenerator.File.planner[1], 1_812_117_376),
        ] as [(String, Int64)]).map { WeightFile(path: $0.0, bytes: $0.1, release: "acestep-v4") }

    var weightFiles: [WeightFile] {
        switch self {
        case .aceStep15:   return Self.shared + Self.neuralEngineFiles
        case .aceStep15XL: return Self.shared + Self.xlFiles
        }
    }

    static let neuralEngineFiles: [WeightFile] = ([
        ("ace_dit_ane_c0.mlmodelc/model.mil", 8_262_897),
        ("ace_dit_ane_c0.mlmodelc/coremldata.bin", 8_186),
        ("ace_dit_ane_c0.mlmodelc/metadata.json", 122_343),
        ("ace_dit_ane_c0.mlmodelc/analytics/coremldata.bin", 243),
        ("ace_dit_ane_c0.mlmodelc/weights/weight.bin", 755_752_384),
        ("ace_dit_ane_c1.mlmodelc/model.mil", 8_266_317),
        ("ace_dit_ane_c1.mlmodelc/coremldata.bin", 8_186),
        ("ace_dit_ane_c1.mlmodelc/metadata.json", 122_343),
        ("ace_dit_ane_c1.mlmodelc/analytics/coremldata.bin", 243),
        ("ace_dit_ane_c1.mlmodelc/weights/weight.bin", 755_752_384),
        ("ace_dit_ane_c2.mlmodelc/model.mil", 8_273_157),
        ("ace_dit_ane_c2.mlmodelc/coremldata.bin", 8_186),
        ("ace_dit_ane_c2.mlmodelc/metadata.json", 122_343),
        ("ace_dit_ane_c2.mlmodelc/analytics/coremldata.bin", 243),
        ("ace_dit_ane_c2.mlmodelc/weights/weight.bin", 755_752_384),
        ("ace_dit_ane_c3.mlmodelc/model.mil", 8_273_157),
        ("ace_dit_ane_c3.mlmodelc/coremldata.bin", 8_186),
        ("ace_dit_ane_c3.mlmodelc/metadata.json", 122_343),
        ("ace_dit_ane_c3.mlmodelc/analytics/coremldata.bin", 243),
        ("ace_dit_ane_c3.mlmodelc/weights/weight.bin", 755_752_384),
        ("ace_dit_outer_f16.safetensors", 130_111_616),
    ] as [(String, Int64)]).map { WeightFile(path: $0.0, bytes: $0.1, release: "acestep-v3") }

    static let xlFiles: [WeightFile] = ([
        ("ace_xl_decoder_q8.part1.safetensors", 1_895_498_560),
        ("ace_xl_decoder_q8.part2.safetensors", 1_888_113_216),
        ("ace_xl_decoder_q8.part3.safetensors", 908_410_816),
    ] as [(String, Int64)]).map { WeightFile(path: $0.0, bytes: $0.1, release: "acestep-v3") }
        + [WeightFile(path: "ace_xl_cond_f16.safetensors", bytes: 8_672_128, release: "acestep-v4")]

    /// Release assets are flat, so a file in a subfolder is published with
    /// its slashes as "__".
    func downloadURL(for file: WeightFile) -> URL? {
        let asset = file.path.replacingOccurrences(of: "/", with: "__")
        #if DEBUG
        // For testing interrupted downloads against a local server:
        // launch with `-MusicWeightsRootOverride http://127.0.0.1:8765`.
        if let root = UserDefaults.standard.string(forKey: "MusicWeightsRootOverride") {
            return URL(string: "\(root)/\(file.release)/\(asset)")
        }
        #endif
        return URL(string: "\(Self.releaseRoot)/\(file.release)/\(asset)")
    }

    var downloadBytes: Int64 { weightFiles.reduce(0) { $0 + $1.bytes } }

    var sizeLabel: String {
        let gigabytes = Double(downloadBytes) / 1_000_000_000
        return String(format: "%.1f GB", gigabytes)
    }

    /// Top-level names under `weightsDirectory` that any version uses; the
    /// rest are left over from earlier builds.
    static var expectedTopLevelNames: Set<String> {
        Set(allCases.flatMap { $0.weightFiles.map { String($0.path.split(separator: "/")[0]) } })
    }

    /// Stable Audio 3, which earlier builds offered, left 1.7 to 3.7 GB here.
    static func removeRetiredWeights() {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let retired = documents.appendingPathComponent("MusicModels/stable-audio-3", isDirectory: true)
        if FileManager.default.fileExists(atPath: retired.path) {
            try? FileManager.default.removeItem(at: retired)
        }
    }
}
