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
    case aceStep15       = "ace-step-1.5-2b"
    case aceStep15XL     = "ace-step-1.5-xl-4b"
    case aceStep15XLFull = "ace-step-1.5-xl-4b-f16"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .aceStep15:       return "ACE-Step 1.5"
        case .aceStep15XL:     return "ACE-Step 1.5 XL"
        case .aceStep15XLFull: return "ACE-Step 1.5 XL Full"
        }
    }

    /// The diffusion transformer's size, as published.
    var parameterLabel: String {
        switch self {
        case .aceStep15:                     return "2B"
        case .aceStep15XL, .aceStep15XLFull: return "4B"
        }
    }

    var tagline: String {
        switch self {
        case .aceStep15:       return "Recommended · runs on the Neural Engine"
        case .aceStep15XL:     return "Richer sound · 4.7 GB, held in memory"
        case .aceStep15XLFull: return "Full precision · 8.1 GB, best with 12 GB or more"
        }
    }

    /// Where the diffusion transformer runs.
    var engineLabel: String {
        switch self {
        case .aceStep15:                     return "NEURAL ENGINE"
        case .aceStep15XL, .aceStep15XLFull: return "GPU"
        }
    }

    /// A caution for this device, or nil. Never a block: an 8 GB iPad Air
    /// M3 reports under 8 GB to apps, and an earlier limit of 7.5 GB shut it
    /// out of XL. The person decides; the app says what to expect.
    var memoryWarning: String? {
        let gigabytes = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        switch self {
        case .aceStep15:
            return nil
        case .aceStep15XL:
            guard gigabytes < 7 else { return nil }
            return String(format: "XL's 4.7 GB transformer is too large to hold in this device's %.0f GB, so it is read from storage as it runs: slower. ACE-Step 1.5 is the faster choice here.", gigabytes.rounded())
        case .aceStep15XLFull:
            guard gigabytes < 11 else { return nil }
            return String(format: "XL Full's transformer is 8.1 GB, more than this device's %.0f GB, so it is read from storage at every step: much slower than XL, which holds its 4.7 GB in memory and measured nearly the same (velocity cosine 0.99878 against 0.99918, both against the official model).", gigabytes.rounded())
        }
    }

    var supportsLyrics: Bool { true }

    // MARK: - Generator settings

    /// Points the generator at this version's files.
    func configure(_ generator: inout ACEGenerator) {
        switch self {
        case .aceStep15:
            generator.transformerEngine = .neuralEngine
        case .aceStep15XL, .aceStep15XLFull:
            generator.transformerEngine = .gpu
            generator.transformerShape = .xl
            generator.conditionerFiles = [ACEGenerator.File.conditioner, "ace_xl_cond_f16.safetensors"]
            if self == .aceStep15XLFull {
                // Float16: against the official XL on the same inputs,
                // velocity cosine 0.99918. At 8.1 GB it is more than an 8 GB
                // device holds — on an iPad Air M3 it is read from storage at
                // every step, 165 s for 30 seconds of music.
                generator.transformerFiles = Self.parts("ace_xl_decoder_f16", 5)
                generator.transformerDType = .float16
            } else {
                // Int8, 0.99878 against the official XL: 4.7 GB, which an
                // 8 GB device holds in memory for the whole run.
                generator.transformerFiles = Self.parts("ace_xl_decoder_q8", 3)
            }
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
        /// Relative to `weightsDirectory`.
        let path: String
        /// Exact, read from the hosted asset: it doubles as the "already
        /// downloaded" test.
        let bytes: Int64
        let release: String
        /// For an Apple Archive: the files it unpacks to, relative to
        /// `weightsDirectory`, with their sizes. The archive is deleted once
        /// unpacked, so these are what show it was downloaded.
        var contents: [(path: String, bytes: Int64)] = []

        var isArchive: Bool { !contents.isEmpty }

        /// Bytes on disk once in place.
        var installedBytes: Int64 { isArchive ? contents.reduce(0) { $0 + $1.bytes } : bytes }

        func isPresent(in directory: URL) -> Bool {
            func size(_ path: String) -> Int64 {
                Int64((try? directory.appendingPathComponent(path)
                    .resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? -1)
            }
            return isArchive ? contents.allSatisfy { size($0.path) == $0.bytes } : size(path) == bytes
        }
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
        case .aceStep15:       return Self.shared + Self.neuralEngineFiles
        case .aceStep15XL:     return Self.shared + Self.xlInt8Files
        case .aceStep15XLFull: return Self.shared + Self.xlFiles
        }
    }

    /// The 2B transformer's four Core ML programs, one archive each — a
    /// compiled model is a folder of five files, and publishing them loose
    /// made a first download 30 files long. Compressed, too: 589 MB each
    /// against 764 MB unpacked.
    static let neuralEngineFiles: [WeightFile] = [
        chunk(0, archive: 588_836_198, program: 8_262_897),
        chunk(1, archive: 588_541_722, program: 8_266_317),
        chunk(2, archive: 588_400_835, program: 8_273_157),
        chunk(3, archive: 588_489_231, program: 8_273_157),
        WeightFile(path: "ace_dit_outer_f16.safetensors", bytes: 130_111_616, release: "acestep-v3"),
    ]

    private static func chunk(_ index: Int, archive: Int64, program: Int64) -> WeightFile {
        let folder = ACENeuralTransformer.chunkDirectory(index)
        return WeightFile(path: "\(folder).aar", bytes: archive, release: "acestep-v3",
                          contents: [("\(folder)/model.mil", program),
                                     ("\(folder)/coremldata.bin", 8_186),
                                     ("\(folder)/metadata.json", 122_343),
                                     ("\(folder)/analytics/coremldata.bin", 243),
                                     ("\(folder)/weights/weight.bin", 755_752_384)])
    }

    static let xlInt8Files: [WeightFile] = ([
        ("ace_xl_decoder_q8.part1.safetensors", 1_895_498_560),
        ("ace_xl_decoder_q8.part2.safetensors", 1_888_113_216),
        ("ace_xl_decoder_q8.part3.safetensors", 908_410_816),
    ] as [(String, Int64)]).map { WeightFile(path: $0.0, bytes: $0.1, release: "acestep-v3") }
        + [WeightFile(path: "ace_xl_cond_f16.safetensors", bytes: 8_672_128, release: "acestep-v4")]

    static let xlFiles: [WeightFile] = ([
        ("ace_xl_decoder_f16.part1.safetensors", 1_893_036_544),
        ("ace_xl_decoder_f16.part2.safetensors", 1_885_197_120),
        ("ace_xl_decoder_f16.part3.safetensors", 1_879_929_856),
        ("ace_xl_decoder_f16.part4.safetensors", 1_879_917_632),
        ("ace_xl_decoder_f16.part5.safetensors", 799_784_704),
        ("ace_xl_cond_f16.safetensors", 8_672_128),
    ] as [(String, Int64)]).map { WeightFile(path: $0.0, bytes: $0.1, release: "acestep-v4") }

    func downloadURL(for file: WeightFile) -> URL? {
        let asset = file.path
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
        Set(allCases.flatMap { model in
            model.weightFiles.flatMap { file in ([file.path] + file.contents.map(\.path)) }
                .map { String($0.split(separator: "/")[0]) }
        })
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
