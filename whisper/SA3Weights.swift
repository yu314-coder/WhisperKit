import Foundation

/// Resolves Stable Audio 3 weight files from the downloaded-models directory.
///
/// The reference implementation reads these out of the app bundle, which would
/// mean shipping 1.7 GB inside the app. They are downloaded instead, so this
/// stands in for `Bundle.main` — deliberately keeping the same argument labels
/// so the adapted model code reads unchanged.
enum SA3Weights {
    static var directory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("MusicModels/stable-audio-3", isDirectory: true)
    }

    static func url(forResource name: String, withExtension ext: String, subdirectory: String? = nil) -> URL? {
        let url = directory.appendingPathComponent("\(name).\(ext)")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Every file the pipeline needs before it can run, and how big each is.
    /// Hosted as a GitHub release rather than bundled or fetched from Hugging
    /// Face directly: MLX can only read `safetensors`, and the published
    /// weights are `npz`, so they have to be converted first.
    static let requiredFiles: [(name: String, bytes: Int64)] = [
        ("t5gemma_f16.safetensors", 567_416_533),
        ("dit_sm-music_f16.safetensors", 919_104_895),
        ("same_s_decoder_f32.safetensors", 218_069_578),
        ("sa3_conditioner_sm-music.safetensors", 792_862),
        ("t5gemma_tokenizer.model", 4_241_003),
    ]

    static var isComplete: Bool {
        requiredFiles.allSatisfy { file in
            let url = directory.appendingPathComponent(file.name)
            guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return false }
            return Int64(size) >= file.bytes
        }
    }

    static var totalBytes: Int64 { requiredFiles.reduce(0) { $0 + $1.bytes } }
}

enum WeightTensorLoaderError: LocalizedError {
    case missing(String)
    var errorDescription: String? {
        switch self {
        case .missing(let fileName): return "\(fileName) is missing — download the model first."
        }
    }
}
