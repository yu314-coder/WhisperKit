import Foundation

/// Resolves Stable Audio 3 weight files from the downloaded-models directory.
///
/// The reference implementation reads these out of the app bundle, which would
/// mean shipping 1.7 GB inside the app. They are downloaded instead, so this
/// stands in for `Bundle.main` — deliberately keeping the same argument labels
/// so the adapted model code reads unchanged.
enum SA3Weights {
    /// Points the pipeline elsewhere, for running it outside the app.
    nonisolated(unsafe) static var directoryOverride: URL?

    static var directory: URL {
        if let directoryOverride { return directoryOverride }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("MusicModels/stable-audio-3", isDirectory: true)
    }

    static func url(forResource name: String, withExtension ext: String, subdirectory: String? = nil) -> URL? {
        let url = directory.appendingPathComponent("\(name).\(ext)")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}

enum WeightTensorLoaderError: LocalizedError {
    case missing(String)
    var errorDescription: String? {
        switch self {
        case .missing(let fileName): return "\(fileName) is missing — download the model first."
        }
    }
}
