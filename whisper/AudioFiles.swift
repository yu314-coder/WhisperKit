import Foundation

/// Manages the persistent location of recorded audio files associated with
/// saved transcripts. Files live in <Documents>/SavedAudio/ and we always
/// store/lookup by *relative* path so the sandbox container ID (which can
/// change between installs / restores) never breaks playback.
enum AudioFiles {
    static let folderName = "SavedAudio"

    static var folderURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent(folderName, isDirectory: true)
    }

    static func ensureFolderExists() {
        try? FileManager.default.createDirectory(at: folderURL, withIntermediateDirectories: true)
    }

    /// Copies a source audio file into the saved-audio folder and returns
    /// the *relative* path stored on the SavedTranscript model.
    static func saveAudio(from sourceURL: URL, suggestedName: String) throws -> String {
        ensureFolderExists()
        let ext = sourceURL.pathExtension.isEmpty ? "m4a" : sourceURL.pathExtension
        let base = suggestedName.replacingOccurrences(of: "/", with: "-")
        var candidate = "\(base).\(ext)"
        var counter = 1
        while FileManager.default.fileExists(atPath: folderURL.appendingPathComponent(candidate).path) {
            candidate = "\(base)_\(counter).\(ext)"
            counter += 1
        }
        let dest = folderURL.appendingPathComponent(candidate)
        try FileManager.default.copyItem(at: sourceURL, to: dest)
        return "\(folderName)/\(candidate)"
    }

    static func urlForRelativePath(_ relative: String) -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent(relative)
    }

    static func deleteAudio(relativePath: String) {
        let url = urlForRelativePath(relativePath)
        try? FileManager.default.removeItem(at: url)
    }
}
