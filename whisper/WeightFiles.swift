import Foundation

/// A file a downloadable model reads, as published on this repository's
/// GitHub releases. Shared by the Music and Image tabs' models.
struct WeightFile {
    /// Relative to the model's weights directory.
    let path: String
    /// Exact, read from the hosted asset: it doubles as the "already
    /// downloaded" test.
    let bytes: Int64
    let release: String
    /// For an Apple Archive: the files it unpacks to, relative to the
    /// weights directory, with their sizes. The archive is deleted once
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

    /// Top-level names under the weights directory that `files` occupy.
    static func topLevelNames(_ files: [WeightFile]) -> Set<String> {
        Set(files.flatMap { file in ([file.path] + file.contents.map(\.path)) }
            .map { String($0.split(separator: "/")[0]) })
    }

    static let releaseRoot = "https://github.com/yu314-coder/WhisperKit/releases/download"

    var downloadURL: URL? {
        #if DEBUG
        // For testing interrupted downloads, or weights not yet published,
        // against a local server: launch with
        // `-MusicWeightsRootOverride http://127.0.0.1:8765`.
        if let root = UserDefaults.standard.string(forKey: "MusicWeightsRootOverride") {
            return URL(string: "\(root)/\(release)/\(path)")
        }
        #endif
        return URL(string: "\(Self.releaseRoot)/\(release)/\(path)")
    }
}

/// Brings a model's weights onto the device: what is missing is downloaded,
/// archives are unpacked, and files no model uses any more are removed.
@MainActor
enum WeightFetcher {
    struct Progress {
        let file: String
        /// Files finished, of `total` missing ones.
        let completed: Int
        let total: Int
        let fraction: Double
        let received: Int64
        let expected: Int64
    }

    /// - Parameters:
    ///   - expected: top-level names in `directory` that any model sharing
    ///     it uses; anything else there is left from an earlier build and
    ///     deleted. Nil keeps everything.
    ///   - progress: called on the main actor as bytes arrive.
    ///   - note: a line for the user while a dropped transfer is recovered
    ///     or an archive unpacked; nil when there is nothing to say.
    static func fetch(_ files: [WeightFile], into directory: URL, expected: Set<String>?,
                      progress: @escaping (Progress) -> Void,
                      note: @escaping (String?) -> Void) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let expected { removeUnused(in: directory, keeping: expected) }

        // Only what is missing is counted and fetched: a file already in
        // place from another version is not downloaded again, and the
        // progress shown is for this download alone, by bytes across all of
        // it rather than file by file.
        let missing = files.filter { !$0.isPresent(in: directory) }
        guard !missing.isEmpty else { return }
        let totalBytes = missing.reduce(Int64(0)) { $0 + $1.bytes }

        // Checked up front: a download that fills the disk does not fail, it
        // stalls, with nothing on screen to say why. An archive and what it
        // unpacks to are both on disk for a moment.
        let needed = missing.reduce(Int64(0)) { $0 + $1.bytes + ($1.isArchive ? $1.installedBytes : 0) }
            + 300_000_000
        if let available = (try? directory.resourceValues(
                forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?.volumeAvailableCapacityForImportantUsage,
           available < needed {
            throw MusicDownloaderError.insufficientSpace(needed: needed, available: available)
        }
        defer { note(nil) }

        var done: Int64 = 0
        for (index, file) in missing.enumerated() {
            try Task.checkCancellation()
            guard let remote = file.downloadURL else { throw WeightFetchError.badURL(file.path) }
            let before = done
            let show = { (received: Int64) in
                progress(Progress(file: file.path, completed: index, total: missing.count,
                                  fraction: Double(before + received) / Double(max(totalBytes, 1)),
                                  received: before + received, expected: totalBytes))
            }
            show(0)
            let destination = directory.appendingPathComponent(file.path)
            // Progress hops to the main actor in separate tasks, which are not
            // guaranteed to run in order; a late one for the previous file
            // could otherwise put its bar back on screen after the next file
            // had started.
            let current = CurrentFile(file.path)
            defer { current.end() }
            // Interruptions are resumed inside the service; what reaches here
            // has failed repeatedly or been cancelled.
            try await WeightDownloadService.shared.download(
                from: remote, to: destination,
                progress: { update in
                    Task { @MainActor in if current.isCurrent { show(update.received) } }
                },
                status: { line in
                    Task { @MainActor in if current.isCurrent { note(line) } }
                })

            let written = (try? destination.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            guard Int64(written) == file.bytes else {
                throw WeightFetchError.wrongSize(file.path, Int64(written), file.bytes)
            }
            if file.isArchive {
                note("Unpacking")
                try await Task.detached(priority: .userInitiated) {
                    try WeightArchive.unpack(destination, into: directory)
                }.value
                try? FileManager.default.removeItem(at: destination)
                note(nil)
                guard file.isPresent(in: directory) else { throw WeightFetchError.unpackFailed(file.path) }
            }
            done += file.bytes
        }
    }

    /// Deletes what an earlier build downloaded that no model uses now.
    ///
    /// ACE-Step's first layout was two 1.3 GB shards and a trimmed silence
    /// file; its replacement splits the checkpoint by stage instead. Without
    /// this, updating the app would leave 2.7 GB behind in Documents that
    /// nothing reads and nothing in the app can remove.
    private static func removeUnused(in directory: URL, keeping expected: Set<String>) {
        // Compared by top-level name: a Neural Engine model is a folder, and
        // its files' paths start with the folder's name.
        guard !expected.isEmpty,
              let present = try? FileManager.default.contentsOfDirectory(atPath: directory.path)
        else { return }
        for name in present where !expected.contains(name) && !name.hasPrefix(".") {
            try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        }
    }

    /// Whether progress for a file may still be shown: false once its
    /// download has returned.
    private final class CurrentFile: @unchecked Sendable {
        let path: String
        private let lock = NSLock()
        private var ended = false
        init(_ path: String) { self.path = path }
        var isCurrent: Bool { lock.withLock { !ended } }
        func end() { lock.withLock { ended = true } }
    }
}

enum WeightFetchError: LocalizedError {
    case badURL(String)
    case wrongSize(String, Int64, Int64)
    case unpackFailed(String)

    var errorDescription: String? {
        switch self {
        case .badURL(let name):     return "No download address for \(name)."
        case .unpackFailed(let name):
            return "\(name) downloaded but did not unpack completely. Try again to fetch it anew."
        case .wrongSize(let name, let got, let want):
            return "\(name) downloaded \(got) bytes, expected \(want). The file is incomplete."
        }
    }
}
