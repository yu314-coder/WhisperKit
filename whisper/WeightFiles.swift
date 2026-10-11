import Foundation

/// A file a downloadable model reads, as published on this repository's
/// GitHub releases — or, for the Transcribe tab's models, at an address of
/// its own on Hugging Face.
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
    /// Where the file is published, when that isn't `release`.
    var source: URL? = nil

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
        if let source { return source }
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

    /// Files bound for one folder.
    struct Batch {
        let files: [WeightFile]
        let directory: URL
        /// Top-level names in `directory` that any model sharing it uses;
        /// anything else there is left from an earlier build and deleted.
        /// Nil keeps everything.
        var expected: Set<String>?
    }

    /// - Parameters:
    ///   - progress: called on the main actor as bytes arrive.
    ///   - note: a line for the user while a dropped transfer is recovered
    ///     or an archive unpacked; nil when there is nothing to say.
    static func fetch(_ files: [WeightFile], into directory: URL, expected: Set<String>?,
                      progress: @escaping (Progress) -> Void,
                      note: @escaping (String?) -> Void) async throws {
        try await fetch([Batch(files: files, directory: directory, expected: expected)], progress: progress, note: note)
    }

    /// Several folders' files as one download: all of them are in flight at
    /// once and the progress covers them together.
    static func fetch(_ batches: [Batch], progress: @escaping (Progress) -> Void,
                      note: @escaping (String?) -> Void) async throws {
        defer { note(nil) }
        for batch in batches {
            try FileManager.default.createDirectory(at: batch.directory, withIntermediateDirectories: true)
            if let expected = batch.expected { removeUnused(in: batch.directory, keeping: expected) }

            // An archive that finished downloading while the app was closed
            // is already here whole; it only needs unpacking.
            for file in batch.files where file.isArchive && !file.isPresent(in: batch.directory) {
                let archive = batch.directory.appendingPathComponent(file.path)
                guard (try? archive.resourceValues(forKeys: [.fileSizeKey]))?.fileSize == Int(file.bytes) else { continue }
                note("Unpacking")
                try await unpack(file, at: archive, into: batch.directory)
            }
        }

        // Only what is missing is counted and fetched: a file already in
        // place from another version is not downloaded again, and the
        // progress shown is for this download alone, by bytes across all of
        // it rather than file by file.
        let missing = batches.flatMap { batch in
            batch.files.filter { !$0.isPresent(in: batch.directory) }.map { (file: $0, directory: batch.directory) }
        }
        guard !missing.isEmpty else { return }
        let totalBytes = missing.reduce(Int64(0)) { $0 + $1.file.bytes }

        // Checked up front: a download that fills the disk does not fail, it
        // stalls, with nothing on screen to say why. An archive and what it
        // unpacks to are both on disk for a moment.
        let needed = missing.reduce(Int64(0)) { $0 + $1.file.bytes + ($1.file.isArchive ? $1.file.installedBytes : 0) }
            + 300_000_000
        if let available = (try? missing[0].directory.resourceValues(
                forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?.volumeAvailableCapacityForImportantUsage,
           available < needed {
            throw MusicDownloaderError.insufficientSpace(needed: needed, available: available)
        }

        // Every file is handed to the system at once. Its download daemon then
        // works through all of them — a few at a time, as it chooses — while
        // the app is in the background, suspended or even closed. Fetched one
        // after another, only the file in flight kept going: the next one
        // waited for the app to run again.
        let tally = Tally(files: missing.map(\.file), total: totalBytes, progress: progress, note: note)
        tally.show(missing[0].file.path)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for (file, directory) in missing {
                guard let remote = file.downloadURL else { throw WeightFetchError.badURL(file.path) }
                let destination = directory.appendingPathComponent(file.path)
                group.addTask {
                    // Interruptions are resumed inside the service; what
                    // reaches here has failed repeatedly or been cancelled.
                    try await WeightDownloadService.shared.download(
                        from: remote, to: destination,
                        progress: { update in Task { @MainActor in tally.received(update.received, for: file.path) } },
                        status: { line in Task { @MainActor in tally.status(line, for: file.path) } })
                    let written = (try? destination.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                    guard Int64(written) == file.bytes else {
                        throw WeightFetchError.wrongSize(file.path, Int64(written), file.bytes)
                    }
                    if file.isArchive {
                        await tally.status("Unpacking", for: file.path)
                        try await unpack(file, at: destination, into: directory)
                        await tally.status(nil, for: file.path)
                    }
                    await tally.finished(file.path)
                }
            }
            try await group.waitForAll()
        }
    }

    /// Unpacks an archive off the main thread, deletes it, and checks that
    /// everything it holds arrived.
    private static func unpack(_ file: WeightFile, at archive: URL, into directory: URL) async throws {
        try await Task.detached(priority: .userInitiated) {
            try WeightArchive.unpack(archive, into: directory)
        }.value
        try? FileManager.default.removeItem(at: archive)
        guard file.isPresent(in: directory) else { throw WeightFetchError.unpackFailed(file.path) }
    }

    /// Progress across files downloading at once: bytes summed over all of
    /// them, the file named being the one that last moved, and any recovery
    /// note that is still current.
    @MainActor private final class Tally {
        private let files: [WeightFile]
        private let total: Int64
        private let progress: (Progress) -> Void
        private let note: (String?) -> Void
        private var received: [String: Int64] = [:]
        private var done: Set<String> = []
        private var notes: [String: String] = [:]

        init(files: [WeightFile], total: Int64, progress: @escaping (Progress) -> Void,
             note: @escaping (String?) -> Void) {
            (self.files, self.total, self.progress, self.note) = (files, total, progress, note)
        }

        func received(_ bytes: Int64, for path: String) {
            // Updates hop here in separate tasks, which needn't arrive in
            // order: one late for a finished file must not take it back.
            guard !done.contains(path) else { return }
            received[path] = bytes
            show(path)
        }

        func status(_ line: String?, for path: String) {
            notes[path] = line
            note(notes.values.first)
        }

        func finished(_ path: String) {
            done.insert(path)
            received[path] = files.first { $0.path == path }?.bytes ?? received[path]
            notes[path] = nil
            note(notes.values.first)
            show(path)
        }

        func show(_ path: String) {
            let bytes = received.values.reduce(0, +)
            progress(Progress(file: path, completed: min(done.count, files.count - 1), total: files.count,
                              fraction: Double(bytes) / Double(max(total, 1)), received: bytes, expected: total))
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
