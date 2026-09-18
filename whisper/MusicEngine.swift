import Foundation
import Observation

/// Fetches model weights and runs generation for the Music tab.
///
/// Weights are downloaded rather than bundled — the Stable Audio set alone is
/// 1.7 GB, and Medium is 3.7 GB. Files land in Documents and survive app
/// updates, so the download happens once per model.
@Observable
@MainActor
final class MusicEngine {
    enum Phase: Equatable {
        case idle
        case downloading(file: String, completed: Int, total: Int, fraction: Double)
        case generating(stage: String)
        case failed(String)
    }

    private(set) var phase: Phase = .idle
    private(set) var lastResult: URL?
    private(set) var lastDuration: Double = 0
    private(set) var elapsedMilliseconds: Int = 0

    private let pipeline = StableAudioPipeline()
    private var work: Task<Void, Never>?

    /// The message from a failed run, if the last attempt failed. The view
    /// needs this to decide whether to show the progress area at all: a
    /// failure leaves `isBusy` false and `lastResult` nil, so without it the
    /// explanation had nowhere to appear and the button just reset itself.
    var failureMessage: String? {
        if case .failed(let message) = phase { return message }
        return nil
    }

    var isBusy: Bool {
        switch phase {
        case .idle, .failed: return false
        case .downloading, .generating: return true
        }
    }

    func cancel() {
        work?.cancel()
        work = nil
        phase = .idle
    }

    /// Downloads whatever is missing, then generates. Both phases report into
    /// `phase` so the view can show one continuous progress story.
    func generate(model: MusicModel, prompt: String, seconds: Double) {
        guard !isBusy else { return }
        work = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.fetchWeightsIfNeeded(for: model)
                try Task.checkCancellation()
                try await self.runGeneration(model: model, prompt: prompt, seconds: seconds)
            } catch is CancellationError {
                self.phase = .idle
            } catch {
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    // MARK: - Weights

    private func fetchWeightsIfNeeded(for model: MusicModel) async throws {
        let directory = SA3Weights.directory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let files = model.weightFiles
        let missing = files.enumerated().filter { _, file in
            let url = directory.appendingPathComponent(file.path)
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return Int64(size) != file.bytes
        }
        guard !missing.isEmpty else { return }

        for (index, file) in missing {
            try Task.checkCancellation()
            guard let remote = model.downloadURL(for: file.path) else {
                throw MusicEngineError.badURL(file.path)
            }
            phase = .downloading(file: file.path, completed: index, total: files.count, fraction: 0)
            let destination = directory.appendingPathComponent(file.path)
            try await download(from: remote, to: destination) { [weak self] fraction in
                self?.phase = .downloading(file: file.path, completed: index,
                                           total: files.count, fraction: fraction)
            }
        }
    }

    /// A download task rather than an `AsyncBytes` loop.
    ///
    /// Iterating `URLSession.bytes` yields one `UInt8` at a time, which for a
    /// 919 MB weight file is 919 million iterations — far too slow to finish.
    /// `URLSessionDownloadTask` streams to disk itself and reports progress.
    private func download(from remote: URL, to destination: URL,
                          onProgress: @escaping @MainActor (Double) -> Void) async throws {
        let delegate = DownloadProgressDelegate { fraction in
            Task { @MainActor in onProgress(fraction) }
        }
        let (temporary, response) = try await URLSession.shared.download(from: remote, delegate: delegate)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            try? FileManager.default.removeItem(at: temporary)
            throw MusicEngineError.http((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
        // Moved into place only once complete, so an interrupted transfer can
        // never be mistaken for a finished model on the next launch.
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: temporary, to: destination)
    }

    // MARK: - Generation

    private func runGeneration(model: MusicModel, prompt: String, seconds: Double) async throws {
        guard let kind = model.stableAudioKind else {
            throw MusicEngineError.notImplemented(model.displayName)
        }
        // MLX asks the Metal device for its architecture and the Simulator's
        // MTLSimDevice returns null, which MLX turns straight into a
        // std::string — strlen(NULL), a hard crash inside the library before
        // any of our code runs. Downloading is worth testing here; generating
        // is not, so refuse it with an explanation instead of segfaulting.
        #if targetEnvironment(simulator)
        throw MusicEngineError.simulatorUnsupported
        #else
        phase = .generating(stage: "Starting")
        let result = try await pipeline.generate(
            model: kind,
            prompt: prompt,
            seconds: Float(seconds),
            steps: 8,
            progress: { stage in
                Task { @MainActor [weak self] in self?.phase = .generating(stage: stage) }
            }
        )
        lastResult = result.url
        lastDuration = Double(result.duration)
        elapsedMilliseconds = Int(result.elapsedSeconds * 1000)
        phase = .idle
        #endif
    }
}

enum MusicEngineError: LocalizedError {
    case badURL(String)
    case http(Int)
    case notImplemented(String)
    case simulatorUnsupported

    var errorDescription: String? {
        switch self {
        case .badURL(let name):     return "No download address for \(name)."
        case .http(let code):       return "Download failed (HTTP \(code))."
        case .notImplemented(let n): return "\(n) has no on-device implementation yet."
        case .simulatorUnsupported:
            return """
            Music generation needs a real device. The Simulator's Metal \
            device cannot run MLX. Downloading models works here; generating \
            does not.
            """
        }
    }
}

extension MusicModel {
    /// The Stable Audio pipeline variant backing this entry, if any.
    var stableAudioKind: StableAudioModelKind? {
        switch self {
        case .stableAudio3Small:  return .smallMusic
        case .stableAudio3Medium: return .medium
        default:                  return nil
        }
    }
}


/// Reports download progress. `URLSession`'s async `download(from:delegate:)`
/// has no progress callback of its own; this supplies one.
private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: @Sendable (Double) -> Void
    private var lastReported = 0.0

    init(onProgress: @escaping @Sendable (Double) -> Void) {
        self.onProgress = onProgress
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let fraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        // A gigabyte file would otherwise post thousands of updates a second.
        guard fraction - lastReported >= 0.005 else { return }
        lastReported = fraction
        onProgress(fraction)
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // The async form of `download` takes ownership of the file; nothing to
        // do here, but the delegate protocol requires the method.
    }
}
