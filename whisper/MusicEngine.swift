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
        case downloading(file: String, completed: Int, total: Int,
                         fraction: Double, received: Int64, expected: Int64)
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
            phase = .downloading(file: file.path, completed: index, total: files.count,
                                 fraction: 0, received: 0, expected: file.bytes)
            let destination = directory.appendingPathComponent(file.path)

            // Progress arrives from the shared background session, so it is
            // pointed at the file being fetched right now.
            WeightDownloadService.shared.onProgress = { [weak self] progress in
                Task { @MainActor in
                    self?.phase = .downloading(file: file.path, completed: index,
                                               total: files.count, fraction: progress.fraction,
                                               received: progress.received, expected: progress.expected)
                }
            }
            do {
                try await WeightDownloadService.shared.download(from: remote, to: destination)
            } catch {
                // A background transfer that fails is retried once; iOS has
                // already done its own reconnection attempts by this point.
                try Task.checkCancellation()
                try await WeightDownloadService.shared.download(from: remote, to: destination)
            }

            let written = (try? destination.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            guard Int64(written) == file.bytes else {
                throw MusicEngineError.wrongSize(file.path, Int64(written), file.bytes)
            }
        }
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
    case wrongSize(String, Int64, Int64)

    var errorDescription: String? {
        switch self {
        case .badURL(let name):     return "No download address for \(name)."
        case .http(let code):       return "Download failed (HTTP \(code))."
        case .notImplemented(let n): return "\(n) has no on-device implementation yet."
        case .wrongSize(let name, let got, let want):
            return "\(name) downloaded \(got) bytes, expected \(want). The file is incomplete."
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
