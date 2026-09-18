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
            // Sizes are approximate megabytes, so allow a margin rather than
            // demanding an exact match.
            return Int64(size) < Int64(Double(file.megabytes) * 900_000)
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

    private func download(from remote: URL, to destination: URL,
                          onProgress: @escaping @MainActor (Double) -> Void) async throws {
        let (stream, response) = try await URLSession.shared.bytes(from: remote)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw MusicEngineError.http((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
        let expected = response.expectedContentLength

        // Written to a partial file and moved into place at the end, so an
        // interrupted download never looks like a complete one.
        let partial = destination.appendingPathExtension("partial")
        FileManager.default.createFile(atPath: partial.path, contents: nil)
        let handle = try FileHandle(forWritingTo: partial)
        defer { try? handle.close() }

        var buffer = Data()
        buffer.reserveCapacity(1 << 20)
        var received: Int64 = 0
        var lastReported = 0.0

        for try await byte in stream {
            buffer.append(byte)
            if buffer.count >= (1 << 20) {
                try handle.write(contentsOf: buffer)
                received += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                if expected > 0 {
                    let fraction = Double(received) / Double(expected)
                    if fraction - lastReported >= 0.01 {
                        lastReported = fraction
                        onProgress(fraction)
                    }
                }
                try Task.checkCancellation()
            }
        }
        if !buffer.isEmpty { try handle.write(contentsOf: buffer) }
        try handle.close()
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: partial, to: destination)
    }

    // MARK: - Generation

    private func runGeneration(model: MusicModel, prompt: String, seconds: Double) async throws {
        guard let kind = model.stableAudioKind else {
            throw MusicEngineError.notImplemented(model.displayName)
        }
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
    }
}

enum MusicEngineError: LocalizedError {
    case badURL(String)
    case http(Int)
    case notImplemented(String)

    var errorDescription: String? {
        switch self {
        case .badURL(let name):     return "No download address for \(name)."
        case .http(let code):       return "Download failed (HTTP \(code))."
        case .notImplemented(let n): return "\(n) has no on-device implementation yet."
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
