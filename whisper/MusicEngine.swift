import Foundation
import Observation
import SwiftData
import MLX

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

    /// Set by the view so a finished clip can be filed in the library.
    var modelContext: ModelContext?

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

    /// A download stops at once. A generation stops at its next step, and
    /// until then the engine stays busy — letting Generate start another run
    /// while the first is still unwinding would hold both in memory.
    func cancel() {
        work?.cancel()
        if case .generating = phase {
            phase = .generating(stage: "Stopping")
        } else {
            work = nil
            phase = .idle
        }
    }

    /// Downloads whatever is missing, then generates. Both phases report into
    /// `phase` so the view can show one continuous progress story.
    func generate(model: MusicModel, prompt: String, lyrics: String = "",
                  language: String = "en", seconds: Double) {
        guard !isBusy else { return }
        work = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.fetchWeightsIfNeeded(for: model)
                try Task.checkCancellation()
                try await self.runGeneration(model: model, prompt: prompt, lyrics: lyrics,
                                             language: language, seconds: seconds)
            } catch is CancellationError {
                self.phase = .idle
            } catch {
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    // MARK: - Weights

    private func fetchWeightsIfNeeded(for model: MusicModel) async throws {
        let directory = model.weightsDirectory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        removeSupersededFiles(in: directory)

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

    /// Deletes files an earlier version downloaded that no model uses now.
    ///
    /// ACE-Step's first layout was two 1.3 GB shards and a trimmed silence
    /// file; its replacement splits the checkpoint by stage instead. Without
    /// this, updating the app would leave 2.7 GB behind in Documents that
    /// nothing reads and nothing in the app can remove.
    private func removeSupersededFiles(in directory: URL) {
        let expected = Set(MusicModel.allCases
            .filter { $0.weightsDirectory == directory }
            .flatMap { $0.weightFiles.map(\.path) })
        guard !expected.isEmpty,
              let present = try? FileManager.default.contentsOfDirectory(atPath: directory.path)
        else { return }
        for name in present where !expected.contains(name) && !name.hasPrefix(".") {
            try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        }
    }

    // MARK: - Generation

    private func runGeneration(model: MusicModel, prompt: String, lyrics: String,
                               language: String, seconds: Double) async throws {
        // MLX asks the Metal device for its architecture and the Simulator's
        // MTLSimDevice returns null, which MLX turns straight into a
        // std::string — strlen(NULL), a hard crash inside the library before
        // any of our code runs. Downloading is worth testing here; generating
        // is not, so refuse it with an explanation instead of segfaulting.
        #if targetEnvironment(simulator)
        throw MusicEngineError.simulatorUnsupported
        #else
        phase = .generating(stage: "Starting")

        // Routing comes before the Stable Audio check, not after it.
        // ACE-Step has no Stable Audio variant, so asking for one first threw
        // "no on-device implementation" and never reached this branch.
        if model == .aceStep15 {
            try await runACEStep(prompt: prompt, lyrics: lyrics, language: language,
                                 seconds: seconds, model: model)
            return
        }
        guard let kind = model.stableAudioKind else {
            throw MusicEngineError.notImplemented(model.displayName)
        }
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
        save(result.url, model: model, prompt: prompt, seconds: Double(result.duration))
        phase = .idle
        #endif
    }
}

extension MusicEngine {
    /// ACE-Step: prompt and lyrics to 48 kHz stereo in eight steps.
    ///
    /// Runs off the main actor — the first version ran the whole pipeline on
    /// it, which froze the screen, memory gauge included, for the length of a
    /// generation. Each run draws a new seed, so the same prompt twice gives
    /// two different takes, as it does with Stable Audio.
    func runACEStep(prompt: String, lyrics: String, language: String,
                    seconds: Double, model: MusicModel) async throws {
        let generator = ACEGenerator(directory: model.weightsDirectory)
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("acestep-\(Int(Date().timeIntervalSince1970)).wav")
        let seed = UInt64.random(in: 0 ... UInt64(UInt32.max))
        let started = Date()

        phase = .generating(stage: "Reading prompt")
        // Cancelling `work` does not reach a detached task, and a run left
        // going after Cancel would still hold its memory when the next one
        // starts. The flag carries the request across; the generator checks
        // it between steps.
        let flag = CancellationFlag()
        try await withTaskCancellationHandler {
            try await Task.detached(priority: .userInitiated) { [weak self] in
            _ = try generator.generate(caption: prompt, lyrics: lyrics, language: language,
                                       seconds: seconds, seed: seed, to: destination,
                                       isCancelled: { flag.isCancelled }) { stage in
                let label: String
                switch stage {
                case .readingPrompt:          label = "Reading prompt"
                case .conditioning:           label = lyrics.isEmpty ? "Conditioning" : "Reading lyrics"
                case .step(let step, let of): label = "Step \(step) of \(of)"
                case .decoding(let fraction): label = "Decoding audio \(Int(fraction * 100))%"
                }
                Task { @MainActor [weak self] in
                    guard let self, !flag.isCancelled else { return }
                    self.phase = .generating(stage: label)
                }
            }
            }.value
        } onCancel: {
            flag.cancel()
        }

        lastResult = destination
        lastDuration = seconds
        elapsedMilliseconds = Int(Date().timeIntervalSince(started) * 1000)
        save(destination, model: model, prompt: prompt, seconds: seconds)
        phase = .idle
    }

    /// Moves the finished clip out of the temporary directory into the app's
    /// audio folder and records it, so it survives the app being closed.
    /// A failure here must not lose the generated audio, so `lastResult` keeps
    /// pointing at whatever the caller can still play.
    func save(_ url: URL, model: MusicModel, prompt: String, seconds: Double) {
        guard let modelContext else { return }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            let name = "Music_\(Int(Date().timeIntervalSince1970)).wav"
            let relative = try AudioFiles.saveAudio(from: url, suggestedName: name, movingSource: true)
            let stored = AudioFiles.urlForRelativePath(relative)
            let envelope = AudioConverter.peakEnvelope(of: stored, bucketCount: 120)
            let clip = SavedMusic(prompt: trimmed.isEmpty ? "Untitled" : trimmed,
                                  duration: seconds,
                                  modelName: model.displayName,
                                  audioFilePath: relative,
                                  waveform: envelope?.buckets)
            modelContext.insert(clip)
            try modelContext.save()
            lastResult = stored
        } catch {
            print("Could not file generated clip: \(error)")
        }
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
