import Foundation
import Observation
import SwiftData
import UIKit
import UserNotifications
import MLX

/// Fetches model weights and runs generation for the Music tab.
///
/// Weights are downloaded rather than bundled — several gigabytes per
/// version. Files land in Documents and survive app updates, so the download
/// happens once.
@Observable
@MainActor
final class MusicEngine {
    enum Phase: Equatable {
        case idle
        case downloading(file: String, completed: Int, total: Int,
                         fraction: Double, received: Int64, expected: Int64)
        /// `progress` runs 0 to 1 over the whole run, weighted by how long
        /// each stage takes, so one bar can show it.
        case generating(stage: String, progress: Double = 0)
        case failed(String)
    }

    private(set) var phase: Phase = .idle {
        didSet { reportProgressToSystem() }
    }
    private(set) var lastResult: URL?
    private(set) var lastDuration: Double = 0
    private(set) var elapsedMilliseconds: Int = 0
    /// When the current run started, for the elapsed time shown while it
    /// works; and the prompt the last result was made from.
    private(set) var startedAt: Date?
    private(set) var lastPrompt: String = ""
    private(set) var lastLyrics: String = ""
    /// Shown under the download bar while a dropped transfer is recovered.
    private(set) var downloadNote: String?
    private var currentDownload: String?

    private var work: Task<Void, Never>?
    /// Keeps the run going when the app is left; see `BackgroundWork`.
    private var backgroundJob: BackgroundWork.Job?
    /// iOS ended the run in the background, to say so rather than just stop.
    private var endedInBackground = false

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
        if case .generating(_, let progress) = phase {
            phase = .generating(stage: "Stopping", progress: progress)
        } else {
            work = nil
            phase = .idle
        }
    }

    /// Downloads whatever is missing, then generates. Both phases report into
    /// `phase` so the view can show one continuous progress story.
    /// - Parameter expandsPrompt: let the planner write a fuller
    ///   description of the music from the prompt, as Suno does with a short
    ///   one; see `ACEGenerator.plannerRewritesCaption`.
    func generate(model: MusicModel, prompt: String, lyrics: String = "",
                  language: String = "en", seconds: Double, expandsPrompt: Bool = false) {
        guard !isBusy else { return }
        startedAt = Date()
        endedInBackground = false
        // The whole run — download, then generation — keeps going when the
        // app is left. It needs the GPU throughout (MLX), so where the device
        // can't lend it to the background, it pauses until the app is back.
        let job = BackgroundWork.shared.begin(title: "Making music", subtitle: Self.shortened(prompt), usesGPU: true)
        job.onExpire = { [weak self] in
            self?.endedInBackground = true
            self?.cancel()
        }
        backgroundJob = job
        work = Task { [weak self] in
            guard let self else { return }
            defer {
                MusicRunMarker.end()
                self.backgroundJob = nil
            }
            do {
                try await self.fetchWeightsIfNeeded(for: model)
                try Task.checkCancellation()
                try await self.runGeneration(model: model, prompt: prompt, lyrics: lyrics,
                                             language: language, seconds: seconds,
                                             expandsPrompt: expandsPrompt)
                job.finish(success: true)
                self.notifyIfAway(title: "Your music is ready", body: Self.shortened(prompt))
            } catch is CancellationError {
                job.finish(success: false)
                if self.endedInBackground {
                    self.phase = .failed("Stopped in the background — iOS needed the device's resources. Generate again with the app open.")
                    self.notifyIfAway(title: "Music stopped", body: "iOS stopped the generation in the background. Open Whisper to try again.")
                } else {
                    self.phase = .idle
                }
            } catch {
                job.finish(success: false)
                self.phase = .failed(error.localizedDescription)
                self.notifyIfAway(title: "Music couldn't be made", body: error.localizedDescription)
            }
        }
    }

    /// Mirrors the run's progress into the system's background banner.
    private func reportProgressToSystem() {
        guard let backgroundJob else { return }
        switch phase {
        case .downloading(_, _, _, let fraction, _, _):
            backgroundJob.update(fraction, "Downloading the music model")
        case .generating(let stage, let progress):
            backgroundJob.update(progress, stage)
        case .idle, .failed:
            break
        }
    }

    /// A notification when a run ends while the user is in another app.
    private func notifyIfAway(title: String, body: String) {
        guard UIApplication.shared.applicationState != .active else { return }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil))
    }

    private static func shortened(_ text: String) -> String {
        let line = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? text
        return line.count > 60 ? String(line.prefix(57)) + "…" : line
    }

    // MARK: - Weights

    private func fetchWeightsIfNeeded(for model: MusicModel) async throws {
        let directory = model.weightsDirectory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        removeSupersededFiles(in: directory)

        // Only what is missing is counted and fetched: a model already in
        // place from an earlier version is not downloaded again, and the
        // progress shown is for this download alone, by bytes across all of
        // it rather than file by file.
        let missing = model.weightFiles.filter { !$0.isPresent(in: directory) }
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

        WeightDownloadService.shared.onStatus = { [weak self] note in
            Task { @MainActor in self?.downloadNote = note }
        }
        defer {
            downloadNote = nil
            currentDownload = nil
        }

        var done: Int64 = 0
        for (index, file) in missing.enumerated() {
            try Task.checkCancellation()
            guard let remote = model.downloadURL(for: file) else {
                throw MusicEngineError.badURL(file.path)
            }
            let before = done
            func show(_ received: Int64) {
                phase = .downloading(file: file.path, completed: index, total: missing.count,
                                     fraction: Double(before + received) / Double(max(totalBytes, 1)),
                                     received: before + received, expected: totalBytes)
            }
            show(0)
            let destination = directory.appendingPathComponent(file.path)

            // Progress arrives from the shared background session, so it is
            // pointed at the file being fetched right now.
            // Progress hops to the main actor in separate tasks, which are not
            // guaranteed to run in order; a late one for the previous file
            // could otherwise put its bar back on screen after the next file
            // had started.
            currentDownload = file.path
            WeightDownloadService.shared.onProgress = { [weak self] progress in
                Task { @MainActor in
                    guard let self, self.currentDownload == file.path else { return }
                    show(progress.received)
                }
            }
            // Interruptions are resumed inside the service; what reaches here
            // has failed repeatedly or been cancelled.
            try await WeightDownloadService.shared.download(from: remote, to: destination)

            let written = (try? destination.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            guard Int64(written) == file.bytes else {
                throw MusicEngineError.wrongSize(file.path, Int64(written), file.bytes)
            }
            if file.isArchive {
                downloadNote = "Unpacking"
                try await Task.detached(priority: .userInitiated) {
                    try WeightArchive.unpack(destination, into: directory)
                }.value
                try? FileManager.default.removeItem(at: destination)
                downloadNote = nil
                guard file.isPresent(in: directory) else { throw MusicEngineError.unpackFailed(file.path) }
            }
            done += file.bytes
        }
    }

    /// Deletes files an earlier version downloaded that no model uses now.
    ///
    /// ACE-Step's first layout was two 1.3 GB shards and a trimmed silence
    /// file; its replacement splits the checkpoint by stage instead. Without
    /// this, updating the app would leave 2.7 GB behind in Documents that
    /// nothing reads and nothing in the app can remove.
    private func removeSupersededFiles(in directory: URL) {
        // Compared by top-level name: a Neural Engine model is a folder, and
        // its files' paths start with the folder's name.
        let expected = MusicModel.expectedTopLevelNames
        guard !expected.isEmpty,
              let present = try? FileManager.default.contentsOfDirectory(atPath: directory.path)
        else { return }
        for name in present where !expected.contains(name) && !name.hasPrefix(".") {
            try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        }
    }

    // MARK: - Generation

    private func runGeneration(model: MusicModel, prompt: String, lyrics: String,
                               language: String, seconds: Double, expandsPrompt: Bool) async throws {
        // MLX asks the Metal device for its architecture and the Simulator's
        // MTLSimDevice returns null, which MLX turns straight into a
        // std::string — strlen(NULL), a hard crash inside the library before
        // any of our code runs. Downloading is worth testing here; generating
        // is not, so refuse it with an explanation instead of segfaulting.
        #if targetEnvironment(simulator)
        throw MusicEngineError.simulatorUnsupported
        #else
        phase = .generating(stage: "Starting")
        MusicRunMarker.begin()
        try await runACEStep(prompt: prompt, lyrics: lyrics, language: language,
                             seconds: seconds, model: model, expandsPrompt: expandsPrompt)
        #endif
    }
}

extension MusicEngine {
    /// ACE-Step: prompt and lyrics to 48 kHz stereo in eight steps.
    ///
    /// Runs off the main actor — the first version ran the whole pipeline on
    /// it, which froze the screen, memory gauge included, for the length of a
    /// generation. Each run draws a new seed, so the same prompt twice gives
    /// two different takes.
    func runACEStep(prompt: String, lyrics: String, language: String,
                    seconds: Double, model: MusicModel, expandsPrompt: Bool = false) async throws {
        var generator = ACEGenerator(directory: model.weightsDirectory)
        generator.plannerExpandsCaption = expandsPrompt
        model.configure(&generator)
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("acestep-\(Int(Date().timeIntervalSince1970)).wav")
        let seed = UInt64.random(in: 0 ... UInt64(UInt32.max))
        let started = Date()

        phase = .generating(stage: "Starting", progress: 0.01)
        // Cancelling `work` does not reach a detached task, and a run left
        // going after Cancel would still hold its memory when the next one
        // starts. The flag carries the request across; the generator checks
        // it between steps.
        let flag = CancellationFlag()
        try await withTaskCancellationHandler {
            try await Task.detached(priority: .userInitiated) { [weak self] in
            _ = try generator.generate(caption: prompt, lyrics: lyrics, language: language,
                                       seconds: seconds, seed: seed, to: destination,
                                       known: PromptMetadata(parsing: prompt),
                                       // Off screen without background GPU time,
                                       // wait here: MLX would crash on a
                                       // refused command buffer.
                                       isCancelled: { GPUGate.waitUntilOpen(); return flag.isCancelled }) { stage in
                GPUGate.waitUntilOpen()
                let label: String
                let progress: Double
                switch stage {
                case .planning:
                    (label, progress) = ("Planning the song", 0.02)
                case .writing(let done, let of):
                    let part = Double(done) / Double(max(of, 1))
                    (label, progress) = ("Writing the song", 0.02 + 0.38 * part)
                case .readingPrompt:
                    (label, progress) = ("Reading the prompt", 0.41)
                case .conditioning:
                    (label, progress) = (lyrics.isEmpty ? "Setting the style" : "Reading the lyrics", 0.44)
                // iOS compiles each length for the Neural Engine once and
                // keeps it; a new length can take a minute here.
                case .preparingEngine:
                    (label, progress) = ("Preparing the Neural Engine — slow the first time at a new length", 0.47)
                case .step(let step, let of):
                    (label, progress) = ("Rendering, step \(step) of \(of)", 0.5 + 0.38 * Double(step) / Double(max(of, 1)))
                case .lengthening:
                    (label, progress) = ("Filling the full length", 0.5)
                case .decoding(let fraction):
                    (label, progress) = ("Making the audio", 0.88 + 0.12 * fraction)
                }
                Task { @MainActor [weak self] in
                    guard let self, !flag.isCancelled else { return }
                    self.phase = .generating(stage: label, progress: progress)
                }
            }
            }.value
        } onCancel: {
            flag.cancel()
        }

        lastResult = destination
        lastDuration = seconds
        lastPrompt = prompt
        lastLyrics = lyrics
        elapsedMilliseconds = Int(Date().timeIntervalSince(started) * 1000)
        save(destination, model: model, prompt: prompt, seconds: seconds,
             lyrics: lyrics, language: language, fullerArrangement: expandsPrompt)
        phase = .idle
    }

    /// Moves the finished clip out of the temporary directory into the app's
    /// audio folder and records it, so it survives the app being closed.
    /// A failure here must not lose the generated audio, so `lastResult` keeps
    /// pointing at whatever the caller can still play.
    func save(_ url: URL, model: MusicModel, prompt: String, seconds: Double,
              lyrics: String = "", language: String? = nil, fullerArrangement: Bool? = nil) {
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
                                  waveform: envelope?.buckets,
                                  lyrics: lyrics.isEmpty ? nil : lyrics,
                                  language: lyrics.isEmpty ? nil : language,
                                  fullerArrangement: fullerArrangement)
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
    case unpackFailed(String)

    var errorDescription: String? {
        switch self {
        case .badURL(let name):     return "No download address for \(name)."
        case .http(let code):       return "Download failed (HTTP \(code))."
        case .notImplemented(let n): return "\(n) has no on-device implementation yet."
        case .unpackFailed(let name):
            return "\(name) downloaded but did not unpack completely. Generate again to fetch it anew."
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
