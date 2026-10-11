import Foundation
import Observation
import SwiftData
import UIKit
import UserNotifications
import CoreGraphics

/// A picture's size, as the models are built for it.
struct ImageSize: Hashable, Identifiable {
    let width: Int
    let height: Int
    var id: String { "\(width)x\(height)" }
    var label: String { "\(width) × \(height)" }
    var shapeName: String { width == height ? "Square" : (width > height ? "Wide" : "Tall") }
    var symbol: String { width == height ? "square" : (width > height ? "rectangle" : "rectangle.portrait") }
}

extension ImageModel {
    /// The sizes each model makes. DreamShaper and SANA-Sprint are built
    /// for one size each; FLUX.2 klein and Z-Image take any multiple of 16
    /// (32 for Z-Image's patches), and these are the ones offered.
    var sizes: [ImageSize] {
        switch self {
        case .dreamShaper: return [ImageSize(width: 512, height: 512)]
        case .sanaSprint, .sanaSprintLarge: return [ImageSize(width: 1024, height: 1024)]
        case .fluxKlein, .zImage:
                           return [ImageSize(width: 768, height: 768), ImageSize(width: 1024, height: 1024),
                                   ImageSize(width: 768, height: 1024), ImageSize(width: 1024, height: 768),
                                   ImageSize(width: 512, height: 512)]
        }
    }
}

/// Fetches an image model's weights and makes pictures with it.
@Observable
@MainActor
final class ImageEngine {
    enum Phase: Equatable {
        case idle
        case downloading(file: String, completed: Int, total: Int,
                         fraction: Double, received: Int64, expected: Int64)
        /// `progress` runs 0 to 1 over the whole run.
        case generating(stage: String, progress: Double = 0)
        case failed(String)
        /// The picture was made but looked sexually explicit, so it was
        /// neither shown nor saved (SafetyChecker).
        case withheld
    }

    /// Thrown by `render` for a picture the safety check keeps back.
    private struct Withheld: Error {}

    private struct SimulatorUnsupported: LocalizedError {
        let model: String
        var errorDescription: String? {
            "\(model) needs a real device: the Simulator's Metal device cannot run MLX. DreamShaper works here."
        }
    }

    private(set) var phase: Phase = .idle {
        didSet { reportProgressToSystem() }
    }
    /// The picture just made, until the next run starts.
    private(set) var lastImage: SavedImage?
    private(set) var startedAt: Date?
    private(set) var downloadNote: String?

    private var work: Task<Void, Never>?
    private var backgroundJob: BackgroundWork.Job?
    private var endedInBackground = false

    var modelContext: ModelContext?

    var failureMessage: String? {
        if case .failed(let message) = phase { return message }
        return nil
    }

    var isBusy: Bool {
        switch phase {
        case .idle, .failed, .withheld: return false
        case .downloading, .generating: return true
        }
    }

    /// A download stops at once; a generation at its next step.
    func cancel() {
        work?.cancel()
        if case .generating(_, let progress) = phase {
            phase = .generating(stage: "Stopping", progress: progress)
        } else {
            work = nil
            phase = .idle
        }
    }

    func generate(model: ImageModel, prompt: String, size: ImageSize, seed: UInt64? = nil) {
        guard !isBusy else { return }
        startedAt = Date()
        endedInBackground = false
        lastImage = nil
        let job = BackgroundWork.shared.begin(title: "Making a picture", subtitle: Self.shortened(prompt),
                                              usesGPU: model.usesGPU)
        job.onExpire = { [weak self] in
            self?.endedInBackground = true
            self?.cancel()
        }
        backgroundJob = job
        work = Task { [weak self] in
            guard let self else { return }
            defer { self.backgroundJob = nil }
            do {
                // The model and the picture check all three share, as one
                // download: every file in flight at once, so all of it
                // arrives even if the app is closed partway.
                try await WeightFetcher.fetch([
                    .init(files: model.weightFiles, directory: model.weightsDirectory,
                          expected: WeightFile.topLevelNames(model.folderFiles)),
                    .init(files: ImageModel.safetyFiles, directory: ImageModel.safetyDirectory,
                          expected: WeightFile.topLevelNames(ImageModel.safetyFiles)),
                ], progress: { [weak self] update in
                    self?.phase = .downloading(file: update.file, completed: update.completed, total: update.total,
                                               fraction: update.fraction, received: update.received,
                                               expected: update.expected)
                }, note: { [weak self] line in
                    self?.downloadNote = line
                })
                try Task.checkCancellation()
                try await self.run(model: model, prompt: prompt, size: size,
                                   seed: seed ?? UInt64.random(in: 0 ... UInt64(UInt32.max)))
                job.finish(success: true)
                self.notifyIfAway(title: "Your picture is ready", body: Self.shortened(prompt))
            } catch is CancellationError {
                job.finish(success: false)
                if self.endedInBackground {
                    self.phase = .failed("Stopped in the background — iOS needed the device's resources. Try again with the app open.")
                    self.notifyIfAway(title: "Picture stopped", body: "iOS stopped it in the background. Open Whisper to try again.")
                } else {
                    self.phase = .idle
                }
            } catch is Withheld {
                job.finish(success: true)
                self.phase = .withheld
                self.notifyIfAway(title: "Picture not shown", body: "It looked sexually explicit. Open Whisper to try another.")
            } catch {
                job.finish(success: false)
                self.phase = .failed(error.localizedDescription)
                self.notifyIfAway(title: "The picture couldn't be made", body: error.localizedDescription)
            }
        }
    }

    // MARK: - Generation

    private func run(model: ImageModel, prompt: String, size: ImageSize, seed: UInt64) async throws {
        // MLX crashes on the Simulator's Metal device (see MusicEngine);
        // DreamShaper is Core ML only and runs there, slowly.
        #if targetEnvironment(simulator)
        if model.usesGPU { throw SimulatorUnsupported(model: model.displayName) }
        #endif
        phase = .generating(stage: "Starting", progress: 0.01)
        let started = Date()
        let directory = model.weightsDirectory
        let flag = CancellationFlag()
        let report: @Sendable (String, Double) -> Void = { [weak self] label, progress in
            Task { @MainActor [weak self] in
                guard let self, !flag.isCancelled else { return }
                self.phase = .generating(stage: label, progress: progress)
            }
        }
        let image: CGImage = try await withTaskCancellationHandler {
            try await Task.detached(priority: .userInitiated) {
                let image = try Self.render(model: model, prompt: prompt, size: size, seed: seed, directory: directory,
                                            isCancelled: { flag.isCancelled }, report: report)
                report("Checking the picture", 0.97)
                if try SafetyChecker(directory: ImageModel.safetyDirectory).isExplicit(image) { throw Withheld() }
                return image
            }.value
        } onCancel: {
            flag.cancel()
        }
        try save(image, model: model, prompt: prompt, size: size, seed: seed,
                 seconds: Date().timeIntervalSince(started))
        phase = .idle
    }

    /// Runs the model off the main actor and returns the picture.
    nonisolated private static func render(model: ImageModel, prompt: String, size: ImageSize, seed: UInt64,
                                           directory: URL, isCancelled: @escaping () -> Bool,
                                           report: @escaping (String, Double) -> Void) throws -> CGImage {
        let stage: (SDPipeline.Stage) -> Void = { stage in
            switch stage {
            case .loading:
                report("Loading the model — slow the first time, while iOS prepares the Neural Engine", 0.03)
            case .readingPrompt:
                report("Reading the prompt", 0.08)
            case .step(let step, let of):
                report("Painting, step \(step) of \(of)", 0.12 + 0.72 * Double(step - 1) / Double(of))
            case .decoding:
                report("Finishing the picture", 0.88)
            }
        }
        switch model {
        case .dreamShaper:
            return try SDPipeline(directory: directory).generate(prompt: prompt, seed: seed,
                                                                 isCancelled: isCancelled, progress: stage)
        case .sanaSprint, .sanaSprintLarge:
            // Gemma and the decoder's first half run on the GPU. Off screen
            // without background GPU time they wait between layers: MLX
            // would crash on a refused command buffer.
            return try SanaPipeline(directory: directory, large: model == .sanaSprintLarge).generate(prompt: prompt, seed: seed,
                                                                   isCancelled: { GPUGate.waitUntilOpen(); return isCancelled() },
                                                                   progress: stage)
        case .fluxKlein:
            // All on the GPU: off screen without background GPU time it
            // waits between blocks.
            return try FluxPipeline(directory: directory).generate(prompt: prompt, width: size.width, height: size.height,
                                                                   seed: seed,
                                                                   isCancelled: { GPUGate.waitUntilOpen(); return isCancelled() },
                                                                   progress: stage)
        case .zImage:
            return try ZImagePipeline(directory: directory).generate(prompt: prompt, width: size.width, height: size.height,
                                                                     seed: seed,
                                                                     isCancelled: { GPUGate.waitUntilOpen(); return isCancelled() },
                                                                     progress: stage)
        }
    }

    private func save(_ image: CGImage, model: ImageModel, prompt: String, size: ImageSize,
                      seed: UInt64, seconds: Double) throws {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let path = try ImageFiles.save(image, prompt: trimmed, model: model.displayName)
        let saved = SavedImage(prompt: trimmed.isEmpty ? "Untitled" : trimmed, modelName: model.displayName,
                               filePath: path, width: image.width, height: image.height,
                               seed: Int64(seed), seconds: seconds)
        if let modelContext {
            modelContext.insert(saved)
            try? modelContext.save()
        }
        lastImage = saved
    }

    // MARK: - Background

    private func reportProgressToSystem() {
        guard let backgroundJob else { return }
        switch phase {
        case .downloading(_, _, _, let fraction, _, _):
            backgroundJob.update(fraction, "Downloading the image model")
        case .generating(let stage, let progress):
            backgroundJob.update(progress, stage)
        case .idle, .failed, .withheld:
            break
        }
    }

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
}
