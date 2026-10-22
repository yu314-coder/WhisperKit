import Foundation
import CoreML
import UIKit
import WhisperKit


/// Which processor Core ML runs the transcription models on.
///
/// The Neural Engine is the most power-efficient option, but Core ML must
/// "specialise" a model for the ANE the first time it loads — a compile that
/// takes minutes for a large model, even on an M3. The GPU needs no such step,
/// so it reaches a usable state far sooner at some cost in battery.
///
/// But the GPU pays again on every launch: Core ML keeps the Neural Engine's
/// compile across launches and rebuilds the GPU's graph each time. Timed on
/// an M4 with Turbo in a fresh process, the Neural Engine was ready and had
/// transcribed a clip in 3.2 s, the GPU in 12.2 s. Automatic takes the GPU
/// only until the Neural Engine is ready, compiling it in the background.
enum ComputeMode: String, CaseIterable, Identifiable {
    case automatic
    case gpu
    case neuralEngine

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .automatic:    return "Automatic"
        case .gpu:          return "GPU"
        case .neuralEngine: return "Neural Engine"
        }
    }

    var shortName: String { displayName }

    var detail: String {
        switch self {
        case .automatic:    return "Starts on the GPU while the Neural Engine is prepared in the background, then uses the Neural Engine: the quickest to load from then on."
        case .gpu:          return "Loads in seconds. Best for getting started quickly."
        case .neuralEngine: return "Uses less battery, but the first load after each update takes minutes."
        }
    }

    /// The options to load with: the ones the user picked.
    ///
    /// Turbo and the large models have closed the app on iPhone in GPU mode —
    /// the encoder does not fit a phone GPU's working set, while the same
    /// model runs on the Neural Engine, and over the GPU on iPad. This used
    /// to switch such a choice to the Neural Engine without asking. The
    /// choice is now kept and `gpuMayFail(for:requested:)` warns beside it;
    /// if a load is cut short anyway, the next launch does not retry it
    /// (`crashedModelVariant`) and says why.
    func computeOptions(for model: ContentView.WhisperModel) -> ModelComputeOptions {
        computeOptions
    }

    /// Whether the GPU is a risky choice for this model on this device.
    static func gpuMayFail(for model: ContentView.WhisperModel, requested: ComputeMode) -> Bool {
        requested == .gpu
            && model.exceedsPhoneGPU
            && UIDevice.current.userInterfaceIdiom == .phone
    }

    /// Automatic has no options of its own: it loads as whichever of the two
    /// it resolves to (`ContentView.effectiveMode(for:)`).
    var computeOptions: ModelComputeOptions {
        switch self {
        case .gpu, .automatic:
            // No ANE anywhere, so nothing has to be specialised.
            return ModelComputeOptions(
                melCompute: .cpuAndGPU,
                audioEncoderCompute: .cpuAndGPU,
                textDecoderCompute: .cpuAndGPU
            )
        case .neuralEngine:
            return ModelComputeOptions(
                melCompute: .cpuAndGPU,
                audioEncoderCompute: .cpuAndNeuralEngine,
                textDecoderCompute: .cpuAndNeuralEngine
            )
        }
    }
}

/// Automatic mode's background half: compiles a downloaded model for the
/// Neural Engine without keeping it, as WhisperKit's prewarm does — one
/// sub-model at a time — so the next load finds Core ML's compile cached.
/// The Mel model runs on the GPU in both modes and has nothing to compile.
enum NeuralEngineWarmup {
    @MainActor private static var running: Set<String> = []
    private static let attemptsKey = "neuralEngineWarmupAttempts"

    /// Claims `key` for a compile, or says why not: one is already running,
    /// or two have been cut short (the app closed, or iOS ended it for
    /// memory), in which case Automatic stays on the GPU for this model.
    @MainActor static func claim(_ key: String) -> Bool {
        var attempts = UserDefaults.standard.dictionary(forKey: attemptsKey) as? [String: Int] ?? [:]
        guard !running.contains(key), attempts[key, default: 0] < 2 else { return false }
        attempts[key, default: 0] += 1
        UserDefaults.standard.set(attempts, forKey: attemptsKey)
        running.insert(key)
        return true
    }

    @MainActor static func release(_ key: String, succeeded: Bool) {
        running.remove(key)
        guard succeeded else { return }
        var attempts = UserDefaults.standard.dictionary(forKey: attemptsKey) as? [String: Int] ?? [:]
        attempts[key] = nil
        UserDefaults.standard.set(attempts, forKey: attemptsKey)
    }

    static func compile(folder: URL) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        for name in ["TextDecoder", "AudioEncoder"] {
            try Task.checkCancellation()
            _ = try await MLModel.load(contentsOf: ModelUtilities.detectModelURL(inFolder: folder, named: name),
                                       configuration: config)
        }
    }
}
