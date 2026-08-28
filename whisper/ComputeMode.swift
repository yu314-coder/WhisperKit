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
enum ComputeMode: String, CaseIterable, Identifiable {
    case gpu
    case neuralEngine

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .gpu:          return "GPU"
        case .neuralEngine: return "Neural Engine"
        }
    }

    var shortName: String { displayName }

    var detail: String {
        switch self {
        case .gpu:          return "Loads in seconds. Best for getting started quickly."
        case .neuralEngine: return "Uses less battery, but the first load after each update takes minutes."
        }
    }

    /// The options to actually load with, which are not always the ones the
    /// user picked.
    ///
    /// Turbo and the large models kill the app on iPhone in GPU mode, during
    /// whatever happens to be running when the encoder's weights are resident.
    /// Confirmed on device: the same model, same build, transcribes fine on
    /// the Neural Engine, and fine over GPU on iPad. The encoder simply does
    /// not fit a phone GPU's working set. Rather than offer a setting that
    /// terminates the app, GPU quietly resolves to the Neural Engine there —
    /// `gpuIsUnavailable(for:)` lets the UI say so.
    func computeOptions(for model: ContentView.WhisperModel) -> ModelComputeOptions {
        Self.gpuIsUnavailable(for: model, requested: self) ? Self.neuralEngine.computeOptions : computeOptions
    }

    /// Whether the user asked for the GPU on a device that cannot host this
    /// model there. Drives both the fallback and the explanation next to it.
    static func gpuIsUnavailable(for model: ContentView.WhisperModel, requested: ComputeMode) -> Bool {
        requested == .gpu
            && model.exceedsPhoneGPU
            && UIDevice.current.userInterfaceIdiom == .phone
    }

    var computeOptions: ModelComputeOptions {
        switch self {
        case .gpu:
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
