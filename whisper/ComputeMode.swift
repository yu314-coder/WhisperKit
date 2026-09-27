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
