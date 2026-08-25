import Foundation
import CoreML
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
