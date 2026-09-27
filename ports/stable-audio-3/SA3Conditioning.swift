//  Adapted from StableAudio3-IOS (https://github.com/kellyvv/StableAudio3-IOS),
//  MIT licensed, Copyright (c) 2026 stableaudio3-ios contributors.
//  Changed here: weights are resolved from the downloaded-models directory
//  rather than the app bundle, so no model files ship inside the app.
import Foundation
import MLX

struct SA3Conditioning {
    let weights: [String: MLXArray]

    func makeConditioning(promptEncoding: T5PromptEncoding, seconds: Float) -> (crossAttention: MLXArray, globalCondition: MLXArray) {
        let paddedPrompt = applyPromptPadding(
            embeddings: promptEncoding.embeddings,
            mask: promptEncoding.mask,
            paddingEmbedding: weights["cond.padding_embedding"]!
        )
        let secondsEmbedding = secondsTotalEmbedding(seconds)
        let crossAttention = concatenated([paddedPrompt, secondsEmbedding.asType(paddedPrompt.dtype)], axis: 1)
        let globalCondition = secondsEmbedding[0..., 0, 0...].asType(.float16)
        eval(crossAttention, globalCondition)
        return (crossAttention, globalCondition)
    }

    private func applyPromptPadding(embeddings: MLXArray, mask: MLXArray, paddingEmbedding: MLXArray) -> MLXArray {
        let m = mask.asType(embeddings.dtype).expandedDimensions(axis: -1)
        let pad = paddingEmbedding.asType(embeddings.dtype).reshaped(1, 1, -1)
        return embeddings * m + pad * (1.0 - m)
    }

    private func secondsTotalEmbedding(_ seconds: Float) -> MLXArray {
        let minValue: Float = 0
        let maxValue: Float = 384
        let fourierDimension = 256
        let half = fourierDimension / 2

        var value = MLXArray([seconds], [1]).asType(.float32)
        value = clip(value, min: minValue, max: maxValue)
        let normalized = ((value - minValue) / (maxValue - minValue)).reshaped(-1, 1)

        let ramp = MLX.arange(half, dtype: .float32) / Float(max(half - 1, 1))
        let minFrequency: Float = 0.5
        let maxFrequency: Float = 10_000
        let frequencies = exp(ramp * (log(maxFrequency) - log(minFrequency)) + log(minFrequency))
        let args = normalized * frequencies * (2.0 * Float.pi)
        let features = concatenated([cos(args), sin(args)], axis: -1)

        let out = matmul(features, weights["cond.seconds_total_weight"]!.T) + weights["cond.seconds_total_bias"]!
        return out.expandedDimensions(axis: 1)
    }
}

enum SA3ConditioningTester {
    struct Report {
        let crossShape: String
        let globalShape: String
        let tokenCount: Int
        let elapsedSeconds: TimeInterval
    }

    static func run(prompt: String, seconds: Float = 10) -> Result<Report, Error> {
        do {
            let startedAt = Date()
            return try Stream.withNewDefaultStream(device: .gpu) {
                guard let url = SA3Weights.url(
                    forResource: "sa3_conditioner",
                    withExtension: "safetensors",
                    subdirectory: "Weights"
                ) else {
                    throw WeightTensorLoaderError.missing("sa3_conditioner.safetensors")
                }

                let promptEncoding = try T5PromptEncoder.encode(prompt: prompt, maxLength: 256)
                let weights = try loadArrays(url: url, stream: .cpu)
                let conditioner = SA3Conditioning(weights: weights)
                let result = conditioner.makeConditioning(promptEncoding: promptEncoding, seconds: seconds)
                let crossShape = result.crossAttention.shape.map(String.init).joined(separator: "x")
                let globalShape = result.globalCondition.shape.map(String.init).joined(separator: "x")
                return .success(
                    Report(
                        crossShape: crossShape,
                        globalShape: globalShape,
                        tokenCount: promptEncoding.tokenCount,
                        elapsedSeconds: Date().timeIntervalSince(startedAt)
                    )
                )
            }
        } catch {
            return .failure(error)
        }
    }
}
