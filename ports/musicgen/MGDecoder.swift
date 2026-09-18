import Foundation
import MLX

/// MusicGen's decoder: a 24-layer transformer that predicts four codebook
/// streams at once, conditioned on the T5 encoding through cross-attention.
///
/// Shape convention follows the checkpoint: token ids arrive as
/// (codebooks, steps) and logits come back the same way, one vocabulary
/// distribution per codebook per step. The four codebook embeddings are
/// **summed** into a single hidden state rather than concatenated — the model
/// is one transformer, not four.
///
/// Unlike the T5 encoder alongside it, this one does scale attention by
/// 1/sqrt(head dimension), uses ordinary LayerNorm with bias, and carries no
/// bias on its projections.
/// Per-layer attention state for incremental decoding.
///
/// Self-attention keys and values grow by one column per step; cross-attention
/// keys and values depend only on the text encoding, so they are computed on
/// the first step and reused for the whole clip.
final class MGDecoderCache {
    var selfKeys: [MLXArray?]
    var selfValues: [MLXArray?]
    var crossKeys: [MLXArray?]
    var crossValues: [MLXArray?]
    private(set) var length = 0

    init(layerCount: Int) {
        selfKeys = Array(repeating: nil, count: layerCount)
        selfValues = Array(repeating: nil, count: layerCount)
        crossKeys = Array(repeating: nil, count: layerCount)
        crossValues = Array(repeating: nil, count: layerCount)
    }

    func advance(by steps: Int) { length += steps }
}

struct MGDecoder {
    static let layerCount = 24
    static let headCount = 16
    static let headDimension = 64
    static let modelDimension = 1024
    static let codebookCount = 4
    static let vocabularySize = 2048
    static let epsilon: Float = 1e-5

    let weights: [String: MLXArray]

    private func w(_ key: String) -> MLXArray {
        guard let value = weights["decoder.\(key)"] else {
            fatalError("missing decoder weight: decoder.\(key)")
        }
        return value.asType(.float32)
    }

    /// - Parameters:
    ///   - tokens: (codebooks, steps) ids already in delay-pattern order.
    ///   - encoderHidden: (1, textLength, 1024) from `encoderToDecoder`.
    ///   - encoderMask: (1, textLength) with 1 for real tokens.
    ///   - offset: position of the first step, so cached decoding continues
    ///     the positional embedding rather than restarting it.
    /// - Returns: (codebooks, steps, vocabulary) logits.
    func callAsFunction(tokens: MLXArray,
                        encoderHidden: MLXArray,
                        encoderMask: MLXArray,
                        offset: Int = 0,
                        cache: MGDecoderCache? = nil) -> MLXArray {
        let steps = tokens.dim(1)

        // Sum the per-codebook embeddings into one sequence.
        var h = MLXArray.zeros([1, steps, Self.modelDimension], dtype: .float32)
        for codebook in 0 ..< Self.codebookCount {
            let ids = tokens[codebook, 0...]
            h = h + take(w("model.decoder.embed_tokens.\(codebook).weight"), ids, axis: 0)
                .reshaped(1, steps, Self.modelDimension)
        }

        // Sinusoidal table stored in the checkpoint; sliced at the offset so a
        // cached step gets the position it actually occupies.
        let positions = w("model.decoder.embed_positions.weights")[offset ..< (offset + steps), 0...]
        h = h + positions.reshaped(1, steps, Self.modelDimension)

        let causal = causalMask(steps: steps, offset: offset)
        let crossBias = (1.0 - encoderMask.asType(.float32))
            .reshaped(1, 1, 1, encoderMask.dim(1)) * -1e9

        for layer in 0 ..< Self.layerCount {
            let prefix = "model.decoder.layers.\(layer)"
            h = h + attention(prefix: "\(prefix).self_attn",
                              x: layerNorm(h, prefix: "\(prefix).self_attn_layer_norm"),
                              memory: nil, bias: causal, cache: cache, layer: layer, isCross: false)
            h = h + attention(prefix: "\(prefix).encoder_attn",
                              x: layerNorm(h, prefix: "\(prefix).encoder_attn_layer_norm"),
                              memory: encoderHidden, bias: crossBias,
                              cache: cache, layer: layer, isCross: true)
            h = h + feedForward(prefix: prefix,
                                x: layerNorm(h, prefix: "\(prefix).final_layer_norm"))
        }
        h = layerNorm(h, prefix: "model.decoder.layer_norm")

        // One head per codebook, stacked back into (codebooks, steps, vocab).
        var heads: [MLXArray] = []
        for codebook in 0 ..< Self.codebookCount {
            heads.append(matmul(h, w("lm_heads.\(codebook).weight").T))
        }
        return concatenated(heads, axis: 0)
    }

    private func layerNorm(_ x: MLXArray, prefix: String) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let centred = x - mean
        let variance = (centred * centred).mean(axis: -1, keepDims: true)
        return centred * rsqrt(variance + Self.epsilon) * w("\(prefix).weight") + w("\(prefix).bias")
    }

    /// Self-attention when `memory` is nil, cross-attention otherwise.
    private func attention(prefix: String, x: MLXArray, memory: MLXArray?, bias: MLXArray,
                           cache: MGDecoderCache? = nil, layer: Int = 0,
                           isCross: Bool = false) -> MLXArray {
        let steps = x.dim(1)
        let source = memory ?? x
        let sourceLength = source.dim(1)

        func split(_ input: MLXArray, _ key: String, _ length: Int) -> MLXArray {
            matmul(input, w("\(prefix).\(key)_proj.weight").T)
                .reshaped(1, length, Self.headCount, Self.headDimension)
                .transposed(0, 2, 1, 3)
        }
        let q = split(x, "q", steps)
        var k: MLXArray
        var v: MLXArray
        if isCross, let cache, let cachedK = cache.crossKeys[layer], let cachedV = cache.crossValues[layer] {
            // Text conditioning never changes, so these are projected once.
            k = cachedK; v = cachedV
        } else {
            k = split(source, "k", sourceLength)
            v = split(source, "v", sourceLength)
            if isCross, let cache {
                cache.crossKeys[layer] = k; cache.crossValues[layer] = v
            } else if let cache {
                if let priorK = cache.selfKeys[layer], let priorV = cache.selfValues[layer] {
                    k = concatenated([priorK, k], axis: 2)
                    v = concatenated([priorV, v], axis: 2)
                }
                cache.selfKeys[layer] = k; cache.selfValues[layer] = v
            }
        }

        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * pow(Float(Self.headDimension), -0.5)
        scores = softmax(scores + bias, axis: -1)
        let out = matmul(scores, v)
            .transposed(0, 2, 1, 3)
            .reshaped(1, steps, Self.modelDimension)
        return matmul(out, w("\(prefix).out_proj.weight").T)
    }

    private func feedForward(prefix: String, x: MLXArray) -> MLXArray {
        let hidden = geluApproximate(matmul(x, w("\(prefix).fc1.weight").T))
        return matmul(hidden, w("\(prefix).fc2.weight").T)
    }

    private func causalMask(steps: Int, offset: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: steps * (steps + offset))
        for query in 0 ..< steps {
            for key in 0 ..< (steps + offset) where key > query + offset {
                values[query * (steps + offset) + key] = -1e9
            }
        }
        return MLXArray(values, [1, 1, steps, steps + offset])
    }
}

/// Exact erf-based GELU, which is what the checkpoint was trained with.
func geluApproximate(_ x: MLXArray) -> MLXArray {
    x * 0.5 * (1.0 + erf(x / 1.4142135623730951))
}
