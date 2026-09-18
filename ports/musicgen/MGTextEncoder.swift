import Foundation
import MLX

/// T5-base encoder, as MusicGen uses it to turn a prompt into conditioning.
///
/// Three things differ from an ordinary transformer encoder and each one
/// silently corrupts the output if missed:
///
/// * T5 does **not** scale attention scores by 1/sqrt(head dimension). The
///   scaling is folded into initialisation instead.
/// * Position is carried entirely by a bucketed relative bias added to the
///   scores, computed from block 0's table and shared by every later block.
///   There are no positional embeddings.
/// * Its "layer norm" is RMS norm — no mean subtraction and no bias.
struct MGTextEncoder {
    static let layerCount = 12
    static let headCount = 12
    static let headDimension = 64
    static let modelDimension = 768
    static let bucketCount = 32
    static let maxDistance = 128
    static let epsilon: Float = 1e-6

    let weights: [String: MLXArray]

    private func w(_ key: String) -> MLXArray {
        guard let value = weights["text_encoder.\(key)"] else {
            fatalError("missing T5 weight: text_encoder.\(key)")
        }
        return value.asType(.float32)
    }

    func callAsFunction(inputIDs: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let length = inputIDs.dim(1)
        var h = take(w("shared.weight"), inputIDs.reshaped(-1), axis: 0)
            .reshaped(1, length, Self.modelDimension)

        // Padding positions are pushed to -inf before the softmax. The bias is
        // additive, so it composes with the relative position bias.
        let maskBias = (1.0 - attentionMask.asType(.float32))
            .reshaped(1, 1, 1, length) * -1e9
        let positionBias = relativeAttentionBias(length: length) + maskBias

        for block in 0 ..< Self.layerCount {
            let prefix = "encoder.block.\(block)"
            h = h + selfAttention(prefix: "\(prefix).layer.0",
                                  x: rmsNorm(h, weight: w("\(prefix).layer.0.layer_norm.weight"),
                                             eps: Self.epsilon),
                                  bias: positionBias)
            h = h + feedForward(prefix: "\(prefix).layer.1",
                                x: rmsNorm(h, weight: w("\(prefix).layer.1.layer_norm.weight"),
                                           eps: Self.epsilon))
        }
        return rmsNorm(h, weight: w("encoder.final_layer_norm.weight"), eps: Self.epsilon)
    }

    private func selfAttention(prefix: String, x: MLXArray, bias: MLXArray) -> MLXArray {
        let length = x.dim(1)
        func heads(_ key: String) -> MLXArray {
            matmul(x, w("\(prefix).SelfAttention.\(key).weight").T)
                .reshaped(1, length, Self.headCount, Self.headDimension)
                .transposed(0, 2, 1, 3)
        }
        let q = heads("q"), k = heads("k"), v = heads("v")
        // No 1/sqrt(d) here — that is the T5 quirk.
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) + bias
        scores = softmax(scores, axis: -1)
        let out = matmul(scores, v)
            .transposed(0, 2, 1, 3)
            .reshaped(1, length, Self.modelDimension)
        return matmul(out, w("\(prefix).SelfAttention.o.weight").T)
    }

    private func feedForward(prefix: String, x: MLXArray) -> MLXArray {
        let hidden = maximum(matmul(x, w("\(prefix).DenseReluDense.wi.weight").T), 0)
        return matmul(hidden, w("\(prefix).DenseReluDense.wo.weight").T)
    }

    /// The (1, heads, L, L) bias table, looked up through T5's relative
    /// position buckets. Computed in Swift rather than with array ops because
    /// the bucket rule is branchy integer arithmetic and L is small.
    private func relativeAttentionBias(length: Int) -> MLXArray {
        var buckets = [Int32](repeating: 0, count: length * length)
        // Bidirectional: half the buckets are reserved for each direction.
        let half = Self.bucketCount / 2
        let maxExact = half / 2
        for query in 0 ..< length {
            for key in 0 ..< length {
                let relative = key - query
                var bucket = relative > 0 ? half : 0
                let distance = abs(relative)
                if distance < maxExact {
                    bucket += distance
                } else {
                    // Distances beyond maxExact share logarithmically wider
                    // buckets, so far-apart tokens are lumped together.
                    let scaled = Double(maxExact)
                        + log(Double(distance) / Double(maxExact))
                        / log(Double(Self.maxDistance) / Double(maxExact))
                        * Double(half - maxExact)
                    bucket += min(Int(scaled), half - 1)
                }
                buckets[query * length + key] = Int32(bucket)
            }
        }
        let table = w("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
        let indices = MLXArray(buckets, [length * length])
        return take(table, indices, axis: 0)
            .reshaped(length, length, Self.headCount)
            .transposed(2, 0, 1)
            .expandedDimensions(axis: 0)
    }
}

/// The 768 -> 1024 projection between the text encoder and the decoder.
func encoderToDecoder(_ x: MLXArray, weights: [String: MLXArray]) -> MLXArray {
    matmul(x, weights["enc_to_dec_proj.weight"]!.asType(.float32).T)
        + weights["enc_to_dec_proj.bias"]!.asType(.float32)
}
