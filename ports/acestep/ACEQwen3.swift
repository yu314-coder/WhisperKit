import Foundation
import MLX

/// Qwen3, as ACE-Step uses it for text conditioning.
///
/// Two details differ from the transformers already ported here:
///
/// * **Head dimension is independent of hidden size.** 16 query heads of 128
///   over a 1024-wide model, so `q_proj` widens to 2048 and `o_proj` narrows
///   back. Deriving head size from hidden width, as most models allow, gives
///   the wrong shape.
/// * **Grouped-query attention with per-head norms.** Eight key/value heads
///   serve sixteen query heads, and `q_norm`/`k_norm` are RMS norms over the
///   head dimension applied *before* the rotary embedding.
struct ACEQwen3 {
    struct Config {
        let layerCount: Int
        let hiddenSize: Int
        let headCount: Int
        let keyValueHeadCount: Int
        let headDimension: Int
        let ropeTheta: Float
        let epsilon: Float

        /// Qwen3-Embedding-0.6B, ACE-Step's text encoder.
        static let embedder = Config(layerCount: 28, hiddenSize: 1024, headCount: 16,
                                     keyValueHeadCount: 8, headDimension: 128,
                                     ropeTheta: 1_000_000, epsilon: 1e-6)
        /// acestep-5Hz-lm-1.7B, the planner.
        static let planner = Config(layerCount: 28, hiddenSize: 2048, headCount: 16,
                                    keyValueHeadCount: 8, headDimension: 128,
                                    ropeTheta: 1_000_000, epsilon: 1e-6)
    }

    let weights: [String: MLXArray]
    let config: Config
    let prefix: String

    private func w(_ key: String) -> MLXArray {
        guard let value = weights[prefix.isEmpty ? key : "\(prefix).\(key)"] else {
            fatalError("missing Qwen3 weight: \(key)")
        }
        return value.asType(.float32)
    }

    func callAsFunction(inputIDs: MLXArray) -> MLXArray {
        let length = inputIDs.dim(1)
        var h = take(w("embed_tokens.weight"), inputIDs.reshaped(-1), axis: 0)
            .reshaped(1, length, config.hiddenSize)

        let mask = causalMask(length: length)
        for layer in 0 ..< config.layerCount {
            let prefix = "layers.\(layer)"
            h = h + attention(prefix: "\(prefix).self_attn",
                              x: rmsNorm(h, weight: w("\(prefix).input_layernorm.weight"),
                                         eps: config.epsilon),
                              mask: mask)
            h = h + feedForward(prefix: "\(prefix).mlp",
                                x: rmsNorm(h, weight: w("\(prefix).post_attention_layernorm.weight"),
                                           eps: config.epsilon))
        }
        return rmsNorm(h, weight: w("norm.weight"), eps: config.epsilon)
    }

    private func attention(prefix: String, x: MLXArray, mask: MLXArray) -> MLXArray {
        let length = x.dim(1)

        func project(_ key: String, heads: Int) -> MLXArray {
            matmul(x, w("\(prefix).\(key).weight").T)
                .reshaped(1, length, heads, config.headDimension)
                .transposed(0, 2, 1, 3)
        }
        var q = project("q_proj", heads: config.headCount)
        var k = project("k_proj", heads: config.keyValueHeadCount)
        let v = project("v_proj", heads: config.keyValueHeadCount)

        // Norms apply per head, before the rotary embedding.
        q = rmsNorm(q, weight: w("\(prefix).q_norm.weight"), eps: config.epsilon)
        k = rmsNorm(k, weight: w("\(prefix).k_norm.weight"), eps: config.epsilon)
        q = RoPE(q, dimensions: config.headDimension, traditional: false,
                 base: config.ropeTheta, scale: 1, offset: 0)
        k = RoPE(k, dimensions: config.headDimension, traditional: false,
                 base: config.ropeTheta, scale: 1, offset: 0)

        // Grouped-query attention: each key/value head serves several queries.
        let repeats = config.headCount / config.keyValueHeadCount
        let kExpanded = repeated(k, count: repeats, axis: 1)
        let vExpanded = repeated(v, count: repeats, axis: 1)

        var scores = matmul(q, kExpanded.transposed(0, 1, 3, 2))
            * pow(Float(config.headDimension), -0.5)
        scores = softmax(scores + mask, axis: -1)
        let out = matmul(scores, vExpanded)
            .transposed(0, 2, 1, 3)
            .reshaped(1, length, config.headCount * config.headDimension)
        return matmul(out, w("\(prefix).o_proj.weight").T)
    }

    private func feedForward(prefix: String, x: MLXArray) -> MLXArray {
        let gate = matmul(x, w("\(prefix).gate_proj.weight").T)
        let up = matmul(x, w("\(prefix).up_proj.weight").T)
        return matmul(gate * sigmoid(gate) * up, w("\(prefix).down_proj.weight").T)
    }

    /// Repeats each head `count` times along `axis`, matching how grouped
    /// key/value heads are broadcast across their query group.
    private func repeated(_ x: MLXArray, count: Int, axis: Int) -> MLXArray {
        guard count > 1 else { return x }
        let shape = x.shape
        // Interleaved per head — [h0, h0, h1, h1, ...] — which is what the
        // key/value grouping means. Repeating the whole tensor instead would
        // pair every query head with the wrong key.
        let expanded = broadcast(x.expandedDimensions(axis: axis + 1),
                                 to: [shape[0], shape[1], count, shape[2], shape[3]])
        return expanded.reshaped(shape[0], shape[1] * count, shape[2], shape[3])
    }

    private func causalMask(length: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: length * length)
        for query in 0 ..< length {
            for key in (query + 1) ..< length { values[query * length + key] = -1e9 }
        }
        return MLXArray(values, [1, 1, length, length])
    }
}
