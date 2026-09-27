//  Ported from ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5), MIT.
//  Verified against the PyTorch reference stage by stage; see ports/README.md.

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
    /// 0 when dense; 8 when the projections are packed to int8.
    var quantizationBits: Int = 0
    var quantizationGroup: Int = 64
    /// What activations are carried in. Quantized scales must match it.
    var dtype: DType = .float32

    private func linear(_ x: MLXArray, _ key: String) -> MLXArray {
        let full = prefix.isEmpty ? key : "\(prefix).\(key)"
        if quantizationBits > 0, let packed = weights["\(full).wq"] {
            return quantizedMatmul(x, packed,
                                   scales: weights["\(full).scales"]!,
                                   biases: weights["\(full).biases"]!,
                                   transpose: true,
                                   groupSize: quantizationGroup,
                                   bits: quantizationBits)
        }
        return matmul(x, w(key).T)
    }

    private func w(_ key: String) -> MLXArray {
        guard let value = weights[prefix.isEmpty ? key : "\(prefix).\(key)"] else {
            fatalError("missing Qwen3 weight: \(key)")
        }
        return value.asType(dtype)
    }

    /// Looks up token embeddings without the transformer — ACE-Step feeds
    /// lyrics to its own lyric encoder this way, as raw table rows.
    ///
    /// Only the requested rows are dequantized. The table is 151,669 rows of
    /// 1024, so expanding all of it to float32 for a lookup cost 620 MB of
    /// transient memory for what is, at most, a couple of thousand rows.
    func embed(inputIDs: MLXArray) -> MLXArray {
        let key = prefix.isEmpty ? "embed_tokens.weight" : "\(prefix).embed_tokens.weight"
        let ids = inputIDs.reshaped(-1)
        let rows: MLXArray
        if quantizationBits > 0, let packed = weights["\(key).wq"] {
            rows = dequantized(take(packed, ids, axis: 0),
                               scales: take(weights["\(key).scales"]!, ids, axis: 0),
                               biases: take(weights["\(key).biases"]!, ids, axis: 0),
                               groupSize: quantizationGroup,
                               bits: quantizationBits)
        } else {
            rows = take(weights[key]!, ids, axis: 0)
        }
        return rows.asType(dtype).reshaped(inputIDs.dim(0), inputIDs.dim(1), config.hiddenSize)
    }

    func callAsFunction(inputIDs: MLXArray) -> MLXArray {
        let length = inputIDs.dim(1)
        var h = embed(inputIDs: inputIDs)

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
            // One layer at a time. Dense float16 weights are widened to
            // float32 as they are used, and left as one lazy graph MLX
            // widened all 28 layers' worth together — 600 MB for a model
            // that runs once, on a few hundred tokens.
            eval(h)
        }
        return rmsNorm(h, weight: w("norm.weight"), eps: config.epsilon)
    }

    // MARK: - Incremental decoding

    /// Runs `inputIDs` on top of everything `cache` has already seen and
    /// returns the final hidden state of each new position.
    ///
    /// - Parameter mask: additive, (batch or 1, 1, new, seen + new); causal
    ///   when nil. Batched callers pass their own to hide padding.
    func callAsFunction(inputIDs: MLXArray, cache: ACEKVCache, mask explicit: MLXArray? = nil) -> MLXArray {
        let length = inputIDs.dim(1)
        let offset = cache.offset
        var h = embed(inputIDs: inputIDs)
        let mask: MLXArray? = explicit ?? (length > 1 ? causalMask(length: length, offset: offset) : nil)
        for layer in 0 ..< config.layerCount {
            let prefix = "layers.\(layer)"
            h = h + attention(prefix: "\(prefix).self_attn",
                              x: rmsNorm(h, weight: w("\(prefix).input_layernorm.weight"),
                                         eps: config.epsilon),
                              mask: mask, cache: (cache, layer), offset: offset)
            h = h + feedForward(prefix: "\(prefix).mlp",
                                x: rmsNorm(h, weight: w("\(prefix).post_attention_layernorm.weight"),
                                           eps: config.epsilon))
            // Prompts and forced runs a layer at a time: as one graph, a
            // 123-token run held every layer's intermediates at once, 430 MB.
            // Single-token steps are small and stay one graph — a sync per
            // layer would cost more than it saves.
            if length > 1 { eval(h) }
        }
        cache.advance(by: length)
        return rmsNorm(h, weight: w("norm.weight"), eps: config.epsilon)
    }

    /// Next-token scores for `rows` of the vocabulary, through the tied
    /// embedding. Scoring only the rows that may be chosen — 64,000 music
    /// tokens out of 217,204 — is most of the planner's per-token work saved.
    func logits(_ hidden: MLXArray, rows: Range<Int>) -> MLXArray {
        let key = prefix.isEmpty ? "embed_tokens.weight" : "\(prefix).embed_tokens.weight"
        if quantizationBits > 0, let packed = weights["\(key).wq"] {
            return quantizedMatmul(hidden, packed[rows],
                                   scales: weights["\(key).scales"]![rows],
                                   biases: weights["\(key).biases"]![rows],
                                   transpose: true, groupSize: quantizationGroup, bits: quantizationBits)
        }
        return matmul(hidden, weights[key]![rows].asType(dtype).T)
    }

    private func attention(prefix: String, x: MLXArray, mask: MLXArray?,
                           cache: (ACEKVCache, Int)? = nil, offset: Int = 0) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

        func project(_ key: String, heads: Int) -> MLXArray {
            linear(x, "\(prefix).\(key).weight")
                .reshaped(batch, length, heads, config.headDimension)
                .transposed(0, 2, 1, 3)
        }
        var q = project("q_proj", heads: config.headCount)
        var k = project("k_proj", heads: config.keyValueHeadCount)
        let v = project("v_proj", heads: config.keyValueHeadCount)

        // Norms apply per head, before the rotary embedding.
        q = rmsNorm(q, weight: w("\(prefix).q_norm.weight"), eps: config.epsilon)
        k = rmsNorm(k, weight: w("\(prefix).k_norm.weight"), eps: config.epsilon)
        q = RoPE(q, dimensions: config.headDimension, traditional: false,
                 base: config.ropeTheta, scale: 1, offset: offset)
        k = RoPE(k, dimensions: config.headDimension, traditional: false,
                 base: config.ropeTheta, scale: 1, offset: offset)

        // Grouped-query attention: the fused kernel pairs each key/value head
        // with its query group itself, so k and v are passed untiled.
        let scale = pow(Float(config.headDimension), -0.5)
        let attended: MLXArray
        if let (cache, layer) = cache {
            // The cache is half precision — a six-minute song is 2,000-odd
            // positions over 28 layers, twice over for guidance — so the
            // attention runs in it too, with MLX's softmax in float32.
            let (keys, values) = cache.update(layer: layer, keys: k, values: v)
            attended = scaledDotProductAttention(queries: q.asType(.float16), keys: keys, values: values,
                                                 scale: scale, mask: mask?.asType(.float16))
                .asType(dtype)
        } else {
            attended = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        }
        let out = attended
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, config.headCount * config.headDimension)
        return linear(out, "\(prefix).o_proj.weight")
    }

    private func feedForward(prefix: String, x: MLXArray) -> MLXArray {
        let gate = linear(x, "\(prefix).gate_proj.weight")
        let up = linear(x, "\(prefix).up_proj.weight")
        return linear(gate * sigmoid(gate) * up, "\(prefix).down_proj.weight")
    }

    /// Each of `length` new positions sees everything before it, including
    /// the `offset` positions already in the cache.
    private func causalMask(length: Int, offset: Int = 0) -> MLXArray {
        let width = offset + length
        var values = [Float](repeating: 0, count: length * width)
        for query in 0 ..< length {
            for key in (offset + query + 1) ..< width { values[query * width + key] = -1e9 }
        }
        return MLXArray(values, [1, 1, length, width]).asType(dtype)
    }
}

/// Keys and values of every position a sequence has seen, so each new token
/// costs one position rather than the whole prefix again.
///
/// Allocated once at its full length and written in place, as mlx-lm does;
/// concatenating per token would copy the whole cache every step.
final class ACEKVCache {
    private(set) var offset = 0
    private let capacity: Int
    private var keys: [Int: MLXArray] = [:]
    private var values: [Int: MLXArray] = [:]

    init(capacity: Int) { self.capacity = capacity }

    /// Stores (1, heads, n, dim) new keys and values for `layer` and returns
    /// everything up to them.
    func update(layer: Int, keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let count = newKeys.dim(2)
        precondition(offset + count <= capacity, "planner cache overflow")
        if keys[layer] == nil {
            let shape = [newKeys.dim(0), newKeys.dim(1), capacity, newKeys.dim(3)]
            keys[layer] = MLXArray.zeros(shape, dtype: .float16)
            values[layer] = MLXArray.zeros(shape, dtype: .float16)
        }
        keys[layer]![0..., 0..., offset ..< offset + count, 0...] = newKeys.asType(.float16)
        values[layer]![0..., 0..., offset ..< offset + count, 0...] = newValues.asType(.float16)
        return (keys[layer]![0..., 0..., 0 ..< offset + count, 0...],
                values[layer]![0..., 0..., 0 ..< offset + count, 0...])
    }

    func advance(by count: Int) { offset += count }
}
