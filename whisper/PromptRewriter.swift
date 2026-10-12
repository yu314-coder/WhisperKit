//  The Image tab's prompt rewriter: Qwen-Image 2.1's PE-T2I (Qwen, Qwen
//  RESEARCH LICENSE) — a fine-tuned Qwen3.5 9B — in its abliterated build.
//  Weights from ports/imagegen/export_rewriter_mlx.py. The gated delta rule's
//  Metal kernel is mlx-swift-lm's (MIT, Copyright (c) 2024 ml-explore).
//
//  Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c)
//  2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.

import Foundation
import MLX

/// Qwen3.5's language model, as the rewriter uses it: 32 layers, three in
/// four of them linear attention (a gated delta rule over a fixed-size
/// state), every fourth full attention with an output gate. 4-bit weights,
/// bfloat16 activations, the recurrent state in float32.
struct Qwen35 {
    private static let layers = 32
    private static let width = 4096
    private static let eps: Float = 1e-6
    // Full attention.
    private static let heads = 16
    private static let keyValueHeads = 4
    private static let headDimension = 256
    private static let rotaryDimensions = 64
    private static let ropeBase: Float = 10_000_000
    // Linear attention.
    private static let keyHeads = 16
    private static let valueHeads = 32
    private static let keyDimension = 128
    private static let valueDimension = 128
    private static let convolution = 4

    let weights: [String: MLXArray]

    /// What every layer carries from one call to the next.
    struct Cache {
        var offset = 0
        var keys: [Int: MLXArray] = [:]
        var values: [Int: MLXArray] = [:]
        /// The last three inputs of each linear layer's convolution.
        var convolution: [Int: MLXArray] = [:]
        /// Each linear layer's state, (1, 32, 128, 128) float32.
        var state: [Int: MLXArray] = [:]

        var arrays: [String: MLXArray] {
            var all: [String: MLXArray] = ["offset": MLXArray([Int32(offset)])]
            for (layer, a) in keys { all["k.\(layer)"] = a }
            for (layer, a) in values { all["v.\(layer)"] = a }
            for (layer, a) in convolution { all["c.\(layer)"] = a }
            for (layer, a) in state { all["s.\(layer)"] = a }
            return all
        }

        init() {}

        init?(arrays: [String: MLXArray]) {
            guard let offset = arrays["offset"] else { return nil }
            self.offset = Int(offset.item(Int32.self))
            for (name, a) in arrays {
                let parts = name.split(separator: ".")
                guard parts.count == 2, let layer = Int(parts[1]) else { continue }
                switch parts[0] {
                case "k": keys[layer] = a
                case "v": values[layer] = a
                case "c": convolution[layer] = a
                case "s": state[layer] = a
                default: break
                }
            }
        }
    }

    /// Runs `ids` after everything in `cache`, and returns the last
    /// position's logits, (vocabulary,) float32.
    func callAsFunction(_ ids: [Int32], cache: inout Cache, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let length = ids.count
        var x = embed(ids)
        let mask = length > 1 ? causalMask(length: length, offset: cache.offset) : nil
        for layer in 0 ..< Self.layers {
            if isCancelled() { throw CancellationError() }
            let name = "layers.\(layer)"
            let normed = rmsNorm(x, weight: weights["\(name).input_layernorm.weight"]!, eps: Self.eps)
            if (layer + 1) % 4 == 0 {
                x = x + attention(normed, "\(name).self_attn", layer: layer, mask: mask, cache: &cache)
            } else {
                x = x + linearAttention(normed, "\(name).linear_attn", layer: layer, cache: &cache)
            }
            let fed = rmsNorm(x, weight: weights["\(name).post_attention_layernorm.weight"]!, eps: Self.eps)
            let m = "\(name).mlp"
            x = x + linear(silu(linear(fed, "\(m).gate_proj")) * linear(fed, "\(m).up_proj"), "\(m).down_proj")
            // A long prompt is evaluated layer by layer, to keep its
            // activations few; one new token runs as a single graph, since
            // waiting on the GPU 32 times a token cost a third of the speed.
            if length > 1 { eval(x) }
        }
        cache.offset += length
        let last = rmsNorm(x[0..., (length - 1)...], weight: weights["norm.weight"]!, eps: Self.eps)
        return linear(last, "lm_head").reshaped(-1).asType(.float32)
    }

    // MARK: - Pieces

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let packed = weights["\(name).weight.wq"]!, scales = weights["\(name).weight.scales"]!
        let bits = packed.dim(-1) * 32 / (scales.dim(-1) * 64)
        return quantizedMatmul(x, packed, scales: scales, biases: weights["\(name).weight.biases"]!,
                               transpose: true, groupSize: 64, bits: bits)
    }

    /// Only the rows asked for are dequantized: the table is 248,320 × 4096.
    private func embed(_ ids: [Int32]) -> MLXArray {
        let key = "embed_tokens.weight"
        let rows = MLXArray(ids)
        let packed = weights["\(key).wq"]!, scales = weights["\(key).scales"]!
        let bits = packed.dim(-1) * 32 / (scales.dim(-1) * 64)
        return dequantized(take(packed, rows, axis: 0), scales: take(scales, rows, axis: 0),
                           biases: take(weights["\(key).biases"]!, rows, axis: 0), groupSize: 64, bits: bits)
            .asType(.bfloat16).expandedDimensions(axis: 0)
    }

    private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    private func causalMask(length: Int, offset: Int) -> MLXArray {
        let width = offset + length
        var values = [Float](repeating: 0, count: length * width)
        for query in 0 ..< length {
            for key in (offset + query + 1) ..< max(offset + query + 1, width) { values[query * width + key] = -1e9 }
        }
        return MLXArray(values, [1, 1, length, width]).asType(.bfloat16)
    }

    /// Grouped-query attention over 256-wide heads, rotary on their first 64
    /// dimensions, the output gated by the query projection's second half.
    private func attention(_ x: MLXArray, _ name: String, layer: Int, mask: MLXArray?,
                           cache: inout Cache) -> MLXArray {
        let length = x.dim(1)
        let d = Self.headDimension
        let projected = linear(x, "\(name).q_proj").reshaped(1, length, Self.heads, 2 * d)
        var q = projected[.ellipsis, 0 ..< d]
        let gate = projected[.ellipsis, d...].reshaped(1, length, Self.heads * d)
        var k = linear(x, "\(name).k_proj").reshaped(1, length, Self.keyValueHeads, d)
        var v = linear(x, "\(name).v_proj").reshaped(1, length, Self.keyValueHeads, d)
        q = rmsNorm(q, weight: weights["\(name).q_norm.weight"]!, eps: Self.eps).transposed(0, 2, 1, 3)
        k = rmsNorm(k, weight: weights["\(name).k_norm.weight"]!, eps: Self.eps).transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        q = RoPE(q, dimensions: Self.rotaryDimensions, traditional: false, base: Self.ropeBase, scale: 1,
                 offset: cache.offset)
        k = RoPE(k, dimensions: Self.rotaryDimensions, traditional: false, base: Self.ropeBase, scale: 1,
                 offset: cache.offset)
        if let previous = cache.keys[layer] {
            k = concatenated([previous, k], axis: 2)
            v = concatenated([cache.values[layer]!, v], axis: 2)
        }
        cache.keys[layer] = k
        cache.values[layer] = v
        let out = scaledDotProductAttention(queries: q, keys: k, values: v, scale: pow(Float(d), -0.5), mask: mask)
            .transposed(0, 2, 1, 3).reshaped(1, length, Self.heads * d)
        return linear(out * sigmoid(gate), "\(name).o_proj")
    }

    /// Gated DeltaNet: a short causal convolution over the projected
    /// queries, keys and values, then the gated delta rule — each head's
    /// (128 × 128) state decays by g, takes in β of each new key's error,
    /// and is read by the query — then an RMS norm gated by silu(z).
    private func linearAttention(_ x: MLXArray, _ name: String, layer: Int, cache: inout Cache) -> MLXArray {
        let length = x.dim(1)
        let keyWidth = Self.keyHeads * Self.keyDimension
        let valueWidth = Self.valueHeads * Self.valueDimension
        let channels = 2 * keyWidth + valueWidth
        let qkv = linear(x, "\(name).in_proj_qkv")
        let z = linear(x, "\(name).in_proj_z").reshaped(1, length, Self.valueHeads, Self.valueDimension)
        let b = linear(x, "\(name).in_proj_b")
        let a = linear(x, "\(name).in_proj_a")

        let history = cache.convolution[layer] ?? MLXArray.zeros([1, Self.convolution - 1, channels], dtype: qkv.dtype)
        let input = concatenated([history, qkv], axis: 1)
        cache.convolution[layer] = input[0..., (input.dim(1) - (Self.convolution - 1))...]
        let kernel = weights["\(name).conv1d.weight"]!.reshaped(channels, Self.convolution)
        var mixed = input[0..., 0 ..< length] * kernel[0..., 0]
        for tap in 1 ..< Self.convolution {
            mixed = mixed + input[0..., tap ..< (tap + length)] * kernel[0..., tap]
        }
        mixed = silu(mixed)

        var q = mixed[.ellipsis, 0 ..< keyWidth].reshaped(1, length, Self.keyHeads, Self.keyDimension)
        var k = mixed[.ellipsis, keyWidth ..< (2 * keyWidth)].reshaped(1, length, Self.keyHeads, Self.keyDimension)
        let v = mixed[.ellipsis, (2 * keyWidth)...].reshaped(1, length, Self.valueHeads, Self.valueDimension)
        // L2-normalized queries and keys, the readout scale folded into q.
        let inverse = pow(Float(Self.keyDimension), -0.5)
        let ones = MLXArray.ones([Self.keyDimension], dtype: q.dtype)
        q = (inverse * inverse) * rmsNorm(q, weight: ones, eps: 1e-6 * inverse * inverse)
        k = inverse * rmsNorm(k, weight: ones, eps: 1e-6 * inverse * inverse)

        let aLog = weights["\(name).A_log"]!.asType(.float32)
        let decay = exp(-exp(aLog) * logAddExp(a.asType(.float32) + weights["\(name).dt_bias"]!.asType(.float32), MLXArray(Float(0))))
        let beta = sigmoid(b)
        let state = cache.state[layer]
            ?? MLXArray.zeros([1, Self.valueHeads, Self.valueDimension, Self.keyDimension], dtype: .float32)
        let (y, next) = GatedDelta.update(q: q, k: k, v: v, g: decay, beta: beta, state: state)
        cache.state[layer] = next

        let normed = rmsNorm(y, weight: weights["\(name).norm.weight"]!, eps: Self.eps).asType(.float32)
        let gated = (silu(z.asType(.float32)) * normed).asType(x.dtype)
        return linear(gated.reshaped(1, length, valueWidth), "\(name).out_proj")
    }
}

/// The gated delta rule over a whole sequence in one Metal dispatch: one
/// threadgroup row per value dimension of each head, 32 threads sharing the
/// key dimension, the state kept in registers across time steps.
/// mlx-swift-lm's kernel (MIT), as mlx-lm ships it.
enum GatedDelta {
    private static let kernel = MLXFast.metalKernel(
        name: "gated_delta_step",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g, beta: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              float kv_mem = 0.0f;
              {
                // Kahan summation, kept under Metal's default fast math.
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                float kv_compensation = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * g_[hv_idx];
                  auto product = state[i] * k_[s_idx];
                  auto corrected = product - kv_compensation;
                  auto next_sum = kv_mem + corrected;
                  kv_compensation = (next_sum - kv_mem) - corrected;
                  kv_mem = next_sum;
                }
              }
              kv_mem = simd_sum(kv_mem);

              auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
              }
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
            """)

    /// - Parameters: q, k (1, T, Hk, Dk); v (1, T, Hv, Dv); g, beta (1, T, Hv);
    ///   state (1, Hv, Dv, Dk).
    /// - Returns: y (1, T, Hv, Dv) in q's type, and the new state.
    static func update(q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray,
                       state: MLXArray) -> (MLXArray, MLXArray) {
        let (batch, steps, keyHeads, keyDimension) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3))
        let (valueHeads, valueDimension) = (v.dim(2), v.dim(3))
        let outputs = kernel([q, k, v, g, beta, state, MLXArray(Int32(steps))],
                             template: [("InT", q.dtype), ("StT", state.dtype), ("Dk", keyDimension),
                                        ("Dv", valueDimension), ("Hk", keyHeads), ("Hv", valueHeads)],
                             grid: (32, valueDimension, batch * valueHeads),
                             threadGroup: (32, 4, 1),
                             outputShapes: [[batch, steps, valueHeads, valueDimension], state.shape],
                             outputDTypes: [q.dtype, state.dtype])
        return (outputs[0], outputs[1])
    }
}

/// Turns a short image request into the long, concrete English description
/// Qwen-Image was trained on, with Qwen's own rewriter — the abliterated
/// build, so it describes what's asked rather than steering away.
///
/// It runs without its thinking pass (the chat template's empty think
/// block): on the M4 the answer comes in about 400 tokens instead of 1,000
/// to 1,600, a third of the time, and still holds the user's names, text
/// and counts. Sampling is the model card's: temperature 1, top-p 0.95,
/// top-k 20. The 2,400-token system prompt's state is worked out once and
/// kept beside the weights, so each request reads only its own tokens.
struct PromptRewriter {
    enum File {
        static let weights = (1 ... 3).map { "rewriter_q4.part\($0).safetensors" }
        static let vocabulary = "rewriter_vocab.json"
        static let merges = "rewriter_merges.txt"
        static let specials = "rewriter_specials.json"
        static let system = "rewriter_system_prompt.txt"
        /// Written on first use, not downloaded.
        static let systemState = "rewriter_system_state.safetensors"
    }

    struct Result {
        let prompt: String
        /// The shape it would have chosen, such as "16:9".
        let ratio: String?
    }

    enum Failure: LocalizedError {
        case unreadable
        var errorDescription: String? { "The prompt rewriter's answer couldn't be read." }
    }

    let directory: URL
    private static let maximumTokens = 1200

    /// - Parameters:
    ///   - ratio: the picture's shape, such as "3:4", passed on as part of
    ///     the request so the description fits the canvas.
    ///   - progress: the number of tokens written so far.
    func rewrite(_ request: String, ratio: String, isCancelled: () -> Bool = { false },
                 progress: (Int) -> Void = { _ in }) throws -> Result {
        try StageMemory.withoutCache {
            let specials = try JSONDecoder().decode([String: Int32].self,
                                                    from: Data(contentsOf: url(File.specials)))
            let tokenizer = try ACETokenizer(vocabularyURL: url(File.vocabulary), mergesURL: url(File.merges),
                                             specials: specials, pattern: ACETokenizer.qwen35Pattern)
            let model = Qwen35(weights: try SanaPipeline.load(File.weights, from: directory))

            var cache = try systemState(model: model, tokenizer: tokenizer, isCancelled: isCancelled)
            let text = request.trimmingCharacters(in: .whitespacesAndNewlines)
            let turn = "<|im_start|>user\n\(text)\n\nAspect ratio: \(ratio)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            var logits = try prefill(tokenizer.encode(turn, appendEndOfText: false), model: model, cache: &cache,
                                     isCancelled: isCancelled)

            let stops = Set([specials["<|im_end|>"], specials["<|endoftext|>"]].compactMap { $0 })
            var generator = SystemRandomNumberGenerator()
            var written: [Int32] = []
            while written.count < Self.maximumTokens {
                if isCancelled() { throw CancellationError() }
                let next = Self.sample(logits, using: &generator)
                if stops.contains(next) { break }
                written.append(next)
                progress(written.count)
                logits = try model([next], cache: &cache, isCancelled: isCancelled)
            }
            return try Self.parse(tokenizer.decode(written))
        }
    }

    private func url(_ name: String) -> URL { directory.appendingPathComponent(name) }

    /// The cache after the system turn: from disk, or worked out and saved.
    private func systemState(model: Qwen35, tokenizer: ACETokenizer, isCancelled: () -> Bool) throws -> Qwen35.Cache {
        let saved = url(File.systemState)
        if let arrays = try? loadArrays(url: saved), let cache = Qwen35.Cache(arrays: arrays) {
            return cache
        }
        let system = try String(contentsOf: url(File.system), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var cache = Qwen35.Cache()
        _ = try prefill(tokenizer.encode("<|im_start|>system\n\(system)<|im_end|>\n", appendEndOfText: false),
                        model: model, cache: &cache, isCancelled: isCancelled)
        try? save(arrays: cache.arrays, url: saved)
        return cache
    }

    /// Runs `ids` in chunks, so a long prompt never builds one huge mask or
    /// activation; returns the last position's logits.
    private func prefill(_ ids: [Int32], model: Qwen35, cache: inout Qwen35.Cache,
                         isCancelled: () -> Bool) throws -> MLXArray {
        var logits = MLXArray(0)
        for start in stride(from: 0, to: ids.count, by: 512) {
            logits = try model(Array(ids[start ..< min(start + 512, ids.count)]), cache: &cache, isCancelled: isCancelled)
            eval(logits)
        }
        return logits
    }

    /// Top-k 20, then top-p 0.95 of what's left, at temperature 1.
    private static func sample(_ logits: MLXArray, using generator: inout some RandomNumberGenerator) -> Int32 {
        let k = 20
        let candidates = argPartition(-logits, kth: k - 1)[0 ..< k]
        let ids = candidates.asType(.int32).asArray(Int32.self)
        let values = take(logits, candidates).asArray(Float.self)
        let order = values.indices.sorted { values[$0] > values[$1] }
        let top = values[order[0]]
        var weights = order.map { exp(Double(values[$0] - top)) }
        let total = weights.reduce(0, +)
        weights = weights.map { $0 / total }
        var kept = 0, mass = 0.0
        while kept < weights.count {
            mass += weights[kept]
            kept += 1
            if mass >= 0.95 { break }
        }
        var draw = Double.random(in: 0 ..< mass, using: &generator)
        for index in 0 ..< kept {
            draw -= weights[index]
            if draw < 0 { return ids[order[index]] }
        }
        return ids[order[kept - 1]]
    }

    /// `{"rewritten_prompt": "…", "wh_ratio": "3:2"}`, possibly with stray
    /// text around it.
    private static func parse(_ text: String) throws -> Result {
        guard let open = text.firstIndex(of: "{"), let close = text.lastIndex(of: "}") else { throw Failure.unreadable }
        let json = Data(text[open ... close].utf8)
        guard let object = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let prompt = (object["rewritten_prompt"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !prompt.isEmpty
        else { throw Failure.unreadable }
        return Result(prompt: prompt, ratio: object["wh_ratio"] as? String)
    }
}
