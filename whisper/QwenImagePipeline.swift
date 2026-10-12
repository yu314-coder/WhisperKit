//  Qwen-Image 2.1 Turbo (Qwen, Qwen RESEARCH LICENSE), with an abliterated
//  Qwen3-VL-8B (Apache 2.0) reading the prompt. Weights from
//  ports/imagegen/export_qwenimage_mlx.py.
//
//  Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c)
//  2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.

import CoreGraphics
import Foundation
import MLX

/// Qwen-Image 2.1's 7B single-stream transformer.
///
/// The prompt and the picture share one sequence, prompt first. Attention is
/// block-causal: prompt tokens see only the tokens before them, the
/// picture's see everything. Prompt tokens are modulated as if at t = 0, so
/// they never depend on the step or on the picture — they run once, and
/// each step only the picture's tokens run, attending to the prompt's cached
/// keys and values. That is upstream's prefix cache, and it is exact.
///
/// One modulation, from the timestep, serves every block: each scales its
/// layer-normed inputs by 1 + scale and gates its outputs by tanh(gate).
/// Rotary positions run over three axes (16, 56 and 56 dimensions, base
/// 10,000): prompt tokens count along all three together; the picture sits
/// at the next position on the first and spreads over the other two,
/// centred on zero.
///
/// bfloat16 throughout with 8-bit weights; rotary and the timestep in float32.
struct QwenImageTransformer {
    static let latentChannels = 64
    private static let width = 4096
    private static let heads = 32
    private static let headDimension = 128
    private static let layers = 32

    let weights: [String: MLXArray]

    /// Rotary angles, cos and sin (positions, 64).
    struct Rotary {
        let cos: MLXArray
        let sin: MLXArray

        private static let frequencies = [16, 56, 56].map { d in
            (0 ..< d / 2).map { pow(10_000.0, -Double(2 * $0) / Double(d)) }
        }

        init(positions: [[Double]]) {
            var cosines: [Float] = [], sines: [Float] = []
            cosines.reserveCapacity(positions.count * 64); sines.reserveCapacity(positions.count * 64)
            for position in positions {
                for axis in 0 ..< 3 {
                    for frequency in Self.frequencies[axis] {
                        let angle = position[axis] * frequency
                        cosines.append(Float(Foundation.cos(angle)))
                        sines.append(Float(Foundation.sin(angle)))
                    }
                }
            }
            cos = MLXArray(cosines, [positions.count, 64])
            sin = MLXArray(sines, [positions.count, 64])
        }

        static func prompt(length: Int) -> Rotary {
            Rotary(positions: (0 ..< length).map { [Double($0), Double($0), Double($0)] })
        }

        static func picture(rows: Int, columns: Int, after promptLength: Int) -> Rotary {
            var positions: [[Double]] = []
            positions.reserveCapacity(rows * columns)
            for row in 0 ..< rows {
                for column in 0 ..< columns {
                    positions.append([Double(promptLength), Double(row - (rows - rows / 2)),
                                      Double(column - (columns - columns / 2))])
                }
            }
            return Rotary(positions: positions)
        }
    }

    /// Every block's keys and values for the prompt, (1, 32, tokens, 128).
    typealias Prefix = [(keys: MLXArray, values: MLXArray)]

    /// Runs the prompt's states (tokens, 4096) through every block once.
    func prefix(_ caption: MLXArray, isCancelled: () -> Bool = { false }) throws -> Prefix {
        let length = caption.dim(0)
        let time = timeEmbedding(sigma: 0)
        let modulation = linear(silu(time), "modulation.1")

        let normed = rmsNorm(caption.asType(.float32),
                             weight: weights["txt_in.text_norm.weight"]!.asType(.float32) + 1, eps: 1e-6)
        var x = linear(gelu(linear(normed.asType(.bfloat16), "txt_in.in_layer")), "txt_in.out_layer")
            .expandedDimensions(axis: 0)

        var values = [Float](repeating: 0, count: length * length)
        for query in 0 ..< length {
            for key in (query + 1) ..< max(query + 1, length) { values[query * length + key] = -1e9 }
        }
        let mask = MLXArray(values, [1, 1, length, length]).asType(.bfloat16)
        let rotary = Rotary.prompt(length: length)

        var prefix: Prefix = []
        for layer in 0 ..< Self.layers {
            if isCancelled() { throw CancellationError() }
            let (h, keys, values) = block(x, "transformer_blocks.\(layer)", modulation: modulation,
                                          rotary: rotary, mask: mask, prefix: nil)
            eval(h, keys, values)
            prefix.append((keys, values))
            x = h
        }
        return prefix
    }

    /// The model's velocity for `latents` (n, 64) at noise level `sigma`.
    /// Returns (n, 64), float32.
    func callAsFunction(_ latents: MLXArray, sigma: Float, prefix: Prefix, rotary: Rotary,
                        isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let time = timeEmbedding(sigma: sigma)
        let modulation = linear(silu(time), "modulation.1")
        var x = linear(latents.asType(.bfloat16), "img_in").expandedDimensions(axis: 0)
        for layer in 0 ..< Self.layers {
            if isCancelled() { throw CancellationError() }
            x = block(x, "transformer_blocks.\(layer)", modulation: modulation, rotary: rotary, mask: nil,
                      prefix: prefix[layer]).0
            eval(x)
        }
        let scale = 1 + linear(silu(time), "norm_out.linear")
        x = layerNorm(x, eps: 1e-6) * scale.expandedDimensions(axis: 1)
        return linear(x, "proj_out")[0].asType(.float32)
    }

    // MARK: - Pieces

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let key = "\(name).weight"
        if let packed = weights["\(key).wq"], let scales = weights["\(key).scales"] {
            // 8 or 4 bits, read off the shapes: each 32-bit word packs 32 / bits weights.
            let bits = packed.dim(-1) * 32 / (scales.dim(-1) * 64)
            return quantizedMatmul(x, packed, scales: scales, biases: weights["\(key).biases"]!,
                                   transpose: true, groupSize: 64, bits: bits)
        }
        return matmul(x, weights[key]!.asType(x.dtype).T)
    }

    private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    private func gelu(_ x: MLXArray) -> MLXArray {
        0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    }

    private func block(_ x: MLXArray, _ name: String, modulation: MLXArray, rotary: Rotary, mask: MLXArray?,
                       prefix: (keys: MLXArray, values: MLXArray)?) -> (MLXArray, MLXArray, MLXArray) {
        let d = Self.width
        let mod = modulation.expandedDimensions(axis: 1)
        let scaleAttention = 1 + mod[.ellipsis, 0 ..< d]
        let gateAttention = tanh(mod[.ellipsis, d ..< 2 * d])
        let scaleFeed = 1 + mod[.ellipsis, (2 * d) ..< (3 * d)]
        let gateFeed = tanh(mod[.ellipsis, (3 * d)...])

        let n = x.dim(1)
        func heads(_ y: MLXArray) -> MLXArray {
            y.reshaped(1, n, Self.heads, Self.headDimension).transposed(0, 2, 1, 3)
        }
        let input = layerNorm(x, eps: 1e-6) * scaleAttention
        let a = "\(name).attn"
        let q = rotate(rmsNorm(heads(linear(input, "\(a).to_q")), weight: weights["\(a).norm_q.weight"]!, eps: 1e-6),
                       rotary).asType(.bfloat16)
        let k = rotate(rmsNorm(heads(linear(input, "\(a).to_k")), weight: weights["\(a).norm_k.weight"]!, eps: 1e-6),
                       rotary).asType(.bfloat16)
        let v = heads(linear(input, "\(a).to_v"))
        let keys = prefix.map { concatenated([$0.keys, k], axis: 2) } ?? k
        let values = prefix.map { concatenated([$0.values, v], axis: 2) } ?? v
        let attended = scaledDotProductAttention(queries: q, keys: keys, values: values,
                                                 scale: pow(Float(Self.headDimension), -0.5), mask: mask)
            .transposed(0, 2, 1, 3).reshaped(1, n, d)
        var h = x + gateAttention * linear(attended, "\(a).to_out.0")

        let fed = layerNorm(h, eps: 1e-6) * scaleFeed
        let f = "\(name).img_mlp"
        h = h + gateFeed * linear(silu(linear(fed, "\(f).gate_layer")) * linear(fed, "\(f).proj"), "\(f).out")
        return (h, k, v)
    }

    /// Each adjacent pair (2i, 2i + 1) as a complex number times e^{iθ}, in float32.
    private func rotate(_ x: MLXArray, _ rotary: Rotary) -> MLXArray {
        let shape = x.shape
        let pairs = x.asType(.float32).reshaped(Array(shape.dropLast()) + [64, 2])
        let (x0, x1) = (pairs[.ellipsis, 0], pairs[.ellipsis, 1])
        let (c, s) = (rotary.cos, rotary.sin)
        return stacked([x0 * c - x1 * s, x0 * s + x1 * c], axis: -1).reshaped(shape)
    }

    /// The timestep's embedding, (1, 4096) in bfloat16. Upstream carries the
    /// timestep in bfloat16 on its way in — as t, then as t / 1000 — and the
    /// model learned on those rounded values, so they are rounded here too.
    private func timeEmbedding(sigma: Float) -> MLXArray {
        let t = Double(Self.bfloat16(Self.bfloat16(sigma * 1000) / 1000)) * 1000
        let half = 128
        let angles = (0 ..< half).map { t * exp(-log(10_000.0) * Double($0) / Double(half)) }
        let sinusoid = MLXArray(angles.map { Float(cos($0)) } + angles.map { Float(sin($0)) }, [1, 2 * half])
            .asType(.bfloat16)
        let e = "time_text_embed.timestep_embedder"
        return linear(silu(linear(sinusoid, "\(e).linear_1")), "\(e).linear_2")
    }

    /// Rounds to the nearest bfloat16, ties to even.
    private static func bfloat16(_ x: Float) -> Float {
        let bits = x.bitPattern
        return Float(bitPattern: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) & 0xFFFF_0000)
    }
}

/// Qwen-Image 2.1 Turbo: eight flow-matching steps, all on the GPU.
///
/// The prompt goes through Qwen's chat template, under a fixed system
/// message, and the abliterated Qwen3-VL-8B; the transformer reads the
/// last layer's states, before the final norm, of the tokens after the
/// system message. The step-distilled model needs no guidance, so each step
/// is one pass, at noise levels upstream ships with the model.
///
/// Two builds of the same model share a folder: 8-bit, and 4-bit at about
/// half the download — the prompt reader and the transformer each in its
/// own bits, the VAE decoder and tokenizer common to both.
struct QwenImagePipeline {
    enum File {
        static let vocabulary = "qwenimage_vocab.json"
        static let merges = "qwenimage_merges.txt"
        static func text(bits: Int) -> [String] {
            (1 ... (bits == 4 ? 3 : 5)).map { "qwenimage_text_q\(bits).part\($0).safetensors" }
        }
        static func transformer(bits: Int) -> [String] {
            (1 ... (bits == 4 ? 3 : 5)).map { "qwenimage_transformer_q\(bits).part\($0).safetensors" }
        }
        static let vae = "qwenimage_vae.safetensors"
    }

    static let sigmas: [Float] = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0]
    static var steps: Int { sigmas.count - 1 }
    private static let system = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"

    let directory: URL
    /// 8, or 4 for the smaller build.
    var bits = 8

    /// - Parameter noise: starting latents (64, height/16, width/16) row-major,
    ///   instead of ones drawn from `seed` — for checking against the reference.
    func generate(prompt: String, width: Int, height: Int, seed: UInt64, noise: [Float]? = nil,
                  isCancelled: () -> Bool = { false },
                  progress: (SDPipeline.Stage) -> Void = { _ in }) throws -> CGImage {
        progress(.readingPrompt)
        let caption = try StageMemory.withoutCache { () throws -> MLXArray in
            let tokenizer = try ACETokenizer(vocabularyURL: directory.appendingPathComponent(File.vocabulary),
                                             mergesURL: directory.appendingPathComponent(File.merges))
            let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? " " : prompt
            let dropped = tokenizer.encode(Self.system, appendEndOfText: false).count
            let ids = tokenizer.encode("\(Self.system)<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n",
                                       appendEndOfText: false)
            var encoder = ACEQwen3(weights: try SanaPipeline.load(File.text(bits: bits), from: directory),
                                   config: .qwenImageText, prefix: "")
            encoder.quantizationBits = bits
            let states = try encoder.states(inputIDs: MLXArray(ids).reshaped(1, -1), valid: ids.count,
                                            after: [36], isCancelled: isCancelled)
            let caption = states[0][0, dropped...].asType(.bfloat16)
            eval(caption)
            return caption
        }

        let (rows, columns) = (height / 16, width / 16)
        let channels = QwenImageTransformer.latentChannels
        var random = SeededNormal(seed: seed)
        var latents = MLXArray(noise ?? random.sample(count: channels * rows * columns), [channels, rows * columns])
            .transposed(1, 0)
        let sigmas = Self.sigmas
        try StageMemory.withoutCache {
            let transformer = QwenImageTransformer(weights: try SanaPipeline.load(File.transformer(bits: bits), from: directory))
            let prefix = try transformer.prefix(caption, isCancelled: isCancelled)
            let rotary = QwenImageTransformer.Rotary.picture(rows: rows, columns: columns, after: caption.dim(0))
            for step in 0 ..< Self.steps {
                if isCancelled() { throw CancellationError() }
                progress(.step(step + 1, of: Self.steps))
                let velocity = try transformer(latents, sigma: sigmas[step], prefix: prefix, rotary: rotary,
                                               isCancelled: isCancelled)
                latents = latents + (sigmas[step + 1] - sigmas[step]) * velocity
                eval(latents)
            }
        }
        if isCancelled() { throw CancellationError() }

        progress(.decoding)
        let planar = try StageMemory.withoutCache { () throws -> [Float] in
            let decoder = QwenImageVAEDecoder(weights: try MappedWeights.load(url: directory.appendingPathComponent(File.vae)))
            let rgba = try decoder(latents, rows: rows, columns: columns, isCancelled: isCancelled)
            // RGBA over white.
            let alpha = (rgba[3 ..< 4] + 1) / 2
            return (rgba[0 ..< 3] * alpha + (1 - alpha)).asArray(Float.self)
        }
        return try RGBImage.make(planar: planar, width: width, height: height)
    }
}
