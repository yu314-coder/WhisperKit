//  Z-Image-Turbo (Tongyi-MAI, Apache 2.0), with an abliterated Qwen3-4B
//  (Apache 2.0) reading the prompt. Weights from
//  ports/imagegen/export_zimage_mlx.py.

import CoreGraphics
import Foundation
import MLX

/// Z-Image-Turbo's 6B single-stream transformer.
///
/// Image patches and the prompt are refined apart — two blocks each, only
/// the image's modulated by the timestep — then joined into one sequence,
/// image first, for thirty blocks. Each modulated block scales its inputs
/// by 1 + the modulation and gates its outputs by its tanh, with RMS norms
/// before and after both attention and the feed-forward. Rotary positions
/// run over three axes (sequence, row, column) with base 256: the prompt
/// counts along the first, the image sits after it there and spreads over
/// the other two.
///
/// bfloat16 throughout with 8-bit weights; rotary and the timestep in float32.
struct ZImageTransformer {
    static let latentChannels = 16
    private static let width = 3840
    private static let heads = 30
    private static let headDimension = 128
    private static let layers = 30
    private static let refiners = 2
    /// The prompt is padded to a multiple of this with a learned token.
    static let sequenceMultiple = 32

    let weights: [String: MLXArray]

    /// Rotary angles, cos and sin (positions, 64): the image's patches, then
    /// the padded prompt's tokens.
    struct Rotary {
        let cos: MLXArray
        let sin: MLXArray

        init(rows: Int, columns: Int, captionLength: Int) {
            let axes = [32, 48, 48]
            let frequencies = axes.map { d in (0 ..< d / 2).map { pow(256.0, -Double(2 * $0) / Double(d)) } }
            var positions: [[Double]] = []
            for row in 0 ..< rows {
                for column in 0 ..< columns {
                    positions.append([Double(captionLength + 1), Double(row), Double(column)])
                }
            }
            for index in 0 ..< captionLength { positions.append([Double(1 + index), 0, 0]) }
            var cosines: [Float] = [], sines: [Float] = []
            cosines.reserveCapacity(positions.count * 64); sines.reserveCapacity(positions.count * 64)
            for position in positions {
                for axis in 0 ..< 3 {
                    for frequency in frequencies[axis] {
                        let angle = position[axis] * frequency
                        cosines.append(Float(Foundation.cos(angle)))
                        sines.append(Float(Foundation.sin(angle)))
                    }
                }
            }
            cos = MLXArray(cosines, [positions.count, 64])
            sin = MLXArray(sines, [positions.count, 64])
        }
    }

    /// The model's output for `patches` (n, 64) at noise level `sigma`, given
    /// the prompt's states (tokens, 2560). Returns (n, 64), float32.
    func callAsFunction(_ patches: MLXArray, caption: MLXArray, sigma: Float, rotary: Rotary,
                        isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let imageCount = patches.dim(0)
        let time = timestepEmbedding((1 - sigma) * 1000)
        let adaln = linear(silu(linear(time, "t_embedder.mlp.0")), "t_embedder.mlp.2")

        // The image, refined on its own.
        var x = linear(patches.expandedDimensions(axis: 0).asType(.bfloat16), "all_x_embedder.2-1")
        let imageRotary = (rotary.cos[..<imageCount], rotary.sin[..<imageCount])
        for block in 0 ..< Self.refiners {
            x = try self.block(x, "noise_refiner.\(block)", rotary: imageRotary, adaln: adaln)
        }

        // The prompt: normed, projected, padded with the learned pad token,
        // refined on its own.
        let count = caption.dim(0)
        let padded = count + (Self.sequenceMultiple - count % Self.sequenceMultiple) % Self.sequenceMultiple
        var c = rmsNorm(caption.asType(.float32), weight: weights["cap_embedder.0.weight"]!.asType(.float32), eps: 1e-5)
        c = linear(c, "cap_embedder.1").asType(.bfloat16)
        if padded > count {
            let pad = broadcast(parameter("cap_pad_token").asType(.bfloat16), to: [padded - count, Self.width])
            c = concatenated([c, pad], axis: 0)
        }
        var cap = c.expandedDimensions(axis: 0)
        let captionRotary = (rotary.cos[imageCount...], rotary.sin[imageCount...])
        for block in 0 ..< Self.refiners {
            cap = try self.block(cap, "context_refiner.\(block)", rotary: captionRotary, adaln: nil)
        }

        // Together, image first.
        var h = concatenated([x, cap], axis: 1)
        for layer in 0 ..< Self.layers {
            if isCancelled() { throw CancellationError() }
            h = try block(h, "layers.\(layer)", rotary: (rotary.cos, rotary.sin), adaln: adaln)
        }

        let final = "all_final_layer.2-1"
        let scale = 1 + linear(silu(adaln), "\(final).adaLN_modulation.1").asType(.bfloat16)
        h = layerNorm(h[0..., ..<imageCount], eps: 1e-6) * scale.expandedDimensions(axis: 1)
        return linear(h, "\(final).linear")[0].asType(.float32)
    }

    // MARK: - Pieces

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let key = "\(name).weight"
        var y: MLXArray
        if let packed = weights["\(key).wq"] {
            y = quantizedMatmul(x, packed, scales: weights["\(key).scales"]!, biases: weights["\(key).biases"]!,
                                transpose: true, groupSize: 64, bits: 8)
        } else {
            y = matmul(x, weights[key]!.asType(x.dtype).T)
        }
        if let bias = weights["\(name).bias"] { y = y + bias.asType(y.dtype) }
        return y
    }

    private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    /// A learned tensor that isn't a matrix to multiply by — the prompt's pad
    /// token. The export packed it to 8 bits like the matrices, (1, 3840).
    private func parameter(_ name: String) -> MLXArray {
        if let packed = weights["\(name).wq"] {
            return dequantized(packed, scales: weights["\(name).scales"]!, biases: weights["\(name).biases"]!,
                               groupSize: 64, bits: 8)
        }
        return weights[name]!
    }

    private func norm(_ x: MLXArray, _ name: String, eps: Float = 1e-5) -> MLXArray {
        rmsNorm(x, weight: weights["\(name).weight"]!.asType(x.dtype), eps: eps)
    }

    private func block(_ x: MLXArray, _ name: String, rotary: (MLXArray, MLXArray), adaln: MLXArray?) throws -> MLXArray {
        let d = Self.width
        var scaleAttention: MLXArray? = nil, gateAttention: MLXArray? = nil
        var scaleFeed: MLXArray? = nil, gateFeed: MLXArray? = nil
        if let adaln {
            let mod = linear(adaln, "\(name).adaLN_modulation.0").asType(.bfloat16).expandedDimensions(axis: 1)
            scaleAttention = 1 + mod[.ellipsis, 0 ..< d]
            gateAttention = tanh(mod[.ellipsis, d ..< 2 * d])
            scaleFeed = 1 + mod[.ellipsis, (2 * d) ..< (3 * d)]
            gateFeed = tanh(mod[.ellipsis, (3 * d)...])
        }
        var input = norm(x, "\(name).attention_norm1")
        if let scaleAttention { input = input * scaleAttention }
        var attended = norm(attention(input, "\(name).attention", rotary: rotary), "\(name).attention_norm2")
        if let gateAttention { attended = gateAttention * attended }
        var h = x + attended

        input = norm(h, "\(name).ffn_norm1")
        if let scaleFeed { input = input * scaleFeed }
        let f = "\(name).feed_forward"
        let gate = linear(input, "\(f).w1")
        var fed = norm(linear(silu(gate) * linear(input, "\(f).w3"), "\(f).w2"), "\(name).ffn_norm2")
        if let gateFeed { fed = gateFeed * fed }
        h = h + fed
        eval(h)
        return h
    }

    private func attention(_ x: MLXArray, _ name: String, rotary: (MLXArray, MLXArray)) -> MLXArray {
        let n = x.dim(1)
        func heads(_ y: MLXArray) -> MLXArray {
            y.reshaped(1, n, Self.heads, Self.headDimension).transposed(0, 2, 1, 3)
        }
        let q = rotate(norm(heads(linear(x, "\(name).to_q")), "\(name).norm_q"), rotary)
        let k = rotate(norm(heads(linear(x, "\(name).to_k")), "\(name).norm_k"), rotary)
        let v = heads(linear(x, "\(name).to_v"))
        let out = scaledDotProductAttention(queries: q.asType(.bfloat16), keys: k.asType(.bfloat16), values: v,
                                            scale: pow(Float(Self.headDimension), -0.5), mask: nil)
            .transposed(0, 2, 1, 3).reshaped(1, n, Self.width)
        return linear(out, "\(name).to_out.0")
    }

    /// Each adjacent pair (2i, 2i + 1) as a complex number times e^{iθ}, in float32.
    private func rotate(_ x: MLXArray, _ rotary: (MLXArray, MLXArray)) -> MLXArray {
        let shape = x.shape
        let pairs = x.asType(.float32).reshaped(Array(shape.dropLast()) + [64, 2])
        let (x0, x1) = (pairs[.ellipsis, 0], pairs[.ellipsis, 1])
        let (c, s) = rotary
        return stacked([x0 * c - x1 * s, x0 * s + x1 * c], axis: -1).reshaped(shape)
    }

    /// Sinusoidal embedding of `t`, cosines first, (1, 256).
    private func timestepEmbedding(_ t: Float) -> MLXArray {
        let half = 128
        let values = (0 ..< half).map { Double(t) * exp(-log(10_000.0) * Double($0) / Double(half)) }
        return MLXArray(values.map { Float(cos($0)) } + values.map { Float(sin($0)) }, [1, 2 * half])
    }
}

/// Z-Image-Turbo: nine flow-matching steps, all on the GPU.
///
/// The prompt goes through Qwen's chat template and the abliterated Qwen3-4B;
/// Z-Image reads the states of its real tokens after layer 35 of 36. The
/// step-distilled model needs no guidance, so each step is one pass. Noise
/// levels are spaced evenly from 1 to 1/9 and shifted by 3 toward the noisy
/// end; the model's output is the negated velocity.
struct ZImagePipeline {
    enum File {
        static let vocabulary = "zimage_vocab.json"
        static let merges = "zimage_merges.txt"
        static let text = ["zimage_text_q8.part1.safetensors", "zimage_text_q8.part2.safetensors",
                           "zimage_text_q8.part3.safetensors"]
        static let transformer = ["zimage_transformer_q8.part1.safetensors", "zimage_transformer_q8.part2.safetensors",
                                  "zimage_transformer_q8.part3.safetensors", "zimage_transformer_q8.part4.safetensors"]
        static let vae = "zimage_vae.safetensors"
    }

    static let steps = 9
    private static let scalingFactor: Float = 0.3611
    private static let shiftFactor: Float = 0.1159

    let directory: URL

    static func sigmas(steps: Int = steps, shift: Double = 3) -> [Float] {
        (0 ..< steps).map { index in
            let s = 1 - Double(index) * (1 - 1 / Double(steps)) / Double(steps - 1)
            return Float(shift * s / (1 + (shift - 1) * s))
        } + [0]
    }

    /// - Parameter noise: starting latents (16, height/8, width/8) row-major,
    ///   instead of ones drawn from `seed` — for checking against the reference.
    func generate(prompt: String, width: Int, height: Int, seed: UInt64, noise: [Float]? = nil,
                  isCancelled: () -> Bool = { false },
                  progress: (SDPipeline.Stage) -> Void = { _ in }) throws -> CGImage {
        progress(.readingPrompt)
        let caption = try StageMemory.withoutCache { () throws -> MLXArray in
            let tokenizer = try ACETokenizer(vocabularyURL: directory.appendingPathComponent(File.vocabulary),
                                             mergesURL: directory.appendingPathComponent(File.merges))
            let text = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
            let ids = Array(tokenizer.encode(text, appendEndOfText: false).prefix(512))
            var encoder = ACEQwen3(weights: try SanaPipeline.load(File.text, from: directory),
                                   config: .zimageText, prefix: "")
            encoder.quantizationBits = 8
            let states = try encoder.states(inputIDs: MLXArray(ids).reshaped(1, -1), valid: ids.count,
                                            after: [35], isCancelled: isCancelled)
            let caption = states[0][0].asType(.bfloat16)
            eval(caption)
            return caption
        }

        let (rows, columns) = (height / 16, width / 16)
        let channels = ZImageTransformer.latentChannels
        var random = SeededNormal(seed: seed)
        let start = MLXArray(noise ?? random.sample(count: channels * rows * 2 * columns * 2),
                             [channels, rows * 2, columns * 2])
        // Into patches: (16, h, w) to (rows × columns, 2 × 2 × 16).
        var latents = start.reshaped(channels, rows, 2, columns, 2).transposed(1, 3, 2, 4, 0)
            .reshaped(rows * columns, 4 * channels)
        let sigmas = Self.sigmas()
        let padded = caption.dim(0) + (ZImageTransformer.sequenceMultiple - caption.dim(0) % ZImageTransformer.sequenceMultiple)
            % ZImageTransformer.sequenceMultiple
        try StageMemory.withoutCache {
            let transformer = ZImageTransformer(weights: try SanaPipeline.load(File.transformer, from: directory))
            let rotary = ZImageTransformer.Rotary(rows: rows, columns: columns, captionLength: padded)
            for step in 0 ..< Self.steps {
                if isCancelled() { throw CancellationError() }
                progress(.step(step + 1, of: Self.steps))
                let out = try transformer(latents, caption: caption, sigma: sigmas[step], rotary: rotary,
                                          isCancelled: isCancelled)
                latents = latents - (sigmas[step + 1] - sigmas[step]) * out
                eval(latents)
            }
        }
        if isCancelled() { throw CancellationError() }

        progress(.decoding)
        let planar = try StageMemory.withoutCache { () throws -> [Float] in
            let decoder = FluxVAEDecoder(weights: try MappedWeights.load(url: directory.appendingPathComponent(File.vae)),
                                         dtype: .bfloat16)
            let z = (latents.reshaped(rows, columns, 2, 2, channels).transposed(0, 2, 1, 3, 4)
                .reshaped(1, rows * 2, columns * 2, channels) / Self.scalingFactor + Self.shiftFactor)
            return try decoder.decode(z, isCancelled: isCancelled).asArray(Float.self)
        }
        return try RGBImage.make(planar: planar, width: width, height: height)
    }
}
