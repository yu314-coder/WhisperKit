//  Ming-Image-0.2-Design (inclusionAI / Ant Group, MIT). Weights from
//  ports/imagegen/export_ming_mlx.py.

import CoreGraphics
import Foundation
import MLX

/// Ming's prompt reader: Ling-lite, a 20-layer mixture of experts (16.8B in
/// all, about 1B per token), reading the prompt and then 256 learned query
/// tokens set where an image would go.
///
/// Attention is grouped (16 query heads over 4 key/value heads of 128), with
/// RMS norms on queries and keys and rotary positions on the first half of
/// each head only — over three axes for the query tokens, which sit in one
/// row of a 1 × 256 grid. Every layer but the first is 256 experts, 8 chosen
/// per token by a sigmoid router (best 4 of 8 groups first) plus one shared;
/// the query tokens have a router of their own.
///
/// bfloat16 activations, 4-bit experts and 8-bit everything else; routing
/// in float32 as upstream.
struct MingTextReader {
    static let width = 2048
    private static let heads = 16
    private static let keyValueHeads = 4
    private static let headDimension = 128
    private static let rotaryDimension = 64
    private static let layers = 20
    private static let experts = 256
    private static let chosen = 8
    private static let groups = 8
    private static let chosenGroups = 4
    private static let routedScale: Float = 2.5

    static let patchToken: Int32 = 157157
    static let imageStart: Int32 = 157158
    static let imageEnd: Int32 = 157159
    static let queryCount = 256

    let weights: [String: MLXArray]

    struct States {
        /// After layers 5 and 12, and the normed last: (tokens, 2048) each.
        let after5: MLXArray
        let after12: MLXArray
        let final: MLXArray
    }

    /// Positions (temporal, row, column) of `count` prompt tokens followed by
    /// the image start, the 256 query tokens and the image end: the prompt
    /// counts along all three axes, the queries sit at its end on the first
    /// two and are centred on the third, and the end token follows the start.
    static func positions(promptTokens count: Int) -> [(Int, Int, Int)] {
        let n = count + 1
        var result = (0 ..< n).map { ($0, $0, $0) }
        for column in 0 ..< queryCount { result.append((n, n, n + column - (queryCount - 1) / 2)) }
        result.append((n + 1, n + 1, n + 1))
        return result
    }

    /// - Parameter trace: sees each layer's output, for checking against upstream.
    func callAsFunction(_ ids: [Int32], isCancelled: () -> Bool = { false },
                        trace: (Int, MLXArray) -> Void = { _, _ in }) throws -> States {
        let n = ids.count
        let isQuery = ids.map { $0 == Self.patchToken }
        let idArray = MLXArray(ids.map { Int32(min(Int($0), 157_183)) })
        var h = dequantized(weights["embed.wq"]![idArray], scales: weights["embed.scales"]![idArray],
                            biases: weights["embed.biases"]![idArray], groupSize: 64, bits: 8).asType(.bfloat16)
        if let first = isQuery.firstIndex(of: true) {
            h = concatenated([h[..<first], weights["query_tokens"]!.asType(.bfloat16),
                              h[(first + Self.queryCount)...]], axis: 0)
        }
        h = h.expandedDimensions(axis: 0)
        let queryMask = MLXArray(isQuery).reshaped(n, 1)
        let rotary = Self.rotary(Self.positions(promptTokens: (isQuery.firstIndex(of: true) ?? n + 1) - 1))
        let mask = MLXFast.ScaledDotProductAttentionMaskMode.causal

        var after5 = h, after12 = h
        for layer in 0 ..< Self.layers {
            if isCancelled() { throw CancellationError() }
            let p = "layers.\(layer)."
            let attended = attention(norm(h, p + "input_layernorm"), p + "attention.", rotary: rotary, mask: mask)
            h = h + attended
            let x = norm(h, p + "post_attention_layernorm")
            h = h + (layer == 0 ? mlp(x, p + "mlp.") : moe(x[0], p + "mlp.", queryMask: queryMask).expandedDimensions(axis: 0))
            eval(h)
            trace(layer, h[0])
            if layer == 4 { after5 = h }
            if layer == 11 { after12 = h }
        }
        let final = norm(h, "norm")
        eval(final)
        return States(after5: after5[0], after12: after12[0], final: final[0])
    }

    // MARK: - Pieces

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        quantizedMatmul(x, weights["\(name).wq"]!, scales: weights["\(name).scales"]!, biases: weights["\(name).biases"]!,
                        transpose: true, groupSize: 64, bits: 8)
    }

    private func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        rmsNorm(x, weight: weights["\(name).weight"]!, eps: 1e-6)
    }

    private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    private func mlp(_ x: MLXArray, _ name: String) -> MLXArray {
        linear(silu(linear(x, name + "gate_proj.weight")) * linear(x, name + "up_proj.weight"), name + "down_proj.weight")
    }

    /// cos and sin, (tokens, 64), laid out as upstream's "video_rope": of
    /// each half's 32 frequencies the first 24 alternate row and column, the
    /// last 8 are temporal.
    private static func rotary(_ positions: [(Int, Int, Int)]) -> (MLXArray, MLXArray) {
        let half = rotaryDimension / 2
        let inverse = (0 ..< half).map { pow(600_000.0, -Double(2 * $0) / Double(rotaryDimension)) }
        var cosines: [Float] = [], sines: [Float] = []
        for (t, row, column) in positions {
            var angles: [Double] = []
            for f in 0 ..< half {
                let position = f < 24 ? (f % 2 == 0 ? row : column) : t
                angles.append(Double(position) * inverse[f])
            }
            for angle in angles + angles {
                cosines.append(Float(Foundation.cos(angle)))
                sines.append(Float(Foundation.sin(angle)))
            }
        }
        let count = positions.count
        return (MLXArray(cosines, [count, rotaryDimension]).asType(.bfloat16),
                MLXArray(sines, [count, rotaryDimension]).asType(.bfloat16))
    }

    /// Rotary on the first 64 of each head's 128, in halves (rotate_half).
    private func rotate(_ x: MLXArray, _ rotary: (MLXArray, MLXArray)) -> MLXArray {
        let d = Self.rotaryDimension
        let r = x[.ellipsis, ..<d], pass = x[.ellipsis, d...]
        let rotated = concatenated([-r[.ellipsis, (d / 2)...], r[.ellipsis, ..<(d / 2)]], axis: -1)
        return concatenated([r * rotary.0 + rotated * rotary.1, pass], axis: -1)
    }

    private func attention(_ x: MLXArray, _ name: String, rotary: (MLXArray, MLXArray),
                           mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let n = x.dim(1)
        let (h, kv, d) = (Self.heads, Self.keyValueHeads, Self.headDimension)
        let qkv = linear(x, name + "query_key_value.weight").reshaped(1, n, h + 2 * kv, d)
        var q = qkv[0..., 0..., ..<h].transposed(0, 2, 1, 3)
        var k = qkv[0..., 0..., h ..< (h + kv)].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., (h + kv)...].transposed(0, 2, 1, 3)
        q = rotate(norm(q, name + "q_norm"), rotary)
        k = rotate(norm(k, name + "k_norm"), rotary)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                    scale: pow(Float(d), -0.5), mask: mask)
            .transposed(0, 2, 1, 3).reshaped(1, n, h * d)
        return linear(out, name + "dense.weight")
    }

    /// The experts' 8 picks per token and their weights, (tokens, 8) each.
    private func route(_ x: MLXArray, _ gate: String) -> (MLXArray, MLXArray) {
        let n = x.dim(0)
        let logits = matmul(x.asType(.float32), weights[gate + ".weight"]!.asType(.float32).T)
        let scores = sigmoid(logits)
        let routing = scores + weights[gate + ".expert_bias"]!.asType(.float32)
        // Best 4 of 8 groups by the sum of each group's top two.
        let grouped = routing.reshaped(n, Self.groups, Self.experts / Self.groups)
        let topTwo = sorted(grouped, axis: -1)[.ellipsis, (Self.experts / Self.groups - 2)...].sum(axis: -1)
        let keptGroups = argPartition(-topTwo, kth: Self.chosenGroups - 1, axis: -1)[.ellipsis, ..<Self.chosenGroups]
        let groupKept = (MLXArray(0 ..< Int32(Self.groups)).reshaped(1, 1, Self.groups)
            .== keptGroups.expandedDimensions(axis: -1)).any(axis: 1)
        let expertKept = repeated(groupKept, count: Self.experts / Self.groups, axis: 1)
        let masked = MLX.where(expertKept, routing, MLXArray(-Float.infinity))
        let picks = argPartition(-masked, kth: Self.chosen - 1, axis: -1)[.ellipsis, ..<Self.chosen]
        var chosenScores = takeAlong(scores, picks, axis: -1)
        chosenScores = chosenScores / (chosenScores.sum(axis: -1, keepDims: true) + 1e-20) * Self.routedScale
        return (picks.asType(.uint32), chosenScores)
    }

    /// (tokens, 2048): each token through its 8 experts, weighted, plus the shared one.
    private func moe(_ x: MLXArray, _ name: String, queryMask: MLXArray) -> MLXArray {
        let (textPicks, textWeights) = route(x, name + "gate")
        let (imagePicks, imageWeights) = route(x, name + "image_gate")
        let picks = MLX.where(queryMask, imagePicks, textPicks)
        let pickWeights = MLX.where(queryMask, imageWeights, textWeights)

        func expert(_ input: MLXArray, _ projection: String) -> MLXArray {
            let w = name + "experts." + projection
            return gatherQuantizedMM(input, weights[w + ".wq"]!, scales: weights[w + ".scales"]!,
                                     biases: weights[w + ".biases"]!, rhsIndices: picks,
                                     transpose: true, groupSize: 64, bits: 4)
        }
        let input = x.expandedDimensions(axes: [-2, -3])            // (n, 1, 1, 2048)
        let hidden = silu(expert(input, "gate_proj")) * expert(input, "up_proj")   // (n, 8, 1, 512)
        let out = expert(hidden, "down_proj").squeezed(axis: -2)    // (n, 8, 2048)
        let routed = (out.asType(.float32) * pickWeights.expandedDimensions(axis: -1)).sum(axis: 1)
        return routed.asType(.bfloat16) + mlp(x, name + "shared_experts.")
    }
}

/// What turns the reader's states into the transformer's two captions.
///
/// The 256 query tokens' final states go through a projection, the
/// connector — Qwen2 1.5B's 28 layers, attending in both directions — and
/// another projection, to 2560 (`caption`). The prompt's own tokens, their
/// states after layers 5, 12 and 20 side by side, are RMS-normed and
/// projected to the transformer's width, 3840 (`direct`).
struct MingCondition {
    private static let width = 1536
    private static let heads = 12
    private static let keyValueHeads = 2
    private static let headDimension = 128
    private static let layers = 28

    let weights: [String: MLXArray]

    func callAsFunction(_ states: MingTextReader.States, promptTokens: Int, queryStart: Int,
                        isCancelled: () -> Bool = { false }) throws -> (caption: MLXArray, direct: MLXArray) {
        let joined = concatenated([states.after5[..<promptTokens], states.after12[..<promptTokens],
                                   states.final[..<promptTokens]], axis: -1)
        let direct = linear(rmsNorm(joined, weight: weights["proj_directvlm.0.weight"]!, eps: 1e-5), "proj_directvlm.1")
        eval(direct)

        let queries = states.final[queryStart ..< (queryStart + MingTextReader.queryCount)]
        var h = linear(queries, "proj_in").expandedDimensions(axis: 0)
        let rotary = Self.rotary(count: MingTextReader.queryCount)
        for layer in 0 ..< Self.layers {
            if isCancelled() { throw CancellationError() }
            let p = "connector.layers.\(layer)."
            h = h + attention(norm(h, p + "input_layernorm"), p + "self_attn.", rotary: rotary)
            let x = norm(h, p + "post_attention_layernorm")
            let m = p + "mlp."
            h = h + linear(silu(linear(x, m + "gate_proj")) * linear(x, m + "up_proj"), m + "down_proj")
            eval(h)
        }
        let caption = linear(norm(h, "connector.norm"), "proj_out")[0]
        eval(caption)
        return (caption, direct)
    }

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        var y = quantizedMatmul(x, weights["\(name).weight.wq"]!, scales: weights["\(name).weight.scales"]!,
                                biases: weights["\(name).weight.biases"]!, transpose: true, groupSize: 64, bits: 8)
        if let bias = weights["\(name).bias"] { y = y + bias }
        return y
    }

    private func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        rmsNorm(x, weight: weights["\(name).weight"]!, eps: 1e-6)
    }

    private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    private static func rotary(count: Int) -> (MLXArray, MLXArray) {
        let half = headDimension / 2
        let inverse = (0 ..< half).map { pow(1_000_000.0, -Double(2 * $0) / Double(headDimension)) }
        var cosines: [Float] = [], sines: [Float] = []
        for position in 0 ..< count {
            let angles = inverse.map { Double(position) * $0 }
            for angle in angles + angles {
                cosines.append(Float(Foundation.cos(angle)))
                sines.append(Float(Foundation.sin(angle)))
            }
        }
        return (MLXArray(cosines, [count, headDimension]).asType(.bfloat16),
                MLXArray(sines, [count, headDimension]).asType(.bfloat16))
    }

    private func attention(_ x: MLXArray, _ name: String, rotary: (MLXArray, MLXArray)) -> MLXArray {
        let n = x.dim(1)
        let d = Self.headDimension
        func rotate(_ y: MLXArray) -> MLXArray {
            let rotated = concatenated([-y[.ellipsis, (d / 2)...], y[.ellipsis, ..<(d / 2)]], axis: -1)
            return y * rotary.0 + rotated * rotary.1
        }
        let q = rotate(linear(x, name + "q_proj").reshaped(1, n, Self.heads, d).transposed(0, 2, 1, 3))
        let k = rotate(linear(x, name + "k_proj").reshaped(1, n, Self.keyValueHeads, d).transposed(0, 2, 1, 3))
        let v = linear(x, name + "v_proj").reshaped(1, n, Self.keyValueHeads, d).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: pow(Float(d), -0.5), mask: nil)
            .transposed(0, 2, 1, 3).reshaped(1, n, Self.heads * d)
        return linear(out, name + "o_proj")
    }
}

/// The Qwen-Image VAE decoder (Wan 2.1's, one frame): 16 latent channels,
/// 8× compression, RGBA out, on the GPU in bfloat16, channels-last.
///
/// For a single frame each causal 3D convolution is a 2D one (its last time
/// slice) and the temporal upsamplers are skipped. Norms are RMS over
/// channels scaled by √channels; the mid block has one single-head attention.
struct MingVAEDecoder {
    let weights: [String: MLXArray]

    private static let unfoldBudget = 256 << 20
    static let scalingFactor: Float = 8.0064

    /// - Parameter latents: (16, 8·rows... ) as (rows, columns, 16), float32.
    /// - Returns: (4, 8 × rows, 8 × columns) in -1...1, float32.
    func callAsFunction(_ latents: MLXArray, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        var x = conv((latents / Self.scalingFactor).expandedDimensions(axis: 0).asType(.bfloat16),
                     "post_quant_conv", padding: 0)
        x = conv(x, "decoder.conv_in")
        x = resnet(x, "decoder.mid_block.resnets.0")
        x = midAttention(x, "decoder.mid_block.attentions.0")
        x = resnet(x, "decoder.mid_block.resnets.1")
        eval(x)
        for block in 0 ..< 4 {
            let name = "decoder.up_blocks.\(block)"
            for layer in 0 ..< 3 {
                if isCancelled() { throw CancellationError() }
                x = resnet(x, "\(name).resnets.\(layer)")
                eval(x)
            }
            if block < 3 {
                x = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
                x = conv(x, "\(name).upsamplers.0.resample.1")
                eval(x)
            }
        }
        x = conv(silu(norm(x, "decoder.norm_out")), "decoder.conv_out").asType(.float32)
        return clip(x[0].transposed(2, 0, 1), min: -1, max: 1)
    }

    private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    private func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        rmsNorm(x, weight: weights["\(name).gamma"]!.reshaped(-1), eps: 1e-12)
    }

    private func conv(_ x: MLXArray, _ name: String, padding: Int = 1) -> MLXArray {
        let weight = weights["\(name).weight"]!
        let (height, width, channels) = (x.dim(1), x.dim(2), x.dim(3))
        let kernel = weight.dim(1) * weight.dim(2)
        let rowBytes = width * channels * kernel * 2
        var y: MLXArray
        if kernel > 1, rowBytes * height > Self.unfoldBudget {
            let rows = max(1, Self.unfoldBudget / rowBytes)
            let padded = padded(x, widths: [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            var strips: [MLXArray] = []
            for start in stride(from: 0, to: height, by: rows) {
                let strip = conv2d(padded[0..., start ..< min(start + rows, height) + 2 * padding], weight)
                eval(strip)
                strips.append(strip)
            }
            y = concatenated(strips, axis: 1)
        } else {
            y = conv2d(x, weight, padding: [padding, padding])
        }
        return y + weights["\(name).bias"]!
    }

    private func resnet(_ x: MLXArray, _ name: String) -> MLXArray {
        var y = conv(silu(norm(x, "\(name).norm1")), "\(name).conv1")
        eval(y)
        y = conv(silu(norm(y, "\(name).norm2")), "\(name).conv2")
        let shortcut = weights["\(name).conv_shortcut.weight"] != nil ? conv(x, "\(name).conv_shortcut", padding: 0) : x
        return shortcut + y
    }

    private func midAttention(_ x: MLXArray, _ name: String) -> MLXArray {
        let (height, width, channels) = (x.dim(1), x.dim(2), x.dim(3))
        let count = height * width
        let y = norm(x, "\(name).norm").reshaped(1, count, channels)
        let qkv = matmul(y, weights["\(name).to_qkv.weight"]!.reshaped(3 * channels, channels).T)
            + weights["\(name).to_qkv.bias"]!
        let q = qkv[.ellipsis, 0 ..< channels].expandedDimensions(axis: 1)
        let k = qkv[.ellipsis, channels ..< 2 * channels].expandedDimensions(axis: 1)
        let v = qkv[.ellipsis, (2 * channels)...].expandedDimensions(axis: 1)
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: count, by: 2048) {
            let chunk = MLXFast.scaledDotProductAttention(queries: q[0..., 0..., start ..< min(start + 2048, count)],
                                                          keys: k, values: v, scale: pow(Float(channels), -0.5), mask: nil)
            eval(chunk)
            chunks.append(chunk)
        }
        let attended = concatenated(chunks, axis: 2)[0..., 0]
        let out = matmul(attended, weights["\(name).proj.weight"]!.reshaped(channels, channels).T)
            + weights["\(name).proj.bias"]!
        return x + out.reshaped(1, height, width, channels)
    }
}

/// Ming-Image-0.2-Design: the prompt through Ling's chat template, the
/// mixture-of-experts reader and the connector, then Z-Image's transformer
/// for 12 flow-matching steps (no guidance), then the VAE decoder.
///
/// Noise levels run evenly from 1 to 0, shifted by 6; the last step lands
/// on zero and changes nothing, so it isn't run.
struct MingPipeline {
    enum File {
        static let vocabulary = "ming_vocab.json"
        static let merges = "ming_merges.txt"
        static let text = (1 ... 5).map { "ming_text.part\($0).safetensors" }
        static let condition = ["ming_condition.part1.safetensors"]
        static let transformer = (1 ... 4).map { "ming_transformer.part\($0).safetensors" }
        static let vae = "ming_vae.safetensors"
    }

    static let steps = 12
    static let specials: [String: Int32] = ["<role>": 157_151, "</role>": 157_152, "<|role_end|>": 156_895,
                                            "<|endoftext|>": 156_892]
    static let pattern = "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
    private static let system = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|>"

    let directory: URL
    /// The text and condition files, if not in `directory` (they're the same for every size).
    var files: (text: [String], condition: [String], transformer: [String]) = (File.text, File.condition, File.transformer)

    /// Noise levels: `steps` evenly from 1 down to 0, shifted by 6 toward the
    /// noisy end, then a final 0. (Upstream's pipeline asks for a shift by
    /// image size, but its scheduler config is frozen and the static shift
    /// of 6 is what runs — checked against the reference.)
    static func sigmas(patches: Int = 0, steps: Int = steps, shift: Double = 6) -> [Float] {
        (0 ..< steps).map { index in
            let s = 1 - Double(index) / Double(steps - 1)
            return Float(shift * s / (1 + (shift - 1) * s))
        } + [0]
    }

    func promptIDs(_ prompt: String) throws -> [Int32] {
        let tokenizer = try ACETokenizer(vocabularyURL: directory.appendingPathComponent(File.vocabulary),
                                         mergesURL: directory.appendingPathComponent(File.merges),
                                         specials: Self.specials, pattern: Self.pattern)
        return tokenizer.encode("\(Self.system)<role>HUMAN</role>\(prompt)<|role_end|><role>ASSISTANT</role>",
                                appendEndOfText: false)
    }

    /// The transformer's two captions for `prompt`: (256, 2560) and (prompt tokens, 3840), bfloat16.
    func condition(_ prompt: String, isCancelled: () -> Bool = { false }) throws -> (caption: MLXArray, direct: MLXArray) {
        let promptIDs = try self.promptIDs(prompt)
        let ids = promptIDs + [MingTextReader.imageStart]
            + Array(repeating: MingTextReader.patchToken, count: MingTextReader.queryCount) + [MingTextReader.imageEnd]
        let states = try StageMemory.withoutCache {
            try MingTextReader(weights: try SanaPipeline.load(files.text, from: directory))(ids, isCancelled: isCancelled)
        }
        return try StageMemory.withoutCache {
            try MingCondition(weights: try SanaPipeline.load(files.condition, from: directory))(
                states, promptTokens: promptIDs.count, queryStart: promptIDs.count + 1, isCancelled: isCancelled)
        }
    }

    /// - Parameter noise: starting latents (16, height/8, width/8) row-major,
    ///   instead of ones drawn from `seed` — for checking against the reference.
    func generate(prompt: String, width: Int, height: Int, seed: UInt64, noise: [Float]? = nil,
                  isCancelled: () -> Bool = { false },
                  progress: (SDPipeline.Stage) -> Void = { _ in }) throws -> CGImage {
        progress(.readingPrompt)
        let (caption, direct) = try condition(prompt, isCancelled: isCancelled)
        let latents = try denoise(caption: caption, direct: direct, width: width, height: height, seed: seed,
                                  noise: noise, isCancelled: isCancelled, progress: progress)
        if isCancelled() { throw CancellationError() }
        progress(.decoding)
        let planar = try decode(latents, width: width, height: height, isCancelled: isCancelled)
        return try RGBImage.make(planar: planar, width: width, height: height)
    }

    /// The finished latents, (rows × columns, 64) patches.
    func denoise(caption: MLXArray, direct: MLXArray, width: Int, height: Int, seed: UInt64, noise: [Float]?,
                 isCancelled: () -> Bool, progress: (SDPipeline.Stage) -> Void) throws -> MLXArray {
        let (rows, columns) = (height / 16, width / 16)
        let channels = ZImageTransformer.latentChannels
        var random = SeededNormal(seed: seed)
        let start = MLXArray(noise ?? random.sample(count: channels * rows * 2 * columns * 2),
                             [channels, rows * 2, columns * 2])
        var latents = start.reshaped(channels, rows, 2, columns, 2).transposed(1, 3, 2, 4, 0)
            .reshaped(rows * columns, 4 * channels)
        let sigmas = Self.sigmas(patches: rows * columns)
        let count = caption.dim(0) + direct.dim(0)
        let padded = count + (ZImageTransformer.sequenceMultiple - count % ZImageTransformer.sequenceMultiple)
            % ZImageTransformer.sequenceMultiple
        let passes = sigmas.indices.dropLast().filter { sigmas[$0] != sigmas[$0 + 1] }
        try StageMemory.withoutCache {
            let transformer = ZImageTransformer(weights: try SanaPipeline.load(files.transformer, from: directory))
            let rotary = ZImageTransformer.Rotary(rows: rows, columns: columns, captionLength: padded, captionTokens: count)
            for (index, step) in passes.enumerated() {
                if isCancelled() { throw CancellationError() }
                progress(.step(index + 1, of: passes.count))
                let out = try transformer(latents, caption: caption, extra: direct, sigma: sigmas[step], rotary: rotary,
                                          isCancelled: isCancelled)
                latents = latents - (sigmas[step + 1] - sigmas[step]) * out
                eval(latents)
            }
        }
        return latents
    }

    /// Planar RGB over white, (3, height, width), 0...1-ish floats for `RGBImage`.
    func decode(_ latents: MLXArray, width: Int, height: Int, isCancelled: () -> Bool) throws -> [Float] {
        let (rows, columns) = (height / 16, width / 16)
        let channels = ZImageTransformer.latentChannels
        return try StageMemory.withoutCache { () throws -> [Float] in
            let decoder = MingVAEDecoder(weights: try MappedWeights.load(url: directory.appendingPathComponent(File.vae)))
            let z = latents.reshaped(rows, columns, 2, 2, channels).transposed(0, 2, 1, 3, 4)
                .reshaped(rows * 2, columns * 2, channels)
            let rgba = try decoder(z, isCancelled: isCancelled)
            let alpha = (rgba[3 ..< 4] + 1) / 2
            return (rgba[0 ..< 3] * alpha + (1 - alpha)).asArray(Float.self)
        }
    }
}
