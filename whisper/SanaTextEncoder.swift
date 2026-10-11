//  Gemma 2 2B (Google, Gemma Terms of Use), as SANA-Sprint (NVIDIA, Apache
//  2.0) reads prompts with it. Weights from ports/imagegen/export_sana_mlx.py.

import Foundation
import MLX

/// Gemma 2 2B without its language-model head: SANA conditions on the last
/// hidden states.
///
/// What sets Gemma 2 apart from the Qwen3 used for music:
///
/// * **Norms around both halves of a layer**, before and after attention
///   and before and after the feed-forward, each scaling by `1 + weight`.
/// * **Attention logits soft-capped** at ±50 by `50 tanh(x / 50)`, as the
///   eager attention SANA was trained with does (newer `transformers`
///   skips it under SDPA; on SANA's prompts the two agree to cosine
///   0.999994).
/// * **Embeddings scaled** by √2304 = 48 on the way in.
/// * **Float32 activations.** Gemma's residual stream overflows float16.
///
/// Only the prompt's real tokens are computed. SANA pads to a fixed length
/// on the right, attention is causal, so padding never reaches a real
/// position, and SANA masks the padded positions out anyway.
struct Gemma2Encoder {
    static let width = 2304
    private static let layers = 26
    private static let heads = 8
    private static let keyValueHeads = 4
    private static let headDimension = 256
    private static let groupSize = 64
    private static let bits = 8

    let weights: [String: MLXArray]

    /// (tokens, 2304), float32. `isCancelled` is asked before each layer.
    func callAsFunction(_ ids: [Int32], isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let count = ids.count
        let key = "embed_tokens.weight"
        let rows = MLXArray(ids)
        var h = dequantized(take(weights["\(key).wq"]!, rows, axis: 0),
                            scales: take(weights["\(key).scales"]!, rows, axis: 0),
                            biases: take(weights["\(key).biases"]!, rows, axis: 0),
                            groupSize: Self.groupSize, bits: Self.bits)
            .asType(.float32) * 48
        h = h.reshaped(1, count, Self.width)

        var mask = [Float](repeating: 0, count: count * count)
        for query in 0 ..< count {
            for key in (query + 1) ..< count { mask[query * count + key] = -1e9 }
        }
        let causal = MLXArray(mask, [1, 1, count, count])

        for layer in 0 ..< Self.layers {
            if isCancelled() { throw CancellationError() }
            let prefix = "layers.\(layer)"
            let attended = attention(norm(h, "\(prefix).input_layernorm"), prefix: "\(prefix).self_attn",
                                     mask: causal)
            h = h + norm(attended, "\(prefix).post_attention_layernorm")
            let fed = feedForward(norm(h, "\(prefix).pre_feedforward_layernorm"), prefix: "\(prefix).mlp")
            h = h + norm(fed, "\(prefix).post_feedforward_layernorm")
            // One layer at a time, so MLX doesn't hold every layer's
            // intermediates in one graph.
            eval(h)
        }
        return norm(h, "norm").reshaped(count, Self.width)
    }

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let key = "\(name).weight"
        return quantizedMatmul(x, weights["\(key).wq"]!, scales: weights["\(key).scales"]!,
                               biases: weights["\(key).biases"]!, transpose: true,
                               groupSize: Self.groupSize, bits: Self.bits)
    }

    private func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        rmsNorm(x, weight: 1 + weights["\(name).weight"]!, eps: 1e-6)
    }

    private func attention(_ x: MLXArray, prefix: String, mask: MLXArray) -> MLXArray {
        let count = x.dim(1)
        func project(_ name: String, heads: Int) -> MLXArray {
            linear(x, "\(prefix).\(name)")
                .reshaped(1, count, heads, Self.headDimension)
                .transposed(0, 2, 1, 3)
        }
        func rotate(_ x: MLXArray) -> MLXArray {
            RoPE(x, dimensions: Self.headDimension, traditional: false, base: 10_000, scale: 1, offset: 0)
        }
        let q = rotate(project("q_proj", heads: Self.heads))
        // Each key/value head serves two query heads, the neighbouring pair.
        let repeats = Self.heads / Self.keyValueHeads
        let k = repeated(rotate(project("k_proj", heads: Self.keyValueHeads)), count: repeats, axis: 1)
        let v = repeated(project("v_proj", heads: Self.keyValueHeads), count: repeats, axis: 1)

        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * pow(Float(Self.headDimension), -0.5)
        scores = tanh(scores / 50) * 50 + mask
        let out = matmul(softmax(scores, axis: -1, precise: true), v)
            .transposed(0, 2, 1, 3)
            .reshaped(1, count, Self.heads * Self.headDimension)
        return linear(out, "\(prefix).o_proj")
    }

    private func feedForward(_ x: MLXArray, prefix: String) -> MLXArray {
        let gate = linear(x, "\(prefix).gate_proj")
        let up = linear(x, "\(prefix).up_proj")
        return linear(geluTanh(gate) * up, "\(prefix).down_proj")
    }
}

/// GELU, tanh approximation — Gemma's and PixArt's caption projection's.
func geluTanh(_ x: MLXArray) -> MLXArray {
    0.5 * x * (1 + tanh(0.797_884_56 * (x + 0.044_715 * x * x * x)))
}

/// What SANA does with Gemma's output before its transformer sees it.
enum SanaCaption {
    /// Prompt positions the transformer attends over.
    static let length = 300
    /// SANA's instruction prefix and prompt together are cut to this many
    /// tokens (the prefix's 208, BOS included, plus 298).
    static let maximumTokens = 506
    /// Positions kept: BOS, then from the prefix's last token on.
    static let firstKept = 207

    /// SANA-Sprint's "complex human instruction": the prompt goes after it,
    /// so Gemma reads the prompt as something to be described in detail.
    /// Upstream's default, verbatim.
    static let instruction = [
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ].joined(separator: "\n")

    /// The token ids Gemma reads: the instruction, then the prompt lower-cased
    /// and trimmed, as upstream prepares it. Foundation's lower-casing rather
    /// than Swift's: like Python's, it ends a Greek word with ς, not σ.
    static func tokens(for prompt: String, tokenizer: GemmaTokenizer) -> [Int32] {
        let cleaned = prompt.lowercased(with: nil).trimmingCharacters(in: .whitespacesAndNewlines)
        return Array(tokenizer.encode(instruction + cleaned).prefix(maximumTokens))
    }

    /// Gemma's states for `tokens` to the transformer's caption: the kept
    /// positions, projected and normed, zero past the prompt's end.
    /// Returns the caption (1, 300, width: 1152 for the 0.6B, 2240 for the
    /// 1.6B) and how many positions are real.
    static func project(_ states: MLXArray, extras: [String: MLXArray]) -> (MLXArray, Int) {
        let count = states.dim(0)
        let kept = [0] + Array(min(firstKept, count) ..< count)
        var rows = take(states, MLXArray(kept.map(Int32.init)), axis: 0)
        rows = padded(rows, widths: [[0, length - kept.count], [0, 0]])

        func weight(_ name: String) -> MLXArray { extras[name]!.asType(.float32) }
        var x = matmul(rows, weight("caption_projection.linear_1.weight").T) + weight("caption_projection.linear_1.bias")
        x = matmul(geluTanh(x), weight("caption_projection.linear_2.weight").T) + weight("caption_projection.linear_2.bias")
        x = rmsNorm(x, weight: weight("caption_norm.weight"), eps: 1e-5)
        return (x.reshaped(1, length, x.dim(-1)), kept.count)
    }
}
