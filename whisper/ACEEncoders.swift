//  Ported from ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5), MIT.
//  Verified against the PyTorch reference stage by stage; see ports/README.md.

import Foundation
import MLX

/// The lyric and timbre encoders, which share one shape.
///
/// Both are short bidirectional stacks of the same Qwen3 block used
/// everywhere else in ACE-Step, differing only in depth and in what they take
/// as input: the lyric encoder projects 1024-wide text embeddings through 8
/// layers, the timbre encoder projects 64-wide acoustic latents through 4 and
/// reads its summary from position zero.
///
/// Attention is bidirectional here — these describe conditioning that exists
/// all at once, not a sequence being generated, so there is no causal mask.
/// Even-numbered layers are local, though: see `ACESlidingWindow`.
struct ACEEncoderStack {
    static let hiddenSize = 2048
    static let headCount = 16
    static let keyValueHeadCount = 8
    static let headDimension = 128
    static let ropeTheta: Float = 1_000_000
    static let epsilon: Float = 1e-6

    let weights: [String: MLXArray]
    let prefix: String
    let layerCount: Int
    var quantizationBits: Int = 0
    var quantizationGroup: Int = 64

    private func w(_ key: String) -> MLXArray {
        guard let value = weights["\(prefix).\(key)"] else {
            fatalError("missing encoder weight: \(prefix).\(key)")
        }
        return value.asType(.float32)
    }

    private func linear(_ x: MLXArray, _ key: String) -> MLXArray {
        let full = "\(prefix).\(key)"
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

    /// - Parameter inputs: (1, length, inputDimension).
    ///
    /// The timbre encoder ships a `special_token` weight that looks like a
    /// CLS lead token, but the reference has that concatenation commented
    /// out — it is dead weight, and position zero of the projected input
    /// carries the summary instead. Prepending it costs cosine 0.85 against
    /// the reference, which is why this takes the checkpoint's word for
    /// nothing.
    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        var h = linear(inputs, "embed_tokens.weight") + w("embed_tokens.bias")
        let (cos, sin) = rotaryTable(length: h.dim(1))
        let local = ACESlidingWindow.mask(length: h.dim(1))

        for index in 0 ..< layerCount {
            let layer = "layers.\(index)"
            h = h + attention(prefix: "\(layer).self_attn",
                              x: rmsNorm(h, weight: w("\(layer).input_layernorm.weight"),
                                         eps: Self.epsilon),
                              cos: cos, sin: sin,
                              mask: ACESlidingWindow.isLocal(layer: index) ? local : nil)
            h = h + feedForward(prefix: "\(layer).mlp",
                                x: rmsNorm(h, weight: w("\(layer).post_attention_layernorm.weight"),
                                           eps: Self.epsilon))
        }
        return rmsNorm(h, weight: w("norm.weight"), eps: Self.epsilon)
    }

    private func attention(prefix: String, x: MLXArray, cos: MLXArray, sin: MLXArray,
                           mask: MLXArray?) -> MLXArray {
        let length = x.dim(1)
        func heads(_ key: String, _ count: Int) -> MLXArray {
            linear(x, "\(prefix).\(key)_proj.weight")
                .reshaped(1, length, count, Self.headDimension)
        }
        var q = rmsNorm(heads("q", Self.headCount),
                        weight: w("\(prefix).q_norm.weight"), eps: Self.epsilon)
            .transposed(0, 2, 1, 3)
        var k = rmsNorm(heads("k", Self.keyValueHeadCount),
                        weight: w("\(prefix).k_norm.weight"), eps: Self.epsilon)
            .transposed(0, 2, 1, 3)
        let v = heads("v", Self.keyValueHeadCount).transposed(0, 2, 1, 3)

        q = applyRotary(q, cos: cos, sin: sin)
        k = applyRotary(k, cos: cos, sin: sin)

        let out = scaledDotProductAttention(queries: q, keys: k, values: v,
                                            scale: pow(Float(Self.headDimension), -0.5),
                                            mask: mask)
            .transposed(0, 2, 1, 3)
            .reshaped(1, length, Self.headCount * Self.headDimension)
        return linear(out, "\(prefix).o_proj.weight")
    }

    private func feedForward(prefix: String, x: MLXArray) -> MLXArray {
        let gate = linear(x, "\(prefix).gate_proj.weight")
        let up = linear(x, "\(prefix).up_proj.weight")
        return linear(gate * sigmoid(gate) * up, "\(prefix).down_proj.weight")
    }

    private func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = Self.headDimension / 2
        let c = cos.reshaped(1, 1, cos.dim(1), cos.dim(2))
        let s = sin.reshaped(1, 1, sin.dim(1), sin.dim(2))
        let first = x[0..., 0..., 0..., 0 ..< half]
        let second = x[0..., 0..., 0..., half ..< Self.headDimension]
        return x * c + concatenated([-second, first], axis: -1) * s
    }

    private func rotaryTable(length: Int) -> (MLXArray, MLXArray) {
        let half = Self.headDimension / 2
        var cosValues = [Float](repeating: 0, count: length * Self.headDimension)
        var sinValues = [Float](repeating: 0, count: length * Self.headDimension)
        for position in 0 ..< length {
            for i in 0 ..< half {
                let inverse = 1.0 / pow(Self.ropeTheta, Float(2 * i) / Float(Self.headDimension))
                let angle = Float(position) * inverse
                cosValues[position * Self.headDimension + i] = cos(angle)
                cosValues[position * Self.headDimension + i + half] = cos(angle)
                sinValues[position * Self.headDimension + i] = sin(angle)
                sinValues[position * Self.headDimension + i + half] = sin(angle)
            }
        }
        return (MLXArray(cosValues, [1, length, Self.headDimension]),
                MLXArray(sinValues, [1, length, Self.headDimension]))
    }
}

/// ACE-Step alternates local and global attention: even-numbered layers of
/// every stack — lyric, timbre and diffusion alike — see only positions
/// within 128 of their own, odd ones see everything.
///
/// Invisible on short inputs, which is how it went unported: below 130
/// positions the window covers the whole sequence and both kinds of layer
/// agree. A 30-second clip is 375 positions in the diffusion transformer, and
/// the timbre encoder always reads 750, so without the window half the layers
/// attended to context they were never trained to see.
enum ACESlidingWindow {
    static let radius = 128

    static func isLocal(layer: Int) -> Bool { layer % 2 == 0 }

    /// Additive mask, or nil when the window already spans the sequence.
    static func mask(length: Int) -> MLXArray? {
        guard length > radius + 1 else { return nil }
        let positions = MLXArray(Array(Int32(0) ..< Int32(length)))
        let distance = abs(positions.reshaped(length, 1) - positions.reshaped(1, length))
        return MLX.where(lessEqual(distance, MLXArray(Int32(radius))),
                         MLXArray(Float(0)), MLXArray(-Float.infinity))
            .reshaped(1, 1, length, length)
    }
}
