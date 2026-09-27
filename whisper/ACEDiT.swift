//  Ported from ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5), MIT.
//  Verified against the PyTorch reference stage by stage; see ports/README.md.

import Foundation
import MLX

/// One layer of ACE-Step's diffusion transformer.
///
/// Structure per layer:
///   1. self-attention under adaptive layer norm, with a **gated** residual
///   2. cross-attention to the conditioning, with a plain residual
///   3. MLP under adaptive layer norm, gated residual again
///
/// The six modulation terms come from `scale_shift_table + temb`, split along
/// the middle axis: shift/scale/gate for the attention, then the same three
/// for the MLP. The table is a learned per-layer offset, so the timestep
/// embedding alone is not enough.
///
/// As in the other transformers here, rotary embedding belongs to
/// self-attention only — the cross-attention path never sees cos/sin, for
/// queries or keys.
struct ACEDiT {
    static let hiddenSize = 2048
    static let headCount = 16
    static let keyValueHeadCount = 8
    static let headDimension = 128
    static let epsilon: Float = 1e-6

    let weights: [String: MLXArray]
    /// 0 when the checkpoint is dense; 4 or 8 when its projections are packed.
    var quantizationBits: Int = 0
    var evaluatesPerLayer = true
    /// What activations are carried in; quantized scales must match it.
    var dtype: DType = .float32
    var quantizationGroup: Int = 64

    private func w(_ key: String) -> MLXArray {
        guard let value = weights[key] else { fatalError("missing DiT weight: \(key)") }
        return value.asType(dtype)
    }

    /// `x @ W.T`, reading either a dense weight or a quantized triple.
    ///
    /// Quantizing only the 2-D projections is deliberate: they carry almost
    /// all the size, while norms and embedding tables are small and lose
    /// disproportionate accuracy when packed.
    private func linear(_ x: MLXArray, _ key: String) -> MLXArray {
        if quantizationBits > 0, let packed = weights["\(key).wq"] {
            return quantizedMatmul(x, packed,
                                   scales: weights["\(key).scales"]!,
                                   biases: weights["\(key).biases"]!,
                                   transpose: true,
                                   groupSize: quantizationGroup,
                                   bits: quantizationBits)
        }
        return matmul(x, w(key).T)
    }

    func layer(_ index: Int,
               hidden: MLXArray,
               encoder: MLXArray,
               temb: MLXArray,
               cos: MLXArray,
               sin: MLXArray,
               localMask: MLXArray? = nil) -> MLXArray {
        let prefix = "decoder.layers.\(index)"
        let modulation = w("\(prefix).scale_shift_table") + temb
        let parts = modulation.split(parts: 6, axis: 1)
        let (shift, scale, gate) = (parts[0], parts[1], parts[2])
        let (mlpShift, mlpScale, mlpGate) = (parts[3], parts[4], parts[5])

        var h = hidden
        var normed = rmsNorm(h, weight: w("\(prefix).self_attn_norm.weight"), eps: Self.epsilon)
            * (1 + scale) + shift
        h = h + attention(prefix: "\(prefix).self_attn", x: normed, source: nil,
                          cos: cos, sin: sin,
                          mask: ACESlidingWindow.isLocal(layer: index) ? localMask : nil) * gate

        normed = rmsNorm(h, weight: w("\(prefix).cross_attn_norm.weight"), eps: Self.epsilon)
        h = h + attention(prefix: "\(prefix).cross_attn", x: normed, source: encoder,
                          cos: nil, sin: nil)

        normed = rmsNorm(h, weight: w("\(prefix).mlp_norm.weight"), eps: Self.epsilon)
            * (1 + mlpScale) + mlpShift
        h = h + feedForward(prefix: "\(prefix).mlp", x: normed) * mlpGate
        return h
    }

    private func attention(prefix: String, x: MLXArray, source: MLXArray?,
                           cos: MLXArray?, sin: MLXArray?, mask: MLXArray? = nil) -> MLXArray {
        let queryLength = x.dim(1)
        let memory = source ?? x
        let memoryLength = memory.dim(1)

        // Head split happens before the norm, which applies over the head
        // dimension rather than the full width.
        func heads(_ input: MLXArray, _ key: String, _ count: Int, _ length: Int) -> MLXArray {
            linear(input, "\(prefix).\(key)_proj.weight")
                .reshaped(1, length, count, Self.headDimension)
        }
        var q = rmsNorm(heads(x, "q", Self.headCount, queryLength),
                        weight: w("\(prefix).q_norm.weight"), eps: Self.epsilon)
            .transposed(0, 2, 1, 3)
        var k = rmsNorm(heads(memory, "k", Self.keyValueHeadCount, memoryLength),
                        weight: w("\(prefix).k_norm.weight"), eps: Self.epsilon)
            .transposed(0, 2, 1, 3)
        let v = heads(memory, "v", Self.keyValueHeadCount, memoryLength).transposed(0, 2, 1, 3)

        if let cos, let sin, source == nil {
            q = applyRotary(q, cos: cos, sin: sin)
            k = applyRotary(k, cos: cos, sin: sin)
        }

        // The fused kernel never materializes the full score matrix, which
        // is what makes long clips possible at all: at the 6:24 maximum the
        // diffusion transformer sees 4,800 positions, and the unfused
        // product would be 16 x 4,800 x 4,800 floats — 1.5 GB per layer.
        let out = scaledDotProductAttention(queries: q, keys: k, values: v,
                                            scale: pow(Float(Self.headDimension), -0.5),
                                            mask: mask)
            .transposed(0, 2, 1, 3)
            .reshaped(1, queryLength, Self.headCount * Self.headDimension)
        return linear(out, "\(prefix).o_proj.weight")
    }

    private func feedForward(prefix: String, x: MLXArray) -> MLXArray {
        let gate = linear(x, "\(prefix).gate_proj.weight")
        let up = linear(x, "\(prefix).up_proj.weight")
        return linear(gate * sigmoid(gate) * up, "\(prefix).down_proj.weight")
    }

    /// `x * cos + rotate_half(x) * sin`, with cos/sin supplied per position.
    private func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = Self.headDimension / 2
        let c = cos.reshaped(1, 1, cos.dim(1), cos.dim(2))
        let s = sin.reshaped(1, 1, sin.dim(1), sin.dim(2))
        let first = x[0..., 0..., 0..., 0 ..< half]
        let second = x[0..., 0..., 0..., half ..< Self.headDimension]
        let rotated = concatenated([-second, first], axis: -1)
        return x * c + rotated * s
    }

}

// MARK: - Full forward

extension ACEDiT {
    static let patchSize = 2
    static let outputChannels = 64
    static let ropeTheta: Float = 1_000_000

    /// Predicts the flow-matching velocity for one diffusion step.
    ///
    /// - Parameters:
    ///   - xt: (1, frames, 64) current noisy latent.
    ///   - context: (1, frames, 128) conditioning latents, concatenated onto
    ///     `xt` along channels to make the 192 the patch convolution expects.
    ///   - encoder: (1, tokens, 2048) conditioning sequence.
    ///   - timestep: the current t.
    func forward(xt: MLXArray, context: MLXArray, encoder: MLXArray, timestep: Float) -> MLXArray {
        let frames = xt.dim(1)

        // Two timestep embeddings are summed: one for t, one for t - r. With
        // r == t the second reduces to the embedding of zero, which is not the
        // same as omitting it — it still contributes learned bias.
        let (tembT, projT) = timeEmbedding(timestep, prefix: "decoder.time_embed")
        let (tembR, projR) = timeEmbedding(0, prefix: "decoder.time_embed_r")
        let temb = tembT + tembR
        let modulation = projT + projR

        var h = concatenated([context, xt], axis: -1).asType(dtype)
        let originalLength = h.dim(1)
        if h.dim(1) % Self.patchSize != 0 {
            let pad = Self.patchSize - (h.dim(1) % Self.patchSize)
            h = concatenated([h, MLXArray.zeros([1, pad, h.dim(2)], dtype: h.dtype)], axis: 1)
        }

        // Patch embedding: stride equals kernel, so this both projects and
        // halves the sequence. Stored (out, in, k); MLX wants (out, k, in).
        h = conv1d(h, w("decoder.proj_in.1.weight").transposed(0, 2, 1),
                   stride: Self.patchSize, padding: 0)
            + w("decoder.proj_in.1.bias")

        let conditioning = linear(encoder.asType(dtype), "decoder.condition_embedder.weight")
            + w("decoder.condition_embedder.bias")

        let (cos, sin) = rotaryTable(length: h.dim(1))
        let localMask = ACESlidingWindow.mask(length: h.dim(1))
        for index in 0 ..< 24 {
            h = layer(index, hidden: h, encoder: conditioning, temb: modulation,
                      cos: cos, sin: sin, localMask: localMask)
            // Layer by layer: left as one graph, MLX kept many layers'
            // intermediates alive together — 1.6 GB at 81 seconds, where one
            // layer's worth is a small fraction of that.
            if evaluatesPerLayer { eval(h) }
        }

        let outParts = (w("decoder.scale_shift_table") + temb.expandedDimensions(axis: 1))
            .split(parts: 2, axis: 1)
        h = rmsNorm(h, weight: w("decoder.norm_out.weight"), eps: Self.epsilon)
            * (1 + outParts[1]) + outParts[0]

        // De-patchify: transposed convolution back to one frame per input.
        // Stored (in, out, k) for a transposed conv; MLX wants (out, k, in).
        h = convTransposed1d(h, w("decoder.proj_out.1.weight").transposed(1, 2, 0),
                             stride: Self.patchSize, padding: 0)
            + w("decoder.proj_out.1.bias")
        return h[0..., 0 ..< originalLength, 0...].asType(.float32)
    }

    /// Sinusoidal features, an MLP, and a six-way modulation projection.
    private func timeEmbedding(_ t: Float, prefix: String) -> (MLXArray, MLXArray) {
        let dimension = 256
        let half = dimension / 2
        let scaled = t * 1000   // the module's own scale
        var frequencies = [Float](repeating: 0, count: half)
        for i in 0 ..< half {
            frequencies[i] = exp(-log(Float(10000)) * Float(i) / Float(half)) * scaled
        }
        let args = MLXArray(frequencies, [1, half])
        let features = concatenated([cos(args), sin(args)], axis: -1).asType(dtype)

        var temb = linear(features, "\(prefix).linear_1.weight") + w("\(prefix).linear_1.bias")
        temb = temb * sigmoid(temb)
        temb = linear(temb, "\(prefix).linear_2.weight") + w("\(prefix).linear_2.bias")

        let activated = temb * sigmoid(temb)
        let projected = linear(activated, "\(prefix).time_proj.weight")
            + w("\(prefix).time_proj.bias")
        return (temb, projected.reshaped(1, 6, Self.hiddenSize))
    }

    /// Qwen3 rotary table: half as many inverse frequencies as the head
    /// dimension, each used twice.
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
