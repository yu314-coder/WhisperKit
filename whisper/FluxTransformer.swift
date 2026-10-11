//  FLUX.2 [klein] 4B (Black Forest Labs, Apache 2.0). Weights from
//  ports/imagegen/export_flux_mlx.py; see ports/README.md.

import Foundation
import MLX

/// FLUX.2's 3.9B transformer: five double-stream blocks, where text and
/// image keep their own weights but attend jointly, then twenty
/// single-stream blocks over both together, each fusing attention and the
/// feed-forward into one projection in and one out.
///
/// bfloat16 throughout, with 8-bit weights: in float16 its activations
/// overflow from the second step on. Rotary positions are applied, and the
/// timestep embedded, in float32.
struct FluxTransformer {
    static let width = 3072
    static let channels = 128
    private static let heads = 24
    private static let headDimension = 128
    private static let doubleBlocks = 5
    private static let singleBlocks = 20

    let weights: [String: MLXArray]

    /// Rotary angles, as cos and sin (positions, 64), for the text's
    /// positions then the image's. FLUX.2 rotates four axes of 32
    /// dimensions each — (time, row, column, token) — with base 2000: text
    /// tokens count along the last axis, image patches along the middle two.
    struct Rotary {
        let cos: MLXArray
        let sin: MLXArray

        init(rows: Int, columns: Int, textLength: Int) {
            let count = textLength + rows * columns
            let frequencies = (0 ..< 16).map { pow(2000.0, -Double(2 * $0) / 32) }
            var cosines = [Float](repeating: 1, count: count * 64)
            var sines = [Float](repeating: 0, count: count * 64)
            for position in 0 ..< count {
                let coordinates: [Double]
                if position < textLength {
                    coordinates = [0, 0, 0, Double(position)]
                } else {
                    let patch = position - textLength
                    coordinates = [0, Double(patch / columns), Double(patch % columns), 0]
                }
                for axis in 0 ..< 4 {
                    for (index, frequency) in frequencies.enumerated() {
                        let angle = coordinates[axis] * frequency
                        cosines[position * 64 + axis * 16 + index] = Float(Foundation.cos(angle))
                        sines[position * 64 + axis * 16 + index] = Float(Foundation.sin(angle))
                    }
                }
            }
            cos = MLXArray(cosines, [count, 64])
            sin = MLXArray(sines, [count, 64])
        }
    }

    /// One prediction of the velocity at noise level `sigma`.
    /// - Parameters:
    ///   - latents: (patches, 128), float32, raster order.
    ///   - context: (512, 7680), Qwen3's three layers side by side.
    /// - Returns: (patches, 128), float32.
    func callAsFunction(_ latents: MLXArray, context: MLXArray, sigma: Float, rotary: Rotary,
                        isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let d = Self.width
        let (textCount, imageCount) = (context.dim(0), latents.dim(0))

        let time = timestepEmbedding(sigma * 1000)
        let temb = linear(silu(linear(time, "time_guidance_embed.timestep_embedder.linear_1")),
                          "time_guidance_embed.timestep_embedder.linear_2")
        let modImage = linear(silu(temb), "double_stream_modulation_img.linear").asType(.bfloat16)
        let modText = linear(silu(temb), "double_stream_modulation_txt.linear").asType(.bfloat16)
        let modSingle = linear(silu(temb), "single_stream_modulation.linear").asType(.bfloat16)
        /// The `index`th of a modulation's chunks of `d`, (1, 1, d).
        func chunk(_ mod: MLXArray, _ index: Int) -> MLXArray {
            mod[0..., (index * d) ..< ((index + 1) * d)].expandedDimensions(axis: 1)
        }

        var h = linear(latents.expandedDimensions(axis: 0).asType(.bfloat16), "x_embedder")
        var e = linear(context.expandedDimensions(axis: 0).asType(.bfloat16), "context_embedder")

        for block in 0 ..< Self.doubleBlocks {
            if isCancelled() { throw CancellationError() }
            let p = "transformer_blocks.\(block).attn"
            let nh = (1 + chunk(modImage, 1)) * norm(h) + chunk(modImage, 0)
            let ne = (1 + chunk(modText, 1)) * norm(e) + chunk(modText, 0)
            let q = concatenated([headNorm(heads(linear(ne, "\(p).add_q_proj")), "\(p).norm_added_q"),
                                  headNorm(heads(linear(nh, "\(p).to_q")), "\(p).norm_q")], axis: 2)
            let k = concatenated([headNorm(heads(linear(ne, "\(p).add_k_proj")), "\(p).norm_added_k"),
                                  headNorm(heads(linear(nh, "\(p).to_k")), "\(p).norm_k")], axis: 2)
            let v = concatenated([heads(linear(ne, "\(p).add_v_proj")), heads(linear(nh, "\(p).to_v"))], axis: 2)
            let attended = attention(q, k, v, rotary).transposed(0, 2, 1, 3).reshaped(1, textCount + imageCount, d)
            h = h + chunk(modImage, 2) * linear(attended[0..., textCount...], "\(p).to_out.0")
            e = e + chunk(modText, 2) * linear(attended[0..., ..<textCount], "\(p).to_add_out")

            let f = "transformer_blocks.\(block)"
            h = h + chunk(modImage, 5) * feedForward((1 + chunk(modImage, 4)) * norm(h) + chunk(modImage, 3), "\(f).ff")
            e = e + chunk(modText, 5) * feedForward((1 + chunk(modText, 4)) * norm(e) + chunk(modText, 3), "\(f).ff_context")
            eval(h, e)
        }

        h = concatenated([e, h], axis: 1)
        let count = textCount + imageCount
        for block in 0 ..< Self.singleBlocks {
            if isCancelled() { throw CancellationError() }
            let p = "single_transformer_blocks.\(block).attn"
            let nh = (1 + chunk(modSingle, 1)) * norm(h) + chunk(modSingle, 0)
            let projected = linear(nh, "\(p).to_qkv_mlp_proj")
            let q = headNorm(heads(projected[.ellipsis, 0 ..< d]), "\(p).norm_q")
            let k = headNorm(heads(projected[.ellipsis, d ..< 2 * d]), "\(p).norm_k")
            let v = heads(projected[.ellipsis, (2 * d) ..< (3 * d)])
            let attended = attention(q, k, v, rotary).transposed(0, 2, 1, 3).reshaped(1, count, d)
            let out = linear(concatenated([attended, swiglu(projected[.ellipsis, (3 * d)...])], axis: -1), "\(p).to_out")
            h = h + chunk(modSingle, 2) * out
            eval(h)
        }

        h = h[0..., textCount...]
        let emb = linear(silu(temb), "norm_out.linear").asType(.bfloat16)
        h = norm(h) * (1 + emb[0..., ..<d].expandedDimensions(axis: 1)) + emb[0..., d...].expandedDimensions(axis: 1)
        return linear(h, "proj_out")[0].asType(.float32)
    }

    // MARK: - Pieces

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let key = "\(name).weight"
        if let packed = weights["\(key).wq"] {
            return quantizedMatmul(x, packed, scales: weights["\(key).scales"]!, biases: weights["\(key).biases"]!,
                                   transpose: true, groupSize: 64, bits: 8)
        }
        return matmul(x, weights[key]!.asType(x.dtype).T)
    }

    private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    private func swiglu(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let gate = x[.ellipsis, ..<half]
        return silu(gate) * x[.ellipsis, half...]
    }

    private func feedForward(_ x: MLXArray, _ name: String) -> MLXArray {
        linear(swiglu(linear(x, "\(name).linear_in")), "\(name).linear_out")
    }

    /// LayerNorm without its own scale or shift; the modulation supplies them.
    private func norm(_ x: MLXArray) -> MLXArray { layerNorm(x, eps: 1e-6) }

    /// (1, n, 3072) to (1, heads, n, 128).
    private func heads(_ x: MLXArray) -> MLXArray {
        x.reshaped(1, x.dim(1), Self.heads, Self.headDimension).transposed(0, 2, 1, 3)
    }

    private func headNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        rmsNorm(x, weight: weights["\(name).weight"]!.asType(x.dtype), eps: 1e-6)
    }

    private func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ rotary: Rotary) -> MLXArray {
        scaledDotProductAttention(queries: rotate(q, rotary).asType(.bfloat16), keys: rotate(k, rotary).asType(.bfloat16),
                                  values: v, scale: pow(Float(Self.headDimension), -0.5), mask: nil)
    }

    /// Rotates each adjacent pair (2i, 2i + 1) by its angle, in float32.
    private func rotate(_ x: MLXArray, _ rotary: Rotary) -> MLXArray {
        let shape = x.shape
        let pairs = x.asType(.float32).reshaped(Array(shape.dropLast()) + [64, 2])
        let (x0, x1) = (pairs[.ellipsis, 0], pairs[.ellipsis, 1])
        return stacked([x0 * rotary.cos - x1 * rotary.sin, x1 * rotary.cos + x0 * rotary.sin], axis: -1)
            .reshaped(shape)
    }

    /// The timestep's sinusoidal embedding, cosines first, (1, 256).
    private func timestepEmbedding(_ t: Float) -> MLXArray {
        let half = 128
        let values = (0 ..< half).map { Double(t) * exp(-log(10_000.0) * Double($0) / Double(half)) }
        return MLXArray((values.map { Float(cos($0)) } + values.map { Float(sin($0)) }), [1, 2 * half])
    }
}

/// FLUX.2 [klein]'s four-step schedule: noise levels spaced evenly from 1
/// to 1/4, then shifted toward the noisy end by an amount that grows with
/// the picture's size ("empirical mu"), and 0 at the end.
enum FluxSchedule {
    static func sigmas(imagePatches: Int, steps: Int = 4) -> [Float] {
        let n = Double(imagePatches)
        let (a1, b1, a2, b2) = (8.73809524e-05, 1.89833333, 0.00016927, 0.45666666)
        let mu: Double
        if n > 4300 {
            mu = a2 * n + b2
        } else {
            let (m200, m10) = (a2 * n + b2, a1 * n + b1)
            let a = (m200 - m10) / 190
            mu = a * Double(steps) + (m200 - 200 * a)
        }
        let shifted = (0 ..< steps).map { index -> Float in
            let s = 1 - Double(index) * (1 - 1 / Double(steps)) / Double(steps - 1)
            return Float(exp(mu) / (exp(mu) + (1 / s - 1)))
        }
        return shifted + [0]
    }
}
