//  FLUX.2's autoencoder (Black Forest Labs, Apache 2.0): its decoder.
//  Weights from ports/imagegen/export_flux_mlx.py.

import Foundation
import MLX

/// The FLUX.2 VAE decoder: Stable Diffusion's decoder shape with 32 latent
/// channels, on the GPU in float16, channels-last.
///
/// Two things keep a 1024 px decode inside an iPhone's memory:
/// * **Group norms in strips.** Their statistics need float32 — a float16
///   sum over a group overflows (the picture came out NaN) — and a float32
///   copy of a 1024 px stage is 512 MB, so the sums run over strips of rows.
/// * **Convolutions in strips**, since MLX unfolds a 3 × 3 convolution's
///   whole input nine times over.
struct FluxVAEDecoder {
    let weights: [String: MLXArray]
    /// float16 for FLUX.2's VAE. Z-Image's (FLUX.1's) overflows float16 —
    /// upstream marks it force_upcast, and here it decoded to NaN — so it
    /// runs in bfloat16: float32's range at float16's size.
    let dtype: DType

    init(weights: [String: MLXArray], dtype: DType = .float16) {
        self.dtype = dtype
        if dtype == .float16 {
            self.weights = weights
        } else {
            let cast = weights.mapValues { $0.dtype == .float16 ? $0.asType(dtype) : $0 }
            eval(Array(cast.values))
            self.weights = cast
        }
    }

    private static let unfoldBudget = 256 << 20
    private static let statisticsBudget = 64 << 20
    private static let groups = 32

    /// - Parameters:
    ///   - latents: (patches, 128) from the transformer, float32.
    /// - Returns: (3, height, width) in -1...1, float32.
    func callAsFunction(_ latents: MLXArray, height: Int, width: Int,
                        isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let (rows, columns) = (height / 16, width / 16)
        // Undo the latents' batch norm, then unpack each patch's 128 values to
        // 2 × 2 pixels of 32 channels.
        var z = latents.reshaped(rows, columns, 128)
        z = z * sqrt(weights["bn.running_var"]! + 1e-4) + weights["bn.running_mean"]!
        z = z.reshaped(rows, columns, 32, 2, 2).transposed(0, 3, 1, 4, 2)
            .reshaped(1, 2 * rows, 2 * columns, 32).asType(.float16)

        return try decode(conv(z, "post_quant_conv", padding: 0), isCancelled: isCancelled)
    }

    /// The decoder proper, from latents (1, h, w, channels) channels-last in
    /// float16 — shared with Z-Image, whose VAE is FLUX.1's: the same decoder
    /// over 16 channels, without FLUX.2's latent batch norm or post-quant
    /// convolution. Returns (3, height, width) in -1...1, float32.
    func decode(_ z: MLXArray, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        var x = conv(z.asType(dtype), "decoder.conv_in")
        x = resnet(x, "decoder.mid_block.resnets.0")
        x = midAttention(x, "decoder.mid_block.attentions.0")
        x = resnet(x, "decoder.mid_block.resnets.1")
        eval(x)
        for block in 0 ..< 4 {
            for layer in 0 ..< 3 {
                if isCancelled() { throw CancellationError() }
                x = resnet(x, "decoder.up_blocks.\(block).resnets.\(layer)")
                eval(x)
            }
            if block < 3 {
                x = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
                x = conv(x, "decoder.up_blocks.\(block).upsamplers.0.conv")
                eval(x)
            }
        }
        x = conv(silu(groupNorm(x, "decoder.conv_norm_out")), "decoder.conv_out")
        return x[0].transposed(2, 0, 1).asType(.float32)
    }

    // MARK: - Pieces

    private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

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

    /// GroupNorm (32 groups, eps 1e-6) with float32 statistics gathered over
    /// strips of rows; the normalizing itself stays in `dtype`.
    private func groupNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        let (height, width, channels) = (x.dim(1), x.dim(2), x.dim(3))
        let perGroup = channels / Self.groups
        let rows = max(1, Self.statisticsBudget / (width * channels * 4))
        var sum = MLXArray.zeros([Self.groups], dtype: .float32)
        var squares = MLXArray.zeros([Self.groups], dtype: .float32)
        for start in stride(from: 0, to: height, by: rows) {
            let strip = x[0..., start ..< min(start + rows, height)].asType(.float32)
                .reshaped(-1, Self.groups, perGroup)
            sum = sum + strip.sum(axes: [0, 2])
            squares = squares + (strip * strip).sum(axes: [0, 2])
            eval(sum, squares)
        }
        let count = Float(height * width * perGroup)
        let mean = sum / count
        let variance = maximum(squares / count - mean * mean, 0)
        let scale = repeated(rsqrt(variance + 1e-6), count: perGroup, axis: 0).asType(dtype)
        let shift = repeated(mean, count: perGroup, axis: 0).asType(dtype)
        return (x - shift) * (scale * weights["\(name).weight"]!) + weights["\(name).bias"]!
    }

    private func resnet(_ x: MLXArray, _ name: String) -> MLXArray {
        var y = conv(silu(groupNorm(x, "\(name).norm1")), "\(name).conv1")
        eval(y)
        y = conv(silu(groupNorm(y, "\(name).norm2")), "\(name).conv2")
        let shortcut = weights["\(name).conv_shortcut.weight"] != nil ? conv(x, "\(name).conv_shortcut", padding: 0) : x
        return shortcut + y
    }

    /// The mid block's single-head attention over every latent pixel, in
    /// chunks of queries so the score matrix stays small.
    private func midAttention(_ x: MLXArray, _ name: String) -> MLXArray {
        let (height, width, channels) = (x.dim(1), x.dim(2), x.dim(3))
        let count = height * width
        let y = groupNorm(x, "\(name).group_norm").reshaped(1, count, channels)
        func project(_ key: String) -> MLXArray {
            (matmul(y, weights["\(name).\(key).weight"]!.T) + weights["\(name).\(key).bias"]!).expandedDimensions(axis: 1)
        }
        let (q, k, v) = (project("to_q"), project("to_k"), project("to_v"))
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: count, by: 2048) {
            let chunk = scaledDotProductAttention(queries: q[0..., 0..., start ..< min(start + 2048, count)], keys: k,
                                                  values: v, scale: pow(Float(channels), -0.5), mask: nil)
            eval(chunk)
            chunks.append(chunk)
        }
        let attended = concatenated(chunks, axis: 2)[0..., 0]
        let out = matmul(attended, weights["\(name).to_out.0.weight"]!.T) + weights["\(name).to_out.0.bias"]!
        return x + out.reshaped(1, height, width, channels)
    }
}
