//  Qwen-Image 2.1's autoencoder (Qwen, Qwen RESEARCH LICENSE): its decoder.
//  Weights from ports/imagegen/export_qwenimage_mlx.py.

import Foundation
import MLX

/// The Qwen-Image 2.1 VAE decoder: Wan's video decoder specialised to one
/// frame — 64 latent channels, 16× compression, RGBA out — on the GPU in
/// bfloat16, channels-last.
///
/// For a single frame the causal 3D convolutions are plain 2D ones, the
/// temporal upsamplers are skipped (upstream marks the first chunk "Rep"),
/// and the shortcut that duplicates channels into pixels keeps only the last
/// of its two frames. Norms are RMS over channels, scaled by √channels.
///
/// Convolutions run in strips, since MLX unfolds a 3 × 3 convolution's whole
/// input nine times over, and each stage is evaluated before the next.
struct QwenImageVAEDecoder {
    let weights: [String: MLXArray]

    static let latentChannels = 64
    private static let unfoldBudget = 256 << 20

    /// The latents' per-channel statistics: the transformer works in units
    /// of these, the decoder in the VAE's own.
    static let latentsMean: [Float] = [
        0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732,
        0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
        -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555,
        0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
        -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632,
        0.5547, -1.7121, -0.1562, 0.0304,
    ]
    static let latentsStd: [Float] = [
        3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904,
        2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
        3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075,
        3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
        3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149,
        3.7662, 3.6746, 3.4563, 3.8161,
    ]

    /// The last three up blocks run in strips of this many of their input
    /// rows (a quarter of the output's), with `halo` rows of context either
    /// side: past the mid block's attention every layer is a convolution or
    /// per-pixel, so a strip comes out as the whole picture would — and the
    /// full-size stages, 288 channels at 1024 px, are 600 MB each.
    private static let stripRows = 32
    /// What the strip's seventeen 3 × 3 convolutions see beyond it, in its
    /// input's rows: 6 + ½ + 3 + ¼ + 1¾, rounded up.
    private static let halo = 12

    /// - Parameters:
    ///   - latents: (rows × columns, 64) from the transformer, float32, row-major.
    ///   - stage: sees each stage's output, for checking against upstream.
    /// - Returns: (4, 16 × rows, 16 × columns) — red, green, blue, alpha — in -1...1, float32.
    func callAsFunction(_ latents: MLXArray, rows: Int, columns: Int,
                        isCancelled: () -> Bool = { false },
                        stage: (String, MLXArray) -> Void = { _, _ in }) throws -> MLXArray {
        var z = latents.reshaped(1, rows, columns, Self.latentChannels)
        z = z * MLXArray(Self.latentsStd) + MLXArray(Self.latentsMean)
        var x = conv(z.asType(.bfloat16), "post_quant_conv", padding: 0)
        x = conv(x, "decoder.conv_in")
        stage("conv_in", x)
        x = resnet(x, "decoder.mid_block.resnets.0")
        x = midAttention(x, "decoder.mid_block.attentions.0")
        x = resnet(x, "decoder.mid_block.resnets.1")
        eval(x)
        stage("mid", x)
        for block in 0 ..< 2 {
            x = try upBlock(x, block, isCancelled: isCancelled)
            stage("up\(block)", x)
        }

        let height = x.dim(1)
        var strips: [MLXArray] = []
        for start in stride(from: 0, to: height, by: Self.stripRows) {
            let end = min(start + Self.stripRows, height)
            let (low, high) = (max(0, start - Self.halo), min(height, end + Self.halo))
            var y = x[0..., low ..< high]
            for block in 2 ..< 5 { y = try upBlock(y, block, isCancelled: isCancelled) }
            y = conv(silu(norm(y, "decoder.norm_out")), "decoder.conv_out")
            y = y[0..., ((start - low) * 4) ..< ((end - low) * 4)].asType(.float32)
            eval(y)
            strips.append(y)
        }
        let picture = concatenated(strips, axis: 1)
        return clip(picture[0].transposed(2, 0, 1), min: -1, max: 1)
    }

    // MARK: - Pieces

    /// Three residual blocks, then — but for the last — a 2× upsampling plus
    /// the shortcut that duplicates the block's input channels into pixels.
    private func upBlock(_ x: MLXArray, _ block: Int, isCancelled: () -> Bool) throws -> MLXArray {
        let name = "decoder.up_blocks.\(block)"
        var y = x
        for layer in 0 ..< 3 {
            if isCancelled() { throw CancellationError() }
            y = resnet(y, "\(name).resnets.\(layer)")
            eval(y)
        }
        guard block < 4 else { return y }
        y = repeated(repeated(y, count: 2, axis: 1), count: 2, axis: 2)
        y = conv(y, "\(name).upsampler.resample.1")
        eval(y)
        y = y + duplicated(x, channels: y.dim(3), frames: block < 3 ? 2 : 1)
        eval(y)
        return y
    }

    private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    /// RMS over channels times √channels — upstream divides by the L2 norm,
    /// the same thing.
    private func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        rmsNorm(x, weight: weights["\(name).gamma"]!, eps: 1e-12)
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

    /// Upstream's `DupUp3D` for the first frame: each input channel repeated
    /// into consecutive output slots, read as (channel, frame, row, column)
    /// of a 2 × 2 pixel block, keeping the last frame. Gathered straight into
    /// the kept slots, so nothing larger than the output is built.
    private func duplicated(_ x: MLXArray, channels: Int, frames: Int) -> MLXArray {
        let (height, width, inputs) = (x.dim(1), x.dim(2), x.dim(3))
        let repeats = channels * frames * 4 / inputs
        var sources: [Int32] = []
        sources.reserveCapacity(channels * 4)
        for channel in 0 ..< channels {
            for slot in 0 ..< 4 {
                sources.append(Int32((((channel * frames) + frames - 1) * 4 + slot) / repeats))
            }
        }
        return take(x, MLXArray(sources), axis: 3)
            .reshaped(1, height, width, channels, 2, 2)
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped(1, 2 * height, 2 * width, channels)
    }

    /// The mid block's single-head attention over every latent pixel, in
    /// chunks of queries so the score matrix stays small.
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
            let chunk = scaledDotProductAttention(queries: q[0..., 0..., start ..< min(start + 2048, count)], keys: k,
                                                  values: v, scale: pow(Float(channels), -0.5), mask: nil)
            eval(chunk)
            chunks.append(chunk)
        }
        let attended = concatenated(chunks, axis: 2)[0..., 0]
        let out = matmul(attended, weights["\(name).proj.weight"]!.reshaped(channels, channels).T)
            + weights["\(name).proj.bias"]!
        return x + out.reshaped(1, height, width, channels)
    }
}
