//  DC-AE (MIT Han Lab, Apache 2.0), SANA's autoencoder: the first half of its
//  decoder. Weights from ports/imagegen/export_sana_mlx.py; the second half
//  is a Core ML program (convert_sana.py decoder_tail).

import Foundation
import MLX

/// DC-AE's decoder from the 32 × 32 latent up to 128 × 128 × 512, on the GPU.
///
/// These are the stages with linear attention ("EfficientViT" blocks), whose
/// sums over every position reach 400,000 — past float16, and the Neural
/// Engine's matmul got them wrong outright. Here the attention is computed
/// in float32 and everything else in float16. The stages after it are only
/// convolutions, 90% of the decoder's work, and run on the Neural Engine.
///
/// Channels-last throughout, as MLX convolves.
struct SanaDecoderHead {
    let weights: [String: MLXArray]

    private static let latentChannels = 32
    /// Channels of up blocks 3, 4 and 5, which this runs, last first.
    private static let channels = [3: 512, 4: 1024, 5: 1024]
    private static let headDimension = 32
    /// Bytes of unfolded input one convolution call may make. MLX unfolds a
    /// 3 × 3 convolution's whole input nine times over; past this the image
    /// is convolved in strips of rows.
    private static let unfoldBudget = 256 << 20

    /// - Parameter latent: (1, 32, 32, 32) channels-first, already divided
    ///   by the VAE's scaling factor.
    /// - Returns: (1, 128, 128, 512), float16, channels-last.
    /// `isCancelled` is asked before each block.
    func callAsFunction(_ latent: MLXArray, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        var x = latent.transposed(0, 2, 3, 1).asType(.float16)
        // The input shortcut: each latent channel repeated to fill 1,024.
        x = conv(x, "conv_in") + repeated(x, count: 1024 / Self.latentChannels, axis: 3)
        for block in [5, 4, 3] {
            if block < 5 { x = upsample(x, block: block, to: Self.channels[block]!) }
            for layer in 0 ..< 3 {
                if isCancelled() { throw CancellationError() }
                x = vitBlock(x, "up_blocks.\(block).\(layer + (block < 5 ? 1 : 0))")
                eval(x)
            }
        }
        return x
    }

    // MARK: - Pieces

    private func conv(_ x: MLXArray, _ name: String, padding: Int = 1, groups: Int = 1) -> MLXArray {
        let weight = weights["\(name).weight"]!
        let (height, width, channels) = (x.dim(1), x.dim(2), x.dim(3))
        let kernel = weight.dim(1) * weight.dim(2)
        let rowBytes = width * channels * kernel * 2
        var y: MLXArray
        if groups == 1, kernel > 1, rowBytes * height > Self.unfoldBudget {
            let rows = max(1, Self.unfoldBudget / rowBytes)
            let padded = padded(x, widths: [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            var strips: [MLXArray] = []
            for start in stride(from: 0, to: height, by: rows) {
                let end = min(start + rows, height)
                let strip = conv2d(padded[0..., start ..< end + 2 * padding], weight)
                eval(strip)
                strips.append(strip)
            }
            y = concatenated(strips, axis: 1)
        } else {
            y = conv2d(x, weight, padding: [padding, padding], groups: groups)
        }
        if let bias = weights["\(name).bias"] { y = y + bias }
        return y
    }

    /// DC-AE's RMSNorm over channels, with a bias. MLX's fused norm squares
    /// and sums in float32, so float16 input can't overflow it.
    private func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        rmsNorm(x, weight: weights["\(name).weight"]!, eps: 1e-5) + weights["\(name).bias"]!
    }

    private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    /// DCUpBlock2d: a convolution of the nearest-neighbour upsampling, plus a
    /// shortcut of the input's channels repeated and pixel-shuffled.
    private func upsample(_ x: MLXArray, block: Int, to outChannels: Int) -> MLXArray {
        let (height, width, channels) = (x.dim(1), x.dim(2), x.dim(3))
        let shortcut = repeated(x, count: outChannels * 4 / channels, axis: 3)
            .reshaped(1, height, width, outChannels, 2, 2)
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped(1, 2 * height, 2 * width, outChannels)
        let nearest = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
        return conv(nearest, "up_blocks.\(block).0.conv") + shortcut
    }

    /// EfficientViTBlock: multi-scale linear attention, then a gated
    /// depthwise feed-forward, each with a norm and a residual.
    private func vitBlock(_ x: MLXArray, _ name: String) -> MLXArray {
        let (height, width, channels) = (x.dim(1), x.dim(2), x.dim(3))
        let attention = "\(name).attn"
        func project(_ key: String) -> MLXArray { matmul(x, weights["\(attention).\(key).weight"]!.T) }
        let qkv = concatenated([project("to_q"), project("to_k"), project("to_v")], axis: 3)
        // The second scale: a 5 × 5 depthwise convolution of q, k and v, then
        // a 1 × 1 one within each head's 32 channels.
        var scaled = conv(qkv, "\(attention).to_qkv_multiscale.0.proj_in", padding: 2, groups: 3 * channels)
        scaled = conv(scaled, "\(attention).to_qkv_multiscale.0.proj_out", padding: 0,
                      groups: 3 * channels / Self.headDimension)
        // Upstream regroups both scales' channels into heads of 96 — query,
        // key, value of 32 each — in the order they were concatenated.
        let d = Self.headDimension
        let heads = concatenated([qkv, scaled], axis: 3).asType(.float32)
            .reshaped(1, height * width, -1, 3 * d)
            .transposed(0, 2, 1, 3)                                  // (1, heads, positions, 96)
        let q = maximum(heads[.ellipsis, 0 ..< d], 0)
        let k = maximum(heads[.ellipsis, d ..< 2 * d], 0)
        let v = heads[.ellipsis, (2 * d)...]
        // ReLU linear attention: q (kᵀv) / q (kᵀ1), summed over positions.
        let numerator = matmul(q, matmul(k.transposed(0, 1, 3, 2), v))
        let denominator = matmul(q, k.sum(axis: 2, keepDims: true).transposed(0, 1, 3, 2))
        let attended = (numerator / (denominator + 1e-15))
            .transposed(0, 2, 1, 3)
            .reshaped(1, height, width, -1)
            .asType(.float16)
        let out = norm(matmul(attended, weights["\(attention).to_out.weight"]!.T), "\(attention).norm_out") + x
        eval(out)

        let feed = "\(name).conv_out"
        var g = silu(conv(out, "\(feed).conv_inverted", padding: 0))
        g = conv(g, "\(feed).conv_depth", padding: 1, groups: g.dim(3))
        let (hidden, gate) = split(g, axis: 3)
        g = conv(hidden * silu(gate), "\(feed).conv_point", padding: 0)
        return norm(g, "\(feed).norm") + out
    }
}
