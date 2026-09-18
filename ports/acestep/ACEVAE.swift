import Foundation
import MLX

/// ACE-Step's audio decoder — diffusers' Oobleck autoencoder, decode side.
///
/// Latents at 25 Hz become 48 kHz stereo through five upsampling blocks whose
/// strides multiply to 1920. Two details are specific to this family:
///
/// * **Snake activation**, `x + sin(αx)² / β`, with per-channel α and β. It is
///   periodic, which is what lets the decoder produce oscillations rather than
///   the smooth curves a ReLU family gives.
/// * **Residual units trim their skip connection.** The dilated 7-tap
///   convolution returns a shorter sequence than it was given, so the residual
///   is centre-cropped to match instead of padded.
struct ACEVAE {
    /// Reversed relative to the config's downsampling ratios — the weight
    /// shapes confirm it: the first block's kernel is 20, so its stride is 10.
    static let strides = [10, 6, 4, 4, 2]
    static let channels = 128
    static let audioChannels = 2

    let weights: [String: MLXArray]

    private func w(_ key: String) -> MLXArray {
        guard let value = weights[key] else { fatalError("missing VAE weight: \(key)") }
        return value.asType(.float32)
    }
    private func maybe(_ key: String) -> MLXArray? { weights[key]?.asType(.float32) }

    /// - Parameter latent: (1, 64, frames), channels-first as the checkpoint expects.
    /// - Returns: (1, 2, samples) stereo audio.
    func decode(latent: MLXArray) -> MLXArray {
        // MLX convolves over (batch, length, channels); the model is stored
        // channels-first, so transpose once here and once at the end.
        var h = latent.transposed(0, 2, 1)
        h = convolve(h, prefix: "decoder.conv1", kernel: 7, dilation: 1)

        for (index, stride) in Self.strides.enumerated() {
            let prefix = "decoder.block.\(index)"
            h = snake(h, prefix: "\(prefix).snake1")
            h = upsample(h, prefix: "\(prefix).conv_t1", stride: stride)
            for (unit, dilation) in [(1, 1), (2, 3), (3, 9)] {
                h = residualUnit(h, prefix: "\(prefix).res_unit\(unit)", dilation: dilation)
            }
        }

        h = snake(h, prefix: "decoder.snake1")
        h = convolve(h, prefix: "decoder.conv2", kernel: 7, dilation: 1)
        return h.transposed(0, 2, 1)
    }

    // MARK: - Pieces

    private func residualUnit(_ x: MLXArray, prefix: String, dilation: Int) -> MLXArray {
        var h = convolve(snake(x, prefix: "\(prefix).snake1"),
                         prefix: "\(prefix).conv1", kernel: 7, dilation: dilation)
        h = convolve(snake(h, prefix: "\(prefix).snake2"),
                     prefix: "\(prefix).conv2", kernel: 1, dilation: 1)
        // Centre-crop the skip when the convolution shortened the sequence.
        let trim = (x.dim(1) - h.dim(1)) / 2
        let skip = trim > 0 ? x[0..., trim ..< (x.dim(1) - trim), 0...] : x
        return skip + h
    }

    private func convolve(_ x: MLXArray, prefix: String, kernel: Int, dilation: Int) -> MLXArray {
        let padding = ((kernel - 1) * dilation) / 2
        var y = conv1d(x, w("\(prefix).weight"), stride: 1, padding: padding, dilation: dilation)
        if let bias = maybe("\(prefix).bias") { y = y + bias }
        return y
    }

    private func upsample(_ x: MLXArray, prefix: String, stride: Int) -> MLXArray {
        let kernel = 2 * stride
        let padding = Int(ceil(Double(stride) / 2))
        var y = convTransposed1d(x, w("\(prefix).weight"), stride: stride, padding: padding)
        if let bias = maybe("\(prefix).bias") { y = y + bias }
        _ = kernel
        return y
    }

    /// `x + sin(alpha * x)^2 / beta`, per channel.
    ///
    /// Oobleck's Snake defaults to `logscale=True`, so the checkpoint stores
    /// the logarithms of alpha and beta — they must be exponentiated before
    /// use. Reading them raw flips the sign of beta wherever it is negative
    /// and the decoder output explodes rather than failing outright.
    private func snake(_ x: MLXArray, prefix: String) -> MLXArray {
        // Stored as (1, channels, 1); the working layout is (batch, length, channels).
        let alpha = exp(w("\(prefix).alpha").reshaped(1, 1, -1))
        let beta = exp(w("\(prefix).beta").reshaped(1, 1, -1))
        let s = sin(alpha * x)
        return x + (s * s) / (beta + 1e-9)
    }
}
