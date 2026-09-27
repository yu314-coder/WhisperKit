//  Ported from ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5), MIT.
//  Verified against the PyTorch reference stage by stage; see ports/README.md.

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
    /// Samples per latent frame: the product of the strides.
    static let hopLength = 1920

    let weights: [String: MLXArray]
    /// What the decoder computes in. Weights are stored as float16.
    var dtype: DType = .float32
    /// Routes upsampling through MLX's transposed convolution, for checking.
    var usesReferenceUpsample = false

    private func w(_ key: String) -> MLXArray {
        guard let value = weights[key] else { fatalError("missing VAE weight: \(key)") }
        return value.asType(dtype)
    }
    private func maybe(_ key: String) -> MLXArray? { weights[key]?.asType(dtype) }

    /// - Parameter latent: (1, 64, frames), channels-first as the checkpoint expects.
    /// - Returns: (1, 2, samples) stereo audio.
    func decode(latent: MLXArray) -> MLXArray {
        // MLX convolves over (batch, length, channels); the model is stored
        // channels-first, so transpose once here and once at the end.
        run(latent.transposed(0, 2, 1)).transposed(0, 2, 1)
    }

    /// Decodes in overlapping windows and hands each finished stretch of
    /// audio to `emit`, so a long clip never exists at full rate inside the
    /// decoder.
    ///
    /// The last stage works at 48 kHz across 128 channels: 740 MB per
    /// intermediate for a 30-second clip decoded whole, several of them live
    /// at once. Windows of 48 frames plus 12 either side cap the working set
    /// regardless of length.
    ///
    /// Overlap-discard, as the reference does it: each window decodes with
    /// context on both sides and only its middle is kept. The decoder's reach
    /// is under 12 frames — measured, 12 reproduces a whole-clip decode
    /// exactly and 8 does not — so the join is not merely inaudible but
    /// absent.
    ///
    /// - Parameter latent: (1, frames, 64), frames-first as diffusion leaves it.
    /// - Parameter emit: (1, samples, 2) audio and the fraction done.
    func decodeTiled(latent: MLXArray, core: Int = 48, overlap: Int = 12,
                     emit: (MLXArray, Double) throws -> Void) rethrows {
        let frames = latent.dim(1)
        var start = 0
        while start < frames {
            let end = min(start + core, frames)
            let lower = max(0, start - overlap)
            let upper = min(frames, end + overlap)
            let audio = run(latent[0..., lower ..< upper, 0...].asType(dtype)).asType(.float32)
            let kept = audio[0..., ((start - lower) * Self.hopLength) ..< ((end - lower) * Self.hopLength), 0...]
            eval(kept)
            try emit(kept, Double(end) / Double(frames))
            start = end
        }
    }

    /// (1, frames, 64) to (1, frames × 1920, 2).
    private func run(_ latent: MLXArray) -> MLXArray {
        var h = convolve(latent, prefix: "decoder.conv1", kernel: 7, dilation: 1)

        for (index, stride) in Self.strides.enumerated() {
            let prefix = "decoder.block.\(index)"
            h = snake(h, prefix: "\(prefix).snake1")
            h = usesReferenceUpsample
                ? referenceUpsample(h, prefix: "\(prefix).conv_t1", stride: stride)
                : upsample(h, prefix: "\(prefix).conv_t1", stride: stride)
            for (unit, dilation) in [(1, 1), (2, 3), (3, 9)] {
                h = residualUnit(h, prefix: "\(prefix).res_unit\(unit)", dilation: dilation)
            }
        }

        h = snake(h, prefix: "decoder.snake1")
        return convolve(h, prefix: "decoder.conv2", kernel: 7, dilation: 1)
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

    /// Transposed convolution as one matrix product and a shift.
    ///
    /// MLX lowers a strided transposed convolution to an explicit unfold of
    /// its input. For the fourth block of a five-second window that is a 1 GB
    /// temporary to produce 60 MB of output, and it was most of what the
    /// decoder cost in memory. Here every kernel is exactly twice its stride,
    /// so each output sample draws on just two input frames: multiply every
    /// frame by all taps at once, then add each frame's overhang into its
    /// neighbours. Same arithmetic, no unfold.
    ///
    /// With padding p = stride/2, output phase r of frame i takes tap r+p
    /// from frame i, plus tap r+p+s from frame i-1 when r+p < s, or tap
    /// r+p-s from frame i+1 otherwise.
    ///
    /// The weights are stored ready for this — (in, taps, out) under
    /// `.taps` — because transposing them here happened once per window: an
    /// 84 MB copy for the first block alone, dozens of times per clip.
    private func upsample(_ x: MLXArray, prefix: String, stride s: Int) -> MLXArray {
        let weight = w("\(prefix).taps")     // (in, 2s, out)
        let (inChannels, taps, outChannels) = (weight.dim(0), weight.dim(1), weight.dim(2))
        precondition(taps == 2 * s && s % 2 == 0, "upsample assumes kernel = 2 × even stride")
        let p = s / 2
        let length = x.dim(1)

        // z[i, k] = x[i] · W_k, for every frame i and tap k.
        let z = matmul(x, weight.reshaped(inChannels, taps * outChannels))
            .reshaped(1, length, taps, outChannels)

        let own = z[0..., 0..., p ..< (p + s), 0...]
        let fromPrevious = z[0..., 0 ..< (length - 1), (p + s) ..< taps, 0...]
        let fromNext = z[0..., 1 ..< length, 0 ..< p, 0...]
        let early = concatenated([MLXArray.zeros([1, 1, s - p, outChannels], dtype: z.dtype), fromPrevious],
                                 axis: 1)
        let late = concatenated([fromNext, MLXArray.zeros([1, 1, p, outChannels], dtype: z.dtype)],
                                axis: 1)

        var y = (own + concatenated([early, late], axis: 2)).reshaped(1, length * s, outChannels)
        if let bias = maybe("\(prefix).bias") { y = y + bias }
        return y
    }

    /// The same, through MLX's own transposed convolution — kept to check
    /// `upsample` against.
    private func referenceUpsample(_ x: MLXArray, prefix: String, stride: Int) -> MLXArray {
        let padding = Int(ceil(Double(stride) / 2))
        var y = convTransposed1d(x, w("\(prefix).taps").transposed(2, 1, 0), stride: stride, padding: padding)
        if let bias = maybe("\(prefix).bias") { y = y + bias }
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
        return Self.fusedSnake(x, alpha, beta)
    }

    /// One kernel instead of five. Unfused, each step of the formula wrote a
    /// full-size intermediate — at 48 kHz across 128 channels, several
    /// hundred megabytes live at once for a single window.
    private static let fusedSnake = compile(shapeless: true) {
        (x: MLXArray, alpha: MLXArray, beta: MLXArray) -> MLXArray in
        let s = sin(alpha * x)
        return x + (s * s) / (beta + 1e-9)
    }
}
