import Foundation
import MLX

/// Assembles ACE-Step's three conditioning streams into one sequence.
///
/// Lyrics, timbre and text are encoded separately and then *packed*: the
/// streams are concatenated and stably sorted so every valid token precedes
/// every padding one. That ordering matters — the diffusion transformer
/// cross-attends to this sequence, and interleaved padding would leave holes
/// mid-sequence that the mask has to work around.
struct ACEConditioning {
    let weights: [String: MLXArray]
    var quantizationBits: Int = 0

    private var lyricEncoder: ACEEncoderStack {
        var stack = ACEEncoderStack(weights: weights, prefix: "encoder.lyric_encoder", layerCount: 8)
        stack.quantizationBits = quantizationBits
        return stack
    }

    private var timbreEncoder: ACEEncoderStack {
        var stack = ACEEncoderStack(weights: weights, prefix: "encoder.timbre_encoder", layerCount: 4)
        stack.quantizationBits = quantizationBits
        return stack
    }

    /// - Parameters:
    ///   - text: (1, n, 1024) from the Qwen3 embedder.
    ///   - lyric: (1, n, 1024) lyric embeddings, or nil for instrumental.
    ///   - reference: (1, n, 64) acoustic latents of a reference clip, or nil.
    /// - Returns: the packed sequence and its mask.
    func encode(text: MLXArray,
                textProjection: MLXArray,
                lyric: MLXArray?,
                reference: MLXArray?) -> (MLXArray, MLXArray) {
        let projected = matmul(text, textProjection.asType(.float32).T)

        var streams: [(MLXArray, Int)] = []
        if let lyric {
            streams.append((lyricEncoder(lyric), lyric.dim(1)))
        }
        if let reference {
            // The learned lead token's final state is the timbre summary, so
            // only position zero survives.
            let encoded = timbreEncoder(reference)
            streams.append((encoded[0..., 0 ..< 1, 0...], 1))
        }
        streams.append((projected, text.dim(1)))

        let sequence = concatenated(streams.map { $0.0 }, axis: 1)
        let total = streams.reduce(0) { $0 + $1.1 }
        let mask = MLXArray([Int32](repeating: 1, count: total), [1, total])
        return (sequence, mask)
    }
}
