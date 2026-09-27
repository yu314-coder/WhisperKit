//  Ported from ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5), MIT.
//  Verified against the PyTorch reference stage by stage; see ports/README.md.

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

    /// `encoder.text_projector`: 1024 to 2048, no bias.
    private func projectText(_ text: MLXArray) -> MLXArray {
        let key = "encoder.text_projector.weight"
        if quantizationBits > 0, let packed = weights["\(key).wq"] {
            return quantizedMatmul(text, packed,
                                   scales: weights["\(key).scales"]!,
                                   biases: weights["\(key).biases"]!,
                                   transpose: true, groupSize: 64, bits: quantizationBits)
        }
        return matmul(text, weights[key]!.asType(.float32).T)
    }

    /// Every stream is always present. A song without words still sends its
    /// lyric slot — the text "[Instrumental]" — and a prompt without a
    /// reference clip still sends a timbre, read from the silence bed. That is
    /// how the model was trained; leaving either out, as this once did,
    /// conditions it on a sequence shape it never saw.
    ///
    /// - Parameters:
    ///   - text: (1, n, 1024) caption states from the Qwen3 embedder.
    ///   - lyric: (1, n, 1024) lyric token embeddings — table rows looked up,
    ///     not run through the embedder.
    ///   - timbre: (1, frames, 64) acoustic latents: a reference clip, or the
    ///     first 750 frames of silence when there is none.
    /// - Returns: (1, lyric + 1 + text, 2048). Packing sorts valid tokens
    ///   ahead of padding; with none padded it is plain concatenation in the
    ///   order lyric, timbre, text.
    func encode(text: MLXArray, lyric: MLXArray, timbre: MLXArray) -> MLXArray {
        let lyricStates = lyricEncoder(lyric)
        // The timbre encoder's summary is its first position.
        let timbreSummary = timbreEncoder(timbre)[0..., 0 ..< 1, 0...]
        return concatenated([lyricStates, timbreSummary, projectText(text)], axis: 1)
    }
}
