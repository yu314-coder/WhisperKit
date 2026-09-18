import Foundation
import MLX

/// Autoregressive sampling for MusicGen.
///
/// The four codebooks are not predicted in lockstep. Codebook q is delayed by q
/// steps, so at column s it carries the token for frame s - q and anything
/// before that is the special token. That staggering is what lets one
/// transformer predict four streams without them having to agree on a frame
/// simultaneously — and getting the offset wrong produces audio that decodes
/// without error but sounds like noise.
///
/// Guidance runs the decoder twice per step: once on the prompt, once on an
/// empty conditioning, combining as `uncond + scale * (cond - uncond)`.
struct MGGenerator {
    static let specialToken: Int32 = 2048
    static let codebookCount = 4

    let decoder: MGDecoder

    func generate(
        encoderHidden: MLXArray,
        encoderMask: MLXArray,
        columns: Int,
        guidanceScale: Float = 3.0,
        onStep: ((Int, Int) -> Void)? = nil
    ) -> MLXArray {
        let frames = columns - (Self.codebookCount - 1)

        // Unconditional branch: zeros, with a mask that hides every position.
        let emptyHidden = MLXArray.zeros(encoderHidden.shape, dtype: .float32)
        let emptyMask = MLXArray.zeros(encoderMask.shape, dtype: encoderMask.dtype)

        // Column 0 is the start token for every codebook.
        var tokens = [[Int32]](repeating: [Self.specialToken], count: Self.codebookCount)

        // One cache per guidance branch. Each step feeds only the column just
        // produced; without this every step re-ran the whole prefix, which is
        // quadratic and was four times slower than the reference.
        let conditionalCache = MGDecoderCache(layerCount: MGDecoder.layerCount)
        let unconditionalCache = MGDecoderCache(layerCount: MGDecoder.layerCount)

        for step in 0 ..< columns {
            var column: [Int32] = []
            column.reserveCapacity(Self.codebookCount)
            for book in 0 ..< Self.codebookCount { column.append(tokens[book][step]) }
            let input = MLXArray(column, [Self.codebookCount, 1])

            let conditional = decoder(tokens: input, encoderHidden: encoderHidden,
                                      encoderMask: encoderMask, offset: step,
                                      cache: conditionalCache)
            let unconditional = decoder(tokens: input, encoderHidden: emptyHidden,
                                        encoderMask: emptyMask, offset: step,
                                        cache: unconditionalCache)
            let condLast = conditional[0..., 0, 0...]
            let uncondLast = unconditional[0..., 0, 0...]
            let guided = uncondLast + (condLast - uncondLast) * guidanceScale
            // One read of all four, rather than four separate `item()` calls:
            // each of those is a GPU sync, and four per step is most of the
            // per-step overhead at this size.
            let choice = argMax(guided, axis: -1).asArray(Int32.self)

            for book in 0 ..< Self.codebookCount {
                // Before its delay has elapsed a codebook emits nothing real.
                tokens[book].append(step >= book ? choice[book] : Self.specialToken)
            }
            onStep?(step + 1, columns)
        }

        // Undo the delay: frame f of codebook q sits at column 1 + f + q.
        var codes = [Int32](repeating: 0, count: Self.codebookCount * frames)
        for book in 0 ..< Self.codebookCount {
            for frame in 0 ..< frames {
                codes[book * frames + frame] = tokens[book][1 + frame + book]
            }
        }
        return MLXArray(codes, [Self.codebookCount, frames])
    }
}
