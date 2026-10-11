//  FLUX.2 [klein] 4B (Black Forest Labs, Apache 2.0), with Qwen3-4B
//  (Apache 2.0) reading the prompt. Converted by ports/imagegen/
//  export_flux_mlx.py; see ports/README.md.

import CoreGraphics
import Foundation
import MLX

/// FLUX.2 [klein] 4B: four flow-matching steps, all on the GPU.
///
/// | stage | precision |
/// |---|---|
/// | Qwen3-4B reads the prompt (27 of 36 layers) | 8-bit, float32 activations |
/// | the transformer, four times | 8-bit, bfloat16 |
/// | the VAE decoder | float16, float32 norm statistics |
///
/// The prompt goes in Qwen's chat template, padded to 512 tokens; FLUX
/// reads the hidden states after layers 9, 18 and 27 side by side, padding
/// included. The step-distilled model needs no guidance, so each step is
/// one pass.
struct FluxPipeline {
    enum File {
        static let vocabulary = "flux_vocab.json"
        static let merges = "flux_merges.txt"
        /// In parts: MLX can't view a file over 2 GB as one array.
        static let text = ["flux_text_q8.part1.safetensors", "flux_text_q8.part2.safetensors"]
        static let transformer = ["flux_transformer_q8.part1.safetensors", "flux_transformer_q8.part2.safetensors",
                                  "flux_transformer_q8.part3.safetensors"]
        static let vae = "flux_vae.safetensors"
    }

    static let steps = 4
    static let textLength = 512
    private static let textLayers = [9, 18, 27]

    let directory: URL

    /// - Parameters:
    ///   - width, height: multiples of 16.
    ///   - noise: the starting latents, (patches × 128) row-major, instead
    ///     of ones drawn from `seed` — for checking against the reference.
    func generate(prompt: String, width: Int, height: Int, seed: UInt64, noise: [Float]? = nil,
                  isCancelled: () -> Bool = { false },
                  progress: (SDPipeline.Stage) -> Void = { _ in }) throws -> CGImage {
        // The prompt.
        progress(.readingPrompt)
        let context = try StageMemory.withoutCache { () throws -> MLXArray in
            let tokenizer = try ACETokenizer(vocabularyURL: directory.appendingPathComponent(File.vocabulary),
                                             mergesURL: directory.appendingPathComponent(File.merges))
            let (ids, valid) = Self.tokens(for: prompt, tokenizer: tokenizer)
            var encoder = ACEQwen3(weights: try SanaPipeline.load(File.text, from: directory),
                                   config: .fluxText, prefix: "")
            encoder.quantizationBits = 8
            let states = try encoder.states(inputIDs: MLXArray(ids).reshaped(1, -1), valid: valid,
                                            after: Self.textLayers, isCancelled: isCancelled)
            let context = concatenated(states, axis: -1)[0].asType(.bfloat16)
            eval(context)
            return context
        }

        // Four steps.
        let (rows, columns) = (height / 16, width / 16)
        let patches = rows * columns
        var random = SeededNormal(seed: seed)
        var latents = MLXArray(noise ?? random.sample(count: patches * FluxTransformer.channels),
                               [patches, FluxTransformer.channels])
        let sigmas = FluxSchedule.sigmas(imagePatches: patches, steps: Self.steps)
        try StageMemory.withoutCache {
            let transformer = FluxTransformer(weights: try SanaPipeline.load(File.transformer, from: directory))
            let rotary = FluxTransformer.Rotary(rows: rows, columns: columns, textLength: Self.textLength)
            for step in 0 ..< Self.steps {
                if isCancelled() { throw CancellationError() }
                progress(.step(step + 1, of: Self.steps))
                let velocity = try transformer(latents, context: context, sigma: sigmas[step], rotary: rotary,
                                               isCancelled: isCancelled)
                // An Euler step along the flow to the next noise level.
                latents = latents + (sigmas[step + 1] - sigmas[step]) * velocity
                eval(latents)
            }
        }
        if isCancelled() { throw CancellationError() }

        // The picture.
        progress(.decoding)
        let planar = try StageMemory.withoutCache { () throws -> [Float] in
            let decoder = FluxVAEDecoder(weights: try MappedWeights.load(url: directory.appendingPathComponent(File.vae)))
            return try decoder(latents, height: height, width: width, isCancelled: isCancelled).asArray(Float.self)
        }
        return try RGBImage.make(planar: planar, width: width, height: height)
    }

    /// The prompt in Qwen's chat template, thinking switched off, as
    /// upstream applies it; padded with `<|endoftext|>` to 512 tokens.
    /// Returns the ids and how many are real.
    static func tokens(for prompt: String, tokenizer: ACETokenizer) -> ([Int32], Int) {
        let text = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        let ids = Array(tokenizer.encode(text, appendEndOfText: false).prefix(textLength))
        return (ids + [Int32](repeating: ACETokenizer.endOfText, count: textLength - ids.count), ids.count)
    }
}
