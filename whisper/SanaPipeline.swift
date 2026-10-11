//  SANA-Sprint 0.6B, 1024 px (NVIDIA, Apache 2.0), with Gemma 2 2B (Google,
//  Gemma Terms of Use) reading the prompt. Converted by ports/imagegen/
//  convert_sana.py and export_sana_mlx.py; see ports/README.md.

import CoreGraphics
import CoreML
import Foundation
import MLX

/// A 1024 × 1024 picture in two steps.
///
/// Four stages, split between the GPU and the Neural Engine by what each
/// does best and what float16 can carry:
///
/// | stage | where | why |
/// |---|---|---|
/// | Gemma 2 reads the prompt | GPU (MLX), 8-bit, float32 activations | overflows float16 |
/// | the transformer, twice | GPU (Core ML), 8-bit palettized | see below |
/// | decoder to 128 px | GPU (MLX) | linear attention needs float32 sums |
/// | decoder to 1024 px | Neural Engine | plain convolutions, most of the work |
///
/// The transformer runs on the Neural Engine too, twice as fast per step
/// (0.26 s against 0.52 s on an M4 Mac), but Core ML took 11 s to load it
/// there on every run, compiled program cached or not; on the GPU it loads
/// in 2 s once and then at once. Two steps don't pay that back.
///
/// SANA-Sprint is distilled with sCM ("TrigFlow"): noise and picture are
/// mixed by an angle, the model predicts a velocity along it, and two steps
/// from pure noise (t = π/2, then 1.3) give the picture.
struct SanaPipeline {
    enum File {
        /// In parts: MLX can't view a file over 2 GB as one array.
        static let gemma = ["sana_gemma_q8.part1.safetensors", "sana_gemma_q8.part2.safetensors"]
        static let extras = "sana_extras.safetensors"
        static let tokenizer = "gemma_bpe.bin"
        static let transformer = "sana_transformer.mlmodelc"
        static let decoderHead = "sana_decoder_head.safetensors"
        static let decoderTail = "sana_decoder_tail.mlmodelc"
    }

    static let width = 1024
    static let height = 1024
    static let steps = 2
    /// Latents are a 32nd of the image on each side, 32 channels.
    private static let latentShape = [1, 32, 32, 32]
    private static let sigmaData = 0.5
    private static let latentScale = 0.41407
    /// The step angles (upstream rounds π/2 to 1.5708) and where the second
    /// step re-noises to.
    private static let angles = [1.5708, 1.3]

    let directory: URL
    /// SANA-Sprint 1.6B rather than 0.6B: its own transformer and caption
    /// projection, the same Gemma, decoder and tokenizer.
    var large = false

    private var transformerFile: String { large ? "sana16_transformer.mlmodelc" : File.transformer }
    private var extrasFile: String { large ? "sana16_extras.safetensors" : File.extras }

    /// - Parameter noise: the starting latents and the second step's noise,
    ///   instead of ones drawn from `seed` — for checking against the
    ///   reference with its own.
    func generate(prompt: String, seed: UInt64, noise: (start: [Float], step: [Float])? = nil,
                  isCancelled: () -> Bool = { false },
                  progress: (SDPipeline.Stage) -> Void = { _ in }) throws -> CGImage {
        let count = Self.latentShape.reduce(1, *)
        var random = SeededNormal(seed: seed)
        var latents = (noise?.start ?? random.sample(count: count)).map { $0 * Float(Self.sigmaData) }
        var prediction = latents

        try autoreleasepool {
            // The transformer first: its first load, while Core ML prepares
            // its GPU programs, is the slow part of a first run, and this way
            // "Loading" is what's on screen meanwhile. Gemma's weights are
            // mapped, not allocated, so holding both costs little.
            progress(.loading)
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndGPU
            let transformer = try MLModel(contentsOf: directory.appendingPathComponent(transformerFile),
                                          configuration: configuration)
            if isCancelled() { throw CancellationError() }

            // The prompt: Gemma, then SANA's caption projection, on the GPU.
            progress(.readingPrompt)
            let (caption, captionBias, modulation, embedded) = try StageMemory.withoutCache {
                () throws -> (MLMultiArray, MLMultiArray, [[Float]], [[Float]]) in
                let tokenizer = try GemmaTokenizer(url: directory.appendingPathComponent(File.tokenizer))
                let ids = SanaCaption.tokens(for: prompt, tokenizer: tokenizer)
                let gemma = Gemma2Encoder(weights: try Self.load(File.gemma, from: directory))
                let states = try gemma(ids, isCancelled: isCancelled)
                eval(states)
                let extras = try MappedWeights.load(url: directory.appendingPathComponent(extrasFile))
                let (projected, real) = SanaCaption.project(states, extras: extras)
                var bias = [Float](repeating: -10_000, count: SanaCaption.length)
                for index in 0 ..< real { bias[index] = 0 }
                return (try SDPipeline.multiArray(projected.asArray(Float.self), shape: [1, SanaCaption.length, projected.dim(2)]),
                        try SDPipeline.multiArray(bias, shape: [1, 1, SanaCaption.length]),
                        (0 ..< Self.steps).map { extras["modulation"]![$0].asArray(Float.self) },
                        (0 ..< Self.steps).map { extras["embedded"]![$0].asArray(Float.self) })
            }
            if isCancelled() { throw CancellationError() }

            // Two steps of the transformer.
            for step in 0 ..< Self.steps {
                if isCancelled() { throw CancellationError() }
                progress(.step(step + 1, of: Self.steps))
                let t = Self.angles[step]
                // sCM's input scaling for this angle.
                let s = sin(t) / (cos(t) + sin(t))
                let scale = (s * s + (1 - s) * (1 - s)).squareRoot()
                let x = latents.map { Float(Double($0) / Self.sigmaData * scale) }
                let output = try transformer.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                    "hidden": try SDPipeline.multiArray(x, shape: Self.latentShape),
                    "caption": caption, "caption_bias": captionBias,
                    "modulation": try SDPipeline.multiArray(modulation[step], shape: [1, modulation[step].count]),
                    "embedded": try SDPipeline.multiArray(embedded[step], shape: [1, embedded[step].count]),
                ]))
                guard let out = output.featureValue(for: "out")?.multiArrayValue else {
                    throw SDPipeline.Failure.noOutput("image model")
                }
                let f = SDPipeline.floats(out)
                // The model's output to a velocity, and a step back along
                // the angle to the clean picture.
                let (a, b) = (Float((1 - 2 * s) / scale * Self.sigmaData),
                              Float((1 - 2 * s + 2 * s * s) / scale * Self.sigmaData))
                let (c, d) = (Float(cos(t)), Float(sin(t)))
                for i in 0 ..< count { prediction[i] = c * latents[i] - d * (a * x[i] + b * f[i]) }
                if step + 1 < Self.steps {
                    // Re-noise to the next angle with fresh noise.
                    let next = Self.angles[step + 1]
                    let fresh = noise?.step ?? random.sample(count: count)
                    let (e, g) = (Float(cos(next)), Float(sin(next) * Self.sigmaData))
                    for i in 0 ..< count { latents[i] = e * prediction[i] + g * fresh[i] }
                }
                if isCancelled() { throw CancellationError() }
            }
        }

        // Decoding: to 128 px on the GPU, the rest on the Neural Engine.
        progress(.decoding)
        let half = try StageMemory.withoutCache { () throws -> MLMultiArray in
            let head = SanaDecoderHead(weights: try MappedWeights.load(url: directory.appendingPathComponent(File.decoderHead)))
            let z = MLXArray(prediction.map { Float(Double($0) / Self.sigmaData / Self.latentScale) }, Self.latentShape)
            let h = try head(z, isCancelled: isCancelled).transposed(0, 3, 1, 2)
            eval(h)
            let shape = [1, h.dim(1), h.dim(2), h.dim(3)]
            let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
            let values = h.asArray(Float16.self)
            array.withUnsafeMutableBufferPointer(ofType: Float16.self) { out, _ in
                _ = out.initialize(from: values)
            }
            return array
        }
        if isCancelled() { throw CancellationError() }
        return try autoreleasepool {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            let tail = try MLModel(contentsOf: directory.appendingPathComponent(File.decoderTail),
                                   configuration: configuration)
            let output = try tail.prediction(from: MLDictionaryFeatureProvider(dictionary: ["hidden": half]))
            guard let image = output.featureValue(for: "image")?.multiArrayValue else {
                throw SDPipeline.Failure.noOutput("decoder")
            }
            return try RGBImage.make(planar: SDPipeline.floats(image), width: Self.width, height: Self.height)
        }
    }

    static func load(_ parts: [String], from directory: URL) throws -> [String: MLXArray] {
        try parts.reduce(into: [:]) { weights, part in
            weights.merge(try MappedWeights.load(url: directory.appendingPathComponent(part))) { first, _ in first }
        }
    }
}
