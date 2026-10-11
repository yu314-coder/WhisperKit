//  DreamShaper 8 (Stable Diffusion 1.5, CreativeML OpenRAIL-M) with ByteDance's
//  Hyper-SD 4-step distillation (CreativeML OpenRAIL++-M) fused in. Converted by
//  ports/imagegen/convert_sd.py; see ports/README.md.

import CoreGraphics
import CoreML
import Foundation

/// Stable Diffusion on the Neural Engine, in four steps.
///
/// Hyper-SD distils the model to four steps without guidance, so each step
/// is one pass of the UNet on one image — no second, unconditioned pass —
/// and the whole run is the text encoder, four UNet passes and the decoder.
/// All three are Core ML programs; nothing here needs the GPU, so a run
/// carries on in the background on any device.
struct SDPipeline {
    enum Stage {
        case loading
        case readingPrompt
        case step(Int, of: Int)
        case decoding
    }

    enum File {
        static let textEncoder = "sd_text_encoder.mlmodelc"
        static let unet = "sd_unet.mlmodelc"
        static let decoder = "sd_vae_decoder.mlmodelc"
        static let vocabulary = "clip_vocab.json"
        static let merges = "clip_merges.txt"
    }

    enum Failure: LocalizedError {
        case noOutput(String)

        var errorDescription: String? {
            switch self {
            case .noOutput(let stage): return "The \(stage) returned no result."
            }
        }
    }

    static let width = 512
    static let height = 512
    static let steps = 4
    /// Latents are an eighth of the image on each side, four channels.
    private static let latentShape = [1, 4, height / 8, width / 8]
    /// SD 1.5's VAE scaling.
    private static let latentScale: Float = 0.18215

    let directory: URL

    /// - Parameter noise: the starting latents, instead of ones drawn from
    ///   `seed` — for checking against the reference with its own noise.
    func generate(prompt: String, seed: UInt64, noise: [Float]? = nil,
                  isCancelled: () -> Bool = { false },
                  progress: (Stage) -> Void = { _ in }) throws -> CGImage {
        let configuration = MLModelConfiguration()
        // As for the music transformer: not .all, so an operation the Neural
        // Engine turns down falls back to the CPU, never the iPhone GPU.
        configuration.computeUnits = .cpuAndNeuralEngine

        progress(.loading)
        let tokenizer = try CLIPTokenizer(vocabularyURL: directory.appendingPathComponent(File.vocabulary),
                                          mergesURL: directory.appendingPathComponent(File.merges))
        // One model in memory at a time: the text encoder is done before the
        // UNet loads, and the UNet before the decoder.
        progress(.readingPrompt)
        let hidden: MLMultiArray = try autoreleasepool {
            let encoder = try MLModel(contentsOf: directory.appendingPathComponent(File.textEncoder),
                                      configuration: configuration)
            let ids = tokenizer.encode(prompt).map { Float($0) }
            let input = try MLMultiArray(shape: [1, NSNumber(value: CLIPTokenizer.length)], dataType: .float32)
            for (index, id) in ids.enumerated() { input[index] = NSNumber(value: id) }
            let output = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": input]))
            guard let states = output.featureValue(for: "last_hidden_state")?.multiArrayValue else {
                throw Failure.noOutput("text encoder")
            }
            return try Self.hiddenStates(states)
        }
        if isCancelled() { throw CancellationError() }

        var random = SeededNormal(seed: seed)
        var latents = noise ?? random.sample(count: Self.latentShape.reduce(1, *))
        let schedule = DDIMSchedule(steps: Self.steps)
        try autoreleasepool {
            let unet = try MLModel(contentsOf: directory.appendingPathComponent(File.unet), configuration: configuration)
            for (index, timestep) in schedule.timesteps.enumerated() {
                progress(.step(index + 1, of: Self.steps))
                let sample = try Self.multiArray(latents, shape: Self.latentShape)
                let time = try MLMultiArray(shape: [1], dataType: .float32)
                time[0] = NSNumber(value: Float(timestep))
                let output = try unet.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                    "sample": sample, "timestep": time, "encoder_hidden_states": hidden,
                ]))
                guard let noise = output.featureValue(for: "noise_pred")?.multiArrayValue else {
                    throw Failure.noOutput("image model")
                }
                latents = schedule.step(sample: latents, noise: Self.floats(noise), at: index)
                if isCancelled() { throw CancellationError() }
            }
        }

        progress(.decoding)
        return try autoreleasepool {
            let decoder = try MLModel(contentsOf: directory.appendingPathComponent(File.decoder),
                                      configuration: configuration)
            let z = try Self.multiArray(latents.map { $0 / Self.latentScale }, shape: Self.latentShape)
            let output = try decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["z": z]))
            guard let image = output.featureValue(for: "image")?.multiArrayValue else {
                throw Failure.noOutput("decoder")
            }
            return try RGBImage.make(planar: Self.floats(image), width: Self.width, height: Self.height)
        }
    }

    // MARK: - Arrays

    /// (1, 77, 768) from the text encoder, as the UNet takes it: (1, 768, 1, 77).
    private static func hiddenStates(_ states: MLMultiArray) throws -> MLMultiArray {
        let values = floats(states)
        let (tokens, width) = (states.shape[1].intValue, states.shape[2].intValue)
        let result = try MLMultiArray(shape: [1, NSNumber(value: width), 1, NSNumber(value: tokens)], dataType: .float32)
        result.withUnsafeMutableBufferPointer(ofType: Float.self) { out, _ in
            for token in 0 ..< tokens {
                for channel in 0 ..< width {
                    out[channel * tokens + token] = values[token * width + channel]
                }
            }
        }
        return result
    }

    static func multiArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        array.withUnsafeMutableBufferPointer(ofType: Float.self) { out, _ in
            for index in 0 ..< values.count { out[index] = values[index] }
        }
        return array
    }

    /// Any multiarray's values in row-major order, as Float, honouring the
    /// padded strides the Neural Engine can return.
    static func floats(_ array: MLMultiArray) -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let count = shape.reduce(1, *)
        var result = [Float](repeating: 0, count: count)
        func read<T: MLShapedArrayScalar>(_ type: T.Type, _ convert: (T) -> Float) {
            array.withUnsafeBufferPointer(ofType: type) { source in
                var index = [Int](repeating: 0, count: shape.count)
                for flat in 0 ..< count {
                    var offset = 0
                    for axis in 0 ..< shape.count { offset += index[axis] * strides[axis] }
                    result[flat] = convert(source[offset])
                    var axis = shape.count - 1
                    while axis >= 0 {
                        index[axis] += 1
                        if index[axis] < shape[axis] { break }
                        index[axis] = 0
                        axis -= 1
                    }
                }
            }
        }
        switch array.dataType {
        case .float16: read(Float16.self) { Float($0) }
        case .float32: read(Float.self) { $0 }
        case .double:  read(Double.self) { Float($0) }
        default:       read(Float.self) { $0 }
        }
        return result
    }
}

/// DDIM with "trailing" timesteps and no added noise, as Hyper-SD's
/// N-step distillations are sampled: for four steps, 999, 749, 499, 249.
struct DDIMSchedule {
    let timesteps: [Int]
    private let alphasCumulative: [Double]
    private let stride: Int

    /// SD 1.5's training schedule: "scaled linear" betas, 0.00085 to 0.012
    /// over 1,000 steps.
    init(steps: Int, trainingSteps: Int = 1000) {
        let start = 0.00085.squareRoot(), end = 0.012.squareRoot()
        var product = 1.0
        alphasCumulative = (0 ..< trainingSteps).map { index in
            let root = start + (end - start) * Double(index) / Double(trainingSteps - 1)
            product *= 1 - root * root
            return product
        }
        stride = trainingSteps / steps
        timesteps = (0 ..< steps).map { trainingSteps - 1 - $0 * trainingSteps / steps }
    }

    /// One step from `timesteps[index]` to the next, from the model's
    /// estimate of the noise in `sample`.
    func step(sample: [Float], noise: [Float], at index: Int) -> [Float] {
        let t = timesteps[index]
        let previous = t - stride
        let alpha = alphasCumulative[t]
        // Not "set alpha to one": the last step lands on the first
        // training step's alpha, as SD 1.5's scheduler config says.
        let alphaPrevious = previous >= 0 ? alphasCumulative[previous] : alphasCumulative[0]
        let (a, b) = (Float(alpha.squareRoot()), Float((1 - alpha).squareRoot()))
        let (c, d) = (Float(alphaPrevious.squareRoot()), Float((1 - alphaPrevious).squareRoot()))
        var result = [Float](repeating: 0, count: sample.count)
        for i in 0 ..< sample.count {
            let original = (sample[i] - b * noise[i]) / a
            result[i] = c * original + d * noise[i]
        }
        return result
    }
}

/// Standard normal samples from a seed, the same on every device: a run is
/// reproducible from the seed it records.
struct SeededNormal {
    private var state: UInt64

    init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }

    /// SplitMix64.
    private mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in (0, 1], never zero, so the logarithm below is finite.
    private mutating func uniform() -> Double {
        (Double(next() >> 11) + 1) / Double(1 << 53)
    }

    /// Box–Muller, two samples per pair of uniforms.
    mutating func sample(count: Int) -> [Float] {
        var values = [Float]()
        values.reserveCapacity(count + 1)
        while values.count < count {
            let radius = (-2 * log(uniform())).squareRoot()
            let angle = 2 * Double.pi * uniform()
            values.append(Float(radius * cos(angle)))
            values.append(Float(radius * sin(angle)))
        }
        return Array(values.prefix(count))
    }
}

/// Turns a model's image output into a picture.
enum RGBImage {
    enum Failure: Error { case unrenderable }

    /// - Parameter planar: (3, height, width), red then green then blue,
    ///   in -1...1.
    static func make(planar: [Float], width: Int, height: Int) throws -> CGImage {
        let plane = width * height
        var pixels = [UInt8](repeating: 255, count: plane * 4)
        for i in 0 ..< plane {
            for channel in 0 ..< 3 {
                let value = (planar[channel * plane + i] + 1) * 0.5
                pixels[i * 4 + channel] = UInt8(max(0, min(1, value)) * 255 + 0.5)
            }
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: true,
                                  intent: .defaultIntent)
        else { throw Failure.unrenderable }
        return image
    }
}
