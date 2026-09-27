//  Ported from ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5), MIT.
//  Built by ports/acestep/convert_neural_engine.py; see ports/README.md.

import CoreML
import Foundation
import MLX

/// ACE-Step's diffusion transformer on the Neural Engine.
///
/// MLX has no Neural Engine backend, so the 24 layers run as Core ML
/// programs, in float16, in four chunks of six. What surrounds them — the
/// timestep embedding, the patch projections and the output norm — is a
/// small fraction of the work and stays in MLX (`ACEDiT.prepare` and
/// `finish`), reading `ace_dit_outer_f16`.
///
/// The Neural Engine only runs fixed shapes, so each chunk holds one
/// function per length and conditioning size, all sharing one copy of the
/// weights. A clip is padded up to the next size and the padding is masked
/// out of every attention; the positions that are kept see exactly what they
/// would unpadded. The first run at a size compiles that function for the
/// Neural Engine, which iOS then keeps.
final class ACENeuralTransformer {
    static let chunkCount = 4
    static let block = 128
    /// Patched positions, 12.5 a second; must match the converter.
    static let lengthBuckets = [384, 512, 640, 768, 896, 1024, 1280, 1536, 1792,
                                2048, 2560, 3072, 3840, 4608, 5760]
    /// Conditioning tokens: caption, lyrics and the timbre token.
    static let conditioningBuckets = [512, 2368]
    static let outerFile = "ace_dit_outer_f16.safetensors"

    static func chunkDirectory(_ index: Int) -> String { "ace_dit_ane_c\(index).mlmodelc" }

    enum Failure: LocalizedError {
        case tooLong(Int)
        case noOutput

        var errorDescription: String? {
            switch self {
            case .tooLong(let positions):
                return "\(positions * 2 / 25) seconds is longer than the Neural Engine models were built for."
            case .noOutput:
                return "The Neural Engine returned no result."
            }
        }
    }

    private let outer: ACEDiT
    private let chunks: [MLModel]
    /// The bucket this run uses.
    let length: Int
    let tokens: Int
    private let cos: MLMultiArray
    private let sin: MLMultiArray
    private var masks: [Int: (local: MLMultiArray, full: MLMultiArray)] = [:]
    private var encoderMasks: [Int: MLMultiArray] = [:]

    static func bucket(_ value: Int, in buckets: [Int]) -> Int? { buckets.first { $0 >= value } }

    /// - Parameters:
    ///   - frames: latent frames to be rendered.
    ///   - tokens: the longest conditioning sequence the run will pass.
    init(directory: URL, outer: ACEDiT, frames: Int, tokens: Int) throws {
        let positions = (frames + ACEDiT.patchSize - 1) / ACEDiT.patchSize
        guard let length = Self.bucket(positions, in: Self.lengthBuckets) else {
            throw Failure.tooLong(positions)
        }
        let tokenBucket = Self.bucket(tokens, in: Self.conditioningBuckets) ?? Self.conditioningBuckets.last!
        self.length = length
        self.tokens = tokenBucket
        self.outer = outer

        let configuration = MLModelConfiguration()
        // Not .all: if the Neural Engine turns an operation down, the CPU is
        // slow but safe, where the iPhone GPU has ended the app outright for
        // large Core ML models (see ComputeMode).
        configuration.computeUnits = .cpuAndNeuralEngine
        configuration.functionName = "p\(length)_e\(tokenBucket)"
        chunks = try (0 ..< Self.chunkCount).map { index in
            try MLModel(contentsOf: directory.appendingPathComponent(Self.chunkDirectory(index)),
                        configuration: configuration)
        }

        let (cosTable, sinTable) = ACEDiT.rotaryTable(length: length)
        cos = try Self.multiArray(cosTable.reshaped(1, 1, length, ACEDiT.headDimension))
        sin = try Self.multiArray(sinTable.reshaped(1, 1, length, ACEDiT.headDimension))
    }

    /// The same step `ACEDiT.forward` computes, with the layers on the
    /// Neural Engine.
    func forward(xt: MLXArray, context: MLXArray, encoder: MLXArray, timestep: Float) throws -> MLXArray {
        let prepared = outer.prepare(xt: xt, context: context, encoder: encoder, timestep: timestep)
        let positions = prepared.hidden.dim(1)
        let count = min(prepared.conditioning.dim(1), tokens)

        let hidden = padded(prepared.hidden, to: length)
        let conditioning = padded(prepared.conditioning[0..., 0 ..< count, 0...], to: tokens)
        let (local, full) = try mask(valid: positions)
        var inputs: [String: MLFeatureValue] = [
            "encoder": MLFeatureValue(multiArray: try Self.multiArray(conditioning)),
            "encoder_mask": MLFeatureValue(multiArray: try encoderMask(valid: count)),
            "temb": MLFeatureValue(multiArray: try Self.multiArray(prepared.modulation)),
            "cos": MLFeatureValue(multiArray: cos),
            "sin": MLFeatureValue(multiArray: sin),
            "local_mask": MLFeatureValue(multiArray: local),
            "full_mask": MLFeatureValue(multiArray: full),
        ]
        var state = try Self.multiArray(hidden)
        for chunk in chunks {
            inputs["hidden"] = MLFeatureValue(multiArray: state)
            let output = try chunk.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
            guard let next = output.featureValue(for: "out")?.multiArrayValue else { throw Failure.noOutput }
            state = next
        }
        let result = Self.array(state)[0..., 0 ..< positions, 0...]
        return outer.finish(result, prepared)
    }

    private func padded(_ x: MLXArray, to length: Int) -> MLXArray {
        let extra = length - x.dim(1)
        guard extra > 0 else { return x }
        return concatenated([x, MLXArray.zeros([1, extra, x.dim(2)], dtype: x.dtype)], axis: 1)
    }

    /// The sliding window in block layout — query q of block b sees key
    /// slot s, position (b - 1) * 128 + s, within 128 of it — and the
    /// padding, for the layers that attend to everything.
    private func mask(valid: Int) throws -> (MLMultiArray, MLMultiArray) {
        if let cached = masks[valid] { return cached }
        let block = Self.block, blocks = length / block
        var local = [Float](repeating: -1e4, count: blocks * block * 3 * block)
        for b in 0 ..< blocks {
            for q in 0 ..< block {
                let query = b * block + q
                let row = (b * block + q) * 3 * block
                for s in 0 ..< 3 * block {
                    let key = (b - 1) * block + s
                    if key >= 0, key < valid, abs(query - key) <= ACESlidingWindow.radius {
                        local[row + s] = 0
                    }
                }
            }
        }
        let full = (0 ..< length).map { $0 < valid ? Float(0) : -1e4 }
        let result = (try Self.multiArray(MLXArray(local, [1, blocks, block, 3 * block])),
                      try Self.multiArray(MLXArray(full, [1, 1, 1, length])))
        masks[valid] = result
        return result
    }

    private func encoderMask(valid: Int) throws -> MLMultiArray {
        if let cached = encoderMasks[valid] { return cached }
        let values = (0 ..< tokens).map { $0 < valid ? Float(0) : -1e4 }
        let result = try Self.multiArray(MLXArray(values, [1, 1, 1, tokens]))
        encoderMasks[valid] = result
        return result
    }

    // MARK: - Conversion

    /// A float16 copy in a buffer the multiarray owns, laid out row-major.
    static func multiArray(_ x: MLXArray) throws -> MLMultiArray {
        let data = x.asType(.float16).asData(access: .copy).data
        let shape = x.shape
        var strides = [Int](repeating: 1, count: shape.count)
        for axis in stride(from: shape.count - 2, through: 0, by: -1) {
            strides[axis] = strides[axis + 1] * shape[axis + 1]
        }
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: max(data.count, 2), alignment: 64)
        data.withUnsafeBytes { buffer.copyMemory(from: $0.baseAddress!, byteCount: data.count) }
        return try MLMultiArray(dataPointer: buffer, shape: shape.map { NSNumber(value: $0) },
                                dataType: .float16, strides: strides.map { NSNumber(value: $0) },
                                deallocator: { $0.deallocate() })
    }

    /// Reads a float16 multiarray back, honouring any row padding the
    /// Neural Engine's output layout carries.
    static func array(_ m: MLMultiArray) -> MLXArray {
        let shape = m.shape.map(\.intValue)
        let strides = m.strides.map(\.intValue)
        let width = shape.last!
        let rows = shape.dropLast().reduce(1, *)
        let rowStride = shape.count > 1 ? strides[shape.count - 2] : width
        return m.withUnsafeBufferPointer(ofType: Float16.self) { source in
            if rowStride == width, strides.last == 1 {
                return MLXArray(UnsafeBufferPointer(start: source.baseAddress, count: rows * width), shape)
            }
            var values = [Float16](repeating: 0, count: rows * width)
            for row in 0 ..< rows {
                for column in 0 ..< width {
                    values[row * width + column] = source[row * rowStride + column * strides.last!]
                }
            }
            return MLXArray(values, shape)
        }
    }
}
