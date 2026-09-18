//  Adapted from StableAudio3-IOS (https://github.com/kellyvv/StableAudio3-IOS),
//  MIT licensed, Copyright (c) 2026 stableaudio3-ios contributors.
//  Changed here: weights are resolved from the downloaded-models directory
//  rather than the app bundle, so no model files ship inside the app.
import Foundation
import MLX
import AVFoundation
import SentencepieceTokenizer

enum StableAudioModelKind: String, CaseIterable, Identifiable, Sendable {
    case smallMusic
    case smallSFX
    case medium

    var id: String { rawValue }

    var title: String {
        switch self {
        case .smallMusic: return "Music"
        case .smallSFX: return "SFX"
        case .medium: return "Medium"
        }
    }

    var displayName: String {
        switch self {
        case .smallMusic: return "Small Music"
        case .smallSFX: return "Small SFX"
        case .medium: return "Medium"
        }
    }

    var ditResourceName: String {
        switch self {
        case .smallMusic: return "dit_sm-music_f16"
        case .smallSFX: return "dit_sm-sfx_f16"
        case .medium: return "dit_medium_f16"
        }
    }

    var conditionerResourceName: String {
        switch self {
        case .smallMusic: return "sa3_conditioner_sm-music"
        case .smallSFX: return "sa3_conditioner_sm-sfx"
        case .medium: return "sa3_conditioner_medium"
        }
    }

    var ditConfig: SA3DiTConfig {
        switch self {
        case .smallMusic, .smallSFX: return .smallMusic
        case .medium: return .medium
        }
    }

    /// Medium ships as two shards: a single safetensors would be 2.9 GB, over
    /// the 2 GB cap on release assets.
    var ditShardCount: Int { self == .medium ? 2 : 1 }

    var missingDiTFileName: String {
        "\(ditResourceName).safetensors"
    }
}

actor StableAudioPipeline {
    static let sampleRate = 44_100
    static let samplesPerLatent = 4_096

    private var cachedTokenizer: SentencepieceTokenizer?
    private var cachedT5Encoder: T5GemmaEncoder?
    private var cachedConditioners: [StableAudioModelKind: SA3Conditioning] = [:]
    private var cachedDiTWeights: [StableAudioModelKind: [String: MLXArray]] = [:]
    private var cachedDecoder: SAMESDecoder?

    struct Result {
        let url: URL
        let duration: Float
        let latentLength: Int
        let elapsedSeconds: TimeInterval
    }

    func generate(model: StableAudioModelKind, prompt: String, seconds: Float = 5, steps: Int = 8, seed: UInt64 = 20260522, progress: @escaping @Sendable (String) -> Void) throws -> Result {
        let totalStartedAt = Date()
        let latentLength = Self.latentLength(for: seconds)

        let conditioning = try Stream.withNewDefaultStream(device: .gpu) {
            let t5StartedAt = Date()
            progress("T5")
            Self.logStart("T5", totalStartedAt: totalStartedAt)
            let promptEncoding = try encodePrompt(prompt: prompt, maxLength: 256)
            Self.logEnd("T5", startedAt: t5StartedAt, totalStartedAt: totalStartedAt)

            let conditioningStartedAt = Date()
            progress("Conditioning")
            Self.logStart("Conditioning", totalStartedAt: totalStartedAt)
            let conditioner = try loadConditioner(model: model)
            let conditioned = conditioner.makeConditioning(promptEncoding: promptEncoding, seconds: seconds)
            let crossAttention = conditioned.crossAttention.asType(.float16)
            let globalCondition = conditioned.globalCondition.asType(.float16)
            eval(crossAttention, globalCondition)
            Self.logEnd("Conditioning", startedAt: conditioningStartedAt, totalStartedAt: totalStartedAt)
            return (crossAttention, globalCondition)
        }

        let latents = try Stream.withNewDefaultStream(device: .gpu) {
            let loadStartedAt = Date()
            progress("DiT load")
            Self.logStart("DiT load", totalStartedAt: totalStartedAt)
            let dit = try loadDiT(model: model, latentLength: latentLength)
            Self.logEnd("DiT load", startedAt: loadStartedAt, totalStartedAt: totalStartedAt)

            let samplingStartedAt = Date()
            progress("Sampling")
            Self.logStart("Sampling", totalStartedAt: totalStartedAt)
            let latents = sample(dit: dit, latentLength: latentLength, steps: steps, seed: seed, totalStartedAt: totalStartedAt, crossAttention: conditioning.0, globalCondition: conditioning.1, progress: progress)
            eval(latents)
            Self.logEnd("Sampling", startedAt: samplingStartedAt, totalStartedAt: totalStartedAt)
            return latents
        }

        let audio = try Stream.withNewDefaultStream(device: .gpu) {
            let loadStartedAt = Date()
            progress("Decoder load")
            Self.logStart("Decoder load", totalStartedAt: totalStartedAt)
            let decoder = try loadDecoder()
            Self.logEnd("Decoder load", startedAt: loadStartedAt, totalStartedAt: totalStartedAt)

            let forwardStartedAt = Date()
            progress("Decoder forward")
            Self.logStart("Decoder forward", totalStartedAt: totalStartedAt)
            let patches = decoder.decodeChunked(latents: latents.asType(.float32))
            let audio = Self.patchedDecode(patches).asType(.float32)
            eval(audio)
            Self.logEnd("Decoder forward", startedAt: forwardStartedAt, totalStartedAt: totalStartedAt)
            return audio
        }

        let wavStartedAt = Date()
        progress("Writing WAV")
        Self.logStart("Writing WAV", totalStartedAt: totalStartedAt)
        let requestedSamples = Int((seconds * Float(Self.sampleRate)).rounded())
        let trimmed = audio[0, 0..., 0 ..< requestedSamples]
        let url = try WAVWriter.writeStereoFloat32(trimmed, sampleRate: Self.sampleRate)
        Self.logEnd("Writing WAV", startedAt: wavStartedAt, totalStartedAt: totalStartedAt)
        progress("Done")
        let elapsedSeconds = Date().timeIntervalSince(totalStartedAt)
        print("[SA3] total \(Self.formatMilliseconds(elapsedSeconds))ms model=\(model.displayName) prompt=\"\(prompt)\" seconds=\(seconds) steps=\(steps) latentLength=\(latentLength)")
        return Result(url: url, duration: seconds, latentLength: latentLength, elapsedSeconds: elapsedSeconds)
    }

    private func sample(
        dit: SA3DiT,
        latentLength: Int,
        steps: Int,
        seed: UInt64,
        totalStartedAt: Date,
        crossAttention: MLXArray,
        globalCondition: MLXArray,
        progress: @escaping @Sendable (String) -> Void
    ) -> MLXArray {
        let schedule = Self.buildSchedule(steps: steps)
        var key = MLXRandom.key(seed)
        var x = MLXRandom.normal([1, 256, latentLength], dtype: .float16, key: key)
        eval(x)

        for index in 0 ..< steps {
            let stepStartedAt = Date()
            progress("Sampling \(index + 1)/\(steps)")
            let current = schedule[index]
            let next = schedule[index + 1]
            let t = MLXArray([current], [1]).asType(.float16)
            let velocity = dit(x, timestep: t, crossAttention: crossAttention, globalCondition: globalCondition)
            let denoised = x - MLXArray(current, dtype: x.dtype) * velocity
            if index < steps - 1 && next > 0 {
                let split = MLXRandom.split(key: key)
                key = split.0
                let noise = MLXRandom.normal(x.shape, dtype: x.dtype, key: split.1)
                x = (1.0 - next) * denoised + next * noise
            } else {
                x = denoised
            }
            eval(x)
            print("[SA3] step \(index + 1)/\(steps) \(Self.formatMilliseconds(Date().timeIntervalSince(stepStartedAt)))ms total=\(Self.formatMilliseconds(Date().timeIntervalSince(totalStartedAt)))ms")
        }
        return x
    }

    private func encodePrompt(prompt: String, maxLength: Int) throws -> T5PromptEncoding {
        let batch = try tokenize(prompt: prompt, maxLength: maxLength)
        let encoder = try loadT5Encoder()
        let embeddings = encoder.encodeTokenIDs(batch.tokenIDs, attentionMask: batch.mask)
        eval(embeddings, batch.mask)
        return T5PromptEncoding(embeddings: embeddings, mask: batch.mask, tokenCount: batch.tokenCount)
    }

    private func tokenize(prompt: String, maxLength: Int) throws -> (tokenIDs: MLXArray, mask: MLXArray, tokenCount: Int) {
        let tokenizer = try loadTokenizer()
        let clipped = Array(try tokenizer.encode(prompt).prefix(maxLength)).map(Int32.init)
        let safeTokens = clipped.isEmpty ? [Int32(1)] : clipped
        let tokenCount = safeTokens.count

        var ids = Array(repeating: Int32(0), count: maxLength)
        var mask = Array(repeating: Int32(0), count: maxLength)
        for index in 0 ..< tokenCount {
            ids[index] = safeTokens[index]
            mask[index] = 1
        }

        return (
            MLXArray(ids, [1, maxLength]),
            MLXArray(mask, [1, maxLength]),
            tokenCount
        )
    }

    private func loadTokenizer() throws -> SentencepieceTokenizer {
        if let cachedTokenizer {
            print("[SA3] cache hit tokenizer")
            return cachedTokenizer
        }

        guard let url = SA3Weights.url(
            forResource: "t5gemma_tokenizer",
            withExtension: "model",
            subdirectory: "Weights"
        ) else {
            throw WeightTensorLoaderError.missing("t5gemma_tokenizer.model")
        }

        print("[SA3] cache miss tokenizer, loading model")
        let tokenizer = try SentencepieceTokenizer(modelPath: url.path(percentEncoded: false), tokenOffset: 0)
        cachedTokenizer = tokenizer
        return tokenizer
    }

    private func loadT5Encoder() throws -> T5GemmaEncoder {
        if let cachedT5Encoder {
            print("[SA3] cache hit T5Gemma")
            return cachedT5Encoder
        }

        guard let url = SA3Weights.url(
            forResource: "t5gemma_f16",
            withExtension: "safetensors",
            subdirectory: "Weights"
        ) else {
            throw WeightTensorLoaderError.missing("t5gemma_f16.safetensors")
        }

        print("[SA3] cache miss T5Gemma, loading weights")
        let weights = try loadArrays(url: url, stream: .cpu)
        let encoder = try T5GemmaEncoder(weights: weights)
        cachedT5Encoder = encoder
        return encoder
    }

    private func loadConditioner(model: StableAudioModelKind) throws -> SA3Conditioning {
        if let cachedConditioner = cachedConditioners[model] {
            print("[SA3] cache hit conditioner \(model.displayName)")
            return cachedConditioner
        }

        let fallbackMusicConditioner = model == .smallMusic
            ? SA3Weights.url(
                forResource: "sa3_conditioner",
                withExtension: "safetensors",
                subdirectory: "Weights"
            )
            : nil
        guard let url = SA3Weights.url(
            forResource: model.conditionerResourceName,
            withExtension: "safetensors",
            subdirectory: "Weights"
        ) ?? fallbackMusicConditioner else {
            throw WeightTensorLoaderError.missing("\(model.conditionerResourceName).safetensors")
        }

        print("[SA3] cache miss conditioner \(model.displayName), loading weights")
        let weights = try loadArrays(url: url, stream: .cpu)
        let conditioner = SA3Conditioning(weights: weights)
        cachedConditioners[model] = conditioner
        return conditioner
    }

    private func loadDiT(model: StableAudioModelKind, latentLength: Int) throws -> SA3DiT {
        if let cachedDiTWeights = cachedDiTWeights[model] {
            print("[SA3] cache hit DiT \(model.displayName)")
            return SA3DiT(weights: cachedDiTWeights, latentLength: latentLength, config: model.ditConfig)
        }

        if model.ditShardCount > 1 {
            var merged: [String: MLXArray] = [:]
            for shard in 1 ... model.ditShardCount {
                guard let url = SA3Weights.url(forResource: "\(model.ditResourceName).part\(shard)",
                                               withExtension: "safetensors") else {
                    throw WeightTensorLoaderError.missing("\(model.ditResourceName).part\(shard).safetensors")
                }
                merged.merge(try loadArrays(url: url, stream: .cpu)) { current, _ in current }
            }
            cachedDiTWeights[model] = merged
            return SA3DiT(weights: merged, latentLength: latentLength, config: model.ditConfig)
        }

        guard let url = SA3Weights.url(
            forResource: model.ditResourceName,
            withExtension: "safetensors",
            subdirectory: "Weights"
        ) else {
            throw WeightTensorLoaderError.missing(model.missingDiTFileName)
        }

        print("[SA3] cache miss DiT \(model.displayName), loading weights")
        let weights = try loadArrays(url: url, stream: .cpu)
        cachedDiTWeights[model] = weights
        return SA3DiT(weights: weights, latentLength: latentLength, config: model.ditConfig)
    }

    private func loadDecoder() throws -> SAMESDecoder {
        if let cachedDecoder {
            print("[SA3] cache hit decoder")
            return cachedDecoder
        }

        guard let url = SA3Weights.url(
            forResource: "same_s_decoder_f32",
            withExtension: "safetensors",
            subdirectory: "Weights"
        ) else {
            throw WeightTensorLoaderError.missing("same_s_decoder_f32.safetensors")
        }

        print("[SA3] cache miss decoder, loading weights")
        let weights = try loadArrays(url: url, stream: .cpu)
        let decoder = SAMESDecoder(weights: weights)
        cachedDecoder = decoder
        return decoder
    }

    static func latentLength(for seconds: Float) -> Int {
        var length = max(1, Int(ceil(seconds * Float(sampleRate) / Float(samplesPerLatent))))
        if length % 2 != 0 {
            length += 1
        }
        return length
    }

    static func buildSchedule(steps: Int) -> [Float] {
        var values: [Float] = []
        for index in 0 ... steps {
            let t = 1.0 - Float(index) / Float(steps)
            var shifted = logSNRShift(t)
            if index == 0 { shifted = 1.0 }
            if index == steps { shifted = 0.0 }
            values.append(shifted)
        }
        return values
    }

    private static func logSNRShift(_ t: Float, anchorLogSNR: Float = -6.2, logSNREnd: Float = 2.0) -> Float {
        if t <= 0 { return 0 }
        if t >= 1 { return 1 }
        let logSNR = logSNREnd - t * (logSNREnd - anchorLogSNR)
        return 1.0 / (1.0 + exp(logSNR))
    }

    static func patchedDecode(_ patches: MLXArray) -> MLXArray {
        let batch = patches.dim(0)
        let length = patches.dim(2)
        var x = patches.reshaped(batch, 2, 256, length)
        x = x.transposed(0, 1, 3, 2)
        return x.reshaped(batch, 2, length * 256)
    }

    private static func logStart(_ stage: String, totalStartedAt: Date) {
        print("[SA3] -> \(stage) total=\(formatMilliseconds(Date().timeIntervalSince(totalStartedAt)))ms")
    }

    private static func logEnd(_ stage: String, startedAt: Date, totalStartedAt: Date) {
        print("[SA3] <- \(stage) stage=\(formatMilliseconds(Date().timeIntervalSince(startedAt)))ms total=\(formatMilliseconds(Date().timeIntervalSince(totalStartedAt)))ms")
    }

    private static func formatMilliseconds(_ seconds: TimeInterval) -> Int {
        Int((seconds * 1000).rounded())
    }
}

enum WAVWriter {
    static func writeStereoFloat32(_ audio: MLXArray, sampleRate: Int) throws -> URL {
        let shape = audio.shape
        precondition(shape.count == 2 && shape[0] == 2)
        let samples = shape[1]
        let values = audio.asArray(Float.self)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("stable-audio-\(Int(Date().timeIntervalSince1970)).wav")

        var data = Data()
        appendString("RIFF", to: &data)
        appendUInt32(UInt32(36 + samples * 2 * 2), to: &data)
        appendString("WAVE", to: &data)
        appendString("fmt ", to: &data)
        appendUInt32(16, to: &data)
        appendUInt16(1, to: &data)
        appendUInt16(2, to: &data)
        appendUInt32(UInt32(sampleRate), to: &data)
        appendUInt32(UInt32(sampleRate * 2 * 2), to: &data)
        appendUInt16(4, to: &data)
        appendUInt16(16, to: &data)
        appendString("data", to: &data)
        appendUInt32(UInt32(samples * 2 * 2), to: &data)

        for index in 0 ..< samples {
            appendPCM(values[index], to: &data)
            appendPCM(values[samples + index], to: &data)
        }

        try data.write(to: url, options: [.atomic])
        return url
    }

    private static func appendPCM(_ value: Float, to data: inout Data) {
        let clipped = max(-1.0, min(1.0, value))
        appendInt16(Int16(clipped * 32767.0), to: &data)
    }

    private static func appendString(_ string: String, to data: inout Data) {
        data.append(string.data(using: .ascii)!)
    }

    private static func appendUInt16(_ value: UInt16, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    private static func appendInt16(_ value: Int16, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    private static func appendUInt32(_ value: UInt32, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }
}

func linear(_ x: MLXArray, weight: MLXArray, bias: MLXArray? = nil) -> MLXArray {
    var y = matmul(x, weight.T)
    if let bias {
        y = y + bias
    }
    return y
}

func silu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}
