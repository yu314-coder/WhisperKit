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

/// Prompt to audio with Stable Audio 3, one stage at a time.
///
/// This once kept every model it had loaded — text encoder, transformer,
/// decoder — for the life of the app, which is how Medium came to need
/// 3.5 GB and be killed on an 8 GB iPad. Now each stage maps its weights
/// (`MappedWeights`: file-backed pages, not memory charged to the app),
/// runs, and hands everything back before the next begins. Only the
/// tokenizer, a few megabytes, is kept between runs.
actor StableAudioPipeline {
    static let sampleRate = 44_100
    static let samplesPerLatent = 4_096

    private var cachedTokenizer: SentencepieceTokenizer?

    struct Result {
        let url: URL
        let duration: Float
        let latentLength: Int
        let elapsedSeconds: TimeInterval
    }

    func generate(model: StableAudioModelKind, prompt: String, seconds: Float = 5, steps: Int = 8,
                  seed: UInt64 = 20260522, progress: @escaping @Sendable (String) -> Void) throws -> Result {
        let totalStartedAt = Date()
        let latentLength = Self.latentLength(for: seconds)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("stable-audio-\(Int(Date().timeIntervalSince1970)).wav")

        try StageMemory.withoutCache {
            // 1. Text encoder.
            progress("Reading prompt")
            let promptEncoding = try Stream.withNewDefaultStream(device: .gpu) {
                try encodePrompt(prompt: prompt, maxLength: 256)
            }
            StageMemory.release()
            try Task.checkCancellation()

            // 2. Conditioner: prompt plus the requested length.
            progress("Conditioning")
            let conditioning = try Stream.withNewDefaultStream(device: .gpu) {
                let conditioner = SA3Conditioning(weights: try loadWeights(model.conditionerResourceName))
                let conditioned = conditioner.makeConditioning(promptEncoding: promptEncoding, seconds: seconds)
                let crossAttention = conditioned.crossAttention.asType(.float16)
                let globalCondition = conditioned.globalCondition.asType(.float16)
                eval(crossAttention, globalCondition)
                return (crossAttention, globalCondition)
            }
            StageMemory.release()

            // 3. Transformer, eight steps.
            let latents = try Stream.withNewDefaultStream(device: .gpu) {
                let dit = SA3DiT(weights: try loadDiTWeights(model), latentLength: latentLength,
                                 config: model.ditConfig)
                let latents = try sample(dit: dit, latentLength: latentLength, steps: steps, seed: seed,
                                         totalStartedAt: totalStartedAt, crossAttention: conditioning.0,
                                         globalCondition: conditioning.1, progress: progress)
                eval(latents)
                return latents
            }
            StageMemory.release()
            try Task.checkCancellation()

            // 4. Decoder, a chunk at a time, straight to disk.
            progress("Decoding audio")
            try Stream.withNewDefaultStream(device: .gpu) {
                let decoder = SAMESDecoder(weights: try loadWeights("same_s_decoder_f32"))
                let requested = Int((seconds * Float(Self.sampleRate)).rounded())
                let writer = try StreamingWAVWriter(url: url, frames: requested, sampleRate: Self.sampleRate)
                var written = 0
                try decoder.decodeChunked(latents: latents.asType(.float32)) { patches in
                    guard written < requested else { return }
                    let audio = Self.patchedDecode(patches).asType(.float32)      // (1, 2, n)
                    let take = min(audio.dim(2), requested - written)
                    try writer.append(audio[0..., 0..., 0 ..< take].transposed(0, 2, 1))
                    written += take
                    try Task.checkCancellation()
                }
                try writer.finish()
            }
        }

        progress("Done")
        let elapsedSeconds = Date().timeIntervalSince(totalStartedAt)
        print("[SA3] total \(Self.formatMilliseconds(elapsedSeconds))ms model=\(model.displayName) seconds=\(seconds) steps=\(steps) latentLength=\(latentLength)")
        return Result(url: url, duration: seconds, latentLength: latentLength, elapsedSeconds: elapsedSeconds)
    }

    private func loadWeights(_ name: String) throws -> [String: MLXArray] {
        guard let url = SA3Weights.url(forResource: name, withExtension: "safetensors") else {
            throw WeightTensorLoaderError.missing("\(name).safetensors")
        }
        return try MappedWeights.load(url: url)
    }

    /// Medium ships as two shards: one file would exceed the 2 GB cap on a
    /// release asset.
    private func loadDiTWeights(_ model: StableAudioModelKind) throws -> [String: MLXArray] {
        guard model.ditShardCount > 1 else { return try loadWeights(model.ditResourceName) }
        var merged: [String: MLXArray] = [:]
        for shard in 1 ... model.ditShardCount {
            merged.merge(try loadWeights("\(model.ditResourceName).part\(shard)")) { current, _ in current }
        }
        return merged
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
    ) throws -> MLXArray {
        let schedule = Self.buildSchedule(steps: steps)
        var key = MLXRandom.key(seed)
        var x = MLXRandom.normal([1, 256, latentLength], dtype: .float16, key: key)
        eval(x)

        for index in 0 ..< steps {
            let stepStartedAt = Date()
            try Task.checkCancellation()
            progress("Step \(index + 1) of \(steps)")
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
        try T5GemmaEncoder(weights: try loadWeights("t5gemma_f16"))
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
