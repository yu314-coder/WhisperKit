//  Ported from ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5), MIT.
//  Verified against the PyTorch reference stage by stage; see ports/README.md.

import Foundation
import Darwin
import MLX
import MLXRandom

/// Text prompt (and lyrics) to audio, using ACE-Step's turbo schedule.
///
/// This is the text path: no reference audio and no planner. That removes
/// three of the checkpoint's parts — the 1.7B planner (its hints are optional
/// and this path passes none), and the FSQ tokenizer and detokenizer, which
/// exist to condition on *supplied* audio rather than to produce it.
///
/// Sampling is flow matching: the transformer predicts a velocity and the
/// sampler walks from noise toward data by Euler steps, solving directly for
/// the clean latent on the last one.
enum ACEPipeline {
    /// The turbo schedule for shift 3.0 at 8 steps, as published — these are
    /// exact values the model was distilled for, not a curve to re-derive.
    static let schedule: [Float] = [1.0, 0.9545454545454546, 0.9, 0.8333333333333334,
                                    0.75, 0.6428571428571429, 0.5, 0.3]
    static let framesPerSecond = 25
    static let latentChannels = 64
    /// Frames of silence the timbre encoder reads when no reference clip is
    /// given, as the reference pipeline does.
    static let timbreFrames = 750

    // MARK: - Prompt layout

    /// The caption as the model was trained to read it: an instruction, the
    /// caption, and a metadata block. Sending the bare caption — as this port
    /// once did — puts the text encoder's output somewhere the diffusion
    /// transformer never saw during training.
    static func captionPrompt(_ caption: String, seconds: Int) -> String {
        let metas = "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: \(seconds) seconds\n"
        return "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n"
            + "# Caption\n\(caption)\n\n# Metas\n\(metas)<|endoftext|>\n"
    }

    /// Lyrics with their language. An instrumental is not an empty lyric
    /// slot but the literal "[Instrumental]" with language "unknown".
    static func lyricPrompt(_ lyrics: String, language: String) -> String {
        let trimmed = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        let body = trimmed.isEmpty ? "[Instrumental]" : trimmed
        let code = trimmed.isEmpty ? "unknown" : language
        return "# Languages\n\(code)\n\n# Lyric\n\(body)<|endoftext|>"
    }

    // MARK: - Diffusion

    /// - Parameters:
    ///   - conditioning: the packed sequence from `ACEConditioning`.
    ///   - silence: (1, ≥ frames, 64), the bed the generation starts from.
    ///   - noise: the starting latent; drawn from `seed` when nil.
    /// - Returns: (1, frames, 64), the clean latent.
    static func diffuse(dit: ACEDiT,
                        conditioning: MLXArray,
                        silence: MLXArray,
                        frames: Int,
                        seed: UInt64,
                        noise: MLXArray? = nil,
                        onStep: (Int, Int) throws -> Void = { _, _ in }) rethrows -> MLXArray {
        // Context is the silence bed plus a chunk mask of ones: every frame
        // is to be generated. An earlier version sent zeros here — the value
        // meaning "keep this frame" — so the model was told to preserve the
        // silence it was meant to replace.
        let bed = silence[0..., 0 ..< frames, 0...].asType(.float32)
        let chunkMask = MLXArray.ones([1, frames, latentChannels], dtype: .float32)
        let context = concatenated([bed, chunkMask], axis: -1)

        var x = noise ?? MLXRandom.normal([1, frames, latentChannels], key: MLXRandom.key(seed))
        eval(x, context)

        for (index, t) in schedule.enumerated() {
            let velocity = dit.forward(xt: x, context: context, encoder: conditioning, timestep: t)
            if index == schedule.count - 1 {
                // Last step solves for the clean latent rather than stepping.
                x = x - velocity * t
            } else {
                x = x - velocity * (t - schedule[index + 1])
            }
            eval(x)
            try onStep(index + 1, schedule.count)
        }
        return x
    }
}

/// Runs ACE-Step one stage at a time, holding only the weights that stage
/// needs.
///
/// The checkpoint is four models that never run together: a text encoder
/// used once, a condition encoder used once, the diffusion transformer used
/// eight times, and a decoder used at the end. Loading all of them up front —
/// as the first version did, alongside an audio tokenizer the text path never
/// touches — kept 3 GB resident for a job whose largest stage needs 1.7 GB.
struct ACEGenerator {
    enum Stage {
        case readingPrompt
        case conditioning
        case step(Int, Int)
        case decoding(Double)
    }

    enum File {
        static let textEncoder = "ace_qwen_f16.safetensors"
        static let conditioner = "ace_cond_q8.safetensors"
        static let transformer = "ace_decoder_q8.safetensors"
        static let decoder = "ace_vae_f16.safetensors"
        static let silence = "ace_silence_full.safetensors"
        static let vocabulary = "ace_vocab.json"
        static let merges = "ace_merges.txt"
    }

    let directory: URL
    var quantizationBits = 8
    /// Map weight files in place rather than reading them into memory; see
    /// `MappedWeights`. Off only to measure the difference.
    var mapsWeights = true
    /// How much freed memory MLX may keep for reuse while generating.
    var cacheLimit = 0

    private func load(_ name: String) throws -> [String: MLXArray] {
        let url = directory.appendingPathComponent(name)
        return mapsWeights ? try MappedWeights.load(url: url) : try loadArrays(url: url, stream: .cpu)
    }

    /// Hands a finished stage's memory back before the next one starts.
    ///
    /// Two things delay it. Freed arrays go to MLX's buffer cache rather than
    /// the system, so the cache is cleared. And the system takes freed GPU
    /// memory back asynchronously — measured at a few hundred milliseconds,
    /// during which it still counts against the app. Loading the next stage
    /// inside that window stacks both stages' memory, which is how the
    /// transition, not either stage, became the peak. So this waits, briefly,
    /// for the footprint to stop falling.
    private func release() {
        MLX.Memory.clearCache()
        var previous = MemoryFootprint.current
        for _ in 0 ..< 20 {
            usleep(50_000)
            let now = MemoryFootprint.current
            if previous - now < 8 * 1_048_576 { break }
            previous = now
        }
    }

    /// - Parameters:
    ///   - noise: a fixed starting latent, for verification only.
    ///   - isCancelled: polled between stages, steps and decoding windows;
    ///     returning true ends the run with `CancellationError`.
    func generate(caption: String,
                  lyrics: String,
                  language: String,
                  seconds: Double,
                  seed: UInt64,
                  to destination: URL,
                  noise: MLXArray? = nil,
                  isCancelled: () -> Bool = { false },
                  progress: (Stage) -> Void = { _ in }) throws -> MLXArray {
        func checkCancellation() throws {
            if isCancelled() { throw CancellationError() }
        }
        let frames = max(1, Int(seconds * Double(ACEPipeline.framesPerSecond)))
        let previousCacheLimit = MLX.Memory.cacheLimit
        MLX.Memory.cacheLimit = cacheLimit
        defer {
            MLX.Memory.cacheLimit = previousCacheLimit
            release()
        }

        let silenceBed = try load(File.silence)["silence"]!
        let silence = Self.silence(silenceBed, covering: max(frames, ACEPipeline.timbreFrames))

        // 1. Text encoder: caption states and lyric table rows, then gone.
        progress(.readingPrompt)
        let (text, lyric): (MLXArray, MLXArray) = try {
            let tokenizer = try ACETokenizer(
                vocabularyURL: directory.appendingPathComponent(File.vocabulary),
                mergesURL: directory.appendingPathComponent(File.merges))
            // The reference truncates at these lengths, so longer input
            // would be conditioning the model never received.
            let captionIDs = Array(tokenizer.encode(
                ACEPipeline.captionPrompt(caption, seconds: Int(seconds))).prefix(256))
            let lyricIDs = Array(tokenizer.encode(
                ACEPipeline.lyricPrompt(lyrics, language: language)).prefix(2048))

            var encoder = ACEQwen3(weights: try load(File.textEncoder), config: .embedder, prefix: "")
            encoder.quantizationBits = quantizationBits
            let text = encoder(inputIDs: MLXArray(captionIDs, [1, captionIDs.count]))
            let lyric = encoder.embed(inputIDs: MLXArray(lyricIDs, [1, lyricIDs.count]))
            eval(text, lyric)
            return (text, lyric)
        }()
        release()
        try checkCancellation()

        // 2. Condition encoder: lyric and timbre stacks plus the projection.
        progress(.conditioning)
        let conditioning: MLXArray = try {
            var packer = ACEConditioning(weights: try load(File.conditioner))
            packer.quantizationBits = quantizationBits
            let packed = packer.encode(
                text: text, lyric: lyric,
                timbre: silence[0..., 0 ..< ACEPipeline.timbreFrames, 0...].asType(.float32))
            eval(packed)
            return packed
        }()
        release()
        try checkCancellation()

        // 3. Diffusion transformer, eight steps.
        let latent: MLXArray = try {
            var dit = ACEDiT(weights: try load(File.transformer))
            dit.quantizationBits = quantizationBits
            return try ACEPipeline.diffuse(dit: dit, conditioning: conditioning, silence: silence,
                                           frames: frames, seed: seed, noise: noise) { step, total in
                progress(.step(step, total))
                try checkCancellation()
            }
        }()
        release()

        // 4. Decoder, a window at a time, straight to disk.
        progress(.decoding(0))
        var vae = ACEVAE(weights: try load(File.decoder))
        // Half precision halves the decoder's working set; against float32
        // the audio agrees to cosine 0.999998.
        vae.dtype = .float16
        let writer = try ACEWAVWriter(url: destination, frames: frames * ACEVAE.hopLength)
        try vae.decodeTiled(latent: latent) { audio, fraction in
            try checkCancellation()
            try writer.append(audio)
            progress(.decoding(fraction))
        }
        try writer.finish()
        return latent
    }

    /// The silence bed at least `frames` long. The shipped one is ten
    /// minutes; if a clip ever outgrows it, the reference repeats it rather
    /// than refusing, and so does this.
    static func silence(_ bed: MLXArray, covering frames: Int) -> MLXArray {
        guard bed.dim(1) < frames else { return bed }
        let repeats = (frames + bed.dim(1) - 1) / bed.dim(1)
        return concatenated(Array(repeating: bed, count: repeats), axis: 1)
    }
}

/// 48 kHz stereo 16-bit WAV, written as the decoder produces it.
///
/// Streaming matters at length: six minutes of stereo float is 147 MB, and
/// building it whole before writing — then again as Swift arrays, then again
/// as `Data` — tripled that at the very end of a run that had already used
/// the most memory it would.
final class ACEWAVWriter {
    private let handle: FileHandle
    private let expectedFrames: Int
    private var writtenFrames = 0

    init(url: URL, frames: Int, sampleRate: Int = 48000) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        expectedFrames = frames

        var header = Data()
        func string(_ s: String) { header.append(s.data(using: .ascii)!) }
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        string("RIFF"); u32(UInt32(36 + frames * 4)); string("WAVE")
        string("fmt "); u32(16); u16(1); u16(2)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 4)); u16(4); u16(16)
        string("data"); u32(UInt32(frames * 4))
        try handle.write(contentsOf: header)
    }

    /// - Parameter audio: (1, samples, 2) — already interleaved, which is
    ///   exactly WAV's sample order.
    func append(_ audio: MLXArray) throws {
        let pcm = (clip(audio, min: -1, max: 1) * 32767).asType(.int16)
        let samples = pcm.asArray(Int16.self)
        try samples.withUnsafeBytes { try handle.write(contentsOf: Data($0)) }
        writtenFrames += audio.dim(1)
    }

    func finish() throws {
        precondition(writtenFrames == expectedFrames, "wrote \(writtenFrames) of \(expectedFrames) frames")
        try handle.close()
    }
}

/// The app's physical footprint — the number iOS compares against its limit
/// when deciding what to kill.
enum MemoryFootprint {
    static var current: Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }
}

/// A cancellation request that crosses into a detached task, which does not
/// inherit its parent's cancellation.
final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isCancelled: Bool { lock.withLock { cancelled } }
    func cancel() { lock.withLock { cancelled = true } }
}
