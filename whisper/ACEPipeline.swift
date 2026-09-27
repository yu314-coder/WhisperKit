//  Ported from ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5), MIT.
//  Verified against the PyTorch reference stage by stage; see ports/README.md.

import Foundation
import MLX
import MLXRandom

/// Text prompt (and lyrics) to audio, using ACE-Step's turbo schedule.
///
/// The official pipeline, minus reference audio: the planner writes the song
/// out as music tokens, and the diffusion transformer renders them. The FSQ
/// audio *encoder* side is still absent — it exists to turn supplied audio
/// into tokens, and here the planner supplies them.
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

    /// Instruction for a plain text-to-music run.
    static let textInstruction = "Fill the audio semantic mask based on the given conditions:"
    /// Instruction once planner tokens guide the run — the official pipeline
    /// switches the task to "cover" whenever tokens are supplied.
    static let plannedInstruction = "Generate audio semantic tokens based on the given conditions:"

    /// The caption as the model was trained to read it: an instruction, the
    /// caption, and a metadata block. Sending the bare caption — as this port
    /// once did — puts the text encoder's output somewhere the diffusion
    /// transformer never saw during training.
    static func captionPrompt(_ caption: String, seconds: Int,
                              instruction: String = textInstruction,
                              bpm: Int? = nil, keyscale: String? = nil, timeSignature: Int? = nil) -> String {
        let metas = "- bpm: \(bpm.map(String.init) ?? "N/A")\n"
            + "- timesignature: \(timeSignature.map(String.init) ?? "N/A")\n"
            + "- keyscale: \(keyscale ?? "N/A")\n"
            + "- duration: \(seconds) seconds\n"
        return "# Instruction\n\(instruction)\n\n"
            + "# Caption\n\(caption)\n\n# Metas\n\(metas)<|endoftext|>\n"
    }

    /// The lyric slot's text: the lyrics, or "[Instrumental]" for none.
    static func lyricBody(_ lyrics: String) -> String {
        let trimmed = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "[Instrumental]" : trimmed
    }

    /// Lyrics with their language. An instrumental is not an empty lyric
    /// slot but the literal "[Instrumental]", and without a planner to say
    /// otherwise its language is "unknown".
    static func lyricPrompt(_ lyrics: String, language: String) -> String {
        "# Languages\n\(language)\n\n# Lyric\n\(lyricBody(lyrics))<|endoftext|>"
    }

    // MARK: - Diffusion

    /// - Parameters:
    ///   - conditioning: the packed sequence from `ACEConditioning`.
    ///   - source: (1, ≥ frames, 64) — the planner's guide when there is
    ///     one, otherwise silence. The transformer reads it alongside the
    ///     noisy latent as what the result should follow.
    ///   - noise: the starting latent; drawn from `seed` when nil.
    /// - Returns: (1, frames, 64), the clean latent.
    static func diffuse(dit: ACEDiT,
                        conditioning: MLXArray,
                        source: MLXArray,
                        frames: Int,
                        seed: UInt64,
                        noise: MLXArray? = nil,
                        onStep: (Int, Int) throws -> Void = { _, _ in }) rethrows -> MLXArray {
        // Context is the source plus a chunk mask of ones: every frame is to
        // be generated. An earlier version sent zeros here — the value
        // meaning "keep this frame" — so the model was told to preserve the
        // silence it was meant to replace.
        let bed = source[0..., 0 ..< frames, 0...].asType(.float32)
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
        case planning
        case writing(Int, Int)
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
        static let planner = "ace_lm_q8.safetensors"
        static let hints = "ace_hints_q8.safetensors"
    }

    let directory: URL
    var quantizationBits = 8
    /// Map weight files in place rather than reading them into memory; see
    /// `MappedWeights`. Off only to measure the difference.
    var mapsWeights = true
    /// How much freed memory MLX may keep for reuse while generating.
    var cacheLimit = 0
    /// Run the planner first, as the official pipeline does by default.
    /// Off only to compare against the transformer alone.
    var usesPlanner = true

    private func load(_ name: String) throws -> [String: MLXArray] {
        let url = directory.appendingPathComponent(name)
        return mapsWeights ? try MappedWeights.load(url: url) : try loadArrays(url: url, stream: .cpu)
    }

    private func release() { StageMemory.release() }

    /// - Parameters:
    ///   - noise: a fixed starting latent, for verification only.
    ///   - isCancelled: polled between stages, steps and decoding windows;
    ///     returning true ends the run with `CancellationError`.
    ///   - known: tempo, key and meter the prompt states; the planner
    ///     writes these in instead of choosing its own.
    func generate(caption: String,
                  lyrics: String,
                  language: String,
                  seconds: Double,
                  seed: UInt64,
                  to destination: URL,
                  known: PromptMetadata? = nil,
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

        let tokenizer = try ACETokenizer(
            vocabularyURL: directory.appendingPathComponent(File.vocabulary),
            mergesURL: directory.appendingPathComponent(File.merges))
        let hasLyrics = !lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty

        // 0. Planner: the song's layout, five tokens a second.
        var plan: ACEPlanner.Plan?
        if usesPlanner {
            progress(.planning)
            plan = try {
                var model = ACEQwen3(weights: try load(File.planner), config: .planner, prefix: "")
                model.quantizationBits = quantizationBits
                let planner = ACEPlanner(model: model, tokenizer: tokenizer)
                // The caption is the user's, written in verbatim. Upstream
                // lets the planner rewrite it by default, and at its sampling
                // temperature a "restrained underscore with soft granular
                // pads" came back as energetic synthwave with a four-on-the-
                // floor kick — which the music then followed. Upstream offers
                // this switch (use_cot_caption); the prompt should decide.
                // The vocal language is likewise the user's; an
                // instrumental's is left to the planner, as upstream does.
                let given = ACEPlanner.Metadata(
                    bpm: known?.bpm, caption: caption, duration: Int(seconds),
                    keyscale: known?.keyscale,
                    language: hasLyrics ? language : nil,
                    timeSignature: known?.timeSignature)
                return try planner.plan(caption: caption, lyrics: ACEPipeline.lyricBody(lyrics),
                                        known: given, seed: seed, isCancelled: isCancelled) { stage in
                    if case .writing(let done, let total) = stage { progress(.writing(done, total)) }
                }
            }()
            release()
            try checkCancellation()
        }
        let metadata = plan?.metadata

        // 1. Text encoder: caption states and lyric table rows, then gone.
        // With a plan, the transformer reads the planner's caption and
        // metadata, as the official pipeline passes them on.
        progress(.readingPrompt)
        let (text, lyric): (MLXArray, MLXArray) = try {
            // The reference truncates at these lengths, so longer input
            // would be conditioning the model never received.
            let captionIDs = Array(tokenizer.encode(
                ACEPipeline.captionPrompt(
                    metadata?.caption ?? caption, seconds: Int(seconds),
                    instruction: plan == nil ? ACEPipeline.textInstruction : ACEPipeline.plannedInstruction,
                    bpm: metadata?.bpm, keyscale: metadata?.keyscale,
                    timeSignature: metadata?.timeSignature)).prefix(256))
            let lyricLanguage = metadata?.language ?? (hasLyrics ? language : "unknown")
            let lyricIDs = Array(tokenizer.encode(
                ACEPipeline.lyricPrompt(lyrics, language: lyricLanguage)).prefix(2048))

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

        // 2b. The plan as a 25 Hz guide; silence when there is none.
        var source = silence
        if let plan {
            var hints = ACEHints(weights: try load(File.hints))
            hints.quantizationBits = quantizationBits
            let guide = hints(codes: plan.codes)
            // Tokens cover whole seconds; any remainder is silence, as the
            // reference pads it.
            source = guide.dim(1) >= frames
                ? guide
                : concatenated([guide, silence[0..., 0 ..< (frames - guide.dim(1)), 0...]], axis: 1)
            eval(source)
            release()
        }

        // 3. Diffusion transformer, eight steps.
        let latent: MLXArray = try {
            var dit = ACEDiT(weights: try load(File.transformer))
            dit.quantizationBits = quantizationBits
            return try ACEPipeline.diffuse(dit: dit, conditioning: conditioning, source: source,
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
        let writer = try StreamingWAVWriter(url: destination, frames: frames * ACEVAE.hopLength, sampleRate: 48000)
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
