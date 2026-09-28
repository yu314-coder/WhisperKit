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
    ///   - release: after this many steps, continue with `released` — the
    ///     same prompt conditioned as plain text-to-music, over silence —
    ///     instead of following the source. Upstream's audio_cover_strength.
    /// One velocity prediction: the noisy latent, the context, the
    /// conditioning and t. `ACEDiT.forward`, or the Neural Engine's.
    typealias Transformer = (MLXArray, MLXArray, MLXArray, Float) throws -> MLXArray

    static func diffuse(dit: Transformer,
                        conditioning: MLXArray,
                        source: MLXArray,
                        frames: Int,
                        seed: UInt64,
                        noise: MLXArray? = nil,
                        release: (afterStep: Int, conditioning: MLXArray, source: MLXArray)? = nil,
                        dcw: DCW = .think,
                        onStep: (Int, Int) throws -> Void = { _, _ in }) rethrows -> MLXArray {
        // Context is the source plus a chunk mask of ones: every frame is to
        // be generated. An earlier version sent zeros here — the value
        // meaning "keep this frame" — so the model was told to preserve the
        // silence it was meant to replace.
        let chunkMask = MLXArray.ones([1, frames, latentChannels], dtype: .float32)
        func context(_ bed: MLXArray) -> MLXArray {
            concatenated([bed[0..., 0 ..< frames, 0...].asType(.float32), chunkMask], axis: -1)
        }
        var guide = (conditioning: conditioning, context: context(source))

        var x = noise ?? MLXRandom.normal([1, frames, latentChannels], key: MLXRandom.key(seed))
        eval(x, guide.context)

        for (index, t) in schedule.enumerated() {
            if let release, index == release.afterStep {
                guide = (release.conditioning, context(release.source))
            }
            let velocity = try dit(x, guide.context, guide.conditioning, t)
            let denoised = x - velocity * t
            if index == schedule.count - 1 {
                // Last step solves for the clean latent rather than stepping.
                x = denoised
            } else {
                x = x - velocity * (t - schedule[index + 1])
                x = waveletCorrection(x, denoised: denoised, t: t, strength: dcw)
            }
            eval(x)
            try onStep(index + 1, schedule.count)
        }
        return x
    }
}

extension ACEPipeline {
    /// Whole seconds of silence at the end of a latent, read without
    /// decoding it. Frames of music sit about 1.2 from the silence latent
    /// (relative distance), frames of silence about 0.1; 0.3 separates them
    /// with room to spare.
    static func trailingSilence(_ latent: MLXArray, silence: MLXArray) -> Int {
        let frames = latent.dim(1)
        let bed = silence[0..., 0 ..< frames, 0...].asType(.float32)
        let distance = sqrt(sum(square(latent - bed), axis: -1)) / sqrt(sum(square(bed), axis: -1))
        let seconds = frames / framesPerSecond
        guard seconds > 0 else { return 0 }
        let perSecond = distance[0, 0 ..< seconds * framesPerSecond]
            .reshaped(seconds, framesPerSecond).mean(axis: -1).asArray(Float.self)
        return perSecond.reversed().prefix { $0 < 0.3 }.count
    }

    /// DCW's two strengths.
    struct DCW: Equatable {
        var low: Float
        var high: Float
        /// What upstream's interface uses with Think on — the planner — and
        /// so what its web demo runs: 0.02 low, 0.06 high.
        static let think = DCW(low: 0.02, high: 0.06)
        /// Upstream's library default, and its interface's with Think off.
        static let plain = DCW(low: 0.05, high: 0.02)
    }

    /// DCW, the correction the official sampler applies after every turbo
    /// step by default (CVPR 2026, arXiv:2604.16044): split the latent and
    /// the predicted clean sample into low and high bands with a one-level
    /// Haar transform along time, and push each band of the latent away from
    /// the prediction — the low by t × low strength, the high by
    /// (1 − t) × high strength.
    ///
    /// For Haar the transform pairs neighbouring frames, so the whole
    /// correction is a per-pair formula: with d = x − denoised, frames 2i and
    /// 2i+1 gain low·(d₀+d₁)/2 ± high·(d₀−d₁)/2. An odd final frame pairs
    /// with zero, as pytorch_wavelets pads it. Checked against
    /// pytorch_wavelets to 1e-6.
    static func waveletCorrection(_ x: MLXArray, denoised: MLXArray, t: Float,
                                  strength: DCW = .think) -> MLXArray {
        let (low, high) = (t * strength.low, (1 - t) * strength.high)
        let (frames, channels) = (x.dim(1), x.dim(2))
        var d = x - denoised
        if frames % 2 == 1 {
            d = concatenated([d, MLXArray.zeros([1, 1, channels], dtype: d.dtype)], axis: 1)
        }
        let pairs = d.reshaped(1, d.dim(1) / 2, 2, channels)
        let (a, b) = (pairs[0..., 0..., 0, 0...], pairs[0..., 0..., 1, 0...])
        let lowBand = (a + b) * (low / 2)
        let highBand = (a - b) * (high / 2)
        let correction = stacked([lowBand + highBand, lowBand - highBand], axis: 2)
            .reshaped(1, -1, channels)[0..., 0 ..< frames, 0...]
        return x + correction
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
        case lengthening
        /// Loading the Neural Engine programs; slow only the first time at
        /// a length, while iOS compiles them.
        case preparingEngine
        case planning
        case writing(Int, Int)
        case readingPrompt
        case conditioning
        case step(Int, Int)
        case decoding(Double)
    }

    enum File {
        static let textEncoder = "ace_qwen_f16.safetensors"
        static let conditioner = "ace_cond_f16.safetensors"
        static let transformer = "ace_decoder_q8.safetensors"
        static let decoder = "ace_vae_f16.safetensors"
        static let silence = "ace_silence_full.safetensors"
        static let vocabulary = "ace_vocab.json"
        static let merges = "ace_merges.txt"
        /// The planner as published, bfloat16, in two parts.
        static let planner = ["ace_lm_bf16.part1.safetensors", "ace_lm_bf16.part2.safetensors"]
        static let hints = "ace_hints_f16.safetensors"
    }

    let directory: URL
    var quantizationBits = 8
    /// Map weight files in place rather than reading them into memory; see
    /// `MappedWeights`. Off only to measure the difference.
    var mapsWeights = true
    /// Read a stage's weights into memory when they fit; see `load`.
    var holdsWeightsInMemory = true
    /// How much freed memory MLX may keep for reuse while generating.
    var cacheLimit = 0
    /// Run the planner first, as the official pipeline does by default.
    /// Off only to compare against the transformer alone.
    var usesPlanner = true
    /// Let the planner rewrite the caption, upstream's default. Off: the
    /// user's words are used verbatim (see `generate`).
    var plannerRewritesCaption = false
    /// Fraction of the eight steps that follow the plan before the rest
    /// continue as plain text-to-music — upstream's audio_cover_strength,
    /// whose default is the whole run.
    ///
    /// Build 21 followed the plan for two steps of eight, which matched the
    /// words more closely (CLAP 0.583 against 0.493 over seven prompts) but
    /// let the transformer rewrite each section on its own: past a minute
    /// the piece came apart into pieces. Scored as the similarity of 3 s
    /// windows 15 s or more apart, an 81-second song held together at 0.965
    /// following the plan throughout and 0.892 following it for two steps;
    /// with no plan at all, 0.690. The words are better served by a better
    /// planner — see `MusicModel.aceStep15XL` — than by letting go of it.
    var planStrength: Double = 1.0
    /// Re-render at full plan strength when a song would end early.
    var retriesEarlyEndings = true
    /// Render past the requested length and cut; see `renderedSeconds`.
    var rendersPastEnd = true
    /// The planner's classifier-free guidance; upstream's lm_cfg_scale.
    var plannerGuidance: Float = 2.0
    /// See `ACEPlanner.copyAllowance`.
    var plannerCopyAllowance: Int? = 12
    /// Plan a prompt's timeline part by part; see `ACEPlanner.plan`.
    var plansBySection = true
    /// What the planner computes in: bfloat16, as upstream runs it. Its
    /// activations overflow float16.
    var plannerDType: DType = .bfloat16
    /// What the transformer computes in. Float16 needs float16 weights.
    var transformerDType: DType = .float32
    /// Which planner, and its shape. Files over 2 GB come in parts.
    var plannerFiles = File.planner
    var plannerConfig = ACEQwen3.Config.planner
    /// Which diffusion transformer, and its shape.
    var transformerFiles = [File.transformer]
    var transformerShape = ACEDiT.Shape.base
    /// The condition encoder, with any later file's tensors replacing the
    /// earlier's: XL retrained only its input embeddings and projection.
    var conditionerFiles = [File.conditioner]
    /// Where the transformer's layers run.
    enum Engine { case gpu, neuralEngine }
    var transformerEngine = Engine.gpu
    /// DCW strengths. With the planner, upstream's Think values; without,
    /// its plain ones. Build 27 and earlier used the plain values with the
    /// planner, unlike the official demo.
    var dcw: ACEPipeline.DCW?
    /// Keep the silence token out of the plan; see `ACEPlanner.bansSilence`.
    var plannerBansSilence = true
    /// Sees the planner's output; for diagnostics.
    var onPlan: ((ACEPlanner.Plan) -> Void)?

    private func load(_ name: String) throws -> [String: MLXArray] {
        try load([name])
    }

    /// A stage's weights, read into memory when they fit and mapped when
    /// they do not.
    ///
    /// Mapped weights arrive a 16 KB page at a time as the GPU first touches
    /// them, and when a model is larger than the device's memory the pages
    /// are dropped and read again at every step. On an 8 GB iPad, XL at
    /// float16 (8.1 GB) took 165 s for 30 seconds of music that way, with
    /// the memory gauge near empty. Read in one pass instead, a stage that
    /// fits is paid for once, sequentially, and stays put until the stage
    /// ends. Mapping remains for what does not fit — it is what lets such a
    /// model run at all.
    private func load(_ names: [String]) throws -> [String: MLXArray] {
        let urls = names.map { directory.appendingPathComponent($0) }
        let bytes = urls.reduce(Int64(0)) {
            $0 + Int64((try? $1.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
        let intoMemory = !mapsWeights || (holdsWeightsInMemory && StageMemory.canHold(bytes))
        var merged: [String: MLXArray] = [:]
        for url in urls {
            let part = intoMemory ? try loadArrays(url: url, stream: .cpu) : try MappedWeights.load(url: url)
            if intoMemory { eval(Array(part.values)) }
            merged.merge(part) { $1 }
        }
        return merged
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
        // Rendered longer than asked, then cut: see `renderedSeconds`.
        let rendered = rendersPastEnd ? Self.renderedSeconds(for: Int(seconds)) : Int(seconds)
        let renderFrames = max(frames, rendered * ACEPipeline.framesPerSecond)
        let previousCacheLimit = MLX.Memory.cacheLimit
        MLX.Memory.cacheLimit = cacheLimit
        defer {
            MLX.Memory.cacheLimit = previousCacheLimit
            release()
        }

        let silenceBed = try load(File.silence)["silence"]!
        let silence = Self.silence(silenceBed, covering: max(renderFrames, ACEPipeline.timbreFrames))

        let tokenizer = try ACETokenizer(
            vocabularyURL: directory.appendingPathComponent(File.vocabulary),
            mergesURL: directory.appendingPathComponent(File.merges))
        let hasLyrics = !lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        // A timeline in the prompt is planned part by part (see
        // `ACEPlanner.plan`) — for instrumentals, where nothing else gives
        // the piece its shape. The rest of the prompt is then the caption:
        // the text encoder keeps 256 tokens, and a long timeline would push
        // the style out.
        let sections = hasLyrics || !usesPlanner || !plansBySection ? [] : known?.sections ?? []
        let caption = sections.isEmpty ? caption
            : known.map { $0.style.isEmpty ? sections.map(\.description).joined(separator: ", ") : $0.style } ?? caption

        // 0. Planner: the song's layout, five tokens a second.
        var plan: ACEPlanner.Plan?
        if usesPlanner {
            progress(.planning)
            plan = try {
                var model = ACEQwen3(weights: try load(plannerFiles), config: plannerConfig, prefix: "")
                model.quantizationBits = quantizationBits
                model.dtype = plannerDType
                var planner = ACEPlanner(model: model, tokenizer: tokenizer)
                planner.guidance = plannerGuidance
                planner.copyAllowance = plannerCopyAllowance
                planner.bansSilence = plannerBansSilence
                // The caption is the user's, written in verbatim. Upstream
                // lets the planner rewrite it by default, and at its sampling
                // temperature a "restrained underscore with soft granular
                // pads" came back as energetic synthwave with a four-on-the-
                // floor kick — which the music then followed. Upstream offers
                // this switch (use_cot_caption); the prompt should decide.
                // The vocal language is likewise the user's; an
                // instrumental's is left to the planner, as upstream does.
                let given = ACEPlanner.Metadata(
                    bpm: known?.bpm, caption: plannerRewritesCaption ? nil : caption, duration: rendered,
                    keyscale: known?.keyscale,
                    language: hasLyrics ? language : nil,
                    timeSignature: known?.timeSignature)
                return try planner.plan(caption: caption, lyrics: ACEPipeline.lyricBody(lyrics),
                                        known: given, seconds: rendered, seed: seed, sections: sections,
                                        isCancelled: isCancelled) { stage in
                    if case .writing(let done, let total) = stage { progress(.writing(done, total)) }
                }
            }()
            release()
            try checkCancellation()
            if let plan { onPlan?(plan) }
        }
        let metadata = plan?.metadata

        let releaseStep = Int(Double(ACEPipeline.schedule.count) * planStrength)
        let releasesPlan = plan != nil && releaseStep < ACEPipeline.schedule.count
        var plainText: MLXArray?

        // 1. Text encoder: caption states and lyric table rows, then gone.
        // With a plan, the transformer reads the planner's caption and
        // metadata, as the official pipeline passes them on.
        progress(.readingPrompt)
        let (text, lyric): (MLXArray, MLXArray) = try {
            // The reference truncates at these lengths, so longer input
            // would be conditioning the model never received.
            let captionIDs = Array(tokenizer.encode(
                ACEPipeline.captionPrompt(
                    metadata?.caption ?? caption, seconds: rendered,
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
            if releasesPlan {
                let plainIDs = Array(tokenizer.encode(ACEPipeline.captionPrompt(
                    metadata?.caption ?? caption, seconds: rendered,
                    bpm: metadata?.bpm, keyscale: metadata?.keyscale,
                    timeSignature: metadata?.timeSignature)).prefix(256))
                plainText = encoder(inputIDs: MLXArray(plainIDs, [1, plainIDs.count]))
                eval(plainText!)
            }
            return (text, lyric)
        }()
        release()
        try checkCancellation()

        var plainConditioning: MLXArray?

        // 2. Condition encoder: lyric and timbre stacks plus the projection.
        progress(.conditioning)
        let conditioning: MLXArray = try {
            var packer = ACEConditioning(weights: try load(conditionerFiles))
            packer.quantizationBits = quantizationBits
            let timbre = silence[0..., 0 ..< ACEPipeline.timbreFrames, 0...].asType(.float32)
            let packed = packer.encode(text: text, lyric: lyric, timbre: timbre)
            eval(packed)
            if let plainText {
                plainConditioning = packer.encode(text: plainText, lyric: lyric, timbre: timbre)
                eval(plainConditioning!)
            }
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
            source = guide.dim(1) >= renderFrames
                ? guide
                : concatenated([guide, silence[0..., 0 ..< (renderFrames - guide.dim(1)), 0...]], axis: 1)
            eval(source)
            release()
        }

        // 3. Diffusion transformer, eight steps.
        let latent: MLXArray = try {
            let transformer: ACEPipeline.Transformer
            switch transformerEngine {
            case .gpu:
                var dit = ACEDiT(weights: try load(transformerFiles))
                dit.shape = transformerShape
                dit.quantizationBits = quantizationBits
                dit.dtype = transformerDType
                transformer = { dit.forward(xt: $0, context: $1, encoder: $2, timestep: $3) }
            case .neuralEngine:
                progress(.preparingEngine)
                // The layers' weights live in the Core ML programs; MLX
                // holds only what surrounds them, in float32.
                let outer = ACEDiT(weights: try load(ACENeuralTransformer.outerFile))
                let engine = try ACENeuralTransformer(
                    directory: directory, outer: outer, frames: renderFrames,
                    tokens: max(conditioning.dim(1), plainConditioning?.dim(1) ?? 0))
                try checkCancellation()
                transformer = { try engine.forward(xt: $0, context: $1, encoder: $2, timestep: $3) }
            }
            func render(release: (Int, MLXArray, MLXArray)?) throws -> MLXArray {
                try ACEPipeline.diffuse(dit: transformer, conditioning: conditioning, source: source,
                                        frames: renderFrames, seed: seed, noise: noise, release: release,
                                        dcw: dcw ?? (plan == nil ? .plain : .think)) { step, total in
                    progress(.step(step, total))
                    try checkCancellation()
                }
            }
            // Only the part that is kept matters.
            func kept(_ latent: MLXArray) -> MLXArray { latent[0..., 0 ..< frames, 0...] }
            let first = kept(try render(release: plainConditioning.map { (releaseStep, $0, silence) }))
            // If the piece still ends inside what is kept, the same plan and
            // seed are rendered again following the plan throughout — which
            // fills the length, at some cost to how closely it follows the
            // words.
            guard retriesEarlyEndings, plainConditioning != nil,
                  ACEPipeline.trailingSilence(first, silence: silence) >= 2 else { return first }
            progress(.lengthening)
            return kept(try render(release: nil))
        }()
        release()

        // 4. Decoder, a window at a time, straight to disk.
        progress(.decoding(0))
        var vae = ACEVAE(weights: try load(File.decoder))
        // Half precision halves the decoder's working set; against float32
        // the audio agrees to cosine 0.999998.
        vae.dtype = .float16
        let total = frames * ACEVAE.hopLength
        let writer = try StreamingWAVWriter(url: destination, frames: total, sampleRate: 48000)
        // The piece runs past this point, so it has not ended here; a short
        // fade makes the cut a close.
        let fade = min(total, Int(Self.fadeSeconds * 48000))
        var written = 0
        try vae.decodeTiled(latent: latent) { audio, fraction in
            try checkCancellation()
            let count = audio.dim(1)
            var chunk = audio
            if rendered > Int(seconds), written + count > total - fade {
                let positions = MLXArray((written ..< written + count).map { Float($0) })
                let gain = clip((Float(total) - positions) / Float(fade), min: 0, max: 1)
                chunk = audio * gain.reshaped(1, count, 1)
            }
            try writer.append(chunk)
            written += count
            progress(.decoding(fraction))
        }
        try writer.finish()
        return latent
    }

    /// How much is generated for `seconds` of kept audio.
    ///
    /// The transformer shapes a whole piece to the window it renders —
    /// intro, body, ending, and often a few seconds of silence — whatever
    /// the metadata says the length is. At 30 seconds, 13 of 21 test runs
    /// went quiet 3 to 5 seconds early. So the window is made longer than
    /// what is kept, the piece's ending lands past the cut, and a short fade
    /// closes the part that is kept.
    static func renderedSeconds(for seconds: Int) -> Int {
        min(600, seconds + max(8, seconds / 5))
    }

    static let fadeSeconds = 1.5

    /// The silence bed at least `frames` long. The shipped one is ten
    /// minutes; if a clip ever outgrows it, the reference repeats it rather
    /// than refusing, and so does this.
    static func silence(_ bed: MLXArray, covering frames: Int) -> MLXArray {
        guard bed.dim(1) < frames else { return bed }
        let repeats = (frames + bed.dim(1) - 1) / bed.dim(1)
        return concatenated(Array(repeating: bed, count: repeats), axis: 1)
    }
}
