import AVFoundation
import Foundation
import MLX
import WhisperKit

/// What the Transcribe tab transcribes with: Whisper (WhisperKit, Core ML),
/// or one of NVIDIA's models, run on the GPU with MLX by the code in
/// SpeechModels/ (from mlx-audio-swift, MIT).
///
/// Measured on FLEURS (100 clips a language, this repository's
/// ports/speech notes): Canary 1B v2 makes the fewest English errors of any
/// model here — 4.3% of words against Whisper Turbo's 6.6% — but knows only
/// English and 24 other European languages. Nemotron 3.5 ASR is a streaming
/// model, so it can write while someone is still speaking; on files it is
/// about Whisper Small's accuracy (Chinese 19% of characters against Turbo's
/// 13%).
enum SpeechEngine: String, CaseIterable, Identifiable {
    case whisper
    case canary = "canary-1b-v2"
    case nemotron = "nemotron-3.5-asr"

    var id: String { rawValue }
    /// In the Transcribe tab's model picker. Nemotron has the Live tab.
    static let nvidia: [SpeechEngine] = [.canary]

    var displayName: String {
        switch self {
        case .whisper:  return "Whisper"
        case .canary:   return "Canary 1B v2"
        case .nemotron: return "Nemotron 3.5 ASR"
        }
    }

    var tagline: String {
        switch self {
        case .whisper:  return ""
        case .canary:   return "English & European languages"
        case .nemotron: return "Live transcription · text as you speak"
        }
    }

    var detail: String {
        switch self {
        case .whisper:
            return ""
        case .canary:
            return "The most accurate English here, and 24 other European languages. Not for Chinese, Japanese or Korean."
        case .nemotron:
            return "Writes while you record, in 30-odd languages. Less accurate than Whisper Turbo on files, so best for live notes."
        }
    }

    /// The badge on its card.
    var badge: String? {
        switch self {
        case .whisper:  return nil
        case .canary:   return "English"
        case .nemotron: return "Live"
        }
    }

    // MARK: - Weights

    private static let release = "speech-v1"

    /// Published on this repository's `speech-v1` release, unchanged from
    /// the MLX conversions on Hugging Face, each under its own prefix there.
    var files: [WeightFile] {
        func file(_ prefix: String, _ name: String, _ bytes: Int64) -> WeightFile {
            WeightFile(path: name, bytes: bytes, release: Self.release,
                       source: URL(string: "\(WeightFile.releaseRoot)/\(Self.release)/\(prefix)_\(name)"))
        }
        switch self {
        case .whisper:
            return []
        case .canary:
            // Mediform/canary-1b-v2-mlx-q8: 8-bit; the tokenizer rides in the config.
            return [file("canary", "config.json", 673_039),
                    file("canary", "model.safetensors", 1_136_436_574),
                    file("canary", "NOTICE.txt", 380)]
        case .nemotron:
            // mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit; the vocabulary rides in the config.
            return [file("nemotron", "config.json", 159_605),
                    file("nemotron", "model.safetensors", 755_598_923),
                    file("nemotron", "LICENSE.txt", 3_164)]
        }
    }

    /// Where the models are kept, one folder each; tests point it elsewhere.
    nonisolated(unsafe) static var root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("SpeechModels", isDirectory: true)

    var directory: URL { Self.root.appendingPathComponent(rawValue, isDirectory: true) }

    var isDownloaded: Bool { !files.isEmpty && files.allSatisfy { $0.isPresent(in: directory) } }

    var sizeLabel: String {
        let bytes = Double(files.reduce(0) { $0 + $1.bytes })
        return bytes >= 1e9 ? String(format: "%.1f GB", bytes / 1e9) : String(format: "%.0f MB", bytes / 1e6)
    }

    /// For the help page.
    var licence: String {
        switch self {
        case .whisper:  return "Whisper (OpenAI): MIT."
        case .canary:   return "Canary 1B v2 (NVIDIA): CC BY 4.0, converted to MLX by Mediform."
        case .nemotron: return "Nemotron 3.5 ASR (NVIDIA): OpenMDW 1.1, converted to MLX by mlx-community."
        }
    }

    // MARK: - Languages

    /// Canary's languages, by Whisper's codes (the language picker's).
    private static let canaryLanguages: Set<String> = [
        "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
        "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
    ]

    /// Nemotron's prompt for each language it was trained to write, by
    /// Whisper's codes; anything else is left to its own detection.
    private static let nemotronLocales: [String: String] = [
        "en": "en-US", "es": "es-US", "fr": "fr-FR", "it": "it-IT", "pt": "pt-BR", "nl": "nl-NL",
        "de": "de-DE", "tr": "tr-TR", "ru": "ru-RU", "ar": "ar-AR", "hi": "hi-IN", "ja": "ja-JP",
        "ko": "ko-KR", "vi": "vi-VN", "uk": "uk-UA", "pl": "pl-PL", "sv": "sv-SE", "cs": "cs-CZ",
        "no": "nb-NO", "da": "da-DK", "bg": "bg-BG", "fi": "fi-FI", "hr": "hr-HR", "sk": "sk-SK",
        "zh": "zh-CN", "hu": "hu-HU", "ro": "ro-RO", "et": "et-EE",
    ]

    /// The Live tab's choices: detection, Nemotron's languages by Whisper's
    /// codes, and Chinese in Traditional characters.
    static var liveLanguages: [String] {
        ["auto"] + (Array(nemotronLocales.keys) + [Languages.traditionalChinese])
            .sorted { Languages.displayName($0) < Languages.displayName($1) }
    }

    /// What to ask the model for, given the language picker's choice (a
    /// Whisper code, "auto", or Traditional Chinese's own code).
    func modelLanguage(for picked: String) -> String {
        let code = picked == Languages.traditionalChinese ? "zh" : picked
        switch self {
        case .whisper:  return code
        case .canary:   return Self.canaryLanguages.contains(code) ? code : "en"
        case .nemotron: return Self.nemotronLocales[code] ?? "auto"
        }
    }

    /// Languages written without spaces between words.
    static func joinsWithoutSpaces(_ code: String) -> Bool {
        ["zh", "ja", "th", "yue"].contains(code.split(separator: "-").first.map(String.init) ?? code)
    }
}

/// Runs the NVIDIA engines: one model resident at a time, on one actor —
/// MLX isn't safe to drive from two threads at once.
actor SpeechRunner {
    static let shared = SpeechRunner()

    private var engine: SpeechEngine?
    private var canary: CanaryModel?
    private var nemotron: NemotronASRModel?

    /// Loads `engine`'s model, letting go of the other.
    func load(_ engine: SpeechEngine) async throws {
        guard self.engine != engine else { return }
        unload()
        switch engine {
        case .whisper:
            return
        case .canary:
            canary = try await CanaryModel.fromModelDirectory(engine.directory)
        case .nemotron:
            nemotron = try NemotronASRModel.fromDirectory(engine.directory)
        }
        self.engine = engine
    }

    func unload() {
        canary = nil
        nemotron = nil
        live = nil
        engine = nil
        Memory.clearCache()
    }

    /// One stretch of speech, in seconds from the start.
    struct Line {
        var start: Double
        var end: Double
        var text: String
    }

    /// Transcribes 16 kHz mono audio in pieces cut at pauses, so the work
    /// can report its progress, stop when asked, and give each piece its
    /// time.
    ///
    /// - Parameters:
    ///   - progress: fraction done, and the text so far.
    func transcribe(_ engine: SpeechEngine, samples: [Float], language: String,
                    progress: @escaping @Sendable (Double, String) -> Void,
                    isCancelled: @escaping @Sendable () -> Bool) async throws -> TranscriptionResult {
        try await load(engine)
        let rate = 16_000
        // Canary reads up to 40 s at a time; pieces of 10 to 20 s also make
        // subtitle-sized lines. Nemotron's alignment times its own sentences,
        // so it takes longer pieces.
        let pieces = engine == .canary
            ? PauseChunker.chunks(samples, sampleRate: rate, target: 10, maximum: 20)
            : PauseChunker.chunks(samples, sampleRate: rate, target: 45, maximum: 60)
        let parameters = STTGenerateParameters(language: language)
        var lines: [Line] = []
        for (index, piece) in pieces.enumerated() {
            if isCancelled() { throw CancellationError() }
            let offset = Double(piece.lowerBound) / Double(rate)
            let audio = MLXArray(Array(samples[piece]))
            switch engine {
            case .canary:
                guard let canary else { break }
                let text = canary.generate(audio: audio, generationParameters: parameters).text
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                lines += Self.sentences(text, from: offset, to: Double(piece.upperBound) / Double(rate))
            case .nemotron:
                guard let nemotron else { break }
                let output = nemotron.generate(audio: audio, generationParameters: parameters)
                lines += Self.lines(output.segments ?? [], offset: offset)
            case .whisper:
                break
            }
            Memory.clearCache()
            progress(Double(index + 1) / Double(pieces.count), Self.text(lines, language: language))
        }
        return Self.result(lines, language: language)
    }

    // MARK: - Live

    /// Nemotron's streaming session, restarted at a pause every half minute
    /// or so: the session re-reads all the audio it was given at each step,
    /// and the model only looks 4.5 s back anyway.
    private struct Live {
        var session: NemotronASRStreamSession
        var language: String
        /// Where the current session starts, in seconds.
        var offset: Double = 0
        var sessionSamples = 0
        var finished: [Line] = []
    }
    private var live: Live?

    func startLive(language: String) async throws {
        try await load(.nemotron)
        guard let nemotron else { return }
        live = Live(session: nemotron.makeStreamSession(language: language, chunkMs: 560), language: language)
    }

    /// Feeds 16 kHz samples; returns the whole transcript so far.
    func feed(_ samples: [Float]) -> String {
        guard var state = live, let nemotron, !samples.isEmpty else { return "" }
        state.session.step(samples)
        state.sessionSamples += samples.count
        let seconds = Double(state.sessionSamples) / 16_000
        if seconds > 60 || (seconds > 25 && PauseChunker.isQuiet(samples)) {
            state.session.finish()
            state.finished += Self.lines(state.session.segments, offset: state.offset)
            state.offset += seconds
            state.sessionSamples = 0
            state.session = nemotron.makeStreamSession(language: state.language, chunkMs: 560)
        }
        live = state
        Memory.clearCache()
        return Self.text(state.finished, language: state.language)
            + (state.finished.isEmpty ? "" : " ") + state.session.text
    }

    /// Flushes the session and returns everything heard.
    func finishLive() -> TranscriptionResult {
        guard let state = live else { return Self.result([], language: "auto") }
        state.session.finish()
        let lines = state.finished + Self.lines(state.session.segments, offset: state.offset)
        live = nil
        return Self.result(lines, language: state.language)
    }

    // MARK: - Pieces

    private static func lines(_ segments: [[String: Any]], offset: Double) -> [Line] {
        segments.compactMap { segment in
            guard let text = (segment["text"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else { return nil }
            let start = (segment["start"] as? Double) ?? Double(segment["start"] as? Float ?? 0)
            let end = (segment["end"] as? Double) ?? Double(segment["end"] as? Float ?? 0)
            return Line(start: offset + start, end: offset + max(start, end), text: text)
        }
    }

    /// A piece's text cut into sentences, its time shared out by length:
    /// Canary gives text without times.
    private static func sentences(_ text: String, from start: Double, to end: Double) -> [Line] {
        guard !text.isEmpty else { return [] }
        var parts: [String] = []
        var current = ""
        for character in text {
            current.append(character)
            if ".!?。！？".contains(character) {
                parts.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
            }
        }
        if !current.trimmingCharacters(in: .whitespaces).isEmpty { parts.append(current.trimmingCharacters(in: .whitespaces)) }
        let total = Double(parts.reduce(0) { $0 + $1.count })
        var cursor = start
        return parts.map { part in
            let length = (end - start) * Double(part.count) / max(total, 1)
            defer { cursor += length }
            return Line(start: cursor, end: cursor + length, text: part)
        }
    }

    private static func text(_ lines: [Line], language: String) -> String {
        lines.map(\.text).joined(separator: SpeechEngine.joinsWithoutSpaces(language) ? "" : " ")
    }

    /// A line longer than a subtitle should be, cut at commas, else spaces,
    /// into parts of about `seconds` each, the time shared out by length.
    /// Nemotron's sentences can run half a minute where speech has no full
    /// stops.
    private static func shortened(_ line: Line, seconds: Double = 10) -> [Line] {
        let duration = line.end - line.start
        guard duration > seconds * 1.4 else { return [line] }
        let parts = Int((duration / seconds).rounded(.up))
        let target = Double(line.text.count) / Double(parts)
        let commas = ",，、;；:："
        var pieces: [String] = [], current = "", lastBreak: String.Index?
        for character in line.text {
            current.append(character)
            if commas.contains(character) || character == " " { lastBreak = current.endIndex }
            if Double(current.count) >= target, pieces.count < parts - 1 {
                let cut = lastBreak ?? current.endIndex
                pieces.append(String(current[..<cut]).trimmingCharacters(in: .whitespaces))
                current = String(current[cut...])
                lastBreak = nil
            }
        }
        pieces.append(current.trimmingCharacters(in: .whitespaces))
        pieces = pieces.filter { !$0.isEmpty }
        let total = Double(pieces.reduce(0) { $0 + $1.count })
        var cursor = line.start
        return pieces.map { piece in
            let length = duration * Double(piece.count) / max(total, 1)
            defer { cursor += length }
            return Line(start: cursor, end: cursor + length, text: piece)
        }
    }

    /// In WhisperKit's shape, so saving, subtitles and playback treat it as
    /// they treat Whisper's.
    private static func result(_ lines: [Line], language: String) -> TranscriptionResult {
        let lines = lines.flatMap { shortened($0) }
        let segments = lines.enumerated().map { index, line in
            TranscriptionSegment(id: index, start: Float(line.start), end: Float(line.end), text: line.text)
        }
        let code = language == "auto" ? "" : String(language.split(separator: "-").first ?? "")
        return TranscriptionResult(text: text(lines, language: language), segments: segments,
                                   language: code, timings: TranscriptionTimings())
    }
}

/// Cuts audio at pauses.
enum PauseChunker {
    private static let frame = 400            // 25 ms at 16 kHz
    private static let quietFrames = 8         // 200 ms

    /// Pieces of about `target` seconds, never over `maximum`: each cut at
    /// the quietest 200 ms between the two.
    static func chunks(_ samples: [Float], sampleRate: Int, target: Double, maximum: Double) -> [Range<Int>] {
        let energy = frameEnergy(samples)
        var pieces: [Range<Int>] = []
        var start = 0
        let longest = Int(maximum * Double(sampleRate))
        let shortest = Int(target * Double(sampleRate))
        while start < samples.count {
            if samples.count - start <= longest {
                pieces.append(start ..< samples.count)
                break
            }
            let first = (start + shortest) / frame, last = (start + longest) / frame - quietFrames
            var best = first, bestEnergy = Float.greatestFiniteMagnitude
            if last > first {
                for f in first ..< last {
                    let window = energy[f ..< min(f + quietFrames, energy.count)].reduce(0, +)
                    if window < bestEnergy { bestEnergy = window; best = f }
                }
            }
            let cut = min(samples.count, (best + quietFrames / 2) * frame)
            pieces.append(start ..< cut)
            start = cut
        }
        return pieces
    }

    /// Whether a stretch of microphone audio is below speech, about −40 dBFS
    /// — a pause to restart at.
    static func isQuiet(_ samples: [Float]) -> Bool {
        !samples.isEmpty && rms(samples[...]) < 0.01
    }

    private static func frameEnergy(_ samples: [Float]) -> [Float] {
        stride(from: 0, to: samples.count, by: frame).map { start in
            rms(samples[start ..< min(start + frame, samples.count)])
        }
    }

    private static func rms(_ slice: ArraySlice<Float>) -> Float {
        guard !slice.isEmpty else { return 0 }
        return (slice.reduce(0) { $0 + $1 * $1 } / Float(slice.count)).squareRoot()
    }
}

/// One big job at a time gets the memory.
///
/// Music and pictures need every gigabyte they can get — they check what's
/// free before starting (GenerationSupport) — so as they start, the
/// transcription models let go: the NVIDIA engines here, and Whisper in the
/// Transcribe tab, which loads it again when next shown.
enum ModelMemory {
    static let releaseTranscription = Notification.Name("ModelMemory.releaseTranscription")

    @MainActor static func makeRoomForGeneration() {
        Task { await SpeechRunner.shared.unload() }
        NotificationCenter.default.post(name: releaseTranscription, object: nil)
    }
}
