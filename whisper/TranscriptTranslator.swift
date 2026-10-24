import Foundation
import NaturalLanguage
import SwiftUI
import Translation
import WhisperKit

/// Turns a transcript into the language picked for it when the speech was
/// in another: speak Chinese with German picked, and the text comes out in
/// German. Apple's Translation framework, on the device like everything
/// else; its languages are downloaded by iOS, once, on the first request
/// for a pair (iOS asks first).
///
/// The models can't do this themselves. Nemotron told one language and
/// hearing another writes nothing at all; Whisper writes a rough
/// translation at best (Turbo was never trained to translate); Canary
/// translates, but only between English and other European languages.
/// So the models write what was said, in the language it was said in, and
/// this translates the result.
@MainActor
final class TranscriptTranslator: ObservableObject {
    static let shared = TranscriptTranslator()

    /// For `TranslationHost`: the pair a session is wanted for.
    @Published fileprivate var configuration: TranslationSession.Configuration?
    private var waiting: [Job] = []
    /// While a session is at work, a new request waits its turn: asking
    /// for another session would cancel the one running.
    private var serving = false

    private struct Job {
        let id = UUID()
        let source: Locale.Language
        let target: Locale.Language
        let texts: [String]
        let continuation: CheckedContinuation<[String], Error>
    }

    // MARK: - Languages

    /// The Translation framework's language for a picker choice or a
    /// Whisper code; nil where it has none of its own.
    nonisolated static func language(_ code: String) -> Locale.Language? {
        switch code {
        case "auto", "": return nil
        case Languages.traditionalChinese: return Locale.Language(identifier: "zh-Hant")
        case "zh": return Locale.Language(identifier: "zh-Hans")
        // Written Cantonese reads closest to Traditional Chinese.
        case "yue": return Locale.Language(identifier: "zh-Hant")
        case "no", "nn": return Locale.Language(identifier: "nb")
        case "jw": return Locale.Language(identifier: "jv")
        default: return Locale.Language(identifier: code)
        }
    }

    /// Whether text in `heard` already reads as `wanted`: the same language,
    /// or both Chinese, whose script `Languages.traditional` sees to.
    nonisolated static func reads(_ heard: String, as wanted: String) -> Bool {
        let chinese: Set<String> = ["zh", "yue", Languages.traditionalChinese]
        if chinese.contains(heard) && chinese.contains(wanted) { return true }
        let norwegian: Set<String> = ["no", "nn", "nb"]
        if norwegian.contains(heard) && norwegian.contains(wanted) { return true }
        return heard == wanted
    }

    /// The language a piece of text is written in, as a Whisper code.
    nonisolated static func detect(_ text: String) -> String? {
        guard let found = NLLanguageRecognizer.dominantLanguage(for: text) else { return nil }
        switch found {
        case .simplifiedChinese, .traditionalChinese: return "zh"
        case .norwegian: return "no"
        default: return found.rawValue
        }
    }

    /// Whether iOS can translate between the two, downloaded or not.
    static func canTranslate(from source: String, to target: String) async -> Bool {
        guard let s = language(source), let t = language(target) else { return false }
        return await LanguageAvailability().status(from: s, to: t) != .unsupported
    }

    /// Whether iOS translates into `target` at all.
    static func canTranslate(into target: String) async -> Bool {
        guard let wanted = language(target) else { return false }
        return await LanguageAvailability().supportedLanguages.contains {
            $0.languageCode == wanted.languageCode && (wanted.script == nil || $0.script == wanted.script)
        }
    }

    // MARK: - Translating

    /// `texts` in `target`, in order; empty ones stay empty.
    func translate(_ texts: [String], from source: String, to target: String) async throws -> [String] {
        guard let s = Self.language(source), let t = Self.language(target) else { throw Unavailable() }
        let wanted = texts.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard wanted.contains(where: { !$0.isEmpty }) else { return texts }
        // Where the pair is already on the device a session needs no view,
        // so this keeps going with the app in the background.
        if #available(iOS 26.0, *), await LanguageAvailability().status(from: s, to: t) == .installed {
            return try await Self.run(TranslationSession(installedSource: s, target: t), wanted)
        }
        // Otherwise through `TranslationHost`, where iOS can ask to download.
        return try await withCheckedThrowingContinuation { continuation in
            let job = Job(source: s, target: t, texts: wanted, continuation: continuation)
            waiting.append(job)
            if !serving && waiting.count == 1 { ask(for: job) }
            // A session that never comes mustn't hold the transcript up for
            // good. (Once one has come, iOS's download sheet waits for the
            // user, and so does this.)
            Task {
                try? await Task.sleep(for: .seconds(180))
                self.expire(job.id)
            }
        }
    }

    private func expire(_ id: UUID) {
        guard let index = waiting.firstIndex(where: { $0.id == id }) else { return }
        waiting.remove(at: index).continuation.resume(throwing: NotReady())
        if !serving, let next = waiting.first { ask(for: next) }
    }

    /// Every segment's text in `target`, times kept, so subtitles and
    /// playback line up as before.
    func translate(_ results: [TranscriptionResult], from source: String,
                   to target: String) async throws -> [TranscriptionResult] {
        var translated: [TranscriptionResult] = []
        for result in results {
            let texts = try await translate(result.segments.map { WhisperText.stripSpecialTokens($0.text) },
                                            from: source, to: target)
            translated.append(Self.result(result, texts: texts, language: target))
        }
        return translated
    }

    /// Each text in `target`: kept where it already reads as `target`,
    /// translated from whatever language it is in otherwise, and kept as
    /// it was where that can't be done. For Live, whose speaker may switch
    /// languages from one sentence to the next.
    func intoLanguage(_ texts: [String], target: String) async -> [String] {
        var out = texts
        var bySource: [String: [Int]] = [:]
        for (index, text) in texts.enumerated() {
            guard let source = Self.detect(text), !Self.reads(source, as: target) else { continue }
            bySource[source, default: []].append(index)
        }
        for (source, indices) in bySource {
            guard let translated = try? await translate(indices.map { texts[$0] }, from: source, to: target)
            else { continue }
            for (index, text) in zip(indices, translated) { out[index] = text }
        }
        return out
    }

    /// `result` with its segments' text replaced, times kept.
    static func result(_ result: TranscriptionResult, texts: [String], language target: String) -> TranscriptionResult {
        let segments = zip(result.segments, texts).map { segment, text in
            TranscriptionSegment(id: segment.id, seek: segment.seek, start: segment.start, end: segment.end,
                                 text: text, tokens: [], tokenLogProbs: [], temperature: segment.temperature,
                                 avgLogprob: segment.avgLogprob, compressionRatio: segment.compressionRatio,
                                 noSpeechProb: segment.noSpeechProb)
        }
        let code = Languages.whisperCode(target)
        let joiner = SpeechEngine.joinsWithoutSpaces(code) ? "" : " "
        return TranscriptionResult(text: texts.filter { !$0.isEmpty }.joined(separator: joiner),
                                   segments: segments, language: code, timings: result.timings)
    }

    private func ask(for job: Job) {
        if configuration?.source == job.source && configuration?.target == job.target {
            configuration?.invalidate()
        } else {
            configuration = TranslationSession.Configuration(source: job.source, target: job.target)
        }
    }

    /// Called by `TranslationHost` with a session for `configuration`'s
    /// pair: serves every request waiting for that pair, then asks for the
    /// next pair if any are left.
    fileprivate func serve(_ session: TranslationSession) async {
        guard let pair = configuration else { return }
        let jobs = waiting.filter { $0.source == pair.source && $0.target == pair.target }
        waiting.removeAll { $0.source == pair.source && $0.target == pair.target }
        guard !jobs.isEmpty else { return }
        serving = true
        defer {
            serving = false
            if let next = waiting.first { ask(for: next) }
        }
        do {
            // Shows iOS's download sheet when the languages aren't here yet.
            try await session.prepareTranslation()
            for job in jobs {
                do { job.continuation.resume(returning: try await Self.run(session, job.texts)) }
                catch { job.continuation.resume(throwing: error) }
            }
        } catch {
            for job in jobs { job.continuation.resume(throwing: error) }
        }
    }

    private static func run(_ session: TranslationSession, _ texts: [String]) async throws -> [String] {
        let requests = texts.enumerated().compactMap { index, text in
            text.isEmpty ? nil : TranslationSession.Request(sourceText: text, clientIdentifier: String(index))
        }
        var out = texts
        for response in try await session.translations(from: requests) {
            if let id = response.clientIdentifier, let index = Int(id) { out[index] = response.targetText }
        }
        return out
    }

    struct NotReady: LocalizedError {
        var errorDescription: String? { "iOS didn't get translation ready in time." }
    }

    struct Unavailable: LocalizedError {
        var errorDescription: String? { "This device can't translate between these languages." }
    }
}

/// Gives `TranscriptTranslator` its sessions. Attached once, at the root,
/// so a translation can be asked for from any tab.
struct TranslationHost: ViewModifier {
    @ObservedObject private var translator = TranscriptTranslator.shared

    func body(content: Content) -> some View {
        content.translationTask(translator.configuration) { session in
            await translator.serve(session)
        }
    }
}
