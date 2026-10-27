import SwiftUI

/// What a translated transcript shows: the translation, what was said, or
/// both — each line with the words as spoken under it. One choice, kept,
/// for the Transcribe result, the library and Live alike; copying and
/// exporting follow it.
enum TranslationDisplay: String, CaseIterable, Identifiable {
    case both, translation, original

    static let storageKey = "translationDisplay"
    var id: String { rawValue }

    static var current: TranslationDisplay {
        TranslationDisplay(rawValue: UserDefaults.standard.string(forKey: storageKey) ?? "") ?? .both
    }

    /// The lines of one segment as this asks for them; a segment that was
    /// never translated has only its text.
    func lines(text: String, original: String?) -> [String] {
        guard let original, !original.isEmpty, original != text else { return [text] }
        switch self {
        case .both:        return [text, original]
        case .translation: return [text]
        case .original:    return [original]
        }
    }
}

extension SavedTranscript {
    /// Saved from speech in another language and translated.
    var isTranslation: Bool { originalLanguageCode != nil }

    /// The transcript as `display` asks, one segment per line (both: the
    /// translation, then what was said).
    func text(for display: TranslationDisplay) -> String {
        let sorted = segments.sorted { $0.startTime < $1.startTime }
        guard isTranslation, !sorted.isEmpty else {
            return display == .original ? originalFullText ?? fullText : fullText
        }
        return sorted.map { display.lines(text: $0.text, original: $0.originalText).joined(separator: "\n") }
            .joined(separator: "\n")
    }
}

/// One segment's words: the translation, what was said, or the two, the
/// spoken words smaller and quieter under the translation.
struct SegmentWords: View {
    let text: String
    let original: String?
    let display: TranslationDisplay
    var size: CGFloat = 17
    var color: Color = Studio.ink

    var body: some View {
        let lines = display.lines(text: text, original: original)
        VStack(alignment: .leading, spacing: 3) {
            Text(lines[0])
                .font(.system(size: size))
                .foregroundColor(color)
            if lines.count > 1 {
                Text(lines[1])
                    .font(.system(size: size * 0.86))
                    .foregroundColor(Studio.mute)
            }
        }
    }
}

/// The switch on a translated transcript: Both, the language it was
/// translated into, or the one it was spoken in.
struct TranslationDisplayPicker: View {
    /// Picker or Whisper codes; nil or "auto" where the spoken language
    /// varied or isn't known.
    let translatedInto: String?
    let spokenIn: String?
    @AppStorage(TranslationDisplay.storageKey) private var raw = TranslationDisplay.both.rawValue

    var body: some View {
        Picker("Show", selection: $raw) {
            Text("Both").tag(TranslationDisplay.both.rawValue)
            Text(name(translatedInto, otherwise: "Translation")).tag(TranslationDisplay.translation.rawValue)
            Text(name(spokenIn, otherwise: "As spoken")).tag(TranslationDisplay.original.rawValue)
        }
        .pickerStyle(.segmented)
        .accessibilityLabel("Show translation, original or both")
    }

    private func name(_ code: String?, otherwise: String) -> String {
        guard let code, code != "auto", !code.isEmpty else { return otherwise }
        return Languages.englishName(code)
    }
}
