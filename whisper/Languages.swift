import Foundation

/// Language names shared by both tabs.
///
/// Codes are the models' own, not ISO lookups: Whisper says `jw` for
/// Javanese where ISO says `jv`, and both models use `yue` for Cantonese
/// separately from `zh`. A code the model does not know is silently ignored
/// by it, so these lists are copied from the models rather than derived.
enum Languages {
    /// Every language Whisper transcribes — the 99 of the original release
    /// plus Cantonese, added in large-v3 — in the order OpenAI lists them,
    /// which is roughly by amount of training audio.
    static let whisperCodes: [String] = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
        "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
        "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
        "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
        "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
        "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
        "su", "yue",
    ]

    /// The vocal languages ACE-Step 1.5 was trained to sing, from its
    /// `VALID_LANGUAGES`. Anything else is sent as "unknown", which is also
    /// what an instrumental sends.
    static let aceStepCodes: [String] = [
        "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr",
        "he", "hi", "hr", "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne",
        "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te",
        "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
    ]

    /// English names, as the models' own tables give them.
    private static let englishNames: [String: String] = [
        "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian",
        "ko": "Korean", "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish",
        "pl": "Polish", "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
        "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
        "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay", "cs": "Czech",
        "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
        "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian",
        "la": "Latin", "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
        "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali", "sr": "Serbian",
        "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", "et": "Estonian",
        "mk": "Macedonian", "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian",
        "ne": "Nepali", "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian",
        "sw": "Swahili", "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala",
        "km": "Khmer", "sn": "Shona", "yo": "Yoruba", "so": "Somali", "af": "Afrikaans",
        "oc": "Occitan", "ka": "Georgian", "be": "Belarusian", "tg": "Tajik", "sd": "Sindhi",
        "gu": "Gujarati", "am": "Amharic", "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek",
        "fo": "Faroese", "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen",
        "nn": "Norwegian Nynorsk", "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish",
        "my": "Burmese", "bo": "Tibetan", "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese",
        "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa", "ba": "Bashkir",
        "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
    ]

    static func englishName(_ code: String) -> String {
        code == traditionalChinese ? "Chinese (Traditional)" : englishNames[code] ?? code.uppercased()
    }

    // MARK: - Chinese script

    /// A transcription choice, not a Whisper language: Whisper has one
    /// Chinese, `zh`, and writes it in simplified characters. Choosing this
    /// transcribes as `zh` and converts the text.
    static let traditionalChinese = "zh-Hant"

    /// The language Whisper is asked for, for a choice in the picker.
    static func whisperCode(_ choice: String) -> String {
        choice == traditionalChinese ? "zh" : choice
    }

    /// Simplified to traditional characters, with the system's own ICU
    /// transform. Character by character, with context for the ambiguous
    /// ones (发展 → 發展 but 头发 → 頭髮, 后来 → 後來); it does not swap
    /// vocabulary, so 软件 becomes 軟件 rather than Taiwan's 軟體.
    static func traditional(_ text: String) -> String {
        text.applyingTransform(StringTransform("Hans-Hant"), reverse: false) ?? text
    }

    /// The language's name for itself — "Українська" for `uk` — when the
    /// system knows it and it differs from the English one.
    static func nativeName(_ code: String) -> String? {
        if code == traditionalChinese { return "繁體中文" }
        // Whisper's `jw` is ISO `jv`; Locale only knows the latter.
        let iso = code == "jw" ? "jv" : code
        guard let name = Locale(identifier: iso).localizedString(forLanguageCode: iso) else {
            return nil
        }
        let native = name.prefix(1).uppercased() + name.dropFirst()
        return native.caseInsensitiveCompare(englishName(code)) == .orderedSame ? nil : native
    }

    /// "Ukrainian (Українська)", or just the English name.
    static func displayName(_ code: String) -> String {
        guard let native = nativeName(code) else { return englishName(code) }
        return "\(englishName(code)) (\(native))"
    }

    /// Alphabetical by English name, which is how someone scanning for their
    /// language expects to find it.
    static func sortedByName(_ codes: [String]) -> [String] {
        codes.sorted { englishName($0).localizedCaseInsensitiveCompare(englishName($1)) == .orderedAscending }
    }

    /// The device's language if the given list has it, for a sensible first
    /// choice.
    static func preferred(among codes: [String], fallback: String = "en") -> String {
        for identifier in Locale.preferredLanguages {
            let code = Locale(identifier: identifier).language.languageCode?.identifier ?? ""
            if codes.contains(code) { return code }
        }
        return fallback
    }
}
