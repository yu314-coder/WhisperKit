import Foundation
import NaturalLanguage

/// How the words were said, from the transcript: how many and how fast, how
/// the pace changed, who spoke for how long, the "um"s, and the words used
/// most. Translated transcripts are measured on the words as spoken.
struct SpeechTextAnalysis: Sendable {
    struct Line: Sendable {
        let start: Double
        let end: Double
        let text: String
        let speaker: String?
    }

    struct Count: Sendable, Hashable {
        let word: String
        let count: Int
    }

    struct Bucket: Sendable, Hashable {
        let start: Double
        let perMinute: Double
    }

    struct Speaker: Sendable, Hashable {
        let name: String
        let seconds: Double
    }

    /// Counted in characters (Chinese, Japanese), which is how their pace is
    /// usually given; in words otherwise.
    let inCharacters: Bool
    let count: Int
    let lines: Int
    /// From the first line's start to the last line's end.
    let span: Double
    let pace: [Bucket]
    let bucketSeconds: Double
    /// Empty unless someone has been named.
    let speakers: [Speaker]
    /// Nil where there is no list for the language.
    let fillers: [Count]?
    let frequent: [Count]
    /// Whether `frequent` is nouns and adjectives, picked out by the
    /// language tagger, or simply the longer words where it can't help.
    let frequentByWordClass: Bool

    var unit: String { inCharacters ? "characters" : "words" }
    var perMinute: Double { span > 0 ? Double(count) / (span / 60) : 0 }

    init(lines: [Line], duration: Double, language: String?) {
        let text = lines.map(\.text).joined(separator: "\n")
        let language = Self.language(of: text, given: language)
        let unspaced = Self.writtenWithoutSpaces(text)
        inCharacters = unspaced
        self.lines = lines.count

        let counts = lines.map { unspaced ? Self.characters(in: $0.text) : Self.words(in: $0.text, language: language).count }
        count = counts.reduce(0, +)
        span = max(0, (lines.map(\.end).max() ?? 0) - (lines.map(\.start).min() ?? 0))

        // Each line's words spread evenly over its time, into half-minute
        // buckets (a minute or two for long recordings).
        let length = max(duration, lines.map(\.end).max() ?? 0)
        let bucket: Double = length <= 600 ? 30 : length <= 3_600 ? 60 : 120
        bucketSeconds = bucket
        var totals = [Double](repeating: 0, count: max(1, Int((length / bucket).rounded(.up))))
        for (line, words) in zip(lines, counts) where words > 0 {
            let lineLength = max(line.end - line.start, 0.001)
            var index = max(0, Int(line.start / bucket))
            while index < totals.count, Double(index) * bucket < line.end {
                let overlap = min(line.end, Double(index + 1) * bucket) - max(line.start, Double(index) * bucket)
                totals[index] += Double(words) * max(0, overlap) / lineLength
                index += 1
            }
        }
        pace = totals.enumerated().map { index, words in
            let seconds = min(bucket, length - Double(index) * bucket)
            return Bucket(start: Double(index) * bucket, perMinute: seconds > 0 ? words / (seconds / 60) : 0)
        }

        var talk: [String: Double] = [:]
        var unnamed = 0.0
        for line in lines {
            let seconds = max(0, line.end - line.start)
            if let name = line.speaker?.trimmingCharacters(in: .whitespaces), !name.isEmpty {
                talk[name, default: 0] += seconds
            } else {
                unnamed += seconds
            }
        }
        if !talk.isEmpty, unnamed > 0 { talk["No name", default: 0] += unnamed }
        speakers = talk.map { Speaker(name: $0.key, seconds: $0.value) }.sorted { $0.seconds > $1.seconds }

        fillers = Self.fillers(in: text, language: language, inCharacters: unspaced)
        (frequent, frequentByWordClass) = Self.frequentWords(in: text, language: language)
    }

    // MARK: - Counting

    private static func language(of text: String, given: String?) -> String? {
        if let given, given != "auto", !given.isEmpty {
            return given == Languages.traditionalChinese || given == "yue" ? "zh" : given
        }
        return NLLanguageRecognizer.dominantLanguage(for: String(text.prefix(4_000)))
            .map { $0 == .simplifiedChinese || $0 == .traditionalChinese ? "zh" : $0.rawValue }
    }

    /// Mostly Chinese characters or Japanese kana.
    private static func writtenWithoutSpaces(_ text: String) -> Bool {
        var letters = 0
        var unspaced = 0
        for scalar in text.unicodeScalars where scalar.properties.isAlphabetic {
            letters += 1
            switch scalar.value {
            case 0x3040 ... 0x30FF, 0x3400 ... 0x4DBF, 0x4E00 ... 0x9FFF, 0xF900 ... 0xFAFF: unspaced += 1
            default: break
            }
        }
        return letters > 0 && unspaced * 2 > letters
    }

    private static func characters(in text: String) -> Int {
        text.unicodeScalars.filter { $0.properties.isAlphabetic || $0.properties.numericType != nil }.count
    }

    private static func words(in text: String, language: String?) -> [String] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        if let language { tokenizer.setLanguage(NLLanguage(language)) }
        return tokenizer.tokens(for: text.startIndex ..< text.endIndex).map { String(text[$0]) }
    }

    // MARK: - Fillers

    /// Hesitations that are only ever hesitations, so a count of them means
    /// something. Speech recognisers often leave them out.
    private static let fillerWords: [String: [String]] = [
        "en": ["um", "umm", "uh", "uhh", "uhm", "erm", "er", "hmm", "mm"],
        "es": ["eh", "em", "ehm", "mmm"],
        "fr": ["euh", "heu", "bah", "hum"],
        "de": ["äh", "ähm", "öhm", "hm"],
        "it": ["ehm", "eh", "uhm"],
        "pt": ["hum", "ahn", "hã"],
        "nl": ["eh", "ehm", "uh", "uhm"],
        "ko": ["음", "어"],
        "zh": ["嗯", "呃"],
        "ja": ["えーと", "えっと", "あのー", "うーん", "えー"],
    ]

    private static func fillers(in text: String, language: String?, inCharacters: Bool) -> [Count]? {
        guard let language, let list = fillerWords[String(language.prefix(2))] else { return nil }
        var counts: [String: Int] = [:]
        if inCharacters {
            // Longest first, each taken out once counted, so "えー" isn't
            // counted again inside "えーと".
            var rest = text
            for filler in list.sorted(by: { $0.count > $1.count }) {
                let pieces = rest.components(separatedBy: filler)
                if pieces.count > 1 { counts[filler] = pieces.count - 1 }
                rest = pieces.joined(separator: " ")
            }
        } else {
            let wanted = Set(list)
            for word in words(in: text, language: language) {
                let lower = word.lowercased()
                if wanted.contains(lower) { counts[lower, default: 0] += 1 }
            }
        }
        return counts.map { Count(word: $0.key, count: $0.value) }.sorted { $0.count > $1.count }
    }

    // MARK: - Frequent words

    /// Common in any talk and so saying little about this one.
    private static let everyday: Set<String> = [
        "thing", "things", "lot", "way", "time", "kind", "sort", "bit", "something", "anything",
        "everything", "nothing", "everyone", "everybody", "someone", "somebody", "anyone", "anybody",
        "people", "yeah", "okay", "ok", "good", "great", "right", "sure",
        "other", "same", "different", "little", "much", "many", "few", "more", "most", "own",
        "able", "last", "next", "new", "old", "big", "small", "first", "second", "whole", "real",
    ]

    /// The nouns and adjectives used most, by dictionary form where the
    /// language has one (so "projects" counts as "project").
    private static func frequentWords(in text: String, language: String?) -> (words: [Count], byWordClass: Bool) {
        let tagger = NLTagger(tagSchemes: [.lexicalClass, .lemma])
        tagger.string = text
        if let language { tagger.setLanguage(NLLanguage(language), range: text.startIndex ..< text.endIndex) }
        var counts: [String: Int] = [:]
        var tagged = false
        tagger.enumerateTags(in: text.startIndex ..< text.endIndex, unit: .word, scheme: .lexicalClass,
                             options: [.omitPunctuation, .omitWhitespace, .omitOther]) { tag, range in
            guard tag == .noun || tag == .adjective else { return true }
            tagged = true
            let lemma = tagger.tag(at: range.lowerBound, unit: .word, scheme: .lemma).0?.rawValue
            let word = (lemma ?? String(text[range])).lowercased()
            if keeps(word) { counts[word, default: 0] += 1 }
            return true
        }
        // A language the tagger can't read, or a device without its data
        // (every word comes back "other"): the longer words, less the
        // commonest English ones.
        if !tagged {
            for word in words(in: text, language: language) {
                let lower = word.lowercased()
                if keeps(lower), !functionWords.contains(lower),
                   lower.count >= (writtenWithoutSpaces(lower) ? 2 : 4) { counts[lower, default: 0] += 1 }
            }
        }
        let top = counts.filter { $0.value >= 2 }
            .map { Count(word: $0.key, count: $0.value) }
            .sorted { $0.count != $1.count ? $0.count > $1.count : $0.word < $1.word }
            .prefix(12)
        return (Array(top), tagged)
    }

    private static let functionWords: Set<String> = [
        "have", "that", "this", "with", "they", "them", "then", "than", "there", "their", "what", "when",
        "where", "which", "were", "been", "from", "will", "would", "could", "should", "about", "just",
        "like", "some", "into", "also", "very", "really", "your", "well", "here", "does", "doing", "done",
        "going", "know", "think", "want", "said", "says", "make", "made", "only", "over", "such", "even",
        "because", "after", "before", "these", "those", "being", "still", "again", "thank", "thanks",
    ]

    private static func keeps(_ word: String) -> Bool {
        guard let first = word.unicodeScalars.first, first.properties.isAlphabetic else { return false }
        return !everyday.contains(word) && (writtenWithoutSpaces(word) ? word.count >= 2 : word.count >= 3)
    }
}
