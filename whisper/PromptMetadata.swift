import Foundation

/// What a music prompt says about its own length, tempo, key and meter.
///
/// The prompt is the control. Someone who writes "a 1 minute 21 second
/// underscore at 120 BPM in C minor" has already said how long it runs, so
/// there is no separate length control to disagree with it; a prompt that
/// names no length gets a stated default.
struct PromptMetadata: Equatable {
    var seconds: Int?
    var bpm: Int?
    /// As the planner writes it: "C minor", "F# major".
    var keyscale: String?
    /// Beats per bar: 2, 3, 4 or 6.
    var timeSignature: Int?

    /// 6:24 is the longest the Neural Engine transformer was built for: a
    /// piece is rendered 20% past its length and cut, and 6:24 plus that
    /// fills its largest size. A longer request is made at 6:24, and the
    /// summary under the prompt shows it.
    static let secondsRange = 5 ... 384

    init(parsing prompt: String) {
        seconds = Self.duration(in: prompt)
        bpm = Self.firstInt(in: prompt, patterns: [#"(\d{2,3})\s*-?\s*bpm\b"#, #"\bbpm\s*[:=]?\s*(\d{2,3})\b"#])
            .flatMap { (30 ... 300).contains($0) ? $0 : nil }
        keyscale = Self.key(in: prompt)
        timeSignature = Self.meter(in: prompt)
    }

    /// Whether anything was found, for showing what the prompt set.
    var isEmpty: Bool { seconds == nil && bpm == nil && keyscale == nil && timeSignature == nil }

    // MARK: - Length

    /// The first length named, clamped to what the models can make. The
    /// first, because a prompt describing "a 90-second spot with a 10 second
    /// intro" means the whole piece before its parts.
    static func duration(in text: String) -> Int? {
        let lowered = text.lowercased()
        var found: [(position: Int, seconds: Double)] = []
        func scan(_ pattern: String, _ value: ([Double?]) -> Double?) {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { return }
            let range = NSRange(lowered.startIndex ..< lowered.endIndex, in: lowered)
            for match in regex.matches(in: lowered, range: range) {
                let groups: [Double?] = (1 ..< match.numberOfRanges).map { index in
                    Range(match.range(at: index), in: lowered).flatMap { Double(lowered[$0]) }
                }
                if let seconds = value(groups) { found.append((match.range.location, seconds)) }
            }
        }
        // 1:21
        scan(#"(?<![\d:/.])(\d{1,2}):([0-5]\d)(?![\d:])"#) { g in
            guard let m = g[0], let s = g[1] else { return nil }
            return m * 60 + s
        }
        // 1 minute 21 seconds, 1-min 21-sec, 1m21s, 2 minutes, 1.5 min
        scan(#"(\d+(?:\.\d+)?)\s*-?\s*(?:minutes?|mins?|m)(?:\b|(?=\d))(?:\s*(?:and\s*)?(\d+)\s*-?\s*(?:seconds?|secs?|s)\b)?"#) { g in
            guard let m = g[0] else { return nil }
            return m * 60 + (g.count > 1 ? g[1] ?? 0 : 0)
        }
        // 81 seconds, 81-second, 81 sec — but not a drum machine's "808s".
        scan(#"(?<![\d.])(\d+(?:\.\d+)?)\s*-?\s*(seconds?|secs?)\b"#) { g in g[0] }
        scan(#"(?<![\d.:])(\d{1,3})s\b"#) { g in
            guard let s = g[0], ![303, 606, 707, 808, 909].contains(Int(s)) else { return nil }
            return s
        }
        // 1分21秒, 2分钟, 81秒
        scan(#"(\d+)\s*分(?:钟|鐘)?\s*(?:(\d+)\s*秒)?"#) { g in
            guard let m = g[0] else { return nil }
            return m * 60 + (g.count > 1 ? g[1] ?? 0 : 0)
        }
        scan(#"(?<!分)(?<!\d)(\d+)\s*秒"#) { g in g[0] }

        // A "1 minute 21 seconds" also contains a bare "21 seconds"; the
        // earliest match is the whole phrase.
        guard let first = found.min(by: { $0.position < $1.position }) else { return nil }
        let rounded = Int(first.seconds.rounded())
        return min(max(rounded, secondsRange.lowerBound), secondsRange.upperBound)
    }

    // MARK: - Tempo, key, meter

    private static func firstInt(in text: String, patterns: [String]) -> Int? {
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
                  let match = regex.firstMatch(in: text, range: NSRange(text.startIndex ..< text.endIndex, in: text)),
                  let range = Range(match.range(at: 1), in: text),
                  let value = Int(text[range]) else { continue }
            return value
        }
        return nil
    }

    /// "C minor", "F# major", "B flat minor", "Ebm"-style shorthand is left
    /// alone. The note must be a capital: "a minor detail" is not a key.
    private static func key(in text: String) -> String? {
        let pattern = #"\b([A-G])\s?(#|♯|b|♭|-?sharp|-?flat)?\s+(major|minor|maj|min)\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: text, range: NSRange(text.startIndex ..< text.endIndex, in: text)),
              let note = Range(match.range(at: 1), in: text).map({ String(text[$0]) }),
              let mode = Range(match.range(at: 3), in: text).map({ String(text[$0]) })
        else { return nil }
        let accidental = Range(match.range(at: 2), in: text).map { String(text[$0]) } ?? ""
        let symbol: String
        switch accidental {
        case "#", "♯", "sharp", "-sharp": symbol = "#"
        case "b", "♭", "flat", "-flat":   symbol = "b"
        default:                          symbol = ""
        }
        return "\(note)\(symbol) \(mode.hasPrefix("maj") ? "major" : "minor")"
    }

    private static func meter(in text: String) -> Int? {
        guard let regex = try? NSRegularExpression(pattern: #"(?<![\d/])([2346])/(4|8)(?![\d/])"#),
              let match = regex.firstMatch(in: text, range: NSRange(text.startIndex ..< text.endIndex, in: text)),
              let beats = Range(match.range(at: 1), in: text).flatMap({ Int(text[$0]) }),
              let unit = Range(match.range(at: 2), in: text).map({ String(text[$0]) })
        else { return nil }
        // The planner knows 2, 3, 4 and 6 beats; 6/8 is the only eighth-note
        // meter it has a slot for.
        if unit == "8" { return beats == 6 ? 6 : nil }
        return beats
    }
}
