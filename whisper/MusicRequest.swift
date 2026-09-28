import Foundation

/// What was typed on the Music tab, sorted into what the model needs: a
/// description of the music and the words to sing, wherever they were
/// typed.
///
/// People paste a whole song into the prompt box, as Suno takes it — a
/// style line, then the lyrics. Those lyrics used to reach the model as
/// part of the description, so nothing was sung.
struct MusicRequest: Equatable {
    /// The prompt without the lyrics found in it. Still multi-line: a
    /// timeline in it is read line by line (`PromptMetadata`); the pipeline
    /// joins it onto one line for the model (`ACEPipeline.oneLine`).
    var description: String
    /// The lyrics box, then any lyrics found in the prompt; headings such as
    /// "Chorus:" written as the model's tags.
    var lyrics: String
    /// Lines of lyrics taken from the prompt, so the screen can say so.
    var lyricLinesFromPrompt: Int

    /// - Parameter found: lyric lines `LyricsFinder` found in `prompt`, or
    ///   nil to use the rules below alone. Its answer is used when every line
    ///   is found in the prompt; an answer of none gives way only to lyrics
    ///   the prompt marks outright (quotes after "lyrics", a "Lyrics:" label,
    ///   section headings).
    init(prompt: String, lyrics box: String, found aiLines: [String]? = nil) {
        let rules = Self.splitLyrics(prompt)
        var split = (description: rules.description, lyrics: rules.lyrics)
        if let aiLines {
            if aiLines.isEmpty {
                if !rules.marked { split = (prompt.trimmingCharacters(in: .whitespacesAndNewlines), "") }
            } else if let located = Self.locate(aiLines, in: prompt) {
                split = located
            }
        }
        let (description, found) = split
        self.description = description
        lyricLinesFromPrompt = found.split(whereSeparator: \.isNewline)
            .filter { !Self.isHeading(String($0)) }.count
        let typed = box.trimmingCharacters(in: .whitespacesAndNewlines)
        let combined: String
        if found.isEmpty || Self.comparable(found) == Self.comparable(typed) {
            combined = typed
        } else if typed.isEmpty {
            combined = found
        } else {
            combined = typed + "\n\n" + found
        }
        lyrics = Self.bracketHeadings(combined)
    }

    var hasLyrics: Bool { !lyrics.isEmpty }

    // MARK: - Finding lyrics in a prompt

    /// Splits a prompt into its description and any lyrics after it.
    ///
    /// Lyrics start at, in order of certainty: a "Lyrics:" heading (or
    /// 歌词：/歌詞：), taking the rest of its line; a section heading on a
    /// line of its own ("[Verse]", "Chorus:", "副歌"); or, after a blank
    /// line, a block of two or more lines that read as sung lines rather
    /// than a description — a few words each, no "name: value" pairs, no
    /// timeline, and mostly free of the words descriptions are made of.
    /// The first block is always the description unless it opens with one
    /// of the headings.
    static func splitLyrics(_ prompt: String) -> (description: String, lyrics: String, marked: Bool) {
        let lines = prompt.components(separatedBy: .newlines)
        func joined(_ slice: ArraySlice<String>) -> String {
            slice.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Quoted, after a word that says they are lyrics: lyric is "…",
        // the lyrics “…”, she sings "…", 歌词是「…」.
        let whole = NSRange(prompt.startIndex..., in: prompt)
        for quote in quotation.matches(in: prompt, range: whole) {
            guard let opening = Range(quote.range, in: prompt),
                  prompt[..<opening.lowerBound].range(of: lyricLabel + #"$"#, options: .regularExpression) != nil,
                  let inner = (1 ..< quote.numberOfRanges).lazy.compactMap({ Range(quote.range(at: $0), in: prompt) }).first,
                  let located = locate(lyricLines(String(prompt[inner])), in: prompt) else { continue }
            return (located.description, located.lyrics, true)
        }
        // After "Lyrics:" anywhere in a line, to the end.
        for (index, line) in lines.enumerated() {
            guard let label = line.range(of: #"(?i:\blyrics?(?:\s+(?:is|are|go|goes|should\s+be|will\s+be))?|(?:歌词|歌詞)(?:是|为|為|应该是|應該是)?)\s*[:：]\s*"#, options: .regularExpression) else { continue }
            let rest = String(line[label.upperBound...])
            let text = ([rest] + lines[(index + 1)...]).joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            let sung = text.contains(where: \.isNewline) ? text : lyricLines(text).joined(separator: "\n")
            let before = (lines[..<index] + [String(line[..<label.lowerBound])]).joined(separator: "\n")
            return (tidy(before), sung, true)
        }
        if let index = lines.firstIndex(where: isHeading) {
            return (joined(lines[..<index]), joined(lines[index...]), true)
        }

        var blocks: [[String]] = [[]]
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { blocks[blocks.count - 1].append(trimmed) }
            else if !blocks[blocks.count - 1].isEmpty { blocks.append([]) }
        }
        blocks.removeAll { $0.isEmpty }
        guard blocks.count >= 2,
              let first = blocks.indices.dropFirst().first(where: { readsAsLyrics(blocks[$0]) }) else {
            return (prompt.trimmingCharacters(in: .whitespacesAndNewlines), "", false)
        }
        func text(_ part: ArraySlice<[String]>) -> String {
            part.map { $0.joined(separator: "\n") }.joined(separator: "\n\n")
        }
        return (text(blocks[..<first]), text(blocks[first...]), false)
    }

    // MARK: - Taking known lyrics out of a prompt

    /// Finds `lines` in `prompt` — in order, ignoring case, spacing,
    /// punctuation and quotes — and returns the prompt without them (or
    /// the label and quotation marks around them), and the lyrics in the
    /// prompt's own characters. Nil unless every line is there: a line
    /// that isn't was written, not found.
    static func locate(_ lines: [String], in prompt: String) -> (description: String, lyrics: String)? {
        let source = letters(prompt)
        var cursor = 0
        var spans: [Range<String.Index>] = []
        for line in lines {
            let wanted = letters(line).map(\.letter)
            guard !wanted.isEmpty else { continue }
            guard let start = (cursor ... max(cursor, source.count - wanted.count)).first(where: { at in
                at + wanted.count <= source.count && (0 ..< wanted.count).allSatisfy { source[at + $0].letter == wanted[$0] }
            }) else { return nil }
            let end = start + wanted.count - 1
            spans.append(source[start].index ..< prompt.index(after: source[end].index))
            cursor = end + 1
        }
        guard let first = spans.first, let last = spans.last else { return nil }
        var start = first.lowerBound
        var end = last.upperBound

        // Section headings on the lines just above the lyrics belong to them.
        let firstLineStart = prompt[..<start].lastIndex(of: "\n").map { prompt.index(after: $0) } ?? prompt.startIndex
        if prompt[firstLineStart ..< start].allSatisfy(\.isWhitespace) {
            var lineStart = firstLineStart
            while lineStart > prompt.startIndex {
                let newline = prompt.index(before: lineStart)
                let previousStart = prompt[..<newline].lastIndex(of: "\n").map { prompt.index(after: $0) } ?? prompt.startIndex
                guard isHeading(String(prompt[previousStart ..< newline])) else { break }
                lineStart = previousStart
                start = previousStart
            }
        }

        let multiLine = prompt[start ..< end].contains(where: \.isNewline)
        let sung = multiLine
            ? String(prompt[start ..< end]).trimmingCharacters(in: .whitespacesAndNewlines)
            : spans.map { String(prompt[$0]) }.joined(separator: "\n")

        // The label and opening quote before, the closing quote after.
        if let label = String(prompt[..<start]).range(of: lyricLabel + #"\s*$"#, options: .regularExpression) {
            start = label.lowerBound
        } else if let quote = String(prompt[..<start]).range(of: #"["“„«「『'‘]\s*$"#, options: .regularExpression) {
            start = quote.lowerBound
        }
        let after = String(prompt[end...])
        if let close = after.range(of: #"^[^\S\n]*[!！?？.。…]*[^\S\n]*["”“»」』'’]"#, options: .regularExpression) {
            end = prompt.index(end, offsetBy: after.distance(from: after.startIndex, to: close.upperBound))
        }
        let description = tidy(String(prompt[..<start]) + " " + String(prompt[end...]))
        return (description, sung)
    }

    /// A word or phrase that says what follows is lyrics, with any colon
    /// and opening quotation mark: "lyric is \"", ", the lyrics: “",
    /// "she sings \"", "歌词是「". For matching at the end of the text
    /// before a quotation.
    private static let lyricLabel =
        #"(?i:[\s,，;；]*(?:(?:(?:and|with)\s+)?(?:the\s+)?(?:lyrics?|words)(?:\s+(?:is|are|go|goes|say|says|read|reads|should\s+be|will\s+be))?|(?:(?:she|he|they|it|the\s+singer)\s+)?(?:sings?|singing|sung)|(?:that|which)\s+goes|(?:歌词|歌詞)(?:是|为|為|应该是|應該是)?|唱(?:着|著|的是)?)[\s:：=]*["“„«「『'‘]?)"#

    /// Text in quotation marks: straight, curly, low, guillemets, corner
    /// brackets.
    private static let quotation = try! NSRegularExpression(
        pattern: #""([^"]+)"|“([^”]+)”|„([^“]+)“|«([^»]+)»|「([^」]+)」|『([^』]+)』|'([^']{3,})'"#)

    /// The letters and digits of `text`, lowercased, with where each is.
    private static func letters(_ text: String) -> [(letter: String, index: String.Index)] {
        var result: [(letter: String, index: String.Index)] = []
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            if character.isLetter || character.isNumber { result.append((character.lowercased(), index)) }
            index = text.index(after: index)
        }
        return result
    }

    /// Lyrics written on one line, as lines: at " / " or "|", after a
    /// sentence's end, and for Chinese and Japanese at their commas and
    /// spaces.
    static func lyricLines(_ text: String) -> [String] {
        var split = text.replacingOccurrences(of: #"\s*[/|｜]\s*"#, with: "\n", options: .regularExpression)
            .replacingOccurrences(of: #"([.!?])\s+"#, with: "$1\n", options: .regularExpression)
        if ["zh", "ja"].contains(language(of: text) ?? "") {
            split = split.replacingOccurrences(of: #"[，、。！？\s]+"#, with: "\n", options: .regularExpression)
        }
        return split.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// What is left of a prompt once lyrics are taken out: no doubled or
    /// dangling separators, no trailing "and" or "with".
    static func tidy(_ text: String) -> String {
        let lines = text.components(separatedBy: .newlines).map { line -> String in
            var line = line.replacingOccurrences(of: #"[ \t]{2,}"#, with: " ", options: .regularExpression)
                .replacingOccurrences(of: #"\s+([,，.。;；!！?？])"#, with: "$1", options: .regularExpression)
                .replacingOccurrences(of: #"([,，;；])(?:\s*[,，;；])+"#, with: "$1", options: .regularExpression)
            while let tail = line.range(of: #"(?i)(?:[\s,，;；:：\-–—、]+|\s+(?:and|with|the|a))$"#, options: .regularExpression) {
                line.removeSubrange(tail)
            }
            while let lead = line.range(of: #"^[\s,，;；:：\-–—、]+"#, options: .regularExpression) {
                line.removeSubrange(lead)
            }
            return line
        }
        return lines.joined(separator: "\n")
            .replacingOccurrences(of: #"\n{3,}"#, with: "\n\n", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// A section heading on a line of its own: "[Chorus]", "Verse 2:",
    /// "Pre-Chorus", "(Bridge)", "副歌：".
    static func isHeading(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.range(of: #"^\[[^\]\n]{1,40}\]$"#, options: .regularExpression) != nil { return true }
        return trimmed.range(of: sectionName, options: .regularExpression) != nil
    }

    private static let sectionName =
        #"^[(（]?(?i:intro|verse|pre-?chorus|chorus|post-?chorus|hook|refrain|bridge|breakdown|interlude|outro|主歌|副歌|前奏|间奏|間奏|尾奏|桥段|橋段|导歌|導歌|预副歌|預副歌)\s*\d*[)）]?\s*[:：]?$"#

    /// Sung lines, not a description: short lines of a few words, none a
    /// "name: value" pair, a timeline range or a bullet, and fewer than
    /// half naming instruments or production.
    private static func readsAsLyrics(_ block: [String]) -> Bool {
        guard block.count >= 2 else { return false }
        for line in block {
            if line.count > 90 { return false }
            if line.hasPrefix("-") || line.hasPrefix("•") || line.hasPrefix("*") { return false }
            if line.range(of: #"^[^:：]{1,24}[:：]\s*\S"#, options: .regularExpression) != nil { return false }
            if line.range(of: #"\d{1,2}:[0-5]\d\s*(–|—|-|to)"#, options: .regularExpression) != nil { return false }
        }
        let wordy = block.filter { line in
            line.contains(" ") ? line.split(separator: " ").count >= 3 : line.count >= 4
        }
        guard wordy.count * 2 >= block.count else { return false }
        let describing = block.filter { line in
            let lower = line.lowercased()
            return musicWords.contains { lower.contains($0) }
        }
        return describing.count * 2 < block.count
    }

    /// Words that describe music rather than sing it.
    private static let musicWords = [
        "piano", "guitar", "drum", "bass", "synth", "string", "violin", "cello", "brass", "horn",
        "tempo", "bpm", "beat", "groove", "melody", "chord", "vocal", "instrumental", "arrangement",
        "reverb", "mix", "lo-fi", "lofi", "orchestra", "percussion", "pad", "builds", "genre",
        "钢琴", "鋼琴", "吉他", "鼓", "贝斯", "貝斯", "弦乐", "弦樂", "合成器", "节奏", "節奏",
        "旋律", "编曲", "編曲", "伴奏", "混音", "乐器", "樂器",
    ]

    // MARK: - Lyrics as the model reads them

    /// "Chorus:" and "(Verse 2)" as the model's "[Chorus]" and "[Verse 2]".
    /// Bracketed tags stay as they are.
    static func bracketHeadings(_ lyrics: String) -> String {
        lyrics.components(separatedBy: .newlines).map { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.hasPrefix("["), isHeading(trimmed) else { return line }
            let name = trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "()（）:： "))
            return "[\(name.prefix(1).uppercased() + name.dropFirst())]"
        }.joined(separator: "\n")
    }

    /// Lyrics compared without case, spacing or blank lines, so the same
    /// words in both boxes are sung once.
    private static func comparable(_ text: String) -> String {
        text.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.joined(separator: " ")
    }

    // MARK: - Sung language

    /// The language lyrics are written in, when their script says it: Han
    /// is Chinese unless kana appear (Japanese), Hangul Korean, and so on.
    /// Nil for Latin script, which many languages share.
    static func language(of lyrics: String) -> String? {
        var counts: [String: Int] = [:]
        for scalar in lyrics.unicodeScalars {
            let v = scalar.value
            let code: String?
            switch v {
            case 0x3040 ... 0x30FF: code = "ja"
            case 0x4E00 ... 0x9FFF, 0x3400 ... 0x4DBF: code = "zh"
            case 0xAC00 ... 0xD7AF, 0x1100 ... 0x11FF: code = "ko"
            case 0x0400 ... 0x04FF: code = "ru"
            case 0x0600 ... 0x06FF: code = "ar"
            case 0x0590 ... 0x05FF: code = "he"
            case 0x0900 ... 0x097F: code = "hi"
            case 0x0980 ... 0x09FF: code = "bn"
            case 0x0A00 ... 0x0A7F: code = "pa"
            case 0x0B80 ... 0x0BFF: code = "ta"
            case 0x0C00 ... 0x0C7F: code = "te"
            case 0x0E00 ... 0x0E7F: code = "th"
            case 0x0370 ... 0x03FF: code = "el"
            default: code = nil
            }
            if let code { counts[code, default: 0] += 1 }
        }
        if counts["ja", default: 0] > 0 { return "ja" }
        guard let (code, _) = counts.max(by: { $0.value < $1.value }) else { return nil }
        return Languages.aceStepCodes.contains(code) ? code : nil
    }
}
