import SwiftUI

/// Text that can run to thousands of words: a transcript as it is written,
/// or one shown whole.
///
/// A single `Text` that long is one paragraph, and laying out a paragraph
/// takes time growing with the square of its length — measured with Core
/// Text on an M4, an English transcript took 8 ms at ten minutes, 83 ms at
/// thirty and 300 ms at an hour, on every change. Live and Transcribe change
/// theirs several times a second, so a long session got steadily laggier.
/// Cut into paragraphs, each is laid out on its own (an hour takes 5 ms), and
/// as the text grows only the last one, the one still being written, changes.
///
/// Font, colour and line spacing come from the environment, as for `Text`.
struct ParagraphText: View {
    let text: String
    /// Shown fainter after the text: what is still being heard.
    var tail: String = ""
    var tailColor: Color = Studio.mute
    var spacing: CGFloat = 12

    var body: some View {
        let paragraphs = Paragraphs.split(text)
        VStack(alignment: .leading, spacing: spacing) {
            ForEach(paragraphs.indices, id: \.self) { index in
                if index == paragraphs.count - 1, !tail.isEmpty {
                    Text("\(Text(verbatim: paragraphs[index])) \(Text(verbatim: tail).foregroundColor(tailColor))")
                } else {
                    Text(verbatim: paragraphs[index])
                }
            }
            if paragraphs.isEmpty, !tail.isEmpty {
                Text(verbatim: tail).foregroundColor(tailColor)
            }
        }
    }
}

enum Paragraphs {
    /// `text` in paragraphs of a few sentences, cut after a full stop. Text
    /// without full stops — some speech comes out unpunctuated — is cut at a
    /// space once a paragraph has grown long, or anywhere for scripts written
    /// without spaces. Line breaks already in the text are kept.
    ///
    /// By Unicode scalar rather than by `Character`: this runs on every
    /// change, and grapheme breaking made it several times slower.
    static func split(_ text: String, sentences: Int = 6, maxLength: Int = 700) -> [String] {
        let scalars = text.unicodeScalars
        var paragraphs: [String] = []
        var start = scalars.startIndex
        var count = 0
        var ended = 0
        var previous: Unicode.Scalar = " "

        func cut(at end: String.UnicodeScalarView.Index) {
            let piece = String(scalars[start ..< end]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty { paragraphs.append(piece) }
            (start, count, ended) = (end, 0, 0)
        }

        var index = scalars.startIndex
        while index < scalars.endIndex {
            let scalar = scalars[index]
            if scalar == "\n" {
                cut(at: index)
                start = scalars.index(after: index)
            } else {
                let space = scalar.isASCII ? scalar == " " || scalar == "\t" : scalar.properties.isWhitespace
                // An ASCII full stop ends a sentence only before a space, so
                // "3.5" stays whole; a wide one ends it at once (below).
                if space, previous == "." || previous == "!" || previous == "?" || previous == "…" { ended += 1 }
                if ended >= sentences || (count >= maxLength && space) {
                    cut(at: index)
                } else if count >= maxLength * 2,
                          // Never inside a character: not before a combining
                          // mark or either side of a joiner.
                          !scalar.properties.isGraphemeExtend, scalar != "\u{200D}", previous != "\u{200D}" {
                    cut(at: index)
                }
                if scalar == "。" || scalar == "！" || scalar == "？" { ended += 1 }
                count += 1
            }
            previous = scalar
            index = scalars.index(after: index)
        }
        cut(at: scalars.endIndex)
        return paragraphs
    }
}
