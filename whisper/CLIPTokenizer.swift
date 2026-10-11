import Foundation

/// CLIP's tokenizer, as Stable Diffusion 1.5 reads prompts.
///
/// Byte-level BPE like `ACETokenizer`, with CLIP's differences: text is
/// lower-cased and its whitespace collapsed first, a word's last symbol
/// carries an end-of-word mark (`</w>`) so "art" alone and "art" inside
/// "artist" are different tokens, and every prompt is exactly 77 ids —
/// start, at most 75 tokens, end, then end-of-text as padding.
struct CLIPTokenizer {
    static let length = 77
    static let start: Int32 = 49406
    static let end: Int32 = 49407

    private let vocabulary: [String: Int32]
    private let ranks: [String: Int]
    private let expression: NSRegularExpression
    private let byteToCharacter: [UInt8: Character]

    init(vocabularyURL: URL, mergesURL: URL) throws {
        guard let table = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabularyURL)) as? [String: Int] else {
            throw ACETokenizer.TokenizerError.badVocabulary
        }
        vocabulary = table.mapValues(Int32.init)
        var order: [String: Int] = [:]
        var rank = 0
        for line in try String(contentsOf: mergesURL, encoding: .utf8).split(separator: "\n") {
            if line.hasPrefix("#version") { continue }
            let pieces = line.split(separator: " ")
            guard pieces.count == 2 else { continue }
            order["\(pieces[0]) \(pieces[1])"] = rank
            rank += 1
        }
        ranks = order
        expression = try NSRegularExpression(
            pattern: "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            options: [.caseInsensitive])
        byteToCharacter = Self.byteMapping()
    }

    /// The 77 ids the text encoder reads. A longer prompt is cut, as it is
    /// upstream: the model never sees past its 75th token.
    func encode(_ text: String) -> [Int32] {
        let cleaned = text.precomposedStringWithCanonicalMapping
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .lowercased()
        var tokens: [Int32] = []
        let range = NSRange(cleaned.startIndex ..< cleaned.endIndex, in: cleaned)
        for match in expression.matches(in: cleaned, range: range) {
            guard let piece = Range(match.range, in: cleaned) else { continue }
            let word = String(cleaned[piece])
            if word == "<|startoftext|>" { tokens.append(Self.start); continue }
            if word == "<|endoftext|>" { tokens.append(Self.end); continue }
            let mapped = Array(word.utf8).compactMap { byteToCharacter[$0] }.map(String.init)
            tokens.append(contentsOf: merge(mapped).compactMap { vocabulary[$0] })
        }
        var ids = [Self.start] + tokens.prefix(Self.length - 2) + [Self.end]
        ids += [Int32](repeating: Self.end, count: Self.length - ids.count)
        return ids
    }

    /// Merges a word's symbols, the last carrying the end-of-word mark.
    private func merge(_ characters: [String]) -> [String] {
        var symbols = characters
        guard !symbols.isEmpty else { return [] }
        symbols[symbols.count - 1] += "</w>"
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0 ..< (symbols.count - 1) {
                if let rank = ranks["\(symbols[index]) \(symbols[index + 1])"], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            guard bestIndex >= 0 else { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }
        return symbols
    }

    /// GPT-2's byte-to-printable mapping, which CLIP shares.
    private static func byteMapping() -> [UInt8: Character] {
        var bytes: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var mapped = bytes
        var next = 0
        for byte in 0...255 where !bytes.contains(byte) {
            bytes.append(byte)
            mapped.append(256 + next)
            next += 1
        }
        var table: [UInt8: Character] = [:]
        for (byte, code) in zip(bytes, mapped) {
            table[UInt8(byte)] = Character(UnicodeScalar(code)!)
        }
        return table
    }
}
