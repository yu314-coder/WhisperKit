//  Ported from ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5), MIT.
//  Verified against the PyTorch reference stage by stage; see ports/README.md.

import Foundation

/// Byte-level BPE, as Qwen3 uses it.
///
/// Written rather than taken from a package because the alternative pulls five
/// transitive dependencies into the app for one tokenizer, and this algorithm
/// is small and exactly checkable — the reference token ids either match or
/// they do not.
///
/// Three stages, in order:
///   1. split the text on the model's pre-tokenizer pattern;
///   2. map each UTF-8 byte to a printable stand-in character, so any byte
///      sequence becomes a string the merge table can address;
///   3. merge adjacent pairs, always taking the lowest-ranked merge available,
///      until none applies.
struct ACETokenizer {
    /// `<|endoftext|>`, which this model appends to every prompt.
    static let endOfText: Int32 = 151643

    private let vocabulary: [String: Int32]
    private let ranks: [String: Int]
    private let expression: NSRegularExpression
    private let byteToCharacter: [UInt8: Character]

    init(vocabularyURL: URL, mergesURL: URL) throws {
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabularyURL))
        guard let table = raw as? [String: Int] else {
            throw TokenizerError.badVocabulary
        }
        vocabulary = table.mapValues(Int32.init)

        // Merge order is priority: the earlier a pair appears, the sooner it
        // is applied.
        var order: [String: Int] = [:]
        let lines = try String(contentsOf: mergesURL, encoding: .utf8).split(separator: "\n")
        var rank = 0
        for line in lines {
            if line.hasPrefix("#version") { continue }
            let pieces = line.split(separator: " ")
            guard pieces.count == 2 else { continue }
            order["\(pieces[0]) \(pieces[1])"] = rank
            rank += 1
        }
        ranks = order

        expression = try NSRegularExpression(
            pattern: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")

        byteToCharacter = Self.byteMapping()
    }

    func encode(_ text: String, appendEndOfText: Bool = true) -> [Int32] {
        var ids: [Int32] = []
        let range = NSRange(text.startIndex ..< text.endIndex, in: text)
        for match in expression.matches(in: text, range: range) {
            guard let piece = Range(match.range, in: text) else { continue }
            let mapped = String(Array(text[piece].utf8).compactMap { byteToCharacter[$0] })
            ids.append(contentsOf: merge(mapped).compactMap { vocabulary[$0] })
        }
        if appendEndOfText { ids.append(Self.endOfText) }
        return ids
    }

    /// Repeatedly applies the best-ranked adjacent merge.
    private func merge(_ text: String) -> [String] {
        var symbols = text.map(String.init)
        guard symbols.count > 1 else { return symbols }

        while true {
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
            if symbols.count == 1 { break }
        }
        return symbols
    }

    /// The byte-to-printable mapping every byte-level BPE shares: printable
    /// ASCII and Latin-1 keep their own character, and the remaining bytes are
    /// lifted above U+0100 so nothing collides with a real character.
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

    enum TokenizerError: Error { case badVocabulary }
}
