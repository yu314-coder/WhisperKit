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

    /// Written literally in prompt templates, read as single tokens. The
    /// planner's chat format uses the turn markers and reasoning tags; the
    /// text encoder only ever sees `<|endoftext|>`.
    static let specialTokens: [String: Int32] = [
        "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
        "<think>": 151667, "</think>": 151668,
    ]

    private let vocabulary: [String: Int32]
    private let tokenText: [Int32: String]
    private let ranks: [String: Int]
    private let expression: NSRegularExpression
    private let byteToCharacter: [UInt8: Character]
    private let characterToByte: [Character: UInt8]

    init(vocabularyURL: URL, mergesURL: URL) throws {
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabularyURL))
        guard let table = raw as? [String: Int] else {
            throw TokenizerError.badVocabulary
        }
        vocabulary = table.mapValues(Int32.init)
        var reverse: [Int32: String] = [:]
        for (text, id) in table { reverse[Int32(id)] = text }
        tokenText = reverse

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
        characterToByte = Dictionary(uniqueKeysWithValues: byteToCharacter.map { ($1, $0) })
    }

    /// A literal special token inside a template is that token, not a dozen
    /// characters of text. ACE-Step's prompt templates close with
    /// `<|endoftext|>` and the text encoder's tokenizer then appends another,
    /// so both must come out as id 151643 — byte-level merging the literal
    /// gives a sequence the model never saw.
    ///
    /// - Parameter appendEndOfText: the text encoder's tokenizer appends one;
    ///   the planner's does not.
    func encode(_ text: String, appendEndOfText: Bool = true) -> [Int32] {
        // Qwen's tokenizer normalizes to NFC first; a decomposed accent would
        // otherwise split into different byte pairs.
        var rest = Substring(text.precomposedStringWithCanonicalMapping)
        var ids: [Int32] = []
        while !rest.isEmpty {
            let next = Self.specialTokens.keys
                .compactMap { marker in rest.range(of: marker).map { (marker, $0) } }
                .min { $0.1.lowerBound < $1.1.lowerBound }
            guard let (marker, range) = next else {
                ids.append(contentsOf: encodeOrdinary(String(rest)))
                break
            }
            ids.append(contentsOf: encodeOrdinary(String(rest[..<range.lowerBound])))
            ids.append(Self.specialTokens[marker]!)
            rest = rest[range.upperBound...]
        }
        if appendEndOfText { ids.append(Self.endOfText) }
        return ids
    }

    /// Text for ids from the base vocabulary; special and planner code tokens
    /// decode to nothing.
    func decode(_ ids: [Int32]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            guard let text = tokenText[id] else { continue }
            bytes.append(contentsOf: text.compactMap { characterToByte[$0] })
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func encodeOrdinary(_ text: String) -> [Int32] {
        var ids: [Int32] = []
        let range = NSRange(text.startIndex ..< text.endIndex, in: text)
        for match in expression.matches(in: text, range: range) {
            guard let piece = Range(match.range, in: text) else { continue }
            let mapped = String(Array(text[piece].utf8).compactMap { byteToCharacter[$0] })
            ids.append(contentsOf: merge(mapped).compactMap { vocabulary[$0] })
        }
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
