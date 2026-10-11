import Foundation

/// Gemma 2's tokenizer, as SANA-Sprint reads prompts, from the integer
/// tables in `gemma_bpe.bin` (written by ports/imagegen/export_sana_mlx.py)
/// rather than the 34 MB `tokenizer.json`.
///
/// Hugging Face's rules for this tokenizer, which this follows:
///
/// * **Added tokens first.** Newline runs, "<start_of_turn>", HTML tags and
///   the like are found in the raw text before anything else, the longest
///   at each position, and the text between them is tokenized on its own.
///   So each line of SANA's instruction prefix is a separate piece, with
///   "\n" its own token between them.
/// * **One word per piece.** Spaces become "▁" and nothing splits on them:
///   tokens like "▁cat" come out of merges.
/// * **Characters, then bytes.** Each Unicode scalar starts as its own
///   symbol; one with no token of its own becomes its UTF-8 bytes'
///   fallback tokens (<0xE2>…).
/// * **Merges by rank.** The lowest-ranked adjacent pair merges first, the
///   leftmost among equals, until no pair has a merge.
struct GemmaTokenizer {
    static let bos: Int32 = 2

    enum Failure: LocalizedError {
        case badTables

        var errorDescription: String? { "The prompt reader's tables are damaged. Delete the model and download it again." }
    }

    private let characters: [UInt32: UInt32]
    private let byteTokens: [UInt32]
    /// Merge pairs as `left << 32 | right`, sorted, with what each makes and
    /// its rank alongside.
    private let pairs: [UInt64]
    private let merged: [UInt32]
    private let ranks: [UInt32]
    /// Added tokens by first byte, longest first.
    private let added: [UInt8: [(bytes: [UInt8], id: UInt32)]]

    init(url: URL) throws {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let words: [UInt32] = data.withUnsafeBytes { raw in
            (0 ..< raw.count / 4).map { UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)) }
        }
        guard words.count >= 4, words[0] == 0x4550_4247 else { throw Failure.badTables }  // "GBPE"
        let (characterCount, mergeCount, addedCount) = (Int(words[1]), Int(words[2]), Int(words[3]))
        var at = 4
        let tableEnd = at + 2 * characterCount + 256 + 4 * mergeCount + 2 * addedCount
        guard words.count >= tableEnd else { throw Failure.badTables }

        var characters: [UInt32: UInt32] = [:]
        characters.reserveCapacity(characterCount)
        for index in 0 ..< characterCount { characters[words[at + 2 * index]] = words[at + 2 * index + 1] }
        self.characters = characters
        at += 2 * characterCount
        byteTokens = Array(words[at ..< at + 256])
        at += 256

        var pairs = [UInt64](), merged = [UInt32](), ranks = [UInt32]()
        pairs.reserveCapacity(mergeCount); merged.reserveCapacity(mergeCount); ranks.reserveCapacity(mergeCount)
        for index in 0 ..< mergeCount {
            let base = at + 4 * index
            pairs.append(UInt64(words[base]) << 32 | UInt64(words[base + 1]))
            merged.append(words[base + 2])
            ranks.append(words[base + 3])
        }
        (self.pairs, self.merged, self.ranks) = (pairs, merged, ranks)
        at += 4 * mergeCount

        var added: [UInt8: [(bytes: [UInt8], id: UInt32)]] = [:]
        var textOffset = tableEnd * 4
        for index in 0 ..< addedCount {
            let (id, length) = (words[at + 2 * index], Int(words[at + 2 * index + 1]))
            guard length > 0, textOffset + length <= data.count else { throw Failure.badTables }
            let bytes = [UInt8](data[textOffset ..< textOffset + length])
            added[bytes[0], default: []].append((bytes, id))
            textOffset += length
        }
        self.added = added.mapValues { $0.sorted { $0.bytes.count > $1.bytes.count } }
    }

    /// BOS, then the text's tokens.
    func encode(_ text: String) -> [Int32] {
        var ids: [Int32] = [Self.bos]
        let bytes = Array(text.utf8)
        var pieceStart = 0, index = 0
        while index < bytes.count {
            if let match = added[bytes[index]]?.first(where: { token in
                index + token.bytes.count <= bytes.count && bytes[index ..< index + token.bytes.count].elementsEqual(token.bytes)
            }) {
                ids += piece(bytes[pieceStart ..< index])
                ids.append(Int32(match.id))
                index += match.bytes.count
                pieceStart = index
            } else {
                index += 1
            }
        }
        ids += piece(bytes[pieceStart ..< bytes.count])
        return ids
    }

    private func piece(_ bytes: ArraySlice<UInt8>) -> [Int32] {
        guard !bytes.isEmpty else { return [] }
        let text = String(decoding: bytes, as: UTF8.self).replacingOccurrences(of: " ", with: "\u{2581}")
        var symbols: [UInt32] = []
        for scalar in text.unicodeScalars {
            if let id = characters[scalar.value] {
                symbols.append(id)
            } else {
                symbols += Array(String(scalar).utf8).map { byteTokens[Int($0)] }
            }
        }
        return merge(symbols).map { Int32($0) }
    }

    private func merge(_ start: [UInt32]) -> [UInt32] {
        var symbols = start
        guard symbols.count > 1 else { return symbols }
        // Each adjacent pair's merge, or nil.
        var candidates = (0 ..< symbols.count - 1).map { lookup(symbols[$0], symbols[$0 + 1]) }
        while true {
            var best = -1
            var bestRank = UInt32.max
            for (index, candidate) in candidates.enumerated() {
                if let candidate, candidate.rank < bestRank {
                    bestRank = candidate.rank
                    best = index
                }
            }
            guard best >= 0, let made = candidates[best]?.id else { break }
            symbols[best] = made
            symbols.remove(at: best + 1)
            candidates.remove(at: best)
            if best < candidates.count { candidates[best] = lookup(symbols[best], symbols[best + 1]) }
            if best > 0 { candidates[best - 1] = lookup(symbols[best - 1], symbols[best]) }
        }
        return symbols
    }

    private func lookup(_ left: UInt32, _ right: UInt32) -> (id: UInt32, rank: UInt32)? {
        let key = UInt64(left) << 32 | UInt64(right)
        var (low, high) = (0, pairs.count)
        while low < high {
            let middle = (low + high) / 2
            if pairs[middle] < key { low = middle + 1 } else { high = middle }
        }
        guard low < pairs.count, pairs[low] == key else { return nil }
        return (merged[low], ranks[low])
    }
}
