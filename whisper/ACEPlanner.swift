//  Ported from ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5), MIT.
//  Verified against the PyTorch reference; see ports/README.md.

import Foundation
import MLX
import MLXRandom

/// ACE-Step's planner: a 1.7B language model that writes the song out before
/// the diffusion transformer renders it.
///
/// Without it the transformer decides on its own when a song is over, and it
/// often decides early — at 81 seconds, four of four test runs went silent
/// between 5 and 24 seconds before the end. The official pipeline runs the
/// planner by default, and it is what fills the length: it must write exactly
/// five tokens per second of audio, and the transformer then follows that
/// layout frame by frame.
///
/// Two phases, as in `llm_inference.py`:
///   1. Reasoning. Tempo, a rewritten caption, duration, key, language and
///      time signature, in a fixed YAML block. Values the prompt already
///      states are written in rather than sampled.
///   2. Codes. Exactly `seconds × 5` music tokens, with classifier-free
///      guidance against an empty prompt.
struct ACEPlanner {
    struct Metadata: Equatable {
        var bpm: Int?
        var caption: String?
        var duration: Int
        var keyscale: String?
        var language: String?
        var timeSignature: Int?
    }

    struct Plan {
        let metadata: Metadata
        /// Phase 1 as the model wrote it, for inspection.
        let reasoning: String
        /// Codebook indices, 0..<64000, five per second.
        let codes: [Int32]
    }

    enum Stage { case reasoning, writing(Int, Int) }

    static let instruction = "Generate audio semantic tokens based on the given conditions:"
    static let codeBase = 151_669
    static let codebookSize = 64_000
    static let codesPerSecond = 5
    static let silenceCodes = [35847, 32855]

    let model: ACEQwen3
    let tokenizer: ACETokenizer
    var temperature: Float = 0.85
    var topP: Float = 0.9
    var guidance: Float = 2.0

    static func chatPrompt(user: String) -> String {
        "<|im_start|>system\n# Instruction\n\(instruction)\n\n<|im_end|>\n"
            + "<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
    }

    static func userPrompt(caption: String, lyrics: String) -> String {
        "# Caption\n\(caption)\n\n# Lyric\n\(lyrics)\n"
    }

    /// - Parameters:
    ///   - lyrics: the lyric text, or "[Instrumental]".
    ///   - known: fields the user stated; they are written in, not sampled.
    func plan(caption: String, lyrics: String, known: Metadata, seed: UInt64,
              isCancelled: () -> Bool, progress: (Stage) -> Void) throws -> Plan {
        var key = MLXRandom.key(seed)
        func nextKey() -> MLXArray {
            let (a, b) = MLXRandom.split(key: key)
            key = a
            return b
        }
        let user = Self.userPrompt(caption: caption, lyrics: lyrics)

        progress(.reasoning)
        let (metadata, reasoning) = try reason(user: user, known: known, nextKey: nextKey, isCancelled: isCancelled)
        let codes = try writeCodes(user: user, metadata: metadata, nextKey: nextKey,
                                   isCancelled: isCancelled, progress: progress)
        return Plan(metadata: metadata, reasoning: reasoning, codes: codes)
    }

    // MARK: - Phase 1: reasoning

    /// The official version drives this with a 2,300-line state machine. Its
    /// job reduces to three things, done here directly: write the field
    /// names in their fixed order, write in any value the user supplied, and
    /// keep sampled values to one line of plain text. Values that come out
    /// malformed are dropped rather than passed on, as the official parser
    /// does.
    private func reason(user: String, known: Metadata, nextKey: () -> MLXArray,
                        isCancelled: () -> Bool) throws -> (Metadata, String) {
        let prompt = tokenizer.encode(Self.chatPrompt(user: user), appendEndOfText: false)
        let cache = ACEKVCache(capacity: prompt.count + 512)
        var last = feed(prompt, cache: cache)
        var written: [Int32] = []
        let textRows = 0 ..< Int(ACETokenizer.endOfText)   // no special or music tokens

        func force(_ text: String) {
            let ids = tokenizer.encode(text, appendEndOfText: false)
            written += ids
            last = feed(ids, cache: cache)
        }
        /// Samples a value and returns it without its newline.
        ///
        /// A caption may run over several lines: the planner writes YAML,
        /// which folds long text at column 80 onto an indented continuation
        /// line. As the official state machine does, a newline ends the
        /// value only when what the model would write next is not indented.
        /// Stopping at the first newline — as this once did — cut captions
        /// off mid-phrase.
        func sampleValue(limit: Int, multiline: Bool = false) -> String {
            var ids: [Int32] = []
            func text() -> String { tokenizer.decode(ids) }
            for _ in 0 ..< limit {
                let logits = model.logits(last, rows: textRows)
                let id = Int32(sample(logits, key: nextKey()).item(Int32.self))
                ids.append(id)
                written.append(id)
                last = feed([id], cache: cache)
                // Only a newline in this token can end the value; the text
                // as a whole keeps its earlier ones.
                guard tokenizer.decode([id]).contains("\n") else { continue }
                if multiline {
                    let next = Int32(argMax(model.logits(last, rows: textRows), axis: -1).item(Int32.self))
                    if let first = tokenizer.decode([next]).first, first == " " || first == "\t" { continue }
                }
                break
            }
            if !text().hasSuffix("\n") && !text().contains("\n") { force("\n") }
            // Unfold: continuation lines join with a single space.
            return text().split(separator: "\n")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
                .joined(separator: " ")
        }
        func field(_ name: String, _ given: String?, limit: Int, multiline: Bool = false) -> String {
            if let given {
                force("\(name): \(given)\n")
                return given
            }
            force("\(name):")
            return sampleValue(limit: limit, multiline: multiline)
        }

        force("<think>\n")
        var result = Metadata(duration: known.duration)
        result.bpm = Int(field("bpm", known.bpm.map(String.init), limit: 6)).flatMap { (30 ... 300).contains($0) ? $0 : nil }
        try checkCancellation(isCancelled)
        let caption = field("caption", known.caption, limit: 512, multiline: true)
        result.caption = caption.isEmpty ? nil : caption
        _ = field("duration", String(known.duration), limit: 1)
        let keyscale = field("keyscale", known.keyscale, limit: 8)
        result.keyscale = Self.isValidKeyscale(keyscale) ? keyscale : nil
        let language = field("language", known.language, limit: 6)
        result.language = (Languages.aceStepCodes + ["unknown"]).contains(language) ? language : nil
        result.timeSignature = Int(field("timesignature", known.timeSignature.map(String.init), limit: 4))
            .flatMap { [2, 3, 4, 6].contains($0) ? $0 : nil }
        force("</think>")
        return (result, tokenizer.decode(written))
    }

    static func isValidKeyscale(_ text: String) -> Bool {
        let parts = text.split(separator: " ")
        guard parts.count == 2, ["major", "minor"].contains(parts[1]),
              let note = parts[0].first, "ABCDEFG".contains(note) else { return false }
        let accidental = parts[0].dropFirst()
        return ["", "#", "b", "♯", "♭"].contains(String(accidental))
    }

    // MARK: - Phase 2: music tokens

    private func writeCodes(user: String, metadata: Metadata, nextKey: () -> MLXArray,
                            isCancelled: () -> Bool, progress: (Stage) -> Void) throws -> [Int32] {
        let target = metadata.duration * Self.codesPerSecond
        let conditional = tokenizer.encode(
            Self.chatPrompt(user: user) + "<think>\n\(ACEPlannerYAML.cot(metadata))\n</think>\n\n",
            appendEndOfText: false)
        // Guidance compares against the prompt the model saw when training
        // dropped its conditions: the literal "NO USER INPUT", empty reasoning.
        let unconditional = tokenizer.encode(
            Self.chatPrompt(user: "NO USER INPUT") + "<think>\n\n</think>\n\n", appendEndOfText: false)

        // Both prompts run as one batch of two: a pass costs the same for two
        // rows as for one (measured, 48 vs 47 ms on an M4), because reading
        // the weights is the work. The shorter prompt is padded on the left
        // and the padding hidden from real positions; rotary attention sees
        // only relative position, so shifting that row changes nothing.
        let padding = conditional.count - unconditional.count
        precondition(padding >= 0, "the unconditional prompt is the shorter one")
        let batch = conditional + [Int32](repeating: ACETokenizer.endOfText, count: padding) + unconditional
        let cache = ACEKVCache(capacity: conditional.count + target)
        var last = feedPair(MLXArray(batch, [2, conditional.count]), cache: cache,
                            mask: Self.pairMask(queries: conditional.count, offset: 0, padding: padding))
        let codeRows = Self.codeBase ..< Self.codeBase + Self.codebookSize

        // The planner learned from recordings that often end in silence, and
        // it plans that too — the last 5 to 15 seconds of a requested length
        // written as nothing. The silence token (35847; 32855 once in a
        // minute of it, measured by tokenizing the silence latent) is kept
        // out of everything but the final second, so a song asked to last
        // 81 seconds has music for 81 seconds and still gets to end.
        var silenceMask = [Float](repeating: 0, count: Self.codebookSize)
        for code in Self.silenceCodes { silenceMask[code] = -Float.infinity }
        let noSilence = MLXArray(silenceMask, [1, Self.codebookSize])
        let endingStart = max(0, target - Self.codesPerSecond)

        var codes: [Int32] = []
        codes.reserveCapacity(target)
        for index in 0 ..< target {
            // Only music tokens are scored, and the end token never is: the
            // count is fixed, which is the whole point.
            let scores = model.logits(last, rows: codeRows)        // (2, codes)
            let (c, u) = (scores[0 ..< 1], scores[1 ..< 2])
            var guided = u + guidance * (c - u)
            if index < endingStart { guided = guided + noSilence }
            let code = sample(guided, key: nextKey()).item(Int32.self)
            codes.append(code)
            let token = Int32(Self.codeBase) + code
            last = feedPair(MLXArray([token, token], [2, 1]), cache: cache,
                            mask: Self.pairMask(queries: 1, offset: cache.offset, padding: padding))
            if index % 25 == 0 {
                try checkCancellation(isCancelled)
                progress(.writing(index, target))
            }
        }
        progress(.writing(target, target))
        return codes
    }

    // MARK: - Pieces

    /// Runs tokens through the model and returns the last position's hidden
    /// state, (1, hidden).
    private func feed(_ ids: [Int32], cache: ACEKVCache) -> MLXArray {
        let hidden = model(inputIDs: MLXArray(ids, [1, ids.count]), cache: cache)
        let last = hidden[0..., (ids.count - 1) ..< ids.count, 0...].reshaped(1, -1)
        eval(last)
        return last
    }

    /// The batched guidance pair's last hidden states, (2, hidden).
    private func feedPair(_ ids: MLXArray, cache: ACEKVCache, mask: MLXArray) -> MLXArray {
        let hidden = model(inputIDs: ids, cache: cache, mask: mask)
        let last = hidden[0..., (ids.dim(1) - 1) ..< ids.dim(1), 0...].reshaped(2, -1)
        eval(last)
        return last
    }

    /// Causal for both rows; in the second, the first `padding` positions
    /// are hidden from every real position. Padding positions may still see
    /// themselves, so no row is ever fully masked — a fully masked row would
    /// be NaN, and NaN in a key poisons everything that attends to it.
    static func pairMask(queries: Int, offset: Int, padding: Int) -> MLXArray {
        let width = offset + queries
        var values = [Float](repeating: 0, count: 2 * queries * width)
        for row in 0 ..< 2 {
            for query in 0 ..< queries {
                let position = offset + query
                for key in 0 ..< width {
                    let future = key > position
                    let hiddenPad = row == 1 && key < padding && position >= padding
                    if future || hiddenPad { values[(row * queries + query) * width + key] = -1e9 }
                }
            }
        }
        return MLXArray(values, [2, 1, queries, width])
    }

    /// Top-p on the untempered distribution, then temperature — mlx-lm's
    /// order, which is what the official MLX path uses. Returns the index
    /// within `logits`.
    private func sample(_ logits: MLXArray, key: MLXArray) -> MLXArray {
        let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
        let order = argSort(logprobs, axis: -1)                      // ascending
        let sorted = takeAlong(logprobs, order, axis: -1)
        let cumulative = cumsum(exp(sorted), axis: -1)
        let kept = MLX.where(cumulative .> (1 - topP), sorted, MLXArray(-Float.infinity))
        let choice = MLXRandom.categorical(kept * (1 / temperature), key: key)
        return takeAlong(order, choice.reshaped(1, 1), axis: -1).reshaped([])
    }

    private func checkCancellation(_ isCancelled: () -> Bool) throws {
        if isCancelled() { throw CancellationError() }
    }
}

/// The reasoning block as PyYAML's `dump(sort_keys=True, allow_unicode=True)`
/// writes it, because that is the text the planner was trained on — down to
/// long captions folding onto a second line at column 80 with a two-space
/// indent.
enum ACEPlannerYAML {
    static func cot(_ metadata: ACEPlanner.Metadata) -> String {
        var lines: [String] = []
        if let bpm = metadata.bpm { lines.append("bpm: \(bpm)") }
        if let caption = metadata.caption { lines.append(scalar(key: "caption", caption)) }
        lines.append("duration: \(metadata.duration)")
        if let keyscale = metadata.keyscale { lines.append(scalar(key: "keyscale", keyscale)) }
        if let language = metadata.language { lines.append(scalar(key: "language", language)) }
        if let timeSignature = metadata.timeSignature { lines.append("timesignature: \(timeSignature)") }
        return lines.joined(separator: "\n")
    }

    static let width = 80

    /// One `key: value` entry, plain where PyYAML would write it plain and
    /// single-quoted otherwise, folded at spaces past column 80.
    static func scalar(key: String, _ value: String) -> String {
        let prefix = "\(key):"
        if needsQuotes(value) {
            let escaped = value.replacingOccurrences(of: "'", with: "''")
            return prefix + fold(" '" + escaped + "'", column: prefix.count, quoted: true)
        }
        return prefix + fold(" " + value, column: prefix.count, quoted: false)
    }

    /// Writes words, and at a single space past the width starts a new
    /// indented line instead — PyYAML's `write_plain` / `write_single_quoted`.
    private static func fold(_ text: String, column start: Int, quoted: Bool) -> String {
        let characters = Array(text)
        var output = ""
        var column = start
        var index = 0
        while index < characters.count {
            let character = characters[index]
            let isSingleSpace = character == " "
                && index > 0 && characters[index - 1] != " "
                && index + 1 < characters.count && characters[index + 1] != " "
            // The first space separates the key; quoted scalars never fold at
            // their very first or last character.
            let foldable = isSingleSpace && index > (quoted ? 2 : 0)
                && !(quoted && index + 1 == characters.count - 1)
            if foldable && column > width {
                output += "\n  "
                column = 2
            } else {
                output.append(character)
                column += character.unicodeScalars.count
            }
            index += 1
        }
        return output
    }

    /// PyYAML quotes what would otherwise read back as something else, or
    /// break the block: YAML 1.1 booleans and nulls (Norwegian's code "no"
    /// among them), numbers, leading indicators, and ": " or " #" inside.
    private static func needsQuotes(_ value: String) -> Bool {
        guard let first = value.first, let lastCharacter = value.last else { return true }
        let reserved: Set<String> = ["y", "yes", "no", "n", "true", "false", "on", "off", "null", "~"]
        if reserved.contains(value.lowercased()) { return true }
        if Double(value) != nil { return true }
        if first == " " || lastCharacter == " " { return true }
        if ",[]{}#&*!|>'\"%@`".contains(first) { return true }
        if "-?:".contains(first), value.count == 1 || value.dropFirst().first == " " { return true }
        if value.hasPrefix("---") || value.hasPrefix("...") { return true }
        if value.contains(": ") || value.contains(" #") || lastCharacter == ":" { return true }
        return false
    }
}
