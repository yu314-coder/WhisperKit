import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

/// Finds the words to sing in a music prompt with Apple's on-device
/// language model (Apple Intelligence): `lyric is "…"`, `歌词是「…」`,
/// `she sings “…”`, lines of verse after the description — however they
/// were written. Rules alone missed inline lyrics like these, and the
/// song came back instrumental.
///
/// The model only says which words are lyrics. Every line it returns must
/// appear in the prompt, letter for letter, or the answer is not used:
/// asked about "a birthday song for my mom named Linda" it wrote lyrics of
/// its own, and once it turned 你 into "You". The prompt's own characters
/// are what get sung (`MusicRequest.locate`). It runs on the device, like
/// everything else here; where it isn't available the rules in
/// `MusicRequest` are used alone.
enum LyricsFinder {
    /// iOS 26 or later with Apple Intelligence turned on.
    static var isAvailable: Bool {
        #if canImport(FoundationModels)
        if #available(iOS 26.0, macOS 26.0, *) { return SystemLanguageModel.default.isAvailable }
        #endif
        return false
    }

    /// Whether a prompt could hold lyrics at all, so a short description
    /// like "Soft piano" isn't sent to the model: more than one line,
    /// quotation marks, a colon, a word about lyrics or singing, or more
    /// than a sentence of text.
    static func mightHoldLyrics(_ prompt: String) -> Bool {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return false }
        if text.contains(where: \.isNewline) { return true }
        if text.contains(where: { "\"“”„«»「」『』'‘’:：".contains($0) }) { return true }
        if text.range(of: #"(?i)lyric|words|sing|sung|goes|歌词|歌詞|唱"#, options: .regularExpression) != nil { return true }
        return text.split(separator: " ").count > 24 || text.count > 120
    }

    /// The lyric lines the model found — empty when it found none — or nil
    /// when it couldn't answer: not available, declined, or too slow.
    static func lyricLines(in prompt: String) async -> [String]? {
        #if canImport(FoundationModels)
        if #available(iOS 26.0, macOS 26.0, *) {
            guard SystemLanguageModel.default.isAvailable, prompt.count < 6000 else { return nil }
            return await withTaskGroup(of: [String]?.self) { group in
                group.addTask { await ask(prompt) }
                group.addTask {
                    try? await Task.sleep(for: .seconds(10))
                    return nil
                }
                let first = await group.next() ?? nil
                group.cancelAll()
                return first
            }
        }
        #endif
        return nil
    }

    #if canImport(FoundationModels)
    @available(iOS 26.0, macOS 26.0, *)
    @Generable
    struct Found {
        @Guide(description: "The words to be sung, copied character for character from the request, one sung line per line. Empty when the request has no words to be sung.")
        var lyrics: String
    }

    /// Tried on 19 prompts on the Mac's copy of the model (greedy, 0.4 to
    /// 1 s each): lyrics after "lyric is", "lyrics:", "sings", "that
    /// goes", "歌词是", "歌詞：", in straight, curly, single or corner
    /// quotes, bare quotations, and verse after a blank line or under
    /// section headings were all found; titles, topics ("lyrics about
    /// love"), styles and a list of instruments were left alone.
    @available(iOS 26.0, macOS 26.0, *)
    private static let instructions = """
        You find the lyrics in a request for a song. Lyrics are the exact words the person wants sung: often in quotation marks, after "lyrics", "lyric is", "sings", "歌词" or "歌詞", or written as lines of verse.
        Not lyrics: a topic ("a song about love", "lyrics about summer"), a title ("called Summer Rain"), a genre, an instrument, a mood, a length.
        Copy lyrics character for character in their original language and script. Never translate, correct, complete or add words. Leave out quotation marks and labels such as "lyrics:". If there are none, answer with empty lyrics.
        """

    @available(iOS 26.0, macOS 26.0, *)
    private static func ask(_ prompt: String) async -> [String]? {
        do {
            let session = LanguageModelSession(instructions: instructions)
            let answer = try await session.respond(to: prompt, generating: Found.self,
                                                   options: GenerationOptions(sampling: .greedy))
            return answer.content.lyrics.components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
        } catch {
            return nil
        }
    }
    #endif
}
