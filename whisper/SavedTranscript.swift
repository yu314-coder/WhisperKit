import Foundation
import SwiftData

@Model
final class SavedTranscript {
    var id: UUID
    var title: String
    var createdAt: Date
    var duration: Double           // seconds
    var languageCode: String?      // detected (or selected) ISO code, e.g. "ja"
    var modelName: String?
    var fullText: String
    var audioFilePath: String?     // relative path under app sandbox

    @Relationship(deleteRule: .cascade, inverse: \SavedSegment.transcript)
    var segments: [SavedSegment] = []

    init(
        id: UUID = UUID(),
        title: String,
        createdAt: Date = Date(),
        duration: Double,
        languageCode: String? = nil,
        modelName: String? = nil,
        fullText: String,
        audioFilePath: String? = nil
    ) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.duration = duration
        self.languageCode = languageCode
        self.modelName = modelName
        self.fullText = fullText
        self.audioFilePath = audioFilePath
    }
}

@Model
final class SavedSegment {
    var id: UUID
    var startTime: Double          // seconds
    var endTime: Double            // seconds
    var text: String
    var speaker: String?           // user-editable label
    var transcript: SavedTranscript?

    init(
        id: UUID = UUID(),
        startTime: Double,
        endTime: Double,
        text: String,
        speaker: String? = nil
    ) {
        self.id = id
        self.startTime = startTime
        self.endTime = endTime
        self.text = text
        self.speaker = speaker
    }
}

// MARK: - Segment text sanitising

enum WhisperText {
    /// Strips Whisper's special tokens from decoded segment text.
    ///
    /// `TranscriptionResult.text` arrives already cleaned, but
    /// `TranscriptionResult.segments[].text` does not — WhisperKit only drops
    /// them when `DecodingOptions.skipSpecialTokens` is set, and that defaults
    /// to false. Segments feed both the detail view and every export format, so
    /// unsanitised text meant SRT/VTT subtitles shipped with
    /// `<|startoftranscript|><|en|><|transcribe|><|0.00|>` baked in.
    ///
    /// Applied defensively even with `skipSpecialTokens: true`, so old records
    /// and any future decoder change are both covered.
    static func stripSpecialTokens(_ raw: String) -> String {
        guard raw.contains("<|") else {
            return raw.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let cleaned = raw.replacingOccurrences(
            of: "<\\|[^|]*\\|>",
            with: "",
            options: .regularExpression
        )
        // Collapse the runs of whitespace the removals leave behind.
        return cleaned
            .replacingOccurrences(of: "[ \\t]{2,}", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Raised when loading a model outlives its deadline. Distinct from a generic
/// failure so the retry can respond to it specifically.
struct ModelLoadTimeout: Error, LocalizedError {
    var errorDescription: String? {
        "The model took too long to load."
    }
}
