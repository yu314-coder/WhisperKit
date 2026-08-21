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
