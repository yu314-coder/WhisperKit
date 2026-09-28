import Foundation
import SwiftData

/// A clip produced by the Music tab, kept so it can be played again later.
///
/// Generated audio is written into the app's Documents directory and only
/// referenced here, the same arrangement transcripts use: SwiftData holds the
/// description, the filesystem holds the samples.
@Model
final class SavedMusic {
    var id: UUID
    var prompt: String
    var createdAt: Date
    var duration: Double          // seconds
    var modelName: String
    var audioFilePath: String?    // relative path under the app sandbox

    /// Downsampled peak envelope, normalised 0...1, measured once at save
    /// time so the library can draw each row without decoding the audio again.
    var waveform: [Float]?

    /// What else made it, so it can be read and made again. Optional: clips
    /// saved before 1.2 (35) have only the prompt.
    var lyrics: String?
    /// The language the lyrics were sung in.
    var language: String?
    var fullerArrangement: Bool?

    init(
        id: UUID = UUID(),
        prompt: String,
        createdAt: Date = Date(),
        duration: Double,
        modelName: String,
        audioFilePath: String? = nil,
        waveform: [Float]? = nil,
        lyrics: String? = nil,
        language: String? = nil,
        fullerArrangement: Bool? = nil
    ) {
        self.id = id
        self.prompt = prompt
        self.createdAt = createdAt
        self.duration = duration
        self.modelName = modelName
        self.audioFilePath = audioFilePath
        self.waveform = waveform
        self.lyrics = lyrics
        self.language = language
        self.fullerArrangement = fullerArrangement
    }

    /// The stored envelope resampled to `limit` buckets, or nil when this clip
    /// predates waveform capture.
    func envelope(limit: Int) -> [Float]? {
        guard let waveform, !waveform.isEmpty else { return nil }
        guard waveform.count > limit else { return waveform }
        return (0 ..< limit).map { index in
            let start = index * waveform.count / limit
            let end = max(start + 1, (index + 1) * waveform.count / limit)
            return waveform[start ..< end].max() ?? 0
        }
    }

    var audioURL: URL? {
        guard let audioFilePath else { return nil }
        return AudioFiles.urlForRelativePath(audioFilePath)
    }
}
