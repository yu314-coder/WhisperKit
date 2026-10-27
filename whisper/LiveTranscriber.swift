import AVFoundation
import Foundation
import WhisperKit

/// Records from the microphone and transcribes as it goes, with Nemotron
/// (SpeechRunner's live session).
///
/// The audio is converted to 16 kHz mono as it arrives and goes two ways:
/// into a WAV file, kept with the transcript for playback as a recording
/// made with the plain recorder is, and into a buffer the model reads about
/// three times a second. Off screen the model waits (GPUGate) while the
/// recording carries on, and catches up on return.
@MainActor
final class LiveTranscriber {
    let url: URL
    private let pending = PendingSamples()
    private var pump: Task<Void, Never>?
    private var mic: MicRecorder?

    init() {
        url = FileManager.default.temporaryDirectory.appendingPathComponent("live_\(UUID().uuidString).wav")
    }

    /// - Parameter onText: the whole transcript so far, on the main actor.
    func start(language: String, onText: @escaping @MainActor (LiveText) -> Void) async throws {
        try await SpeechRunner.shared.startLive(language: language)
        let pending = self.pending
        let mic = try MicRecorder(url: url) { pending.append($0) }
        try mic.start()
        self.mic = mic

        pump = Task.detached(priority: .userInitiated) {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 320_000_000)
                let samples = pending.take()
                guard !samples.isEmpty else { continue }
                GPUGate.waitUntilOpen()
                let text = await SpeechRunner.shared.feed(samples)
                await onText(text)
            }
        }
    }

    /// Stops the microphone and returns the recording and its transcript.
    func stop() async -> (url: URL, result: TranscriptionResult) {
        mic?.stop()   // closes the file, finishing the WAV header
        mic = nil
        pump?.cancel()
        await pump?.value
        let rest = pending.take()
        if !rest.isEmpty {
            GPUGate.waitUntilOpen()
            _ = await SpeechRunner.shared.feed(rest)
        }
        return (url, await SpeechRunner.shared.finishLive())
    }
}

/// Samples the audio thread hands over, under a lock.
private final class PendingSamples: @unchecked Sendable {
    private var samples: [Float] = []
    private let lock = NSLock()

    func append(_ more: [Float]) { lock.withLock { samples += more } }

    func take() -> [Float] {
        lock.withLock {
            defer { samples.removeAll(keepingCapacity: true) }
            return samples
        }
    }
}
