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
    private let engine = AVAudioEngine()
    private let pending = PendingSamples()
    private var pump: Task<Void, Never>?
    private var file: AVAudioFile?

    init() {
        url = FileManager.default.temporaryDirectory.appendingPathComponent("live_\(UUID().uuidString).wav")
    }

    /// - Parameter onText: the whole transcript so far, on the main actor.
    func start(language: String, onText: @escaping @MainActor (String) -> Void) async throws {
        try await SpeechRunner.shared.startLive(language: language)

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1,
                                         interleaved: false),
              let converter = AVAudioConverter(from: inputFormat, to: target)
        else { throw LiveError.unsupportedInput }
        let file = try AVAudioFile(forWriting: url,
                                   settings: [AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: 16_000.0,
                                              AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                                              AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false],
                                   commonFormat: .pcmFormatFloat32, interleaved: false)
        self.file = file
        let pending = self.pending

        input.installTap(onBus: 0, bufferSize: 4_096, format: inputFormat) { buffer, _ in
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * 16_000 / inputFormat.sampleRate) + 32
            guard let converted = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            var supplied = false
            converter.convert(to: converted, error: nil) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true
                status.pointee = .haveData
                return buffer
            }
            guard converted.frameLength > 0, let channel = converted.floatChannelData?[0] else { return }
            try? file.write(from: converted)
            pending.append(Array(UnsafeBufferPointer(start: channel, count: Int(converted.frameLength))))
        }
        engine.prepare()
        try engine.start()

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
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        pump?.cancel()
        await pump?.value
        let rest = pending.take()
        if !rest.isEmpty {
            GPUGate.waitUntilOpen()
            _ = await SpeechRunner.shared.feed(rest)
        }
        file = nil   // closes it, finishing the WAV header
        return (url, await SpeechRunner.shared.finishLive())
    }

    enum LiveError: LocalizedError {
        case unsupportedInput
        var errorDescription: String? { "This microphone's audio format can't be converted for live transcription." }
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
