import AVFoundation

/// The microphone through AVAudioEngine into a 16 kHz mono WAV, the format
/// the models read, as the microphone hears it.
///
/// No noise reduction, on purpose. Measured on FLEURS with noise mixed in
/// (Whisper Turbo, 200 clips), every denoiser tried — Apple's Voice
/// Isolation, and RMT low-rank cleaning — made the transcripts worse than
/// the noisy original in nearly every test: Whisper was trained on noisy
/// audio and copes with it, while cleaning removes or rebuilds speech it
/// can't get back.
final class MicRecorder {
    let url: URL

    private let engine = AVAudioEngine()
    private var file: AVAudioFile?
    private let onSamples: (([Float]) -> Void)?
    private let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1,
                                       interleaved: false)!

    /// - Parameter onSamples: each piece of 16 kHz mono audio as it comes,
    ///   on the audio thread (Live feeds its model with these).
    init(url: URL, onSamples: (([Float]) -> Void)? = nil) throws {
        self.url = url
        self.onSamples = onSamples
        file = try AVAudioFile(forWriting: url,
                               settings: [AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: 16_000.0,
                                          AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                                          AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false],
                               commonFormat: .pcmFormatFloat32, interleaved: false)
    }

    func start() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0,
              let converter = AVAudioConverter(from: format, to: target)
        else { throw MicError.unsupportedInput }
        converter.downmix = true
        let target = self.target
        input.installTap(onBus: 0, bufferSize: 4_096, format: format) { [weak self] buffer, _ in
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * 16_000 / format.sampleRate) + 32
            guard let self, let converted = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            var supplied = false
            converter.convert(to: converted, error: nil) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true
                status.pointee = .haveData
                return buffer
            }
            guard converted.frameLength > 0, let channel = converted.floatChannelData?[0] else { return }
            try? self.file?.write(from: converted)
            self.onSamples?(Array(UnsafeBufferPointer(start: channel, count: Int(converted.frameLength))))
        }
        engine.prepare()
        try engine.start()
    }

    /// Stops the microphone and closes the file, finishing its header.
    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        file = nil
    }

    enum MicError: LocalizedError {
        case unsupportedInput
        var errorDescription: String? { "This microphone's audio format can't be converted for transcription." }
    }
}
