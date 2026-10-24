import AVFoundation
import SwiftUI

/// How the microphone is recorded. Picked on the chip above the record
/// button, the same in both tabs, and switchable mid-recording.
///
/// Natural is the default: Whisper and Nemotron were trained on noisy audio
/// and cope with it, cleaning speech up often costs a recogniser more words
/// than the noise did, and voice processing — built for calls — takes a
/// loudspeaker's sound for echo, so a video, a lecture or a meeting played
/// nearby comes out quieter or patchy.
enum MicSound: String, CaseIterable, Identifiable {
    case natural
    case reduceNoise

    static let storageKey = "micSound"
    var id: String { rawValue }

    var title: String {
        switch self {
        case .natural:     return "Natural sound"
        case .reduceNoise: return "Reduce noise"
        }
    }

    var shortTitle: String {
        switch self {
        case .natural:     return "Natural"
        case .reduceNoise: return "Less noise"
        }
    }

    var detail: String {
        switch self {
        case .natural:     return "Everything the mic hears. Best for a room, a lecture or a video."
        case .reduceNoise: return "Turns background noise down. Best for one voice close to the phone."
        }
    }

    var icon: String {
        switch self {
        case .natural:     return "waveform"
        case .reduceNoise: return "person.wave.2"
        }
    }
}

/// The microphone through AVAudioEngine into a 16 kHz mono WAV, the format
/// the models read. With `.reduceNoise` the input runs through Apple's
/// voice processing — the noise suppression and level control calls use —
/// with other audio left at full volume.
final class MicRecorder {
    let url: URL
    private(set) var sound: MicSound
    /// Set when noise reduction couldn't be turned on and it records
    /// natural sound instead.
    private(set) var reductionUnavailable = false

    private let engine = AVAudioEngine()
    private var file: AVAudioFile?
    private let onSamples: (([Float]) -> Void)?
    private let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1,
                                       interleaved: false)!

    /// - Parameter onSamples: each piece of 16 kHz mono audio as it comes,
    ///   on the audio thread (Live feeds its model with these).
    init(url: URL, sound: MicSound, onSamples: (([Float]) -> Void)? = nil) throws {
        self.url = url
        self.sound = sound
        self.onSamples = onSamples
        file = try AVAudioFile(forWriting: url,
                               settings: [AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: 16_000.0,
                                          AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                                          AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false],
                               commonFormat: .pcmFormatFloat32, interleaved: false)
    }

    func start() throws {
        do {
            try run(sound)
        } catch where sound == .reduceNoise {
            // Not every route takes voice processing; record anyway.
            halt()
            try run(.natural)
            sound = .natural
            reductionUnavailable = true
        }
    }

    /// Switches mid-recording: the file carries on, with a moment's gap.
    func switchTo(_ new: MicSound) {
        guard new != sound, file != nil else { return }
        halt()
        do {
            try run(new)
            sound = new
            reductionUnavailable = false
        } catch {
            halt()
            try? run(sound)
            reductionUnavailable = new == .reduceNoise
        }
    }

    /// Stops the microphone and closes the file, finishing its header.
    func stop() {
        halt()
        if engine.inputNode.isVoiceProcessingEnabled { try? engine.inputNode.setVoiceProcessingEnabled(false) }
        file = nil
    }

    private func halt() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }

    private func run(_ sound: MicSound) throws {
        let input = engine.inputNode
        let processed = sound == .reduceNoise
        if input.isVoiceProcessingEnabled != processed {
            try input.setVoiceProcessingEnabled(processed)
        }
        if processed {
            // Calls duck other audio; a recording shouldn't.
            input.voiceProcessingOtherAudioDuckingConfiguration =
                AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: false, duckingLevel: .min)
            // Voice processing works on input and output together, so the
            // engine needs its output side in the graph.
            _ = engine.mainMixerNode
        }
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0,
              let converter = AVAudioConverter(from: format, to: target)
        else { throw MicError.unsupportedInput }
        // Voice processing can hand over more than one channel.
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

    enum MicError: LocalizedError {
        case unsupportedInput
        var errorDescription: String? { "This microphone's audio format can't be converted for transcription." }
    }
}

/// The chip above the record button: which sound to record, with a word on
/// what each is for. While noise is being reduced, it also opens iOS's own
/// mic modes, where Voice Isolation goes further.
struct MicSoundPicker: View {
    var isRecording: Bool
    @AppStorage(MicSound.storageKey) private var raw = MicSound.natural.rawValue

    private var sound: MicSound { MicSound(rawValue: raw) ?? .natural }

    var body: some View {
        Menu {
            ForEach(MicSound.allCases) { option in
                // Title, then the subtitle a menu shows under it.
                Button {
                    raw = option.rawValue
                } label: {
                    Text(option.title)
                    Text(option.detail)
                    Image(systemName: option == sound ? "checkmark" : option.icon)
                }
            }
            if isRecording && sound == .reduceNoise {
                Divider()
                Button {
                    AVCaptureDevice.showSystemUserInterface(.microphoneModes)
                } label: {
                    Text("Voice Isolation…")
                    Text("iOS's mic modes, for even less background.")
                    Image(systemName: "person.and.background.dotted")
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: sound.icon)
                    .font(.system(size: 11, weight: .semibold))
                Text(sound.shortTitle)
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .bold))
            }
            .foregroundColor(sound == .natural ? Studio.ink.opacity(0.7) : Studio.accent)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Capsule().fill(sound == .natural ? Studio.panel : Studio.accent.opacity(0.12)))
            .overlay(Capsule().strokeBorder(Studio.rule, lineWidth: 0.5))
            .contentShape(Capsule())
        }
        // Natural first, though the menu opens upward from the chip.
        .menuOrder(.fixed)
        .accessibilityLabel("Microphone, \(sound.title)")
    }
}
