import Foundation
import AVFoundation

/// Converts any input audio (mp3, m4a, mp4, wav, caf, etc.) into a
/// 16 kHz mono 16-bit Linear PCM WAV that both WhisperKit and
/// AVAudioFile can read on iOS *and* on "Designed for iPad" / Mac Catalyst.
///
/// On Designed-for-iPad-on-Mac, the iOS audio session uses macOS's iOSSupport
/// AVFoundation, where `ExtAudioFile`'s AAC decoder is broken (returns
/// `kAudioFileUnsupportedDataFormatError` / 1685348671). AVAssetReader uses
/// a different decoder pipeline that works on both platforms.
enum AudioConverter {
    /// Returns either the original URL (if already a usable 16 kHz mono WAV)
    /// or a path to a freshly-written converted file in the temp directory.
    static func convertToWhisperReadableWAV(_ inputURL: URL) async throws -> URL {
        // Skip conversion if file is already a WAV — assume it's compatible.
        // If a WAV file still fails downstream we can tighten this check.
        if inputURL.pathExtension.lowercased() == "wav" {
            return inputURL
        }

        let asset = AVURLAsset(url: inputURL)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        guard let audioTrack = tracks.first else {
            throw NSError(
                domain: "AudioConverter",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No audio track found in file."]
            )
        }

        // Reader: pull samples out as 16 kHz mono 16-bit PCM
        // These AVFoundation objects are not Sendable, but every touch below
        // happens on the single serial queue driving requestMediaDataWhenReady,
        // so the capture is safe.
        nonisolated(unsafe) let assetReader = try AVAssetReader(asset: asset)
        let pcmSettings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]
        nonisolated(unsafe) let readerOutput = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: pcmSettings)
        guard assetReader.canAdd(readerOutput) else {
            throw NSError(domain: "AudioConverter", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Reader rejected output settings."])
        }
        assetReader.add(readerOutput)

        // Writer: write to .wav (WAVE) container
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("whisper_in_\(UUID().uuidString).wav")
        try? FileManager.default.removeItem(at: outputURL)
        nonisolated(unsafe) let assetWriter = try AVAssetWriter(outputURL: outputURL, fileType: .wav)
        nonisolated(unsafe) let writerInput = AVAssetWriterInput(mediaType: .audio, outputSettings: pcmSettings)
        writerInput.expectsMediaDataInRealTime = false
        guard assetWriter.canAdd(writerInput) else {
            throw NSError(domain: "AudioConverter", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Writer rejected input."])
        }
        assetWriter.add(writerInput)

        guard assetReader.startReading() else {
            throw assetReader.error ?? NSError(domain: "AudioConverter", code: 4,
                                               userInfo: [NSLocalizedDescriptionKey: "Reader could not start."])
        }
        guard assetWriter.startWriting() else {
            throw assetWriter.error ?? NSError(domain: "AudioConverter", code: 5,
                                               userInfo: [NSLocalizedDescriptionKey: "Writer could not start."])
        }
        assetWriter.startSession(atSourceTime: .zero)

        // Drain the reader into the writer.
        let queue = DispatchQueue(label: "AudioConverter.queue")

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            writerInput.requestMediaDataWhenReady(on: queue) {
                while writerInput.isReadyForMoreMediaData {
                    if assetReader.status != .reading {
                        writerInput.markAsFinished()
                        if let err = assetReader.error {
                            continuation.resume(throwing: err)
                        } else {
                            assetWriter.finishWriting {
                                if assetWriter.status == .completed {
                                    continuation.resume(returning: ())
                                } else {
                                    continuation.resume(throwing: assetWriter.error ?? NSError(
                                        domain: "AudioConverter", code: 6,
                                        userInfo: [NSLocalizedDescriptionKey: "Writer failed."]))
                                }
                            }
                        }
                        return
                    }

                    if let sampleBuffer = readerOutput.copyNextSampleBuffer() {
                        if !writerInput.append(sampleBuffer) {
                            // Append failed — surface the writer error.
                            assetReader.cancelReading()
                            writerInput.markAsFinished()
                            continuation.resume(throwing: assetWriter.error ?? NSError(
                                domain: "AudioConverter", code: 7,
                                userInfo: [NSLocalizedDescriptionKey: "Sample append failed."]))
                            return
                        }
                    } else {
                        // No more samples — done.
                        writerInput.markAsFinished()
                        assetWriter.finishWriting {
                            if assetWriter.status == .completed {
                                continuation.resume(returning: ())
                            } else {
                                continuation.resume(throwing: assetWriter.error ?? NSError(
                                    domain: "AudioConverter", code: 8,
                                    userInfo: [NSLocalizedDescriptionKey: "Writer finished in bad state."]))
                            }
                        }
                        return
                    }
                }
            }
        }

        return outputURL
    }
}

extension AudioConverter {
    /// Peak absolute sample amplitude (0...1) of a PCM file.
    ///
    /// Used to tell "the microphone captured nothing" apart from "the model
    /// found no speech in this audio" — two failures that look identical to the
    /// user but need completely different advice. One linear pass is negligible
    /// next to the transcription that follows it.
    static func peakAmplitude(of url: URL) -> Float? {
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let format = file.processingFormat
        let frameCapacity: AVAudioFrameCount = 16_384
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCapacity) else {
            return nil
        }

        var peak: Float = 0
        while true {
            buffer.frameLength = 0
            do {
                try file.read(into: buffer, frameCount: frameCapacity)
            } catch {
                return peak
            }
            let frames = Int(buffer.frameLength)
            if frames == 0 { break }

            guard let channels = buffer.floatChannelData else { break }
            for ch in 0..<Int(format.channelCount) {
                let samples = channels[ch]
                for i in 0..<frames {
                    let magnitude = abs(samples[i])
                    if magnitude > peak { peak = magnitude }
                }
            }
        }
        return peak
    }
}
