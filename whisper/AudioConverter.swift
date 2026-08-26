import Foundation
import AVFoundation
import Accelerate

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

// MARK: - Peak envelope

/// A downsampled amplitude envelope: what a waveform view draws.
struct PeakEnvelope {
    /// One value per bucket, scaled so the loudest bucket is 1.0. Normalised
    /// because a quiet recording still needs a legible shape; `peak` carries
    /// the true level for anything that needs the absolute figure.
    let buckets: [Float]
    /// Loudest absolute sample in the file, 0...1.
    let peak: Float
    let duration: TimeInterval
}

extension AudioConverter {
    /// Reduces a PCM file to `bucketCount` peak amplitudes in a single pass.
    ///
    /// Peak rather than RMS: peak is what audio editors draw, and it keeps
    /// transients visible at any zoom. Within a bucket the maximum magnitude is
    /// taken with `vDSP_maxmgv` over each channel's slice — a 40-minute
    /// recording is ~38M frames at 16 kHz, and a per-frame Swift loop over that
    /// is slow enough to be felt.
    ///
    /// Synchronous and I/O-bound; call it off the main actor.
    static func peakEnvelope(of url: URL, bucketCount: Int = 120) -> PeakEnvelope? {
        guard bucketCount > 0, let file = try? AVAudioFile(forReading: url) else { return nil }

        let totalFrames = file.length
        let sampleRate = file.processingFormat.sampleRate
        let duration = sampleRate > 0 ? Double(totalFrames) / sampleRate : 0
        guard totalFrames > 0 else {
            return PeakEnvelope(buckets: [Float](repeating: 0, count: bucketCount), peak: 0, duration: 0)
        }

        let format = file.processingFormat
        let channels = Int(format.channelCount)
        let frameCapacity: AVAudioFrameCount = 16_384
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCapacity) else { return nil }

        var envelope = [Float](repeating: 0, count: bucketCount)
        var readCursor: Int64 = 0

        while readCursor < totalFrames {
            buffer.frameLength = 0
            do {
                try file.read(into: buffer, frameCount: frameCapacity)
            } catch {
                break
            }
            let frames = Int(buffer.frameLength)
            if frames == 0 { break }
            guard let data = buffer.floatChannelData else { break }

            let chunkStart = readCursor
            let chunkEnd = readCursor + Int64(frames)

            // Only the buckets this chunk actually overlaps.
            let firstBucket = Int(chunkStart * Int64(bucketCount) / totalFrames)
            let lastBucket = min(Int((chunkEnd - 1) * Int64(bucketCount) / totalFrames), bucketCount - 1)

            for bucket in firstBucket...max(firstBucket, lastBucket) {
                let bucketStart = Int64(bucket) * totalFrames / Int64(bucketCount)
                let bucketEnd = Int64(bucket + 1) * totalFrames / Int64(bucketCount)

                let from = max(bucketStart, chunkStart) - chunkStart
                let to = min(bucketEnd, chunkEnd) - chunkStart
                guard to > from else { continue }

                let offset = Int(from)
                let count = vDSP_Length(to - from)
                var loudest: Float = 0
                for ch in 0..<channels {
                    var m: Float = 0
                    vDSP_maxmgv(data[ch] + offset, 1, &m, count)
                    if m > loudest { loudest = m }
                }
                if loudest > envelope[bucket] { envelope[bucket] = loudest }
            }

            readCursor = chunkEnd
        }

        let peak = envelope.max() ?? 0
        guard peak > 0 else {
            return PeakEnvelope(buckets: envelope, peak: 0, duration: duration)
        }
        // vDSP_vsdiv scales the whole array in one call.
        var normalised = [Float](repeating: 0, count: bucketCount)
        var divisor = peak
        vDSP_vsdiv(envelope, 1, &divisor, &normalised, 1, vDSP_Length(bucketCount))

        return PeakEnvelope(buckets: normalised, peak: peak, duration: duration)
    }
}
