import Accelerate
import AVFoundation
import Foundation

/// How a recording sounds, measured from its audio: how much of it is
/// speech and where the pauses fall, how loud the voice and the background
/// are, the voice's pitch, and a spectrogram. Worked out on this device by
/// `SpeechAnalyzer`, and kept in Caches so a recording is read only once.
struct AudioAnalysis: Codable, Sendable {
    /// Raised when what is measured changes, so older saved results are redone.
    static let currentVersion = 1
    var version = AudioAnalysis.currentVersion

    let duration: Double
    /// One value per second: loudness (dBFS, averaged as energy), the share
    /// of the second that is speech (0...1), and the voice's pitch (Hz, the
    /// median of the voiced moments; 0 where there are none).
    let loudness: [Float]
    let speech: [Float]
    let pitch: [Float]

    let speakingTime: Double
    /// dBFS: the median of the speech, and of everything else.
    let speechLevel: Float
    let background: Float
    let peak: Float
    /// Moments the sound reached the most a recording can hold.
    let clipping: Int
    /// Silences of half a second or more between the first and last words.
    let pauses: [Pause]

    /// Hz; all 0 when there is less than a second of voice.
    let pitchMedian: Float
    let pitchLow: Float
    let pitchHigh: Float
    /// Standard deviation around the median, in semitones.
    let pitchSpread: Float

    let spectrogram: Spectrogram

    struct Pause: Codable, Sendable, Hashable {
        let start: Double
        let length: Double
    }

    /// Mel-spaced bands over time, 0...255, column by column with the lowest
    /// band first.
    struct Spectrogram: Codable, Sendable {
        let columns: Int
        let bands: Int
        let lowHz: Double
        let highHz: Double
        let levels: Data
    }
}

enum SpeechAnalyzer {
    struct Failure: LocalizedError {
        let errorDescription: String?
    }

    /// Reads the recording once, at 16 kHz mono, measuring as it goes, so
    /// even an hour holds only its measurements in memory. Blocking; call it
    /// off the main thread. `progress` gets whole percents.
    static func analyze(_ url: URL, progress: (Double) -> Void) throws -> AudioAnalysis {
        let file = try AVAudioFile(forReading: url)
        let input = file.processingFormat
        guard input.sampleRate > 0, file.length > 0 else {
            throw Failure(errorDescription: "The recording is empty.")
        }
        guard let output = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Measure.rate,
                                         channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: input, to: output)
        else { throw Failure(errorDescription: "This recording's format can't be read.") }
        converter.downmix = true

        let inCapacity: AVAudioFrameCount = 32_768
        let outCapacity = AVAudioFrameCount(Double(inCapacity) * Measure.rate / input.sampleRate) + 4_096
        guard let inBuffer = AVAudioPCMBuffer(pcmFormat: input, frameCapacity: inCapacity),
              let outBuffer = AVAudioPCMBuffer(pcmFormat: output, frameCapacity: outCapacity)
        else { throw Failure(errorDescription: "Not enough memory to read the recording.") }

        let measure = Measure(expectedSamples: Int(Double(file.length) * Measure.rate / input.sampleRate))
        var readAll = false
        var reported = -1
        while true {
            try Task.checkCancellation()
            outBuffer.frameLength = 0
            var error: NSError?
            let status = converter.convert(to: outBuffer, error: &error) { _, state in
                if !readAll {
                    inBuffer.frameLength = 0
                    try? file.read(into: inBuffer, frameCount: inCapacity)
                    if inBuffer.frameLength > 0 {
                        state.pointee = .haveData
                        return inBuffer
                    }
                    readAll = true
                }
                state.pointee = .endOfStream
                return nil
            }
            if status == .error {
                throw error ?? Failure(errorDescription: "The recording couldn't be decoded.")
            }
            if outBuffer.frameLength > 0, let channel = outBuffer.floatChannelData?[0] {
                measure.append(UnsafeBufferPointer(start: channel, count: Int(outBuffer.frameLength)))
            }
            let percent = min(99, Int(Double(file.framePosition) / Double(file.length) * 100))
            if percent != reported {
                reported = percent
                progress(Double(percent) / 100)
            }
            if status == .endOfStream || (readAll && outBuffer.frameLength == 0) { break }
        }
        try Task.checkCancellation()
        return measure.finish()
    }
}

/// Everything measured as the audio streams past at 16 kHz: a level every
/// 10 ms, a pitch every 20 ms, and spectra spread evenly over the
/// spectrogram's columns. Samples are dropped as soon as no measurement
/// needs them.
private final class Measure {
    static let rate: Double = 16_000
    private static let hop = 160              // 10 ms: level and speech
    private static let pitchHop = 320         // 20 ms
    private static let pitchWindow = 1_024    // 64 ms, read at 8 kHz
    private static let fftSize = 512          // 32 ms
    private static let bands = 64
    private static let lowHz = 60.0
    private static let highHz = 8_000.0

    private var samples: [Float] = []
    private var samplesStart = 0              // the absolute index of samples[0]
    private var received = 0

    private var levels: [Float] = []          // dBFS every 10 ms
    private var nextFrame = 0
    private var loudest: Float = 0
    private var clipping = 0
    private var lastClippedFrame = Int.min / 2

    private var pitches: [Float] = []         // Hz every 20 ms, 0 where unvoiced
    private var nextPitch = 0
    private let pitch = PitchDetector()

    private let expected: Int
    private let columns: Int
    private var column = 0
    private var columnStep = 0
    private var lastSpectrum = -1
    private var bandPower: [Float]
    private var bandFrames: [Int]
    private let spectrum = Spectrum(size: fftSize, bands: bands, lowHz: lowHz, highHz: highHz, rate: rate)

    init(expectedSamples: Int) {
        expected = max(expectedSamples, 1)
        // About eight columns a second, up to 1,600: an hour still fits a
        // screen as a whole, and a short clip isn't stretched thin.
        columns = max(1, min(1_600, Int(Double(expected) / Self.rate * 8)))
        bandPower = [Float](repeating: 0, count: columns * Self.bands)
        bandFrames = [Int](repeating: 0, count: columns)
        levels.reserveCapacity(expected / Self.hop + 1)
        pitches.reserveCapacity(expected / Self.pitchHop + 1)
    }

    func append(_ chunk: UnsafeBufferPointer<Float>) {
        samples.append(contentsOf: chunk)
        received += chunk.count
        samples.withUnsafeBufferPointer { buffer in
            let base = buffer.baseAddress!
            while nextFrame + Self.hop <= received {
                measureLevel(base + (nextFrame - samplesStart))
                nextFrame += Self.hop
            }
            while nextPitch + Self.pitchWindow <= received {
                pitches.append(pitch.frequency(base + (nextPitch - samplesStart)))
                nextPitch += Self.pitchHop
            }
            while let start = nextSpectrumStart(), start + Self.fftSize <= received {
                spectrum.accumulate(base + (start - samplesStart),
                                    into: &bandPower, at: column * Self.bands)
                bandFrames[column] += 1
                advanceSpectrum()
            }
        }
        // Keep only what a measurement still needs.
        let needed = min(nextFrame, nextPitch, nextSpectrumStart() ?? received) - samplesStart
        if needed > 65_536 {
            samples.removeFirst(needed)
            samplesStart += needed
        }
    }

    private func measureLevel(_ frame: UnsafePointer<Float>) {
        var meanSquare: Float = 0
        var peak: Float = 0
        vDSP_measqv(frame, 1, &meanSquare, vDSP_Length(Self.hop))
        vDSP_maxmgv(frame, 1, &peak, vDSP_Length(Self.hop))
        let index = levels.count
        levels.append(10 * log10(max(meanSquare, 1e-12)))
        loudest = max(loudest, peak)
        if peak >= 0.999 {
            // One moment, however many samples: clipped frames within half a
            // second of each other count once.
            if index - lastClippedFrame > 50 { clipping += 1 }
            lastClippedFrame = index
        }
    }

    // MARK: Spectrogram positions

    /// Up to three spectra per column, evenly spread through it.
    private func spectraIn(_ column: Int) -> (start: Int, span: Int, count: Int) {
        let start = column * expected / columns
        let span = (column + 1) * expected / columns - start
        return (start, span, max(1, min(3, span / Self.fftSize)))
    }

    private func nextSpectrumStart() -> Int? {
        guard column < columns else { return nil }
        let (start, span, count) = spectraIn(column)
        let position = start + (columnStep + 1) * span / (count + 1) - Self.fftSize / 2
        return max(position, lastSpectrum + 1, samplesStart)
    }

    private func advanceSpectrum() {
        lastSpectrum = nextSpectrumStart() ?? lastSpectrum
        columnStep += 1
        if columnStep >= spectraIn(column).count {
            column += 1
            columnStep = 0
        }
    }

    // MARK: Results

    func finish() -> AudioAnalysis {
        let duration = Double(received) / Self.rate
        let speech = Self.speechFrames(levels)
        let speechCount = speech.lazy.filter { $0 }.count

        var speechLevels: [Float] = []
        var otherLevels: [Float] = []
        for (level, isSpeech) in zip(levels, speech) where level > -100 {
            if isSpeech { speechLevels.append(level) } else { otherLevels.append(level) }
        }
        let speechLevel = Self.percentile(speechLevels, 0.5) ?? -100
        let background = Self.percentile(otherLevels, 0.5) ?? Self.percentile(levels, 0.05) ?? -100

        // A second at a time.
        let seconds = Int((duration).rounded(.up))
        var loudness = [Float](repeating: -100, count: seconds)
        var speechShare = [Float](repeating: 0, count: seconds)
        for second in 0 ..< seconds {
            let first = second * 100
            guard first < levels.count else { continue }
            let range = first ..< min(levels.count, first + 100)
            var energy: Float = 0
            var spoken = 0
            for index in range {
                energy += pow(10, levels[index] / 10)
                if speech[index] { spoken += 1 }
            }
            loudness[second] = 10 * log10(max(energy / Float(range.count), 1e-12))
            speechShare[second] = Float(spoken) / Float(range.count)
        }

        // Pitch: only where there is speech, then smoothed against octave
        // slips with the median of each voiced moment and its neighbours.
        var voiced = pitches.enumerated().map { index, frequency -> Float in
            let level = (index * Self.pitchHop + Self.pitchWindow / 2) / Self.hop
            return frequency > 0 && level < speech.count && speech[level] ? frequency : 0
        }
        voiced = voiced.indices.map { index in
            guard voiced[index] > 0 else { return 0 }
            let around = voiced[max(0, index - 2) ... min(voiced.count - 1, index + 2)].filter { $0 > 0 }
            return Self.percentile(Array(around), 0.5) ?? voiced[index]
        }
        var perSecond = [[Float]](repeating: [], count: seconds)
        for (index, frequency) in voiced.enumerated() where frequency > 0 {
            let second = (index * Self.pitchHop + Self.pitchWindow / 2) / Int(Self.rate)
            if second < seconds { perSecond[second].append(frequency) }
        }
        // A tenth of a second of voice at least, or one frame at the edge of
        // a word would give a silent second a pitch.
        let pitchBySecond = perSecond.map { $0.count >= 5 ? Self.percentile($0, 0.5) ?? 0 : 0 }
        let allVoiced = voiced.filter { $0 > 0 }
        let enoughVoice = allVoiced.count >= 50
        let median = enoughVoice ? Self.percentile(allVoiced, 0.5) ?? 0 : 0
        var spread: Float = 0
        if median > 0 {
            let squares = allVoiced.map { pow(12 * log2($0 / median), 2) }
            spread = sqrt(squares.reduce(0, +) / Float(squares.count))
        }

        return AudioAnalysis(
            duration: duration,
            loudness: loudness,
            speech: speechShare,
            pitch: pitchBySecond,
            speakingTime: Double(speechCount) * Double(Self.hop) / Self.rate,
            speechLevel: speechLevel,
            background: background,
            peak: 20 * log10(max(loudest, 1e-6)),
            clipping: clipping,
            pauses: Self.pauses(speech),
            pitchMedian: median,
            pitchLow: enoughVoice ? Self.percentile(allVoiced, 0.1) ?? 0 : 0,
            pitchHigh: enoughVoice ? Self.percentile(allVoiced, 0.9) ?? 0 : 0,
            pitchSpread: spread,
            spectrogram: spectrogram()
        )
    }

    /// Which 10 ms frames are speech: louder than the background by a margin
    /// that suits the recording (6–12 dB), with gaps under a quarter of a
    /// second bridged and blips under 50 ms ignored.
    private static func speechFrames(_ levels: [Float]) -> [Bool] {
        let audible = levels.filter { $0 > -100 }
        guard let floor = percentile(audible, 0.05), let loud = percentile(audible, 0.95) else {
            return [Bool](repeating: false, count: levels.count)
        }
        let threshold = floor + max(6, min(12, (loud - floor) * 0.35))
        var speech = levels.map { $0 > threshold }
        fill(&speech, runsOf: false, shorterThan: 25, between: true)
        fill(&speech, runsOf: true, shorterThan: 5, between: nil)
        return speech
    }

    /// Flips runs of `value` shorter than `length` frames; with `between`,
    /// only runs that have the other value on both sides.
    private static func fill(_ frames: inout [Bool], runsOf value: Bool, shorterThan length: Int, between: Bool?) {
        var index = 0
        while index < frames.count {
            guard frames[index] == value else { index += 1; continue }
            var end = index
            while end < frames.count, frames[end] == value { end += 1 }
            let enclosed = index > 0 && end < frames.count
            if end - index < length, between == nil || enclosed {
                for flip in index ..< end { frames[flip] = !value }
            }
            index = end
        }
    }

    private static func pauses(_ speech: [Bool]) -> [AudioAnalysis.Pause] {
        guard let first = speech.firstIndex(of: true), let last = speech.lastIndex(of: true) else { return [] }
        var pauses: [AudioAnalysis.Pause] = []
        var index = first
        while index < last {
            guard !speech[index] else { index += 1; continue }
            var end = index
            while end < last, !speech[end] { end += 1 }
            if end - index >= 50 {
                pauses.append(.init(start: Double(index) * Double(hop) / rate,
                                    length: Double(end - index) * Double(hop) / rate))
            }
            index = end
        }
        return pauses
    }

    private func spectrogram() -> AudioAnalysis.Spectrogram {
        var decibels = [Float](repeating: -.infinity, count: bandPower.count)
        for column in 0 ..< columns where bandFrames[column] > 0 {
            for band in 0 ..< Self.bands {
                let index = column * Self.bands + band
                decibels[index] = 10 * log10(bandPower[index] / Float(bandFrames[column]) + 1e-12)
            }
        }
        // 70 dB below the loudest cell is black.
        let top = decibels.max() ?? 0
        let range: Float = 70
        let levels = decibels.map { value -> UInt8 in
            guard value.isFinite else { return 0 }
            return UInt8(max(0, min(255, (value - (top - range)) / range * 255)))
        }
        return .init(columns: columns, bands: Self.bands, lowHz: Self.lowHz, highHz: Self.highHz, levels: Data(levels))
    }

    static func percentile(_ values: [Float], _ fraction: Double) -> Float? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        return sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * fraction))]
    }
}

/// The voice's fundamental frequency in one 64 ms window, by McLeod's
/// normalised square difference: read at 8 kHz, 65–450 Hz, and only where
/// the periodicity is clear (otherwise 0, as for breath, noise or silence).
private final class PitchDetector {
    private static let size = 512             // 64 ms at 8 kHz
    private static let rate: Float = 8_000
    private static let maxLag = 123           // 65 Hz
    private static let minLag = 18            // 444 Hz
    private static let terms = size - maxLag

    private var x = [Float](repeating: 0, count: size)
    private var squares = [Float](repeating: 0, count: size + 1)
    private var correlation = [Float](repeating: 0, count: maxLag + 1)
    private var nsdf = [Float](repeating: 0, count: maxLag + 1)

    func frequency(_ window: UnsafePointer<Float>) -> Float {
        // Pairs averaged: a rough low-pass that is enough for a voice's
        // fundamental, and halves the work.
        for index in 0 ..< Self.size {
            x[index] = 0.5 * (window[2 * index] + window[2 * index + 1])
        }
        var mean: Float = 0
        vDSP_meanv(x, 1, &mean, vDSP_Length(Self.size))
        var negative = -mean
        x.withUnsafeMutableBufferPointer { samples in
            vDSP_vsadd(samples.baseAddress!, 1, &negative, samples.baseAddress!, 1, vDSP_Length(Self.size))
        }
        var meanSquare: Float = 0
        vDSP_measqv(x, 1, &meanSquare, vDSP_Length(Self.size))
        guard 10 * log10(max(meanSquare, 1e-12)) > -55 else { return 0 }

        // r(τ) = Σ x[p]·x[p+τ] over the same number of terms for every lag.
        vDSP_conv(x, 1, x, 1, &correlation, 1, vDSP_Length(Self.maxLag + 1), vDSP_Length(Self.terms))
        squares[0] = 0
        for index in 0 ..< Self.size { squares[index + 1] = squares[index] + x[index] * x[index] }
        let head = squares[Self.terms]
        for lag in 0 ... Self.maxLag {
            let energy = head + squares[lag + Self.terms] - squares[lag]
            nsdf[lag] = energy > 0 ? 2 * correlation[lag] / energy : 0
        }

        // The highest peak of each positive lobe after the one at zero lag;
        // the first within 90% of the best of them is the period.
        var lag = 1
        while lag <= Self.maxLag, nsdf[lag] > 0 { lag += 1 }
        var peaks: [(lag: Int, value: Float)] = []
        while lag <= Self.maxLag {
            while lag <= Self.maxLag, nsdf[lag] <= 0 { lag += 1 }
            var best = (lag: -1, value: Float(0))
            while lag <= Self.maxLag, nsdf[lag] > 0 {
                if nsdf[lag] > best.value { best = (lag, nsdf[lag]) }
                lag += 1
            }
            if best.lag > 0, best.lag < Self.maxLag { peaks.append(best) }
        }
        guard let top = peaks.map(\.value).max(), top >= 0.7,
              let chosen = peaks.first(where: { $0.value >= 0.9 * top && $0.lag >= Self.minLag })
        else { return 0 }

        // Between samples, by the parabola through the peak and its neighbours.
        let (a, b, c) = (nsdf[chosen.lag - 1], nsdf[chosen.lag], nsdf[chosen.lag + 1])
        let curve = a - 2 * b + c
        let shift = curve != 0 ? 0.5 * (a - c) / curve : 0
        let frequency = Self.rate / (Float(chosen.lag) + max(-0.5, min(0.5, shift)))
        return (65 ... 450).contains(frequency) ? frequency : 0
    }
}

/// Power in mel-spaced bands for one Hann-windowed 512-sample frame.
private final class Spectrum {
    private let size: Int
    private let log2n: vDSP_Length
    private let setup: FFTSetup
    private let window: [Float]
    private var windowed: [Float]
    private var real: [Float]
    private var imaginary: [Float]
    private var power: [Float]
    /// Each band's FFT bins and their triangular weights.
    private let filters: [[(bin: Int, weight: Float)]]

    init(size: Int, bands: Int, lowHz: Double, highHz: Double, rate: Double) {
        self.size = size
        log2n = vDSP_Length(log2(Double(size)))
        setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
        window = vDSP.window(ofType: Float.self, usingSequence: .hanningDenormalized, count: size, isHalfWindow: false)
        windowed = [Float](repeating: 0, count: size)
        real = [Float](repeating: 0, count: size / 2)
        imaginary = [Float](repeating: 0, count: size / 2)
        power = [Float](repeating: 0, count: size / 2 + 1)

        func mel(_ hz: Double) -> Double { 2595 * log10(1 + hz / 700) }
        func hz(_ mel: Double) -> Double { 700 * (pow(10, mel / 2595) - 1) }
        let binHz = rate / Double(size)
        let edges = (0 ... bands + 1).map { hz(mel(lowHz) + (mel(highHz) - mel(lowHz)) * Double($0) / Double(bands + 1)) / binHz }
        filters = (0 ..< bands).map { band in
            let (left, center, right) = (edges[band], edges[band + 1], edges[band + 2])
            var taps: [(bin: Int, weight: Float)] = []
            for bin in Int(left.rounded(.up)) ... max(Int(left.rounded(.up)), Int(right.rounded(.down))) where bin <= size / 2 {
                let position = Double(bin)
                let weight = position <= center ? (position - left) / max(center - left, 1e-9)
                                                : (right - position) / max(right - center, 1e-9)
                if weight > 0 { taps.append((bin, Float(weight))) }
            }
            // A low band narrower than one bin takes the nearest bin whole.
            return taps.isEmpty ? [(bin: min(size / 2, Int(center.rounded())), weight: Float(1))] : taps
        }
    }

    deinit { vDSP_destroy_fftsetup(setup) }

    func accumulate(_ frame: UnsafePointer<Float>, into bands: inout [Float], at offset: Int) {
        vDSP_vmul(frame, 1, window, 1, &windowed, 1, vDSP_Length(size))
        let half = size / 2
        real.withUnsafeMutableBufferPointer { realPart in
            imaginary.withUnsafeMutableBufferPointer { imaginaryPart in
                var split = DSPSplitComplex(realp: realPart.baseAddress!, imagp: imaginaryPart.baseAddress!)
                windowed.withUnsafeBufferPointer { samples in
                    samples.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
                        vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
                    }
                }
                vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
            }
        }
        // Packed: the DC term in real[0], Nyquist's in imaginary[0].
        power[0] = real[0] * real[0]
        power[half] = imaginary[0] * imaginary[0]
        for bin in 1 ..< half { power[bin] = real[bin] * real[bin] + imaginary[bin] * imaginary[bin] }
        for (band, taps) in filters.enumerated() {
            var sum: Float = 0
            for tap in taps { sum += power[tap.bin] * tap.weight }
            bands[offset + band] += sum
        }
    }
}

/// Finished analyses, in Caches: one file per transcript, redone when the
/// recording changes or what is measured does. iOS may clear them; they are
/// simply measured again.
enum AudioAnalysisCache {
    private struct Entry: Codable {
        let size: Int
        let modified: Date
        let analysis: AudioAnalysis
    }

    private static var folder: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SpeechAnalysis", isDirectory: true)
    }

    private static func file(for id: UUID) -> URL { folder.appendingPathComponent("\(id.uuidString).json") }

    private static func stamp(_ audio: URL) -> (size: Int, modified: Date)? {
        guard let values = try? audio.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey]),
              let size = values.fileSize, let modified = values.contentModificationDate else { return nil }
        return (size, modified)
    }

    static func load(for id: UUID, audio: URL) -> AudioAnalysis? {
        guard let stamp = stamp(audio),
              let data = try? Data(contentsOf: file(for: id)),
              let entry = try? JSONDecoder().decode(Entry.self, from: data),
              entry.size == stamp.size, entry.modified == stamp.modified,
              entry.analysis.version == AudioAnalysis.currentVersion
        else { return nil }
        return entry.analysis
    }

    static func save(_ analysis: AudioAnalysis, for id: UUID, audio: URL) {
        guard let stamp = stamp(audio),
              let data = try? JSONEncoder().encode(Entry(size: stamp.size, modified: stamp.modified, analysis: analysis))
        else { return }
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try? data.write(to: file(for: id), options: .atomic)
    }

    static func remove(for id: UUID) {
        try? FileManager.default.removeItem(at: file(for: id))
    }
}
