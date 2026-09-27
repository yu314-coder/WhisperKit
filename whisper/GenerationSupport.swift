import Foundation
import Darwin
import MLX

/// Pieces shared by the music pipelines, which all run the same way: one
/// model stage at a time, each handing its memory back before the next.

/// The app's physical footprint — the number iOS compares against its limit
/// when deciding what to kill.
enum MemoryFootprint {
    static var current: Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }
}

enum StageMemory {
    /// Hands a finished stage's memory back before the next one starts.
    ///
    /// Two things delay it. Freed arrays go to MLX's buffer cache rather than
    /// the system, so the cache is cleared. And the system takes freed GPU
    /// memory back asynchronously — measured at a few hundred milliseconds,
    /// during which it still counts against the app. Loading the next stage
    /// inside that window stacks both stages' memory, which is how the
    /// transition, not either stage, became the peak. So this waits, briefly,
    /// for the footprint to stop falling.
    static func release() {
        MLX.Memory.clearCache()
        var previous = MemoryFootprint.current
        for _ in 0 ..< 20 {
            usleep(50_000)
            let now = MemoryFootprint.current
            if previous - now < 8 * 1_048_576 { break }
            previous = now
        }
    }

    /// Runs `body` with MLX keeping no freed buffers for reuse. Measured on
    /// ACE-Step, a zero cache cost nothing in speed and 240 MB less at peak
    /// than a 256 MB one.
    static func withoutCache<T>(_ body: () throws -> T) rethrows -> T {
        let previous = MLX.Memory.cacheLimit
        MLX.Memory.cacheLimit = 0
        defer {
            MLX.Memory.cacheLimit = previous
            release()
        }
        return try body()
    }
}

/// Stereo 16-bit WAV, written as the decoder produces it, peak-normalised.
///
/// Streaming matters at length: six minutes of stereo float is 147 MB, and
/// building it whole before writing — then again as Swift arrays, then again
/// as `Data` — tripled that at the very end of a run that had already used
/// the most memory it would.
///
/// The decoder's output runs past full scale on loud passages, and it was
/// written clipped: every song peaked at exactly 0 dBFS, and a 1:52 one had
/// 5,843 clipped samples, heard as crackle. Upstream scales every song so
/// its peak sits at -1 dBFS (`normalize_audio`, on by default), which also
/// brings quiet ones up. The peak is only known at the end, so samples go to
/// a scratch file as float and are scaled into the WAV on `finish`.
final class StreamingWAVWriter {
    static let peakDecibels: Float = -1

    private let url: URL
    private let scratchURL: URL
    private let scratch: FileHandle
    private let expectedFrames: Int
    private let sampleRate: Int
    private var writtenFrames = 0
    private var peak: Float = 0

    init(url: URL, frames: Int, sampleRate: Int) throws {
        self.url = url
        self.expectedFrames = frames
        self.sampleRate = sampleRate
        scratchURL = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.deletingPathExtension().lastPathComponent).f32")
        FileManager.default.createFile(atPath: scratchURL.path, contents: nil)
        scratch = try FileHandle(forWritingTo: scratchURL)
    }

    deinit {
        try? scratch.close()
        try? FileManager.default.removeItem(at: scratchURL)
    }

    /// - Parameter audio: (1, samples, 2) — already interleaved, which is
    ///   exactly WAV's sample order.
    func append(_ audio: MLXArray) throws {
        let samples = audio.asType(.float32)
        peak = max(peak, abs(samples).max().item(Float.self))
        try samples.asArray(Float.self).withUnsafeBytes { try scratch.write(contentsOf: Data($0)) }
        writtenFrames += audio.dim(1)
    }

    func finish() throws {
        precondition(writtenFrames == expectedFrames, "wrote \(writtenFrames) of \(expectedFrames) frames")
        try scratch.close()
        // Silence stays silence rather than being blown up to full scale.
        let gain = peak > 1e-6 ? pow(10, Self.peakDecibels / 20) / peak : 1

        FileManager.default.createFile(atPath: url.path, contents: nil)
        let output = try FileHandle(forWritingTo: url)
        defer { try? output.close() }
        var header = Data()
        func string(_ s: String) { header.append(s.data(using: .ascii)!) }
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        let frames = writtenFrames
        string("RIFF"); u32(UInt32(36 + frames * 4)); string("WAVE")
        string("fmt "); u32(16); u16(1); u16(2)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 4)); u16(4); u16(16)
        string("data"); u32(UInt32(frames * 4))
        try output.write(contentsOf: header)

        let input = try FileHandle(forReadingFrom: scratchURL)
        defer { try? input.close() }
        while let block = try input.read(upToCount: 1 << 22), !block.isEmpty {
            let pcm: [Int16] = block.withUnsafeBytes { raw in
                raw.bindMemory(to: Float.self).map { sample in
                    Int16(max(-1, min(1, sample * gain)) * 32767)
                }
            }
            try pcm.withUnsafeBytes { try output.write(contentsOf: Data($0)) }
        }
        try? FileManager.default.removeItem(at: scratchURL)
    }
}

/// A cancellation request that crosses into a detached task, which does not
/// inherit its parent's cancellation.
final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isCancelled: Bool { lock.withLock { cancelled } }
    func cancel() { lock.withLock { cancelled = true } }
}

/// Whether the previous run of the app was closed while generating music.
///
/// Set while a generation runs and cleared when it ends, so finding it set
/// at launch means the app was closed mid-run. Read once, at launch, before
/// anything can start a new run.
enum MusicRunMarker {
    private static let key = "musicGenerationInFlight"

    static let wasInterrupted: Bool = {
        let value = UserDefaults.standard.bool(forKey: key)
        UserDefaults.standard.set(false, forKey: key)
        return value
    }()

    static func begin() {
        _ = wasInterrupted          // snapshot before overwriting
        UserDefaults.standard.set(true, forKey: key)
    }

    static func end() { UserDefaults.standard.set(false, forKey: key) }
}
