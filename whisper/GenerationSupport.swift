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

/// Stereo 16-bit WAV, written as the decoder produces it.
///
/// Streaming matters at length: six minutes of stereo float is 147 MB, and
/// building it whole before writing — then again as Swift arrays, then again
/// as `Data` — tripled that at the very end of a run that had already used
/// the most memory it would.
final class StreamingWAVWriter {
    private let handle: FileHandle
    private let expectedFrames: Int
    private var writtenFrames = 0

    init(url: URL, frames: Int, sampleRate: Int) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        expectedFrames = frames

        var header = Data()
        func string(_ s: String) { header.append(s.data(using: .ascii)!) }
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        string("RIFF"); u32(UInt32(36 + frames * 4)); string("WAVE")
        string("fmt "); u32(16); u16(1); u16(2)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 4)); u16(4); u16(16)
        string("data"); u32(UInt32(frames * 4))
        try handle.write(contentsOf: header)
    }

    /// - Parameter audio: (1, samples, 2) — already interleaved, which is
    ///   exactly WAV's sample order.
    func append(_ audio: MLXArray) throws {
        let pcm = (clip(audio, min: -1, max: 1) * 32767).asType(.int16)
        let samples = pcm.asArray(Int16.self)
        try samples.withUnsafeBytes { try handle.write(contentsOf: Data($0)) }
        writtenFrames += audio.dim(1)
    }

    func finish() throws {
        precondition(writtenFrames == expectedFrames, "wrote \(writtenFrames) of \(expectedFrames) frames")
        try handle.close()
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
