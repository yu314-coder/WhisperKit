import AVFoundation
import Foundation

/// Audio waiting to be transcribed: recordings, imported files and links.
///
/// Each is put here before transcription starts, and only leaves when its
/// transcript is saved (it moves into SavedAudio) or the user deletes it. So
/// if iOS closes the app mid-transcription, or transcription fails, the audio
/// is still here on the next launch and the Library offers to transcribe it
/// again. It used to wait in a folder that every launch emptied (GitHub
/// issue #1: a 53-minute recording lost that way).
@MainActor @Observable final class PendingAudio {
    static let shared = PendingAudio()

    struct Item: Identifiable, Hashable {
        let url: URL
        let added: Date
        let bytes: Int
        let duration: Double?
        var id: URL { url }
        var name: String { url.deletingPathExtension().lastPathComponent }
    }

    private(set) var items: [Item] = []
    /// The file being transcribed right now, if any.
    var active: URL?
    /// Set by the Library's "Transcribe" button; the Transcribe tab picks it up.
    var retryRequest: URL?

    nonisolated static var folder: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Pending", isDirectory: true)
    }

    nonisolated static func contains(_ url: URL) -> Bool {
        url.standardizedFileURL.deletingLastPathComponent().path == folder.standardizedFileURL.path
    }

    /// A free name in the folder for `fileName`.
    nonisolated static func destination(for fileName: String) throws -> URL {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let name = (fileName as NSString).deletingPathExtension
        let ext = (fileName as NSString).pathExtension
        var url = folder.appendingPathComponent(fileName)
        var counter = 1
        while FileManager.default.fileExists(atPath: url.path) {
            url = folder.appendingPathComponent(ext.isEmpty ? "\(name)_\(counter)" : "\(name)_\(counter).\(ext)")
            counter += 1
        }
        return url
    }

    func refresh() {
        let keys: [URLResourceKey] = [.fileSizeKey, .creationDateKey]
        let files = (try? FileManager.default.contentsOfDirectory(
            at: Self.folder, includingPropertiesForKeys: keys, options: [.skipsHiddenFiles])) ?? []
        items = files.map { url in
            let values = try? url.resourceValues(forKeys: Set(keys))
            let duration = (try? AVAudioFile(forReading: url)).map { Double($0.length) / $0.fileFormat.sampleRate }
            return Item(url: url, added: values?.creationDate ?? .distantPast,
                        bytes: values?.fileSize ?? 0, duration: duration)
        }
        .sorted { $0.added > $1.added }
    }

    func delete(_ item: Item) {
        try? FileManager.default.removeItem(at: item.url)
        refresh()
    }

    /// At launch: brings back audio that earlier versions left behind when a
    /// transcription was interrupted, instead of deleting it.
    ///
    /// - Imports: where 1.3.2 and earlier kept a file while it was being
    ///   transcribed; anything still there never got its transcript.
    /// - tmp `recording_*.wav`: where earlier versions recorded. They never
    ///   removed these, so most are recordings that were transcribed fine;
    ///   those (same bytes as a file in SavedAudio) are just deleted.
    nonisolated static func recoverLeftovers(imports: URL) {
        let fm = FileManager.default
        let leftovers = ((try? fm.contentsOfDirectory(at: imports, includingPropertiesForKeys: nil,
                                                      options: [.skipsHiddenFiles])) ?? [])
            + ((try? fm.contentsOfDirectory(at: fm.temporaryDirectory, includingPropertiesForKeys: nil,
                                            options: [.skipsHiddenFiles])) ?? [])
                .filter { $0.lastPathComponent.hasPrefix("recording_") && $0.pathExtension == "wav" }
        guard !leftovers.isEmpty else { return }

        let saved = (try? fm.contentsOfDirectory(at: AudioFiles.folderURL, includingPropertiesForKeys: [.fileSizeKey],
                                                 options: [.skipsHiddenFiles])) ?? []
        let savedBySize = Dictionary(grouping: saved) { size(of: $0) }

        for file in leftovers {
            let bytes = size(of: file)
            // A WAV header with no audio, or nothing at all.
            if bytes <= 4_096 { try? fm.removeItem(at: file); continue }
            if savedBySize[bytes]?.contains(where: { sameTail(file, $0) }) == true {
                try? fm.removeItem(at: file)
                continue
            }
            var name = file.lastPathComponent
            if name.hasPrefix("recording_") {
                let date = (try? file.resourceValues(forKeys: [.creationDateKey]))?.creationDate ?? Date()
                name = recordingName(date)
            }
            if let destination = try? destination(for: name) {
                try? fm.moveItem(at: file, to: destination)
            }
        }
    }

    /// "Recording 2026-11-23 1405.wav"
    nonisolated static func recordingName(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HHmm"
        return "Recording \(f.string(from: date)).wav"
    }

    private nonisolated static func size(of url: URL) -> Int {
        (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
    }

    /// Whether two files of the same size end in the same 64 KB.
    private nonisolated static func sameTail(_ a: URL, _ b: URL) -> Bool {
        func tail(_ url: URL) -> Data? {
            guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
            defer { try? handle.close() }
            guard let end = try? handle.seekToEnd() else { return nil }
            try? handle.seek(toOffset: end > 65_536 ? end - 65_536 : 0)
            return try? handle.readToEnd()
        }
        guard let x = tail(a) else { return false }
        return x == tail(b)
    }
}
