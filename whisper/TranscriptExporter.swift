import Foundation

enum ExportFormat: String, CaseIterable, Identifiable {
    case txt, srt, vtt, markdown, json
    var id: String { rawValue }
    var fileExtension: String { self == .markdown ? "md" : rawValue }
    var displayName: String {
        switch self {
        case .txt: return "Plain Text (.txt)"
        case .srt: return "SubRip Subtitles (.srt)"
        case .vtt: return "WebVTT (.vtt)"
        case .markdown: return "Markdown (.md)"
        case .json: return "JSON (.json)"
        }
    }
}

enum TranscriptExporter {
    /// - Parameter includeSpeakers: when false, speaker labels are omitted from
    ///   every format. Exposed so the export screen can preview both.
    static func export(
        _ transcript: SavedTranscript,
        as format: ExportFormat,
        includeSpeakers: Bool = true
    ) -> String {
        switch format {
        case .txt: return exportTXT(transcript, includeSpeakers)
        case .srt: return exportSRT(transcript, includeSpeakers)
        case .vtt: return exportVTT(transcript, includeSpeakers)
        case .markdown: return exportMarkdown(transcript, includeSpeakers)
        case .json: return exportJSON(transcript, includeSpeakers)
        }
    }

    static func write(
        _ transcript: SavedTranscript,
        as format: ExportFormat,
        includeSpeakers: Bool = true
    ) throws -> URL {
        let body = export(transcript, as: format, includeSpeakers: includeSpeakers)
        let safeTitle = transcript.title
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
        let filename = "\(safeTitle).\(format.fileExtension)"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        try body.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    // MARK: - Formatters

    private static func exportTXT(_ t: SavedTranscript, _ includeSpeakers: Bool = true) -> String {
        let sorted = t.segments.sorted { $0.startTime < $1.startTime }
        if sorted.isEmpty { return t.fullText }
        return sorted.map { seg in
            if includeSpeakers, let speaker = seg.speaker, !speaker.isEmpty {
                return "\(speaker): \(seg.text)"
            }
            return seg.text
        }.joined(separator: "\n")
    }

    private static func exportMarkdown(_ t: SavedTranscript, _ includeSpeakers: Bool = true) -> String {
        let df = DateFormatter()
        df.dateStyle = .medium
        df.timeStyle = .short

        var out = "# \(t.title)\n\n"
        out += "**Date:** \(df.string(from: t.createdAt))  \n"
        out += "**Duration:** \(formatHMS(t.duration))  \n"
        if let lang = t.languageCode { out += "**Language:** \(lang.uppercased())  \n" }
        if let model = t.modelName { out += "**Model:** \(model)  \n" }
        out += "\n---\n\n"

        let sorted = t.segments.sorted { $0.startTime < $1.startTime }
        for seg in sorted {
            let stamp = "`[\(formatTimestamp(seg.startTime))]`"
            if includeSpeakers, let speaker = seg.speaker, !speaker.isEmpty {
                out += "\(stamp) **\(speaker):** \(seg.text)\n\n"
            } else {
                out += "\(stamp) \(seg.text)\n\n"
            }
        }
        return out
    }

    private static func exportSRT(_ t: SavedTranscript, _ includeSpeakers: Bool = true) -> String {
        let sorted = t.segments.sorted { $0.startTime < $1.startTime }
        var out = ""
        for (idx, seg) in sorted.enumerated() {
            out += "\(idx + 1)\n"
            out += "\(srtTime(seg.startTime)) --> \(srtTime(seg.endTime))\n"
            if includeSpeakers, let speaker = seg.speaker, !speaker.isEmpty {
                out += "\(speaker): \(seg.text)\n\n"
            } else {
                out += "\(seg.text)\n\n"
            }
        }
        return out
    }

    private static func exportVTT(_ t: SavedTranscript, _ includeSpeakers: Bool = true) -> String {
        let sorted = t.segments.sorted { $0.startTime < $1.startTime }
        var out = "WEBVTT\n\n"
        for seg in sorted {
            out += "\(vttTime(seg.startTime)) --> \(vttTime(seg.endTime))\n"
            if includeSpeakers, let speaker = seg.speaker, !speaker.isEmpty {
                out += "<v \(speaker)>\(seg.text)\n\n"
            } else {
                out += "\(seg.text)\n\n"
            }
        }
        return out
    }

    private static func exportJSON(_ t: SavedTranscript, _ includeSpeakers: Bool = true) -> String {
        struct Seg: Encodable {
            let start: Double
            let end: Double
            let text: String
            let speaker: String?
        }
        struct Payload: Encodable {
            let title: String
            let createdAt: String
            let duration: Double
            let language: String?
            let model: String?
            let text: String
            let segments: [Seg]
        }

        let iso = ISO8601DateFormatter()
        let payload = Payload(
            title: t.title,
            createdAt: iso.string(from: t.createdAt),
            duration: t.duration,
            language: t.languageCode,
            model: t.modelName,
            text: t.fullText,
            segments: t.segments
                .sorted { $0.startTime < $1.startTime }
                .map { Seg(start: $0.startTime, end: $0.endTime, text: $0.text, speaker: includeSpeakers ? $0.speaker : nil) }
        )

        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? enc.encode(payload), let s = String(data: data, encoding: .utf8) {
            return s
        }
        return "{}"
    }

    // MARK: - Time helpers

    static func formatTimestamp(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        let h = total / 3600
        let m = (total % 3600) / 60
        let s = total % 60
        if h > 0 { return String(format: "%d:%02d:%02d", h, m, s) }
        return String(format: "%d:%02d", m, s)
    }

    static func formatHMS(_ seconds: Double) -> String {
        return formatTimestamp(seconds)
    }

    /// Splits a duration into whole milliseconds once, then derives the parts.
    ///
    /// The obvious `Int((total - floor(total)) * 1000)` loses a millisecond on
    /// values that binary floating point cannot hold exactly: 13.04 - 13.0 is
    /// 0.03999999999999915, so the subtitle stamped 13.040 came out 13,039.
    private static func hmsMillis(_ seconds: Double) -> (h: Int, m: Int, s: Int, ms: Int) {
        let totalMS = Int((max(0, seconds) * 1000).rounded())
        return (
            totalMS / 3_600_000,
            (totalMS % 3_600_000) / 60_000,
            (totalMS % 60_000) / 1000,
            totalMS % 1000
        )
    }

    private static func srtTime(_ seconds: Double) -> String {
        let p = hmsMillis(seconds)
        return String(format: "%02d:%02d:%02d,%03d", p.h, p.m, p.s, p.ms)
    }

    private static func vttTime(_ seconds: Double) -> String {
        let p = hmsMillis(seconds)
        return String(format: "%02d:%02d:%02d.%03d", p.h, p.m, p.s, p.ms)
    }
}
