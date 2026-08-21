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
    static func export(_ transcript: SavedTranscript, as format: ExportFormat) -> String {
        switch format {
        case .txt: return exportTXT(transcript)
        case .srt: return exportSRT(transcript)
        case .vtt: return exportVTT(transcript)
        case .markdown: return exportMarkdown(transcript)
        case .json: return exportJSON(transcript)
        }
    }

    static func write(_ transcript: SavedTranscript, as format: ExportFormat) throws -> URL {
        let body = export(transcript, as: format)
        let safeTitle = transcript.title
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
        let filename = "\(safeTitle).\(format.fileExtension)"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        try body.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    // MARK: - Formatters

    private static func exportTXT(_ t: SavedTranscript) -> String {
        let sorted = t.segments.sorted { $0.startTime < $1.startTime }
        if sorted.isEmpty { return t.fullText }
        return sorted.map { seg in
            if let speaker = seg.speaker, !speaker.isEmpty {
                return "\(speaker): \(seg.text)"
            }
            return seg.text
        }.joined(separator: "\n")
    }

    private static func exportMarkdown(_ t: SavedTranscript) -> String {
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
            if let speaker = seg.speaker, !speaker.isEmpty {
                out += "\(stamp) **\(speaker):** \(seg.text)\n\n"
            } else {
                out += "\(stamp) \(seg.text)\n\n"
            }
        }
        return out
    }

    private static func exportSRT(_ t: SavedTranscript) -> String {
        let sorted = t.segments.sorted { $0.startTime < $1.startTime }
        var out = ""
        for (idx, seg) in sorted.enumerated() {
            out += "\(idx + 1)\n"
            out += "\(srtTime(seg.startTime)) --> \(srtTime(seg.endTime))\n"
            if let speaker = seg.speaker, !speaker.isEmpty {
                out += "\(speaker): \(seg.text)\n\n"
            } else {
                out += "\(seg.text)\n\n"
            }
        }
        return out
    }

    private static func exportVTT(_ t: SavedTranscript) -> String {
        let sorted = t.segments.sorted { $0.startTime < $1.startTime }
        var out = "WEBVTT\n\n"
        for seg in sorted {
            out += "\(vttTime(seg.startTime)) --> \(vttTime(seg.endTime))\n"
            if let speaker = seg.speaker, !speaker.isEmpty {
                out += "<v \(speaker)>\(seg.text)\n\n"
            } else {
                out += "\(seg.text)\n\n"
            }
        }
        return out
    }

    private static func exportJSON(_ t: SavedTranscript) -> String {
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
                .map { Seg(start: $0.startTime, end: $0.endTime, text: $0.text, speaker: $0.speaker) }
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

    private static func srtTime(_ seconds: Double) -> String {
        let total = max(0, seconds)
        let h = Int(total) / 3600
        let m = (Int(total) % 3600) / 60
        let s = Int(total) % 60
        let ms = Int((total - floor(total)) * 1000)
        return String(format: "%02d:%02d:%02d,%03d", h, m, s, ms)
    }

    private static func vttTime(_ seconds: Double) -> String {
        let total = max(0, seconds)
        let h = Int(total) / 3600
        let m = (Int(total) % 3600) / 60
        let s = Int(total) % 60
        let ms = Int((total - floor(total)) * 1000)
        return String(format: "%02d:%02d:%02d.%03d", h, m, s, ms)
    }
}
