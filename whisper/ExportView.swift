import SwiftUI

/// Pick a format, see the actual file, then share or copy it.
///
/// The preview is the real output — `TranscriptExporter` generates it, not a
/// mock-up of it — so the timestamp punctuation, the WEBVTT header and the byte
/// count are all whatever the exported file will genuinely contain.
struct ExportView: View {
    let transcript: SavedTranscript
    @Environment(\.dismiss) private var dismiss
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    @State private var format: ExportFormat = .srt
    @State private var includeSpeakers = true
    @State private var shareItem: ShareItem?
    @State private var copied = false

    private var isRegular: Bool { horizontalSizeClass == .regular }

    private var body_: String {
        TranscriptExporter.export(transcript, as: format, includeSpeakers: includeSpeakers)
    }

    private var filename: String {
        let safe = transcript.title
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
        return "\(safe).\(format.fileExtension)"
    }

    /// Whether the chosen format has anywhere to put a timestamp. Plain text
    /// doesn't, and saying so is better than showing a control that lies.
    private var carriesTimestamps: Bool { format != .txt }

    /// True when at least one segment actually has a speaker label — otherwise
    /// the toggle would appear to do nothing.
    private var hasSpeakers: Bool {
        transcript.segments.contains { ($0.speaker ?? "").isEmpty == false }
    }

    var body: some View {
        NavigationStack {
            Group {
                if isRegular {
                    HStack(spacing: 0) {
                        controls
                            .frame(width: 268)
                            .background(Studio.panel)
                        Rectangle().fill(Studio.rule).frame(width: 1)
                        preview
                    }
                } else {
                    VStack(spacing: 0) {
                        compactControls
                        preview
                    }
                }
            }
            .background(Studio.bg.ignoresSafeArea())
            .navigationTitle("Export")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }.foregroundColor(Studio.mute)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Share") { share() }.foregroundColor(Studio.accent).fontWeight(.semibold)
                }
            }
            .sheet(item: $shareItem) { item in
                ShareSheet(activityItems: [item.url])
            }
        }
        .presentationSizing(.page)
    }

    // MARK: - Controls

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                VStack(alignment: .leading, spacing: 8) {
                    StudioLabel(text: "Format")
                    VStack(spacing: 2) {
                        ForEach(ExportFormat.allCases) { f in
                            formatRow(f)
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    StudioLabel(text: "Include")
                    VStack(spacing: 2) {
                        checkRow(
                            title: "Speaker labels",
                            on: includeSpeakers && hasSpeakers,
                            enabled: hasSpeakers,
                            note: hasSpeakers ? nil : "none set"
                        ) { includeSpeakers.toggle() }

                        checkRow(
                            title: "Timestamps",
                            on: carriesTimestamps,
                            enabled: false,
                            note: carriesTimestamps ? nil : "n/a in txt"
                        ) {}
                    }
                }

                Spacer(minLength: 8)

                VStack(spacing: 9) {
                    Button(action: share) {
                        HStack(spacing: 8) {
                            Image(systemName: "square.and.arrow.up")
                            Text("Share \(format.shortName)")
                        }
                        .font(Studio.text(14, weight: .semibold))
                        .foregroundColor(Studio.onAccent)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(RoundedRectangle(cornerRadius: Studio.corner).fill(Studio.accent))
                    }
                    Button(action: copy) {
                        HStack(spacing: 8) {
                            Image(systemName: copied ? "checkmark" : "doc.on.doc")
                            Text(copied ? "Copied" : "Copy to clipboard")
                        }
                        .font(Studio.text(14))
                        .foregroundColor(copied ? Studio.ok : Studio.ink.opacity(0.8))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(
                            RoundedRectangle(cornerRadius: Studio.corner)
                                .strokeBorder(Studio.rule, lineWidth: 1)
                        )
                    }
                }
            }
            .padding(18)
        }
    }

    private var compactControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 7) {
                    ForEach(ExportFormat.allCases) { f in
                        Button { format = f } label: {
                            Text(f.shortName)
                                .font(Studio.mono(12, weight: .semibold))
                                .foregroundColor(f == format ? Studio.onAccent : Studio.ink)
                                .padding(.horizontal, 14)
                                .padding(.vertical, 8)
                                .background(
                                    Capsule()
                                        .fill(f == format ? Studio.accent : Color.clear)
                                        .overlay(Capsule().strokeBorder(f == format ? Color.clear : Studio.rule, lineWidth: 1))
                                )
                        }
                    }
                }
                .padding(.horizontal, 18)
            }

            HStack(spacing: 18) {
                if hasSpeakers {
                    Button { includeSpeakers.toggle() } label: {
                        HStack(spacing: 8) {
                            checkBox(on: includeSpeakers)
                            Text("Speakers").font(Studio.text(12)).foregroundColor(Studio.ink)
                        }
                    }
                }
                HStack(spacing: 8) {
                    checkBox(on: carriesTimestamps)
                    Text("Timestamps").font(Studio.text(12)).foregroundColor(Studio.ink)
                    if !carriesTimestamps {
                        Text("n/a in txt").font(Studio.mono(9)).foregroundColor(Studio.mute)
                    }
                }
                .opacity(carriesTimestamps ? 1 : 0.45)
                Spacer()
            }
            .padding(.horizontal, 18)
        }
        .padding(.vertical, 14)
        .background(Studio.panel)
    }

    private func formatRow(_ f: ExportFormat) -> some View {
        Button { format = f } label: {
            HStack(alignment: .top, spacing: 11) {
                Circle()
                    .strokeBorder(f == format ? Studio.accent : Studio.mute, lineWidth: f == format ? 4.5 : 1)
                    .frame(width: 15, height: 15)
                VStack(alignment: .leading, spacing: 2) {
                    Text(f.shortName)
                        .font(Studio.mono(12, weight: .semibold))
                        .foregroundColor(Studio.ink)
                    Text(f.blurb)
                        .font(Studio.text(11))
                        .foregroundColor(Studio.mute)
                        .multilineTextAlignment(.leading)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 11)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(f == format ? Studio.accent.opacity(0.10) : Color.clear)
            )
            .contentShape(Rectangle())
        }
    }

    private func checkRow(title: String, on: Bool, enabled: Bool, note: String?, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 11) {
                checkBox(on: on)
                Text(title).font(Studio.text(13)).foregroundColor(Studio.ink)
                Spacer(minLength: 0)
                if let note {
                    Text(note).font(Studio.mono(9)).foregroundColor(Studio.mute)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 9)
            .contentShape(Rectangle())
        }
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.45)
    }

    private func checkBox(on: Bool) -> some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(on ? Studio.accent : Color.clear)
            .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(Studio.mute, lineWidth: 1))
            .frame(width: 15, height: 15)
            .overlay {
                if on {
                    Image(systemName: "checkmark")
                        .font(.system(size: 9, weight: .heavy))
                        .foregroundColor(Studio.onAccent)
                }
            }
    }

    // MARK: - Preview

    private var preview: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: "doc.text")
                    .font(.system(size: 12))
                    .foregroundColor(Studio.mute)
                Text(filename)
                    .font(Studio.mono(12))
                    .foregroundColor(Studio.ink.opacity(0.85))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 8)
                Text(byteLabel)
                    .font(Studio.mono(10))
                    .foregroundColor(Studio.mute)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Studio.panel)
            .overlay(alignment: .bottom) { Rectangle().fill(Studio.rule).frame(height: 1) }

            ScrollView {
                Text(body_)
                    .font(Studio.mono(11.5))
                    .foregroundColor(Studio.ink.opacity(0.9))
                    .lineSpacing(5)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
            }
            .background(Studio.sunk)
        }
    }

    private var byteLabel: String {
        let n = body_.utf8.count
        return n < 1024 ? "\(n) bytes" : String(format: "%.1f KB", Double(n) / 1024)
    }

    // MARK: - Actions

    private func share() {
        do {
            let url = try TranscriptExporter.write(transcript, as: format, includeSpeakers: includeSpeakers)
            shareItem = ShareItem(url: url)
        } catch {
            print("⚠️ Export failed: \(error)")
        }
    }

    private func copy() {
        UIPasteboard.general.string = body_
        copied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { copied = false }
    }
}

extension ExportFormat {
    /// Short name for chips and buttons — `displayName` is the long form.
    var shortName: String {
        switch self {
        case .txt: return "TXT"
        case .srt: return "SRT"
        case .vtt: return "VTT"
        case .markdown: return "Markdown"
        case .json: return "JSON"
        }
    }

    var blurb: String {
        switch self {
        case .txt: return "Plain text, no timing"
        case .srt: return "SubRip subtitles"
        case .vtt: return "WebVTT for the web"
        case .markdown: return "Headed, with inline stamps"
        case .json: return "Full data, for tooling"
        }
    }
}
