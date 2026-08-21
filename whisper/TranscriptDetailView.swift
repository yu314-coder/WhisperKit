import SwiftUI
import SwiftData
import UniformTypeIdentifiers

struct TranscriptDetailView: View {
    @Bindable var transcript: SavedTranscript
    @Environment(\.modelContext) private var modelContext
    @Environment(\.dismiss) private var dismiss

    @StateObject private var player = TranscriptPlayer()
    @State private var editingSpeakerForSegment: SavedSegment?
    @State private var speakerDraft: String = ""
    @State private var showExportPicker = false
    @State private var pendingShare: ShareItem?
    @State private var renamingTitle = false
    @State private var titleDraft: String = ""

    /// Sorted once on appear rather than recomputed on every access. Segment
    /// start times never change after transcription, so the order is stable;
    /// `revision` is bumped when a speaker label is edited so the list still
    /// refreshes for the one field that *can* change.
    @State private var sortedSegments: [SavedSegment] = []
    @State private var revision: Int = 0

    /// The segment under the playhead. Held as state and only written when it
    /// actually changes, so 10 Hz playback ticks don't invalidate the list.
    @State private var activeSegmentID: UUID?

    /// Binary search — the list can run to thousands of segments on a long
    /// recording, and this used to be a linear scan performed once per row.
    private func segmentID(at time: Double) -> UUID? {
        var low = 0
        var high = sortedSegments.count - 1
        while low <= high {
            let mid = (low + high) / 2
            let seg = sortedSegments[mid]
            if time < seg.startTime {
                high = mid - 1
            } else if time >= seg.endTime {
                low = mid + 1
            } else {
                return seg.id
            }
        }
        return nil
    }

    var body: some View {
        VStack(spacing: 0) {
            headerCard
            Divider()
            segmentList
            playerBar
        }
        .background(Color(.systemGroupedBackground).ignoresSafeArea())
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { toolbar }
        .onAppear {
            sortedSegments = transcript.segments.sorted { $0.startTime < $1.startTime }
            loadAudio()
        }
        // Recompute the highlight only when the playhead crosses a boundary.
        .onReceive(player.$currentTime) { t in
            let id = segmentID(at: t)
            if id != activeSegmentID { activeSegmentID = id }
        }
        // One alert for the whole screen. It used to be attached to every row,
        // which put N alert modifiers in the view tree.
        .alert("Speaker name", isPresented: Binding(
            get: { editingSpeakerForSegment != nil },
            set: { if !$0 { editingSpeakerForSegment = nil } }
        )) {
            TextField("Speaker", text: $speakerDraft)
            Button("Save") {
                let trimmed = speakerDraft.trimmingCharacters(in: .whitespaces)
                editingSpeakerForSegment?.speaker = trimmed.isEmpty ? nil : trimmed
                editingSpeakerForSegment = nil
                revision &+= 1
            }
            Button("Clear", role: .destructive) {
                editingSpeakerForSegment?.speaker = nil
                editingSpeakerForSegment = nil
                revision &+= 1
            }
            Button("Cancel", role: .cancel) { editingSpeakerForSegment = nil }
        }
        .sheet(item: $pendingShare) { item in
            ShareSheet(activityItems: [item.url])
        }
        .confirmationDialog("Export Transcript", isPresented: $showExportPicker, titleVisibility: .visible) {
            ForEach(ExportFormat.allCases) { fmt in
                Button(fmt.displayName) { exportAs(fmt) }
            }
            Button("Cancel", role: .cancel) {}
        }
        .alert("Rename Transcript", isPresented: $renamingTitle) {
            TextField("Title", text: $titleDraft)
            Button("Save") { transcript.title = titleDraft.trimmingCharacters(in: .whitespaces).isEmpty ? transcript.title : titleDraft }
            Button("Cancel", role: .cancel) {}
        }
    }

    // MARK: - Header

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(transcript.title)
                    .font(.title2.bold())
                    .lineLimit(2)
                Spacer()
                Button {
                    titleDraft = transcript.title
                    renamingTitle = true
                } label: {
                    Image(systemName: "pencil")
                        .foregroundColor(.secondary)
                }
            }

            HStack(spacing: 12) {
                Label(formatDate(transcript.createdAt), systemImage: "calendar")
                Label(TranscriptExporter.formatTimestamp(transcript.duration), systemImage: "clock")
                if let lang = transcript.languageCode {
                    Label(lang.uppercased(), systemImage: "globe")
                }
            }
            .font(.caption)
            .foregroundColor(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground))
    }

    // MARK: - Segment list

    private var segmentList: some View {
        SegmentListView(
            segments: sortedSegments,
            activeID: activeSegmentID,
            revision: revision,
            onSeek: { player.seek(to: $0) },
            onEditSpeaker: { seg in
                speakerDraft = seg.speaker ?? ""
                editingSpeakerForSegment = seg
            }
        )
        .equatable()
    }

    // MARK: - Player bar

    private var playerBar: some View {
        VStack(spacing: 8) {
            if player.duration > 0 {
                ProgressView(value: min(player.currentTime, player.duration), total: max(player.duration, 0.01))
                    .progressViewStyle(.linear)
                HStack {
                    Text(TranscriptExporter.formatTimestamp(player.currentTime))
                        .font(.caption.monospacedDigit())
                        .foregroundColor(.secondary)
                    Spacer()
                    Button {
                        player.seek(to: max(0, player.currentTime - 10))
                    } label: {
                        Image(systemName: "gobackward.10").font(.title3)
                    }
                    Button {
                        player.togglePlayPause()
                    } label: {
                        Image(systemName: player.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                            .font(.system(size: 44))
                            .foregroundStyle(.tint)
                    }
                    Button {
                        player.seek(to: min(player.duration, player.currentTime + 10))
                    } label: {
                        Image(systemName: "goforward.10").font(.title3)
                    }
                    Spacer()
                    Text(TranscriptExporter.formatTimestamp(player.duration))
                        .font(.caption.monospacedDigit())
                        .foregroundColor(.secondary)
                }
            } else {
                Text("Audio file not available for this transcript")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.vertical, 12)
            }
        }
        .padding()
        .background(.ultraThinMaterial)
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .topBarTrailing) {
            Menu {
                Button {
                    showExportPicker = true
                } label: {
                    Label("Export…", systemImage: "square.and.arrow.up")
                }
                Button {
                    UIPasteboard.general.string = transcript.fullText
                } label: {
                    Label("Copy full text", systemImage: "doc.on.doc")
                }
                Divider()
                Button(role: .destructive) {
                    modelContext.delete(transcript)
                    try? modelContext.save()
                    dismiss()
                } label: {
                    Label("Delete", systemImage: "trash")
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
        }
    }

    // MARK: - Helpers

    private func loadAudio() {
        guard let relative = transcript.audioFilePath else { return }
        let url = AudioFiles.urlForRelativePath(relative)
        if FileManager.default.fileExists(atPath: url.path) {
            player.load(audioURL: url)
        }
    }

    private func exportAs(_ fmt: ExportFormat) {
        do {
            let url = try TranscriptExporter.write(transcript, as: fmt)
            pendingShare = ShareItem(url: url)
        } catch {
            print("Export error: \(error)")
        }
    }

    private func formatDate(_ d: Date) -> String {
        let df = DateFormatter()
        df.dateStyle = .medium
        df.timeStyle = .short
        return df.string(from: d)
    }
}

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

/// Wraps an export URL for `.sheet(item:)`. Conforming `URL` itself to
/// `Identifiable` would break if Foundation ever adds that conformance.
struct ShareItem: Identifiable {
    let id = UUID()
    let url: URL
}

// MARK: - Segment list

/// Extracted and `Equatable` so that playback ticks — which republish the
/// player's `currentTime` several times a second — cannot invalidate the whole
/// segment list. SwiftUI re-renders it only when the highlighted segment, the
/// segment count, or the speaker revision actually changes.
private struct SegmentListView: View, Equatable {
    let segments: [SavedSegment]
    let activeID: UUID?
    let revision: Int
    let onSeek: (Double) -> Void
    let onEditSpeaker: (SavedSegment) -> Void

    static func == (a: SegmentListView, b: SegmentListView) -> Bool {
        a.activeID == b.activeID
            && a.revision == b.revision
            && a.segments.count == b.segments.count
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(segments) { seg in
                        row(seg, isActive: seg.id == activeID)
                            .id(seg.id)
                    }
                }
                .padding()
            }
            .onChange(of: activeID) { _, newID in
                guard let id = newID else { return }
                withAnimation(.easeOut(duration: 0.25)) {
                    proxy.scrollTo(id, anchor: .center)
                }
            }
        }
    }

    @ViewBuilder
    private func row(_ seg: SavedSegment, isActive: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(TranscriptExporter.formatTimestamp(seg.startTime))
                    .font(.caption.monospacedDigit())
                    .foregroundColor(isActive ? .accentColor : .secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(
                        Capsule().fill(isActive ? Color.accentColor.opacity(0.18) : Color.gray.opacity(0.12))
                    )

                Button {
                    onEditSpeaker(seg)
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "person.fill")
                            .font(.caption2)
                        Text(seg.speaker?.isEmpty == false ? seg.speaker! : "Add speaker")
                            .font(.caption.weight(.semibold))
                    }
                    .foregroundColor(seg.speaker?.isEmpty == false ? .blue : .secondary)
                }
                .buttonStyle(.plain)

                Spacer()
            }

            Text(seg.text)
                .font(.body)
                .foregroundColor(.primary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(isActive ? Color.accentColor.opacity(0.08) : Color(.tertiarySystemGroupedBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(isActive ? Color.accentColor.opacity(0.5) : Color.clear, lineWidth: 1)
        )
        .contentShape(Rectangle())
        .onTapGesture { onSeek(seg.startTime) }
    }
}
