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
    @State private var pendingShareURL: URL?
    @State private var renamingTitle = false
    @State private var titleDraft: String = ""

    var sortedSegments: [SavedSegment] {
        transcript.segments.sorted { $0.startTime < $1.startTime }
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
        .onAppear(perform: loadAudio)
        .sheet(item: $pendingShareURL) { url in
            ShareSheet(activityItems: [url])
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
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(sortedSegments) { seg in
                        segmentRow(seg)
                            .id(seg.id)
                    }
                }
                .padding()
            }
            .onChange(of: activeSegmentID) { _, newID in
                guard let id = newID else { return }
                withAnimation(.easeOut(duration: 0.25)) {
                    proxy.scrollTo(id, anchor: .center)
                }
            }
        }
    }

    private var activeSegmentID: UUID? {
        sortedSegments.first(where: { player.currentTime >= $0.startTime && player.currentTime < $0.endTime })?.id
    }

    @ViewBuilder
    private func segmentRow(_ seg: SavedSegment) -> some View {
        let isActive = (seg.id == activeSegmentID)
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
                    speakerDraft = seg.speaker ?? ""
                    editingSpeakerForSegment = seg
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
        .onTapGesture {
            player.seek(to: seg.startTime)
        }
        .alert("Speaker name",
               isPresented: Binding(
                get: { editingSpeakerForSegment?.id == seg.id },
                set: { if !$0 { editingSpeakerForSegment = nil } }
               )) {
            TextField("Speaker", text: $speakerDraft)
            Button("Save") {
                let trimmed = speakerDraft.trimmingCharacters(in: .whitespaces)
                seg.speaker = trimmed.isEmpty ? nil : trimmed
            }
            Button("Clear", role: .destructive) {
                seg.speaker = nil
            }
            Button("Cancel", role: .cancel) {}
        }
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
            pendingShareURL = url
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

// Allow URL to be used directly with .sheet(item:)
extension URL: Identifiable {
    public var id: String { absoluteString }
}
