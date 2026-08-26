import SwiftUI
import SwiftData

struct TranscriptLibraryView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \SavedTranscript.createdAt, order: .reverse)
    private var transcripts: [SavedTranscript]

    @State private var searchText: String = ""
    /// Applied on a short delay. Filtering scans every transcript's full body
    /// text, so running it on each keystroke made typing stutter on a large
    /// library.
    @State private var debouncedSearch: String = ""

    /// One shared formatter. Allocating a RelativeDateTimeFormatter is
    /// expensive and this used to happen once per row, on every render.
    private static let relativeFormatter: RelativeDateTimeFormatter = {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .abbreviated
        return f
    }()

    var filtered: [SavedTranscript] {
        let trimmed = debouncedSearch.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return transcripts }
        let q = trimmed.lowercased()
        return transcripts.filter {
            $0.title.lowercased().contains(q)
                || $0.fullText.lowercased().contains(q)
                || ($0.languageCode?.lowercased().contains(q) ?? false)
        }
    }

    var body: some View {
        NavigationStack {
            Group {
                if transcripts.isEmpty {
                    emptyState
                } else {
                    list
                }
            }
            .background(Studio.bg.ignoresSafeArea())
            .navigationTitle("Library")
            .navigationBarTitleDisplayMode(.large)
            .searchable(text: $searchText, prompt: "Search transcripts and text")
            .task(id: searchText) {
                try? await Task.sleep(for: .milliseconds(200))
                guard !Task.isCancelled else { return }
                debouncedSearch = searchText
            }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var list: some View {
        List {
            ForEach(filtered) { t in
                ZStack {
                    NavigationLink {
                        TranscriptDetailView(transcript: t)
                    } label: { EmptyView() }
                    .opacity(0)

                    row(t)
                }
                .listRowInsets(EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14))
                .listRowBackground(Studio.bg)
                .listRowSeparatorTint(Studio.rule)
            }
            .onDelete(perform: delete)
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .background(Studio.bg)
    }

    /// Each row carries its own measured envelope, so the library reads as a
    /// set of recordings with distinct shapes rather than a wall of text. The
    /// envelope is stored on the transcript, so nothing is decoded here.
    private func row(_ t: SavedTranscript) -> some View {
        HStack(spacing: 14) {
            Group {
                if let env = t.envelope(limit: 34) {
                    WaveformView(buckets: env, progress: nil, idleColor: Studio.mute.opacity(0.55))
                } else {
                    WaveformPlaceholder()
                }
            }
            .frame(width: 96, height: 34)

            VStack(alignment: .leading, spacing: 4) {
                Text(t.title)
                    .font(Studio.text(15, weight: .medium))
                    .foregroundColor(Studio.ink)
                    .lineLimit(1)
                Text(t.fullText)
                    .font(Studio.text(12))
                    .foregroundColor(Studio.mute)
                    .lineLimit(1)
                HStack(spacing: 8) {
                    Text(TranscriptExporter.formatTimestamp(t.duration))
                    if let lang = t.languageCode, !lang.isEmpty {
                        Text(lang.uppercased())
                    }
                    if let model = t.modelName {
                        Text(model)
                    }
                    Text(relativeDate(t.createdAt))
                }
                .font(Studio.mono(9))
                .foregroundColor(Studio.mute)
            }

            Spacer(minLength: 4)

            Image(systemName: "chevron.right")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(Studio.mute.opacity(0.7))
        }
        .contentShape(Rectangle())
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform")
                .font(.system(size: 40, weight: .light))
                .foregroundColor(Studio.mute)
            Text("No sessions yet")
                .font(Studio.text(17, weight: .semibold))
                .foregroundColor(Studio.ink)
            Text("Record audio or import a file from the main screen. Everything you transcribe is kept here.")
                .font(Studio.text(13))
                .foregroundColor(Studio.mute)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private func delete(at offsets: IndexSet) {
        for index in offsets {
            let t = filtered[index]
            if let path = t.audioFilePath {
                AudioFiles.deleteAudio(relativePath: path)
            }
            modelContext.delete(t)
        }
        try? modelContext.save()
    }

    private func relativeDate(_ d: Date) -> String {
        Self.relativeFormatter.localizedString(for: d, relativeTo: Date())
    }
}
