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
                NavigationLink {
                    TranscriptDetailView(transcript: t)
                } label: {
                    row(t)
                }
            }
            .onDelete(perform: delete)
        }
        .listStyle(.insetGrouped)
    }

    private func row(_ t: SavedTranscript) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(t.title)
                .font(.headline)
                .lineLimit(1)
            Text(t.fullText)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .lineLimit(2)
            HStack(spacing: 10) {
                Label(TranscriptExporter.formatTimestamp(t.duration), systemImage: "clock")
                if let lang = t.languageCode {
                    Label(lang.uppercased(), systemImage: "globe")
                }
                Text(relativeDate(t.createdAt))
            }
            .font(.caption2)
            .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "text.bubble")
                .font(.system(size: 44))
                .foregroundStyle(.secondary)
            Text("No saved transcripts yet")
                .font(.headline)
            Text("Record audio or import a file from the main screen. Transcripts will be saved here automatically.")
                .font(.subheadline)
                .foregroundColor(.secondary)
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
