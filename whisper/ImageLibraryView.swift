import SwiftUI
import SwiftData

/// Every picture the Image tab has made, newest first, searchable by prompt.
struct ImageLibraryView: View {
    /// Puts a picture's prompt and model back on the Image tab.
    var onReuse: ((SavedImage) -> Void)?

    @Environment(\.modelContext) private var modelContext
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \SavedImage.createdAt, order: .reverse) private var images: [SavedImage]
    @State private var search = ""

    private var shown: [SavedImage] {
        let query = search.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty else { return images }
        return images.filter { $0.prompt.localizedCaseInsensitiveContains(query) }
    }

    var body: some View {
        NavigationStack {
            Group {
                if images.isEmpty {
                    ContentUnavailableView("No pictures yet", systemImage: "photo.artframe",
                                           description: Text("Pictures you make appear here."))
                } else {
                    ScrollView {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 110), spacing: 8)], spacing: 8) {
                            ForEach(shown) { image in
                                NavigationLink(value: image.id) {
                                    StoredImageView(image: image, thumbnail: true)
                                        .aspectRatio(1, contentMode: .fill)
                                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                                }
                                .buttonStyle(.plain)
                                .contextMenu {
                                    ShareLink(item: image.fileURL)
                                    if let onReuse {
                                        Button { onReuse(image); dismiss() } label: {
                                            Label("Use prompt again", systemImage: "arrow.uturn.backward")
                                        }
                                    }
                                    Button(role: .destructive) { delete(image) } label: {
                                        Label("Delete", systemImage: "trash")
                                    }
                                }
                                .accessibilityLabel(image.prompt)
                            }
                        }
                        .padding(14)
                    }
                    .searchable(text: $search, prompt: "Search prompts")
                }
            }
            .background(Studio.bg.ignoresSafeArea())
            .navigationTitle("Pictures")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { dismiss() }
                }
            }
            .navigationDestination(for: UUID.self) { id in
                if let image = images.first(where: { $0.id == id }) {
                    ImageDetail(image: image,
                                onReuse: onReuse.map { reuse in { reuse(image); dismiss() } },
                                onDelete: { delete(image) })
                }
            }
        }
    }

    private func delete(_ image: SavedImage) {
        ImageFiles.delete(image.filePath)
        modelContext.delete(image)
        try? modelContext.save()
    }
}

/// One picture, whole, with what made it.
struct ImageDetail: View {
    let image: SavedImage
    var onReuse: (() -> Void)?
    var onDelete: (() -> Void)?
    @Environment(\.dismiss) private var dismiss
    @State private var confirmDelete = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                StoredImageView(image: image)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                Text(image.prompt)
                    .font(Studio.text(15))
                    .foregroundColor(Studio.ink)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                VStack(alignment: .leading, spacing: 4) {
                    detail("Model", image.modelName)
                    detail("Size", "\(image.width) × \(image.height)")
                    detail("Made in", ImageView.formatSeconds(image.seconds))
                    detail("Seed", "\(image.seed)")
                    detail("Date", image.createdAt.formatted(date: .abbreviated, time: .shortened))
                }
                HStack(spacing: 18) {
                    ShareLink(item: image.fileURL) {
                        Label("Share", systemImage: "square.and.arrow.up")
                    }
                    SaveToPhotosButton(url: image.fileURL)
                    if let onReuse {
                        Button { onReuse() } label: {
                            Label("Use again", systemImage: "arrow.uturn.backward")
                        }
                    }
                    Spacer()
                    if onDelete != nil {
                        Button(role: .destructive) { confirmDelete = true } label: {
                            Image(systemName: "trash")
                        }
                        .accessibilityLabel("Delete")
                    }
                }
                .font(Studio.text(14, weight: .medium))
                .foregroundColor(Studio.accent)
            }
            .padding(18)
            .frame(maxWidth: 760)
            .frame(maxWidth: .infinity)
        }
        .background(Studio.bg.ignoresSafeArea())
        .navigationTitle("Picture")
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog("Delete this picture?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete", role: .destructive) { onDelete?(); dismiss() }
        }
    }

    private func detail(_ label: String, _ value: String) -> some View {
        HStack(spacing: 8) {
            Text(label.uppercased())
                .font(Studio.mono(9, weight: .semibold))
                .foregroundColor(Studio.mute)
                .frame(width: 64, alignment: .leading)
            Text(value)
                .font(Studio.mono(11))
                .foregroundColor(Studio.ink.opacity(0.8))
        }
    }
}
