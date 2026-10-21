import SwiftUI
import SwiftData
import Photos

/// The Image tab: describe a picture, make it, keep it.
struct ImageView: View {
    @State private var prompt: String = ""
    @FocusState private var promptFocused: Bool
    /// Z-Image Turbo unless chosen otherwise: DreamShaper, the first default,
    /// is 512 px and looked blurry and fake full-screen on an iPad.
    @AppStorage("imageModel") private var selectedModel: ImageModel = .zImage
    @AppStorage("imageSize") private var sizeID: String = ""
    @AppStorage("imageEnhancePrompt") private var enhancePrompt = false
    @State private var engine = ImageEngine()
    @State private var memory = MusicMemoryMonitor()
    @State private var showModelPicker = false
    @State private var showLibrary = false
    @State private var openedImage: SavedImage?
    @State private var showDetails = false
    @State private var reuseCount = 0
    @Environment(\.modelContext) private var modelContext
    /// Only what Recent shows; the library sheet queries the rest.
    @Query(ImageView.recentFetch) private var images: [SavedImage]
    @State private var imageCount = 0
    /// Bytes the chosen model (and Enhance, when on) still has to download.
    @State private var neededBytes: Int64 = 0
    @State private var confirmDownload = false
    /// A reused picture's seed, so the same prompt makes it again.
    @State private var reuseSeed: UInt64?
    private static let top = "top"

    private static var recentFetch: FetchDescriptor<SavedImage> {
        var fetch = FetchDescriptor<SavedImage>(sortBy: [SortDescriptor(\.createdAt, order: .reverse)])
        fetch.fetchLimit = 6
        return fetch
    }

    /// Added to the prompt: a look, in the words these models know.
    private static let styles: [(name: String, words: String)] = [
        ("Photo", "photo, natural light, sharp focus"),
        ("Cinematic", "cinematic still, dramatic lighting, film grain"),
        ("Illustration", "digital illustration, clean lines, vibrant colours"),
        ("Anime", "anime style, cel shading"),
        ("Oil painting", "oil painting, visible brush strokes"),
        ("Watercolour", "watercolour painting, soft washes"),
        ("3D render", "3D render, soft studio lighting"),
        ("Pixel art", "pixel art, 16-bit"),
        ("Editorial cartoon", "editorial cartoon, ink and wash, satirical"),
    ]

    private var size: ImageSize {
        selectedModel.sizes.first { $0.id == sizeID } ?? selectedModel.sizes[0]
    }

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Rectangle().fill(Studio.rule).frame(height: 1)
            ScrollViewReader { scroller in
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Color.clear.frame(height: 0).id(Self.top)
                        promptCard
                        if let warning = selectedModel.memoryWarning {
                            noteRow(icon: "memorychip", tint: Studio.mute, text: warning)
                                .padding(.horizontal, 4)
                        }
                        if engine.isBusy || engine.failureMessage != nil || engine.phase == .withheld {
                            progressCard
                        } else if let image = engine.lastImage, !image.isDeleted, image.modelContext != nil {
                            resultCard(image)
                        }
                        if engine.isBusy { MusicMemoryGauge(monitor: memory) }
                        if !images.isEmpty { recentSection }
                        detailsSection
                    }
                    .padding(18)
                    // A column, not the whole width of a 13" iPad.
                    .frame(maxWidth: 760)
                    .frame(maxWidth: .infinity)
                }
                .scrollDismissesKeyboard(.interactively)
                .onChange(of: reuseCount) { _, _ in
                    withAnimation { scroller.scrollTo(Self.top, anchor: .top) }
                }
            }
            generateBar
        }
        .background(Studio.bg.ignoresSafeArea())
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") { promptFocused = false }
            }
        }
        .onAppear {
            engine.modelContext = modelContext
            refreshStorage()
            #if DEBUG
            // `-ImageShowPicker YES`: the model sheet open, for screenshots.
            if UserDefaults.standard.bool(forKey: "ImageShowPicker") { showModelPicker = true }
            if prompt.isEmpty, let preset = UserDefaults.standard.string(forKey: "ImagePromptPreset") {
                prompt = preset
                // `-ImageAutoGenerate YES` starts it, for testing downloads unattended.
                if UserDefaults.standard.bool(forKey: "ImageAutoGenerate"), !engine.isBusy {
                    engine.generate(model: selectedModel, prompt: preset, size: selectedModel.sizes[0])
                }
            }
            #endif
        }
        .onDisappear { memory.stop() }
        .onChange(of: engine.isBusy) { _, busy in
            if !busy {
                memory.stop()
                refreshStorage()
            }
        }
        .onChange(of: selectedModel) { _, _ in refreshStorage() }
        .onChange(of: enhancePrompt) { _, _ in refreshStorage() }
        .onChange(of: showModelPicker) { _, open in if !open { refreshStorage() } }
        .onChange(of: showLibrary) { _, open in if !open { refreshStorage() } }
        .alert("Download \(selectedModel.displayName)?", isPresented: $confirmDownload) {
            Button("Download & generate") { generate() }
            Button("Not now", role: .cancel) {}
        } message: {
            Text("It needs \(ImageModel.formatBytes(neededBytes)), downloaded once, then it works offline. Wi‑Fi is best; the download carries on if you leave the app.")
        }
        .sheet(isPresented: $showModelPicker) {
            ImageModelPicker(selected: $selectedModel).presentationSizing(.page)
        }
        .sheet(isPresented: $showLibrary) {
            ImageLibraryView(onReuse: reuse).presentationSizing(.page)
        }
        .sheet(item: $openedImage) { image in
            NavigationStack {
                ImageDetail(image: image, onReuse: { reuse(image); openedImage = nil },
                            onDelete: { delete(image); openedImage = nil })
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) {
                            Button("Done") { openedImage = nil }
                        }
                    }
            }
            .presentationSizing(.page)
        }
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "photo.artframe")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(Studio.accent)
                Text("image")
                    .font(Studio.mono(13, weight: .semibold))
                    .foregroundColor(Studio.ink)
                Text("BETA")
                    .font(Studio.mono(9, weight: .bold))
                    .tracking(1)
                    .foregroundColor(Studio.accent)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Studio.accent.opacity(0.12)))
                    .accessibilityLabel("Beta")
            }
            Spacer()
            Button { showModelPicker = true } label: {
                HStack(spacing: 5) {
                    Circle().fill(neededBytes == 0 ? Studio.ok : Studio.mute.opacity(0.5)).frame(width: 6, height: 6)
                    Text(selectedModel.displayName)
                        .font(Studio.mono(11))
                        .foregroundColor(Studio.ink.opacity(0.82))
                        .lineLimit(1)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundColor(Studio.mute)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .overlay(Capsule().strokeBorder(Studio.rule, lineWidth: 0.5))
            }
            .buttonStyle(PressableButtonStyle())
            .disabled(engine.isBusy)
            .accessibilityLabel("Model, \(selectedModel.displayName)")

            Button { showLibrary = true } label: {
                Image(systemName: "photo.stack")
                    .font(.system(size: 16))
                    .foregroundColor(Studio.ink.opacity(0.75))
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(PressableButtonStyle())
            .padding(.leading, 8)
            .accessibilityLabel("Library")

            HelpButton(topic: .image)
                .padding(.leading, 6)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }

    // MARK: - Compose

    private func card<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12, content: content)
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(Studio.panel))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Studio.rule, lineWidth: 0.5))
    }

    private var promptCard: some View {
        card {
            HStack(spacing: 14) {
                StudioLabel(text: "Describe the picture")
                Spacer()
                if !prompt.isEmpty {
                    Button("Clear") { prompt = "" }
                        .font(Studio.mono(10, weight: .semibold))
                        .foregroundColor(Studio.mute)
                }
                if !images.isEmpty {
                    Button { showLibrary = true } label: {
                        Label("History", systemImage: "clock.arrow.circlepath")
                            .font(Studio.mono(10, weight: .semibold))
                            .foregroundColor(Studio.accent)
                    }
                }
            }
            ZStack(alignment: .topLeading) {
                if prompt.isEmpty {
                    Text("A lighthouse on a cliff at sunset, waves crashing below")
                        .font(Studio.text(15))
                        .foregroundColor(Studio.mute.opacity(0.7))
                        .padding(.top, 8)
                        .padding(.horizontal, 12)
                        .allowsHitTesting(false)
                }
                TextEditor(text: $prompt)
                    .focused($promptFocused)
                    .font(Studio.text(15))
                    .foregroundColor(Studio.ink)
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .frame(minHeight: 96)
            }
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Studio.sunk))

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 7) {
                    ForEach(Self.styles, id: \.name) { style in
                        Button {
                            let current = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                            prompt = current.isEmpty ? style.words : "\(current), \(style.words)"
                        } label: {
                            Text(style.name)
                                .font(Studio.text(12, weight: .medium))
                                .foregroundColor(Studio.ink.opacity(0.8))
                                .padding(.horizontal, 11)
                                .padding(.vertical, 6)
                                .background(Capsule().fill(Studio.sunk))
                        }
                        .buttonStyle(PressableButtonStyle())
                    }
                }
            }

            Toggle(isOn: $enhancePrompt) {
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .foregroundColor(Studio.accent)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Enhance prompt")
                            .font(Studio.text(13, weight: .medium))
                            .foregroundColor(Studio.ink)
                        Text("Qwen3.5 first writes your idea out as a detailed description, keeping your names, text and counts, then the picture is made from it. About half a minute more; \(ImageModel.rewriterSizeLabel) the first time.\(ImageModel.rewriterMemoryNote.map { " " + $0 } ?? "")")
                            .font(Studio.text(12))
                            .foregroundColor(Studio.mute)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .tint(Studio.accent)

            if let reuseSeed {
                HStack(spacing: 6) {
                    Image(systemName: "dice")
                        .font(.system(size: 11))
                    Text("Same seed as the picture reused (\(reuseSeed)) · the same prompt makes it again")
                        .font(Studio.mono(10))
                        .lineLimit(2)
                    Spacer(minLength: 4)
                    Button { self.reuseSeed = nil } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 14))
                    }
                    .accessibilityLabel("Use a new seed")
                }
                .foregroundColor(Studio.mute)
            }

            if selectedModel.sizes.count > 1 {
                HStack(spacing: 7) {
                    ForEach(selectedModel.sizes) { option in
                        let chosen = option == size
                        Button { sizeID = option.id } label: {
                            HStack(spacing: 5) {
                                Image(systemName: option.symbol).font(.system(size: 11))
                                Text(option.width == option.height ? "\(option.width)" : option.shapeName)
                                    .font(Studio.mono(10, weight: .semibold))
                            }
                            .foregroundColor(chosen ? Studio.onAccent : Studio.ink.opacity(0.75))
                            .padding(.horizontal, 9)
                            .padding(.vertical, 6)
                            .background(Capsule().fill(chosen ? Studio.accent : Studio.sunk))
                        }
                        .buttonStyle(PressableButtonStyle())
                        .disabled(engine.isBusy)
                        .accessibilityLabel("\(option.shapeName), \(option.label)")
                        .accessibilityAddTraits(chosen ? .isSelected : [])
                    }
                }
            } else {
                Text("\(size.label) · \(selectedModel.engineLabel.lowercased())")
                    .font(Studio.mono(10, weight: .semibold))
                    .foregroundColor(Studio.accent)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(Studio.accent.opacity(0.10)))
            }
        }
    }

    // MARK: - Progress and result

    @ViewBuilder
    private var progressCard: some View {
        card {
            switch engine.phase {
            case .downloading(let file, let completed, let total, let fraction, let received, let expected):
                HStack {
                    StudioLabel(text: "Downloading \(selectedModel.displayName)")
                    Spacer()
                    Text("\(Int(fraction * 100))%")
                        .font(Studio.mono(11, weight: .semibold))
                        .foregroundColor(Studio.accent)
                }
                ProgressView(value: fraction).tint(Studio.accent)
                HStack {
                    Text("\(Self.formatBytes(received)) of \(Self.formatBytes(expected))")
                        .font(Studio.mono(10))
                        .foregroundColor(Studio.mute)
                    Spacer()
                    Text(total > 1 ? "\(completed) of \(total) files done" : file)
                        .font(Studio.mono(10))
                        .foregroundColor(Studio.mute)
                        .lineLimit(1)
                }
                Text("Downloaded once. It carries on if you leave the app; the picture starts when it's done.")
                    .font(Studio.text(12))
                    .foregroundColor(Studio.mute)
                    .fixedSize(horizontal: false, vertical: true)
                if let note = engine.downloadNote {
                    Label(note, systemImage: note == "Unpacking" ? "shippingbox" : "arrow.clockwise")
                        .font(Studio.mono(10))
                        .foregroundColor(Studio.hot)
                }
            case .generating(let stage, let progress):
                HStack {
                    StudioLabel(text: "Making your picture")
                    Spacer()
                    Text("\(Int(progress * 100))%")
                        .font(Studio.mono(11, weight: .semibold))
                        .foregroundColor(Studio.accent)
                }
                ProgressView(value: progress).tint(Studio.accent)
                if let rewritten = engine.rewrittenPrompt {
                    Text(rewritten)
                        .font(Studio.text(12))
                        .foregroundColor(Studio.ink.opacity(0.75))
                        .lineLimit(5)
                        .textSelection(.enabled)
                }
                HStack(alignment: .top) {
                    Text(stage)
                        .font(Studio.mono(10))
                        .foregroundColor(Studio.mute)
                        .lineLimit(2)
                    Spacer()
                    if let started = engine.startedAt {
                        TimelineView(.periodic(from: .now, by: 1)) { context in
                            Text("\(Int(context.date.timeIntervalSince(started)))s")
                                .font(Studio.mono(10))
                                .foregroundColor(Studio.mute)
                                .monospacedDigit()
                        }
                    }
                }
            case .failed(let message):
                StudioLabel(text: "Didn't finish")
                Text(message)
                    .font(Studio.text(13))
                    .foregroundColor(Studio.hot)
                    .fixedSize(horizontal: false, vertical: true)
                messageActions
            case .withheld:
                StudioLabel(text: "Not shown")
                Text("This picture looked sexually explicit, so it wasn't shown or saved. Try rewording the prompt, or generate again for a different take.")
                    .font(Studio.text(13))
                    .foregroundColor(Studio.ink.opacity(0.8))
                    .fixedSize(horizontal: false, vertical: true)
                messageActions
            case .idle:
                EmptyView()
            }
        }
    }

    private var messageActions: some View {
        HStack(spacing: 18) {
            Button { startGenerating() } label: {
                Label("Try again", systemImage: "arrow.clockwise")
            }
            .disabled(!canGenerate)
            Button { engine.dismissMessage() } label: {
                Text("Dismiss")
            }
            .foregroundColor(Studio.mute)
            Spacer()
        }
        .font(Studio.text(13, weight: .medium))
        .foregroundColor(Studio.accent)
    }

    private func resultCard(_ image: SavedImage) -> some View {
        card {
            HStack(alignment: .firstTextBaseline) {
                StudioLabel(text: "Your picture")
                Spacer()
                Text("\(image.width) × \(image.height) · \(Self.formatSeconds(image.seconds))")
                    .font(Studio.mono(10))
                    .foregroundColor(Studio.mute)
            }
            Button { openedImage = image } label: {
                StoredImageView(image: image)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Your picture: \(Self.shortLabel(image.prompt)). Opens it full size.")
            Text(image.prompt)
                .font(Studio.text(13))
                .foregroundColor(Studio.ink.opacity(0.8))
                .lineLimit(4)
                .textSelection(.enabled)
            HStack(spacing: 16) {
                ShareLink(item: image.fileURL, preview: SharePreview(Self.shortLabel(image.prompt))) {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                SaveToPhotosButton(url: image.fileURL)
                Spacer()
                Button { startGenerating() } label: {
                    Label("New take", systemImage: "arrow.triangle.2.circlepath")
                }
                .disabled(!canGenerate)
            }
            .font(Studio.text(13, weight: .medium))
            .foregroundColor(Studio.accent)
        }
    }

    // MARK: - Library

    private var recentSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                StudioLabel(text: "Recent")
                Spacer()
                Button { showLibrary = true } label: {
                    Text("All \(imageCount) \u{2192}")
                        .font(Studio.mono(9, weight: .semibold))
                        .foregroundColor(Studio.accent)
                }
            }
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 3), spacing: 8) {
                ForEach(images) { image in
                    Button { openedImage = image } label: {
                        SquareThumbnail(image: image)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Picture: \(Self.shortLabel(image.prompt))")
                }
            }
        }
    }

    private func reuse(_ image: SavedImage) {
        prompt = image.prompt
        reuseSeed = UInt64(clamping: image.seed)
        if let model = ImageModel.allCases.first(where: { $0.displayName == image.modelName }) {
            selectedModel = model
            if let match = model.sizes.first(where: { $0.width == image.width && $0.height == image.height }) {
                sizeID = match.id
            }
        }
        reuseCount += 1
    }

    private func delete(_ image: SavedImage) {
        engine.forget(image)
        ImageThumbnails.forget(image.fileURL)
        ImageFiles.delete(image.filePath)
        modelContext.delete(image)
        try? modelContext.save()
        refreshStorage()
    }

    /// What the chosen model still needs, and how many pictures there are:
    /// both read from disk, so on changes rather than in `body`.
    private func refreshStorage() {
        neededBytes = selectedModel.bytesToDownload(enhance: enhancePrompt)
        imageCount = (try? modelContext.fetchCount(FetchDescriptor<SavedImage>())) ?? images.count
    }

    /// A prompt cut short for labels: an enhanced one runs to a paragraph.
    static func shortLabel(_ prompt: String) -> String {
        prompt.count > 80 ? String(prompt.prefix(77)) + "…" : prompt
    }

    private var detailsSection: some View {
        DisclosureGroup(isExpanded: $showDetails) {
            VStack(alignment: .leading, spacing: 10) {
                noteRow(icon: neededBytes == 0 ? "checkmark.circle" : "arrow.down.circle",
                        tint: neededBytes == 0 ? Studio.ok : Studio.mute,
                        text: neededBytes == 0
                            ? "\(selectedModel.displayName) is on this device and runs offline."
                            : "\(selectedModel.displayName) runs on this device. It needs \(ImageModel.formatBytes(neededBytes)) more, downloaded once.")
                noteRow(icon: "tag", tint: Studio.mute,
                        text: "Pictures are marked as made by AI in their file information, as news sites and Photos read it. Nothing is drawn on them.")
                if !engine.isBusy && memory.peakMB > 0 {
                    MusicMemoryGauge(monitor: memory)
                }
            }
            .padding(.top, 8)
        } label: {
            StudioLabel(text: "Details")
        }
        .tint(Studio.mute)
    }

    private func noteRow(icon: String, tint: Color, text: String) -> some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: icon)
                .font(.system(size: 13))
                .foregroundColor(tint)
                .padding(.top, 1)
            Text(text)
                .font(Studio.text(13))
                .foregroundColor(Studio.ink.opacity(0.75))
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }

    private static func formatBytes(_ bytes: Int64) -> String {
        bytes >= 1_000_000_000
            ? String(format: "%.2f GB", Double(bytes) / 1_000_000_000)
            : "\(bytes / 1_000_000) MB"
    }

    static func formatSeconds(_ seconds: Double) -> String {
        seconds < 10 ? String(format: "%.1f s", seconds) : "\(Int(seconds.rounded())) s"
    }

    // MARK: - Generate

    private var canGenerate: Bool {
        engine.isBusy || !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Generates — after asking, when that means a big download first.
    private func startGenerating() {
        promptFocused = false
        refreshStorage()
        if neededBytes > 300_000_000 {
            confirmDownload = true
        } else {
            generate()
        }
    }

    private func generate() {
        memory.start()
        engine.generate(model: selectedModel, prompt: prompt, size: size, seed: reuseSeed, enhance: enhancePrompt)
    }

    private var generateBar: some View {
        VStack(spacing: 0) {
            Rectangle().fill(Studio.rule).frame(height: 1)
            Button {
                if engine.isBusy { engine.cancel() } else { startGenerating() }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: engine.isBusy ? "stop.fill" : "wand.and.stars")
                    Text(engine.isBusy ? "Cancel"
                         : neededBytes > 0 ? "Download \(ImageModel.formatBytes(neededBytes)) & generate"
                         : "Generate \(size.label)")
                }
                .font(Studio.text(15, weight: .semibold))
                .foregroundColor(canGenerate ? Studio.onAccent : Studio.mute)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 15)
                .background(
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .fill(canGenerate ? (engine.isBusy ? Studio.hot : Studio.accent) : Studio.sunk)
                )
            }
            .buttonStyle(PressableButtonStyle())
            .disabled(!canGenerate)
            .padding(16)
            .frame(maxWidth: 760)
            .frame(maxWidth: .infinity)
        }
    }
}

// MARK: - Pieces

/// A saved picture, read from its file off the main thread.
struct StoredImageView: View {
    let image: SavedImage
    var thumbnail = false
    @State private var loaded: UIImage?

    var body: some View {
        Group {
            if let loaded {
                Image(uiImage: loaded)
                    .resizable()
                    .aspectRatio(contentMode: thumbnail ? .fill : .fit)
            } else {
                Rectangle()
                    .fill(Studio.sunk)
                    .aspectRatio(CGFloat(image.width) / CGFloat(max(image.height, 1)), contentMode: .fit)
            }
        }
        .task(id: image.filePath) {
            // Grids want about 400 px; the full view the picture's own size.
            let side = thumbnail ? 400 : max(image.width, image.height)
            if let hit = ImageThumbnails.cached(image.fileURL, maxPixel: side) {
                loaded = hit
            } else {
                loaded = await ImageThumbnails.load(image.fileURL, maxPixel: side)
            }
        }
    }
}

/// A saved picture filling a square tile, whatever its shape.
struct SquareThumbnail: View {
    let image: SavedImage

    var body: some View {
        Color.clear
            .aspectRatio(1, contentMode: .fit)
            .overlay { StoredImageView(image: image, thumbnail: true) }
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .contentShape(Rectangle())
    }
}

/// Adds a picture to Photos, asking for add-only access the first time.
struct SaveToPhotosButton: View {
    let url: URL
    @State private var saved = false
    @State private var failed = false

    var body: some View {
        Button {
            Task { await save() }
        } label: {
            Label(saved ? "Saved" : "Save", systemImage: saved ? "checkmark" : "square.and.arrow.down")
        }
        .disabled(saved)
        .alert("Couldn't save to Photos", isPresented: $failed) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("Allow Whisper to add photos in Settings › Privacy & Security › Photos.")
        }
    }

    private func save() async {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else { failed = true; return }
        do {
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetCreationRequest.forAsset().addResource(with: .photo, fileURL: url, options: nil)
            }
            saved = true
        } catch {
            failed = true
        }
    }
}

// MARK: - Model picker

struct ImageModelPicker: View {
    @Binding var selected: ImageModel
    @Environment(\.dismiss) private var dismiss

    /// Read from disk when the sheet opens and after a delete.
    @State private var needed: [ImageModel: Int64] = [:]
    @State private var removable: [ImageModel: Int64] = [:]
    @State private var rewriterBytes: Int64 = 0
    @State private var deleting: ImageModel?
    @State private var deletingRewriter = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 10) {
                    ForEach(ImageModel.allCases) { model in
                        Button { selected = model; dismiss() } label: { card(model) }
                            .buttonStyle(PressableButtonStyle())
                            .accessibilityAddTraits(selected == model ? .isSelected : [])
                            .contextMenu {
                                if let bytes = removable[model], bytes > 0 {
                                    Button(role: .destructive) { deleting = model } label: {
                                        Label("Delete download (\(ImageModel.formatBytes(bytes)))", systemImage: "trash")
                                    }
                                }
                            }
                    }
                    storageFooter
                }
                .padding(16)
                .frame(maxWidth: 760)
                .frame(maxWidth: .infinity)
            }
            .background(Studio.bg.ignoresSafeArea())
            .navigationTitle("Image Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .task { await refresh() }
            .confirmationDialog(deleting.map { "Delete \($0.displayName)?" } ?? "", isPresented: Binding(
                get: { deleting != nil }, set: { if !$0 { deleting = nil } }), titleVisibility: .visible) {
                if let model = deleting {
                    Button("Delete \(ImageModel.formatBytes(removable[model] ?? 0))", role: .destructive) {
                        model.deleteDownload()
                        deleting = nil
                        Task { await refresh() }
                    }
                }
            } message: {
                Text("Its files leave this device. It downloads again the next time you use it.")
            }
            .confirmationDialog("Delete Enhance prompt's model?", isPresented: $deletingRewriter, titleVisibility: .visible) {
                Button("Delete \(ImageModel.formatBytes(rewriterBytes))", role: .destructive) {
                    ImageModel.deleteRewriter()
                    Task { await refresh() }
                }
            } message: {
                Text("It downloads again the next time Enhance prompt is used.")
            }
        }
    }

    private var storageFooter: some View {
        let total = removable.values.reduce(0, +)
        return VStack(alignment: .leading, spacing: 8) {
            Text(total > 0
                 ? "Image models on this device: \(ImageModel.formatBytes(total)). Touch and hold a model to delete its download."
                 : "Nothing is downloaded yet. A model downloads the first time it's used.")
                .font(Studio.text(12))
                .foregroundColor(Studio.mute)
                .fixedSize(horizontal: false, vertical: true)
            if rewriterBytes > 0 {
                Button { deletingRewriter = true } label: {
                    Label("Delete Enhance prompt's model (\(ImageModel.formatBytes(rewriterBytes)))", systemImage: "trash")
                        .font(Studio.text(12, weight: .medium))
                }
                .foregroundColor(Studio.hot)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 6)
        .padding(.horizontal, 4)
    }

    private func refresh() async {
        let (n, r, w) = await Task.detached(priority: .userInitiated) { () -> ([ImageModel: Int64], [ImageModel: Int64], Int64) in
            var needed: [ImageModel: Int64] = [:], removable: [ImageModel: Int64] = [:]
            for model in ImageModel.allCases {
                needed[model] = model.bytesToDownload()
                removable[model] = model.removableBytes
            }
            return (needed, removable, ImageModel.rewriterInstalledBytes)
        }.value
        needed = n
        removable = r
        rewriterBytes = w
    }

    private func card(_ model: ImageModel) -> some View {
        let missing = needed[model]
        return VStack(alignment: .leading, spacing: 7) {
            HStack(alignment: .firstTextBaseline) {
                Text(model.displayName)
                    .font(Studio.text(16, weight: .semibold))
                    .foregroundColor(Studio.ink)
                Spacer()
                Text(model.parameterLabel)
                    .font(Studio.mono(10, weight: .semibold))
                    .foregroundColor(Studio.mute)
            }
            Text(model.tagline)
                .font(Studio.text(12))
                .foregroundColor(Studio.mute)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 7) {
                if missing == 0 {
                    chip("ON DEVICE", tint: Studio.ok)
                } else if let missing, missing < model.downloadBytes {
                    chip("\(ImageModel.formatBytes(missing)) TO DOWNLOAD")
                } else {
                    chip(model.sizeLabel)
                }
                if model.memoryWarning != nil { chip("SLOWER HERE") }
                chip(model.engineLabel)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous)
            .fill(selected == model ? Studio.panel : Studio.sunk))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
            .strokeBorder(selected == model ? Studio.accent : Studio.rule,
                          lineWidth: selected == model ? 1.5 : 0.5))
    }

    private func chip(_ text: String, tint: Color = Studio.mute) -> some View {
        Text(text)
            .font(Studio.mono(9, weight: .semibold))
            .foregroundColor(tint)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Capsule().fill(tint.opacity(0.12)))
    }
}
