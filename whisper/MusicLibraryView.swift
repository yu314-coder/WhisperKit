import SwiftUI
import SwiftData
import AVFoundation

/// Everything the Music tab has generated, newest first, searchable by what
/// made it. A song opens to its full prompt and lyrics, and can be made
/// again from them.
struct MusicLibraryView: View {
    /// Puts a song's prompt, lyrics and settings back on the Music tab.
    var onReuse: ((SavedMusic) -> Void)?

    @Environment(\.modelContext) private var modelContext
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \SavedMusic.createdAt, order: .reverse) private var clips: [SavedMusic]

    @State private var player: AVAudioPlayer?
    @State private var playingID: UUID?
    @State private var search = ""

    private var shown: [SavedMusic] {
        let query = search.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty else { return clips }
        return clips.filter {
            $0.prompt.localizedCaseInsensitiveContains(query)
                || ($0.lyrics?.localizedCaseInsensitiveContains(query) ?? false)
        }
    }

    var body: some View {
        NavigationStack {
            Group {
                if clips.isEmpty { emptyState } else { list }
            }
            .background(Studio.bg.ignoresSafeArea())
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { stop(); dismiss() }
                }
            }
            .navigationDestination(for: UUID.self) { id in
                if let clip = clips.first(where: { $0.id == id }) {
                    MusicClipDetail(clip: clip,
                                    onReuse: onReuse.map { reuse in { stop(); reuse(clip); dismiss() } },
                                    onDelete: { delete(clip) })
                }
            }
        }
        .onDisappear { stop() }
    }

    private var list: some View {
        List {
            ForEach(shown) { clip in
                NavigationLink(value: clip.id) {
                    MusicRow(clip: clip, isPlaying: playingID == clip.id) { toggle(clip) }
                }
                .listRowInsets(EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14))
                .listRowBackground(Studio.bg)
                .listRowSeparatorTint(Studio.rule)
                .swipeActions {
                    Button(role: .destructive) { delete(clip) } label: {
                        Label("Delete", systemImage: "trash")
                    }
                    if let onReuse {
                        Button { stop(); onReuse(clip); dismiss() } label: {
                            Label("Use again", systemImage: "arrow.uturn.backward")
                        }
                        .tint(Studio.accent)
                    }
                }
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .background(Studio.bg)
        .searchable(text: $search, prompt: "Search prompts and lyrics")
        .overlay {
            if shown.isEmpty {
                ContentUnavailableView.search(text: search)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "music.note.list")
                .font(.system(size: 40, weight: .light))
                .foregroundColor(Studio.mute)
            Text("Nothing generated yet")
                .font(Studio.text(17, weight: .semibold))
                .foregroundColor(Studio.ink)
            Text("Describe a piece of music on the Music tab and it will be kept here, with its prompt and lyrics.")
                .font(Studio.text(13))
                .foregroundColor(Studio.mute)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private func toggle(_ clip: SavedMusic) {
        if playingID == clip.id { stop(); return }
        guard let url = clip.audioURL, FileManager.default.fileExists(atPath: url.path) else { return }
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
        player = try? AVAudioPlayer(contentsOf: url)
        player?.play()
        playingID = clip.id
    }

    private func stop() {
        player?.stop()
        player = nil
        playingID = nil
    }

    private func delete(_ clip: SavedMusic) {
        if playingID == clip.id { stop() }
        // The audio file is ours to remove; leaving it behind would grow the
        // container with clips nothing references any more.
        if let path = clip.audioFilePath { AudioFiles.deleteAudio(relativePath: path) }
        modelContext.delete(clip)
        try? modelContext.save()
    }
}

/// One clip: its measured waveform, the prompt that made it (in full, up to
/// a few lines), the first line of its lyrics, and how it was made.
struct MusicRow: View {
    let clip: SavedMusic
    let isPlaying: Bool
    let onToggle: () -> Void

    static let relativeFormatter: RelativeDateTimeFormatter = {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .abbreviated
        return f
    }()

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Button(action: onToggle) {
                Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 32))
                    .foregroundColor(Studio.accent)
            }
            .buttonStyle(PressableButtonStyle())
            .accessibilityLabel(isPlaying ? "Pause" : "Play")

            VStack(alignment: .leading, spacing: 5) {
                Text(clip.prompt)
                    .font(Studio.text(14, weight: .medium))
                    .foregroundColor(Studio.ink)
                    .lineLimit(3)
                    .multilineTextAlignment(.leading)
                if let firstLine = clip.firstLyricLine {
                    Label(firstLine, systemImage: "music.mic")
                        .font(Studio.text(12))
                        .foregroundColor(Studio.ink.opacity(0.65))
                        .lineLimit(1)
                }
                HStack(spacing: 8) {
                    Group {
                        if let envelope = clip.envelope(limit: 28) {
                            WaveformView(buckets: envelope, progress: nil, idleColor: Studio.mute.opacity(0.55))
                        } else {
                            WaveformPlaceholder()
                        }
                    }
                    .frame(width: 56, height: 14)
                    Text(clip.summary)
                        .font(Studio.mono(9))
                        .foregroundColor(Studio.mute)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 0)
        }
        .contentShape(Rectangle())
    }
}

/// A song on its own page: play and scrub it, read its whole prompt and
/// lyrics, and make it again.
struct MusicClipDetail: View {
    let clip: SavedMusic
    var onReuse: (() -> Void)?
    var onDelete: (() -> Void)?

    @Environment(\.dismiss) private var dismiss
    @State private var player: AVAudioPlayer?
    @State private var copied = false
    @State private var confirmDelete = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                playerCard
                section("Prompt") {
                    Text(clip.prompt)
                        .font(Studio.text(16))
                        .foregroundColor(Studio.ink)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let lyrics = clip.lyrics, !lyrics.isEmpty {
                    section("Lyrics" + (clip.language.map { " · sung in \(Languages.englishName($0))" } ?? "")) {
                        Text(lyrics)
                            .font(Studio.text(15))
                            .foregroundColor(Studio.ink)
                            .lineSpacing(3)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                } else {
                    section("Lyrics") {
                        // Every clip saved since lyrics were kept records
                        // the arrangement setting; older ones have neither.
                        Text(clip.fullerArrangement == nil
                             ? "Not recorded for songs made before this version."
                             : "Instrumental")
                            .font(Studio.text(14))
                            .foregroundColor(Studio.mute)
                    }
                }
                section("Made with") {
                    VStack(alignment: .leading, spacing: 6) {
                        detailRow("Length", TranscriptExporter.formatTimestamp(clip.duration))
                        detailRow("Model", clip.modelName)
                        if let fuller = clip.fullerArrangement {
                            detailRow("Fuller arrangement", fuller ? "On" : "Off")
                        }
                        detailRow("Made", clip.createdAt.formatted(date: .abbreviated, time: .shortened))
                    }
                }
                actions
            }
            .padding(18)
        }
        .background(Studio.bg.ignoresSafeArea())
        .navigationTitle("Song")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear { player?.stop() }
        .confirmationDialog("Delete this song?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                player?.stop()
                onDelete?()
                dismiss()
            }
        }
    }

    private var playerCard: some View {
        TimelineView(.periodic(from: .now, by: 0.1)) { _ in
            let playing = player?.isPlaying ?? false
            let position = (player?.duration ?? 0) > 0 ? player!.currentTime / player!.duration : 0
            HStack(spacing: 12) {
                Button { toggle() } label: {
                    Image(systemName: playing ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 48))
                        .foregroundColor(Studio.accent)
                }
                .buttonStyle(PressableButtonStyle())
                .accessibilityLabel(playing ? "Pause" : "Play")
                Group {
                    if let envelope = clip.envelope(limit: 80) {
                        WaveformView(buckets: envelope, progress: position, onScrub: { fraction in
                            if player == nil { prepare() }
                            if let player { player.currentTime = fraction * player.duration }
                        })
                    } else {
                        WaveformPlaceholder()
                    }
                }
                .frame(height: 48)
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(Studio.panel))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(Studio.rule, lineWidth: 0.5))
    }

    private var actions: some View {
        VStack(spacing: 10) {
            if let onReuse {
                Button(action: onReuse) {
                    Label("Use this prompt again", systemImage: "arrow.uturn.backward")
                        .font(Studio.text(15, weight: .semibold))
                        .foregroundColor(Studio.onAccent)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 13)
                        .background(RoundedRectangle(cornerRadius: 13, style: .continuous).fill(Studio.accent))
                }
                .buttonStyle(PressableButtonStyle())
            }
            HStack(spacing: 10) {
                Button {
                    UIPasteboard.general.string = clip.lyrics.map { clip.prompt + "\n\n" + $0 } ?? clip.prompt
                    copied = true
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .frame(maxWidth: .infinity)
                }
                if let url = clip.audioURL {
                    ShareLink(item: url) {
                        Label("Share", systemImage: "square.and.arrow.up")
                            .frame(maxWidth: .infinity)
                    }
                }
                if onDelete != nil {
                    Button(role: .destructive) { confirmDelete = true } label: {
                        Label("Delete", systemImage: "trash")
                            .frame(maxWidth: .infinity)
                    }
                }
            }
            .font(Studio.text(13, weight: .medium))
            .foregroundColor(Studio.accent)
            .padding(.vertical, 4)
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            StudioLabel(text: title)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func detailRow(_ name: String, _ value: String) -> some View {
        HStack {
            Text(name).foregroundColor(Studio.mute)
            Spacer()
            Text(value).foregroundColor(Studio.ink)
        }
        .font(Studio.text(13))
    }

    private func prepare() {
        guard let url = clip.audioURL, FileManager.default.fileExists(atPath: url.path) else { return }
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
        player = try? AVAudioPlayer(contentsOf: url)
    }

    private func toggle() {
        if player?.isPlaying == true { player?.pause(); return }
        if player == nil { prepare() }
        player?.play()
    }
}

extension SavedMusic {
    /// The first sung line, skipping section tags.
    var firstLyricLine: String? {
        lyrics?.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { !$0.isEmpty && !MusicRequest.isHeading($0) }
    }

    /// "0:45 · ACE-Step 1.5 · 2h ago".
    var summary: String {
        [TranscriptExporter.formatTimestamp(duration), modelName,
         MusicRow.relativeFormatter.localizedString(for: createdAt, relativeTo: Date())]
            .joined(separator: " · ")
    }
}
