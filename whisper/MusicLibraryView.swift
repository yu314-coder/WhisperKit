import SwiftUI
import SwiftData
import AVFoundation

/// Everything the Music tab has generated, oldest pushed to the bottom.
struct MusicLibraryView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \SavedMusic.createdAt, order: .reverse) private var clips: [SavedMusic]

    @State private var player: AVAudioPlayer?
    @State private var playingID: UUID?

    var body: some View {
        NavigationStack {
            Group {
                if clips.isEmpty { emptyState } else { list }
            }
            .background(Studio.bg.ignoresSafeArea())
            .navigationTitle("Generated Music")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { stop(); dismiss() }
                }
            }
        }
        .onDisappear { stop() }
    }

    private var list: some View {
        List {
            ForEach(clips) { clip in
                MusicRow(clip: clip, isPlaying: playingID == clip.id) { toggle(clip) }
                    .listRowInsets(EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14))
                    .listRowBackground(Studio.bg)
                    .listRowSeparatorTint(Studio.rule)
                    .swipeActions {
                        Button(role: .destructive) { delete(clip) } label: {
                            Label("Delete", systemImage: "trash")
                        }
                        if let url = clip.audioURL {
                            ShareLink(item: url) { Label("Share", systemImage: "square.and.arrow.up") }
                                .tint(Studio.accent)
                        }
                    }
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .background(Studio.bg)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "music.note.list")
                .font(.system(size: 40, weight: .light))
                .foregroundColor(Studio.mute)
            Text("Nothing generated yet")
                .font(Studio.text(17, weight: .semibold))
                .foregroundColor(Studio.ink)
            Text("Describe a piece of music on the Music tab and it will be kept here.")
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

/// One clip: its measured waveform, the prompt that made it, and how it was made.
struct MusicRow: View {
    let clip: SavedMusic
    let isPlaying: Bool
    let onToggle: () -> Void

    private static let relativeFormatter: RelativeDateTimeFormatter = {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .abbreviated
        return f
    }()

    var body: some View {
        HStack(spacing: 14) {
            Button(action: onToggle) {
                Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 30))
                    .foregroundColor(Studio.accent)
            }
            .buttonStyle(PressableButtonStyle())

            Group {
                if let envelope = clip.envelope(limit: 34) {
                    WaveformView(buckets: envelope, progress: nil, idleColor: Studio.mute.opacity(0.55))
                } else {
                    WaveformPlaceholder()
                }
            }
            .frame(width: 74, height: 32)

            VStack(alignment: .leading, spacing: 4) {
                Text(clip.prompt)
                    .font(Studio.text(14, weight: .medium))
                    .foregroundColor(Studio.ink)
                    .lineLimit(2)
                HStack(spacing: 8) {
                    Text(TranscriptExporter.formatTimestamp(clip.duration))
                    Text(clip.modelName)
                    Text(Self.relativeFormatter.localizedString(for: clip.createdAt, relativeTo: Date()))
                }
                .font(Studio.mono(9))
                .foregroundColor(Studio.mute)
            }
            Spacer(minLength: 0)
        }
        .contentShape(Rectangle())
    }
}
