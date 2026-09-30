import SwiftUI

/// How to use a tab, behind the "?" in its top bar.
struct HelpView: View {
    enum Topic { case transcribe, music }
    let topic: Topic

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    ForEach(Array(sections.enumerated()), id: \.offset) { _, section in
                        HStack(alignment: .top, spacing: 14) {
                            Image(systemName: section.icon)
                                .font(.system(size: 17, weight: .medium))
                                .foregroundColor(Studio.accent)
                                .frame(width: 26)
                            VStack(alignment: .leading, spacing: 5) {
                                Text(section.title)
                                    .font(Studio.text(15, weight: .semibold))
                                    .foregroundColor(Studio.ink)
                                Text(section.body)
                                    .font(Studio.text(14))
                                    .foregroundColor(Studio.ink.opacity(0.75))
                                    .lineSpacing(2)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                }
                .padding(20)
                .frame(maxWidth: 640, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            .background(Studio.bg.ignoresSafeArea())
            .navigationTitle(topic == .transcribe ? "How to Transcribe" : "How to Make Music")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private struct Section {
        let icon: String
        let title: String
        let body: String
    }

    private var sections: [Section] {
        switch topic {
        case .transcribe:
            return [
                Section(icon: "square.grid.2x2", title: "Choose a model",
                        body: "Tap the model name at the top. Small is fastest, Turbo is the best balance, and the Large models are the most accurate. A model downloads once, then works offline."),
                Section(icon: "cpu", title: "GPU or Neural Engine",
                        body: "In the same menu, under Run on. The GPU loads in seconds. The Neural Engine uses less battery, but its first load after an update takes a few minutes. Big models can be too large for an iPhone's GPU; if the app closes while loading, switch to the Neural Engine."),
                Section(icon: "mic.fill", title: "Record",
                        body: "Tap the red button to start, and again to stop. The recording is transcribed right away."),
                Section(icon: "folder", title: "Import a file or video",
                        body: "The folder button opens Files; the photo button picks a video from Photos. Audio and video both work."),
                Section(icon: "square.and.arrow.up", title: "Share to Whisper",
                        body: "In another app — Voice Memos, Files, Mail — share an audio or video file and choose Whisperkit. It opens here and starts transcribing."),
                Section(icon: "link", title: "Transcribe a link",
                        body: "The link button takes a web address that leads to an audio or video file (.mp3, .m4a, .mp4…) and downloads it. YouTube, Instagram, Facebook and similar sites can't be downloaded; save your own videos to Photos or Files first."),
                Section(icon: "globe", title: "Language",
                        body: "The language button chooses what is spoken, or detects it. Setting it helps on short or noisy clips. For Chinese, choose Traditional to get traditional characters."),
                Section(icon: "rectangle.on.rectangle", title: "Leave the app while it works",
                        body: "Transcriptions and downloads keep going when you switch to another app, and iOS shows their progress. You get a notification when a transcription is done. (Before iOS 26, work pauses soon after you leave and continues when you come back.)"),
                Section(icon: "books.vertical", title: "Library",
                        body: "Every transcript is kept in the library, where you can search them. Open one to read it, rename it, or export it as text, subtitles (SRT, VTT), Markdown or JSON."),
                Section(icon: "lock.fill", title: "Private",
                        body: "Transcription happens entirely on this device. Your audio never leaves it."),
            ]
        case .music:
            return [
                Section(icon: "flask", title: "Music is in beta",
                        body: "It works, and it keeps improving. Results vary from take to take — generate again for a different one."),
                Section(icon: "text.cursor", title: "1. Describe the music",
                        body: "Genre, mood, instruments, voice: \"a warm acoustic ballad with a female singer\". The style buttons fill in a starting point. Say how long it should be — \"90 seconds\" or \"2:30\" — otherwise it makes 30 seconds. A tempo (\"120 BPM\") or key (\"in C minor\") works too."),
                Section(icon: "music.mic", title: "2. Add lyrics, or don't",
                        body: "Type the words in the Lyrics box, or right in the description (lyric is \"…\" works). Put a blank line between verses; a verse you repeat is sung as the chorus. No lyrics makes an instrumental. The sung language follows the lyrics, or pick one."),
                Section(icon: "sparkles", title: "Fuller arrangement",
                        body: "On by default: adds bass, drums, pads and a build after your words, so a short prompt still sounds full. Turn it off to use your words alone."),
                Section(icon: "wand.and.stars", title: "3. Generate",
                        body: "The first time, the model downloads (several GB). The first song at a new length also prepares the Neural Engine, which takes longer; after that it's quicker."),
                Section(icon: "rectangle.on.rectangle", title: "Leave the app while it works",
                        body: "Music keeps being made when you switch apps, on iPhones and iPads that let apps use the graphics chip in the background; iOS shows the progress and you get a notification when it's ready. On other devices it pauses while you're away and carries on when you come back."),
                Section(icon: "square.stack.3d.up", title: "Models",
                        body: "Tap the model name at the top. ACE-Step 1.5 runs on the Neural Engine and is the fastest. XL sounds richer and needs more memory. XL Full is the largest; on devices with less memory it is slower."),
                Section(icon: "clock.arrow.circlepath", title: "History",
                        body: "Every song is kept with its prompt and lyrics. Tap one to play it, read everything, share it, or use its prompt again."),
                Section(icon: "lock.fill", title: "Private",
                        body: "Music is made entirely on this device. Nothing you type leaves it."),
            ]
        }
    }
}

/// The "?" button in a tab's top bar.
struct HelpButton: View {
    let topic: HelpView.Topic
    @State private var showing = false

    var body: some View {
        Button { showing = true } label: {
            Image(systemName: "questionmark")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(Studio.ink.opacity(0.75))
                .frame(width: 32, height: 32)
                .overlay(Circle().strokeBorder(Studio.rule, lineWidth: 0.5))
        }
        .buttonStyle(PressableButtonStyle())
        .accessibilityLabel("Help")
        .sheet(isPresented: $showing) {
            HelpView(topic: topic).presentationSizing(.page)
        }
    }
}
