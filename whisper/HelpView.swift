import SwiftUI

/// How to use a tab, behind the "?" in its top bar.
struct HelpView: View {
    enum Topic { case transcribe, live, music, image }
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
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var title: String {
        switch topic {
        case .transcribe: return "How to Transcribe"
        case .live:       return "Live Transcription"
        case .music:      return "How to Make Music"
        case .image:      return "How to Make Pictures"
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
                Section(icon: "bolt.horizontal", title: "Canary 1B v2",
                        body: "Under the Whisper models: NVIDIA's model for English and 24 other European languages, which makes the fewest mistakes in English. It doesn't know Chinese, Japanese or Korean. It runs on the GPU. Licence: CC BY 4.0 (NVIDIA; converted to MLX by Mediform)."),
                Section(icon: "captions.bubble", title: "Live",
                        body: "To see the words while you speak, use the Live tab."),
                Section(icon: "cpu", title: "GPU or Neural Engine",
                        body: "In the same menu, under Run on. Automatic, the default, starts on the GPU and prepares the Neural Engine in the background; after that the model loads in a few seconds and uses less battery. You can also pick either one. The GPU rebuilds the model each time the app opens. The Neural Engine's first load after an update takes a few minutes. Big models can be too large for an iPhone's GPU; if the app closes while loading, switch to the Neural Engine."),
                Section(icon: "mic.fill", title: "Record",
                        body: "Tap the red button to start, and again to stop. The recording is transcribed right away."),
                Section(icon: "folder", title: "Import a file or video",
                        body: "The folder button opens Files; the photo button picks a video from Photos. Audio and video both work."),
                Section(icon: "square.and.arrow.up", title: "Share to Whisper",
                        body: "In another app — Voice Memos, Files, Mail — share an audio or video file and choose Whisperkit. It opens here and starts transcribing."),
                Section(icon: "link", title: "Transcribe a link",
                        body: "The link button takes a web address that leads to an audio or video file (.mp3, .m4a, .mp4…) and downloads it. YouTube, Instagram, Facebook and similar sites can't be downloaded; save your own videos to Photos or Files first."),
                Section(icon: "globe", title: "Language",
                        body: "The language button chooses the language to write in. If the speech is in another, it is transcribed and then translated into the one chosen, on this device; iOS downloads a language the first time, after asking. Auto-detect writes what is said as it is said. For Chinese, choose Traditional to get traditional characters."),
                Section(icon: "person.wave.2", title: "Background noise",
                        body: "The chip above the record button: Natural records everything the mic hears, best for a room, a lecture or a video playing nearby. Reduce noise turns the background down for one voice close to the phone; while it records, the chip also opens iOS's Voice Isolation. You can switch mid-recording, and the choice is shared with the Live tab."),
                Section(icon: "rectangle.on.rectangle", title: "Leave the app while it works",
                        body: "Transcriptions and downloads keep going when you switch to another app, and iOS shows their progress. You get a notification when a transcription is done. (Before iOS 26, work pauses soon after you leave and continues when you come back.)"),
                Section(icon: "books.vertical", title: "Library",
                        body: "Every transcript is kept in the library, where you can search them. Open one to read it, rename it, or export it as text, subtitles (SRT, VTT), Markdown or JSON."),
                Section(icon: "lock.fill", title: "Private",
                        body: "Transcription happens entirely on this device. Your audio never leaves it."),
            ]
        case .live:
            return [
                Section(icon: "captions.bubble", title: "Speak and read along",
                        body: "Tap Start and talk: the words appear as you say them, about half a second behind. Tap Stop to finish. The transcript and the recording are kept in the library, the same one the Transcribe tab uses."),
                Section(icon: "square.and.arrow.down", title: "The model",
                        body: "Live uses NVIDIA's Nemotron 3.5 ASR, made for transcribing as speech happens. It downloads once (0.76 GB) and runs on the GPU. It is less accurate than Whisper Turbo, especially in Chinese, so for recordings where every word matters, record in the Transcribe tab instead."),
                Section(icon: "globe", title: "Language",
                        body: "The globe button sets the language to write in. Speak another and each sentence is translated into it as you finish it, on this device; the words still being said show fainter until then. Auto-detect writes what you say as you say it, and follows a switch from one language to another. Choose Chinese (Traditional) for traditional characters."),
                Section(icon: "person.wave.2", title: "Background noise",
                        body: "The chip above Start: Natural, or Reduce noise for one voice in a noisy place. It can be switched while you speak, and is shared with the Transcribe tab."),
                Section(icon: "rectangle.on.rectangle", title: "Leaving the app",
                        body: "The recording carries on if you switch apps; the words catch up when you come back."),
                Section(icon: "doc.text", title: "Licence",
                        body: "Nemotron 3.5 ASR: NVIDIA, OpenMDW 1.1; converted to MLX by mlx-community. Its licence text downloads with it."),
                Section(icon: "lock.fill", title: "Private",
                        body: "Everything happens on this device. Your voice never leaves it."),
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
        case .image:
            return [
                Section(icon: "flask", title: "Pictures are in beta",
                        body: "It works, and it keeps improving. Every picture is different — generate again for another take."),
                Section(icon: "text.cursor", title: "1. Describe the picture",
                        body: "Say what is in it and how it looks: \"a lighthouse on a cliff at sunset, oil painting\". The style buttons add a look. Real people, places and news events all work — the app doesn't filter topics."),
                Section(icon: "wand.and.stars", title: "2. Generate",
                        body: "The first time, the model downloads (1 to 17 GB, shown in the model menu), and the first picture also prepares the chip it runs on, which takes longer. After that DreamShaper takes a few seconds, SANA-Sprint a little longer, FLUX.2 klein and Z-Image Turbo a minute or more, and Qwen-Image 2.1 Turbo a few minutes."),
                Section(icon: "square.stack.3d.up", title: "Models",
                        body: "Tap the model name at the top. DreamShaper is the fastest and smallest, at 512 × 512. SANA-Sprint makes sharp 1024 × 1024 pictures in two steps; its 1.6B version makes better ones, still in seconds, and shares most of its download. FLUX.2 klein follows long, detailed prompts closely and offers several shapes. Z-Image Turbo makes the most lifelike pictures. The two read prompts with the same reader, which has had its refusals removed, so whichever you download second needs less. Qwen-Image 2.1 Turbo draws the finest detail and lettering, also with a prompt reader that has had its refusals removed; it is the slowest and largest download, then Z-Image and FLUX. Its 4-bit build makes nearly the same pictures from about half the download, and suits devices with less memory. With Enhance prompt on, Qwen-Image's own prompt writer, a Qwen3.5 model with its refusals removed, first turns your words into a detailed description, keeping your names, any text to show and counts, and the picture is made from that; it works with every model, adds about half a minute, and downloads once the first time."),
                Section(icon: "photo.stack", title: "History",
                        body: "Every picture is kept with its prompt. Open one to share it, save it to Photos, or use its prompt again."),
                Section(icon: "tag", title: "Marked as AI-made",
                        body: "Each picture's file says it was made by AI (the IPTC \"trained algorithmic media\" tag that Photos, news desks and social networks read). Nothing is drawn on the picture."),
                Section(icon: "checkmark.shield", title: "What you may not make",
                        body: "The models' licences, which you agree to by using them, forbid using them to break the law; to harm, exploit or sexualise minors; to defame or harass people; to make false information meant to harm others, or to pass AI pictures off as real in order to deceive; and to sexualise real people without consent. Every finished picture passes a check on this device, and one that looks sexually explicit isn't shown or saved; prompts themselves are never filtered. Licences: DreamShaper 8 and Stable Diffusion 1.5, CreativeML OpenRAIL-M; Hyper-SD, CreativeML OpenRAIL++-M; SANA-Sprint, FLUX.2 klein, Z-Image-Turbo, Qwen3 (and its abliterated build) and the picture check (Falconsai), Apache 2.0; Gemma (an abliterated build, its refusals removed), the Gemma Terms of Use (ai.google.dev/gemma/terms) and its Prohibited Use Policy; Qwen-Image 2.1 Turbo, the Qwen RESEARCH LICENSE AGREEMENT, for non-commercial research and evaluation only (its text downloads with the model) — Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved. Its prompt reader, Qwen3-VL (abliterated build), Apache 2.0. The Enhance prompt writer, Qwen-Image 2.1 PE-T2I (abliterated build), the same Qwen RESEARCH LICENSE AGREEMENT."),
                Section(icon: "lock.fill", title: "Private",
                        body: "Pictures are made entirely on this device. Nothing you type leaves it."),
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
