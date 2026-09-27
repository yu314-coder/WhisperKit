import SwiftUI
import SwiftData
import AVFoundation

/// The Music tab: a prompt, a version of ACE-Step 1.5, and a generated clip.
struct MusicView: View {
    @State private var prompt: String = ""
    @State private var lyrics: String = ""
    @FocusState private var promptFocused: Bool
    @FocusState private var lyricsFocused: Bool
    @AppStorage("musicModel") private var selectedModel: MusicModel = .aceStep15
    @State private var showModelPicker = false
    @State private var engine = MusicEngine()
    @State private var showLibrary = false
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \SavedMusic.createdAt, order: .reverse) private var clips: [SavedMusic]
    @State private var player: AVAudioPlayer?
    @State private var isPlaying = false
    @State private var playingClipID: UUID?
    @State private var memory = MusicMemoryMonitor()
    @State private var interruptionDismissed = false
    /// The language the lyrics are sung in. ACE-Step conditions on it; an
    /// instrumental ignores it and sends "unknown" instead.
    @AppStorage("musicVocalLanguage") private var vocalLanguage =
        Languages.preferred(among: Languages.aceStepCodes)
    private static let vocalLanguages = Languages.sortedByName(Languages.aceStepCodes)

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Rectangle().fill(Studio.rule).frame(height: 1)
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    if MusicRunMarker.wasInterrupted && !interruptionDismissed { interruptedCard }
                    promptCard
                    if selectedModel.supportsLyrics { lyricsCard }
                    if engine.isBusy || engine.lastResult != nil || engine.failureMessage != nil {
                        progressCard
                    }
                    // Shown while working and kept afterwards, so the peak a
                    // run reached is still readable once it finishes.
                    if engine.isBusy || memory.peakMB > 0 {
                        MusicMemoryGauge(monitor: memory)
                    }
                    statusCard
                    if !clips.isEmpty { recentSection }
                }
                .padding(18)
            }
            // Two text fields now sit above Generate, and the keyboard covers
            // it. Dragging the content dismisses the keyboard, and the
            // toolbar gives a deliberate way out.
            .scrollDismissesKeyboard(.interactively)
            generateBar
        }
        .background(Studio.bg.ignoresSafeArea())
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") { promptFocused = false; lyricsFocused = false }
            }
        }
        .onAppear { engine.modelContext = modelContext }
        .onDisappear { memory.stop() }
        .onChange(of: engine.isBusy) { _, busy in
            if !busy { memory.stop() }
        }
        .sheet(isPresented: $showModelPicker) {
            MusicModelPicker(selected: $selectedModel)
                .presentationSizing(.page)
        }
        .sheet(isPresented: $showLibrary) {
            MusicLibraryView().presentationSizing(.page)
        }
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "music.note")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(Studio.accent)
                Text("music")
                    .font(Studio.mono(13, weight: .semibold))
                    .foregroundColor(Studio.ink)
            }
            Spacer()
            Button { showModelPicker = true } label: {
                HStack(spacing: 5) {
                    Circle()
                        .fill(selectedModel.isRunnable ? Studio.ok : Studio.mute.opacity(0.5))
                        .frame(width: 6, height: 6)
                    Text(selectedModel.displayName)
                        .font(Studio.mono(11))
                        .foregroundColor(Studio.ink.opacity(0.82))
                        .lineLimit(1)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .overlay(Capsule().strokeBorder(Studio.rule, lineWidth: 0.5))
            }
            .buttonStyle(PressableButtonStyle())

            Button { showLibrary = true } label: {
                Image(systemName: "music.note.list")
                    .font(.system(size: 16))
                    .foregroundColor(Studio.ink.opacity(0.75))
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(PressableButtonStyle())
            .padding(.leading, 8)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }

    /// The last few clips, so a generated piece is one tap away rather than
    /// behind a sheet.
    private var recentSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                StudioLabel(text: "Generated")
                Spacer()
                Button { showLibrary = true } label: {
                    Text("All \(clips.count) \u{2192}")
                        .font(Studio.mono(9, weight: .semibold))
                        .foregroundColor(Studio.accent)
                }
            }
            ForEach(clips.prefix(3)) { clip in
                MusicRow(clip: clip, isPlaying: playingClipID == clip.id) { toggleClip(clip) }
            }
        }
    }

    private func toggleClip(_ clip: SavedMusic) {
        if playingClipID == clip.id {
            player?.stop(); playingClipID = nil; return
        }
        guard let url = clip.audioURL, FileManager.default.fileExists(atPath: url.path) else { return }
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
        player = try? AVAudioPlayer(contentsOf: url)
        player?.play()
        playingClipID = clip.id
        isPlaying = false
    }

    // MARK: - Cards

    /// The last run never finished: the app was closed mid-generation. Said
    /// here rather than on the Transcribe tab, which used to take the blame
    /// for it when a model happened to be loading at the same time.
    private var interruptedCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Last generation didn't finish", systemImage: "exclamationmark.triangle")
                .font(Studio.text(14, weight: .semibold))
                .foregroundColor(Studio.hot)
            Text("Whisper was closed while it was generating. If you didn't close it yourself, iOS stopped it — most likely for memory. A shorter piece needs less.")
                .font(Studio.text(13))
                .foregroundColor(Studio.ink)
                .fixedSize(horizontal: false, vertical: true)
            Button("Dismiss") { interruptionDismissed = true }
                .font(Studio.text(13, weight: .medium))
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Studio.sunk))
    }

    private var promptCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            StudioLabel(text: "Prompt")
            ZStack(alignment: .topLeading) {
                if prompt.isEmpty {
                    Text("A slow lo-fi beat with warm bass and vinyl crackle, 1 minute 30 seconds")
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
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Studio.sunk)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(Studio.rule, lineWidth: 0.5)
            )
            // The length comes from the prompt alone; there is no separate
            // control to disagree with it. This line says what was read —
            // or, when the prompt names no length, what will be used.
            if promptSettings.seconds == nil {
                Label("No length in your prompt — \(Self.formatLength(Self.defaultSeconds)) will be made. Add one, like “90 seconds” or “2:30”.",
                      systemImage: "clock")
                    .font(Studio.mono(10))
                    .foregroundColor(Studio.mute)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !promptSettings.isEmpty {
                Label("From your prompt: \(promptSettingsSummary)", systemImage: "text.badge.checkmark")
                    .font(Studio.mono(10))
                    .foregroundColor(Studio.accent)
            }
        }
    }

    /// Length, tempo, key and meter the prompt itself states.
    private var promptSettings: PromptMetadata { PromptMetadata(parsing: prompt) }

    private var promptSettingsSummary: String {
        let settings = promptSettings
        var parts: [String] = []
        if let seconds = settings.seconds { parts.append(Self.formatLength(Double(seconds))) }
        // Tempo, key and meter go to the planner as given.
        if let bpm = settings.bpm { parts.append("\(bpm) BPM") }
        if let key = settings.keyscale { parts.append(key) }
        if let beats = settings.timeSignature { parts.append(beats == 6 ? "6/8" : "\(beats)/4") }
        return parts.joined(separator: " · ")
    }

    /// Used when the prompt names no length.
    private static let defaultSeconds: Double = 30

    /// The length that will be generated: the prompt's, or the default.
    /// `PromptMetadata` already keeps it within 5 seconds to 6:24, the
    /// models' own range.
    private var effectiveSeconds: Double {
        promptSettings.seconds.map(Double.init) ?? Self.defaultSeconds
    }

    /// Lyrics are conditioning, not a caption: the model sings them, so an
    /// empty box means an instrumental rather than a missing field.
    private var lyricsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                StudioLabel(text: "Lyrics")
                Spacer()
                Menu {
                    Picker("Sung in", selection: $vocalLanguage) {
                        ForEach(Self.vocalLanguages, id: \.self) { code in
                            Text(Languages.displayName(code)).tag(code)
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "globe")
                        Text("Sung in \(Languages.englishName(vocalLanguage))")
                        Image(systemName: "chevron.up.chevron.down")
                    }
                    .font(Studio.mono(10, weight: .semibold))
                    .foregroundColor(lyrics.isEmpty ? Studio.mute : Studio.accent)
                }
                .accessibilityLabel("Vocal language, \(Languages.englishName(vocalLanguage))")
            }
            ZStack(alignment: .topLeading) {
                if lyrics.isEmpty {
                    Text("Leave empty for an instrumental, or write a verse to be sung")
                        .font(Studio.text(14))
                        .foregroundColor(Studio.mute.opacity(0.7))
                        .padding(.top, 8)
                        .padding(.horizontal, 12)
                        .allowsHitTesting(false)
                }
                TextEditor(text: $lyrics)
                    .focused($lyricsFocused)
                    .font(Studio.text(14))
                    .foregroundColor(Studio.ink)
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .frame(minHeight: 78)
            }
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Studio.sunk))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Studio.rule, lineWidth: 0.5))
        }
    }

    private static func formatLength(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        guard total >= 60 else { return "\(total)s" }
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    /// What the selected model can actually do on this device, stated plainly.
    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            StudioLabel(text: "Model")
            switch selectedModel.availability {
            case .ready:
                noteRow(icon: "checkmark.circle",
                        tint: Studio.ok,
                        text: "\(selectedModel.displayName) runs on this device. Weights are \(selectedModel.sizeLabel), downloaded once; versions share most of them.")
            case .needsMoreMemory(let note):
                noteRow(icon: "memorychip", tint: Studio.hot, text: note)
            }
        }
    }

    /// Download and generation share one progress area: from the user's side
    /// it is a single wait, even though the first part only happens once.
    @ViewBuilder
    private var progressCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            switch engine.phase {
            case .downloading(let file, let completed, let total, let fraction, let received, let expected):
                HStack {
                    StudioLabel(text: "Downloading \(completed + 1) of \(total)")
                    Spacer()
                    // A number, not just a bar: a stalled download and a slow
                    // one look identical otherwise.
                    Text("\(Int(fraction * 100))%")
                        .font(Studio.mono(11, weight: .semibold))
                        .foregroundColor(Studio.accent)
                }
                ProgressView(value: fraction).tint(Studio.accent)
                HStack {
                    Text(file).font(Studio.mono(10)).foregroundColor(Studio.mute).lineLimit(1)
                    Spacer()
                    Text("\(received / 1_000_000) / \(expected / 1_000_000) MB")
                        .font(Studio.mono(10))
                        .foregroundColor(Studio.mute)
                }
                if let note = engine.downloadNote {
                    Label(note, systemImage: "arrow.clockwise")
                        .font(Studio.mono(10))
                        .foregroundColor(Studio.hot)
                }
            case .generating(let stage):
                StudioLabel(text: "Generating")
                ProgressView().tint(Studio.accent)
                Text(stage).font(Studio.mono(10)).foregroundColor(Studio.mute)
            case .failed(let message):
                StudioLabel(text: "Failed")
                Text(message)
                    .font(Studio.text(13))
                    .foregroundColor(Studio.hot)
                    .fixedSize(horizontal: false, vertical: true)
            case .idle:
                if let url = engine.lastResult {
                    HStack {
                        StudioLabel(text: "Result")
                        Spacer()
                        Text("\(engine.elapsedMilliseconds) ms")
                            .font(Studio.mono(10))
                            .foregroundColor(Studio.mute)
                    }
                    HStack(spacing: 12) {
                        Button { togglePlayback(url) } label: {
                            Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                                .font(.system(size: 38))
                                .foregroundColor(Studio.accent)
                        }
                        .buttonStyle(PressableButtonStyle())
                        ShareLink(item: url) {
                            Image(systemName: "square.and.arrow.up")
                                .font(.system(size: 17))
                                .foregroundColor(Studio.ink.opacity(0.75))
                        }
                        Spacer()
                    }
                }
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Studio.sunk))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
            .strokeBorder(Studio.rule, lineWidth: 0.5))
    }

    private func togglePlayback(_ url: URL) {
        if isPlaying { player?.pause(); isPlaying = false; return }
        if player?.url != url {
            try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try? AVAudioSession.sharedInstance().setActive(true)
            player = try? AVAudioPlayer(contentsOf: url)
        }
        player?.play()
        isPlaying = true
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

    // MARK: - Generate

    private var canGenerate: Bool {
        engine.isBusy
            || (selectedModel.isRunnable && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    private var buttonTitle: String {
        if engine.isBusy { return "Cancel" }
        guard selectedModel.isRunnable else { return "\(selectedModel.displayName) needs more memory" }
        return "Generate"
    }

    private var generateBar: some View {
        VStack(spacing: 0) {
            Rectangle().fill(Studio.rule).frame(height: 1)
            Button {
                if engine.isBusy {
                    engine.cancel()
                } else {
                    isPlaying = false
                    player?.stop()
                    // Started here rather than from a change in `isBusy`: a run
                    // that fails immediately never lets the view observe the
                    // busy state, and the gauge would never appear.
                    memory.start()
                    engine.generate(model: selectedModel, prompt: prompt,
                                    lyrics: selectedModel.supportsLyrics ? lyrics : "",
                                    language: vocalLanguage,
                                    seconds: effectiveSeconds)
                }
            } label: {
                Text(buttonTitle)
                    .font(Studio.text(15, weight: .semibold))
                    .foregroundColor(canGenerate ? Studio.onAccent : Studio.mute)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 15)
                    .background(
                        RoundedRectangle(cornerRadius: 13, style: .continuous)
                            .fill(canGenerate ? Studio.accent : Studio.sunk)
                    )
            }
            .buttonStyle(PressableButtonStyle())
            .disabled(!canGenerate)
            .padding(16)
        }
    }
}

// MARK: - Model picker

struct MusicModelPicker: View {
    @Binding var selected: MusicModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 10) {
                    ForEach(MusicModel.allCases) { model in
                        Button { selected = model; dismiss() } label: { card(model) }
                            .buttonStyle(PressableButtonStyle())
                    }
                }
                .padding(16)
            }
            .background(Studio.bg.ignoresSafeArea())
            .navigationTitle("Music Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func card(_ model: MusicModel) -> some View {
        VStack(alignment: .leading, spacing: 7) {
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

            HStack(spacing: 7) {
                chip(model.sizeLabel, tint: Studio.mute)
                switch model.availability {
                case .ready:           chip("RUNS ON DEVICE", tint: Studio.ok)
                case .needsMoreMemory: chip("NEEDS 8 GB", tint: Studio.hot)
                }
                chip(model.engineLabel, tint: Studio.mute)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(selected == model ? Studio.panel : Studio.sunk)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(selected == model ? Studio.accent : Studio.rule,
                              lineWidth: selected == model ? 1.5 : 0.5)
        )
    }

    private func chip(_ text: String, tint: Color) -> some View {
        Text(text)
            .font(Studio.mono(9, weight: .semibold))
            .foregroundColor(tint)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Capsule().fill(tint.opacity(0.12)))
    }
}
