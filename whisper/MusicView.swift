import SwiftUI
import SwiftData
import AVFoundation

/// The Music tab: describe the music, add lyrics if it should be sung,
/// generate, and play what comes back.
struct MusicView: View {
    @State private var prompt: String = ""
    @State private var lyrics: String = ""
    @FocusState private var promptFocused: Bool
    @FocusState private var lyricsFocused: Bool
    @AppStorage("musicModel") private var selectedModel: MusicModel = .aceStep15
    /// Let the planner turn the prompt into a full arrangement first. On by
    /// default: a short prompt otherwise gets one instrument, thin next to
    /// what Suno makes of the same words.
    @AppStorage("musicExpandsPrompt") private var expandsPrompt = true
    @State private var showModelPicker = false
    @State private var engine = MusicEngine()
    @State private var showLibrary = false
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \SavedMusic.createdAt, order: .reverse) private var clips: [SavedMusic]
    @State private var player: AVAudioPlayer?
    @State private var playingClipID: UUID?
    @State private var resultEnvelope: [Float]?
    @State private var memory = MusicMemoryMonitor()
    @State private var interruptionDismissed = false
    @State private var showDetails = false
    @State private var openedClip: SavedMusic?
    /// What the on-device model found in a prompt: the lines it called
    /// lyrics, or nil when it couldn't answer. See `LyricsFinder`.
    @State private var detected: DetectedLyrics?
    @State private var findingLyrics = false
    private struct DetectedLyrics {
        var prompt: String
        var lines: [String]?
    }
    /// Bumped when a song is used again, to scroll back to the boxes.
    @State private var reuseCount = 0
    private static let top = "top"
    /// The language the lyrics are sung in — ACE-Step conditions on it —
    /// or "auto", read from the lyrics' script. A new key: the old one
    /// held a fixed language, and Chinese lyrics sung as English came out
    /// garbled.
    @AppStorage("musicSungLanguage") private var sungLanguage = Self.autoLanguage
    private static let autoLanguage = "auto"
    private static let vocalLanguages = Languages.sortedByName(Languages.aceStepCodes)
    /// Latin-script languages, for lyrics whose script doesn't say which.
    private static let latinLanguages = ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "sv", "da", "no", "fi",
                                         "cs", "ro", "hu", "tr", "id", "ms", "vi", "ca", "hr", "sk", "lt", "is"]
        .filter(Languages.aceStepCodes.contains)

    /// One tap from an empty box to something that makes good music.
    private static let styles: [(name: String, prompt: String)] = [
        ("Lo-fi", "Warm lo-fi hip hop beat with mellow keys, soft drums and vinyl crackle"),
        ("Cinematic", "Epic cinematic orchestral score with soaring strings, brass and big drums"),
        ("Acoustic", "Warm acoustic folk song with fingerpicked guitar and soft percussion"),
        ("Pop", "Upbeat modern pop with bright synths, punchy drums and a catchy groove"),
        ("Jazz", "Smooth jazz trio with piano, upright bass and brushed drums"),
        ("Ambient", "Calm ambient soundscape with soft pads, gentle textures and slow evolving chords"),
        ("Rock", "Energetic rock anthem with driving electric guitars, bass and big drums"),
        ("EDM", "Festival EDM track with a pulsing bassline, bright leads and a big drop"),
    ]

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Rectangle().fill(Studio.rule).frame(height: 1)
            ScrollViewReader { scroller in
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Color.clear.frame(height: 0).id(Self.top)
                    if MusicRunMarker.wasInterrupted && !interruptionDismissed { interruptedCard }
                    promptCard
                    lyricsCard
                    if let warning = selectedModel.memoryWarning {
                        // Advice about the user's own choice, so not in the
                        // alarm colour: in red it read as "not allowed".
                        noteRow(icon: "memorychip", tint: Studio.mute, text: warning)
                            .padding(.horizontal, 4)
                    }
                    if engine.isBusy || engine.failureMessage != nil {
                        progressCard
                    } else if engine.lastResult != nil {
                        resultCard
                    }
                    // Live while working: how much memory the models hold.
                    if engine.isBusy { MusicMemoryGauge(monitor: memory) }
                    if !clips.isEmpty { recentSection }
                    detailsSection
                }
                .padding(18)
            }
            // Two text fields sit above Generate, and the keyboard covers
            // it. Dragging the content dismisses the keyboard, and the
            // toolbar gives a deliberate way out.
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
                Button("Done") { promptFocused = false; lyricsFocused = false }
            }
        }
        .onAppear {
            engine.modelContext = modelContext
            #if DEBUG
            // For testing downloads in the Simulator, whose keyboard covers
            // Generate: launch with `-MusicPromptPreset "…"`.
            if prompt.isEmpty, let preset = UserDefaults.standard.string(forKey: "MusicPromptPreset") {
                prompt = preset
            }
            // History can't be made in the Simulator, which can't generate:
            // `-MusicSeedHistory YES` adds two songs to look at.
            if clips.isEmpty, UserDefaults.standard.bool(forKey: "MusicSeedHistory") {
                modelContext.insert(SavedMusic(
                    prompt: "Soft piano ballad with a female singer, strings in the chorus, 1 minute 30 seconds",
                    createdAt: Date().addingTimeInterval(-3600), duration: 90, modelName: "ACE-Step 1.5 XL",
                    waveform: (0 ..< 120).map { _ in Float.random(in: 0.2 ... 1) },
                    lyrics: "[Verse]\nWalking down the empty street tonight\nCity lights are shining oh so bright\n\n[Chorus]\nHold on to me, don't let go\nWe will find our way back home",
                    language: "en", fullerArrangement: true))
                modelContext.insert(SavedMusic(
                    prompt: "Warm lo-fi hip hop beat with mellow keys, soft drums and vinyl crackle",
                    createdAt: Date().addingTimeInterval(-86400), duration: 30, modelName: "ACE-Step 1.5",
                    waveform: (0 ..< 120).map { _ in Float.random(in: 0.1 ... 0.8) },
                    fullerArrangement: true))
            }
            #endif
        }
        .onDisappear { memory.stop() }
        .onChange(of: engine.isBusy) { _, busy in
            if !busy { memory.stop() }
        }
        // Once typing pauses, ask which words are lyrics.
        .task(id: prompt) { await findLyrics(in: prompt, afterPause: true) }
        .onChange(of: engine.lastResult) { _, url in
            resultEnvelope = url.flatMap { AudioConverter.peakEnvelope(of: $0, bucketCount: 90)?.buckets }
        }
        .sheet(isPresented: $showModelPicker) {
            MusicModelPicker(selected: $selectedModel)
                .presentationSizing(.page)
        }
        .sheet(isPresented: $showLibrary) {
            MusicLibraryView(onReuse: reuse).presentationSizing(.page)
        }
        .sheet(item: $openedClip) { clip in
            NavigationStack {
                MusicClipDetail(clip: clip,
                                onReuse: { reuse(clip); openedClip = nil },
                                onDelete: { delete(clip) })
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) {
                            Button("Done") { openedClip = nil }
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
                        .fill(Studio.ok)
                        .frame(width: 6, height: 6)
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
            .accessibilityLabel("Model, \(selectedModel.displayName)")

            Button { showLibrary = true } label: {
                Image(systemName: "music.note.list")
                    .font(.system(size: 16))
                    .foregroundColor(Studio.ink.opacity(0.75))
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(PressableButtonStyle())
            .padding(.leading, 8)
            .accessibilityLabel("Library")
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

    private func editor(_ text: Binding<String>, placeholder: String, focus: FocusState<Bool>.Binding,
                        minHeight: CGFloat, size: CGFloat) -> some View {
        ZStack(alignment: .topLeading) {
            if text.wrappedValue.isEmpty {
                Text(placeholder)
                    .font(Studio.text(size))
                    .foregroundColor(Studio.mute.opacity(0.7))
                    .padding(.top, 8)
                    .padding(.horizontal, 12)
                    .allowsHitTesting(false)
            }
            TextEditor(text: text)
                .focused(focus)
                .font(Studio.text(size))
                .foregroundColor(Studio.ink)
                .scrollContentBackground(.hidden)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .frame(minHeight: minHeight)
        }
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Studio.sunk))
    }

    private var promptCard: some View {
        card {
            HStack(spacing: 14) {
                StudioLabel(text: "Describe the music")
                Spacer()
                if !prompt.isEmpty {
                    Button("Clear") { prompt = "" }
                        .font(Studio.mono(10, weight: .semibold))
                        .foregroundColor(Studio.mute)
                }
                if !clips.isEmpty {
                    Button { showLibrary = true } label: {
                        Label("History", systemImage: "clock.arrow.circlepath")
                            .font(Studio.mono(10, weight: .semibold))
                            .foregroundColor(Studio.accent)
                    }
                    .accessibilityHint("Earlier prompts and lyrics, to play or use again")
                }
            }
            // Multi-line on purpose: return starts a new line, a pasted
            // song keeps its shape, and lyrics typed here are sung.
            editor($prompt, placeholder: "A slow lo-fi beat with warm bass and vinyl crackle, 1 minute 30 seconds\n\nLyrics can go here too, after a blank line",
                   focus: $promptFocused, minHeight: 96, size: 15)

            // Styles: fill an empty box, or add to what is there.
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 7) {
                    ForEach(Self.styles, id: \.name) { style in
                        Button {
                            let current = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                            prompt = current.isEmpty ? style.prompt : "\(current), \(style.name.lowercased())"
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

            // What the prompt set — length always, the rest when stated.
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(settingPills, id: \.self) { pill($0) }
                }
            }

            Toggle(isOn: $expandsPrompt) {
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .foregroundColor(Studio.accent)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Fuller arrangement")
                            .font(Studio.text(13, weight: .medium))
                            .foregroundColor(Studio.ink)
                        Text("Adds a full arrangement after your words: bass, drums, pads, how it builds. May bring in instruments you didn't name.")
                            .font(Studio.text(12))
                            .foregroundColor(Studio.mute)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .tint(Studio.accent)
        }
    }

    private func pill(_ text: String) -> some View {
        Text(text)
            .font(Studio.mono(10, weight: .semibold))
            .foregroundColor(Studio.accent)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Capsule().fill(Studio.accent.opacity(0.10)))
    }

    /// Length, tempo, key and meter the prompt itself states; the length is
    /// always shown, since it is the default when the prompt names none.
    private var settingPills: [String] {
        let settings = promptSettings
        var pills: [String] = []
        // First, so it is never scrolled out of sight behind the others.
        let fromPrompt = request.lyricLinesFromPrompt
        if findingLyrics {
            pills.append("✦ Looking for lyrics…")
        } else if fromPrompt > 0 {
            pills.append("♪ \(fromPrompt) line\(fromPrompt == 1 ? "" : "s") of lyrics found — sung")
        }
        if let seconds = settings.seconds {
            pills.append("⏱ \(Self.formatLength(Double(seconds)))")
        } else {
            pills.append("⏱ \(Self.formatLength(Self.defaultSeconds)) · say “90 seconds” or “2:30” to change")
        }
        if let bpm = settings.bpm { pills.append("\(bpm) BPM") }
        if let key = settings.keyscale { pills.append(key) }
        if let beats = settings.timeSignature { pills.append(beats == 6 ? "6/8" : "\(beats)/4") }
        // A timeline is planned part by part; with lyrics, the lyrics lead.
        if !settings.sections.isEmpty && !sendsLyrics {
            pills.append("\(settings.sections.count) timed sections")
        }
        return pills
    }

    /// Length, tempo, key and meter the prompt itself states — read from
    /// the description, not from lyrics pasted after it.
    private var promptSettings: PromptMetadata { PromptMetadata(parsing: request.description) }

    /// The two boxes sorted into a description and lyrics, wherever the
    /// lyrics were typed.
    private var request: MusicRequest {
        MusicRequest(prompt: prompt, lyrics: lyrics,
                     found: detected?.prompt == prompt ? detected?.lines : nil)
    }

    /// Asks the on-device model which words in `text` are lyrics, unless it
    /// already has; the rules in `MusicRequest` answer until it does, and
    /// wherever it can't run.
    private func findLyrics(in text: String, afterPause: Bool) async {
        guard LyricsFinder.isAvailable, LyricsFinder.mightHoldLyrics(text), detected?.prompt != text else { return }
        if afterPause {
            try? await Task.sleep(for: .milliseconds(700))
            guard !Task.isCancelled else { return }
        }
        findingLyrics = true
        let lines = await LyricsFinder.lyricLines(in: text)
        findingLyrics = false
        if afterPause && Task.isCancelled { return }
        detected = DetectedLyrics(prompt: text, lines: lines)
    }

    /// The language to sing in: the one chosen, or the lyrics' own.
    private func sungLanguage(for lyrics: String) -> String {
        guard sungLanguage == Self.autoLanguage else { return sungLanguage }
        return MusicRequest.language(of: lyrics) ?? Languages.preferred(among: Self.latinLanguages, fallback: "en")
    }

    /// Used when the prompt names no length.
    private static let defaultSeconds: Double = 30

    /// The length that will be generated: the prompt's, or the default.
    /// `PromptMetadata` already keeps it within 5 seconds to 6:24.
    private var effectiveSeconds: Double {
        promptSettings.seconds.map(Double.init) ?? Self.defaultSeconds
    }

    private var sendsLyrics: Bool { request.hasLyrics }

    /// Just a place to type the words. Section tags and a vocals switch
    /// made writing lyrics a chore; the tags are now added for the model
    /// (`ACEPipeline.sectioned`), and no lyrics means an instrumental.
    private var lyricsCard: some View {
        card {
            HStack(spacing: 14) {
                StudioLabel(text: "Lyrics · optional")
                Spacer()
                if !lyrics.isEmpty {
                    Button("Clear") { lyrics = "" }
                        .font(Studio.mono(10, weight: .semibold))
                        .foregroundColor(Studio.mute)
                }
                Menu {
                    Picker("Sung in", selection: $sungLanguage) {
                        Text("Automatic, from the lyrics").tag(Self.autoLanguage)
                        Divider()
                        ForEach(Self.vocalLanguages, id: \.self) { code in
                            Text(Languages.displayName(code)).tag(code)
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "globe")
                        Text(languageLabel)
                        Image(systemName: "chevron.up.chevron.down")
                    }
                    .font(Studio.mono(10, weight: .semibold))
                    .foregroundColor(Studio.accent)
                }
                .lineLimit(1)
                .accessibilityLabel("Sung in \(languageLabel)")
            }
            editor($lyrics, placeholder: "Type or paste the words to sing",
                   focus: $lyricsFocused, minHeight: 120, size: 14)
            Text(lyricsHint)
                .font(Studio.text(12))
                .foregroundColor(Studio.mute)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var languageLabel: String {
        let language = Languages.englishName(sungLanguage(for: request.lyrics))
        return sungLanguage == Self.autoLanguage
            ? (sendsLyrics ? "\(language) · auto" : "Language · auto")
            : language
    }

    private var lyricsHint: String {
        if request.lyricLinesFromPrompt > 0 && lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "The lyrics in your prompt will be sung. They can go here instead — either works."
        }
        return sendsLyrics
            ? "Put a blank line between verses. A verse you repeat is sung as the chorus."
            : "Leave empty for an instrumental, or put lyrics in the prompt above."
    }

    // MARK: - Progress and result

    /// Download and generation share one progress area: from the user's side
    /// it is a single wait, even though the first part only happens once.
    @ViewBuilder
    private var progressCard: some View {
        card {
            switch engine.phase {
            case .downloading(let file, let completed, let total, let fraction, let received, let expected):
                HStack {
                    // What is missing, not everything the model uses: files
                    // already here from an earlier version are not counted.
                    StudioLabel(text: total == 1 ? "Downloading" : "Downloading file \(completed + 1) of \(total)")
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
                    Text("\(Self.formatBytes(received)) / \(Self.formatBytes(expected))")
                        .font(Studio.mono(10))
                        .foregroundColor(Studio.mute)
                }
                if let note = engine.downloadNote {
                    Label(note, systemImage: note == "Unpacking" ? "shippingbox" : "arrow.clockwise")
                        .font(Studio.mono(10))
                        .foregroundColor(Studio.hot)
                }
            case .generating(let stage, let progress):
                HStack {
                    StudioLabel(text: "Making your music")
                    Spacer()
                    Text("\(Int(progress * 100))%")
                        .font(Studio.mono(11, weight: .semibold))
                        .foregroundColor(Studio.accent)
                }
                ProgressView(value: progress).tint(Studio.accent)
                HStack {
                    Text(stage)
                        .font(Studio.mono(10))
                        .foregroundColor(Studio.mute)
                        .lineLimit(2)
                    Spacer()
                    if let started = engine.startedAt {
                        TimelineView(.periodic(from: .now, by: 1)) { context in
                            Text(Self.formatLength(context.date.timeIntervalSince(started)))
                                .font(Studio.mono(10))
                                .foregroundColor(Studio.mute)
                                .monospacedDigit()
                        }
                    }
                }
                stageSteps(progress)
            case .failed(let message):
                StudioLabel(text: "Failed")
                Text(message)
                    .font(Studio.text(13))
                    .foregroundColor(Studio.hot)
                    .fixedSize(horizontal: false, vertical: true)
            case .idle:
                EmptyView()
            }
        }
    }

    /// Plan, write, render, audio — each ticked off as the run passes it.
    private func stageSteps(_ progress: Double) -> some View {
        let steps: [(String, Double)] = [("Plan", 0.02), ("Write", 0.40), ("Render", 0.50), ("Audio", 0.88)]
        return HStack(spacing: 0) {
            ForEach(Array(steps.enumerated()), id: \.offset) { index, step in
                let done = index + 1 < steps.count ? progress >= steps[index + 1].1 : progress >= 1
                let active = !done && progress >= step.1
                HStack(spacing: 4) {
                    Image(systemName: done ? "checkmark.circle.fill" : (active ? "circle.dotted" : "circle"))
                        .font(.system(size: 11))
                    Text(step.0).font(Studio.mono(10, weight: active ? .semibold : .regular))
                }
                .foregroundColor(done || active ? Studio.accent : Studio.mute.opacity(0.7))
                if index + 1 < steps.count { Spacer(minLength: 4) }
            }
        }
    }

    private var resultCard: some View {
        card {
            HStack(alignment: .firstTextBaseline) {
                StudioLabel(text: "Your track")
                Spacer()
                Text("\(Self.formatLength(engine.lastDuration)) · made in \(Self.formatLength(Double(engine.elapsedMilliseconds) / 1000))")
                    .font(Studio.mono(10))
                    .foregroundColor(Studio.mute)
            }
            if !engine.lastPrompt.isEmpty {
                Text(engine.lastPrompt)
                    .font(Studio.text(14, weight: .medium))
                    .foregroundColor(Studio.ink)
                    .lineLimit(3)
            }
            // The words it sang, so a song made from lyrics in the prompt
            // shows them too; all of them are in History.
            let sung = engine.lastLyrics.components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty && !MusicRequest.isHeading($0) }
            if !sung.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "music.mic")
                        .foregroundColor(Studio.accent)
                    Text(sung.prefix(4).joined(separator: "\n") + (sung.count > 4 ? "\n…" : ""))
                        .foregroundColor(Studio.ink.opacity(0.75))
                        .lineSpacing(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .font(Studio.text(13))
            }
            if let url = engine.lastResult {
                TimelineView(.periodic(from: .now, by: 0.1)) { _ in
                    let playing = playingClipID == nil && player?.url == url && (player?.isPlaying ?? false)
                    let position = player?.url == url && (player?.duration ?? 0) > 0
                        ? (player!.currentTime / player!.duration) : 0
                    HStack(spacing: 12) {
                        Button { toggleResult(url) } label: {
                            Image(systemName: playing ? "pause.circle.fill" : "play.circle.fill")
                                .font(.system(size: 44))
                                .foregroundColor(Studio.accent)
                        }
                        .buttonStyle(PressableButtonStyle())
                        .accessibilityLabel(playing ? "Pause" : "Play")
                        Group {
                            if let envelope = resultEnvelope {
                                WaveformView(buckets: envelope, progress: position, onScrub: { fraction in
                                    if player?.url != url { preparePlayer(url) }
                                    if let player { player.currentTime = fraction * player.duration }
                                })
                            } else {
                                WaveformPlaceholder()
                            }
                        }
                        .frame(height: 44)
                    }
                }
                HStack(spacing: 10) {
                    ShareLink(item: url) {
                        Label("Share", systemImage: "square.and.arrow.up")
                            .font(Studio.text(13, weight: .medium))
                    }
                    Spacer()
                    Button { startGenerating() } label: {
                        Label("New take", systemImage: "arrow.triangle.2.circlepath")
                            .font(Studio.text(13, weight: .medium))
                    }
                    .disabled(!canGenerate)
                }
                .foregroundColor(Studio.accent)
            }
        }
    }

    private func preparePlayer(_ url: URL) {
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
        player = try? AVAudioPlayer(contentsOf: url)
        playingClipID = nil
    }

    private func toggleResult(_ url: URL) {
        if player?.url == url, playingClipID == nil, player?.isPlaying == true {
            player?.pause()
            return
        }
        if player?.url != url || playingClipID != nil { preparePlayer(url) }
        player?.play()
    }

    // MARK: - Library and details

    /// The last few clips, so a generated piece is one tap away rather than
    /// behind a sheet.
    private var recentSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                StudioLabel(text: "Recent")
                Spacer()
                Button { showLibrary = true } label: {
                    Text("All \(clips.count) \u{2192}")
                        .font(Studio.mono(9, weight: .semibold))
                        .foregroundColor(Studio.accent)
                }
            }
            ForEach(clips.prefix(3)) { clip in
                Button { openedClip = clip } label: {
                    HStack(alignment: .center, spacing: 6) {
                        MusicRow(clip: clip, isPlaying: playingClipID == clip.id) { toggleClip(clip) }
                        Image(systemName: "chevron.right")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(Studio.mute.opacity(0.7))
                    }
                }
                .buttonStyle(.plain)
            }
        }
    }

    /// Puts a song's prompt, lyrics and settings back in the boxes.
    private func reuse(_ clip: SavedMusic) {
        player?.stop()
        playingClipID = nil
        prompt = clip.prompt
        lyrics = clip.lyrics ?? ""
        if let language = clip.language, language != sungLanguage(for: lyrics) { sungLanguage = language }
        if let fuller = clip.fullerArrangement { expandsPrompt = fuller }
        reuseCount += 1
    }

    private func delete(_ clip: SavedMusic) {
        if playingClipID == clip.id { player?.stop(); playingClipID = nil }
        if let path = clip.audioFilePath { AudioFiles.deleteAudio(relativePath: path) }
        modelContext.delete(clip)
        try? modelContext.save()
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
    }

    /// The model and memory, out of the way until wanted.
    private var detailsSection: some View {
        DisclosureGroup(isExpanded: $showDetails) {
            VStack(alignment: .leading, spacing: 10) {
                noteRow(icon: "checkmark.circle",
                        tint: Studio.ok,
                        text: "\(selectedModel.displayName) runs on this device. Weights are \(selectedModel.sizeLabel), downloaded once; versions share most of them.")
                // Kept after a run, so the peak it reached is still readable.
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

    private static func formatLength(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        guard total >= 60 else { return "\(total)s" }
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    // MARK: - Generate

    private var canGenerate: Bool {
        engine.isBusy || !request.description.isEmpty || request.hasLyrics
    }

    private func startGenerating() {
        player?.stop()
        playingClipID = nil
        promptFocused = false
        lyricsFocused = false
        // Started here rather than from a change in `isBusy`: a run that
        // fails immediately never lets the view observe the busy state, and
        // the gauge would never appear.
        memory.start()
        Task { @MainActor in
            // A prompt edited within the last moment hasn't been read yet.
            await findLyrics(in: prompt, afterPause: false)
            let request = self.request
            // Lyrics with no description: the planner writes one.
            engine.generate(model: selectedModel, prompt: request.description,
                            lyrics: request.lyrics,
                            language: sungLanguage(for: request.lyrics),
                            seconds: effectiveSeconds,
                            expandsPrompt: expandsPrompt || request.description.isEmpty)
        }
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
                         : "Generate \(Self.formatLength(effectiveSeconds)) · \(sendsLyrics ? "with vocals" : "instrumental")")
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
                chip("RUNS ON DEVICE", tint: Studio.ok)
                if model.memoryWarning != nil { chip("SLOWER HERE", tint: Studio.mute) }
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
