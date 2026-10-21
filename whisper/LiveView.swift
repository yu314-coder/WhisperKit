import AVFoundation
import SwiftData
import SwiftUI
import WhisperKit

/// The Live tab: speak, and the words appear as you say them — NVIDIA's
/// Nemotron 3.5 ASR, a streaming model (LiveTranscriber, SpeechRunner).
///
/// A tab of its own because it works unlike the Transcribe tab: nothing to
/// import and no waiting at the end. What it hears is kept like any other
/// transcript, recording included, in the library the two tabs share.
struct LiveView: View {
    @AppStorage("liveLanguage") private var language = "auto"
    @Environment(\.modelContext) private var modelContext
    @Environment(\.scenePhase) private var scenePhase

    @State private var transcriber: LiveTranscriber?
    @State private var text = ""
    @State private var startedAt: Date?
    @State private var finishing = false
    @State private var downloading = false
    @State private var downloadProgress: Double?
    @State private var downloaded = SpeechEngine.nemotron.isDownloaded
    @State private var message: String?
    @State private var saved: SavedTranscript?
    @State private var copied = false
    @State private var showLanguages = false
    @State private var showLibrary = false
    private static let bottom = "bottom"

    private var engine: SpeechEngine { .nemotron }
    private var isRecording: Bool { transcriber != nil }

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Rectangle().fill(Studio.rule).frame(height: 1)
            ScrollViewReader { scroller in
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        if !downloaded { setupCard }
                        transcriptCard
                        if let message { note(message, icon: "exclamationmark.circle") }
                        Color.clear.frame(height: 1).id(Self.bottom)
                    }
                    .padding(18)
                    .frame(maxWidth: 760)
                    .frame(maxWidth: .infinity)
                }
                .onChange(of: text) { _, _ in
                    if isRecording { scroller.scrollTo(Self.bottom, anchor: .bottom) }
                }
            }
            recordBar
        }
        .background(Studio.bg.ignoresSafeArea())
        .sheet(isPresented: $showLanguages) { languageSheet }
        .sheet(isPresented: $showLibrary) { TranscriptLibraryView().presentationSizing(.page) }
        .onAppear { downloaded = engine.isDownloaded }
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "captions.bubble")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(Studio.accent)
                Text("live")
                    .font(Studio.mono(13, weight: .semibold))
                    .foregroundColor(Studio.ink)
            }
            Spacer()
            Button { showLanguages = true } label: {
                HStack(spacing: 5) {
                    Image(systemName: "globe").font(.system(size: 11))
                    Text(languageName(language))
                        .font(Studio.mono(11))
                        .lineLimit(1)
                }
                .foregroundColor(Studio.ink.opacity(0.82))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .overlay(Capsule().strokeBorder(Studio.rule, lineWidth: 0.5))
            }
            .buttonStyle(PressableButtonStyle())
            .disabled(isRecording || finishing)
            .accessibilityLabel("Language, \(languageName(language))")

            Button { showLibrary = true } label: {
                Image(systemName: "books.vertical")
                    .font(.system(size: 16))
                    .foregroundColor(Studio.ink.opacity(0.75))
                    .frame(width: 32, height: 32)
                    .overlay(Circle().strokeBorder(Studio.rule, lineWidth: 0.5))
            }
            .buttonStyle(PressableButtonStyle())
            .padding(.leading, 8)
            .accessibilityLabel("Library")

            HelpButton(topic: .live)
                .padding(.leading, 6)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }

    // MARK: - Cards

    private func card<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12, content: content)
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(Studio.panel))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Studio.rule, lineWidth: 0.5))
    }

    private var setupCard: some View {
        card {
            StudioLabel(text: engine.displayName)
            Text("Live transcription uses NVIDIA's Nemotron 3.5 ASR, which writes while you speak, on the GPU. It downloads once (\(engine.sizeLabel)), then works offline. It is less accurate than Whisper Turbo, especially in Chinese: for recordings where every word matters, use the Transcribe tab.")
                .font(Studio.text(13))
                .foregroundColor(Studio.ink.opacity(0.75))
                .fixedSize(horizontal: false, vertical: true)
            if downloading {
                ProgressView(value: downloadProgress ?? 0).tint(Studio.accent)
                Text(downloadProgress.map { "\(Int($0 * 100))% of \(engine.sizeLabel)" } ?? "Starting…")
                    .font(Studio.mono(10))
                    .foregroundColor(Studio.mute)
            } else {
                Button(action: download) {
                    Label("Download \(engine.sizeLabel)", systemImage: "arrow.down")
                        .font(Studio.text(14, weight: .semibold))
                        .foregroundColor(Studio.onAccent)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Studio.accent))
                }
                .buttonStyle(PressableButtonStyle())
            }
        }
    }

    private var transcriptCard: some View {
        card {
            HStack(spacing: 8) {
                if isRecording {
                    Circle().fill(Studio.hot).frame(width: 7, height: 7)
                    StudioLabel(text: "Listening")
                } else {
                    StudioLabel(text: saved == nil ? "Transcript" : "Saved")
                }
                Spacer()
                if !text.isEmpty, !isRecording {
                    Button {
                        UIPasteboard.general.string = text
                        copied = true
                    } label: {
                        Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                            .font(Studio.mono(10, weight: .semibold))
                            .foregroundColor(Studio.accent)
                    }
                }
            }
            if text.isEmpty {
                Text(isRecording ? "Start speaking…" : "Tap Start and speak. The words appear as you say them, and the recording is kept with its transcript in the library.")
                    .font(Studio.text(15))
                    .foregroundColor(Studio.mute)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text(text)
                    .font(Studio.text(18))
                    .foregroundColor(Studio.ink)
                    .lineSpacing(6)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if finishing {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Finishing…").font(Studio.mono(10)).foregroundColor(Studio.mute)
                }
            } else if saved != nil, !isRecording {
                Text("Kept in the library, with the recording.")
                    .font(Studio.mono(10))
                    .foregroundColor(Studio.mute)
            }
        }
    }

    private func note(_ text: String, icon: String) -> some View {
        Label(text, systemImage: icon)
            .font(Studio.text(13))
            .foregroundColor(Studio.hot)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 4)
    }

    // MARK: - Record

    private var recordBar: some View {
        VStack(spacing: 0) {
            Rectangle().fill(Studio.rule).frame(height: 1)
            Button(action: isRecording ? stop : start) {
                HStack(spacing: 8) {
                    Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                    if isRecording, let startedAt {
                        TimelineView(.periodic(from: .now, by: 1)) { context in
                            Text("Stop · \(Self.clock(context.date.timeIntervalSince(startedAt)))")
                                .monospacedDigit()
                        }
                    } else {
                        Text("Start")
                    }
                }
                .font(Studio.text(15, weight: .semibold))
                .foregroundColor(canStart || isRecording ? Studio.onAccent : Studio.mute)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 15)
                .background(
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .fill(isRecording ? Studio.hot : (canStart ? Studio.accent : Studio.sunk))
                )
            }
            .buttonStyle(PressableButtonStyle())
            .disabled(!canStart && !isRecording)
            .padding(16)
        }
        .background(Studio.bg)
    }

    private var canStart: Bool { downloaded && !finishing && !downloading }

    private static func clock(_ seconds: TimeInterval) -> String {
        let s = Int(seconds)
        return s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
                         : String(format: "%d:%02d", s / 60, s % 60)
    }

    // MARK: - Actions

    private func download() {
        downloading = true
        downloadProgress = 0
        message = nil
        Task {
            do {
                try await WeightFetcher.fetch(engine.files, into: engine.directory,
                                              expected: WeightFile.topLevelNames(engine.files),
                                              progress: { downloadProgress = $0.fraction },
                                              note: { _ in })
                downloaded = engine.isDownloaded
            } catch is CancellationError {
            } catch {
                message = "The download stopped: \(error.localizedDescription) Tap Download to carry on."
            }
            downloading = false
        }
    }

    private func start() {
        message = nil
        AVAudioApplication.requestRecordPermission { granted in
            Task { @MainActor in
                guard granted else {
                    message = "Microphone access is off. Enable it in Settings › Whisper."
                    return
                }
                let session = AVAudioSession.sharedInstance()
                do {
                    #if targetEnvironment(simulator)
                    throw SimulatorUnsupported()
                    #endif
                    #if targetEnvironment(macCatalyst)
                    try? session.setCategory(.playAndRecord, mode: .default, options: [])
                    #else
                    try? session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothHFP])
                    #endif
                    try session.setActive(true)
                    text = ""
                    saved = nil
                    copied = false
                    let live = LiveTranscriber()
                    let script = language
                    try await live.start(language: engine.modelLanguage(for: language)) { heard in
                        text = script == Languages.traditionalChinese ? Languages.traditional(heard) : heard
                    }
                    transcriber = live
                    startedAt = Date()
                } catch {
                    message = "Couldn't start: \(error.localizedDescription)"
                }
            }
        }
    }

    private func stop() {
        guard let live = transcriber else { return }
        transcriber = nil
        startedAt = nil
        finishing = true
        let script = language
        Task {
            let (url, result) = await live.stop()
            // 0.76 GB the other tabs may want; a second takes it back.
            await SpeechRunner.shared.unload()
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            let convert: (String) -> String = { script == Languages.traditionalChinese ? Languages.traditional($0) : $0 }
            let full = convert(result.text.trimmingCharacters(in: .whitespacesAndNewlines))
            if full.isEmpty {
                try? FileManager.default.removeItem(at: url)
                message = "Nothing was heard. Check that the microphone isn't muted, and speak a little closer."
            } else {
                text = full
                let envelope = await Task.detached(priority: .userInitiated) { AudioConverter.peakEnvelope(of: url)?.buckets }.value
                save(url: url, result: result, text: full, envelope: envelope, convert: convert)
            }
            finishing = false
        }
    }

    /// Into the library, as the Transcribe tab keeps a recording.
    private func save(url: URL, result: TranscriptionResult, text: String, envelope: [Float]?,
                      convert: (String) -> String) {
        let stamp = Self.stampFormatter.string(from: Date())
        let audio = try? AudioFiles.saveAudio(from: url, suggestedName: "Live_\(stamp)", movingSource: true)
        let firstLine = text.components(separatedBy: .newlines).first ?? ""
        let record = SavedTranscript(
            title: firstLine.count > 50 ? String(firstLine.prefix(50)) + "…" : firstLine,
            duration: Double(result.segments.last?.end ?? 0),
            languageCode: language == "auto" ? nil : language,
            modelName: "\(engine.displayName) · live",
            fullText: text,
            audioFilePath: audio,
            waveform: envelope
        )
        record.segments = result.segments.map {
            SavedSegment(startTime: Double($0.start), endTime: Double($0.end), text: convert($0.text))
        }
        modelContext.insert(record)
        try? modelContext.save()
        saved = record
    }

    private static let stampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HHmm"
        return formatter
    }()

    private struct SimulatorUnsupported: LocalizedError {
        var errorDescription: String? { "Live transcription needs a real device: the Simulator's GPU can't run it." }
    }

    // MARK: - Language

    private func languageName(_ code: String) -> String {
        code == "auto" ? "Auto-detect" : Languages.displayName(code)
    }

    private var languageSheet: some View {
        NavigationStack {
            List(SpeechEngine.liveLanguages, id: \.self) { code in
                Button {
                    language = code
                    showLanguages = false
                } label: {
                    HStack {
                        Text(languageName(code)).foregroundColor(Studio.ink)
                        Spacer()
                        if code == language { Image(systemName: "checkmark").foregroundColor(Studio.accent) }
                    }
                }
            }
            .navigationTitle("Language")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) { Button("Done") { showLanguages = false } }
            }
        }
        .presentationDetents([.medium, .large])
    }
}
