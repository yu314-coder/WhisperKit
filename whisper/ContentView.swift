import SwiftUI
import SwiftData
import WhisperKit
import UniformTypeIdentifiers
import PhotosUI
import Charts
import Metal
import AVFoundation
import UserNotifications
import ActivityKit
import BackgroundTasks

struct ContentView: View {
    // MARK: - State Properties
    @Environment(\.modelContext) private var modelContext
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    /// Recent work, surfaced on the idle page. On iPad the single column left
    /// most of the screen empty; showing the library's most recent entries
    /// gives the page something to be.
    @Query(sort: \SavedTranscript.createdAt, order: .reverse)
    private var recentTranscripts: [SavedTranscript]
    @State private var quickOpenTranscript: SavedTranscript?

    /// Sampling model. Isolated from ContentView's own state so its 1 Hz tick
    /// redraws only the meter, not the whole screen.
    @State private var monitor = SystemMonitor()

    /// In-flight model download/load. Held so selecting another model can
    /// cancel it instead of leaving two WhisperKit inits racing for the same
    /// `whisperKit` slot.
    @State private var modelPrepTask: Task<Void, Never>?
    @State private var isPreparingModel = false
    /// Which variant the in-flight preparation is for, so tapping the same
    /// model again doesn't cancel and restart a download that's already going.
    @State private var preparingVariant: WhisperModel?
    /// 0...1 while downloading; nil once we're loading into the Neural Engine
    /// (that phase reports no progress).
    @State private var downloadProgress: Double?
    /// True while Core ML is specialising the model for this chip — the
    /// one-time cost that used to land on the user's first recording.
    @State private var isOptimizingModel = false

    /// Variants already specialised on this device. Core ML caches the
    /// specialised model outside the app and evicts that cache on OS updates,
    /// so the record is keyed by OS build as well as variant.
    @AppStorage("prewarmedVariants") private var prewarmedVariantsRaw: String = ""
    /// Guards the one-time repair of segments written by 1.1 (1)–(4).
    @AppStorage("didSanitiseStoredSegments") private var didSanitiseStoredSegments = false

    /// Which processor Core ML runs the model on. Defaults to GPU because the
    /// Neural Engine's one-time "specialisation" compile is what made the first
    /// load take minutes even on an M3; Metal skips that step entirely.
    @AppStorage("computeMode") private var computeModeRaw: String = ComputeMode.gpu.rawValue
    private var computeMode: ComputeMode {
        get { ComputeMode(rawValue: computeModeRaw) ?? .gpu }
        nonmutating set { computeModeRaw = newValue.rawValue }
    }

    /// Set across a model load and cleared when it settles. Finding it still
    /// set at launch means the previous load never finished — the app was
    /// killed part-way, which on iOS almost always means it was jetsammed for
    /// memory while Core ML specialised the model.
    @AppStorage("modelLoadInFlight") private var modelLoadInFlight = false
    // Previous tick counters for instantaneous CPU & task time deltas.
    // Without this baseline the sampler returns the cumulative since-boot
    // average, which is nearly constant — making the chart look frozen.
    @State private var showLibrary: Bool = false
    @State private var showModelPicker: Bool = false
    @State private var showLanguagePicker: Bool = false
    @State private var lastTranscribedAudioURL: URL? = nil
    @State private var lastTranscriptionResults: [TranscriptionResult] = []
    @State private var showModelDetails: Bool = false
    @State private var detectedLanguageCode: String? = nil
    /// The transcript body only — clean text, no decoration. Metadata lives in
    /// `transcriptMeta` and failures in `errorMessage`, so this is always
    /// exactly what the user would want to copy or export.
    @State private var transcript: String = ""
    @State private var transcriptMeta: TranscriptMeta? = nil
    @State private var errorMessage: String? = nil
    @State private var streamingTranscript: String = ""
    @State private var isProcessing = false
    @State private var isModelLoaded = false
    @AppStorage("selectedModel") private var selectedModelRaw: String = WhisperModel.turbo.rawValue
    private var selectedModel: WhisperModel {
        get { WhisperModel.migrating(selectedModelRaw) ?? .turbo }
        nonmutating set { selectedModelRaw = newValue.rawValue }
    }
    @State private var downloadStatus: [WhisperModel: Bool] = [:]
    @State private var showingFilePicker = false
    @State private var statusMessage = ""
    @State private var selectedPhotoItem: PhotosPickerItem?
    @AppStorage("selectedLanguage") private var selectedLanguage: String = "auto"
    @State private var transcriptionProgress: Double = 0.0
    @State private var currentSegment: Int = 0
    @State private var totalSegments: Int = 0
    @State private var showCopySuccess = false
    
    // TQDM-style progress tracking
    @State private var processStartTime: Date?
    @State private var estimatedTotalTime: TimeInterval = 0
    @State private var segmentsPerSecond: Double = 0
    @State private var lastProgressUpdate: Date?
    
    // MARK: - WhisperKit
    @State private var whisperKit: WhisperKit?
    
    // Real-time monitoring
    
    // Background handling
    @Environment(\.scenePhase) private var scenePhase
    @State private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    @State private var backgroundTaskProgress: Progress?
    
    // Live Activity
    @State private var currentActivity: Activity<TranscriptionAttributes>?
    @State private var activeTranscriptionID: String?

    // Audio recording
    @State private var audioRecorder: AVAudioRecorder?
    @State private var isRecording = false
    @State private var recordingURL: URL?
    @State private var recordingStartTime: Date?
    
    // MARK: - Transcript Result Metadata

    /// Structured stats for a finished run. These used to be interpolated into
    /// the transcript string itself, which made the on-screen text impossible
    /// to copy or restyle without the decoration coming along.
    struct TranscriptMeta {
        var detectedLanguage: String?
        var selectedLanguageName: String
        var audioDuration: Double
        var processingTime: Double
        var modelName: String
        var segmentCount: Int

        /// Seconds of audio handled per second of wall clock.
        var realtimeFactor: Double? {
            guard processingTime > 0, audioDuration > 0 else { return nil }
            return audioDuration / processingTime
        }
    }

    // MARK: - Models
    //
    // All four are Argmax's quantized CoreML builds. The previous lineup used
    // unquantized variants whose advertised sizes were badly wrong — "Large V3
    // Turbo • ~950MB" actually pulled 3.2 GB, and "Medium • ~750MB" pulled
    // 1.5 GB. Sizes below are measured from the model repo.
    //
    // Base and Medium are gone. Small-quantized beats base outright at a
    // similar size (base is close to unusable for Chinese), and the quantized
    // large-v3-turbo beats medium at *less than half* medium's download.
    enum WhisperModel: String, CaseIterable {
        case small        = "openai_whisper-small_216MB"
        case turbo        = "openai_whisper-large-v3-v20240930_turbo_632MB"
        case largeV3      = "openai_whisper-large-v3_947MB"
        case largeV3Turbo = "openai_whisper-large-v3_turbo_954MB"

        var displayName: String {
            switch self {
            case .small:        return "Small"
            case .turbo:        return "Turbo"
            case .largeV3:      return "Large V3"
            case .largeV3Turbo: return "Large V3 Turbo"
            }
        }

        /// Measured download size of the model folder, in megabytes.
        var sizeMB: Int {
            switch self {
            case .small:        return 217
            case .turbo:        return 646
            case .largeV3:      return 948
            case .largeV3Turbo: return 1053
            }
        }

        var sizeLabel: String {
            sizeMB >= 1000
                ? String(format: "%.1f GB", Double(sizeMB) / 1000)
                : "\(sizeMB) MB"
        }

        var tagline: String {
            switch self {
            case .small:        return "Fastest"
            case .turbo:        return "Recommended"
            case .largeV3:      return "Best quality"
            case .largeV3Turbo: return "Best overall"
            }
        }

        var description: String { "\(tagline) • \(sizeLabel)" }

        /// Maps a previously stored raw value onto the current lineup, so an
        /// upgrade doesn't silently reset the user to the default.
        static func migrating(_ raw: String) -> WhisperModel? {
            if let exact = WhisperModel(rawValue: raw) { return exact }
            switch raw {
            case "openai_whisper-base", "openai_whisper-tiny",
                 "openai_whisper-small":                       return .small
            case "openai_whisper-medium":                      return .turbo
            case "openai_whisper-large-v3":                    return .largeV3
            case "openai_whisper-large-v3_turbo":              return .largeV3Turbo
            default:                                           return nil
            }
        }
    }

    // MARK: - Languages
    let supportedLanguages: [(code: String, name: String)] = [
        ("auto", "🌍 Auto Detect"),
        ("en", "🇺🇸 English"),
        ("zh", "🇨🇳 Chinese (中文)"),
        ("es", "🇪🇸 Spanish (Español)"),
        ("fr", "🇫🇷 French (Français)"),
        ("de", "🇩🇪 German (Deutsch)"),
        ("ja", "🇯🇵 Japanese (日本語)"),
        ("ko", "🇰🇷 Korean (한국어)"),
        ("ru", "🇷🇺 Russian (Русский)"),
        ("pt", "🇵🇹 Portuguese (Português)"),
        ("it", "🇮🇹 Italian (Italiano)")
    ]
    
    var body: some View {
        editorialLayout
            .animation(.spring(response: 0.45, dampingFraction: 0.85), value: isProcessing)
            .animation(.spring(response: 0.45, dampingFraction: 0.85), value: isModelLoaded)
            .animation(.spring(response: 0.45, dampingFraction: 0.85), value: isRecording)
            // .page rather than the default form sheet: on a 13" iPad the
            // default is a small panel marooned in a dimmed screen, which
            // wastes most of the display for content that wants the room.
            // No effect in compact width, where sheets are already full width.
            .sheet(isPresented: $showLibrary) {
                TranscriptLibraryView()
                    .presentationSizing(.page)
            }
            .sheet(item: $quickOpenTranscript) { t in
                NavigationStack {
                    TranscriptDetailView(transcript: t)
                }
                .presentationSizing(.page)
            }
        .fileImporter(
            isPresented: $showingFilePicker,
            allowedContentTypes: [.audio, .movie, .mpeg4Movie, .quickTimeMovie, .mpeg4Audio],
            allowsMultipleSelection: false
        ) { result in
            handleFileImport(result)
        }
        .onAppear {
            // Start the model loading first — it is the long pole on launch.
            // The one-time maintenance below walks every stored segment, which
            // on a large library is slow enough to visibly delay it.
            checkModelStatus()
            Task { @MainActor in
                removeRetiredModelDownloads()
                sanitiseStoredSegmentsIfNeeded()
                createDebugFiles()
            }
            monitor.initializeGPUMonitoring()
            monitor.onSample = {
                if isProcessing && scenePhase == .background {
                    updateLiveActivity()
                }
            }
            setupBackgroundAudio()
            requestNotificationPermissions()
            registerBackgroundTasks()
            cleanupStaleActivities()
        }
        .onDisappear {
            monitor.stopSystemMonitoring()
            endLiveActivity()

            // Clean up all activities when view disappears
            if #available(iOS 16.2, *) {
                Task {
                    for activity in Activity<TranscriptionAttributes>.activities {
                        await activity.end(nil, dismissalPolicy: .immediate)
                    }
                }
            }
        }
        .onChange(of: scenePhase) { oldPhase, newPhase in
            handleScenePhaseChange(oldPhase: oldPhase, newPhase: newPhase)
        }
    }

    // MARK: - Palette (warm-paper aesthetic)
    //
    // Light is warm paper; dark is the same page under lamplight — warm
    // near-black stock with off-white ink, so the editorial feel survives the
    // switch instead of turning into generic gray-on-black.

    /// Builds a color that resolves per trait collection, so every call site
    /// stays a plain `Self.paperInk` while still following the system appearance.
    private static func adaptive(
        light: (r: Double, g: Double, b: Double, a: Double),
        dark: (r: Double, g: Double, b: Double, a: Double)
    ) -> Color {
        Color(UIColor { traits in
            let c = traits.userInterfaceStyle == .dark ? dark : light
            return UIColor(red: c.r, green: c.g, blue: c.b, alpha: c.a)
        })
    }

    static let paperBG = adaptive(
        light: (0.980, 0.965, 0.940, 1.0),   // #FAF6F0 warm paper
        dark:  (0.075, 0.068, 0.059, 1.0)    // warm near-black stock
    )
    static let paperInk = adaptive(
        light: (0.110, 0.095, 0.080, 1.0),   // warm near-black
        dark:  (0.949, 0.929, 0.894, 1.0)    // warm off-white
    )
    static let paperMute = adaptive(
        light: (0.520, 0.495, 0.460, 1.0),   // warm gray
        dark:  (0.620, 0.592, 0.548, 1.0)
    )
    static let paperRule = adaptive(
        light: (0.000, 0.000, 0.000, 0.10),
        dark:  (1.000, 0.980, 0.940, 0.16)
    )
    static let paperAccent = adaptive(
        light: (0.780, 0.290, 0.180, 1.0),   // terracotta
        dark:  (0.910, 0.450, 0.320, 1.0)    // lifted to carry on dark stock
    )
    static let paperCard = adaptive(
        light: (1.000, 1.000, 1.000, 0.55),
        dark:  (1.000, 0.960, 0.900, 0.06)
    )
    static let paperAccentSoft = adaptive(
        light: (0.940, 0.825, 0.770, 1.0),   // pale terracotta
        dark:  (0.310, 0.150, 0.110, 1.0)    // deep terracotta wash
    )

    // MARK: - Adaptive metrics (compact iPhone vs regular iPad)

    /// True on iPad and in wide multitasking splits. Driven by size class, not
    /// idiom, so a Slide Over pane correctly gets the compact layout.
    private var isRegularWidth: Bool { horizontalSizeClass == .regular }

    /// Measure cap for the reading column. Serif body text gets hard to track
    /// past ~75 characters a line, so on iPad the page stays a centered column
    /// instead of stretching to a 13" bezel.
    private var contentMeasure: CGFloat { isRegularWidth ? 720 : .infinity }

    /// Outer gutter — wider on iPad to keep the column off the edges.
    private var gutter: CGFloat { isRegularWidth ? 40 : 22 }

    /// Body type scales up slightly on the larger canvas.
    private var bodyTextSize: CGFloat { isRegularWidth ? 21 : 19 }

    /// Capturing audio needs no model — only transcribing does. Loading a model
    /// takes real time on every launch (the weights live in memory, so it can't
    /// be skipped), and gating the record button on it made the app unusable
    /// for that whole window. Record now, transcribe when the model arrives.
    private var canCapture: Bool { isModelLoaded || isPreparingModel }

    // MARK: - Editorial / Paper Layout
    var editorialLayout: some View {
        ZStack {
            Self.paperBG.ignoresSafeArea()
            VStack(spacing: 0) {
                editorialTopBar
                editorialTranscriptCanvas
                editorialControlBar
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .sheet(isPresented: $showModelPicker) {
            modelPickerSheet
                .presentationSizing(.page)
        }
        .sheet(isPresented: $showLanguagePicker) {
            languagePickerSheet
                .presentationSizing(.page)
        }
    }

    // Clean top bar — serif wordmark left, model chip + library right
    var editorialTopBar: some View {
        HStack(alignment: .center, spacing: 0) {
            (Text("Whisper")
                .font(.system(size: 28, weight: .semibold, design: .serif))
                .foregroundColor(Self.paperInk)
             + Text(".")
                .font(.system(size: 28, weight: .semibold, design: .serif))
                .foregroundColor(Self.paperAccent))

            Spacer()

            HStack(spacing: 8) {
                Button {
                    showModelPicker = true
                } label: {
                    HStack(spacing: 5) {
                        Circle()
                            .fill(isModelLoaded ? Color(red: 0.30, green: 0.65, blue: 0.40) : Self.paperMute.opacity(0.4))
                            .frame(width: 6, height: 6)
                        Text(selectedModel.displayName)
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(Self.paperInk.opacity(0.75))
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .overlay(Capsule().strokeBorder(Self.paperRule, lineWidth: 0.5))
                }
                .buttonStyle(PressableButtonStyle())

                Button {
                    showLibrary = true
                } label: {
                    Image(systemName: "books.vertical")
                        .font(.system(size: 16, weight: .regular))
                        .foregroundColor(Self.paperInk.opacity(0.75))
                        .frame(width: 32, height: 32)
                        .overlay(Circle().strokeBorder(Self.paperRule, lineWidth: 0.5))
                }
                .buttonStyle(PressableButtonStyle())
            }
        }
        .padding(.horizontal, gutter)
        // Content tracks the reading column; the hairline rule still spans the
        // full width so the page reads as one sheet on iPad.
        .frame(maxWidth: contentMeasure)
        .frame(maxWidth: .infinity)
        .padding(.top, 8)
        .padding(.bottom, 14)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Self.paperRule).frame(height: 0.5)
        }
    }

    // Transcript canvas — large serif body text on warm paper
    var editorialTranscriptCanvas: some View {
        GeometryReader { geo in
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    if isProcessing {
                        if !streamingTranscript.isEmpty {
                            liveTranscriptView
                        } else {
                            editorialProcessingState
                        }
                    } else if let errorMessage {
                        editorialErrorState(errorMessage)
                    } else if !transcript.isEmpty {
                        finishedTranscriptView
                    } else {
                        editorialEmptyState
                    }
                }
                .padding(.horizontal, isRegularWidth ? gutter : 26)
                .padding(.top, isRegularWidth ? 44 : 32)
                .padding(.bottom, 28)
                .frame(maxWidth: contentMeasure, alignment: .leading)
                .frame(maxWidth: .infinity)
                // Short states used to sit in the top-left corner of a mostly
                // empty iPad screen. Centre them in the available height so the
                // page reads as composed rather than unfinished.
                .frame(minHeight: geo.size.height, alignment: shortFormCanvas ? .center : .top)
            }
        }
        .frame(maxHeight: .infinity)
    }

    /// True when the canvas is showing something too small to fill the screen.
    private var shortFormCanvas: Bool {
        guard !isProcessing, transcript.isEmpty else { return false }
        if errorMessage != nil { return true }
        return recentTranscripts.isEmpty
    }

    var editorialEmptyState: some View {
        VStack(alignment: .leading, spacing: 22) {
            // Date — gives the page a sense of "today"
            Text(todayPaperDate())
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .tracking(2)
                .foregroundColor(Self.paperMute)

            Text("A quiet place\nfor your voice.")
                .font(.system(size: isRegularWidth ? 44 : 36, weight: .semibold, design: .serif))
                .foregroundColor(Self.paperInk)
                .lineSpacing(2)

            Text(emptyStateBlurb)
                .font(.system(size: 16, design: .serif))
                .foregroundColor(Self.paperInk.opacity(0.65))
                .lineSpacing(4)

            if isModelLoaded {
                HStack(spacing: 8) {
                    editorialChip(icon: "globe", text: "99 languages")
                    editorialChip(icon: "lock.fill", text: "On-device")
                    editorialChip(icon: "books.vertical", text: "Library")
                }
                .padding(.top, 4)
            } else if isReloadingKnownModel {
                // The model is already on disk and simply loading. Showing the
                // whole "choose a model" grid here implied a decision the user
                // had already made, and hid the fact that they can just record.
                modelPreparationStatus
                    .padding(.top, 4)
            } else {
                inlineModelPicker
                    .padding(.top, 8)
            }

            recentSection
        }
    }

    /// True when the selected model is already downloaded and is merely being
    /// loaded — as opposed to the user not having chosen one yet.
    private var isReloadingKnownModel: Bool {
        isPreparingModel && (downloadStatus[selectedModel] ?? false)
    }

    private var emptyStateBlurb: String {
        if isModelLoaded {
            return "Press the round button below to start a new recording, or pull in an audio file. Everything you transcribe stays on this device — every page is kept in your library."
        }
        if isReloadingKnownModel {
            return "\(selectedModel.displayName) is loading onto the \(computeMode.shortName) — this happens once each time you open the app. Go ahead and record; it will transcribe as soon as the model is ready."
        }
        return "First, choose a transcription model. It downloads once and runs entirely on this device. Larger models are slower but more accurate."
    }

    /// The most recent transcripts, inline on the idle page.
    @ViewBuilder
    var recentSection: some View {
        let items = Array(recentTranscripts.prefix(isRegularWidth ? 6 : 3))
        if !items.isEmpty {
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 8) {
                    Rectangle()
                        .fill(Self.paperInk.opacity(0.20))
                        .frame(width: 18, height: 1)
                    Text("Recent")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .tracking(2)
                        .foregroundColor(Self.paperMute)
                    Spacer()
                    Button {
                        showLibrary = true
                    } label: {
                        Text("All \(recentTranscripts.count) →")
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundColor(Self.paperAccent)
                    }
                    .buttonStyle(PressableButtonStyle())
                }
                .padding(.bottom, 10)

                ForEach(items) { t in
                    Button {
                        quickOpenTranscript = t
                    } label: {
                        recentRow(t)
                    }
                    .buttonStyle(PressableButtonStyle())

                    if t.id != items.last?.id {
                        Rectangle()
                            .fill(Self.paperRule)
                            .frame(height: 0.5)
                    }
                }
            }
            .padding(.top, 26)
        }
    }

    func recentRow(_ t: SavedTranscript) -> some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(t.title)
                    .font(.system(size: 15, weight: .medium, design: .serif))
                    .foregroundColor(Self.paperInk)
                    .lineLimit(1)
                    .multilineTextAlignment(.leading)

                HStack(spacing: 10) {
                    Text(TranscriptExporter.formatTimestamp(t.duration))
                    if let lang = t.languageCode, !lang.isEmpty {
                        Text(lang.uppercased())
                    }
                    if let model = t.modelName {
                        Text(model)
                    }
                }
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundColor(Self.paperMute)
            }

            Spacer(minLength: 8)

            Image(systemName: "chevron.right")
                .font(.system(size: 10, weight: .semibold))
                .foregroundColor(Self.paperMute.opacity(0.6))
                .padding(.top, 4)
        }
        .padding(.vertical, 11)
        .contentShape(Rectangle())
    }

    // Compact inline model picker shown on the empty state when nothing's loaded yet.
    var inlineModelPicker: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Rectangle()
                    .fill(Self.paperInk.opacity(0.20))
                    .frame(width: 18, height: 1)
                Text("Choose a model")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .tracking(2)
                    .foregroundColor(Self.paperMute)
            }

            LazyVGrid(
                columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)],
                spacing: 10
            ) {
                ForEach(WhisperModel.allCases, id: \.self) { model in
                    paperModelCard(model)
                }
            }

            modelPreparationStatus

            Button {
                if isPreparingModel {
                    modelPrepTask?.cancel()
                    isPreparingModel = false
                    statusMessage = ""
                } else {
                    prepareModel(selectedModel)
                }
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: isPreparingModel
                          ? "xmark"
                          : (downloadStatus[selectedModel] ?? false ? "arrow.clockwise" : "arrow.down"))
                        .font(.system(size: 13, weight: .bold))
                    Text(isPreparingModel
                         ? "Cancel"
                         : (downloadStatus[selectedModel] ?? false
                            ? "Reload \(selectedModel.displayName)"
                            : "Download \(selectedModel.displayName) · \(selectedModel.sizeLabel)"))
                        .font(.system(size: 14, weight: .semibold, design: .serif))
                }
                .foregroundColor(Self.paperBG)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 13)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Self.paperInk)
                        .shadow(color: Self.paperInk.opacity(0.18), radius: 10, y: 4)
                )
            }
            .buttonStyle(PressableButtonStyle())
            .disabled(isProcessing && !isPreparingModel)
            .opacity((isProcessing && !isPreparingModel) ? 0.5 : 1)
        }
    }

    /// Preparation progress. Shown in both the inline picker (first launch)
    /// and the model sheet, so a long first load is never unexplained.
    @ViewBuilder
    var modelPreparationStatus: some View {
        if isPreparingModel {
            VStack(alignment: .leading, spacing: 7) {
                if let p = downloadProgress {
                    ProgressView(value: p)
                        .progressViewStyle(.linear)
                        .tint(Self.paperAccent)
                    Text("\(Int(p * 100))% of \(selectedModel.sizeLabel)")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundColor(Self.paperMute)
                } else {
                    ProgressView()
                        .progressViewStyle(.linear)
                        .tint(Self.paperAccent)
                    Text(statusMessage.isEmpty ? "Preparing…" : statusMessage)
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundColor(Self.paperMute)
                    if isOptimizingModel {
                        Text("iOS compiles each model for this specific chip the first time it runs. This happens once per model — afterwards it loads in a moment.")
                            .font(.system(size: 11, design: .serif))
                            .foregroundColor(Self.paperMute)
                            .lineSpacing(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
    }

    func editorialChip(icon: String, text: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
            Text(text)
                .font(.system(size: 11, weight: .medium))
        }
        .foregroundColor(Self.paperInk.opacity(0.55))
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .overlay(Capsule().strokeBorder(Self.paperRule, lineWidth: 0.5))
    }

    /// Classifies what kind of "processing" is happening so the UI label
    /// can match the actual phase (downloading a model vs. transcribing
    /// audio vs. importing a file).
    enum ProcessingPhase {
        case downloading
        case loadingModel
        case importingAudio
        case transcribing

        var heading: String {
            switch self {
            case .downloading:    return "Downloading the model…"
            case .loadingModel:   return "Warming up the model…"
            case .importingAudio: return "Reading audio…"
            case .transcribing:   return "Listening to the words…"
            }
        }
    }

    var currentProcessingPhase: ProcessingPhase {
        let s = statusMessage.lowercased()
        if s.contains("download")     { return .downloading }
        if s.contains("load") && s.contains("model") { return .loadingModel }
        if s.contains("import") || s.contains("read") || s.contains("photo") || s.contains("convert") {
            return .importingAudio
        }
        // Default: anything else with non-empty progress or status implies transcribing.
        return .transcribing
    }

    var editorialProcessingState: some View {
        let phase = currentProcessingPhase
        return VStack(alignment: .leading, spacing: 16) {
            Text(todayPaperDate())
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .tracking(2)
                .foregroundColor(Self.paperMute)

            HStack(spacing: 10) {
                ProgressView()
                    .tint(Self.paperInk.opacity(0.5))
                Text(phase.heading)
                    .font(.system(size: 22, weight: .regular, design: .serif).italic())
                    .foregroundColor(Self.paperInk.opacity(0.75))
            }

            if transcriptionProgress > 0 {
                ProgressView(value: transcriptionProgress)
                    .progressViewStyle(.linear)
                    .tint(Self.paperAccent)
                    .padding(.top, 2)
            }

            if !statusMessage.isEmpty {
                Text(statusMessage)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Self.paperMute.opacity(0.7))
            }

            // Live CPU / Memory / Neural-Engine sparklines run for ALL phases
            // (download, model warm-up, audio convert, transcription) so the
            // user can see the app actually working.
            PaperPerfMonitor(monitor: monitor)
                .padding(.top, 12)
        }
    }

    // MARK: - Performance monitor (CPU / Memory / Neural Engine)
    //
    // Card lives below the "Listening / Downloading…" heading during any
    // background phase. Each row: colored channel dot, full label, current
    // value as the visual hero (big monospace), peak chip, smooth Bezier
    // sparkline below, peak dotted line.

    // Channel accent colors — pulled from the warm-paper palette
    static let perfCPU    = Color(red: 0.780, green: 0.290, blue: 0.180) // terracotta
    static let perfMem    = Color(red: 0.470, green: 0.380, blue: 0.220) // umber
    static let perfNPU    = Color(red: 0.180, green: 0.380, blue: 0.580) // deep ink-blue


    // Catmull-Rom-ish smooth path through points (with optional fill close)

    var liveTranscriptView: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 6) {
                Circle()
                    .fill(Self.paperAccent)
                    .frame(width: 7, height: 7)
                Text("LIVE")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .tracking(2)
                    .foregroundColor(Self.paperAccent)
            }

            Text(streamingTranscript)
                .font(.system(size: bodyTextSize + 1, weight: .regular, design: .serif))
                .foregroundColor(Self.paperInk)
                .lineSpacing(7)

            if transcriptionProgress > 0 {
                ProgressView(value: transcriptionProgress)
                    .progressViewStyle(.linear)
                    .tint(Self.paperAccent)
                    .padding(.top, 4)
            }

            PaperPerfMonitor(monitor: monitor)
                .padding(.top, 6)
        }
    }

    var finishedTranscriptView: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 8) {
                Text(todayPaperDate())
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .tracking(2)
                    .foregroundColor(Self.paperMute)

                Spacer()

                if let code = detectedLanguageCode, !code.isEmpty {
                    HStack(spacing: 4) {
                        Image(systemName: "sparkles")
                            .font(.system(size: 9, weight: .bold))
                        Text(displayName(forLanguageCode: code))
                            .font(.system(size: 11, weight: .semibold))
                    }
                    .foregroundColor(Self.paperAccent)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(Self.paperAccentSoft.opacity(0.6)))
                }

                Button(action: copyTranscript) {
                    Image(systemName: showCopySuccess ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(showCopySuccess ? Color(red: 0.30, green: 0.65, blue: 0.40) : Self.paperInk.opacity(0.55))
                        .frame(width: 28, height: 28)
                        .overlay(Circle().strokeBorder(Self.paperRule, lineWidth: 0.5))
                }
                .accessibilityLabel(showCopySuccess ? "Transcript copied" : "Copy transcript")
            }

            if let meta = transcriptMeta {
                transcriptMetaStrip(meta)
            }

            Text(transcript)
                .font(.system(size: bodyTextSize, weight: .regular, design: .serif))
                .foregroundColor(Self.paperInk)
                .lineSpacing(8)
                .textSelection(.enabled)
        }
    }

    /// Run stats as a quiet rule-bounded strip, so the transcript body below
    /// stays pure text that copies and exports cleanly.
    func transcriptMetaStrip(_ meta: TranscriptMeta) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Rectangle().fill(Self.paperRule).frame(height: 0.5)

            HStack(spacing: 14) {
                ForEach(metaItems(meta), id: \.label) { item in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.label)
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .tracking(1.2)
                            .foregroundColor(Self.paperMute)
                        Text(item.value)
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundColor(Self.paperInk.opacity(0.8))
                    }
                }
                Spacer(minLength: 0)
            }

            Rectangle().fill(Self.paperRule).frame(height: 0.5)
        }
    }

    private func metaItems(_ meta: TranscriptMeta) -> [(label: String, value: String)] {
        var items: [(label: String, value: String)] = []
        if meta.audioDuration > 0 {
            items.append(("LENGTH", formatDuration(meta.audioDuration)))
        }
        items.append(("MODEL", meta.modelName))
        items.append(("SEGMENTS", "\(meta.segmentCount)"))
        if let factor = meta.realtimeFactor {
            items.append(("SPEED", String(format: "%.1f×", factor)))
        }
        items.append(("TOOK", formatDuration(meta.processingTime)))
        return items
    }

    /// Failures get their own presentation instead of being written into the
    /// transcript text, where they used to be copyable and exportable as if
    /// they were speech.
    func editorialErrorState(_ message: String) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.system(size: 11, weight: .bold))
                Text("SOMETHING WENT WRONG")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .tracking(2)
            }
            .foregroundColor(Self.paperAccent)

            Text(message)
                .font(.system(size: 16, design: .serif))
                .foregroundColor(Self.paperInk.opacity(0.8))
                .lineSpacing(4)
                .fixedSize(horizontal: false, vertical: true)

            Button {
                errorMessage = nil
            } label: {
                Text("Dismiss")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(Self.paperInk.opacity(0.75))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 7)
                    .overlay(Capsule().strokeBorder(Self.paperRule, lineWidth: 0.5))
            }
            .buttonStyle(PressableButtonStyle())
        }
        .padding(isRegularWidth ? 24 : 18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Self.paperAccentSoft.opacity(0.35))
        )
    }

    // Stamp like "MAY 19, 2026"
    private static let paperDateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "MMM d, yyyy"
        return f
    }()

    func todayPaperDate() -> String {
        Self.paperDateFormatter.string(from: Date()).uppercased()
    }

    // Bottom control bar — paper aesthetic, hero round record button
    var editorialControlBar: some View {
        VStack(spacing: 0) {
            // Recording timer pill — only when capturing
            if isRecording {
                HStack(spacing: 6) {
                    Circle().fill(Self.paperAccent).frame(width: 6, height: 6)
                    TimelineView(.periodic(from: .now, by: 1)) { context in
                        Text(formatDuration(context.date.timeIntervalSince(recordingStartTime ?? context.date)))
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundColor(Self.paperAccent)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Capsule().fill(Self.paperAccentSoft.opacity(0.5)))
                .padding(.bottom, 10)
                .transition(.opacity.combined(with: .scale(scale: 0.9)))
            }

            HStack(spacing: 18) {
                editorialIconButton(icon: "folder", disabled: !canCapture || isProcessing) {
                    showingFilePicker = true
                }

                PhotosPicker(selection: $selectedPhotoItem, matching: .videos) {
                    editorialIconButtonLabel(icon: "photo.on.rectangle.angled")
                }
                .disabled(!canCapture || isProcessing)
                .opacity((isModelLoaded && !isProcessing) ? 1.0 : 0.35)
                .onChange(of: selectedPhotoItem) { _, newItem in
                    if let newItem = newItem { handlePhotoSelection(newItem) }
                }

                Button(action: toggleRecording) {
                    ZStack {
                        // Outer hairline ring (always visible)
                        Circle()
                            .strokeBorder(Self.paperInk.opacity(0.15), lineWidth: 1)
                            .frame(width: 76, height: 76)

                        Circle()
                            .fill(isRecording ? Self.paperInk : Self.paperAccent)
                            .frame(width: 64, height: 64)
                            .shadow(color: (isRecording ? Self.paperInk : Self.paperAccent).opacity(0.30), radius: 16, y: 6)

                        if isRecording {
                            // Pulse
                            Circle()
                                .stroke(Self.paperAccent.opacity(0.5), lineWidth: 2)
                                .frame(width: 84, height: 84)
                                .scaleEffect(isRecording ? 1.18 : 1.0)
                                .opacity(isRecording ? 0 : 1)
                                .animation(.easeOut(duration: 1.2).repeatForever(autoreverses: false), value: isRecording)

                            RoundedRectangle(cornerRadius: 3, style: .continuous)
                                .fill(Color.white)
                                .frame(width: 18, height: 18)
                        } else {
                            Image(systemName: "mic.fill")
                                .font(.system(size: 22, weight: .bold))
                                .foregroundColor(.white)
                        }
                    }
                }
                .buttonStyle(PressableButtonStyle())
                .disabled(!canCapture || (isProcessing && !isRecording))
                .opacity((isModelLoaded && (!isProcessing || isRecording)) ? 1.0 : 0.4)

                Button {
                    showLanguagePicker = true
                } label: {
                    languageIndicatorLabel
                }
                .buttonStyle(PressableButtonStyle())
                .disabled(isProcessing && !isRecording)

                Button {
                    showLibrary = true
                } label: {
                    editorialIconButtonLabel(icon: "books.vertical")
                }
                .buttonStyle(PressableButtonStyle())
            }
        }
        .padding(.horizontal, gutter)
        .frame(maxWidth: contentMeasure)
        .frame(maxWidth: .infinity)
        .padding(.top, 16)
        .padding(.bottom, 18)
        .background(
            Self.paperBG
                .overlay(Rectangle().fill(Self.paperRule).frame(height: 0.5), alignment: .top)
        )
    }

    func editorialIconButton(icon: String, disabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            editorialIconButtonLabel(icon: icon)
        }
        .buttonStyle(PressableButtonStyle())
        .disabled(disabled)
        .opacity(disabled ? 0.35 : 1.0)
    }

    func editorialIconButtonLabel(icon: String) -> some View {
        Image(systemName: icon)
            .font(.system(size: 17, weight: .regular))
            .foregroundColor(Self.paperInk.opacity(0.70))
            .frame(width: 44, height: 44)
            .background(Circle().fill(Color.white.opacity(0.5)))
            .overlay(Circle().strokeBorder(Self.paperRule, lineWidth: 0.5))
    }

    // Language button shows the 2-letter code so users see which language is active
    var languageIndicatorLabel: some View {
        let display: String = {
            if selectedLanguage == "auto" { return "AUTO" }
            return selectedLanguage.uppercased()
        }()
        return Text(display)
            .font(.system(size: 11, weight: .semibold, design: .monospaced))
            .tracking(1)
            .foregroundColor(Self.paperInk.opacity(0.70))
            .frame(width: 44, height: 44)
            .background(Circle().fill(Color.white.opacity(0.5)))
            .overlay(Circle().strokeBorder(Self.paperRule, lineWidth: 0.5))
    }

    var modelPickerSheet: some View {
        NavigationStack {
            ScrollView {
                paperModelPicker
            }
            .background(Self.paperBG.ignoresSafeArea())
            .navigationTitle("Transcription Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { showModelPicker = false }
                        .foregroundColor(Self.paperAccent)
                }
            }
        }
    }

    // Paper-style model picker — replaces the old glass card grid
    var paperModelPicker: some View {
        VStack(alignment: .leading, spacing: 18) {
            // Status row
            HStack(spacing: 8) {
                Circle()
                    .fill(isModelLoaded ? Color(red: 0.30, green: 0.65, blue: 0.40) : Self.paperMute.opacity(0.4))
                    .frame(width: 7, height: 7)
                Text(isPreparingModel
                     ? (downloadProgress != nil ? "Downloading"
                        : (isOptimizingModel ? "Optimising" : "Loading"))
                     : (isModelLoaded ? "Ready" : "Choose a model to begin"))
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .tracking(1.5)
                    .textCase(.uppercase)
                    .foregroundColor(Self.paperMute)
                Spacer()
                Button(action: checkModelStatus) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(Self.paperMute)
                        .frame(width: 26, height: 26)
                        .overlay(Circle().strokeBorder(Self.paperRule, lineWidth: 0.5))
                }
                .buttonStyle(PressableButtonStyle())
            }

            modelPreparationStatus

            Text("Each model is downloaded once and runs on this device's Neural Engine. Larger models are slower but more accurate, especially across languages.")
                .font(.system(size: 13, design: .serif))
                .foregroundColor(Self.paperInk.opacity(0.6))
                .lineSpacing(3)

            // 2-column paper card grid
            LazyVGrid(
                columns: [GridItem(.flexible(), spacing: 12), GridItem(.flexible(), spacing: 12)],
                spacing: 12
            ) {
                ForEach(WhisperModel.allCases, id: \.self) { model in
                    paperModelCard(model)
                }
            }

            // Action button
            Button {
                if isPreparingModel {
                    modelPrepTask?.cancel()
                    isPreparingModel = false
                    statusMessage = ""
                } else {
                    prepareModel(selectedModel)
                }
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: isPreparingModel
                          ? "xmark"
                          : (downloadStatus[selectedModel] ?? false ? "arrow.clockwise" : "arrow.down"))
                        .font(.system(size: 14, weight: .bold))
                    Text(isPreparingModel
                         ? "Cancel"
                         : (downloadStatus[selectedModel] ?? false
                            ? "Reload \(selectedModel.displayName)"
                            : "Download \(selectedModel.displayName) · \(selectedModel.sizeLabel)"))
                        .font(.system(size: 15, weight: .semibold, design: .serif))
                }
                .foregroundColor(Self.paperBG)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 15)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(Self.paperInk)
                        .shadow(color: Self.paperInk.opacity(0.18), radius: 12, y: 5)
                )
            }
            .buttonStyle(PressableButtonStyle())
            .disabled(isProcessing && !isPreparingModel)
            .opacity((isProcessing && !isPreparingModel) ? 0.5 : 1)
            .padding(.top, 6)

            computeModePicker

            // Privacy footnote — reinforces that this is differentiated
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "lock.fill")
                    .font(.system(size: 10))
                    .foregroundColor(Self.paperMute)
                    .padding(.top, 2)
                Text("Models are fetched from Hugging Face once and cached on this device. Your audio never leaves the phone.")
                    .font(.system(size: 11, design: .serif).italic())
                    .foregroundColor(Self.paperMute)
                    .lineSpacing(2)
            }
            .padding(.top, 4)
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 22)
        .frame(maxWidth: 620)
        .frame(maxWidth: .infinity)
    }

    /// Processor selector. Exposed because the trade-off is real and personal:
    /// the Neural Engine saves battery but makes the first load after every
    /// model or OS change take minutes, which reads as a hang.
    var computeModePicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Rectangle()
                    .fill(Self.paperInk.opacity(0.20))
                    .frame(width: 18, height: 1)
                Text("Run on")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .tracking(2)
                    .foregroundColor(Self.paperMute)
            }

            ForEach(ComputeMode.allCases) { mode in
                Button {
                    selectComputeMode(mode)
                } label: {
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: mode == computeMode ? "largecircle.fill.circle" : "circle")
                            .font(.system(size: 15))
                            .foregroundColor(mode == computeMode ? Self.paperAccent : Self.paperMute.opacity(0.6))
                            .padding(.top, 1)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(mode.displayName)
                                .font(.system(size: 14, weight: .semibold, design: .serif))
                                .foregroundColor(Self.paperInk)
                            Text(mode.detail)
                                .font(.system(size: 11, design: .serif))
                                .foregroundColor(Self.paperInk.opacity(0.55))
                                .multilineTextAlignment(.leading)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.vertical, 7)
                    .contentShape(Rectangle())
                }
                .buttonStyle(PressableButtonStyle())
                .disabled(isPreparingModel || isProcessing)
                .opacity((isPreparingModel || isProcessing) ? 0.5 : 1)
            }
        }
        .padding(.top, 6)
    }

    func paperModelCard(_ model: WhisperModel) -> some View {
        let isSelected = (selectedModel == model)
        let isDownloaded = downloadStatus[model] ?? false
        let isActive = isLoaded(model)

        return Button {
            selectModel(model)
        } label: {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 6) {
                    Text(model.displayName)
                        .font(.system(size: 16, weight: .semibold, design: .serif))
                        .foregroundColor(Self.paperInk)
                    Spacer()
                    if isActive {
                        Image(systemName: "waveform")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(Self.paperAccent)
                    } else if isDownloaded {
                        Image(systemName: "checkmark")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(Color(red: 0.30, green: 0.65, blue: 0.40))
                    }
                }

                Text(model.tagline)
                    .font(.system(size: 12, design: .serif))
                    .foregroundColor(Self.paperInk.opacity(0.55))
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)

                HStack(spacing: 6) {
                    paperBadge(model.sizeLabel,
                               tint: isDownloaded ? Color(red: 0.30, green: 0.65, blue: 0.40) : Self.paperMute)
                    if isActive {
                        paperBadge("Loaded", tint: Self.paperAccent)
                    } else if isDownloaded {
                        paperBadge("On device", tint: Color(red: 0.30, green: 0.65, blue: 0.40))
                    }
                }

                // Per-card progress, so it's obvious which model is downloading.
                if isSelected, isPreparingModel, let p = downloadProgress {
                    ProgressView(value: p)
                        .progressViewStyle(.linear)
                        .tint(Self.paperAccent)
                }
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(isSelected ? Self.paperAccentSoft.opacity(0.45) : Self.paperCard)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(
                        isSelected ? Self.paperAccent : Self.paperRule,
                        lineWidth: isSelected ? 1.5 : 0.5
                    )
            )
        }
        .buttonStyle(PressableButtonStyle())
        .contextMenu {
            if isDownloaded {
                Button(role: .destructive) {
                    deleteModel(model)
                } label: {
                    Label("Delete download (\(model.sizeLabel))", systemImage: "trash")
                }
            }
        }
    }

    func paperBadge(_ text: String, tint: Color) -> some View {
        Text(text.uppercased())
            .font(.system(size: 9, weight: .bold, design: .monospaced))
            .tracking(1)
            .foregroundColor(tint)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(Capsule().fill(tint.opacity(0.10)))
    }

    func isLoaded(_ model: WhisperModel) -> Bool {
        return isModelLoaded && selectedModel == model
    }

    var languagePickerSheet: some View {
        NavigationStack {
            List {
                ForEach(supportedLanguages, id: \.code) { lang in
                    Button {
                        selectedLanguage = lang.code
                        showLanguagePicker = false
                    } label: {
                        HStack {
                            Text(lang.name)
                                .foregroundColor(Self.paperInk)
                            Spacer()
                            if selectedLanguage == lang.code {
                                Image(systemName: "checkmark")
                                    .foregroundColor(Self.paperAccent)
                            }
                        }
                    }
                }
            }
            .frame(maxWidth: 620)
            .frame(maxWidth: .infinity)
            .background(Self.paperBG.ignoresSafeArea())
            .navigationTitle("Language")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { showLanguagePicker = false }
                }
            }
        }
    }
    
    // MARK: - Transcript Section
    // Friendly display for a detected ISO language code from Whisper.
    func displayName(forLanguageCode code: String) -> String {
        if let match = supportedLanguages.first(where: { $0.code == code }) {
            return match.name
        }
        // Fallback: use Locale to resolve language name
        let locale = Locale(identifier: "en")
        if let name = locale.localizedString(forLanguageCode: code) {
            return name.prefix(1).uppercased() + name.dropFirst()
        }
        return code.uppercased()
    }
    
    // MARK: - Live Activity Functions

    func cleanupStaleActivities() {
        // Remove any stale activities from previous sessions
        if #available(iOS 16.2, *) {
            Task {
                let activities = Activity<TranscriptionAttributes>.activities
                if activities.count > 0 {
                    for activity in activities {
                        await activity.end(nil, dismissalPolicy: .immediate)
                    }
                }
            }
        }
    }

    func startLiveActivity() {
        // Only create if we don't already have an active one
        guard currentActivity == nil else {
            updateLiveActivity()
            return
        }

        if #available(iOS 16.2, *) {
            // End any existing activities first
            Task {
                for activity in Activity<TranscriptionAttributes>.activities {
                    await activity.end(nil, dismissalPolicy: .immediate)
                }

                // Now create new one
                await MainActor.run {
                    let attributes = TranscriptionAttributes(fileName: "Audio File")
                    let initialState = TranscriptionAttributes.ContentState(
                        progress: transcriptionProgress,
                        currentSegment: currentSegment,
                        totalSegments: totalSegments,
                        status: "Starting transcription..."
                    )

                    do {
                        currentActivity = try Activity.request(
                            attributes: attributes,
                            content: .init(state: initialState, staleDate: nil),
                            pushType: nil
                        )
                    } catch {
                        print("Failed to start Live Activity: \(error)")
                    }
                }
            }
        }
    }
    
    func updateLiveActivity() {
        guard let activity = currentActivity else { return }

        if #available(iOS 16.2, *) {
            Task {
                let updatedState = TranscriptionAttributes.ContentState(
                    progress: transcriptionProgress,
                    currentSegment: currentSegment,
                    totalSegments: totalSegments,
                    status: statusMessage.isEmpty ? "Processing..." : statusMessage
                )

                await activity.update(.init(state: updatedState, staleDate: nil))
            }
        }
    }
    
    func endLiveActivity() {
        if #available(iOS 16.2, *) {
            Task {
                if let activity = currentActivity {
                    // Update with final state before ending
                    let finalState = TranscriptionAttributes.ContentState(
                        progress: 1.0,
                        currentSegment: totalSegments,
                        totalSegments: totalSegments,
                        status: "Complete"
                    )

                    await activity.update(.init(state: finalState, staleDate: nil))

                    // End after a short delay to show completion
                    try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
                    await activity.end(nil, dismissalPolicy: .default)

                    await MainActor.run {
                        currentActivity = nil
                    }
                }
            }
        }
    }
    
    // MARK: - Background Processing

    func registerBackgroundTasks() {
        // Register for iOS 26+ BGContinuedProcessingTask
        if #available(iOS 16.0, *) {
            BGTaskScheduler.shared.register(forTaskWithIdentifier: "com.whisper.transcription", using: nil) { task in
                // This will be called when iOS 26+ uses BGContinuedProcessingTask
                self.handleBackgroundTranscription(task: task)
            }
        }
    }

    func handleBackgroundTranscription(task: BGTask) {
        // Setup progress reporting for BGContinuedProcessingTask
        let progress = Progress(totalUnitCount: 100)
        backgroundTaskProgress = progress

        task.expirationHandler = {
            // Task is about to expire
            task.setTaskCompleted(success: false)
        }

        // The actual transcription continues with progress updates
        // Progress is automatically synced through our existing progress reporting
    }

    func setupBackgroundAudio() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            // Use .playback for background audio processing
            // This keeps the app active in background for audio processing
            try audioSession.setCategory(.playback, mode: .spokenAudio, options: [.mixWithOthers, .duckOthers, .allowAirPlay])
            try audioSession.setActive(true)

            // Play silent audio to keep session active
            playSilentAudio()
        } catch {
            print("Failed to setup background audio: \(error)")
        }
    }

    func playSilentAudio() {
        // This keeps the audio session active in background
        // which helps maintain app processing capability
        Task {
            do {
                let audioSession = AVAudioSession.sharedInstance()
                try audioSession.setActive(true)
            } catch {
                print("Failed to activate audio session: \(error)")
            }
        }
    }
    
    func requestNotificationPermissions() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if let error = error {
                print("Notification permission error: \(error)")
            }
        }
    }
    
    func handleScenePhaseChange(oldPhase: ScenePhase, newPhase: ScenePhase) {
        switch newPhase {
        case .background:
            if isProcessing {
                // Ensure audio session is active
                setupBackgroundAudio()
                beginBackgroundTask()
                startLiveActivity()

                // Show alert about background limitations
                if !UserDefaults.standard.bool(forKey: "hasSeenBackgroundWarning") {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        let content = UNMutableNotificationContent()
                        content.title = "Processing in Background"
                        content.body = "Keep app in foreground for best performance. Background processing may be limited."
                        content.sound = .default

                        let request = UNNotificationRequest(identifier: "bg-warning", content: content, trigger: nil)
                        UNUserNotificationCenter.current().add(request)

                        UserDefaults.standard.set(true, forKey: "hasSeenBackgroundWarning")
                    }
                }
            }
        case .active:
            endBackgroundTask()
            // Reactivate audio session when returning to foreground
            if isProcessing {
                setupBackgroundAudio()
            }
        case .inactive:
            break
        @unknown default:
            break
        }
    }
    
    func beginBackgroundTask() {
        // For iOS 26+, submit BGContinuedProcessingTask
        if #available(iOS 16.0, *) {
            submitContinuedProcessingTask()
        }

        // Also use traditional background task as fallback
        backgroundTask = UIApplication.shared.beginBackgroundTask { [self] in
            self.endBackgroundTask()
        }
    }

    func submitContinuedProcessingTask() {
        if #available(iOS 16.0, *) {
            let request = BGProcessingTaskRequest(identifier: "com.whisper.transcription")
            request.requiresNetworkConnectivity = false
            request.requiresExternalPower = false

            do {
                try BGTaskScheduler.shared.submit(request)
            } catch {
                print("Could not schedule background task: \(error)")
            }
        }
    }
    
    func endBackgroundTask() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }
    
    func sendCompletionNotification() {
        guard scenePhase == .background else { return }

        let content = UNMutableNotificationContent()
        content.title = "Transcription Complete"
        content.body = "Your audio transcription has finished"
        content.sound = .default

        let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }
    
    // MARK: - Helper Functions
    
    func copyTranscript() {
        UIPasteboard.general.string = transcript
        showCopySuccess = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            showCopySuccess = false
        }
    }

    /// Clears any previous result and error so a new run starts from a blank page.
    func resetResult() {
        transcript = ""
        transcriptMeta = nil
        errorMessage = nil
    }

    /// Surfaces a failure in the dedicated error channel rather than smuggling
    /// it into the transcript text.
    func showError(_ message: String) {
        transcript = ""
        transcriptMeta = nil
        errorMessage = message
    }

    /// Explains an empty result in terms of what actually happened, rather than
    /// the flat "no speech detected" that gave the user nothing to act on.
    func reportEmptyTranscript(peak: Float?) {
        // -46 dBFS. Room tone and mic self-noise sit below this; anything
        // audible sits well above it.
        let silenceFloor: Float = 0.005

        if let peak, peak < silenceFloor {
            showError("""
            This recording is essentially silent — nothing reached the microphone.

            If you were capturing sound from another device's speaker, move the \
            two closer together and turn the source volume up. Also check that \
            Whisper has microphone access in Settings.
            """)
            return
        }

        let modelHint = selectedModel == .small
            ? "\n\nThe Small model struggles with quiet or accented speech — try Turbo for a much better result."
            : ""
        let languageHint = selectedLanguage == "auto"
            ? "\n\nAuto-detect can pick the wrong language on a short or noisy clip. Setting the language explicitly usually fixes it."
            : ""

        showError("The audio came through, but no speech could be transcribed from it.\(modelHint)\(languageHint)")
    }
    
    /// Switching models used to only do something when the model was already
    /// on disk — otherwise it silently changed the label while `whisperKit`
    /// kept holding the *previous* model, so the chip lied about what was
    /// actually transcribing. Now selecting always (re)prepares.
    func selectModel(_ model: WhisperModel) {
        let alreadyReady = (model == selectedModel && isModelLoaded && !isPreparingModel)
        selectedModel = model
        guard !alreadyReady else { return }
        prepareModel(model)
    }

    /// Downloads the model if needed, then loads it.
    ///
    /// Restarting a preparation that is already running for the same model is
    /// not free: cancelling mid-download leaves Hugging Face's partial
    /// `<name>.<sha>.incomplete` temp file behind, and the next attempt then
    /// fails trying to move a file the cancelled one had already cleaned up.
    /// That is the "couldn't be moved to whisper-large-v3" error. So a repeat
    /// request for the model already being prepared is a no-op.
    func prepareModel(_ model: WhisperModel, force: Bool = false) {
        if !force, isPreparingModel, preparingVariant == model { return }
        modelPrepTask?.cancel()
        isModelLoaded = false
        whisperKit = nil
        preparingVariant = model
        modelPrepTask = Task { await performModelPreparation(model) }
    }

    /// Clears *empty* partial-download placeholders.
    ///
    /// Deliberately narrow: a non-empty `.incomplete` file is resume state —
    /// the downloader measures it and continues with `Range: bytes=N-`, so
    /// deleting one would throw away real progress on a 646 MB model. Only the
    /// zero-byte placeholders the downloader writes up front are debris worth
    /// removing before a retry.
    func clearEmptyIncompleteDownloads() {
        let root = getModelsDirectory()
        guard let e = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return }
        for case let url as URL in e where url.pathExtension == "incomplete" {
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            if size == 0 {
                try? FileManager.default.removeItem(at: url)
            }
        }
    }

    /// Core ML specialises per compute-unit configuration, so a model prewarmed
    /// for the Neural Engine is *not* warm for the GPU. The mode belongs in the
    /// key, or switching would skip a prewarm that had never actually happened.
    private func prewarmKey(_ model: WhisperModel) -> String {
        "\(model.rawValue)@\(ProcessInfo.processInfo.operatingSystemVersionString)#\(computeMode.rawValue)"
    }

    private func hasBeenPrewarmed(_ model: WhisperModel) -> Bool {
        prewarmedVariantsRaw.split(separator: "\n").contains(Substring(prewarmKey(model)))
    }

    /// Drops a variant's prewarm marker. Core ML evicts its specialised-model
    /// cache on OS updates *and* after a model goes unused for a while, and
    /// there is no API to ask whether the cache is still warm. When a load
    /// stalls we assume it went cold and re-arm prewarming for next time.
    private func clearPrewarmed(_ model: WhisperModel) {
        let key = prewarmKey(model)
        prewarmedVariantsRaw = prewarmedVariantsRaw
            .split(separator: "\n")
            .filter { $0 != Substring(key) }
            .joined(separator: "\n")
    }

    private func markPrewarmed(_ model: WhisperModel) {
        guard !hasBeenPrewarmed(model) else { return }
        let key = prewarmKey(model)
        prewarmedVariantsRaw = prewarmedVariantsRaw.isEmpty ? key : prewarmedVariantsRaw + "\n" + key
    }

    @MainActor
    func performModelPreparation(_ model: WhisperModel) async {
        isPreparingModel = true
        downloadProgress = nil
        isOptimizingModel = false
        errorMessage = nil
        monitor.startSystemMonitoring()
        defer {
            isPreparingModel = false
            downloadProgress = nil
            isOptimizingModel = false
            preparingVariant = nil
            monitor.stopSystemMonitoring()
        }

        do {
            try await attemptModelPreparation(model)
        } catch is CancellationError {
            // Superseded by another selection — leave state to the newer task.
        } catch {
            if Task.isCancelled { return }

            // An interrupted transfer leaves a partial file behind and the next
            // attempt trips over it. Clear the debris and try once more before
            // bothering the user — this class of failure is transient, and the
            // retry is what they would have done by hand anyway.
            clearEmptyIncompleteDownloads()

            // A stall almost always means Core ML has to re-specialise the
            // model and did it the memory-hungry way, because we believed our
            // own "already prewarmed" note. Forget it so the retry prewarms —
            // that loads one sub-model at a time and keeps peak memory down.
            if error is ModelLoadTimeout {
                clearPrewarmed(model)
            }
            do {
                statusMessage = "Retrying \(model.displayName)…"
                try await attemptModelPreparation(model)
                return
            } catch is CancellationError {
                return
            } catch let retryError {
                if Task.isCancelled { return }
                // If something else finished the job while we were failing,
                // there is nothing to report.
                if whisperKit != nil, isModelLoaded { return }
                isModelLoaded = false
                statusMessage = ""
                showError("""
                Couldn't prepare \(model.displayName). This is usually a network \
                hiccup part-way through the download.

                Tap Download to try again — finished parts are kept, so it \
                resumes rather than starting over.

                \(retryError.localizedDescription)
                """)
            }
        }
    }

    @MainActor
    private func attemptModelPreparation(_ model: WhisperModel) async throws {
        let modelsDir = getModelsDirectory()
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        do {

            if findModelDirectory(for: model) == nil {
                statusMessage = "Downloading \(model.displayName) — \(model.sizeLabel)"
                downloadProgress = 0
                _ = try await WhisperKit.download(
                    variant: model.rawValue,
                    downloadBase: modelsDir,
                    progressCallback: { progress in
                        Task { @MainActor in
                            self.downloadProgress = progress.fractionCompleted
                        }
                    }
                )
                try Task.checkCancellation()
                downloadStatus[model] = true
            }

            // Loading reports no progress, so drop the bar.
            downloadProgress = nil

            // Two things used to be deferred to the user's first recording:
            //
            // 1. WhisperKit's `load` parameter resolves to
            //    `config.load ?? (config.modelFolder != nil)`. We pass `model:`
            //    rather than `modelFolder:`, so it defaulted to FALSE and the
            //    init never actually loaded anything — `runTranscribeTask` hit
            //    `modelState != .loaded` and loaded the whole model mid-recording,
            //    while `isModelLoaded` had already been showing "Ready".
            //
            // 2. Core ML specialises an .mlmodelc for the specific chip the
            //    first time it is loaded, caching the result outside the app.
            //    On a large model that alone is tens of seconds.
            //
            // Both now happen here, behind the progress UI. Prewarm runs only
            // for a variant this device hasn't specialised yet: it costs a
            // load-unload-load cycle, so paying it on every launch would make
            // the common case slower.
            // If the last load never settled, the app died during it. Prewarm
            // this time regardless of what our notes say: it loads one
            // sub-model at a time instead of all at once, which is the
            // difference between fitting in memory and being killed again.
            let crashedLastLoad = modelLoadInFlight
            if crashedLastLoad {
                clearPrewarmed(model)
            }

            let needsPrewarm = crashedLastLoad || !hasBeenPrewarmed(model)
            isOptimizingModel = needsPrewarm
            modelLoadInFlight = true
            statusMessage = needsPrewarm
                ? "Optimising \(model.displayName) for the \(computeMode.shortName)…"
                : "Loading \(model.displayName)…"

            // A load that never returns used to leave the UI on "Loading"
            // forever with no way forward. Race it against a deadline so a
            // stall becomes a recoverable error instead of a dead end.
            // Generous, because a cold Core ML specialisation of a large model
            // genuinely can take minutes on older hardware.
            let kit = try await withModelLoadTimeout(seconds: needsPrewarm ? 420 : 180) {
                try await WhisperKit(
                    model: model.rawValue,
                    downloadBase: modelsDir,
                    computeOptions: computeMode.computeOptions,
                    verbose: false,
                    logLevel: .error,
                    prewarm: needsPrewarm,
                    load: true
                )
            }
            try Task.checkCancellation()
            modelLoadInFlight = false
            markPrewarmed(model)
            isOptimizingModel = false

            whisperKit = kit
            isModelLoaded = true
            downloadStatus[model] = true
            statusMessage = ""
            resetResult()
        } catch {
            isOptimizingModel = false
            downloadProgress = nil
            // Cleared on failure too — only an outright kill should leave it set.
            modelLoadInFlight = false
            throw error
        }
    }

    /// Variants shipped by earlier versions. Anyone who had downloaded Medium
    /// or the unquantized Large builds is sitting on up to 7.8 GB of weights
    /// that nothing can load or delete any more, so retire them on launch.
    static let retiredModelVariants = [
        "openai_whisper-base",
        "openai_whisper-base.en",
        "openai_whisper-tiny",
        "openai_whisper-medium",
        "openai_whisper-large-v3",
        "openai_whisper-large-v3_turbo",
    ]

    func removeRetiredModelDownloads() {
        let root = getModelsDirectory()
            .appendingPathComponent("models/argmaxinc/whisperkit-coreml", isDirectory: true)
        let live = Set(WhisperModel.allCases.map(\.rawValue))

        for variant in Self.retiredModelVariants where !live.contains(variant) {
            let dir = root.appendingPathComponent(variant, isDirectory: true)
            guard FileManager.default.fileExists(atPath: dir.path) else { continue }
            do {
                try FileManager.default.removeItem(at: dir)
                print("Removed retired model download: \(variant)")
            } catch {
                print("Could not remove retired model \(variant): \(error)")
            }
        }
    }

    /// Runs `work`, cancelling it and failing with `ModelLoadTimeout` if it
    /// outlives `seconds`.
    ///
    /// Stays on the main actor throughout — `WhisperKit` is not `Sendable`, and
    /// keeping the result inside one isolation domain avoids passing it across
    /// one. Best-effort: it can only interrupt a load that honours
    /// cancellation, but that is the difference between an error the user can
    /// act on and a spinner that never ends.
    @MainActor
    private func withModelLoadTimeout<T>(
        seconds: Double,
        _ work: @escaping @MainActor () async throws -> T
    ) async throws -> T {
        let workTask = Task { @MainActor in try await work() }
        let watchdog = Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            if !Task.isCancelled { workTask.cancel() }
        }
        defer { watchdog.cancel() }

        do {
            return try await workTask.value
        } catch is CancellationError {
            // Ours if the watchdog fired; the caller's otherwise.
            if Task.isCancelled { throw CancellationError() }
            throw ModelLoadTimeout()
        }
    }

    /// Repairs transcripts saved by 1.1 (1)–(4), whose segments were stored
    /// with Whisper's special tokens still embedded. Without this, existing
    /// library entries keep showing `<|startoftranscript|>` and keep exporting
    /// broken subtitles even after the decoder fix.
    func sanitiseStoredSegmentsIfNeeded() {
        guard !didSanitiseStoredSegments else { return }
        guard let segments = try? modelContext.fetch(FetchDescriptor<SavedSegment>()) else { return }

        var repaired = 0
        for segment in segments where segment.text.contains("<|") {
            let cleaned = WhisperText.stripSpecialTokens(segment.text)
            if cleaned != segment.text {
                segment.text = cleaned
                repaired += 1
            }
        }
        if repaired > 0 {
            try? modelContext.save()
            print("Repaired \(repaired) segment(s) containing special tokens")
        }
        didSanitiseStoredSegments = true
    }

    /// Switching processor requires a fresh load — Core ML bakes the compute
    /// units into the loaded model.
    func selectComputeMode(_ mode: ComputeMode) {
        guard mode != computeMode else { return }
        computeMode = mode
        prepareModel(selectedModel, force: true)
    }

    /// Frees a downloaded model's files. The full lineup runs to ~2.9 GB, so
    /// reclaiming space needs to be possible in-app.
    func deleteModel(_ model: WhisperModel) {
        if let dir = findModelDirectory(for: model) {
            try? FileManager.default.removeItem(at: dir)
        }
        downloadStatus[model] = false
        if model == selectedModel {
            modelPrepTask?.cancel()
            whisperKit = nil
            isModelLoaded = false
        }
    }
    
    func getElapsedTime() -> TimeInterval {
        guard let startTime = processStartTime else { return 0 }
        return Date().timeIntervalSince(startTime)
    }
    
    func getRemainingTime() -> TimeInterval {
        guard transcriptionProgress > 0.05 else { return 0 }
        let elapsed = getElapsedTime()
        let total = elapsed / transcriptionProgress
        return max(0, total - elapsed)
    }
    
    func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        if minutes > 0 {
            return String(format: "%02d:%02d", minutes, seconds)
        }
        return String(format: "%ds", seconds)
    }
    
    func getProcessorInfo() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        return String(cString: machine)
    }
    
    func getTotalMemoryGB() -> String {
        let totalGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        return String(format: "%.1f", totalGB)
    }
    
    // MARK: - File Management
    
    func getDocumentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
    
    func getApplicationSupportDirectory() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    }
    
    func getModelsDirectory() -> URL {
        getDocumentsDirectory().appendingPathComponent("Models")
    }
    
    func getImportsDirectory() -> URL {
        getApplicationSupportDirectory().appendingPathComponent("Imports")
    }
    
    func createDebugFiles() {
        let appSupport = getApplicationSupportDirectory()
        let debugFile = appSupport.appendingPathComponent("debug_info.txt")
        
        try? FileManager.default.createDirectory(at: appSupport, withIntermediateDirectories: true)
        
        let debugInfo = """
        Whisper iOS App - Debug Information
        ====================================
        Launch Time: \(Date())
        Documents: \(getDocumentsDirectory().path)
        App Support: \(appSupport.path)
        Models Dir: \(getModelsDirectory().path)
        Imports Dir: \(getImportsDirectory().path)
        Processor: \(getProcessorInfo())
        Total Memory: \(getTotalMemoryGB()) GB
        GPU Device: \(monitor.metalDevice?.name ?? "Unknown")
        """
        
        try? debugInfo.write(to: debugFile, atomically: true, encoding: .utf8)
    }
    
    // MARK: - Model Management
    
    /// WhisperKit lays models out at <base>/models/<owner>/<repo>/<variant>.
    /// This used to enumerate the entire tree — every file of every downloaded
    /// model — once per model at launch, which is thousands of stat() calls
    /// against gigabytes of weights just to answer "is it there?". Probe the
    /// known layout first, and only fall back to a walk if that misses.
    func findModelDirectory(for model: WhisperModel) -> URL? {
        let modelsDir = getModelsDirectory()

        let direct = modelsDir
            .appendingPathComponent("models/argmaxinc/whisperkit-coreml", isDirectory: true)
            .appendingPathComponent(model.rawValue, isDirectory: true)
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: direct.path),
           !contents.isEmpty {
            return direct
        }

        guard let enumerator = FileManager.default.enumerator(
            at: modelsDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return nil }

        for case let url as URL in enumerator where url.lastPathComponent == model.rawValue {
            if let contents = try? FileManager.default.contentsOfDirectory(atPath: url.path),
               !contents.isEmpty {
                return url
            }
        }
        return nil
    }

    func checkModelStatus() {
        for model in WhisperModel.allCases {
            downloadStatus[model] = findModelDirectory(for: model) != nil
        }

        if downloadStatus[selectedModel] == true && !isModelLoaded && !isProcessing && !isPreparingModel {
            prepareModel(selectedModel)
        }
    }
    
    // MARK: - File Import
    
    func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let sourceURL = urls.first else { return }
            Task {
                await importFileDirectly(from: sourceURL)
            }
        case .failure(let error):
            showError("File selection failed: \(error.localizedDescription)")
        }
    }
    
    func importFileDirectly(from sourceURL: URL) async {
        let accessing = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if accessing {
                sourceURL.stopAccessingSecurityScopedResource()
            }
        }
        
        await MainActor.run {
            statusMessage = "Reading file..."
        }
        
        if FileManager.default.fileExists(atPath: sourceURL.path) {
            await importAndTranscribeFile(from: sourceURL)
            return
        }
        
        let coordinationResult: Result<URL, Error> = await withCheckedContinuation { continuation in
            var error: NSError?
            var coordinatedURL: URL?
            
            NSFileCoordinator().coordinate(
                readingItemAt: sourceURL,
                options: [.withoutChanges],
                error: &error
            ) { actualURL in
                coordinatedURL = actualURL
            }
            
            if let error = error {
                continuation.resume(returning: .failure(error))
            } else if let url = coordinatedURL {
                continuation.resume(returning: .success(url))
            } else {
                continuation.resume(returning: .failure(NSError(
                    domain: "FileCoordination",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to coordinate file access"]
                )))
            }
        }
        
        switch coordinationResult {
        case .success(let coordinatedURL):
            await importAndTranscribeFile(from: coordinatedURL)
            
        case .failure(let error):
            await MainActor.run {
                showError("Couldn't open that file: \(error.localizedDescription)\n\nIf it lives in iCloud, download it first, or import the video from Photos instead.")
                isProcessing = false
                statusMessage = ""
            }
        }
    }
    
    func handlePhotoSelection(_ item: PhotosPickerItem) {
        isProcessing = true
        statusMessage = "Loading from Photos..."
        
        Task {
            do {
                guard let movie = try await item.loadTransferable(type: Data.self) else {
                    await MainActor.run {
                        showError("Couldn't load that video from Photos.")
                        isProcessing = false
                        selectedPhotoItem = nil
                    }
                    return
                }
                
                let tempURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString)
                    .appendingPathExtension("mp4")
                
                try movie.write(to: tempURL)
                await importAndTranscribeFile(from: tempURL)
                try? FileManager.default.removeItem(at: tempURL)
                
                await MainActor.run {
                    selectedPhotoItem = nil
                }
            } catch {
                await MainActor.run {
                    showError("Photo import failed: \(error.localizedDescription)")
                    isProcessing = false
                    selectedPhotoItem = nil
                }
            }
        }
    }
    
    func importAndTranscribeFile(from sourceURL: URL) async {
        // End any existing activity before starting new one
        endLiveActivity()

        await MainActor.run {
            statusMessage = "Importing file..."
            isProcessing = true
            streamingTranscript = ""
            resetResult()
            transcriptionProgress = 0.0
            currentSegment = 0
            totalSegments = 0
            processStartTime = Date()
            lastProgressUpdate = Date()
            segmentsPerSecond = 0.0
            currentActivity = nil // Reset activity
        }
        
        do {
            guard FileManager.default.fileExists(atPath: sourceURL.path) else {
                throw NSError(domain: "FileNotFound", code: 404)
            }
            
            let importsDir = getImportsDirectory()
            try FileManager.default.createDirectory(at: importsDir, withIntermediateDirectories: true)
            
            let fileName = sourceURL.lastPathComponent
            var destinationURL = importsDir.appendingPathComponent(fileName)
            var counter = 1
            
            while FileManager.default.fileExists(atPath: destinationURL.path) {
                let nameWithoutExt = sourceURL.deletingPathExtension().lastPathComponent
                let ext = sourceURL.pathExtension
                let newName = "\(nameWithoutExt)_\(counter).\(ext)"
                destinationURL = importsDir.appendingPathComponent(newName)
                counter += 1
            }
            
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            
            await MainActor.run {
                statusMessage = "Transcribing audio..."
            }
            
            try await transcribeAudio(at: destinationURL)
            
        } catch {
            await MainActor.run {
                let detail = error.localizedDescription

                // iOS denies Neural Engine / GPU access to backgrounded apps, so
                // this failure needs a different explanation than a generic one.
                if detail.contains("ML Programs") || detail.contains("asynchronous prediction") {
                    showError("""
                    Background processing is limited. iOS restricts Neural Engine \
                    access while the app is in the background — keep Whisper in the \
                    foreground during transcription.

                    Details: \(detail)
                    """)
                } else {
                    showError("Import failed: \(detail)")
                }

                isProcessing = false
                statusMessage = ""
                monitor.stopSystemMonitoring()
                endBackgroundTask()
                endLiveActivity()
            }
        }
    }
    
    // MARK: - Audio Recording

    func toggleRecording() {
        if isRecording {
            stopRecordingAndTranscribe()
        } else {
            startRecording()
        }
    }

    func startRecording() {
        let session = AVAudioSession.sharedInstance()
        AVAudioApplication.requestRecordPermission { granted in
            DispatchQueue.main.async {
                guard granted else {
                    showError("Microphone access is off. Enable it in Settings › Whisper.")
                    return
                }
                do {
                    // Mac Catalyst / Designed-for-iPad-on-Mac doesn't support
                    // .defaultToSpeaker; configure category defensively.
                    #if targetEnvironment(macCatalyst)
                    try? session.setCategory(.playAndRecord, mode: .default, options: [])
                    #else
                    try? session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothHFP])
                    #endif
                    try session.setActive(true)

                    // Record in Linear PCM / WAV. AAC on Designed-for-iPad-on-Mac
                    // fails AVAudioFile open (error 1685348671 = 'dta?'). LPCM
                    // is universally readable and is what Whisper consumes
                    // internally anyway, so no quality loss.
                    let url = FileManager.default.temporaryDirectory
                        .appendingPathComponent("recording_\(UUID().uuidString).wav")

                    let settings: [String: Any] = [
                        AVFormatIDKey: Int(kAudioFormatLinearPCM),
                        AVSampleRateKey: 16000.0,
                        AVNumberOfChannelsKey: 1,
                        AVLinearPCMBitDepthKey: 16,
                        AVLinearPCMIsBigEndianKey: false,
                        AVLinearPCMIsFloatKey: false,
                        AVLinearPCMIsNonInterleaved: false
                    ]

                    let recorder = try AVAudioRecorder(url: url, settings: settings)
                    recorder.prepareToRecord()
                    recorder.record()

                    audioRecorder = recorder
                    recordingURL = url
                    isRecording = true
                    recordingStartTime = Date()
                    resetResult()
                    streamingTranscript = ""
                } catch {
                    showError("Couldn't start recording: \(error.localizedDescription)")
                }
            }
        }
    }

    func stopRecordingAndTranscribe() {
        // Finalize recorder
        audioRecorder?.stop()
        let recorder = audioRecorder
        audioRecorder = nil
        recordingStartTime = nil
        isRecording = false

        guard let url = recordingURL else { return }
        recordingURL = nil

        // Release the .playAndRecord session so transcription has full Neural Engine + audio I/O
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            // Non-fatal — log but continue.
            print("⚠️ Audio session deactivation warning: \(error)")
        }

        Task {
            // Wait until the AAC encoder has fully flushed the file to disk.
            // AVAudioRecorder.stop() is documented as synchronous, but in practice
            // the file can be momentarily unreadable / 0 bytes right after.
            let ready = await waitForRecordingFile(at: url, recorder: recorder)
            guard ready else {
                await MainActor.run {
                    showError("That recording was too short to save. Record at least one second of audio.")
                    isProcessing = false
                    statusMessage = ""
                }
                return
            }

            await importAndTranscribeFile(from: url)
        }
    }

    /// Polls the recording file until it has a usable size (or times out).
    /// Returns false if the file never reached a viable state.
    func waitForRecordingFile(at url: URL, recorder: AVAudioRecorder?, timeout: TimeInterval = 3.0) async -> Bool {
        let start = Date()
        // LPCM 16 kHz mono 16-bit ≈ 32 KB / second. 4 KB ≈ 125 ms of real audio
        // plus the WAV header — anything less is a misclick.
        let minBytes: Int64 = 4_000
        var lastSize: Int64 = -1
        var stableHits = 0

        while Date().timeIntervalSince(start) < timeout {
            let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
            let size = (attrs?[.size] as? Int64) ?? 0

            if size >= minBytes {
                // Wait for size to stop changing for two consecutive polls (encoder flushed).
                if size == lastSize {
                    stableHits += 1
                    if stableHits >= 2 { return true }
                } else {
                    stableHits = 0
                }
                lastSize = size
            }

            // Hint the recorder to release file handles if still alive
            _ = recorder?.isRecording

            try? await Task.sleep(nanoseconds: 80_000_000) // 80 ms
        }
        return lastSize >= minBytes
    }

    // MARK: - SwiftData Library Persistence

    func saveTranscriptToLibrary(audioURL: URL, results: [TranscriptionResult], fullText: String, waveform: [Float]? = nil) {
        // Move audio file into the durable SavedAudio folder so the user can
        // still play it back even after temp gets cleaned.
        var savedRelativePath: String? = nil
        let stampFormatter = DateFormatter()
        stampFormatter.dateFormat = "yyyy-MM-dd_HHmm"
        let stamp = stampFormatter.string(from: Date())
        do {
            savedRelativePath = try AudioFiles.saveAudio(from: audioURL, suggestedName: "Recording_\(stamp)")
        } catch {
            print("⚠️ Could not persist audio file: \(error)")
        }

        // Compute total duration from the furthest segment end across all results.
        var duration: Double = 0
        for r in results {
            if let last = r.segments.last { duration = max(duration, Double(last.end)) }
        }

        // Detected language from first result, or fall back to the user's selected language.
        let detectedCode = results.first?.language
        let langForRecord: String? = (selectedLanguage == "auto") ? detectedCode : selectedLanguage

        // Title — short snippet of the transcript, or timestamp.
        let snippet = fullText
            .components(separatedBy: .newlines)
            .first?
            .trimmingCharacters(in: .whitespaces) ?? ""
        let title: String
        if !snippet.isEmpty {
            let trimmed = snippet.count > 50 ? String(snippet.prefix(50)) + "…" : snippet
            title = trimmed
        } else {
            title = "Transcript \(stamp)"
        }

        let record = SavedTranscript(
            title: title,
            duration: duration,
            languageCode: langForRecord,
            modelName: selectedModel.displayName,
            fullText: fullText,
            audioFilePath: savedRelativePath,
            waveform: waveform
        )

        // Flatten WhisperKit segments across all results into our SavedSegment rows.
        var segs: [SavedSegment] = []
        for r in results {
            for s in r.segments {
                let cleaned = WhisperText.stripSpecialTokens(s.text)
                guard !cleaned.isEmpty else { continue }
                segs.append(SavedSegment(
                    startTime: Double(s.start),
                    endTime: Double(s.end),
                    text: cleaned
                ))
            }
        }
        record.segments = segs

        modelContext.insert(record)
        do {
            try modelContext.save()
        } catch {
            print("⚠️ Failed to save transcript: \(error)")
        }
    }
    
    // MARK: - Transcription
    
    func transcribeAudio(at url: URL) async throws {
        await MainActor.run {
            transcriptionProgress = 0.0
            streamingTranscript = ""
            processStartTime = Date()
            lastProgressUpdate = Date()
            monitor.startSystemMonitoring()

            // Only start Live Activity if in background
            if scenePhase == .background {
                startLiveActivity()
            }
        }

        // The capture buttons stay live while the model loads, so by the time
        // we get here the load may still be running. Join it rather than
        // failing — the user's audio is already recorded and waiting.
        if whisperKit == nil, let prep = modelPrepTask {
            await MainActor.run {
                statusMessage = "Waiting for \(selectedModel.displayName) to finish loading…"
            }
            _ = await prep.value
        }

        guard let whisper = whisperKit else {
            throw NSError(
                domain: "Whisper",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey:
                    "The transcription model isn't loaded. Open the model picker and tap Reload, then try again."]
            )
        }

        // Ensure we're keeping the app active
        await MainActor.run {
            setupBackgroundAudio()
        }

        // Convert any input to 16 kHz mono 16-bit PCM WAV. This avoids
        // ExtAudioFile's broken AAC decoder on Designed-for-iPad-on-Mac
        // (error 1685348671 = 'dta?') and is what Whisper consumes
        // internally anyway, so no quality loss and faster startup.
        let workURL: URL
        do {
            workURL = try await AudioConverter.convertToWhisperReadableWAV(url)
        } catch {
            await MainActor.run {
                statusMessage = "Audio conversion failed: \(error.localizedDescription)"
            }
            throw error
        }

        // One pass over the converted WAV yields both the level (so a failure can
        // say which thing went wrong) and the waveform envelope. Measured on
        // workURL, not the original: an import from Photos is an .mp4, and
        // AVAudioFile cannot open a video container — reading the original would
        // silently leave every such transcript without a waveform.
        let envelope = AudioConverter.peakEnvelope(of: workURL)
        let peakLevel = envelope?.peak

        let languageCode = selectedLanguage == "auto" ? nil : selectedLanguage
        let shouldDetectLanguage = (languageCode == nil)

        // Whisper's compression-ratio check exists to catch repetition loops.
        // It is computed over the UTF-8 bytes of the decoded text, and CJK
        // characters are three bytes each and compress extremely well — so
        // perfectly good Chinese, Japanese and Korean routinely score above the
        // 2.4 default, get treated as failed decodes, exhaust the temperature
        // fallbacks and end up dropped. That is why Chinese came back as
        // "no speech detected". Disable the check for those scripts and lean on
        // the log-prob and no-speech gates instead.
        let isCJK = ["zh", "ja", "ko"].contains(languageCode ?? "")

        // Audio captured off another device's speaker is quiet and reverberant,
        // which pushes average log-prob down and no-speech probability up. Both
        // gates are loosened so far-field recordings aren't silently discarded.
        let options = DecodingOptions(
            verbose: false,
            task: .transcribe,
            language: languageCode,
            temperature: 0.0,
            temperatureFallbackCount: 5,
            detectLanguage: shouldDetectLanguage,
            skipSpecialTokens: true,
            compressionRatioThreshold: isCJK ? nil : 2.8,
            logProbThreshold: -1.5,
            noSpeechThreshold: 0.8
        )

        let audioFile = try? AVAudioFile(forReading: workURL)
        let duration = Double(audioFile?.length ?? 0) / (audioFile?.fileFormat.sampleRate ?? 1.0)
        
        await MainActor.run {
            totalSegments = max(1, Int(ceil(duration / 30.0)))
        }
        
        var lastReportedProgress = 0.0
        // WhisperKit invokes this callback for every decoded token. Hopping to
        // the main actor that often — to format a status string and republish a
        // growing transcript — is what made the UI stutter while transcribing.
        // Coalesce to 10 Hz; the final state is written after transcribe()
        // returns, so dropping trailing intermediate frames costs nothing.
        let throttle = ProgressThrottle(interval: 0.1)

        let results = try await whisper.transcribe(
            audioPath: workURL.path,
            decodeOptions: options,
            callback: { progress in
                guard throttle.shouldPush() else { return nil }
                let windowId = progress.windowId
                let text = progress.text

                Task { @MainActor in
                    let currentTime = Date()

                    // Update segment counter
                    self.currentSegment = windowId + 1  // +1 for 1-based counting
                    
                    // Calculate precise progress
                    if self.totalSegments > 0 {
                        let rawProgress = Double(windowId + 1) / Double(self.totalSegments)
                        // Clamp between 0 and 0.99 (never show 100% until complete)
                        self.transcriptionProgress = min(0.99, max(0.0, rawProgress))

                        // Update background task progress
                        if let bgProgress = self.backgroundTaskProgress {
                            bgProgress.completedUnitCount = Int64(self.transcriptionProgress * 100)
                        }
                    }
                    
                    // Update streaming text
                    if !text.isEmpty {
                        self.streamingTranscript = text
                    }

                    // Segments per second, measured on actual decode windows
                    // rather than callback count.
                    let totalElapsed = currentTime.timeIntervalSince(self.processStartTime ?? currentTime)
                    if totalElapsed > 0 {
                        self.segmentsPerSecond = Double(self.currentSegment) / totalElapsed
                    }
                    
                    // Update status message
                    let elapsed = self.getElapsedTime()
                    let remaining = self.getRemainingTime()
                    self.statusMessage = String(format: "Processing: %.0f%% | %d/%d segments | %.1f seg/s | %@ elapsed | %@ remaining",
                                               self.transcriptionProgress * 100,
                                               self.currentSegment,
                                               self.totalSegments,
                                               self.segmentsPerSecond,
                                               self.formatDuration(elapsed),
                                               self.formatDuration(remaining))
                    
                    // Update Live Activity every 5% progress change (only if in background)
                    if self.scenePhase == .background && abs(self.transcriptionProgress - lastReportedProgress) >= 0.05 {
                        self.updateLiveActivity()
                        lastReportedProgress = self.transcriptionProgress
                    }
                }
                return nil
            }
        )
        
        await MainActor.run {
            transcriptionProgress = 1.0
            
            if scenePhase == .background {
                sendCompletionNotification()
            }
            
            if !results.isEmpty {
                // Trim before testing — a result made only of whitespace used to
                // sail through this check and land an empty transcript in the
                // library.
                let fullText = results
                    .map { $0.text }
                    .joined(separator: "\n")
                    .trimmingCharacters(in: .whitespacesAndNewlines)

                if !fullText.isEmpty {
                    let detectedLanguage = results.first?.language ?? "Unknown"
                    if selectedLanguage == "auto" {
                        self.detectedLanguageCode = results.first?.language
                    } else {
                        self.detectedLanguageCode = nil
                    }

                    // Persist to SwiftData library
                    self.lastTranscribedAudioURL = url
                    self.lastTranscriptionResults = results
                    self.saveTranscriptToLibrary(audioURL: url, results: results, fullText: fullText, waveform: envelope?.buckets)
                    
                    var totalDuration: Double = 0.0
                    for result in results {
                        if !result.segments.isEmpty, let lastSegment = result.segments.last {
                            totalDuration = max(totalDuration, Double(lastSegment.end))
                        }
                    }
                    
                    let selectedLangName = supportedLanguages.first(where: { $0.code == selectedLanguage })?.name ?? "Auto Detect"

                    transcript = fullText
                    transcriptMeta = TranscriptMeta(
                        detectedLanguage: detectedLanguage == "Unknown" ? nil : detectedLanguage,
                        selectedLanguageName: selectedLangName,
                        audioDuration: totalDuration,
                        processingTime: getElapsedTime(),
                        modelName: selectedModel.displayName,
                        segmentCount: results.count
                    )
                    errorMessage = nil
                } else {
                    reportEmptyTranscript(peak: peakLevel)
                }
            } else {
                reportEmptyTranscript(peak: peakLevel)
            }

            streamingTranscript = ""
            isProcessing = false
            statusMessage = ""
            transcriptionProgress = 0.0
            monitor.stopSystemMonitoring()
            endBackgroundTask()
            endLiveActivity()
        }
    }
}



// MARK: - Progress Throttle

/// Coalesces a high-frequency callback down to a fixed rate.
///
/// WhisperKit reports progress per decoded token. Forwarding each one to the
/// main actor meant hundreds of view invalidations a second while a
/// transcription ran. Accessed only from WhisperKit's serial decode thread, so
/// the unsynchronised state is safe.
final class ProgressThrottle: @unchecked Sendable {
    private let interval: TimeInterval
    private var lastPush: Date = .distantPast

    init(interval: TimeInterval) {
        self.interval = interval
    }

    /// True at most once per `interval`.
    func shouldPush() -> Bool {
        let now = Date()
        guard now.timeIntervalSince(lastPush) >= interval else { return false }
        lastPush = now
        return true
    }
}

// MARK: - Pressable Button Style
struct PressableButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .brightness(configuration.isPressed ? -0.04 : 0)
            .animation(.spring(response: 0.25, dampingFraction: 0.7), value: configuration.isPressed)
    }
}

// MARK: - Live Activity Attributes (For Main App)
@available(iOS 16.2, *)
struct TranscriptionAttributes: ActivityAttributes {
    public struct ContentState: Codable, Hashable {
        var progress: Double
        var currentSegment: Int
        var totalSegments: Int
        var status: String
    }
    
    var fileName: String
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
