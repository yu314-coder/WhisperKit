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
    // Previous tick counters for instantaneous CPU & task time deltas.
    // Without this baseline the sampler returns the cumulative since-boot
    // average, which is nearly constant — making the chart look frozen.
    @State private var lastHostCPUTicks: (user: UInt64, system: UInt64, nice: UInt64, idle: UInt64) = (0, 0, 0, 0)
    @State private var lastTaskTimeSeconds: Double = 0
    @State private var lastSampleAt: Date? = nil
    @State private var showLibrary: Bool = false
    @State private var showModelPicker: Bool = false
    @State private var showLanguagePicker: Bool = false
    @State private var lastTranscribedAudioURL: URL? = nil
    @State private var lastTranscriptionResults: [TranscriptionResult] = []
    @State private var showModelDetails: Bool = false
    @State private var detectedLanguageCode: String? = nil
    @State private var transcript: String = ""
    @State private var streamingTranscript: String = ""
    @State private var isProcessing = false
    @State private var isModelLoaded = false
    @AppStorage("selectedModel") private var selectedModelRaw: String = WhisperModel.base.rawValue
    private var selectedModel: WhisperModel {
        get { WhisperModel(rawValue: selectedModelRaw) ?? .base }
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
    @State private var systemStatsHistory: [SystemStatsSnapshot] = []
    @State private var showCopySuccess = false
    
    // TQDM-style progress tracking
    @State private var processStartTime: Date?
    @State private var estimatedTotalTime: TimeInterval = 0
    @State private var segmentsPerSecond: Double = 0
    @State private var lastProgressUpdate: Date?
    
    // MARK: - WhisperKit
    @State private var whisperKit: WhisperKit?
    
    // Real-time monitoring
    @State private var monitoringTimer: Timer?
    @State private var metalDevice: MTLDevice?
    
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
    @State private var recordingElapsed: TimeInterval = 0
    @State private var recordingTimer: Timer?
    
    // MARK: - Enhanced System Stats
    struct SystemStatsSnapshot: Identifiable {
        let id = UUID()
        let timestamp: Date
        let cpuUsage: Double
        let memoryUsage: Double
        let memoryPercent: Double
        let gpuUsage: Double
        let networkIO: Double
    }
    
    // MARK: - Models
    enum WhisperModel: String, CaseIterable {
        case base = "openai_whisper-base"
        case medium = "openai_whisper-medium"
        case largeV3 = "openai_whisper-large-v3"
        case largeV3Turbo = "openai_whisper-large-v3_turbo"
        
        var displayName: String {
            switch self {
            case .base: return "Base"
            case .medium: return "Medium"
            case .largeV3: return "Large V3"
            case .largeV3Turbo: return "Large V3 Turbo"
            }
        }
        
        var description: String {
            switch self {
            case .base: return "Fastest • ~150MB"
            case .medium: return "Balanced • ~750MB"
            case .largeV3: return "Best Quality • ~950MB"
            case .largeV3Turbo: return "Fast & Accurate • ~950MB"
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
            .sheet(isPresented: $showLibrary) {
                TranscriptLibraryView()
            }
        .fileImporter(
            isPresented: $showingFilePicker,
            allowedContentTypes: [.audio, .movie, .mpeg4Movie, .quickTimeMovie, .mpeg4Audio],
            allowsMultipleSelection: false
        ) { result in
            handleFileImport(result)
        }
        .onAppear {
            checkModelStatus()
            createDebugFiles()
            initializeGPUMonitoring()
            setupBackgroundAudio()
            requestNotificationPermissions()
            registerBackgroundTasks()
            cleanupStaleActivities()
        }
        .onDisappear {
            stopSystemMonitoring()
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
    
    // MARK: - Background
    var appBackground: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.93, green: 0.95, blue: 1.0),
                    Color(red: 0.97, green: 0.94, blue: 1.0),
                    Color(red: 0.99, green: 0.96, blue: 0.99)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            // Soft accent orbs for depth
            Circle()
                .fill(Color.blue.opacity(0.18))
                .frame(width: 320, height: 320)
                .blur(radius: 90)
                .offset(x: -160, y: -260)

            Circle()
                .fill(Color.purple.opacity(0.16))
                .frame(width: 360, height: 360)
                .blur(radius: 100)
                .offset(x: 180, y: 320)

            Circle()
                .fill(Color.pink.opacity(0.10))
                .frame(width: 240, height: 240)
                .blur(radius: 80)
                .offset(x: 200, y: -180)
        }
    }

    // MARK: - Compact Model Pill (iPhone, when model loaded)
    var compactModelPill: some View {
        Button {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.85)) {
                showModelDetails = true
            }
        } label: {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [Color.green.opacity(0.25), Color.green.opacity(0.10)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 36, height: 36)
                    Image(systemName: "checkmark.seal.fill")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [Color.green, Color(red: 0.10, green: 0.55, blue: 0.40)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Model Ready")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.primary)
                    Text(selectedModel.displayName)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.down")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.secondary.opacity(0.7))
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color.white.opacity(0.4))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .strokeBorder(
                                LinearGradient(
                                    colors: [Color.white.opacity(0.8), Color.white.opacity(0.15)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    )
                    .shadow(color: .black.opacity(0.05), radius: 12, x: 0, y: 4)
            )
        }
        .buttonStyle(PressableButtonStyle())
    }

    // MARK: - Palette (warm-paper aesthetic)
    static let paperBG       = Color(red: 0.980, green: 0.965, blue: 0.940) // #FAF6F0
    static let paperInk      = Color(red: 0.110, green: 0.095, blue: 0.080) // warm near-black
    static let paperMute     = Color(red: 0.520, green: 0.495, blue: 0.460) // warm gray
    static let paperRule     = Color(red: 0.000, green: 0.000, blue: 0.000).opacity(0.10)
    static let paperAccent   = Color(red: 0.780, green: 0.290, blue: 0.180) // terracotta
    static let paperAccentSoft = Color(red: 0.940, green: 0.825, blue: 0.770) // pale terracotta

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
        .preferredColorScheme(.light)
        .sheet(isPresented: $showModelPicker) {
            modelPickerSheet
        }
        .sheet(isPresented: $showLanguagePicker) {
            languagePickerSheet
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
        .padding(.horizontal, 22)
        .padding(.top, 8)
        .padding(.bottom, 14)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Self.paperRule).frame(height: 0.5)
        }
    }

    // Transcript canvas — large serif body text on warm paper
    var editorialTranscriptCanvas: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                if isProcessing {
                    if !streamingTranscript.isEmpty {
                        liveTranscriptView
                    } else {
                        editorialProcessingState
                    }
                } else if !transcript.isEmpty {
                    finishedTranscriptView
                } else {
                    editorialEmptyState
                }
            }
            .padding(.horizontal, 26)
            .padding(.top, 32)
            .padding(.bottom, 28)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxHeight: .infinity)
    }

    var editorialEmptyState: some View {
        VStack(alignment: .leading, spacing: 22) {
            // Date — gives the page a sense of "today"
            Text(todayPaperDate())
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .tracking(2)
                .foregroundColor(Self.paperMute)

            Text("A quiet place\nfor your voice.")
                .font(.system(size: 36, weight: .semibold, design: .serif))
                .foregroundColor(Self.paperInk)
                .lineSpacing(2)

            Text(isModelLoaded
                 ? "Press the round button below to start a new recording, or pull in an audio file. Everything you transcribe stays on this device — every page is kept in your library."
                 : "First, choose a transcription model. It downloads once and runs on this device's Neural Engine. Larger models are slower but more accurate.")
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
            } else {
                inlineModelPicker
                    .padding(.top, 8)
            }

            Spacer(minLength: 60)
        }
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

            Button(action: downloadSelectedModel) {
                HStack(spacing: 10) {
                    Image(systemName: downloadStatus[selectedModel] ?? false ? "arrow.clockwise" : "arrow.down")
                        .font(.system(size: 13, weight: .bold))
                    Text(downloadStatus[selectedModel] ?? false ? "Reload \(selectedModel.displayName)" : "Download \(selectedModel.displayName)")
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
            .disabled(isProcessing)
            .opacity(isProcessing ? 0.5 : 1)
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
            paperPerfMonitor
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

    var paperPerfMonitor: some View {
        let latest = systemStatsHistory.last
        let elapsed = max(0, systemStatsHistory.count - 1)
        return VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 6) {
                Rectangle()
                    .fill(Self.paperInk.opacity(0.20))
                    .frame(width: 18, height: 1)
                Text("Live performance")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .tracking(2)
                    .foregroundColor(Self.paperMute)
                Spacer()
                Text("\(elapsed)s")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundColor(Self.paperMute.opacity(0.75))
            }

            paperPerfChannel(
                label: "Processor",
                accent: Self.perfCPU,
                value: latest?.cpuUsage ?? 0,
                samples: systemStatsHistory.map { $0.cpuUsage },
                primaryUnit: "%",
                secondary: nil
            )

            Divider().background(Self.paperRule)

            paperPerfChannel(
                label: "Memory",
                accent: Self.perfMem,
                value: latest?.memoryPercent ?? 0,
                samples: systemStatsHistory.map { $0.memoryPercent },
                primaryUnit: "%",
                secondary: String(format: "%.0f MB", latest?.memoryUsage ?? 0)
            )

            Divider().background(Self.paperRule)

            paperPerfChannel(
                label: "Neural Engine",
                accent: Self.perfNPU,
                value: latest?.gpuUsage ?? 0,
                samples: systemStatsHistory.map { $0.gpuUsage },
                primaryUnit: "%",
                secondary: nil
            )
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.white.opacity(0.62))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Self.paperRule, lineWidth: 0.5)
        )
        .shadow(color: Self.paperInk.opacity(0.04), radius: 8, y: 2)
    }

    func paperPerfChannel(
        label: String,
        accent: Color,
        value: Double,
        samples: [Double],
        primaryUnit: String,
        secondary: String?
    ) -> some View {
        let peak = samples.max() ?? 0
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                // Channel dot
                Circle()
                    .fill(accent)
                    .frame(width: 7, height: 7)

                Text(label.uppercased())
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .tracking(1.5)
                    .foregroundColor(Self.paperMute)

                Spacer()

                // Peak chip
                if samples.count > 1 {
                    HStack(spacing: 3) {
                        Text("peak")
                            .font(.system(size: 9, weight: .medium, design: .monospaced))
                            .foregroundColor(Self.paperMute.opacity(0.75))
                        Text(String(format: "%.0f%@", peak, primaryUnit))
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundColor(accent.opacity(0.85))
                    }
                }
            }

            HStack(alignment: .firstTextBaseline, spacing: 0) {
                // Hero value
                Text(String(format: "%.0f", value))
                    .font(.system(size: 28, weight: .semibold, design: .serif))
                    .foregroundColor(Self.paperInk)
                    .monospacedDigit()
                Text(primaryUnit)
                    .font(.system(size: 14, weight: .medium, design: .serif))
                    .foregroundColor(Self.paperInk.opacity(0.55))
                    .padding(.leading, 1)
                if let s = secondary {
                    Text("· \(s)")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundColor(Self.paperMute.opacity(0.85))
                        .padding(.leading, 6)
                }
                Spacer()
            }

            // Sparkline with peak rule
            GeometryReader { geo in
                paperSparklineSmooth(samples: samples, accent: accent, in: geo.size)
            }
            .frame(height: 34)
        }
    }

    @ViewBuilder
    func paperSparklineSmooth(samples: [Double], accent: Color, in size: CGSize) -> some View {
        let h = size.height
        let w = size.width
        if samples.count < 2 {
            // Empty baseline — dotted hairline so the chart has visual presence
            Path { p in
                p.move(to: CGPoint(x: 0, y: h - 1))
                p.addLine(to: CGPoint(x: w, y: h - 1))
            }
            .stroke(Self.paperRule, style: StrokeStyle(lineWidth: 1, dash: [2, 3]))
        } else {
            // Scale to a generous max so a 100% trace doesn't kiss the ceiling
            let maxVal = max(samples.max() ?? 100, 1) * 1.1
            let stepX = w / CGFloat(samples.count - 1)
            let points: [CGPoint] = samples.enumerated().map { (i, v) in
                let x = CGFloat(i) * stepX
                let y = h - CGFloat(v / maxVal) * (h - 2)
                return CGPoint(x: x, y: y)
            }
            ZStack {
                // Gradient fill below curve
                paperSmoothPath(points: points, closeAtY: h)
                    .fill(
                        LinearGradient(
                            colors: [accent.opacity(0.22), accent.opacity(0.02)],
                            startPoint: .top, endPoint: .bottom
                        )
                    )

                // Smooth curve stroke
                paperSmoothPath(points: points, closeAtY: nil)
                    .stroke(accent, style: StrokeStyle(lineWidth: 1.4, lineCap: .round, lineJoin: .round))

                // Latest point indicator
                if let last = points.last {
                    Circle()
                        .fill(accent)
                        .frame(width: 5, height: 5)
                        .position(last)
                    Circle()
                        .stroke(accent.opacity(0.35), lineWidth: 4)
                        .frame(width: 12, height: 12)
                        .position(last)
                }
            }
        }
    }

    // Catmull-Rom-ish smooth path through points (with optional fill close)
    private func paperSmoothPath(points: [CGPoint], closeAtY: CGFloat?) -> Path {
        var path = Path()
        guard let first = points.first else { return path }

        if let baseY = closeAtY {
            path.move(to: CGPoint(x: first.x, y: baseY))
            path.addLine(to: first)
        } else {
            path.move(to: first)
        }

        for i in 1..<points.count {
            let p0 = points[i - 1]
            let p1 = points[i]
            let midX = (p0.x + p1.x) / 2
            let control1 = CGPoint(x: midX, y: p0.y)
            let control2 = CGPoint(x: midX, y: p1.y)
            path.addCurve(to: p1, control1: control1, control2: control2)
        }

        if let baseY = closeAtY, let last = points.last {
            path.addLine(to: CGPoint(x: last.x, y: baseY))
            path.closeSubpath()
        }
        return path
    }

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
                .font(.system(size: 20, weight: .regular, design: .serif))
                .foregroundColor(Self.paperInk)
                .lineSpacing(7)

            if transcriptionProgress > 0 {
                ProgressView(value: transcriptionProgress)
                    .progressViewStyle(.linear)
                    .tint(Self.paperAccent)
                    .padding(.top, 4)
            }

            paperPerfMonitor
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
            }

            Text(transcript)
                .font(.system(size: 19, weight: .regular, design: .serif))
                .foregroundColor(Self.paperInk)
                .lineSpacing(8)
                .textSelection(.enabled)
        }
    }

    // Stamp like "MAY 19, 2026"
    func todayPaperDate() -> String {
        let f = DateFormatter()
        f.dateFormat = "MMM d, yyyy"
        return f.string(from: Date()).uppercased()
    }

    // Bottom control bar — paper aesthetic, hero round record button
    var editorialControlBar: some View {
        VStack(spacing: 0) {
            // Recording timer pill — only when capturing
            if isRecording {
                HStack(spacing: 6) {
                    Circle().fill(Self.paperAccent).frame(width: 6, height: 6)
                    Text(formatDuration(recordingElapsed))
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundColor(Self.paperAccent)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Capsule().fill(Self.paperAccentSoft.opacity(0.5)))
                .padding(.bottom, 10)
                .transition(.opacity.combined(with: .scale(scale: 0.9)))
            }

            HStack(spacing: 18) {
                editorialIconButton(icon: "folder", disabled: !isModelLoaded || isProcessing) {
                    showingFilePicker = true
                }

                PhotosPicker(selection: $selectedPhotoItem, matching: .videos) {
                    editorialIconButtonLabel(icon: "photo.on.rectangle.angled")
                }
                .disabled(!isModelLoaded || isProcessing)
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
                .disabled(!isModelLoaded || (isProcessing && !isRecording))
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
        .padding(.horizontal, 22)
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
            .preferredColorScheme(.light)
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
                Text(isModelLoaded ? "Ready" : "Choose a model to begin")
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
            Button(action: downloadSelectedModel) {
                HStack(spacing: 10) {
                    Image(systemName: downloadStatus[selectedModel] ?? false ? "arrow.clockwise" : "arrow.down")
                        .font(.system(size: 14, weight: .bold))
                    Text(downloadStatus[selectedModel] ?? false ? "Reload \(selectedModel.displayName)" : "Download \(selectedModel.displayName)")
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
            .disabled(isProcessing)
            .opacity(isProcessing ? 0.5 : 1)
            .padding(.top, 6)

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

                Text(model.description)
                    .font(.system(size: 12, design: .serif))
                    .foregroundColor(Self.paperInk.opacity(0.55))
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)

                HStack(spacing: 6) {
                    if isDownloaded {
                        paperBadge("Downloaded", tint: Color(red: 0.30, green: 0.65, blue: 0.40))
                    } else {
                        paperBadge("Not yet", tint: Self.paperMute)
                    }
                    if isActive {
                        paperBadge("Loaded", tint: Self.paperAccent)
                    }
                }
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(isSelected ? Self.paperAccentSoft.opacity(0.45) : Color.white.opacity(0.55))
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
            .background(Self.paperBG.ignoresSafeArea())
            .preferredColorScheme(.light)
            .navigationTitle("Language")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { showLanguagePicker = false }
                }
            }
        }
    }

    // MARK: - Library Button (top-right floating)
    var libraryButton: some View {
        Button {
            showLibrary = true
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "books.vertical.fill")
                    .font(.system(size: 13, weight: .semibold))
                Text("Library")
                    .font(.system(size: 13, weight: .semibold))
            }
            .foregroundStyle(
                LinearGradient(colors: [.blue, .purple], startPoint: .leading, endPoint: .trailing)
            )
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(
                Capsule()
                    .fill(.ultraThinMaterial)
                    .overlay(
                        Capsule().strokeBorder(
                            LinearGradient(colors: [Color.blue.opacity(0.4), Color.purple.opacity(0.4)],
                                           startPoint: .leading, endPoint: .trailing),
                            lineWidth: 1
                        )
                    )
                    .shadow(color: .black.opacity(0.06), radius: 8, y: 2)
            )
        }
        .buttonStyle(PressableButtonStyle())
    }

    // MARK: - Header View
    var headerView: some View {
        VStack(spacing: 14) {
            // App Icon — layered glow
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Color.blue.opacity(0.35), Color.purple.opacity(0.35)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 96, height: 96)
                    .blur(radius: 18)

                Circle()
                    .fill(.ultraThinMaterial)
                    .frame(width: 84, height: 84)
                    .overlay(
                        Circle()
                            .strokeBorder(
                                LinearGradient(
                                    colors: [Color.white.opacity(0.9), Color.white.opacity(0.2)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    )

                Image(systemName: "waveform")
                    .font(.system(size: 38, weight: .semibold))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .symbolEffect(.variableColor.iterative, options: .repeating, isActive: isProcessing || isRecording)
            }

            VStack(spacing: 6) {
                Text("Whisper")
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .tracking(-0.5)

                Text("ON-DEVICE  ·  PRIVATE  ·  NEURAL")
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .tracking(2.0)
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
    }
    
    // MARK: - Model Management Section
    var modelManagementSection: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                HStack(spacing: 8) {
                    Image(systemName: "cpu.fill")
                        .font(.title3)
                        .foregroundStyle(
                            LinearGradient(
                                colors: [.blue, .cyan],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )

                    VStack(alignment: .leading, spacing: 2) {
                        Text("AI Model")
                            .font(.headline)
                            .bold()
                        Text("Neural Engine Optimized")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }

                Spacer()

                if isModelLoaded {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 10, height: 10)
                            .shadow(color: .green.opacity(0.5), radius: 3)
                        Text("Ready")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.green)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.green.opacity(0.1))
                    .cornerRadius(20)
                }

                Button(action: checkModelStatus) {
                    Image(systemName: "arrow.clockwise")
                        .font(.caption)
                        .foregroundColor(.blue)
                }
                .buttonStyle(BorderlessButtonStyle())

                if isModelLoaded {
                    Button {
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.85)) {
                            showModelDetails = false
                        }
                    } label: {
                        Image(systemName: "chevron.up")
                            .font(.caption.bold())
                            .foregroundColor(.secondary)
                            .padding(6)
                            .background(Circle().fill(Color.gray.opacity(0.15)))
                    }
                    .buttonStyle(BorderlessButtonStyle())
                }
            }
            
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                ForEach(WhisperModel.allCases, id: \.self) { model in
                    CompactModelCard(
                        model: model,
                        isSelected: selectedModel == model,
                        isDownloaded: downloadStatus[model] ?? false,
                        isLoaded: isModelLoaded && selectedModel == model
                    ) {
                        selectModel(model)
                    }
                }
            }
            
            Button(action: downloadSelectedModel) {
                HStack(spacing: 10) {
                    Image(systemName: downloadStatus[selectedModel] ?? false ? "arrow.clockwise.circle.fill" : "arrow.down.circle.fill")
                        .font(.title3)
                    Text(downloadStatus[selectedModel] ?? false ? "Reload Model" : "Download Model")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(
                    LinearGradient(
                        colors: [Color(red: 0.20, green: 0.78, blue: 0.55), Color(red: 0.12, green: 0.62, blue: 0.45)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .foregroundColor(.white)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.3), lineWidth: 1)
                )
                .shadow(color: Color.green.opacity(0.35), radius: 14, x: 0, y: 6)
            }
            .buttonStyle(PressableButtonStyle())
            .disabled(isProcessing)
        }
        .padding(22)
        .background(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .fill(Color.white.opacity(0.45))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: [Color.white.opacity(0.8), Color.white.opacity(0.15)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                )
                .shadow(color: .black.opacity(0.06), radius: 18, x: 0, y: 6)
        )
    }

    // MARK: - Transcription Section
    var transcriptionSection: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                HStack(spacing: 8) {
                    Image(systemName: "text.bubble.fill")
                        .font(.title3)
                        .foregroundStyle(
                            LinearGradient(
                                colors: [.purple, .pink],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Transcription")
                            .font(.headline)
                            .bold()
                        Text("Upload & Process Audio")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                if isProcessing {
                    HStack(spacing: 4) {
                        ProgressView()
                            .scaleEffect(0.7)
                        if scenePhase == .background {
                            Image(systemName: "app.badge")
                                .font(.caption)
                                .foregroundColor(.blue)
                        }
                        Text(scenePhase == .background ? "Processing" : "Processing")
                            .font(.caption)
                            .foregroundColor(.primary)
                    }
                }
            }
            
            // Language Selection
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 6) {
                    Image(systemName: "globe")
                        .font(.subheadline)
                        .foregroundColor(.blue)
                    Text("Language")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundColor(.primary)
                }

                Picker("Language", selection: $selectedLanguage) {
                    ForEach(supportedLanguages, id: \.code) { language in
                        Text(language.name).tag(language.code)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                .padding(14)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(UIColor.systemBackground))
                        .shadow(color: .black.opacity(0.05), radius: 2, x: 0, y: 1)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                )
            }
            
            // Record Button
            VStack(spacing: 10) {
                Button(action: toggleRecording) {
                    HStack(spacing: 12) {
                        ZStack {
                            if isRecording {
                                Circle()
                                    .stroke(Color.white.opacity(0.5), lineWidth: 2)
                                    .frame(width: 28, height: 28)
                                    .scaleEffect(isRecording ? 1.3 : 1.0)
                                    .opacity(isRecording ? 0 : 1)
                                    .animation(.easeOut(duration: 1.2).repeatForever(autoreverses: false), value: isRecording)
                            }
                            Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                                .font(.system(size: 18, weight: .bold))
                        }
                        Text(isRecording
                             ? String(format: "Stop & Transcribe  ·  %@", formatDuration(recordingElapsed))
                             : "Record Audio")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 18)
                    .background(
                        LinearGradient(
                            colors: isRecording
                                ? [Color(red: 0.95, green: 0.25, blue: 0.35), Color(red: 0.78, green: 0.12, blue: 0.30)]
                                : [Color(red: 0.99, green: 0.40, blue: 0.55), Color(red: 0.82, green: 0.25, blue: 0.60)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .foregroundColor(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .strokeBorder(Color.white.opacity(0.3), lineWidth: 1)
                    )
                    .shadow(color: (isRecording ? Color.red : Color.pink).opacity(0.4), radius: 16, x: 0, y: 8)
                }
                .buttonStyle(PressableButtonStyle())
                .disabled(!isModelLoaded || (isProcessing && !isRecording))
                .opacity((isModelLoaded && (!isProcessing || isRecording)) ? 1.0 : 0.45)

                if isRecording {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 8, height: 8)
                            .shadow(color: .red, radius: 4)
                        Text("Listening…")
                            .font(.caption.weight(.semibold))
                            .foregroundColor(.red)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 5)
                    .background(Capsule().fill(Color.red.opacity(0.1)))
                }
            }

            // Upload Buttons
            VStack(spacing: 12) {
                Button(action: { showingFilePicker = true }) {
                    uploadButtonContent(icon: "folder.fill", title: "Files / iCloud", color: Color(red: 0.20, green: 0.55, blue: 0.98))
                }
                .buttonStyle(PressableButtonStyle())
                .disabled(!isModelLoaded || isProcessing)
                .opacity((isModelLoaded && !isProcessing) ? 1.0 : 0.45)

                PhotosPicker(selection: $selectedPhotoItem, matching: .videos) {
                    uploadButtonContent(icon: "photo.fill", title: "Photos", color: Color(red: 0.15, green: 0.72, blue: 0.85))
                }
                .buttonStyle(PressableButtonStyle())
                .disabled(!isModelLoaded || isProcessing)
                .opacity((isModelLoaded && !isProcessing) ? 1.0 : 0.45)
                .onChange(of: selectedPhotoItem) { _, newItem in
                    if let newItem = newItem {
                        handlePhotoSelection(newItem)
                    }
                }

            }
            
            // Background Warning Banner
            if isProcessing && scenePhase == .background {
                backgroundWarningBanner
            }

            // TQDM Progress
            if isProcessing {
                tqdmProgressSection
            }
            
            // Enhanced Performance Monitoring
            if isProcessing || !systemStatsHistory.isEmpty {
                enhancedPerformanceSection
            }
            
            // Transcript Display
            transcriptSection
        }
        .padding(22)
        .background(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .fill(Color.white.opacity(0.45))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: [Color.white.opacity(0.8), Color.white.opacity(0.15)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                )
                .shadow(color: .black.opacity(0.06), radius: 18, x: 0, y: 6)
        )
    }
    
    // MARK: - Background Warning Banner
    var backgroundWarningBanner: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title2)
                    .foregroundColor(.orange)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Background Processing Active")
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .foregroundColor(.primary)

                    Text("Tap to return to app for better performance")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()
            }
            .padding(16)
            .background(
                LinearGradient(
                    colors: [Color.orange.opacity(0.15), Color.yellow.opacity(0.1)],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.orange.opacity(0.3), lineWidth: 1.5)
            )
        }
    }

    // MARK: - Upload Button Content
    func uploadButtonContent(icon: String, title: String, color: Color) -> some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color.white.opacity(0.22))
                    .frame(width: 32, height: 32)
                Image(systemName: icon)
                    .font(.system(size: 14, weight: .bold))
            }
            Text(title)
                .fontWeight(.semibold)
            Spacer()
            Image(systemName: "chevron.right")
                .font(.system(size: 12, weight: .bold))
                .opacity(0.65)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 18)
        .padding(.vertical, 15)
        .background(
            LinearGradient(
                colors: [color.opacity(0.95), color.opacity(0.72)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .foregroundColor(.white)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.white.opacity(0.25), lineWidth: 1)
        )
        .shadow(color: color.opacity(0.35), radius: 12, x: 0, y: 6)
    }
    
    // MARK: - TQDM Progress Section
    var tqdmProgressSection: some View {
        VStack(spacing: 16) {
            HStack {
                Text(statusMessage)
                    .font(.system(.body, design: .monospaced))
                    .foregroundColor(.primary)
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 12) {
                    Text("\(Int(transcriptionProgress * 100))%")
                        .font(.system(.title3, design: .monospaced))
                        .bold()
                        .foregroundColor(.blue)
                        .frame(width: 60, alignment: .trailing)
                    
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            Rectangle()
                                .fill(Color.gray.opacity(0.2))
                                .frame(height: 24)
                                .cornerRadius(4)
                            
                            Rectangle()
                                .fill(
                                    LinearGradient(
                                        colors: [.blue, .cyan],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: geometry.size.width * transcriptionProgress, height: 24)
                                .cornerRadius(4)
                                .animation(.easeInOut(duration: 0.3), value: transcriptionProgress)
                            
                            if transcriptionProgress > 0.1 {
                                Text(tqdmProgressBar())
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.white)
                                    .padding(.leading, 8)
                            }
                        }
                    }
                    .frame(height: 24)
                }
                
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 16) {
                        StatLabel(label: "Segments", value: "\(currentSegment)/\(totalSegments)")
                        StatLabel(label: "Speed", value: String(format: "%.2f seg/s", segmentsPerSecond))
                        StatLabel(label: "Elapsed", value: formatDuration(getElapsedTime()))
                        StatLabel(label: "Remaining", value: formatDuration(getRemainingTime()))
                    }
                    .font(.system(.caption, design: .monospaced))
                    
                    if estimatedTotalTime > 0 {
                        Text("ETA: \(formatTime(estimatedTotalTime))")
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundColor(.gray)
                    }
                }
            }
            .padding()
            .background(Color.white)
            .cornerRadius(12)
            .shadow(color: .black.opacity(0.05), radius: 2, x: 0, y: 1)
        }
    }
    
    // MARK: - ENHANCED PERFORMANCE SECTION (FIXED)
    var enhancedPerformanceSection: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundColor(.blue)
                    .font(.title3)
                Text("System Performance Monitor")
                    .font(.headline)
                    .bold()
                
                Spacer()
                
                if let latest = systemStatsHistory.last {
                    HStack(spacing: 3) {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 6, height: 6)
                        Text("LIVE")
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundColor(.green)
                            .bold()
                    }
                }
            }
            
            if systemStatsHistory.count > 1, let latest = systemStatsHistory.last {
                VStack(spacing: 16) {
                    // CPU Graph
                    PerformanceGraphCard(
                        title: "CPU",
                        icon: "cpu.fill",
                        color: .blue,
                        currentValue: systemStatsHistory.last?.cpuUsage ?? 0,
                        unit: "%",
                        data: systemStatsHistory.map { ($0.timestamp, $0.cpuUsage) },
                        maxValue: 100
                    )
                    
                    // Memory Graph
                    PerformanceGraphCard(
                        title: "Memory",
                        icon: "memorychip.fill",
                        color: .green,
                        currentValue: latest.memoryPercent,
                        unit: "%",
                        data: systemStatsHistory.map { ($0.timestamp, $0.memoryPercent) },
                        maxValue: 100,
                        subtitle: String(format: "%.0f MB / %@ GB",
                                       latest.memoryUsage,
                                       getTotalMemoryGB())
                    )
                    
                    // Neural Engine Graph (Fixed Icon)
                    PerformanceGraphCard(
                        title: "Neural Engine",
                        icon: "brain.head.profile",
                        color: .purple,
                        currentValue: latest.gpuUsage,
                        unit: "%",
                        data: systemStatsHistory.map { ($0.timestamp, $0.gpuUsage) },
                        maxValue: 100,
                        subtitle: metalDevice?.name ?? "Apple Neural Engine"
                    )
                }
                
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                    SystemInfoCard(
                        title: "Processor",
                        value: getProcessorInfo(),
                        icon: "cpu",
                        color: .blue
                    )
                    
                    SystemInfoCard(
                        title: "Total Memory",
                        value: getTotalMemoryGB() + " GB",
                        icon: "memorychip",
                        color: .green
                    )
                    
                    SystemInfoCard(
                        title: "Processing Time",
                        value: formatDuration(getElapsedTime()),
                        icon: "clock",
                        color: .orange
                    )
                    
                    SystemInfoCard(
                        title: "Throughput",
                        value: String(format: "%.2f seg/s", segmentsPerSecond),
                        icon: "gauge.with.dots.needle.67percent",
                        color: .purple
                    )
                }
                
            } else {
                HStack {
                    Spacer()
                    VStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(1.2)
                        Text("Initializing performance monitoring...")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.gray)
                    }
                    Spacer()
                }
                .padding(.vertical, 40)
            }
        }
        .padding()
        .background(
            LinearGradient(
                colors: [Color.gray.opacity(0.03), Color.blue.opacity(0.02)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .cornerRadius(16)
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.gray.opacity(0.2), lineWidth: 1)
        )
    }
    
    // MARK: - Performance Graph Card
    struct PerformanceGraphCard: View {
        let title: String
        let icon: String
        let color: Color
        let currentValue: Double
        let unit: String
        let data: [(Date, Double)]
        let maxValue: Double
        var subtitle: String = ""
        
        var body: some View {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    HStack(spacing: 8) {
                        Image(systemName: icon)
                            .font(.title3)
                            .foregroundColor(color)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text(title)
                                .font(.system(.subheadline, design: .default))
                                .bold()
                            
                            if !subtitle.isEmpty {
                                Text(subtitle)
                                    .font(.system(.caption2, design: .monospaced))
                                    .foregroundColor(.gray)
                            }
                        }
                    }
                    
                    Spacer()
                    
                    HStack(alignment: .firstTextBaseline, spacing: 2) {
                        Text(String(format: "%.1f", currentValue))
                            .font(.system(.title2, design: .monospaced))
                            .bold()
                            .foregroundColor(getColorForValue(currentValue))
                        Text(unit)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.gray)
                    }
                }
                
                Chart {
                    ForEach(Array(data.enumerated()), id: \.offset) { _, point in
                        LineMark(
                            x: .value("Time", point.0),
                            y: .value("Value", point.1)
                        )
                        .foregroundStyle(color)
                        .lineStyle(StrokeStyle(lineWidth: 2.5))
                        .interpolationMethod(.catmullRom)
                        
                        AreaMark(
                            x: .value("Time", point.0),
                            y: .value("Value", point.1)
                        )
                        .foregroundStyle(
                            LinearGradient(
                                colors: [color.opacity(0.4), color.opacity(0.05)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        .interpolationMethod(.catmullRom)
                    }
                    
                    if !data.isEmpty {
                        let average = data.map { $0.1 }.reduce(0, +) / Double(data.count)
                        RuleMark(y: .value("Average", average))
                            .foregroundStyle(color.opacity(0.3))
                            .lineStyle(StrokeStyle(lineWidth: 1, dash: [5, 5]))
                    }
                }
                .chartYScale(domain: 0...maxValue)
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 5)) { value in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
                            .foregroundStyle(.gray.opacity(0.2))
                        AxisValueLabel {
                            if let intValue = value.as(Int.self) {
                                Text("\(intValue)")
                                    .font(.system(.caption2, design: .monospaced))
                                    .foregroundColor(.gray)
                            }
                        }
                    }
                }
                .chartXAxis {
                    AxisMarks(values: .automatic(desiredCount: 4)) { _ in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
                            .foregroundStyle(.gray.opacity(0.2))
                    }
                }
                .frame(height: 120)
                
                if !data.isEmpty {
                    HStack(spacing: 20) {
                        MiniStat(label: "Min", value: String(format: "%.1f", data.map { $0.1 }.min() ?? 0), color: .gray)
                        MiniStat(label: "Avg", value: String(format: "%.1f", data.map { $0.1 }.reduce(0, +) / Double(data.count)), color: .gray)
                        MiniStat(label: "Max", value: String(format: "%.1f", data.map { $0.1 }.max() ?? 0), color: .gray)
                        MiniStat(label: "Current", value: String(format: "%.1f", currentValue), color: getColorForValue(currentValue))
                    }
                    .font(.system(.caption2, design: .monospaced))
                }
            }
            .padding()
            .background(Color.white)
            .cornerRadius(14)
            .shadow(color: color.opacity(0.1), radius: 4, x: 0, y: 2)
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .stroke(
                        LinearGradient(
                            colors: [color.opacity(0.3), color.opacity(0.1)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 1
                    )
            )
        }
        
        func getColorForValue(_ value: Double) -> Color {
            if title == "CPU" || title == "Neural Engine" {
                if value > 80 { return .red }
                if value > 60 { return .orange }
                return color
            } else if title == "Memory" {
                if value > 85 { return .red }
                if value > 70 { return .orange }
                return color
            }
            return color
        }
    }
    
    struct MiniStat: View {
        let label: String
        let value: String
        let color: Color
        
        var body: some View {
            VStack(spacing: 2) {
                Text(label)
                    .foregroundColor(.gray)
                Text(value)
                    .foregroundColor(color)
                    .bold()
            }
        }
    }
    
    struct SystemInfoCard: View {
        let title: String
        let value: String
        let icon: String
        let color: Color
        
        var body: some View {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(color)
                    .frame(width: 32)
                
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.system(.caption, design: .default))
                        .foregroundColor(.gray)
                    Text(value)
                        .font(.system(.subheadline, design: .monospaced))
                        .bold()
                        .lineLimit(1)
                }
                
                Spacer()
            }
            .padding()
            .background(Color.white)
            .cornerRadius(10)
            .shadow(color: .black.opacity(0.03), radius: 2, x: 0, y: 1)
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

    var detectedLanguageBadge: some View {
        Group {
            if let code = detectedLanguageCode, !code.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 10, weight: .bold))
                    Text("Detected: \(displayName(forLanguageCode: code))")
                        .font(.system(size: 11, weight: .semibold))
                }
                .foregroundStyle(
                    LinearGradient(
                        colors: [.blue, .purple],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(
                    Capsule()
                        .fill(.ultraThinMaterial)
                        .overlay(
                            Capsule().strokeBorder(
                                LinearGradient(
                                    colors: [Color.blue.opacity(0.4), Color.purple.opacity(0.4)],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                ),
                                lineWidth: 1
                            )
                        )
                )
                .transition(.scale.combined(with: .opacity))
            }
        }
    }

    var transcriptSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Transcript")
                    .font(.subheadline)
                    .bold()

                detectedLanguageBadge

                Spacer()

                if !transcript.isEmpty && !isProcessing {
                    Button(action: copyTranscript) {
                        HStack(spacing: 4) {
                            Image(systemName: showCopySuccess ? "checkmark.circle.fill" : "doc.on.doc")
                            Text(showCopySuccess ? "Copied!" : "Copy")
                        }
                        .font(.caption)
                        .foregroundColor(showCopySuccess ? .green : .blue)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                    }
                }
            }
            
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if isProcessing && !streamingTranscript.isEmpty {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack(spacing: 6) {
                                Circle()
                                    .fill(Color.red)
                                    .frame(width: 8, height: 8)
                                    .shadow(color: .red.opacity(0.5), radius: 2)
                                Text("Live Transcription")
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.red)
                                    .fontWeight(.semibold)
                            }

                            Text(streamingTranscript)
                                .font(.system(.body, design: .default))
                                .foregroundColor(.primary)
                                .padding(14)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(
                                    LinearGradient(
                                        colors: [Color.yellow.opacity(0.15), Color.orange.opacity(0.08)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                                .cornerRadius(10)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(Color.orange.opacity(0.3), lineWidth: 1.5)
                                )
                        }
                    }

                    if !transcript.isEmpty {
                        Text(transcript)
                            .font(.system(.body, design: .default))
                            .textSelection(.enabled)
                            .padding(2)
                    } else if !isProcessing {
                        VStack(spacing: 8) {
                            Image(systemName: "doc.text")
                                .font(.system(size: 32))
                                .foregroundColor(.gray.opacity(0.5))
                            Text("Transcript will appear here")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.top, 60)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)
            }
            .frame(height: 280)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(UIColor.secondarySystemBackground))
            )
        }
    }
    
    // MARK: - Supporting Views
    
    struct StatLabel: View {
        let label: String
        let value: String
        
        var body: some View {
            HStack(spacing: 4) {
                Text(label + ":")
                    .foregroundColor(.gray)
                Text(value)
                    .foregroundColor(.primary)
                    .bold()
            }
        }
    }
    
    struct CompactModelCard: View {
        let model: WhisperModel
        let isSelected: Bool
        let isDownloaded: Bool
        let isLoaded: Bool
        let action: () -> Void

        var body: some View {
            Button(action: action) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(model.displayName)
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(isSelected ? .blue : .primary)

                        Spacer()

                        if isLoaded {
                            Circle()
                                .fill(Color.green)
                                .frame(width: 10, height: 10)
                                .shadow(color: .green.opacity(0.5), radius: 2)
                        } else if isDownloaded {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                                .font(.subheadline)
                        }
                    }

                    Text(model.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(isSelected ? Color.blue.opacity(0.08) : Color(UIColor.systemBackground))
                        .shadow(color: .black.opacity(isSelected ? 0.08 : 0.03), radius: isSelected ? 4 : 2, x: 0, y: 1)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(
                            isSelected ?
                                LinearGradient(colors: [.blue, .blue.opacity(0.6)], startPoint: .topLeading, endPoint: .bottomTrailing) :
                                LinearGradient(colors: [.gray.opacity(0.2), .gray.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing),
                            lineWidth: isSelected ? 2 : 1
                        )
                )
            }
            .buttonStyle(PlainButtonStyle())
        }
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
    
    func selectModel(_ model: WhisperModel) {
        selectedModel = model
        if downloadStatus[model] == true {
            Task {
                await loadExistingModel(model)
            }
        }
    }
    
    // MARK: - System Monitoring
    
    func initializeGPUMonitoring() {
        metalDevice = MTLCreateSystemDefaultDevice()
    }
    
    /// Instantaneous host CPU usage by computing the delta between two
    /// successive `PROCESSOR_CPU_LOAD_INFO` reads. The previous reading is
    /// cached in `lastHostCPUTicks`; the first call returns 0 and seeds the
    /// baseline so the next call is correct.
    func getActualCPUUsage() -> Double {
        var cpuInfo: processor_info_array_t!
        var numCpuInfo: mach_msg_type_number_t = 0
        var numCPUs: uint = 0

        let result = host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &numCPUs, &cpuInfo, &numCpuInfo)
        guard result == KERN_SUCCESS else { return 0 }

        // Aggregate ticks across all logical CPUs
        var user: UInt64 = 0, system: UInt64 = 0, nice: UInt64 = 0, idle: UInt64 = 0
        for i in 0..<Int(numCPUs) {
            let cpuLoad = cpuInfo.advanced(by: Int(CPU_STATE_MAX) * i)
            user   += UInt64(cpuLoad[Int(CPU_STATE_USER)])
            system += UInt64(cpuLoad[Int(CPU_STATE_SYSTEM)])
            nice   += UInt64(cpuLoad[Int(CPU_STATE_NICE)])
            idle   += UInt64(cpuLoad[Int(CPU_STATE_IDLE)])
        }

        // Free the kernel-allocated buffer (otherwise leaks each call)
        let vmAddr = vm_address_t(bitPattern: cpuInfo)
        let size   = vm_size_t(Int(numCpuInfo) * MemoryLayout<integer_t>.size)
        vm_deallocate(mach_task_self_, vmAddr, size)

        let prev = lastHostCPUTicks
        let dUser   = Int64(user)   - Int64(prev.user)
        let dSystem = Int64(system) - Int64(prev.system)
        let dNice   = Int64(nice)   - Int64(prev.nice)
        let dIdle   = Int64(idle)   - Int64(prev.idle)
        let dBusy   = dUser + dSystem + dNice
        let dTotal  = dBusy + dIdle

        // Persist for next call
        DispatchQueue.main.async {
            self.lastHostCPUTicks = (user, system, nice, idle)
        }

        // First call (no baseline yet) — return 0; the next call will report a real %.
        guard prev.user > 0 || prev.system > 0 else { return 0 }
        guard dTotal > 0 else { return 0 }

        return min(100.0, max(0.0, Double(dBusy) / Double(dTotal) * 100.0))
    }
    
    func getActualMemoryUsage() -> (usedMB: Double, totalMB: Double, percent: Double) {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            let usedMemoryMB = Double(info.resident_size) / 1_048_576
            let totalMemoryMB = Double(ProcessInfo.processInfo.physicalMemory) / 1_048_576
            let percent = (usedMemoryMB / totalMemoryMB) * 100.0
            
            return (usedMemoryMB, totalMemoryMB, percent)
        }
        
        return (0, 0, 0)
    }
    
    /// Real "app activity" % — total CPU+ANE time this process consumed since
    /// the previous sample, divided by wall-clock elapsed and normalized to the
    /// number of logical cores. Spikes hard when WhisperKit hammers the
    /// Neural Engine because ANE work shows up as task thread time too.
    func getGPUUsage() -> Double {
        var infoData = task_thread_times_info()
        var count = mach_msg_type_number_t(MemoryLayout<task_thread_times_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &infoData) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_THREAD_TIMES_INFO), $0, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return 0 }

        // total user + system time for live threads, in seconds
        let userSecs = Double(infoData.user_time.seconds) + Double(infoData.user_time.microseconds) / 1_000_000
        let sysSecs  = Double(infoData.system_time.seconds) + Double(infoData.system_time.microseconds) / 1_000_000
        let currentTaskSecs = userSecs + sysSecs

        let now = Date()
        let prevSecs = lastTaskTimeSeconds
        let prevAt   = lastSampleAt

        // Update baseline for next call
        DispatchQueue.main.async {
            self.lastTaskTimeSeconds = currentTaskSecs
            self.lastSampleAt = now
        }

        guard let prevAt, prevSecs > 0 else { return 0 }
        let wallSecs = now.timeIntervalSince(prevAt)
        guard wallSecs > 0.05 else { return 0 } // ignore samples too close together

        let deltaTask = max(0, currentTaskSecs - prevSecs)
        // Normalize per logical core so an app pinning all cores reads ~100%.
        let cores = max(1.0, Double(ProcessInfo.processInfo.activeProcessorCount))
        let activity = (deltaTask / (wallSecs * cores)) * 100.0
        return min(100.0, max(0.0, activity))
    }
    
    func updateSystemStats() {
        let cpuUsage = getActualCPUUsage()
        let (memoryMB, _, memoryPercent) = getActualMemoryUsage()
        let gpuUsage = getGPUUsage()

        let snapshot = SystemStatsSnapshot(
            timestamp: Date(),
            cpuUsage: cpuUsage,
            memoryUsage: memoryMB,
            memoryPercent: memoryPercent,
            gpuUsage: gpuUsage,
            networkIO: 0.0
        )

        // Use a Task on @MainActor so SwiftUI reliably picks up the @State
        // mutation. DispatchQueue.main.async sometimes coalesces away during
        // touch tracking on iPhone, leaving the chart frozen.
        Task { @MainActor in
            self.systemStatsHistory.append(snapshot)

            if self.systemStatsHistory.count > 60 {
                self.systemStatsHistory.removeFirst()
            }

            if self.isProcessing && self.scenePhase == .background {
                self.updateLiveActivity()
            }
        }
    }

    func startSystemMonitoring() {
        stopSystemMonitoring()
        systemStatsHistory.removeAll()

        // Push one snapshot immediately so the chart isn't blank for the
        // first second.
        updateSystemStats()

        // Attach to .common mode so the timer keeps firing while the user is
        // scrolling / pressing buttons (the default mode pauses during touch
        // tracking, which makes the chart look frozen on iPhone).
        let timer = Timer(timeInterval: 1.0, repeats: true) { _ in
            // Capture nothing problematic — read @State through the wrapper.
            self.updateSystemStats()
        }
        RunLoop.main.add(timer, forMode: .common)
        monitoringTimer = timer
    }
    
    func stopSystemMonitoring() {
        monitoringTimer?.invalidate()
        monitoringTimer = nil
    }
    
    // MARK: - TQDM Helpers
    
    func tqdmProgressBar() -> String {
        let filled = Int(transcriptionProgress * 20)
        let empty = 20 - filled
        return String(repeating: "█", count: filled) + String(repeating: "░", count: empty)
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
    
    func formatTime(_ duration: TimeInterval) -> String {
        let hours = Int(duration) / 3600
        let minutes = (Int(duration) % 3600) / 60
        let seconds = Int(duration) % 60
        
        if hours > 0 {
            return String(format: "%02d:%02d:%02d", hours, minutes, seconds)
        }
        return String(format: "%02d:%02d", minutes, seconds)
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
        GPU Device: \(metalDevice?.name ?? "Unknown")
        """
        
        try? debugInfo.write(to: debugFile, atomically: true, encoding: .utf8)
    }
    
    // MARK: - Model Management
    
    func findModelDirectory(for model: WhisperModel) -> URL? {
        let modelsDir = getModelsDirectory()
        guard let enumerator = FileManager.default.enumerator(
            at: modelsDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }

        for case let url as URL in enumerator {
            if url.lastPathComponent == model.rawValue {
                if let contents = try? FileManager.default.contentsOfDirectory(atPath: url.path),
                   !contents.isEmpty {
                    return url
                }
            }
        }
        return nil
    }

    func checkModelStatus() {
        for model in WhisperModel.allCases {
            downloadStatus[model] = findModelDirectory(for: model) != nil
        }

        if downloadStatus[selectedModel] == true && !isModelLoaded && !isProcessing {
            Task {
                await loadExistingModel(selectedModel)
            }
        }
    }
    
    func loadExistingModel(_ model: WhisperModel) async {
        await MainActor.run {
            statusMessage = "Loading model..."
            isProcessing = true
            startSystemMonitoring()
        }

        do {
            let modelsDir = getModelsDirectory()
            whisperKit = try await WhisperKit(
                model: model.rawValue,
                downloadBase: modelsDir,
                verbose: true,
                logLevel: .debug
            )

            await MainActor.run {
                isModelLoaded = true
                isProcessing = false
                statusMessage = "Model loaded successfully"
                stopSystemMonitoring()
                transcript = ""
            }
        } catch {
            await MainActor.run {
                isModelLoaded = false
                isProcessing = false
                statusMessage = "Failed to load model"
                stopSystemMonitoring()
                print("Failed to load model: \(error)")
            }
        }
    }

    func downloadSelectedModel() {
        isProcessing = true
        isModelLoaded = false
        statusMessage = "Downloading \(selectedModel.displayName)..."
        transcript = ""
        startSystemMonitoring()

        Task {
            do {
                let modelsDir = getModelsDirectory()
                try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)

                whisperKit = try await WhisperKit(
                    model: selectedModel.rawValue,
                    downloadBase: modelsDir,
                    verbose: true,
                    logLevel: .debug
                )

                await MainActor.run {
                    downloadStatus[selectedModel] = true
                    isModelLoaded = true
                    isProcessing = false
                    statusMessage = "Model ready"
                    stopSystemMonitoring()
                    transcript = ""
                }
            } catch {
                await MainActor.run {
                    isModelLoaded = false
                    transcript = "❌ Download failed: \(error.localizedDescription)\n\nPlease check your internet connection and try again."
                    isProcessing = false
                    statusMessage = ""
                    stopSystemMonitoring()
                }
            }
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
            transcript = "❌ File selection failed: \(error.localizedDescription)"
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
                transcript = "❌ Failed to access file: \(error.localizedDescription)\n\n💡 Try:\n• Downloading file from iCloud first\n• Using Photos upload (cyan button)\n• Testing on a real device"
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
                        transcript = "❌ Failed to load video from Photos"
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
                    transcript = "❌ Photo import failed: \(error.localizedDescription)"
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
            transcript = ""
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
                let errorMessage = error.localizedDescription

                // Check if it's the background ML error
                if errorMessage.contains("ML Programs") || errorMessage.contains("asynchronous prediction") {
                    transcript = """
                    ⚠️ Background Processing Limited

                    iOS restricts Neural Engine/GPU access when app is in background.

                    ✅ Solution: Keep the app in foreground during transcription.

                    Technical error: \(errorMessage)
                    """
                } else {
                    transcript = "❌ Import failed: \(errorMessage)"
                }

                isProcessing = false
                statusMessage = ""
                stopSystemMonitoring()
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
        session.requestRecordPermission { granted in
            DispatchQueue.main.async {
                guard granted else {
                    transcript = "❌ Microphone permission denied. Enable it in Settings."
                    return
                }
                do {
                    // Mac Catalyst / Designed-for-iPad-on-Mac doesn't support
                    // .defaultToSpeaker; configure category defensively.
                    #if targetEnvironment(macCatalyst)
                    try? session.setCategory(.playAndRecord, mode: .default, options: [])
                    #else
                    try? session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
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
                    recordingElapsed = 0
                    transcript = ""
                    streamingTranscript = ""

                    recordingTimer?.invalidate()
                    recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
                        if let start = recordingStartTime {
                            recordingElapsed = Date().timeIntervalSince(start)
                        }
                    }
                } catch {
                    transcript = "❌ Failed to start recording: \(error.localizedDescription)"
                }
            }
        }
    }

    func stopRecordingAndTranscribe() {
        // Finalize recorder
        audioRecorder?.stop()
        let recorder = audioRecorder
        audioRecorder = nil
        recordingTimer?.invalidate()
        recordingTimer = nil
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
                    transcript = "❌ Recording was too short or failed to save. Please record at least 1 second of audio."
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

    func saveTranscriptToLibrary(audioURL: URL, results: [TranscriptionResult], fullText: String) {
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
            audioFilePath: savedRelativePath
        )

        // Flatten WhisperKit segments across all results into our SavedSegment rows.
        var segs: [SavedSegment] = []
        for r in results {
            for s in r.segments {
                segs.append(SavedSegment(
                    startTime: Double(s.start),
                    endTime: Double(s.end),
                    text: s.text.trimmingCharacters(in: .whitespacesAndNewlines)
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

    func testWithBundledAudio() {
        let audioExtensions = ["m4a", "mp3", "wav", "aac", "mp4"]
        var audioURL: URL?
        
        for ext in audioExtensions {
            if let url = Bundle.main.url(forResource: "test", withExtension: ext) {
                audioURL = url
                break
            }
        }
        
        guard let url = audioURL else {
            transcript = "❌ No bundled audio file found!\n\nAdd a test audio file to the project:\n1. Drag audio file into Xcode\n2. Name it 'test.mp3' or 'test.m4a'\n3. Check 'Add to targets: whisper'\n4. Rebuild and try again"
            return
        }
        
        isProcessing = true
        statusMessage = "Loading bundled test audio..."
        
        Task {
            do {
                let importsDir = getImportsDirectory()
                try FileManager.default.createDirectory(at: importsDir, withIntermediateDirectories: true)
                
                let fileName = url.lastPathComponent
                let destinationURL = importsDir.appendingPathComponent(fileName)
                
                try? FileManager.default.removeItem(at: destinationURL)
                try FileManager.default.copyItem(at: url, to: destinationURL)
                
                await MainActor.run {
                    statusMessage = "Transcribing bundled audio..."
                }
                
                try await transcribeAudio(at: destinationURL)
            } catch {
                await MainActor.run {
                    transcript = "❌ Test failed: \(error.localizedDescription)"
                    isProcessing = false
                    statusMessage = ""
                }
            }
        }
    }
    
    // MARK: - Transcription
    
    func transcribeAudio(at url: URL) async throws {
        await MainActor.run {
            transcriptionProgress = 0.0
            streamingTranscript = ""
            processStartTime = Date()
            lastProgressUpdate = Date()
            startSystemMonitoring()

            // Only start Live Activity if in background
            if scenePhase == .background {
                startLiveActivity()
            }
        }

        guard let whisper = whisperKit else {
            throw NSError(domain: "WhisperKit not initialized", code: -1)
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

        let languageCode = selectedLanguage == "auto" ? nil : selectedLanguage
        let shouldDetectLanguage = (languageCode == nil)

        let options = DecodingOptions(
            verbose: true,
            task: .transcribe,
            language: languageCode,
            temperature: 0.0,
            detectLanguage: shouldDetectLanguage,
            compressionRatioThreshold: 2.4
        )

        let audioFile = try? AVAudioFile(forReading: workURL)
        let duration = Double(audioFile?.length ?? 0) / (audioFile?.fileFormat.sampleRate ?? 1.0)
        
        await MainActor.run {
            totalSegments = max(1, Int(ceil(duration / 30.0)))
        }
        
        var lastSegmentTime = Date()
        var segmentCount = 0
        var lastReportedProgress = 0.0
        
        let results = try await whisper.transcribe(
            audioPath: workURL.path,
            decodeOptions: options,
            callback: { progress in
                Task { @MainActor in
                    let currentTime = Date()
                    
                    // Update segment counter
                    self.currentSegment = progress.windowId + 1  // +1 for 1-based counting
                    
                    // Calculate precise progress
                    if self.totalSegments > 0 {
                        let rawProgress = Double(progress.windowId + 1) / Double(self.totalSegments)
                        // Clamp between 0 and 0.99 (never show 100% until complete)
                        self.transcriptionProgress = min(0.99, max(0.0, rawProgress))

                        // Update background task progress
                        if let bgProgress = self.backgroundTaskProgress {
                            bgProgress.completedUnitCount = Int64(self.transcriptionProgress * 100)
                        }
                    }
                    
                    // Update streaming text
                    if !progress.text.isEmpty {
                        self.streamingTranscript = progress.text
                    }
                    
                    // Calculate segments per second (smooth calculation)
                    segmentCount += 1
                    if segmentCount > 1 {
                        let totalElapsed = currentTime.timeIntervalSince(self.processStartTime ?? currentTime)
                        if totalElapsed > 0 {
                            self.segmentsPerSecond = Double(segmentCount) / totalElapsed
                        }
                    }
                    
                    lastSegmentTime = currentTime
                    
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
                let fullText = results.map { $0.text }.joined(separator: "\n")
                
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
                    self.saveTranscriptToLibrary(audioURL: url, results: results, fullText: fullText)
                    
                    var totalDuration: Double = 0.0
                    for result in results {
                        if !result.segments.isEmpty, let lastSegment = result.segments.last {
                            totalDuration = max(totalDuration, Double(lastSegment.end))
                        }
                    }
                    
                    let durationStr = totalDuration > 0 ? String(format: "%.1f", totalDuration) : "Unknown"
                    let selectedLangName = supportedLanguages.first(where: { $0.code == selectedLanguage })?.name ?? "Auto Detect"
                    let processingTime = getElapsedTime()
                    
                    transcript = """
                    ✅ Transcription Complete
                    
                    📊 Metadata:
                    • Detected Language: \(detectedLanguage.uppercased())
                    • Selected Language: \(selectedLangName)
                    • Audio Duration: \(durationStr)s
                    • Processing Time: \(formatDuration(processingTime))
                    • Speed: \(String(format: "%.2fx real-time", totalDuration / processingTime))
                    • Model: \(selectedModel.displayName)
                    • Segments: \(results.count)
                    • Avg Speed: \(String(format: "%.2f seg/s", segmentsPerSecond))
                    
                    📝 Transcript:
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    
                    \(fullText)
                    """
                } else {
                    transcript = "⚠️ No speech detected in audio file"
                }
            } else {
                transcript = "⚠️ No speech detected in audio file"
            }
            
            streamingTranscript = ""
            isProcessing = false
            statusMessage = ""
            transcriptionProgress = 0.0
            stopSystemMonitoring()
            endBackgroundTask()
            endLiveActivity()
        }
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
