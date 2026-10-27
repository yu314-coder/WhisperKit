import SwiftUI
import UIKit
import SwiftData
import BackgroundTasks

/// Launch setup, and the wake-ups iOS sends for background weight downloads.
///
/// Background work itself is `BackgroundWork`: its continued-processing
/// tasks register when first used, which iOS allows for that kind of task
/// only. Any other `BGTaskScheduler.register` must run before
/// `application(_:didFinishLaunchingWithOptions:)` returns — from a view's
/// `.onAppear` it crashed on device, and again on a second registration.
final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        _ = MusicRunMarker.wasInterrupted   // what the last run left, before anything changes it
        BackgroundWork.shared.start()
        DispatchQueue.global(qos: .utility).async { MusicModel.removeRetiredWeights() }
        // The download session, at once: files that finished while the app
        // was closed are moved into place as soon as it exists.
        _ = WeightDownloadService.shared
        Task { await MediaLink.discardOrphanedDownloads() }
        return true
    }

    /// iOS relaunched (or resumed) the app because a background download
    /// finished. The session's delegate does the work; this hands
    /// back the completion handler it must call once every event has been
    /// delivered, or the system counts the wake-up as a hang.
    func application(_ application: UIApplication,
                     handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        WeightDownloadService.backgroundCompletionHandler = completionHandler
        _ = WeightDownloadService.shared   // recreates the session to receive events
    }
}

@main
struct whisperApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var sharedModelContainer: ModelContainer = {
        let schema = Schema([SavedTranscript.self, SavedSegment.self, SavedMusic.self, SavedImage.self])
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
        do {
            return try ModelContainer(for: schema, configurations: [config])
        } catch {
            // A failed migration used to be a fatalError, which turned any
            // store problem into a crash on every launch with no way out.
            // Fall back to an in-memory store so the app still runs: the user
            // can record and transcribe, and only loses the saved library.
            print("⚠️ Persistent store unavailable, falling back to memory: \(error)")
            let memory = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
            do {
                return try ModelContainer(for: schema, configurations: [memory])
            } catch {
                fatalError("Could not create even an in-memory ModelContainer: \(error)")
            }
        }
    }()

    var body: some Scene {
        WindowGroup {
            RootView()
        }
        .modelContainer(sharedModelContainer)
    }
}

/// Transcription, music and pictures are separate enough to be separate
/// tabs: they share storage and the model downloader, but nothing on screen.
struct RootView: View {
    private enum Tab: String { case transcribe, live, music, image }
    @State private var tab = Self.firstTab

    /// The Transcribe tab, or in Debug builds `-AppStartTab live|music|image`.
    private static var firstTab: Tab {
        #if DEBUG
        if let raw = UserDefaults.standard.string(forKey: "AppStartTab"), let tab = Tab(rawValue: raw) { return tab }
        #endif
        return .transcribe
    }
    @Environment(\.horizontalSizeClass) private var sizeClass
    /// A phone-shaped window. iPhone Duo's wide screen puts the tabs in a
    /// column down the side, where badges stick out past it.
    @State private var tallWindow = true
    private var showsBadges: Bool { sizeClass == .compact && tallWindow }

    init() {
        // The "Beta" badge in the app's colour: in the default red it read
        // as something wrong.
        let appearance = UITabBarAppearance()
        appearance.configureWithDefaultBackground()
        for layout in [appearance.stackedLayoutAppearance, appearance.inlineLayoutAppearance,
                       appearance.compactInlineLayoutAppearance] {
            layout.normal.badgeBackgroundColor = UIColor(Studio.accent)
            layout.selected.badgeBackgroundColor = UIColor(Studio.accent)
        }
        UITabBar.appearance().standardAppearance = appearance
        UITabBar.appearance().scrollEdgeAppearance = appearance
    }

    var body: some View {
        TabView(selection: $tab) {
            ContentView()
                .tabItem { Label("Transcribe", systemImage: "waveform") }
                .tag(Tab.transcribe)
            LiveView()
                .tabItem { Label("Live", systemImage: "captions.bubble") }
                .tag(Tab.live)
            MusicView()
                .tabItem { Label("Music", systemImage: "music.note") }
                // iPad's floating tab bar draws badges only in red, over the
                // label, and iPhone Duo's side column lets them stick out;
                // there the Music page's own BETA tag says it.
                .badge(showsBadges ? Text("Beta") : nil)
                .tag(Tab.music)
            ImageView()
                .tabItem { Label("Image", systemImage: "photo.artframe") }
                .badge(showsBadges ? Text("Beta") : nil)
                .tag(Tab.image)
        }
        // The whole window, insets included: on iPhone Duo the side column
        // is inset, and the rest alone is phone-shaped.
        .onGeometryChange(for: Bool.self) { proxy in
            let insets = proxy.safeAreaInsets
            return proxy.size.width + insets.leading + insets.trailing
                < (proxy.size.height + insets.top + insets.bottom) * 0.62
        } action: { tallWindow = $0 }
        .tint(Studio.accent)
        .modifier(TranslationHost())
        // An audio or video file shared to the app, or opened with it from
        // Files: transcribe it (see Info.plist's document types).
        .onOpenURL { url in
            guard url.isFileURL else { return }
            tab = .transcribe
            IncomingMedia.shared.receive(url)
        }
        .task { IncomingMedia.clearStaleInbox() }
    }
}

/// A file handed to the app from outside, waiting for the Transcribe tab
/// to take it — which it does once a model is loaded.
@Observable
final class IncomingMedia {
    static let shared = IncomingMedia()
    var file: URL?

    /// Takes a newly shared file; one still waiting is replaced, and the
    /// copy iOS made of it is removed.
    func receive(_ url: URL) {
        if let waiting = file, waiting != url { Self.removeIfCopied(waiting) }
        file = url
    }

    /// iOS copies a shared file into Documents/Inbox; the copy is the app's
    /// to remove. A file opened in place from Files is not.
    static func removeIfCopied(_ url: URL) {
        if url.pathComponents.contains("Inbox") { try? FileManager.default.removeItem(at: url) }
    }

    /// Copies left in Documents/Inbox by files shared and never transcribed
    /// (the app closed first) — a day old or more.
    static func clearStaleInbox() {
        guard let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let inbox = documents.appendingPathComponent("Inbox", isDirectory: true)
        let cutoff = Date().addingTimeInterval(-86_400)
        let items = (try? FileManager.default.contentsOfDirectory(at: inbox, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        for item in items {
            let modified = (try? item.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
            if modified < cutoff { try? FileManager.default.removeItem(at: item) }
        }
    }
}
