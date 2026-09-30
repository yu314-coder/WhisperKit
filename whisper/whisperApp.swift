import SwiftUI
import UIKit
import SwiftData
import BackgroundTasks

/// Registers background-task handlers at the only moment iOS allows.
///
/// `BGTaskScheduler.register` must be called before
/// `application(_:didFinishLaunchingWithOptions:)` returns. This used to run
/// from ContentView's `.onAppear`, which is after launch completes — iOS
/// raises NSInternalInconsistencyException for that on device, and again if the
/// same identifier is registered twice. `.onAppear` fires every time a sheet is
/// dismissed, so opening export or the model picker and coming back was enough
/// to hit the second case. The Simulator enforces neither, which is why it only
/// ever crashed on real hardware.
final class AppDelegate: NSObject, UIApplicationDelegate {
    static let transcriptionTaskID = "com.whisper.transcription"

    /// Set by ContentView so the handler can reach the running view's logic.
    static var transcriptionHandler: ((BGTask) -> Void)?

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        _ = MusicRunMarker.wasInterrupted   // what the last run left, before anything changes it
        DispatchQueue.global(qos: .utility).async { MusicModel.removeRetiredWeights() }
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.transcriptionTaskID,
            using: nil
        ) { task in
            guard let handler = Self.transcriptionHandler else {
                // Nothing is listening yet — end cleanly rather than hang.
                task.setTaskCompleted(success: false)
                return
            }
            handler(task)
        }
        return true
    }

    /// iOS relaunched (or resumed) the app because a background weight
    /// download finished. The session's delegate does the work; this hands
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
        let schema = Schema([SavedTranscript.self, SavedSegment.self, SavedMusic.self])
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

/// Transcription and music generation are separate enough to be separate
/// tabs: they share the library and the audio stack, but nothing on screen.
struct RootView: View {
    private enum Tab { case transcribe, music }
    @State private var tab = Tab.transcribe

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
            MusicView()
                .tabItem { Label("Music", systemImage: "music.note") }
                .badge("Beta")
                .tag(Tab.music)
        }
        .tint(Studio.accent)
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
