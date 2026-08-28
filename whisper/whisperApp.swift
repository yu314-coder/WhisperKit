import SwiftUI
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
}

@main
struct whisperApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var sharedModelContainer: ModelContainer = {
        let schema = Schema([SavedTranscript.self, SavedSegment.self])
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
            ContentView()
        }
        .modelContainer(sharedModelContainer)
    }
}
