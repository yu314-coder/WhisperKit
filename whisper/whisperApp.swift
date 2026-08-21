import SwiftUI
import SwiftData

@main
struct whisperApp: App {
    var sharedModelContainer: ModelContainer = {
        let schema = Schema([SavedTranscript.self, SavedSegment.self])
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
        do {
            return try ModelContainer(for: schema, configurations: [config])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(sharedModelContainer)
    }
}
