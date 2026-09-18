import Foundation

/// Downloads model weights, continuing while the app is in the background.
///
/// A background `URLSession` keeps multi-gigabyte transfers alive when the app
/// is suspended, and iOS relaunches the app to hand over the finished files if
/// it was terminated meanwhile. That imposes the shape of this class:
///
/// - There is exactly one session, created once with a fixed identifier.
///   Creating a second one with the same identifier is an error, so this is a
///   singleton rather than an object per download.
/// - The delegate must move each finished file into place itself, because the
///   app that started the download may be gone by the time it lands. The
///   destination travels with the task in `taskDescription`, which the system
///   preserves across relaunches.
/// - Awaiting a download is therefore best-effort: if the app was terminated
///   the continuation is gone, but the file still arrives, and the next launch
///   simply finds it already present.
final class WeightDownloadService: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    static let shared = WeightDownloadService()

    struct Progress: Sendable {
        let destinationName: String
        let fraction: Double
        let received: Int64
        let expected: Int64
    }

    /// Set by the app delegate when iOS wakes the app for this session, and
    /// called once its events have all been delivered.
    static var backgroundCompletionHandler: (@Sendable () -> Void)?

    /// Reports progress for whichever download is running.
    var onProgress: (@Sendable (Progress) -> Void)?

    private var session: URLSession!
    private let lock = NSLock()
    private var continuations: [Int: CheckedContinuation<Void, Error>] = [:]
    private var lastReported: [Int: Double] = [:]

    private override init() {
        super.init()
        let configuration = URLSessionConfiguration.background(
            withIdentifier: "euleryu.whisper.music-weights")
        // Not discretionary: the user is watching a progress bar, so the
        // transfer should start now rather than when iOS finds it convenient.
        configuration.isDiscretionary = false
        configuration.sessionSendsLaunchEvents = true
        configuration.allowsCellularAccess = true
        configuration.timeoutIntervalForResource = 7 * 24 * 3600
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    /// Starts (or rejoins) the download of `remote` into `destination`.
    ///
    /// If a task for this destination is already running — typically because
    /// it was started before the app was last suspended — this waits on that
    /// one instead of starting a second copy.
    func download(from remote: URL, to destination: URL) async throws {
        if let existing = await existingTask(for: destination) {
            try await wait(for: existing)
            return
        }
        let task = session.downloadTask(with: remote)
        task.taskDescription = destination.path
        try await withTaskCancellationHandler {
            try await wait(for: task, start: true)
        } onCancel: {
            task.cancel()
        }
    }

    /// Any unfinished task already targeting this destination.
    private func existingTask(for destination: URL) async -> URLSessionDownloadTask? {
        let tasks = await session.allTasks
        return tasks.compactMap { $0 as? URLSessionDownloadTask }
            .first { $0.taskDescription == destination.path
                && ($0.state == .running || $0.state == .suspended) }
    }

    private func wait(for task: URLSessionDownloadTask, start: Bool = false) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            lock.lock()
            continuations[task.taskIdentifier] = continuation
            lock.unlock()
            if start { task.resume() } else if task.state == .suspended { task.resume() }
        }
    }

    private func finish(_ identifier: Int, with result: Result<Void, Error>) {
        lock.lock()
        let continuation = continuations.removeValue(forKey: identifier)
        lastReported.removeValue(forKey: identifier)
        lock.unlock()
        switch result {
        case .success:          continuation?.resume()
        case .failure(let e):   continuation?.resume(throwing: e)
        }
    }

    // MARK: - URLSessionDownloadDelegate

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let fraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        lock.lock()
        let previous = lastReported[downloadTask.taskIdentifier] ?? -1
        let shouldReport = fraction - previous >= 0.005
        if shouldReport { lastReported[downloadTask.taskIdentifier] = fraction }
        lock.unlock()
        guard shouldReport else { return }
        let name = (downloadTask.taskDescription as NSString?)?.lastPathComponent ?? ""
        onProgress?(Progress(destinationName: name, fraction: fraction,
                             received: totalBytesWritten, expected: totalBytesExpectedToWrite))
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // The file at `location` is deleted the moment this returns, so it has
        // to be moved here and now — not from the awaiting side, which may no
        // longer exist.
        guard let path = downloadTask.taskDescription else { return }
        let destination = URL(fileURLWithPath: path)
        do {
            if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
                throw MusicDownloaderError.http(http.statusCode)
            }
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
            finish(downloadTask.taskIdentifier, with: .success(()))
        } catch {
            finish(downloadTask.taskIdentifier, with: .failure(error))
        }
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        guard let error else { return }   // success is resumed above
        finish(task.taskIdentifier,
               with: .failure(MusicDownloaderError.interrupted(underlying: error.localizedDescription)))
    }

    /// All background events for this session have been delivered; iOS wants
    /// its completion handler called so it can snapshot and suspend the app.
    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        let handler = Self.backgroundCompletionHandler
        Self.backgroundCompletionHandler = nil
        DispatchQueue.main.async { handler?() }
    }
}

enum MusicDownloaderError: LocalizedError {
    case http(Int)
    case interrupted(underlying: String)

    var errorDescription: String? {
        switch self {
        case .http(let code):            return "The server returned HTTP \(code)."
        case .interrupted(let underlying): return underlying
        }
    }
}
