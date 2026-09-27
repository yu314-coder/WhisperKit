import AppleArchive
import Foundation
import System
import UIKit

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
///
/// Transfers resume rather than restart. GitHub's CDN occasionally resets a
/// connection mid-file; the background daemon then retries on its own
/// schedule, which can mean minutes of a frozen progress bar — one tester's
/// sat at 1,764 of 1,773 MB. So a transfer that makes no progress for
/// `stallTimeout` while the app is on screen is cancelled with resume data
/// and continued at once, and a failed one continues from its resume data
/// when there is any. Only when there is none does it start over.
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
    /// A line for the user when a transfer is being recovered, nil otherwise.
    var onStatus: (@Sendable (String?) -> Void)? {
        get { lock.withLock { statusHandler } }
        set { lock.withLock { statusHandler = newValue } }
    }
    private var statusHandler: (@Sendable (String?) -> Void)?

    private func setStatus(_ note: String?) {
        let handler = lock.withLock { () -> (@Sendable (String?) -> Void)? in
            noteShowing = note != nil
            return statusHandler
        }
        handler?(note)
    }

    /// No bytes for this long, on screen, counts as stalled.
    private let stallTimeout: TimeInterval = 30
    private let maximumAttempts = 8

    private var session: URLSession!
    private let lock = NSLock()
    private var continuations: [Int: CheckedContinuation<Void, Error>] = [:]
    private var lastReported: [Int: Date] = [:]
    private var lastActivity: [Int: Date] = [:]
    /// Tasks this class cancelled for stalling. Their completion is delivered
    /// with the resume data from `cancel(byProducingResumeData:)`, not by the
    /// delegate's generic cancellation error.
    private var stalled: Set<Int> = []
    /// Whether a recovery note is on screen, so the first bytes after it
    /// can clear it.
    private var noteShowing = false

    /// A transfer that ended early, with what is needed to continue it.
    private struct Interrupted: Error {
        let resumeData: Data?
        let reason: String
    }

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

    /// Downloads `remote` into `destination`, resuming across interruptions.
    ///
    /// If a task for this destination is already running — typically because
    /// it was started before the app was last suspended — this continues that
    /// one instead of starting a second copy.
    func download(from remote: URL, to destination: URL) async throws {
        var task = await existingTask(for: destination) ?? newTask(from: remote, to: destination)
        var attempt = 1
        while true {
            do {
                try await run(task)
                setStatus(nil)
                return
            } catch let interruption as Interrupted {
                try Task.checkCancellation()
                guard attempt < maximumAttempts else {
                    throw MusicDownloaderError.interrupted(underlying: interruption.reason)
                }
                attempt += 1
                if let data = interruption.resumeData {
                    setStatus("Connection dropped — resuming (try \(attempt) of \(maximumAttempts))")
                    task = session.downloadTask(withResumeData: data)
                    task.taskDescription = destination.path
                } else {
                    // Nothing to resume from; a brief pause so a flapping
                    // connection is not hammered.
                    setStatus("Connection dropped — restarting this file (try \(attempt) of \(maximumAttempts))")
                    try await Task.sleep(for: .seconds(2))
                    task = newTask(from: remote, to: destination)
                }
            } catch MusicDownloaderError.http(let code) where (400 ..< 500).contains(code) && attempt < maximumAttempts {
                // A resumed request goes to the address GitHub redirected to,
                // which expires after an hour. Starting from the release URL
                // again gets a fresh one.
                try Task.checkCancellation()
                attempt += 1
                setStatus("Download link expired — restarting this file")
                task = newTask(from: remote, to: destination)
            }
        }
    }

    private func newTask(from remote: URL, to destination: URL) -> URLSessionDownloadTask {
        let task = session.downloadTask(with: remote)
        task.taskDescription = destination.path
        return task
    }

    /// Any unfinished task already targeting this destination.
    private func existingTask(for destination: URL) async -> URLSessionDownloadTask? {
        let tasks = await session.allTasks
        return tasks.compactMap { $0 as? URLSessionDownloadTask }
            .first { $0.taskDescription == destination.path
                && ($0.state == .running || $0.state == .suspended) }
    }

    /// Runs one task to completion, watching it for stalls.
    private func run(_ task: URLSessionDownloadTask) async throws {
        let identifier = task.taskIdentifier
        lock.withLock { lastActivity[identifier] = Date() }
        let watchdog = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                guard let self, !Task.isCancelled else { return }
                let idle = Date().timeIntervalSince(self.lock.withLock { self.lastActivity[identifier] } ?? Date())
                // Only while on screen: in the background iOS may be
                // deliberately holding the transfer, and this process may be
                // suspended anyway.
                let onScreen = await MainActor.run { UIApplication.shared.applicationState == .active }
                if idle > self.stallTimeout, onScreen {
                    self.recoverStalled(task)
                    return
                }
            }
        }
        defer { watchdog.cancel() }

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                lock.withLock { continuations[identifier] = continuation }
                if task.state == .suspended { task.resume() }
            }
        } onCancel: {
            task.cancel()
        }
    }

    private func recoverStalled(_ task: URLSessionDownloadTask) {
        let identifier = task.taskIdentifier
        _ = lock.withLock { stalled.insert(identifier) }
        setStatus("No data for \(Int(stallTimeout)) s — reconnecting")
        task.cancel { [weak self] data in
            self?.finish(identifier, with: .failure(Interrupted(resumeData: data, reason: "The download stalled.")))
        }
    }

    private func finish(_ identifier: Int, with result: Result<Void, Error>) {
        let continuation = lock.withLock {
            lastReported.removeValue(forKey: identifier)
            lastActivity.removeValue(forKey: identifier)
            stalled.remove(identifier)
            return continuations.removeValue(forKey: identifier)
        }
        switch result {
        case .success:          continuation?.resume()
        case .failure(let e):   continuation?.resume(throwing: e)
        }
    }

    // MARK: - URLSessionDownloadDelegate

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didResumeAtOffset fileOffset: Int64,
                    expectedTotalBytes: Int64) {
        lock.withLock { lastActivity[downloadTask.taskIdentifier] = Date() }
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let now = Date()
        // Reported by time, not by fraction: at 0.5% steps a 1.8 GB file
        // updated only every 9 MB, which on a slow link is indistinguishable
        // from being stuck.
        let clearNote = lock.withLock { noteShowing }
        if clearNote { setStatus(nil) }
        let shouldReport = lock.withLock {
            lastActivity[downloadTask.taskIdentifier] = now
            let previous = lastReported[downloadTask.taskIdentifier] ?? .distantPast
            let due = now.timeIntervalSince(previous) >= 0.25 || totalBytesWritten == totalBytesExpectedToWrite
            if due { lastReported[downloadTask.taskIdentifier] = now }
            return due
        }
        guard shouldReport else { return }
        let name = (downloadTask.taskDescription as NSString?)?.lastPathComponent ?? ""
        onProgress?(Progress(destinationName: name,
                             fraction: Double(totalBytesWritten) / Double(totalBytesExpectedToWrite),
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
            // 206 is success too: a resumed transfer ends with the partial
            // response that completed it. Accepting only 200, as this once
            // did, rejected every download that had been resumed.
            if let http = downloadTask.response as? HTTPURLResponse, !(200 ..< 300).contains(http.statusCode) {
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
        // A stall cancellation is finished by its own handler, with resume data.
        if lock.withLock({ stalled.contains(task.taskIdentifier) }) { return }
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorCancelled,
           nsError.userInfo[NSURLSessionDownloadTaskResumeData] == nil {
            finish(task.taskIdentifier, with: .failure(CancellationError()))
            return
        }
        finish(task.taskIdentifier, with: .failure(Interrupted(
            resumeData: nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data,
            reason: error.localizedDescription)))
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
    case insufficientSpace(needed: Int64, available: Int64)

    var errorDescription: String? {
        switch self {
        case .http(let code):            return "The server returned HTTP \(code)."
        case .interrupted(let underlying): return "The download kept failing: \(underlying)"
        case .insufficientSpace(let needed, let available):
            return String(format: "Not enough storage: this needs %.1f GB free and %.1f GB is available. Free some space and try again — finished files are kept.",
                          Double(needed) / 1e9, Double(available) / 1e9)
        }
    }
}

/// Apple Archives the weights are published in, unpacked where they land.
enum WeightArchive {
    enum Failure: LocalizedError {
        case unreadable(String)

        var errorDescription: String? {
            switch self {
            case .unreadable(let name): return "Could not unpack \(name)."
            }
        }
    }

    /// Unpacks `archive` into `directory`, creating what it holds there.
    static func unpack(_ archive: URL, into directory: URL) throws {
        let name = archive.lastPathComponent
        guard let file = ArchiveByteStream.fileStream(path: FilePath(archive.path), mode: .readOnly,
                                                      options: [], permissions: FilePermissions(rawValue: 0o644))
        else { throw Failure.unreadable(name) }
        defer { try? file.close() }
        guard let decompressed = ArchiveByteStream.decompressionStream(readingFrom: file) else {
            throw Failure.unreadable(name)
        }
        defer { try? decompressed.close() }
        guard let decoded = ArchiveStream.decodeStream(readingFrom: decompressed) else {
            throw Failure.unreadable(name)
        }
        defer { try? decoded.close() }
        guard let extractor = ArchiveStream.extractStream(extractingTo: FilePath(directory.path),
                                                          flags: [.ignoreOperationNotPermitted]) else {
            throw Failure.unreadable(name)
        }
        defer { try? extractor.close() }
        _ = try ArchiveStream.process(readingFrom: decoded, writingTo: extractor)
    }
}
