import Foundation

/// Downloads one weight file, reporting progress.
///
/// Uses its own `URLSession` with a **session**-level delegate rather than
/// `URLSession.shared.download(from:delegate:)`. The async form fulfils its
/// continuation through an internal delegate of its own, and task-level
/// `didWriteData` callbacks do not reliably arrive — which left the progress
/// bar sitting at zero for the whole download on device.
///
/// An interrupted transfer hands back resume data, so a retry continues from
/// where it stopped instead of restarting a gigabyte.
final class MusicDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    struct Progress: Sendable {
        let fraction: Double
        let received: Int64
        let expected: Int64
    }

    private var session: URLSession!
    private var continuation: CheckedContinuation<URL, Error>?
    private var onProgress: (@Sendable (Progress) -> Void)?
    private var destination: URL?
    private var lastReported = -1.0

    override init() {
        super.init()
        let configuration = URLSessionConfiguration.default
        // The whole transfer can legitimately take many minutes on a slow
        // connection; only a stall should count as a timeout.
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 3600
        configuration.waitsForConnectivity = true
        configuration.allowsCellularAccess = true
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    deinit { session.invalidateAndCancel() }

    /// Downloads `remote` to `destination`. `resumeData`, when supplied,
    /// continues a previously interrupted transfer.
    func download(
        from remote: URL,
        to destination: URL,
        resumeData: Data? = nil,
        onProgress: @escaping @Sendable (Progress) -> Void
    ) async throws -> URL {
        self.destination = destination
        self.onProgress = onProgress
        self.lastReported = -1

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                self.continuation = continuation
                let task = resumeData.map { session.downloadTask(withResumeData: $0) }
                    ?? session.downloadTask(with: remote)
                task.resume()
            }
        } onCancel: {
            self.session.invalidateAndCancel()
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
        // A gigabyte file posts these constantly; a redraw per half percent is
        // plenty and keeps the main actor free.
        guard fraction - lastReported >= 0.005 else { return }
        lastReported = fraction
        onProgress?(Progress(fraction: fraction,
                             received: totalBytesWritten,
                             expected: totalBytesExpectedToWrite))
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // This file is deleted as soon as this method returns, so it has to be
        // moved now, synchronously, not from the continuation.
        guard let destination else { return }
        do {
            if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
                throw MusicDownloaderError.http(http.statusCode)
            }
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
            continuation?.resume(returning: destination)
        } catch {
            continuation?.resume(throwing: error)
        }
        continuation = nil
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        guard let error else { return }   // success already resumed above
        let resumeData = (error as NSError)
            .userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        continuation?.resume(throwing: MusicDownloaderError.interrupted(
            underlying: error.localizedDescription, resumeData: resumeData))
        continuation = nil
    }
}

enum MusicDownloaderError: LocalizedError {
    case http(Int)
    case interrupted(underlying: String, resumeData: Data?)

    var errorDescription: String? {
        switch self {
        case .http(let code):
            return "The server returned HTTP \(code)."
        case .interrupted(let underlying, _):
            return underlying
        }
    }

    var resumeData: Data? {
        if case .interrupted(_, let data) = self { return data }
        return nil
    }
}
