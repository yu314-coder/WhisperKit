import Foundation

/// Downloads the audio or video file a pasted link points to, for the
/// Transcribe tab: a direct link (".mp3", ".m4a", ".mp4"…), or a web page
/// that embeds one — many podcast episode pages name their audio in an
/// `og:audio` tag or an `<audio>` element.
///
/// Video platforms are refused by name. App Review guideline 5.2.3 bars
/// downloading media from third-party services such as YouTube without
/// their permission, and their terms forbid it; their pages don't link a
/// file anyway, so there is nothing to follow.
enum MediaLink {
    enum Failure: LocalizedError, Equatable {
        case notALink
        case notSecure
        case platform(String)
        case notMedia
        case http(Int)

        var errorDescription: String? {
            switch self {
            case .notALink:
                return "That doesn't look like a link. Paste a web address that starts with https://."
            case .notSecure:
                return "Only secure links (https://) can be downloaded."
            case .platform(let name):
                return "\(name) videos can't be downloaded — \(name)'s terms and the App Store's rules don't allow it. If the video is yours, save it to Photos or Files and import it from there."
            case .notMedia:
                return "That link leads to a web page, not an audio or video file. Use a link that ends in a file such as .mp3, .m4a or .mp4, or import the file from Files."
            case .http(let code):
                return "The server answered with error \(code). Check the link and try again."
            }
        }
    }

    /// Hosts whose media can't be downloaded, with the name to show.
    private static let platforms: [(host: String, name: String)] = [
        ("youtube.com", "YouTube"), ("youtu.be", "YouTube"), ("youtube-nocookie.com", "YouTube"),
        ("instagram.com", "Instagram"), ("facebook.com", "Facebook"), ("fb.watch", "Facebook"),
        ("fb.com", "Facebook"), ("tiktok.com", "TikTok"), ("twitter.com", "X"), ("x.com", "X"),
        ("vimeo.com", "Vimeo"), ("soundcloud.com", "SoundCloud"), ("spotify.com", "Spotify"),
        ("music.apple.com", "Apple Music"), ("podcasts.apple.com", "Apple Podcasts"),
        ("twitch.tv", "Twitch"), ("bilibili.com", "Bilibili"), ("b23.tv", "Bilibili"),
        ("douyin.com", "Douyin"), ("xiaohongshu.com", "Xiaohongshu"), ("netflix.com", "Netflix"),
        ("dailymotion.com", "Dailymotion"), ("threads.net", "Threads"), ("snapchat.com", "Snapchat"),
    ]

    /// The first web address in pasted text: people paste a whole message
    /// as often as a bare link.
    static func url(in text: String) -> URL? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue),
              let match = detector.firstMatch(in: trimmed, range: NSRange(trimmed.startIndex..., in: trimmed)),
              let url = match.url, let scheme = url.scheme?.lowercased(), scheme == "https" || scheme == "http" else {
            return nil
        }
        return url
    }

    /// Why `url` can't be downloaded, before trying; nil when it may be.
    static func refusal(for url: URL) -> Failure? {
        guard url.scheme?.lowercased() == "https" else { return .notSecure }
        let host = (url.host ?? "").lowercased()
        if let platform = platforms.first(where: { host == $0.host || host.hasSuffix("." + $0.host) }) {
            return .platform(platform.name)
        }
        return nil
    }

    /// Downloads the media `url` leads to into a temporary file named with
    /// the right extension, following one web page to the file it embeds.
    /// `progress` gets the fraction done, or nil while the size is unknown.
    static func download(_ url: URL, progress: @escaping @Sendable (Double?) -> Void) async throws -> URL {
        if let refusal = refusal(for: url) { throw refusal }
        let (file, response) = try await Download.run(url, progress: progress)
        if isMedia(response, url: url) { return try keep(file, response: response, url: url) }

        // A page: look for the file it plays.
        defer { try? FileManager.default.removeItem(at: file) }
        guard let data = try? Data(contentsOf: file), data.count < 5_000_000,
              let html = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1),
              let embedded = embeddedMedia(in: html, base: response.url ?? url) else {
            throw Failure.notMedia
        }
        if let refusal = refusal(for: embedded) { throw refusal }
        let (second, secondResponse) = try await Download.run(embedded, progress: progress)
        guard isMedia(secondResponse, url: embedded) else {
            try? FileManager.default.removeItem(at: second)
            throw Failure.notMedia
        }
        return try keep(second, response: secondResponse, url: embedded)
    }

    // MARK: - Recognising media

    private static let mediaExtensions: Set<String> = [
        "mp3", "m4a", "m4b", "aac", "wav", "aif", "aiff", "caf", "flac", "ogg", "oga", "opus",
        "mp4", "m4v", "mov", "3gp", "webm", "mkv", "amr",
    ]

    private static func isMedia(_ response: HTTPURLResponse, url: URL) -> Bool {
        let type = (response.mimeType ?? "").lowercased()
        if type.hasPrefix("audio/") || type.hasPrefix("video/") || type == "application/ogg" { return true }
        let named = [response.url?.pathExtension, url.pathExtension, (response.suggestedFilename as NSString?)?.pathExtension]
            .compactMap { $0?.lowercased() }
        return (type == "application/octet-stream" || type == "binary/octet-stream" || type.isEmpty)
            && named.contains(where: mediaExtensions.contains)
    }

    /// The media file a page embeds: `og:audio`, `og:video`, or the source
    /// of an `<audio>`, `<video>` or `<source>` element.
    static func embeddedMedia(in html: String, base: URL) -> URL? {
        let patterns = [
            #"<meta[^>]+property=["']og:audio(?::secure_url|:url)?["'][^>]+content=["']([^"']+)["']"#,
            #"<meta[^>]+content=["']([^"']+)["'][^>]+property=["']og:audio(?::secure_url|:url)?["']"#,
            #"<(?:audio|source)[^>]+src=["']([^"']+)["']"#,
            #"<meta[^>]+property=["']og:video(?::secure_url|:url)?["'][^>]+content=["']([^"']+)["']"#,
            #"<video[^>]+src=["']([^"']+)["']"#,
        ]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else { continue }
            for match in regex.matches(in: html, range: NSRange(html.startIndex..., in: html)) {
                guard let range = Range(match.range(at: 1), in: html) else { continue }
                let raw = String(html[range]).replacingOccurrences(of: "&amp;", with: "&")
                guard let found = URL(string: raw, relativeTo: base)?.absoluteURL,
                      found.scheme?.lowercased() == "https",
                      mediaExtensions.contains(found.pathExtension.lowercased()) else { continue }
                return found
            }
        }
        return nil
    }

    /// Moves a finished download to a temporary file whose extension says
    /// what it is, which is how the importer tells audio from video.
    private static func keep(_ file: URL, response: HTTPURLResponse, url: URL) throws -> URL {
        let suggested = response.suggestedFilename.flatMap { $0.isEmpty ? nil : $0 } ?? url.lastPathComponent
        var name = suggested.isEmpty ? "Download" : suggested
        if !mediaExtensions.contains((name as NSString).pathExtension.lowercased()) {
            name += "." + fileExtension(for: response.mimeType ?? "")
        }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("Links", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let destination = folder.appendingPathComponent(name.replacingOccurrences(of: "/", with: "-"))
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: file, to: destination)
        return destination
    }

    private static func fileExtension(for mime: String) -> String {
        switch mime.lowercased() {
        case "audio/mpeg", "audio/mp3": return "mp3"
        case "audio/mp4", "audio/x-m4a", "audio/m4a", "audio/aac", "audio/aacp": return "m4a"
        case "audio/wav", "audio/x-wav", "audio/wave": return "wav"
        case "audio/flac", "audio/x-flac": return "flac"
        case "audio/ogg", "application/ogg": return "ogg"
        case "audio/webm", "video/webm": return "webm"
        case "video/quicktime": return "mov"
        case "audio/aiff", "audio/x-aiff": return "aiff"
        default: return mime.lowercased().hasPrefix("audio/") ? "m4a" : "mp4"
        }
    }

    // MARK: - Downloading with progress

    /// One download, with progress, as an async call.
    private final class Download: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        private let progress: @Sendable (Double?) -> Void
        private var continuation: CheckedContinuation<(URL, HTTPURLResponse), Error>?
        private var kept: URL?

        private init(progress: @escaping @Sendable (Double?) -> Void) { self.progress = progress }

        static func run(_ url: URL, progress: @escaping @Sendable (Double?) -> Void) async throws -> (URL, HTTPURLResponse) {
            let download = Download(progress: progress)
            let session = URLSession(configuration: .default, delegate: download, delegateQueue: nil)
            defer { session.finishTasksAndInvalidate() }
            let task = session.downloadTask(with: url)
            return try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    download.continuation = continuation
                    task.resume()
                }
            } onCancel: {
                task.cancel()
            }
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                        totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
            progress(totalBytesExpectedToWrite > 0 ? Double(totalBytesWritten) / Double(totalBytesExpectedToWrite) : nil)
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
            // The file is deleted when this returns; move it somewhere ours.
            let copy = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try? FileManager.default.moveItem(at: location, to: copy)
            kept = copy
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            defer { continuation = nil }
            if let error {
                if let kept { try? FileManager.default.removeItem(at: kept) }
                continuation?.resume(throwing: error)
                return
            }
            guard let response = task.response as? HTTPURLResponse, let kept else {
                continuation?.resume(throwing: Failure.notMedia)
                return
            }
            guard (200 ..< 300).contains(response.statusCode) else {
                try? FileManager.default.removeItem(at: kept)
                continuation?.resume(throwing: Failure.http(response.statusCode))
                return
            }
            continuation?.resume(returning: (kept, response))
        }
    }
}
