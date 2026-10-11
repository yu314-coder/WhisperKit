import Foundation

/// The Transcribe tab's Whisper models, fetched from Hugging Face with the
/// app's background downloader rather than WhisperKit's.
///
/// WhisperKit's downloader streams each file through a data task, and those
/// stop when iOS suspends the app: a background session carries on only
/// download tasks while the app is away (and WhisperKit's "background"
/// option opens a new session under the same identifier for every file).
/// So this lists the model's files and hands them all to
/// `WeightDownloadService`, which keeps going in the background — through
/// the same fetcher as the Music and Image models.
///
/// Files land where WhisperKit puts them —
/// `<base>/models/argmaxinc/whisperkit-coreml/<variant>/…`, and the
/// tokenizer under `<base>/models/openai/whisper-…` — each with the
/// metadata its Hub client writes beside a download (commit, etag, time).
/// A later load through WhisperKit's checking path then recognises them
/// rather than fetching them again.
enum WhisperModelDownload {
    private static let repo = "argmaxinc/whisperkit-coreml"
    private static let endpoint = "https://huggingface.co"

    enum Failure: LocalizedError {
        case listing(Int)
        case unreadableListing
        case noFiles(String)

        var errorDescription: String? {
            switch self {
            case .listing(let code):  return "Hugging Face answered with error \(code) when asked for the model's files."
            case .unreadableListing:  return "Hugging Face's list of the model's files couldn't be read."
            case .noFiles(let name):  return "Hugging Face lists no files for \(name)."
            }
        }
    }

    /// The tokenizer WhisperKit pairs with a model, from OpenAI's repository
    /// for its size: Small its own, every Large V3 build (Turbo included)
    /// Large V3's. Fetched with the model, or WhisperKit would download it at
    /// the first load, in the foreground.
    private static func tokenizerRepo(for variant: String) -> String {
        variant.contains("small") ? "openai/whisper-small" : "openai/whisper-large-v3"
    }
    private static let tokenizerFiles: Set<String> = ["tokenizer.json", "tokenizer_config.json", "config.json"]

    /// Downloads whatever of `variant`, and its tokenizer, isn't already in
    /// place under `base` — all in one batch, every file in flight at once.
    @MainActor
    static func fetch(variant: String, base: URL,
                      progress: @escaping (Double) -> Void,
                      note: @escaping (String?) -> Void = { _ in }) async throws {
        let model = try await listing(repo: repo, folder: variant)
        let tokenizer = try await listing(repo: tokenizerRepo(for: variant), only: tokenizerFiles)
        let roots = [model, tokenizer].map { base.appendingPathComponent("models/\($0.repo)", isDirectory: true) }
        try await WeightFetcher.fetch([
            .init(files: model.files, directory: roots[0]),
            .init(files: tokenizer.files, directory: roots[1]),
        ], progress: { progress($0.fraction) }, note: note)

        let now = Date().timeIntervalSince1970
        for (listing, root) in zip([model, tokenizer], roots) {
            let metadata = root.appendingPathComponent(".cache/huggingface/download", isDirectory: true)
            for file in listing.files {
                guard let etag = listing.etags[file.path] else { continue }
                let path = metadata.appendingPathComponent(file.path + ".metadata")
                try? FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? "\(listing.commit)\n\(etag)\n\(now)\n".write(to: path, atomically: true, encoding: .utf8)
            }
        }
    }

    private struct Listing {
        let repo: String
        let files: [WeightFile]
        /// An LFS file's etag is its SHA-256; any other's, its git id —
        /// what Hugging Face sends as `X-Linked-Etag`.
        let etags: [String: String]
        let commit: String
    }

    /// The files of `repo` under `folder` (recursively), or at its top level
    /// named in `only`, with their sizes and etags, at the current commit.
    private static func listing(repo: String, folder: String? = nil, only names: Set<String>? = nil) async throws -> Listing {
        struct Revision: Decodable { let sha: String }
        struct Entry: Decodable {
            struct LFS: Decodable { let oid: String }
            let type: String
            let path: String
            let size: Int64
            let oid: String
            let lfs: LFS?
        }

        let revision: Revision = try await decode(URL(string: "\(endpoint)/api/models/\(repo)/revision/main?expand%5B%5D=sha")!).0

        var entries: [Entry] = []
        let scope = folder.map { "/\($0)?recursive=true" } ?? ""
        var next = URL(string: "\(endpoint)/api/models/\(repo)/tree/\(revision.sha)\(scope)")
        while let url = next {
            let (page, response): ([Entry], HTTPURLResponse) = try await decode(url)
            entries += page
            next = nextPage(response)
        }

        var files: [WeightFile] = []
        var etags: [String: String] = [:]
        for entry in entries where entry.type == "file" && (names?.contains(entry.path) ?? true) {
            let encoded = entry.path.split(separator: "/")
                .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
                .joined(separator: "/")
            guard let source = URL(string: "\(endpoint)/\(repo)/resolve/\(revision.sha)/\(encoded)") else { continue }
            files.append(WeightFile(path: entry.path, bytes: entry.size, release: "", source: source))
            etags[entry.path] = entry.lfs?.oid ?? entry.oid
        }
        guard !files.isEmpty else { throw Failure.noFiles(folder ?? repo) }
        return Listing(repo: repo, files: files, etags: etags, commit: revision.sha)
    }

    private static func decode<T: Decodable>(_ url: URL) async throws -> (T, HTTPURLResponse) {
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse else { throw Failure.unreadableListing }
        guard http.statusCode == 200 else { throw Failure.listing(http.statusCode) }
        guard let value = try? JSONDecoder().decode(T.self, from: data) else { throw Failure.unreadableListing }
        return (value, http)
    }

    /// Hugging Face pages long listings with `Link: <…>; rel="next"`.
    private static func nextPage(_ response: HTTPURLResponse) -> URL? {
        guard let link = response.value(forHTTPHeaderField: "Link") else { return nil }
        for part in link.split(separator: ",") where part.contains("rel=\"next\"") {
            guard let open = part.firstIndex(of: "<"), let close = part.firstIndex(of: ">") else { continue }
            return URL(string: String(part[part.index(after: open) ..< close]))
        }
        return nil
    }
}
