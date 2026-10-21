import Foundation
import ImageIO
import UIKit

/// What the image models have on the device: what a run still needs to
/// download, and what deleting a model would free.
///
/// Folders are shared — SANA's two sizes, Qwen-Image's two builds, Z-Image's
/// prompt reader with FLUX — so a model's size and what deleting it frees
/// both count only what is its own or no other downloaded model needs.
/// These read file sizes from disk: call them when needed (opening the
/// picker, tapping Generate), not from a view's body.
extension ImageModel {
    typealias Placed = (file: WeightFile, directory: URL)

    /// The model's own files and any it reads from another model's folder.
    var ownFiles: [Placed] {
        weightFiles.map { ($0, weightsDirectory) }
            + (borrowedFiles.map { borrowed in borrowed.files.map { ($0, borrowed.model.weightsDirectory) } } ?? [])
    }

    /// Bytes a run still has to download: the model, the picture check every
    /// model shares, and Enhance prompt's rewriter when asked for.
    func bytesToDownload(enhance: Bool = false) -> Int64 {
        var needed = ownFiles + ImageModel.safetyFiles.map { ($0, ImageModel.safetyDirectory) }
        if enhance { needed += ImageModel.rewriterFiles.map { ($0, ImageModel.rewriterDirectory) } }
        return needed.filter { !$0.file.isPresent(in: $0.directory) }.reduce(0) { $0 + $1.file.bytes }
    }

    /// Whether everything the model itself reads is in place.
    var isDownloaded: Bool { ownFiles.allSatisfy { $0.file.isPresent(in: $0.directory) } }

    /// The files deleting this model removes: those in place that no other
    /// fully downloaded model reads.
    var removableFiles: [Placed] {
        let others = ImageModel.allCases.filter { $0 != self && $0.isDownloaded }
        return ownFiles.filter { item in
            item.file.isPresent(in: item.directory) && !others.contains { other in
                other.ownFiles.contains { $0.file.path == item.file.path && $0.directory == item.directory }
            }
        }
    }

    var removableBytes: Int64 { removableFiles.reduce(0) { $0 + $1.file.installedBytes } }

    func deleteDownload() {
        for item in removableFiles {
            for name in WeightFile.topLevelNames([item.file]) {
                try? FileManager.default.removeItem(at: item.directory.appendingPathComponent(name))
            }
        }
    }

    // MARK: - Enhance prompt's rewriter

    static var rewriterInstalledBytes: Int64 {
        rewriterFiles.filter { $0.isPresent(in: rewriterDirectory) }.reduce(0) { $0 + $1.bytes }
    }

    static func deleteRewriter() {
        try? FileManager.default.removeItem(at: rewriterDirectory)
    }

    static func formatBytes(_ bytes: Int64) -> String {
        bytes >= 1_000_000_000
            ? String(format: "%.1f GB", Double(bytes) / 1_000_000_000)
            : "\(max(1, bytes / 1_000_000)) MB"
    }
}

/// Saved pictures, decoded small for grids and kept for a while.
///
/// ImageIO makes the thumbnail straight from the file, without decoding the
/// full picture first, and off the main thread.
enum ImageThumbnails {
    private static let cache: NSCache<NSString, UIImage> = {
        let cache = NSCache<NSString, UIImage>()
        cache.countLimit = 240
        return cache
    }()

    static func cached(_ url: URL, maxPixel: Int) -> UIImage? {
        cache.object(forKey: key(url, maxPixel))
    }

    static func load(_ url: URL, maxPixel: Int) async -> UIImage? {
        if let hit = cached(url, maxPixel: maxPixel) { return hit }
        let image = await Task.detached(priority: .userInitiated) { () -> UIImage? in
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceShouldCacheImmediately: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPixel,
            ]
            guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
            return UIImage(cgImage: thumbnail)
        }.value
        if let image { cache.setObject(image, forKey: key(url, maxPixel)) }
        return image
    }

    static func forget(_ url: URL) {
        for size in [400, 1024, 2048] { cache.removeObject(forKey: key(url, size)) }
    }

    private static func key(_ url: URL, _ maxPixel: Int) -> NSString { "\(url.path)#\(maxPixel)" as NSString }
}
