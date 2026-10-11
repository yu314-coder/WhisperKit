import Foundation
import SwiftData
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// A picture made on the Image tab, kept so it can be seen and made again.
///
/// As with music and transcripts, SwiftData holds the description and the
/// file system holds the pixels.
@Model
final class SavedImage {
    var id: UUID
    var prompt: String
    var createdAt: Date
    var modelName: String
    /// Relative to Documents, under `ImageFiles.folderName`.
    var filePath: String
    var width: Int
    var height: Int
    /// With the model and prompt, makes the same picture again.
    var seed: Int64
    /// How long it took, start to finish, in seconds.
    var seconds: Double

    init(id: UUID = UUID(), prompt: String, createdAt: Date = Date(), modelName: String,
         filePath: String, width: Int, height: Int, seed: Int64, seconds: Double) {
        self.id = id
        self.prompt = prompt
        self.createdAt = createdAt
        self.modelName = modelName
        self.filePath = filePath
        self.width = width
        self.height = height
        self.seed = seed
        self.seconds = seconds
    }

    var fileURL: URL { ImageFiles.url(for: filePath) }
}

/// Where made pictures are kept: Documents/MadeImages, by relative path, so
/// a restored or reinstalled container still finds them.
enum ImageFiles {
    static let folderName = "MadeImages"

    static var folderURL: URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documents.appendingPathComponent(folderName, isDirectory: true)
    }

    static func url(for relative: String) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(relative)
    }

    enum Failure: LocalizedError {
        case unwritable
        var errorDescription: String? { "The picture could not be saved." }
    }

    /// Writes `image` as a PNG and returns its relative path.
    ///
    /// The file says what it is: IPTC's "trained algorithmic media" source
    /// type, which Photos, news desks and the big platforms read as "made by
    /// AI", with the prompt as its description and the model as the
    /// software. Nothing is drawn on the picture.
    static func save(_ image: CGImage, prompt: String, model: String) throws -> String {
        try FileManager.default.createDirectory(at: folderURL, withIntermediateDirectories: true)
        let name = "Image_\(Int(Date().timeIntervalSince1970 * 1000)).png"
        let url = folderURL.appendingPathComponent(name)
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw Failure.unwritable
        }
        CGImageDestinationAddImageAndMetadata(destination, image, provenance(prompt: prompt, model: model), nil)
        guard CGImageDestinationFinalize(destination) else { throw Failure.unwritable }
        return "\(folderName)/\(name)"
    }

    static func delete(_ relative: String) {
        try? FileManager.default.removeItem(at: url(for: relative))
    }

    private static func provenance(prompt: String, model: String) -> CGImageMetadata {
        let metadata = CGImageMetadataCreateMutable()
        let iptcExtension = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/" as CFString
        CGImageMetadataRegisterNamespaceForPrefix(metadata, iptcExtension, "Iptc4xmpExt" as CFString, nil)
        func set(_ path: String, _ value: String) {
            _ = CGImageMetadataSetValueWithPath(metadata, nil, path as CFString, value as CFString)
        }
        set("Iptc4xmpExt:DigitalSourceType", "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia")
        set("xmp:CreatorTool", "WhisperKit — \(model), on device")
        // A description is language-tagged text in XMP, not a plain string.
        if let tag = CGImageMetadataTagCreate("http://purl.org/dc/elements/1.1/" as CFString, "dc" as CFString,
                                              "description" as CFString, .alternateText,
                                              ["x-default": prompt] as CFDictionary) {
            CGImageMetadataSetTagWithPath(metadata, nil, "dc:description" as CFString, tag)
        }
        return metadata
    }
}
