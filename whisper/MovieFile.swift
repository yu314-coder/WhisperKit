import Foundation
import CoreTransferable
import UniformTypeIdentifiers

/// A movie received from the Photos picker as a *file*, not as bytes.
///
/// `loadTransferable(type: Data.self)` materialises the entire video in memory
/// before anything can be written to disk — for an hour-long recording that is
/// gigabytes of RAM, no way to report progress, and a real chance of being
/// jetsammed part-way. `FileRepresentation` hands over a URL instead, so the
/// import becomes a filesystem copy at constant memory.
struct MovieFile: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            // The received file is deleted as soon as this closure returns, so
            // it has to be copied somewhere we control first.
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let destination = FileManager.default.temporaryDirectory
                .appendingPathComponent("photos_\(UUID().uuidString)")
                .appendingPathExtension(ext)
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.copyItem(at: received.file, to: destination)
            return MovieFile(url: destination)
        }
    }
}
