import Foundation
import MLX

/// Safetensors weights used in place, straight from the file.
///
/// `loadArrays` reads a file into freshly allocated memory, and that memory
/// is the app's: iOS counts every byte against the limit it kills apps at.
/// Mapping the file instead leaves the weights as clean, file-backed pages —
/// the system can drop and re-read them under pressure, so they are not
/// charged to the app the way allocated memory is. llama.cpp runs large
/// models on phones the same way.
///
/// The GPU reads the mapping directly: MLX wraps page-aligned memory in a
/// Metal buffer without copying (`newBufferWithBytesNoCopy`), and each
/// tensor is a view into that one buffer at its offset.
///
/// Files must be written with every tensor's data aligned (see
/// `ports/acestep/convert_weights.py`); a view at an odd offset would work
/// but read slowly.
enum MappedWeights {
    enum Failure: LocalizedError {
        case unreadable(String)
        case badHeader(String)
        case unsupportedType(String)

        var errorDescription: String? {
            switch self {
            case .unreadable(let name):     return "Could not open \(name)."
            case .badHeader(let name):      return "\(name) is not a valid weights file."
            case .unsupportedType(let type): return "Unsupported tensor type \(type)."
            }
        }
    }

    static func load(url: URL) throws -> [String: MLXArray] {
        let name = url.lastPathComponent
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw Failure.unreadable(name) }
        defer { close(descriptor) }

        var status = stat()
        guard fstat(descriptor, &status) == 0, status.st_size > 8 else { throw Failure.unreadable(name) }
        let size = Int(status.st_size)
        // Metal wants whole pages; the tail past end-of-file reads as zeros.
        let page = Int(vm_page_size)
        let length = (size + page - 1) / page * page

        guard let base = mmap(nil, length, PROT_READ, MAP_SHARED, descriptor, 0),
              base != MAP_FAILED else { throw Failure.unreadable(name) }

        let headerLength = Int(UInt64(littleEndian: base.load(as: UInt64.self)))
        guard headerLength > 0, 8 + headerLength <= size,
              let header = try? JSONSerialization.jsonObject(
                with: Data(bytesNoCopy: base + 8, count: headerLength, deallocator: .none))
                as? [String: Any]
        else {
            munmap(base, length)
            throw Failure.badHeader(name)
        }
        let dataStart = 8 + headerLength

        // One array over the whole mapping; unmapped when the last tensor
        // viewing it is released.
        let whole = MLXArray(rawPointer: base, [length], dtype: .uint8) {
            munmap(base, length)
        }

        var tensors: [String: MLXArray] = [:]
        for (key, value) in header where key != "__metadata__" {
            guard let entry = value as? [String: Any],
                  let type = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int],
                  let offsets = entry["data_offsets"] as? [Int], offsets.count == 2
            else { throw Failure.badHeader(name) }
            let dtype: DType
            switch type {
            case "F16":  dtype = .float16
            case "BF16": dtype = .bfloat16
            case "F32":  dtype = .float32
            case "U32":  dtype = .uint32
            case "I32":  dtype = .int32
            case "U8":   dtype = .uint8
            default:     throw Failure.unsupportedType(type)
            }
            let bytes = whole[(dataStart + offsets[0]) ..< (dataStart + offsets[1])]
            tensors[key] = bytes.view(dtype: dtype).reshaped(shape)
        }
        return tensors
    }
}
