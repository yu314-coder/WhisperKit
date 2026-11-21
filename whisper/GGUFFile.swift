import Foundation

/// A GGUF file (llama.cpp's format), read in place: the header parsed, the
/// tensors left in the mapped file and dequantized a row at a time.
///
/// Only what importing a model needs: metadata, tensor names and shapes,
/// and float rows. The types are the ones published models use — float,
/// the legacy 4/5/8-bit blocks, the K-quants and IQ4 — and anything else
/// is refused by name rather than read wrong.
final class GGUFFile {
    enum Failure: LocalizedError {
        case unreadable(String)
        case notGGUF
        case truncated
        case unsupportedType(tensor: String, type: UInt32)
        case missingTensor(String)

        var errorDescription: String? {
            switch self {
            case .unreadable(let name): return "Could not open \(name)."
            case .notGGUF:              return "This isn't a GGUF file."
            case .truncated:            return "The file is incomplete (did the copy finish?)."
            case .unsupportedType(let tensor, let type):
                return "\(tensor) uses a quantization the app can't read (\(GGUFFile.typeName(type))). F16, BF16, Q8_0, Q4/Q5 and the K-quants work."
            case .missingTensor(let name): return "The file has no tensor \(name)."
            }
        }
    }

    struct Tensor {
        let name: String
        /// Outermost first, the way PyTorch and MLX write shapes (GGUF lists
        /// them innermost first).
        let shape: [Int]
        let type: UInt32
        /// From the start of the tensor data.
        let offset: Int
        var columns: Int { shape.last ?? 1 }
        var rows: Int { shape.dropLast().reduce(1, *) }
        var count: Int { shape.reduce(1, *) }
    }

    let url: URL
    private(set) var metadata: [String: Any] = [:]
    private(set) var tensors: [String: Tensor] = [:]
    private let base: UnsafeMutableRawPointer
    private let length: Int
    private var dataStart = 0

    var architecture: String { metadata["general.architecture"] as? String ?? "" }

    init(url: URL) throws {
        self.url = url
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw Failure.unreadable(url.lastPathComponent) }
        defer { close(descriptor) }
        var status = stat()
        guard fstat(descriptor, &status) == 0, status.st_size > 24 else { throw Failure.notGGUF }
        length = Int(status.st_size)
        guard let mapped = mmap(nil, length, PROT_READ, MAP_SHARED, descriptor, 0), mapped != MAP_FAILED else {
            throw Failure.unreadable(url.lastPathComponent)
        }
        base = mapped
        do { try parse() } catch { munmap(base, length); throw error }
    }

    deinit { munmap(base, length) }

    // MARK: - Header

    private func parse() throws {
        var cursor = 0
        func need(_ bytes: Int) throws { if cursor + bytes > length { throw Failure.truncated } }
        func read<T: FixedWidthInteger>(_: T.Type) throws -> T {
            try need(MemoryLayout<T>.size)
            let value = base.loadUnaligned(fromByteOffset: cursor, as: T.self)
            cursor += MemoryLayout<T>.size
            return T(littleEndian: value)
        }
        func readString() throws -> String {
            let count = Int(try read(UInt64.self))
            try need(count)
            let text = String(decoding: UnsafeRawBufferPointer(start: base + cursor, count: count), as: UTF8.self)
            cursor += count
            return text
        }
        func readValue(_ type: UInt32) throws -> Any {
            switch type {
            case 0: return Int(try read(UInt8.self))
            case 1: return Int(try read(Int8.self))
            case 2: return Int(try read(UInt16.self))
            case 3: return Int(try read(Int16.self))
            case 4: return Int(try read(UInt32.self))
            case 5: return Int(try read(Int32.self))
            case 6: return Double(Float(bitPattern: try read(UInt32.self)))
            case 7: return try read(UInt8.self) != 0
            case 8: return try readString()
            case 9:
                let elementType = try read(UInt32.self)
                let count = Int(try read(UInt64.self))
                // Vocabularies run to hundreds of thousands of entries and
                // nothing here reads them: skip, keeping only the count.
                if count > 64 {
                    for _ in 0 ..< count { _ = try readValue(elementType) }
                    return count
                }
                return try (0 ..< count).map { _ in try readValue(elementType) }
            case 10: return Int(truncatingIfNeeded: try read(UInt64.self))
            case 11: return Int(try read(Int64.self))
            case 12: return Double(bitPattern: try read(UInt64.self))
            default: throw Failure.notGGUF
            }
        }

        guard length > 4, base.loadUnaligned(as: UInt32.self) == 0x4655_4747 else { throw Failure.notGGUF }  // "GGUF"
        cursor = 4
        let version = try read(UInt32.self)
        guard version >= 2 else { throw Failure.notGGUF }
        let tensorCount = Int(try read(UInt64.self))
        let entryCount = Int(try read(UInt64.self))
        for _ in 0 ..< entryCount {
            let key = try readString()
            metadata[key] = try readValue(try read(UInt32.self))
        }
        for _ in 0 ..< tensorCount {
            let name = try readString()
            let dimensions = Int(try read(UInt32.self))
            var shape: [Int] = []
            for _ in 0 ..< dimensions { shape.append(Int(try read(UInt64.self))) }
            let type = try read(UInt32.self)
            let offset = Int(try read(UInt64.self))
            tensors[name] = Tensor(name: name, shape: shape.reversed(), type: type, offset: offset)
        }
        let alignment = (metadata["general.alignment"] as? Int).flatMap { $0 > 0 ? $0 : nil } ?? 32
        dataStart = (cursor + alignment - 1) / alignment * alignment
        for tensor in tensors.values {
            guard let layout = Self.layout(tensor.type) else { continue }
            guard tensor.columns % layout.block == 0 else { throw Failure.unsupportedType(tensor: tensor.name, type: tensor.type) }
            if dataStart + tensor.offset + tensor.count / layout.block * layout.bytes > length { throw Failure.truncated }
        }
    }

    // MARK: - Rows

    /// Dequantizes rows `rows` of `tensor` (rows of its innermost dimension)
    /// into `out`, which holds `rows.count × columns` floats.
    func readRows(_ tensor: Tensor, _ rows: Range<Int>, into out: UnsafeMutablePointer<Float>) throws {
        try readRows(tensor, Array(rows), into: out)
    }

    /// The listed rows, in that order.
    func readRows(_ tensor: Tensor, _ rowList: [Int], into out: UnsafeMutablePointer<Float>) throws {
        guard let layout = Self.layout(tensor.type) else {
            throw Failure.unsupportedType(tensor: tensor.name, type: tensor.type)
        }
        let columns = tensor.columns
        let rowBytes = columns / layout.block * layout.bytes
        let start = base + dataStart + tensor.offset
        let type = tensor.type
        DispatchQueue.concurrentPerform(iterations: rowList.count) { index in
            let row = rowList[index]
            Self.dequantize(type, UnsafeRawPointer(start + row * rowBytes), columns, out + index * columns)
        }
    }

    /// One row of `count` values (a whole number of blocks).
    private static func dequantize(_ type: UInt32, _ source: UnsafeRawPointer, _ count: Int, _ out: UnsafeMutablePointer<Float>) {
        switch type {
        case 0:
            for i in 0 ..< count { out[i] = source.loadUnaligned(fromByteOffset: i * 4, as: Float.self) }
        case 1:
            for i in 0 ..< count { out[i] = Float(Float16(bitPattern: source.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))) }
        case 30:
            for i in 0 ..< count { out[i] = Float(bitPattern: UInt32(source.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)) << 16) }
        default:
            let layout = layout(type)!
            for block in 0 ..< count / layout.block {
                let b = source + block * layout.bytes
                let y = out + block * layout.block
                switch type {
                case 2: q4_0(b, y)
                case 3: q4_1(b, y)
                case 6: q5_0(b, y)
                case 7: q5_1(b, y)
                case 8: q8_0(b, y)
                case 10: q2_k(b, y)
                case 11: q3_k(b, y)
                case 12: q4_k(b, y)
                case 13: q5_k(b, y)
                case 14: q6_k(b, y)
                case 20: iq4_nl(b, y)
                case 23: iq4_xs(b, y)
                default: break
                }
            }
        }
    }

    /// Values per block and bytes per block; nil for a type not read here.
    static func layout(_ type: UInt32) -> (block: Int, bytes: Int)? {
        switch type {
        case 0: return (1, 4)
        case 1, 30: return (1, 2)
        case 2: return (32, 18)
        case 3: return (32, 20)
        case 6: return (32, 22)
        case 7: return (32, 24)
        case 8: return (32, 34)
        case 10: return (256, 84)
        case 11: return (256, 110)
        case 12: return (256, 144)
        case 13: return (256, 176)
        case 14: return (256, 210)
        case 20: return (32, 18)
        case 23: return (256, 136)
        default: return nil
        }
    }

    static func typeName(_ type: UInt32) -> String {
        let names: [UInt32: String] = [0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
                                       9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
                                       15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
                                       20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M",
                                       30: "BF16", 34: "TQ1_0", 35: "TQ2_0"]
        return names[type] ?? "type \(type)"
    }

    // MARK: - Blocks (ggml-quants.c)

    @inline(__always) private static func half(_ p: UnsafeRawPointer, _ offset: Int) -> Float {
        Float(Float16(bitPattern: p.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
    }
    @inline(__always) private static func byte(_ p: UnsafeRawPointer, _ offset: Int) -> UInt8 {
        p.load(fromByteOffset: offset, as: UInt8.self)
    }

    private static func q4_0(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0)
        for j in 0 ..< 16 {
            let q = byte(b, 2 + j)
            y[j] = Float(Int(q & 0xF) - 8) * d
            y[j + 16] = Float(Int(q >> 4) - 8) * d
        }
    }

    private static func q4_1(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0), m = half(b, 2)
        for j in 0 ..< 16 {
            let q = byte(b, 4 + j)
            y[j] = Float(q & 0xF) * d + m
            y[j + 16] = Float(q >> 4) * d + m
        }
    }

    private static func q5_0(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0)
        let qh = b.loadUnaligned(fromByteOffset: 2, as: UInt32.self)
        for j in 0 ..< 16 {
            let q = byte(b, 6 + j)
            let h0 = UInt8(((qh >> UInt32(j)) << 4) & 0x10)
            let h1 = UInt8((qh >> UInt32(j + 12)) & 0x10)
            y[j] = Float(Int((q & 0xF) | h0) - 16) * d
            y[j + 16] = Float(Int((q >> 4) | h1) - 16) * d
        }
    }

    private static func q5_1(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0), m = half(b, 2)
        let qh = b.loadUnaligned(fromByteOffset: 4, as: UInt32.self)
        for j in 0 ..< 16 {
            let q = byte(b, 8 + j)
            let h0 = UInt8(((qh >> UInt32(j)) << 4) & 0x10)
            let h1 = UInt8((qh >> UInt32(j + 12)) & 0x10)
            y[j] = Float((q & 0xF) | h0) * d + m
            y[j + 16] = Float((q >> 4) | h1) * d + m
        }
    }

    private static func q8_0(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0)
        for j in 0 ..< 32 { y[j] = Float(b.load(fromByteOffset: 2 + j, as: Int8.self)) * d }
    }

    private static func q2_k(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        // scales[16], qs[64], d, dmin
        let d = half(b, 80), dmin = half(b, 82)
        var out = 0, scale = 0
        for n in stride(from: 0, to: 256, by: 128) {
            let q = 16 + n / 4
            var shift: UInt8 = 0
            for _ in 0 ..< 4 {
                for half in 0 ..< 2 {
                    let sc = byte(b, scale); scale += 1
                    let dl = d * Float(sc & 0xF), ml = dmin * Float(sc >> 4)
                    for l in 0 ..< 16 {
                        y[out] = dl * Float((byte(b, q + half * 16 + l) >> shift) & 3) - ml
                        out += 1
                    }
                }
                shift += 2
            }
        }
    }

    private static func q3_k(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        // hmask[32], qs[64], scales[12], d
        let d = half(b, 108)
        var aux = [UInt32](repeating: 0, count: 4)
        for i in 0 ..< 3 { aux[i] = b.loadUnaligned(fromByteOffset: 96 + 4 * i, as: UInt32.self) }
        let kmask1: UInt32 = 0x0303_0303, kmask2: UInt32 = 0x0f0f_0f0f
        let tmp = aux[2]
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
        var scales = [Int8](repeating: 0, count: 16)
        for i in 0 ..< 16 { scales[i] = Int8(bitPattern: UInt8((aux[i / 4] >> UInt32(8 * (i % 4))) & 0xFF)) }
        var out = 0, scale = 0
        var m: UInt8 = 1
        for n in stride(from: 0, to: 256, by: 128) {
            let q = 32 + n / 4
            var shift: UInt8 = 0
            for _ in 0 ..< 4 {
                for half in 0 ..< 2 {
                    let dl = d * Float(Int(scales[scale]) - 32); scale += 1
                    for l in 0 ..< 16 {
                        let index = half * 16 + l
                        let low = Int((byte(b, q + index) >> shift) & 3)
                        let high = byte(b, index) & m != 0 ? 0 : 4
                        y[out] = dl * Float(low - high)
                        out += 1
                    }
                }
                shift += 2
                m <<= 1
            }
        }
    }

    @inline(__always) private static func scaleMin(_ j: Int, _ b: UnsafeRawPointer, _ at: Int) -> (Float, Float) {
        if j < 4 {
            return (Float(byte(b, at + j) & 63), Float(byte(b, at + j + 4) & 63))
        }
        let d = (byte(b, at + j + 4) & 0xF) | ((byte(b, at + j - 4) >> 6) << 4)
        let m = (byte(b, at + j + 4) >> 4) | ((byte(b, at + j) >> 6) << 4)
        return (Float(d), Float(m))
    }

    private static func q4_k(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        // d, dmin, scales[12], qs[128]
        let d = half(b, 0), dmin = half(b, 2)
        var out = 0, q = 16, scale = 0
        for _ in stride(from: 0, to: 256, by: 64) {
            let (s1, m1) = scaleMin(scale, b, 4), (s2, m2) = scaleMin(scale + 1, b, 4)
            let d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2
            for l in 0 ..< 32 { y[out + l] = d1 * Float(byte(b, q + l) & 0xF) - mm1 }
            for l in 0 ..< 32 { y[out + 32 + l] = d2 * Float(byte(b, q + l) >> 4) - mm2 }
            out += 64; q += 32; scale += 2
        }
    }

    private static func q5_k(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        // d, dmin, scales[12], qh[32], qs[128]
        let d = half(b, 0), dmin = half(b, 2)
        var out = 0, ql = 48, scale = 0
        var u1: UInt8 = 1, u2: UInt8 = 2
        for _ in stride(from: 0, to: 256, by: 64) {
            let (s1, m1) = scaleMin(scale, b, 4), (s2, m2) = scaleMin(scale + 1, b, 4)
            let d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2
            for l in 0 ..< 32 {
                let h = byte(b, 16 + l)
                y[out + l] = d1 * Float(Int(byte(b, ql + l) & 0xF) + (h & u1 != 0 ? 16 : 0)) - mm1
                y[out + 32 + l] = d2 * Float(Int(byte(b, ql + l) >> 4) + (h & u2 != 0 ? 16 : 0)) - mm2
            }
            out += 64; ql += 32; scale += 2
            u1 <<= 2; u2 <<= 2
        }
    }

    private static func q6_k(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        // ql[128], qh[64], scales[16] (int8), d
        let d = half(b, 208)
        var out = 0, ql = 0, qh = 128, sc = 192
        for _ in stride(from: 0, to: 256, by: 128) {
            for l in 0 ..< 32 {
                let s = l / 16
                let h = byte(b, qh + l)
                let q1 = Int((byte(b, ql + l) & 0xF) | ((h & 3) << 4)) - 32
                let q2 = Int((byte(b, ql + l + 32) & 0xF) | (((h >> 2) & 3) << 4)) - 32
                let q3 = Int((byte(b, ql + l) >> 4) | (((h >> 4) & 3) << 4)) - 32
                let q4 = Int((byte(b, ql + l + 32) >> 4) | (((h >> 6) & 3) << 4)) - 32
                func scale(_ i: Int) -> Float { Float(b.load(fromByteOffset: sc + i, as: Int8.self)) }
                y[out + l] = d * scale(s) * Float(q1)
                y[out + l + 32] = d * scale(s + 2) * Float(q2)
                y[out + l + 64] = d * scale(s + 4) * Float(q3)
                y[out + l + 96] = d * scale(s + 6) * Float(q4)
            }
            out += 128; ql += 64; qh += 32; sc += 8
        }
    }

    private static let iq4Values: [Float] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

    private static func iq4_nl(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        let d = half(b, 0)
        for j in 0 ..< 16 {
            let q = byte(b, 2 + j)
            y[j] = d * iq4Values[Int(q & 0xF)]
            y[j + 16] = d * iq4Values[Int(q >> 4)]
        }
    }

    private static func iq4_xs(_ b: UnsafeRawPointer, _ y: UnsafeMutablePointer<Float>) {
        // d, scales_h (u16), scales_l[4], qs[128]
        let d = half(b, 0)
        let high = b.loadUnaligned(fromByteOffset: 2, as: UInt16.self)
        for ib in 0 ..< 8 {
            let low = (byte(b, 4 + ib / 2) >> UInt8(4 * (ib % 2))) & 0xF
            let ls = Int(low) | (Int((high >> UInt16(2 * ib)) & 3) << 4)
            let dl = d * Float(ls - 32)
            for j in 0 ..< 16 {
                let q = byte(b, 8 + ib * 16 + j)
                y[ib * 32 + j] = dl * iq4Values[Int(q & 0xF)]
                y[ib * 32 + j + 16] = dl * iq4Values[Int(q >> 4)]
            }
        }
    }
}
