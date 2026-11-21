import Foundation
import MLX

// MARK: - Your models

/// A model the user brought as a GGUF, converted for one of the app's own
/// pipelines.
///
/// A GGUF of an image or music model holds only its transformer, and a
/// GGUF of Qwen3.5 only the language model, so each is used with the rest
/// of the matching built-in model (its prompt reader, decoder, tokenizer):
/// the converted folder holds the new transformer and links to the base
/// model's other files.
struct ImportedModel: Codable, Identifiable, Hashable, Sendable {
    enum Kind: String, Codable, Sendable {
        case zImage, fluxKlein, qwenImage, aceStepXL, rewriter
    }

    /// The converted folder's name.
    let id: String
    /// The GGUF's file name without its extension.
    let name: String
    let kind: Kind
    /// The raw value of the built-in model it runs with (`ImageModel`,
    /// `MusicModel`), or "rewriter".
    let base: String
    let sourceFile: String
    let sourceBytes: Int64
    let bytes: Int64

    var directory: URL { YourModels.convertedRoot.appendingPathComponent(id, isDirectory: true) }
    var imageModel: ImageModel? { ImageModel(rawValue: base) }
    var musicModel: MusicModel? { MusicModel(rawValue: base) }
    var isImage: Bool { imageModel != nil }
    var isMusic: Bool { musicModel != nil }
    var isRewriter: Bool { kind == .rewriter }

    var baseName: String {
        imageModel?.displayName ?? musicModel?.displayName ?? "Qwen3.5 (Enhance prompt)"
    }
}

/// The models in Documents/Your Models: dropped there as `.gguf` files in
/// the Files app, converted once, then offered beside the built-in models.
@MainActor @Observable
final class YourModels {
    static let shared = YourModels()

    enum Status: Equatable {
        case converting(Double)
        /// Can't start yet, and why (a base model to download first).
        case waiting(String)
        case failed(String)
    }

    /// Converted and ready.
    private(set) var models: [ImportedModel] = []
    /// GGUF files not (yet) converted, by file name.
    private(set) var pending: [String: Status] = [:]

    private var converting: String?
    private var scanning = false

    // MARK: Selection

    /// The imported model used in place of a built-in one, by id; nil for
    /// the built-in model itself.
    var imageSelection: String? {
        didSet { UserDefaults.standard.set(imageSelection, forKey: "yourImageModel") }
    }
    var musicSelection: String? {
        didSet { UserDefaults.standard.set(musicSelection, forKey: "yourMusicModel") }
    }
    var rewriterSelection: String? {
        didSet { UserDefaults.standard.set(rewriterSelection, forKey: "yourRewriter") }
    }

    /// The imported model to use for `model`, if one is chosen for it.
    func image(for model: ImageModel) -> ImportedModel? {
        models.first { $0.id == imageSelection && $0.imageModel == model }
    }
    func music(for model: MusicModel) -> ImportedModel? {
        models.first { $0.id == musicSelection && $0.musicModel == model }
    }
    var rewriter: ImportedModel? { models.first { $0.id == rewriterSelection && $0.isRewriter } }

    // MARK: Folders

    /// For the Mac test harness (ports/gguf-harness), which converts into
    /// a scratch folder rather than ~/Documents.
    nonisolated(unsafe) static var documentsOverride: URL?
    nonisolated static var documents: URL {
        documentsOverride ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
    /// Shown in Files as On My iPhone › Whisperkit › Your Models.
    nonisolated static var folder: URL { documents.appendingPathComponent("Your Models", isDirectory: true) }
    /// Hidden from Files by its leading dot.
    nonisolated static var convertedRoot: URL { folder.appendingPathComponent(".converted", isDirectory: true) }

    private init() {
        imageSelection = UserDefaults.standard.string(forKey: "yourImageModel")
        musicSelection = UserDefaults.standard.string(forKey: "yourMusicModel")
        rewriterSelection = UserDefaults.standard.string(forKey: "yourRewriter")
        models = Self.readConverted()
    }

    // MARK: Scanning

    /// The size each file had when it last failed or had to wait: tried
    /// again once it changes (a copy still arriving), or, for waiting, on
    /// the next scan a screen asks for.
    private var lastTried: [String: Int64] = [:]

    /// Looks for new GGUF files and converts them, one at a time. Cheap
    /// when nothing changed: screens call it as they appear, which is also
    /// when a file waiting for its base model is tried again.
    func scan(retryWaiting: Bool = true) {
        guard !scanning else { return }
        scanning = true
        defer { scanning = false }
        let manager = FileManager.default
        try? manager.createDirectory(at: Self.convertedRoot, withIntermediateDirectories: true)
        if converting == nil, let leftovers = try? manager.contentsOfDirectory(at: Self.convertedRoot, includingPropertiesForKeys: nil) {
            for folder in leftovers where folder.lastPathComponent.hasSuffix(ModelImporter.inProgressSuffix) {
                try? manager.removeItem(at: folder)
            }
        }
        models = Self.readConverted()

        let files = ((try? manager.contentsOfDirectory(at: Self.folder, includingPropertiesForKeys: [.fileSizeKey])) ?? [])
            .filter { $0.pathExtension.lowercased() == "gguf" }
        var seen = Set<String>()
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let name = file.lastPathComponent
            seen.insert(name)
            let size = Int64((try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
            if models.contains(where: { $0.sourceFile == name && $0.sourceBytes == size }) {
                pending[name] = nil
                continue
            }
            guard converting == nil, converting != name else { continue }
            switch pending[name] {
            case .failed where lastTried[name] == size: continue
            case .waiting where lastTried[name] == size && !retryWaiting: continue
            default: break
            }
            start(file, size: size)
        }
        for name in pending.keys where !seen.contains(name) { pending[name] = nil }
    }

    func delete(_ model: ImportedModel) {
        try? FileManager.default.removeItem(at: model.directory)
        if imageSelection == model.id { imageSelection = nil }
        if musicSelection == model.id { musicSelection = nil }
        if rewriterSelection == model.id { rewriterSelection = nil }
        models.removeAll { $0.id == model.id }
    }

    private func start(_ file: URL, size: Int64) {
        let name = file.lastPathComponent
        converting = name
        pending[name] = .converting(0)
        Task.detached(priority: .utility) { [weak self] in
            let result: Result<ImportedModel, Error>
            do {
                result = .success(try ModelImporter.convert(file, size: size) { fraction in
                    Task { @MainActor in
                        let store = YourModels.shared
                        guard store.converting == name else { return }
                        store.pending[name] = .converting(fraction)
                    }
                })
            } catch {
                result = .failure(error)
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.converting = nil
                self.lastTried[name] = size
                switch result {
                case .success:
                    self.pending[name] = nil
                    self.models = Self.readConverted()
                case .failure(let error as ModelImporter.Waiting):
                    self.pending[name] = .waiting(error.localizedDescription)
                case .failure(let error):
                    self.pending[name] = .failed(error.localizedDescription)
                }
                self.scan(retryWaiting: false)
            }
        }
    }

    /// Converted folders with their description; a folder without one was
    /// left by a conversion that never finished, and is removed.
    nonisolated private static func readConverted() -> [ImportedModel] {
        let manager = FileManager.default
        let folders = (try? manager.contentsOfDirectory(at: convertedRoot, includingPropertiesForKeys: nil)) ?? []
        var result: [ImportedModel] = []
        for folder in folders {
            let info = folder.appendingPathComponent(ModelImporter.infoFile)
            if let data = try? Data(contentsOf: info), let model = try? JSONDecoder().decode(ImportedModel.self, from: data) {
                result.append(model)
            } else if !folder.lastPathComponent.hasSuffix(ModelImporter.inProgressSuffix) {
                try? manager.removeItem(at: folder)
            }
        }
        return result.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }
}

// MARK: - Conversion

/// Turns a GGUF into the built-in model's own file layout.
///
/// The base model's files are the specification: every tensor they hold
/// is read from the GGUF under the base's name, checked against the base's
/// shape, and written with the base's precision — 8-bit (or 4-bit) in
/// groups of 64 where the base is quantized — at the base's offsets, so
/// the pipeline reads it exactly as it reads its own. Files the GGUF has
/// nothing for (prompt reader, decoder, tokenizer) are linked, not copied.
enum ModelImporter {
    static let infoFile = "model.json"
    static let inProgressSuffix = ".partial"

    struct Waiting: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    struct Unsupported: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// Where a tensor of the base model comes from in the GGUF.
    struct Source {
        enum Rows {
            case same
            /// The `index`th of `parts` equal row blocks (a fused q/k/v).
            case part(Int)
            /// The two halves the other way round (shift and scale).
            case swapped
            case map((Int) -> Int)
        }
        var tensor: String
        var rows: Rows = .same
        /// Column `j` of the result is the source's column `columns[j]`.
        var columns: [Int]? = nil
        var value: ((Float) -> Float)? = nil
    }

    struct Plan {
        let kind: ImportedModel.Kind
        let base: String
        let baseName: String
        let directory: URL
        let files: [WeightFile]
        /// Safetensors files that must come wholly from the GGUF.
        let required: (String) -> Bool
        let source: (String) -> Source?
    }

    nonisolated static func convert(_ url: URL, size: Int64, progress: @escaping @Sendable (Double) -> Void) throws -> ImportedModel {
        let gguf = try GGUFFile(url: url)
        let plan = try plan(for: gguf)
        if !plan.files.allSatisfy({ $0.isPresent(in: plan.directory) }) {
            switch plan.kind {
            case .rewriter:
                throw Waiting(message: "Uses the built-in Enhance prompt model's tokenizer. Make one picture with Enhance prompt on so it downloads, then open this list again.")
            case .aceStepXL:
                throw Waiting(message: "Uses \(plan.baseName)'s lyric and text readers. Make one piece with \(plan.baseName) so it downloads, then open this list again.")
            default:
                throw Waiting(message: "Uses \(plan.baseName)'s prompt reader and decoder. Make one picture with \(plan.baseName) so it downloads, then open this list again.")
            }
        }

        let stem = url.deletingPathExtension().lastPathComponent
        let safe = String(stem.unicodeScalars.map { CharacterSet.alphanumerics.contains($0) || "-_.".unicodeScalars.contains($0) ? Character($0) : "-" })
        let id = "\(safe.prefix(60))-\(String(size, radix: 36))"
        let manager = FileManager.default
        let finished = YourModels.convertedRoot.appendingPathComponent(id, isDirectory: true)
        let working = YourModels.convertedRoot.appendingPathComponent(id + inProgressSuffix, isDirectory: true)
        try? manager.removeItem(at: working)
        try? manager.removeItem(at: finished)
        try manager.createDirectory(at: working, withIntermediateDirectories: true)

        do {
            // What to convert: every safetensors file the GGUF covers.
            var convert: [(WeightFile, Safetensors)] = []
            var link: [String] = []
            for file in plan.files {
                guard file.path.hasSuffix(".safetensors"), !file.isArchive else { link.append(file.path); continue }
                let header = try Safetensors(url: plan.directory.appendingPathComponent(file.path))
                let missing = header.logical.filter { plan.source($0.name).flatMap { gguf.tensors[$0.tensor] } == nil }
                if missing.isEmpty {
                    convert.append((file, header))
                } else if plan.required(file.path) {
                    throw Unsupported(message: "This GGUF doesn't match \(plan.baseName): it has no \(missing[0].name).")
                } else {
                    link.append(file.path)
                }
            }
            guard !convert.isEmpty else { throw Unsupported(message: "Nothing in this GGUF matches \(plan.baseName).") }

            let total = convert.reduce(0) { $0 + $1.1.logical.reduce(0) { $0 + $1.count } }
            var done = 0
            var written: Int64 = 0
            for (file, header) in convert {
                let out = working.appendingPathComponent(file.path)
                try write(header, to: out, from: gguf, plan: plan) { count in
                    done += count
                    progress(Double(done) / Double(max(total, 1)))
                }
                written += Int64(header.fileSize)
            }

            // Everything else, linked from the base model's folder: relative,
            // since the app's container moves when the app is updated.
            let target = relativePath(from: working, to: plan.directory)
            for name in WeightFile.topLevelNames(plan.files.filter { link.contains($0.path) || $0.isArchive })
            where !name.contains("system_state") {
                try manager.createSymbolicLink(atPath: working.appendingPathComponent(name).path,
                                               withDestinationPath: target + name)
            }

            let model = ImportedModel(id: id, name: stem, kind: plan.kind, base: plan.base,
                                      sourceFile: url.lastPathComponent, sourceBytes: size, bytes: written)
            try JSONEncoder().encode(model).write(to: working.appendingPathComponent(infoFile))
            try manager.moveItem(at: working, to: finished)
            return model
        } catch {
            try? manager.removeItem(at: working)
            throw error
        }
    }

    /// `../../..` up from `from` to Documents, then down to `to`, with a
    /// trailing slash.
    private static func relativePath(from: URL, to: URL) -> String {
        let documents = YourModels.documents.standardizedFileURL.pathComponents
        let fromParts = from.standardizedFileURL.pathComponents
        let toParts = to.standardizedFileURL.pathComponents
        let up = String(repeating: "../", count: fromParts.count - documents.count)
        return up + toParts.dropFirst(documents.count).joined(separator: "/") + "/"
    }

    // MARK: Writing

    private static func write(_ header: Safetensors, to out: URL, from gguf: GGUFFile, plan: Plan,
                              advanced: (Int) -> Void) throws {
        FileManager.default.createFile(atPath: out.path, contents: nil)
        let handle = try FileHandle(forWritingTo: out)
        defer { try? handle.close() }
        try handle.truncate(atOffset: UInt64(header.fileSize))
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: header.prefix)

        func put(_ data: Data, at offset: Int) throws {
            try handle.seek(toOffset: UInt64(header.dataStart + offset))
            try handle.write(contentsOf: data)
        }

        for tensor in header.logical {
            guard let source = plan.source(tensor.name), let from = gguf.tensors[source.tensor] else { continue }
            let columns = from.columns
            guard tensor.count % columns == 0, tensor.columns == columns || !tensor.quantized else {
                throw Unsupported(message: "This GGUF doesn't match \(plan.baseName): \(source.tensor) has \(from.shape), expected \(tensor.shape).")
            }
            let rows = tensor.count / columns
            let rowMap: (Int) -> Int
            switch source.rows {
            case .same: rowMap = { $0 }
            case .part(let index): rowMap = { $0 + index * rows }
            case .swapped: rowMap = { ($0 + rows / 2) % rows }
            case .map(let map): rowMap = map
            }
            guard (0 ..< rows).allSatisfy({ rowMap($0) < from.rows }),
                  source.columns == nil || source.columns!.count == columns,
                  rows * columns == tensor.count,
                  !(source.rows.isSame && from.count != tensor.count) else {
                throw Unsupported(message: "This GGUF doesn't match \(plan.baseName): \(source.tensor) has \(from.shape), expected \(tensor.shape).")
            }

            // About 32 MB of float32 at a time.
            let chunk = max(1, min(rows, 8_000_000 / columns))
            var buffer = [Float](repeating: 0, count: chunk * columns)
            var reordered = source.columns == nil ? [] : [Float](repeating: 0, count: chunk * columns)
            for start in stride(from: 0, to: rows, by: chunk) {
                let count = min(chunk, rows - start)
                try buffer.withUnsafeMutableBufferPointer { pointer in
                    try gguf.readRows(from, (start ..< start + count).map(rowMap), into: pointer.baseAddress!)
                }
                if let order = source.columns {
                    for r in 0 ..< count {
                        for (j, c) in order.enumerated() { reordered[r * columns + j] = buffer[r * columns + c] }
                    }
                    swap(&buffer, &reordered)
                }
                if let value = source.value {
                    for i in 0 ..< count * columns { buffer[i] = value(buffer[i]) }
                }
                let floats = MLXArray(Array(buffer[0 ..< count * columns]), [count, columns])
                if tensor.quantized, let packed = tensor.packed, let scales = tensor.scales, let biases = tensor.biases {
                    let (wq, s, b) = MLX.quantized(floats, groupSize: 64, bits: tensor.bits)
                    guard let b, wq.shape[1] == packed.shape[1], s.shape[1] == scales.shape[1] else {
                        throw Unsupported(message: "Could not pack \(tensor.name) like \(plan.baseName).")
                    }
                    try put(wq.asData().data, at: packed.offset + start * packed.rowBytes)
                    try put(s.asType(scales.dtype).asData().data, at: scales.offset + start * scales.rowBytes)
                    try put(b.asType(biases.dtype).asData().data, at: biases.offset + start * biases.rowBytes)
                } else if let plain = tensor.plain {
                    let rowBytes = columns * plain.dtype.size
                    try put(floats.asType(plain.dtype).asData().data, at: plain.offset + start * rowBytes)
                }
                advanced(count * columns)
            }
        }
    }

    // MARK: Plans

    private static func plan(for gguf: GGUFFile) throws -> Plan {
        let names = Set(gguf.tensors.keys)
        /// A name as the GGUF spells it: some writers keep ComfyUI's prefix.
        func find(_ name: String) -> String? {
            for prefix in ["", "model.diffusion_model.", "diffusion_model."] where names.contains(prefix + name) {
                return prefix + name
            }
            return nil
        }
        func source(_ name: String, rows: Source.Rows = .same) -> Source? {
            find(name).map { Source(tensor: $0, rows: rows) }
        }

        switch gguf.architecture {
        case "lumina2":
            let model = ImageModel.zImage
            return Plan(kind: .zImage, base: model.rawValue, baseName: model.displayName,
                        directory: model.weightsDirectory, files: model.weightFiles,
                        required: { $0.hasPrefix("zimage_transformer") }) { target in
                var name = target
                name = name.replacingOccurrences(of: "all_x_embedder.2-1.", with: "x_embedder.")
                name = name.replacingOccurrences(of: "all_final_layer.2-1.", with: "final_layer.")
                for (index, part) in ["to_q", "to_k", "to_v"].enumerated() where name.hasSuffix(".attention.\(part).weight") {
                    return source(name.replacingOccurrences(of: ".attention.\(part).weight", with: ".attention.qkv.weight"),
                                  rows: .part(index))
                }
                name = name.replacingOccurrences(of: ".attention.to_out.0.", with: ".attention.out.")
                name = name.replacingOccurrences(of: ".attention.norm_q.", with: ".attention.q_norm.")
                name = name.replacingOccurrences(of: ".attention.norm_k.", with: ".attention.k_norm.")
                return source(name)
            }

        case "flux":
            let model = ImageModel.fluxKlein
            return Plan(kind: .fluxKlein, base: model.rawValue, baseName: model.displayName,
                        directory: model.weightsDirectory, files: model.weightFiles,
                        required: { $0.hasPrefix("flux_transformer") }) { target in
                let fixed: [String: String] = [
                    "x_embedder.weight": "img_in.weight",
                    "context_embedder.weight": "txt_in.weight",
                    "time_guidance_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
                    "time_guidance_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
                    "double_stream_modulation_img.linear.weight": "double_stream_modulation_img.lin.weight",
                    "double_stream_modulation_txt.linear.weight": "double_stream_modulation_txt.lin.weight",
                    "single_stream_modulation.linear.weight": "single_stream_modulation.lin.weight",
                    "proj_out.weight": "final_layer.linear.weight",
                ]
                if let name = fixed[target] { return source(name) }
                // Diffusers puts the final modulation's scale first, BFL its shift.
                if target == "norm_out.linear.weight" { return source("final_layer.adaLN_modulation.1.weight", rows: .swapped) }
                let parts = target.split(separator: ".").map(String.init)
                if parts.count >= 4, parts[0] == "transformer_blocks" {
                    let block = "double_blocks.\(parts[1])"
                    let rest = parts[2...].joined(separator: ".")
                    let split: [String: (String, Int)] = [
                        "attn.to_q.weight": ("img_attn.qkv.weight", 0), "attn.to_k.weight": ("img_attn.qkv.weight", 1),
                        "attn.to_v.weight": ("img_attn.qkv.weight", 2), "attn.add_q_proj.weight": ("txt_attn.qkv.weight", 0),
                        "attn.add_k_proj.weight": ("txt_attn.qkv.weight", 1), "attn.add_v_proj.weight": ("txt_attn.qkv.weight", 2),
                    ]
                    if let (name, index) = split[rest] { return source("\(block).\(name)", rows: .part(index)) }
                    let renamed: [String: String] = [
                        "attn.norm_q.weight": "img_attn.norm.query_norm.scale", "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
                        "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
                        "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
                        "attn.to_out.0.weight": "img_attn.proj.weight", "attn.to_add_out.weight": "txt_attn.proj.weight",
                        "ff.linear_in.weight": "img_mlp.0.weight", "ff.linear_out.weight": "img_mlp.2.weight",
                        "ff_context.linear_in.weight": "txt_mlp.0.weight", "ff_context.linear_out.weight": "txt_mlp.2.weight",
                    ]
                    return renamed[rest].flatMap { source("\(block).\($0)") }
                }
                if parts.count >= 4, parts[0] == "single_transformer_blocks" {
                    let block = "single_blocks.\(parts[1])"
                    let renamed: [String: String] = [
                        "attn.to_qkv_mlp_proj.weight": "linear1.weight", "attn.to_out.weight": "linear2.weight",
                        "attn.norm_q.weight": "norm.query_norm.scale", "attn.norm_k.weight": "norm.key_norm.scale",
                    ]
                    return renamed[parts[2...].joined(separator: ".")].flatMap { source("\(block).\($0)") }
                }
                return nil
            }

        case "qwen_image":
            // Whichever build is on the device; the 8-bit one if both.
            let model = ImageModel.qwenImage.weightFiles.allSatisfy({ $0.isPresent(in: ImageModel.qwenImage.weightsDirectory) })
                || !ImageModel.qwenImageSmall.weightFiles.allSatisfy({ $0.isPresent(in: ImageModel.qwenImageSmall.weightsDirectory) })
                ? ImageModel.qwenImage : ImageModel.qwenImageSmall
            return Plan(kind: .qwenImage, base: model.rawValue, baseName: model.displayName,
                        directory: model.weightsDirectory, files: model.weightFiles,
                        required: { $0.hasPrefix("qwenimage_transformer") }) { target in
                if target.hasSuffix(".img_mlp.gate_layer.weight") {
                    return source(target.replacingOccurrences(of: ".gate_layer.", with: ".gate_up."), rows: .part(0))
                }
                if target.hasSuffix(".img_mlp.proj.weight") {
                    return source(target.replacingOccurrences(of: ".img_mlp.proj.", with: ".img_mlp.gate_up."), rows: .part(1))
                }
                return source(target)
            }

        case "acestep-dit":
            let full = MusicModel.aceStep15XLFull
            let model = full.weightFiles.allSatisfy({ $0.isPresent(in: full.weightsDirectory) })
                && !MusicModel.aceStep15XL.weightFiles.allSatisfy({ $0.isPresent(in: MusicModel.aceStep15XL.weightsDirectory) })
                ? full : MusicModel.aceStep15XL
            return Plan(kind: .aceStepXL, base: model.rawValue, baseName: model.displayName,
                        directory: model.weightsDirectory, files: model.weightFiles,
                        required: { $0.hasPrefix("ace_xl_decoder") }) { source($0) }

        case "qwen35":
            // llama.cpp stores the linear-attention value heads tiled
            // (head r·groups + g) where Hugging Face groups them (g·ratio + r).
            let groups = gguf.metadata["qwen35.ssm.group_count"] as? Int ?? 16
            let valueHeads = gguf.metadata["qwen35.ssm.time_step_rank"] as? Int ?? 32
            let ratio = max(1, valueHeads / max(groups, 1))
            let width = 128
            func head(_ h: Int) -> Int { (h % ratio) * groups + h / ratio }
            func channel(_ c: Int) -> Int { head(c / width) * width + c % width }
            let valueStart = 2 * groups * width
            return Plan(kind: .rewriter, base: "rewriter", baseName: "Enhance prompt's Qwen3.5",
                        directory: ImageModel.rewriterDirectory, files: ImageModel.rewriterFiles,
                        required: { $0.hasPrefix("rewriter_") && $0.hasSuffix(".safetensors") }) { target in
                switch target {
                case "embed_tokens.weight": return Source(tensor: "token_embd.weight")
                case "lm_head.weight": return Source(tensor: names.contains("output.weight") ? "output.weight" : "token_embd.weight")
                case "norm.weight": return Source(tensor: "output_norm.weight")
                default: break
                }
                let parts = target.split(separator: ".").map(String.init)
                guard parts.count >= 3, parts[0] == "layers" else { return nil }
                let block = "blk.\(parts[1])"
                let rest = parts[2...].joined(separator: ".")
                let renamed: [String: String] = [
                    "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "post_attention_norm.weight",
                    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
                    "mlp.down_proj.weight": "ffn_down.weight",
                    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
                    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
                    "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
                    "linear_attn.norm.weight": "ssm_norm.weight",
                ]
                if let name = renamed[rest] { return Source(tensor: "\(block).\(name)") }
                switch rest {
                case "linear_attn.in_proj_qkv.weight":
                    return Source(tensor: "\(block).attn_qkv.weight",
                                  rows: .map { $0 < valueStart ? $0 : valueStart + channel($0 - valueStart) })
                case "linear_attn.conv1d.weight":
                    return Source(tensor: "\(block).ssm_conv1d.weight",
                                  rows: .map { $0 < valueStart ? $0 : valueStart + channel($0 - valueStart) })
                case "linear_attn.in_proj_z.weight":
                    return Source(tensor: "\(block).attn_gate.weight", rows: .map(channel))
                case "linear_attn.in_proj_a.weight":
                    return Source(tensor: "\(block).ssm_alpha.weight", rows: .map(head))
                case "linear_attn.in_proj_b.weight":
                    return Source(tensor: "\(block).ssm_beta.weight", rows: .map(head))
                case "linear_attn.out_proj.weight":
                    return Source(tensor: "\(block).ssm_out.weight", columns: (0 ..< valueHeads * width).map(channel))
                case "linear_attn.dt_bias":
                    return Source(tensor: "\(block).ssm_dt.bias", columns: (0 ..< valueHeads).map(head))
                case "linear_attn.A_log":
                    // llama.cpp keeps A itself, −exp(A_log).
                    return Source(tensor: "\(block).ssm_a", columns: (0 ..< valueHeads).map(head), value: { log(-$0) })
                default: return nil
                }
            }

        default:
            let kind = gguf.architecture.isEmpty ? "an unknown model" : "“\(gguf.architecture)”"
            throw Unsupported(message: "This is \(kind). Whisperkit can use GGUFs of Z-Image-Turbo, FLUX.2 [klein] 4B, Qwen-Image 2.1, ACE-Step 1.5 XL and Qwen3.5 9B.")
        }
    }
}

private extension ModelImporter.Source.Rows {
    var isSame: Bool { if case .same = self { return true } else { return false } }
}

// MARK: - Safetensors headers

/// A base model's weights file, as a specification: names, shapes, types
/// and offsets, read from its header.
struct Safetensors {
    struct Entry {
        let dtype: DType
        let shape: [Int]
        let offset: Int
        var rowBytes: Int { (shape.dropFirst().reduce(1, *)) * dtype.size }
    }

    /// A tensor as the pipeline sees it: plain, or packed with its scales
    /// and biases.
    struct Logical {
        let name: String
        var plain: Entry?
        var packed: Entry?
        var scales: Entry?
        var biases: Entry?
        var quantized: Bool { packed != nil }
        /// The unpacked shape.
        var shape: [Int] {
            if let packed, let scales { return [packed.shape[0], scales.shape[1] * 64] }
            return plain?.shape ?? []
        }
        var columns: Int { shape.last ?? 1 }
        var count: Int { shape.reduce(1, *) }
        var bits: Int {
            guard let packed, let scales else { return 8 }
            return packed.shape[1] * 32 / (scales.shape[1] * 64)
        }
    }

    let prefix: Data
    let dataStart: Int
    let fileSize: Int
    let logical: [Logical]

    init(url: URL) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw MappedWeights.Failure.badHeader(url.lastPathComponent)
        }
        let length = Int(lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        guard let json = try handle.read(upToCount: length), json.count == length,
              let header = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw MappedWeights.Failure.badHeader(url.lastPathComponent)
        }
        prefix = lengthData + json
        dataStart = 8 + length
        fileSize = Int((try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue ?? 0)

        var groups: [String: Logical] = [:]
        for (key, value) in header where key != "__metadata__" {
            guard let entry = value as? [String: Any], let type = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int], let offsets = entry["data_offsets"] as? [Int] else { continue }
            let dtype: DType
            switch type {
            case "F16": dtype = .float16
            case "BF16": dtype = .bfloat16
            case "F32": dtype = .float32
            case "U32": dtype = .uint32
            default: throw MappedWeights.Failure.unsupportedType(type)
            }
            let item = Entry(dtype: dtype, shape: shape, offset: offsets[0])
            for suffix in [".wq", ".scales", ".biases"] where key.hasSuffix(suffix) {
                let name = String(key.dropLast(suffix.count))
                var logical = groups[name] ?? Logical(name: name)
                switch suffix {
                case ".wq": logical.packed = item
                case ".scales": logical.scales = item
                default: logical.biases = item
                }
                groups[name] = logical
            }
            if ![".wq", ".scales", ".biases"].contains(where: key.hasSuffix) {
                var logical = groups[key] ?? Logical(name: key)
                logical.plain = item
                groups[key] = logical
            }
        }
        logical = groups.values.sorted { $0.name < $1.name }
    }
}
