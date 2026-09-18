import Foundation
import MLX

/// EnCodec's decoder: four residual-VQ codebooks back to 32 kHz audio.
///
/// Two pieces have no MLX equivalent and are built here:
///
/// * **Reflect padding.** EnCodec pads every convolution by reflection, not
///   zeros. Zero padding leaves an audible click at each end. Implemented as a
///   gather, since the reflected index pattern is easy to state directly.
/// * **LSTM.** Two layers, 1024 wide, run at frame rate between the input
///   convolution and the upsampling stack, with a skip connection around the
///   pair. PyTorch packs its gates as input, forget, cell, output in one
///   tensor, and that order is what the checkpoint carries.
struct MGCodec {
    static let codebookCount = 4
    static let latentDimension = 128
    static let upsamplingRatios = [8, 5, 4, 4]

    let weights: [String: MLXArray]

    private func w(_ key: String) -> MLXArray {
        guard let value = weights["audio_encoder.\(key)"] else {
            fatalError("missing codec weight: audio_encoder.\(key)")
        }
        return value.asType(.float32)
    }

    /// - Parameter codes: (codebooks, frames) indices.
    /// - Returns: (1, samples) mono audio at 32 kHz.
    func decode(codes: MLXArray) -> MLXArray {
        let frames = codes.dim(1)

        // Residual VQ: the codebooks are summed, each one refining the last.
        var latent = MLXArray.zeros([frames, Self.latentDimension], dtype: .float32)
        for book in 0 ..< Self.codebookCount {
            latent = latent + take(w("quantizer.layers.\(book).codebook.embed"),
                                   codes[book, 0...], axis: 0)
        }
        var h = latent.reshaped(1, frames, Self.latentDimension)

        h = convolve(h, prefix: "decoder.layers.0", kernel: 7)
        h = lstmBlock(h, prefix: "decoder.layers.1")

        // Each stage: activation, transposed-conv upsample, residual block.
        let stages = [(3, 4), (6, 7), (9, 10), (12, 13)]
        for (index, (upsample, residual)) in stages.enumerated() {
            h = elu(h)
            h = upsampleTransposed(h, prefix: "decoder.layers.\(upsample)",
                                   ratio: Self.upsamplingRatios[index])
            h = h + residualBlock(elu(h), prefix: "decoder.layers.\(residual)")
        }

        h = elu(h)
        h = convolve(h, prefix: "decoder.layers.15", kernel: 7)
        return h.reshaped(h.dim(1))
            .reshaped(1, h.dim(1))
    }

    // MARK: - Pieces

    private func residualBlock(_ x: MLXArray, prefix: String) -> MLXArray {
        // block.0 and block.2 are activations and carry no weights.
        var h = convolve(x, prefix: "\(prefix).block.1", kernel: 3)
        h = convolve(elu(h), prefix: "\(prefix).block.3", kernel: 1)
        return h
    }

    /// Reflect-pad by the amount the kernel consumes, then convolve unpadded.
    private func convolve(_ x: MLXArray, prefix: String, kernel: Int) -> MLXArray {
        var input = x
        if kernel > 1 {
            let total = kernel - 1
            let right = total / 2
            input = reflectPad(x, left: total - right, right: right)
        }
        let weight = w("\(prefix).conv.weight")
        let bias = w("\(prefix).conv.bias")
        return conv1d(input, weight, stride: 1, padding: 0) + bias
    }

    /// Transposed convolution, then trim the overhang the stride introduces.
    private func upsampleTransposed(_ x: MLXArray, prefix: String, ratio: Int) -> MLXArray {
        let kernel = ratio * 2
        let weight = w("\(prefix).conv.weight")
        let bias = w("\(prefix).conv.bias")
        var y = convTransposed1d(x, weight, stride: ratio, padding: 0) + bias
        let total = kernel - ratio
        let right = total / 2
        let left = total - right
        y = y[0..., left ..< (y.dim(1) - right), 0...]
        return y
    }

    private func lstmBlock(_ x: MLXArray, prefix: String) -> MLXArray {
        var h = x
        for layer in 0 ..< 2 {
            h = lstmLayer(h, prefix: prefix, layer: layer)
        }
        return h + x   // skip connection around both layers
    }

    private func lstmLayer(_ x: MLXArray, prefix: String, layer: Int) -> MLXArray {
        let inputWeight = w("\(prefix).lstm.weight_ih_l\(layer)").T
        let hiddenWeight = w("\(prefix).lstm.weight_hh_l\(layer)").T
        let bias = w("\(prefix).lstm.bias_ih_l\(layer)") + w("\(prefix).lstm.bias_hh_l\(layer)")
        let width = x.dim(2)
        let steps = x.dim(1)

        var hidden = MLXArray.zeros([1, width], dtype: .float32)
        var cell = MLXArray.zeros([1, width], dtype: .float32)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(steps)

        // Precompute the input half for every step; only the recurrent half
        // has to run sequentially.
        let projected = matmul(x.reshaped(steps, width), inputWeight) + bias

        for step in 0 ..< steps {
            let gates = projected[step ..< (step + 1), 0...] + matmul(hidden, hiddenWeight)
            let parts = gates.split(parts: 4, axis: -1)   // input, forget, cell, output
            let inputGate = sigmoid(parts[0])
            let forgetGate = sigmoid(parts[1])
            let candidate = tanh(parts[2])
            let outputGate = sigmoid(parts[3])
            cell = forgetGate * cell + inputGate * candidate
            hidden = outputGate * tanh(cell)
            outputs.append(hidden)
        }
        return concatenated(outputs, axis: 0).reshaped(1, steps, width)
    }

    /// PyTorch's `reflect` padding: the edge sample is not repeated, so the
    /// left pad walks indices p…1 and the right pad walks L-2…L-1-q.
    private func reflectPad(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        let length = x.dim(1)
        var indices: [Int32] = []
        indices.reserveCapacity(left + length + right)
        for i in stride(from: left, to: 0, by: -1) { indices.append(Int32(i)) }
        for i in 0 ..< length { indices.append(Int32(i)) }
        for i in 0 ..< right { indices.append(Int32(length - 2 - i)) }
        return take(x, MLXArray(indices), axis: 1)
    }

    private func elu(_ x: MLXArray) -> MLXArray {
        which(x .> 0, x, exp(minimum(x, 0)) - 1)
    }
}
