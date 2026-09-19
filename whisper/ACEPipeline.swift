//  Ported from ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5), MIT.
//  Verified against the PyTorch reference stage by stage; see ports/README.md.

import Foundation
import MLX
import MLXRandom

/// Text prompt to audio, using ACE-Step's turbo schedule.
///
/// This is the text-only path: no lyrics, no reference audio. That matters
/// because it removes three of the pipeline's parts — the 1.7B planner (its
/// hints are optional and the generate path passes none), the FSQ audio
/// tokenizer, and the detokenizer, which exist to condition on *supplied*
/// audio rather than to produce it. What remains is a text encoder, the
/// diffusion transformer, and the decoder.
///
/// Sampling is flow matching: the transformer predicts a velocity and the
/// sampler walks from noise toward data by Euler steps, solving directly for
/// the clean latent on the last one.
struct ACEPipeline {
    /// The turbo schedule for shift 3.0 at 8 steps, as published — these are
    /// exact values the model was distilled for, not a curve to re-derive.
    static let schedule: [Float] = [1.0, 0.9545454545454546, 0.9, 0.8333333333333334,
                                    0.75, 0.6428571428571429, 0.5, 0.3]
    static let framesPerSecond = 25
    static let latentChannels = 64

    let textEncoder: ACEQwen3
    let dit: ACEDiT
    let vae: ACEVAE
    /// (1, frames, 64) of silence, the bed the generation starts from.
    let silence: MLXArray
    /// encoder.text_projector.weight — 1024 to 2048, no bias.
    let textProjection: MLXArray

    /// - Parameter lyricIDs: tokens for lyrics to sing, or nil for an
    ///   instrumental. They run through the same text encoder as the prompt
    ///   and then a dedicated 8-layer lyric encoder.
    func generate(tokenIDs: MLXArray,
                  lyricIDs: MLXArray? = nil,
                  seconds: Double,
                  seed: UInt64 = 1234,
                  onStep: ((Int, Int) -> Void)? = nil) -> MLXArray {
        let frames = min(Int(seconds * Double(Self.framesPerSecond)), silence.dim(1))

        let text = textEncoder(inputIDs: tokenIDs)
        let lyricEmbeddings = lyricIDs.map { textEncoder(inputIDs: $0) }
        var packer = ACEConditioning(weights: dit.weights)
        packer.quantizationBits = dit.quantizationBits
        let (conditioning, _) = packer.encode(text: text,
                                              textProjection: textProjection,
                                              lyric: lyricEmbeddings,
                                              reference: nil)

        // Context is the silence bed plus an all-zero chunk mask: nothing is
        // being continued or covered, so every frame is free to be generated.
        let bed = silence[0..., 0 ..< frames, 0...].asType(.float32)
        let chunkMask = MLXArray.zeros([1, frames, Self.latentChannels], dtype: .float32)
        let context = concatenated([bed, chunkMask], axis: -1)

        var x = MLXRandom.normal([1, frames, Self.latentChannels], key: MLXRandom.key(seed))
        eval(x)

        for (index, t) in Self.schedule.enumerated() {
            let velocity = dit.forward(xt: x, context: context, encoder: conditioning, timestep: t)
            if index == Self.schedule.count - 1 {
                // Last step solves for the clean latent rather than stepping.
                x = x - velocity * t
            } else {
                let dt = t - Self.schedule[index + 1]
                x = x - velocity * dt
            }
            eval(x)
            onStep?(index + 1, Self.schedule.count)
        }

        // The decoder wants channels first.
        return vae.decode(latent: x.transposed(0, 2, 1))
    }
}

/// 48 kHz stereo WAV, which is what the Oobleck decoder produces.
enum ACEWAVWriter {
    static func write(_ audio: MLXArray, to url: URL, sampleRate: Int = 48000) throws {
        let samples = audio.dim(2)
        let left = audio[0, 0, 0...].asArray(Float.self)
        let right = audio.dim(1) > 1 ? audio[0, 1, 0...].asArray(Float.self) : left

        var data = Data()
        func string(_ s: String) { data.append(s.data(using: .ascii)!) }
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }

        string("RIFF"); u32(UInt32(36 + samples * 4)); string("WAVE")
        string("fmt "); u32(16); u16(1); u16(2)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 4)); u16(4); u16(16)
        string("data"); u32(UInt32(samples * 4))

        for index in 0 ..< samples {
            for channel in [left, right] {
                let clipped = max(-1, min(1, channel[index]))
                withUnsafeBytes(of: Int16(clipped * 32767).littleEndian) { data.append(contentsOf: $0) }
            }
        }
        try data.write(to: url, options: [.atomic])
    }
}
