//  Stable Audio 3 diffusion transformer, covering the Small and Medium
//  variants. Derived from the DiT in StableAudio3-IOS (MIT, Copyright (c)
//  2026 stableaudio3-ios contributors), generalized here for Medium, whose
//  differential attention that project does not implement.

import Foundation
import MLX

/// Shape differences between the published Stable Audio 3 DiT variants.
///
/// Medium is not simply a wider Small. Its attention projections are sized for
/// five-way and two/three-way splits rather than three and one/two, which is
/// differential attention: a second set of queries and keys whose attention
/// output is subtracted from the first. The tensor *names* are identical across
/// both variants and there is no extra lambda parameter, so the subtraction is
/// unweighted — the same form the SAME-S decoder already uses.
struct SA3DiTConfig {
    let embedDimension: Int
    let layerCount: Int
    let feedForwardInner: Int
    let differential: Bool

    var headDimension: Int { 64 }
    var headCount: Int { embedDimension / headDimension }

    static let smallMusic = SA3DiTConfig(
        embedDimension: 1024, layerCount: 20, feedForwardInner: 4096, differential: false)
    static let medium = SA3DiTConfig(
        embedDimension: 1536, layerCount: 24, feedForwardInner: 6144, differential: true)
}

/// Which of the five self-attention projections is which, and the same for the
/// three packed into `to_kv`. The published weights carry no metadata saying,
/// so the order is determined empirically.
/// Determined by sweeping every plausible packing and scoring the result on a
/// prompt that demands silence between hits. Only this one can render silence:
/// it scored a frame-energy ratio of 17,124 against ~2 for each alternative,
/// which is the difference between drum hits and an undifferentiated wash.
enum SA3AttentionOrder {
    /// `to_qkv` packs q, k, v, qDiff, kDiff in order.
    nonisolated(unsafe) static var selfOrder = [0, 1, 2, 3, 4]
    /// `to_kv` packs k, kDiff, v — the diff key sits *before* the value.
    nonisolated(unsafe) static var crossKV = [0, 2, 1]
    nonisolated(unsafe) static var subtract = true
}

struct SA3DiT {
    static let ioChannels = 256
    static let ropeDimensions = 32
    static let localAddConditionDimension = 257
    static let memoryTokenCount = 64
    static let timestepFeatureDimension = 256

    let weights: [String: MLXArray]
    let latentLength: Int
    let config: SA3DiTConfig

    func callAsFunction(_ x: MLXArray, timestep: MLXArray, crossAttention: MLXArray, globalCondition: MLXArray) -> MLXArray {
        let batch = x.dim(0)

        var context = linear(crossAttention, weight: weights["to_cond_embed.0.weight"]!)
        context = silu(context)
        context = linear(context, weight: weights["to_cond_embed.2.weight"]!)

        var global = linear(globalCondition, weight: weights["to_global_embed.0.weight"]!)
        global = silu(global)
        let globalPre = linear(global, weight: weights["to_global_embed.2.weight"]!)

        var timeFeatures = timestepFeatures(timestep)
        timeFeatures = linear(timeFeatures, weight: weights["to_timestep_embed.0.weight"]!, bias: weights["to_timestep_embed.0.bias"]!)
        timeFeatures = silu(timeFeatures)
        let timeEmbed = linear(timeFeatures, weight: weights["to_timestep_embed.2.weight"]!, bias: weights["to_timestep_embed.2.bias"]!)
        let globalEmbed = globalPre + timeEmbed

        var h = x.transposed(0, 2, 1)
        h = conv1d(h, weights["preprocess_conv.weight"]!) + h
        h = continuousTransformer(h, context: context, globalEmbed: globalEmbed, batch: batch)
        let out = conv1d(h, weights["postprocess_conv.weight"]!) + h
        return out.transposed(0, 2, 1)
    }

    private func continuousTransformer(_ x: MLXArray, context: MLXArray, globalEmbed: MLXArray, batch: Int) -> MLXArray {
        var h = linear(x, weight: weights["transformer.project_in.weight"]!)
        let memory = weights["transformer.memory_tokens"]!.asType(h.dtype).expandedDimensions(axis: 0)
        let memoryBatched = broadcast(memory, to: [batch, Self.memoryTokenCount, config.embedDimension])
        h = concatenated([memoryBatched, h], axis: 1)

        var global = linear(globalEmbed, weight: weights["transformer.global_cond_embedder.0.weight"]!, bias: weights["transformer.global_cond_embedder.0.bias"]!)
        global = silu(global)
        global = linear(global, weight: weights["transformer.global_cond_embedder.2.weight"]!, bias: weights["transformer.global_cond_embedder.2.bias"]!)

        let localZeros = MLXArray.zeros([batch, latentLength, Self.localAddConditionDimension], dtype: h.dtype)
        for index in 0 ..< config.layerCount {
            let prefix = "transformer.layers.\(index)"
            var local = linear(localZeros, weight: weights["\(prefix).to_local_embed.seq.0.weight"]!, bias: weights["\(prefix).to_local_embed.seq.0.bias"]!)
            local = silu(local)
            local = linear(local, weight: weights["\(prefix).to_local_embed.seq.2.weight"]!, bias: weights["\(prefix).to_local_embed.seq.2.bias"]!)
            let localPadding = MLXArray.zeros([batch, Self.memoryTokenCount, config.embedDimension], dtype: local.dtype)
            let localPadded = concatenated([localPadding, local], axis: 1)
            h = transformerBlock(prefix: prefix, x: h, context: context, globalCondition: global, localEmbedded: localPadded)
            eval(h)
        }

        h = h[0..., Self.memoryTokenCount..., 0...]
        return linear(h, weight: weights["transformer.project_out.weight"]!)
    }

    private func transformerBlock(prefix: String, x: MLXArray, context: MLXArray, globalCondition: MLXArray, localEmbedded: MLXArray) -> MLXArray {
        let scaleShiftGate = (weights["\(prefix).to_scale_shift_gate"]! + globalCondition).expandedDimensions(axis: 1)
        let split = scaleShiftGate.split(parts: 6, axis: -1)

        var h = rmsNorm(x, weight: weights["\(prefix).pre_norm.weight"]!, eps: 1e-5)
        h = h * (1.0 + split[0]) + split[1]
        h = selfAttention(prefix: "\(prefix).self_attn", x: h)
        h = h * sigmoid(1.0 - split[2])
        var out = x + h

        out = out + crossAttention(
            prefix: "\(prefix).cross_attn",
            x: rmsNorm(out, weight: weights["\(prefix).cross_attend_norm.weight"]!, eps: 1e-5),
            context: context)
        out = out + localEmbedded

        h = rmsNorm(out, weight: weights["\(prefix).ff_norm.weight"]!, eps: 1e-5)
        h = h * (1.0 + split[3]) + split[4]
        h = feedForward(prefix: "\(prefix).ff.ff", x: h)
        h = h * sigmoid(1.0 - split[5])
        return out + h
    }

    private func selfAttention(prefix: String, x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let qkv = linear(x, weight: weights["\(prefix).to_qkv.weight"]!)
        let parts = qkv.split(parts: config.differential ? 5 : 3, axis: -1)

        let qNorm = weights["\(prefix).q_norm.weight"]!
        let kNorm = weights["\(prefix).k_norm.weight"]!
        let o = config.differential ? SA3AttentionOrder.selfOrder : [0, 1, 2, 3, 4]
        let q = prepare(parts[o[0]], batch: batch, length: length, norm: qNorm, rope: true)
        let k = prepare(parts[o[1]], batch: batch, length: length, norm: kNorm, rope: true)
        let v = toHeads(parts[o[2]], batch: batch, length: length)

        var attended = scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: pow(Float(config.headDimension), -0.5), mask: nil)

        if config.differential {
            // Same norms and rotary embedding as the primary pair; the second
            // attention map is subtracted rather than blended, matching the
            // decoder's formulation. No lambda tensor exists to weight it.
            let qDiff = prepare(parts[o[3]], batch: batch, length: length, norm: qNorm, rope: true)
            let kDiff = prepare(parts[o[4]], batch: batch, length: length, norm: kNorm, rope: true)
            let diff = scaledDotProductAttention(
                queries: qDiff, keys: kDiff, values: v,
                scale: pow(Float(config.headDimension), -0.5), mask: nil)
            attended = SA3AttentionOrder.subtract ? attended - diff : (attended + diff) * 0.5
        }

        let out = attended.transposed(0, 2, 1, 3).reshaped(batch, length, config.embedDimension)
        return linear(out, weight: weights["\(prefix).to_out.weight"]!)
    }

    private func crossAttention(prefix: String, x: MLXArray, context: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let xLength = x.dim(1)
        let contextLength = context.dim(1)

        let qProjected = linear(x, weight: weights["\(prefix).to_q.weight"]!)
        let kvProjected = linear(context, weight: weights["\(prefix).to_kv.weight"]!)
        let qParts = qProjected.split(parts: config.differential ? 2 : 1, axis: -1)
        let kvParts = kvProjected.split(parts: config.differential ? 3 : 2, axis: -1)

        let qNorm = weights["\(prefix).q_norm.weight"]!
        let kNorm = weights["\(prefix).k_norm.weight"]!
        let q = prepare(qParts[0], batch: batch, length: xLength, norm: qNorm, rope: false)
        let c = config.differential ? SA3AttentionOrder.crossKV : [0, 1, 2]
        let k = prepare(kvParts[c[0]], batch: batch, length: contextLength, norm: kNorm, rope: false)
        let v = toHeads(kvParts[c[1]], batch: batch, length: contextLength)

        var attended = scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: pow(Float(config.headDimension), -0.5), mask: nil)

        if config.differential {
            let qDiff = prepare(qParts[1], batch: batch, length: xLength, norm: qNorm, rope: false)
            let kDiff = prepare(kvParts[c[2]], batch: batch, length: contextLength, norm: kNorm, rope: false)
            let diff = scaledDotProductAttention(
                queries: qDiff, keys: kDiff, values: v,
                scale: pow(Float(config.headDimension), -0.5), mask: nil)
            attended = SA3AttentionOrder.subtract ? attended - diff : (attended + diff) * 0.5
        }

        let out = attended.transposed(0, 2, 1, 3).reshaped(batch, xLength, config.embedDimension)
        return linear(out, weight: weights["\(prefix).to_out.weight"]!)
    }

    /// Heads, RMS norm, and rotary embedding only where the reference applies
    /// it. Cross-attention deliberately skips RoPE: its keys index text tokens,
    /// which carry no audio timeline to rotate against. Applying it there
    /// changed the output of a fixed seed, which is how the omission surfaced.
    private func prepare(_ x: MLXArray, batch: Int, length: Int, norm: MLXArray, rope: Bool) -> MLXArray {
        var h = toHeads(x, batch: batch, length: length)
        h = rmsNorm(h, weight: norm, eps: 1e-6)
        guard rope else { return h }
        return RoPE(h, dimensions: Self.ropeDimensions, traditional: false, base: 10_000, scale: 1, offset: 0)
    }

    private func feedForward(prefix: String, x: MLXArray) -> MLXArray {
        let projected = linear(x, weight: weights["\(prefix).0.proj.weight"]!, bias: weights["\(prefix).0.proj.bias"]!)
        let split = projected.split(parts: 2, axis: -1)
        let activated = split[0] * silu(split[1])
        return linear(activated, weight: weights["\(prefix).2.weight"]!, bias: weights["\(prefix).2.bias"]!)
    }

    private func toHeads(_ x: MLXArray, batch: Int, length: Int) -> MLXArray {
        x.reshaped(batch, length, config.headCount, config.headDimension).transposed(0, 2, 1, 3)
    }

    private func timestepFeatures(_ timestep: MLXArray) -> MLXArray {
        let half = Self.timestepFeatureDimension / 2
        let ramp = MLX.linspace(Float(0), Float(1), count: half)
        let frequencies = exp(ramp * (log(Float(10_000)) - log(Float(0.5))) + log(Float(0.5))) * (2.0 * Float.pi)
        let args = timestep.asType(.float32).expandedDimensions(axis: 1) * frequencies
        return concatenated([cos(args), sin(args)], axis: -1)
    }
}
