//  Falconsai's nsfw_image_detection (ViT-base, Apache 2.0), converted by
//  ports/imagegen/convert_safety.py.

import CoreGraphics
import CoreML
import Foundation

/// The Image tab's one filter: a picture that looks sexually explicit is
/// neither shown nor saved (App Store guideline 1.1.4).
///
/// It looks at the finished picture only. Prompts are never read or
/// filtered, so politics, public figures, satire and news all work.
///
/// A ViT classifier on the Neural Engine, 224 × 224, about 20 ms. Pictures
/// the pipelines make of swimwear, a shirtless runner or Michelangelo's
/// David scored at most 0.05, so the bar sits well above that.
struct SafetyChecker {
    static let file = "safety_checker.mlmodelc"
    private static let size = 224
    private static let threshold: Float = 0.7

    let directory: URL

    func isExplicit(_ image: CGImage) throws -> Bool {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: directory.appendingPathComponent(Self.file), configuration: configuration)
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pixels": try SDPipeline.multiArray(Self.pixels(image), shape: [1, 3, Self.size, Self.size]),
        ]))
        guard let probabilities = output.featureValue(for: "probabilities")?.multiArrayValue else {
            throw SDPipeline.Failure.noOutput("picture check")
        }
        return SDPipeline.floats(probabilities)[1] >= Self.threshold
    }

    /// The picture at 224 × 224, RGB in 0...1, channels first.
    private static func pixels(_ image: CGImage) throws -> [Float] {
        var rgba = [UInt8](repeating: 0, count: size * size * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: size, height: size, bitsPerComponent: 8,
                                          bytesPerRow: size * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.interpolationQuality = .medium
            context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
            return true
        }
        guard drawn else { throw RGBImage.Failure.unrenderable }
        let plane = size * size
        var planar = [Float](repeating: 0, count: 3 * plane)
        for i in 0 ..< plane {
            for channel in 0 ..< 3 { planar[channel * plane + i] = Float(rgba[i * 4 + channel]) / 255 }
        }
        return planar
    }
}
