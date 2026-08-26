import SwiftUI

/// Draws a peak envelope as a bar waveform, split at the playhead.
///
/// The envelope comes from `AudioConverter.peakEnvelope` — measured amplitude,
/// not decoration — so silences really are gaps and loud passages really are
/// tall. Rendered with `Canvas` rather than a stack of shapes: a detail view
/// draws 120+ bars and a library row redraws its own on every scroll tick, and
/// one draw call per view is the difference between smooth and not.
struct WaveformView: View {
    /// Normalised 0...1, one value per bucket.
    let buckets: [Float]
    /// 0...1 position of the playhead; nil draws every bar idle.
    var progress: Double? = nil
    var playedColor: Color = Studio.accent
    var idleColor: Color = Studio.idle
    var barSpacing: CGFloat = 1
    var minBarHeight: CGFloat = 1
    /// Called with a 0...1 position when the user taps or drags.
    var onScrub: ((Double) -> Void)? = nil

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                guard !buckets.isEmpty else { return }

                let count = CGFloat(buckets.count)
                let totalSpacing = barSpacing * (count - 1)
                let barWidth = max(0.5, (size.width - totalSpacing) / count)
                let played = progress.map { CGFloat(max(0, min(1, $0))) }

                for (i, value) in buckets.enumerated() {
                    let h = max(minBarHeight, CGFloat(value) * size.height)
                    let x = CGFloat(i) * (barWidth + barSpacing)
                    let rect = CGRect(
                        x: x,
                        y: (size.height - h) / 2,
                        width: barWidth,
                        height: h
                    )
                    // A bar counts as played once its midpoint is behind the head.
                    let mid = (CGFloat(i) + 0.5) / count
                    let color = played.map { mid <= $0 ? playedColor : idleColor } ?? idleColor
                    context.fill(
                        Path(roundedRect: rect, cornerRadius: min(1, barWidth / 2)),
                        with: .color(color)
                    )
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { g in
                        guard let onScrub, geo.size.width > 0 else { return }
                        onScrub(Double(max(0, min(1, g.location.x / geo.size.width))))
                    }
            )
            .allowsHitTesting(onScrub != nil)
        }
    }
}

/// A flat placeholder for transcripts saved before envelopes existed, or whose
/// audio has since gone missing. Deliberately not a fake waveform.
struct WaveformPlaceholder: View {
    var body: some View {
        GeometryReader { geo in
            Path { p in
                p.move(to: CGPoint(x: 0, y: geo.size.height / 2))
                p.addLine(to: CGPoint(x: geo.size.width, y: geo.size.height / 2))
            }
            .stroke(Studio.idle, style: StrokeStyle(lineWidth: 1, dash: [2, 3]))
        }
    }
}

extension SavedTranscript {
    /// Envelope for display, downsampled to at most `limit` bars so a narrow
    /// library row doesn't try to draw 120 sub-pixel slivers.
    func envelope(limit: Int) -> [Float]? {
        guard let waveform, !waveform.isEmpty else { return nil }
        guard waveform.count > limit, limit > 0 else { return waveform }
        let stride = Double(waveform.count) / Double(limit)
        return (0..<limit).map { i in
            let lo = Int(Double(i) * stride)
            let hi = min(waveform.count, max(lo + 1, Int(Double(i + 1) * stride)))
            return waveform[lo..<hi].max() ?? 0
        }
    }
}
