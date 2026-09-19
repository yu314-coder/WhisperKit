import SwiftUI
import Darwin

/// Live memory while generating.
///
/// Reports `phys_footprint`, not resident size. Resident size on a device
/// counts shared framework pages the app never allocated and reads far higher
/// than what iOS actually holds against it — measuring it once sent an
/// investigation in this app chasing a gigabyte that did not exist. Footprint
/// is the number that decides whether a model survives.
///
/// Headroom comes from `os_proc_available_memory`, which is how much more this
/// process may allocate before being killed — the quantity that matters when a
/// 3 GB model runs on an 8 GB device.
@Observable
@MainActor
final class MusicMemoryMonitor {
    private(set) var footprintMB: Double = 0
    private(set) var availableMB: Double = 0
    private(set) var peakMB: Double = 0
    private(set) var history: [Double] = []

    private var timer: Timer?

    var physicalMB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / 1_048_576
    }

    func start() {
        stop()
        peakMB = 0
        history.removeAll()
        sample()
        let timer = Timer(timeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.sample() }
        }
        // .common so samples keep arriving while the user scrolls or holds a
        // control; the default mode pauses during touch tracking.
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    // No deinit: `timer` is main-actor isolated and deinit is not, which
    // Swift rejects. The timer captures self weakly and `stop()` runs when the
    // tab disappears or work ends, so nothing is left running.

    private func sample() {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return }

        footprintMB = Double(info.phys_footprint) / 1_048_576
        availableMB = Double(os_proc_available_memory()) / 1_048_576
        peakMB = max(peakMB, footprintMB)

        history.append(footprintMB)
        if history.count > 90 { history.removeFirst() }
    }
}

/// A compact trace of footprint against the headroom left.
struct MusicMemoryGauge: View {
    let monitor: MusicMemoryMonitor

    /// Whether iOS told us how much more this process may allocate.
    /// `os_proc_available_memory` reports 0 on the Simulator, where it means
    /// nothing — showing that as "0 MB headroom" in red would be alarming and
    /// false.
    private var hasHeadroom: Bool { monitor.availableMB > 0 }

    /// The ceiling the trace is drawn against: what the app holds plus what it
    /// may still take. Without a headroom figure, fall back to the device's
    /// physical memory so the trace still has a sensible scale.
    private var ceilingMB: Double {
        hasHeadroom ? max(monitor.footprintMB + monitor.availableMB, 1)
                    : max(monitor.physicalMB, 1)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                StudioLabel(text: "Memory")
                Spacer()
                Text(String(format: "%.0f MB", monitor.footprintMB))
                    .font(Studio.mono(11, weight: .semibold))
                    .foregroundColor(tint)
            }

            GeometryReader { geometry in
                ZStack(alignment: .bottomLeading) {
                    trace(in: geometry.size)
                }
            }
            .frame(height: 34)

            HStack(spacing: 10) {
                Text(String(format: "peak %.0f MB", monitor.peakMB))
                Spacer()
                if hasHeadroom {
                    Text(String(format: "%.0f MB headroom", monitor.availableMB))
                        .foregroundColor(isTight ? Studio.hot : Studio.mute)
                } else {
                    Text("headroom unavailable")
                        .foregroundColor(Studio.mute)
                }
            }
            .font(Studio.mono(9))
            .foregroundColor(Studio.mute)
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Studio.sunk))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
            .strokeBorder(Studio.rule, lineWidth: 0.5))
    }

    /// Thin headroom is the state that precedes being killed. Unknown
    /// headroom is not thin headroom, so it is not flagged.
    private var isTight: Bool { hasHeadroom && monitor.availableMB < 400 }

    private var tint: Color { isTight ? Studio.hot : Studio.accent }

    @ViewBuilder
    private func trace(in size: CGSize) -> some View {
        let samples = monitor.history
        if samples.count < 2 {
            Path { path in
                path.move(to: CGPoint(x: 0, y: size.height - 1))
                path.addLine(to: CGPoint(x: size.width, y: size.height - 1))
            }
            .stroke(Studio.rule, style: StrokeStyle(lineWidth: 1, dash: [2, 3]))
        } else {
            let ceiling = ceilingMB
            let step = size.width / CGFloat(samples.count - 1)
            let points = samples.enumerated().map { index, value in
                CGPoint(x: CGFloat(index) * step,
                        y: size.height - CGFloat(min(value / ceiling, 1)) * (size.height - 2))
            }
            ZStack {
                Path { path in
                    path.move(to: CGPoint(x: points[0].x, y: size.height))
                    points.forEach { path.addLine(to: $0) }
                    path.addLine(to: CGPoint(x: points[points.count - 1].x, y: size.height))
                    path.closeSubpath()
                }
                .fill(LinearGradient(colors: [tint.opacity(0.24), tint.opacity(0.02)],
                                     startPoint: .top, endPoint: .bottom))
                Path { path in
                    path.move(to: points[0])
                    points.dropFirst().forEach { path.addLine(to: $0) }
                }
                .stroke(tint, style: StrokeStyle(lineWidth: 1.4, lineCap: .round, lineJoin: .round))
            }
        }
    }
}
