import SwiftUI
import Metal

// MARK: - System Monitor
//
// Sampling and its 60-sample history used to live as @State on ContentView, so
// every 1 Hz tick invalidated the entire screen — including the transcript
// canvas — while a transcription was streaming into it. As an @Observable
// model consumed by a dedicated view, a tick now only redraws the meter.

struct SystemStatsSnapshot: Identifiable {
    let id = UUID()
    let timestamp: Date
    let cpuUsage: Double
    let memoryUsage: Double
    let memoryPercent: Double
    let gpuUsage: Double
    let networkIO: Double
}

@Observable
final class SystemMonitor {
    /// Rolling 60-sample window driving the sparklines.
    private(set) var history: [SystemStatsSnapshot] = []

    @ObservationIgnored private var lastHostCPUTicks: (user: UInt64, system: UInt64, nice: UInt64, idle: UInt64) = (0, 0, 0, 0)
    @ObservationIgnored private var lastTaskTimeSeconds: Double = 0
    @ObservationIgnored private var lastSampleAt: Date? = nil
    @ObservationIgnored private var monitoringTimer: Timer?
    @ObservationIgnored private(set) var metalDevice: MTLDevice?

    /// Called on the main actor after each sample lands.
    @ObservationIgnored var onSample: (() -> Void)?


// MARK: - System Monitoring

func initializeGPUMonitoring() {
    metalDevice = MTLCreateSystemDefaultDevice()
}


/// Instantaneous host CPU usage by computing the delta between two
/// successive `PROCESSOR_CPU_LOAD_INFO` reads. The previous reading is
/// cached in `lastHostCPUTicks`; the first call returns 0 and seeds the
/// baseline so the next call is correct.
func getActualCPUUsage() -> Double {
    var cpuInfo: processor_info_array_t!
    var numCpuInfo: mach_msg_type_number_t = 0
    var numCPUs: uint = 0

    let result = host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &numCPUs, &cpuInfo, &numCpuInfo)
    guard result == KERN_SUCCESS else { return 0 }

    // Aggregate ticks across all logical CPUs
    var user: UInt64 = 0, system: UInt64 = 0, nice: UInt64 = 0, idle: UInt64 = 0
    for i in 0..<Int(numCPUs) {
        let cpuLoad = cpuInfo.advanced(by: Int(CPU_STATE_MAX) * i)
        user   += UInt64(cpuLoad[Int(CPU_STATE_USER)])
        system += UInt64(cpuLoad[Int(CPU_STATE_SYSTEM)])
        nice   += UInt64(cpuLoad[Int(CPU_STATE_NICE)])
        idle   += UInt64(cpuLoad[Int(CPU_STATE_IDLE)])
    }

    // Free the kernel-allocated buffer (otherwise leaks each call)
    let vmAddr = vm_address_t(bitPattern: cpuInfo)
    let size   = vm_size_t(Int(numCpuInfo) * MemoryLayout<integer_t>.size)
    vm_deallocate(mach_task_self_, vmAddr, size)

    let prev = lastHostCPUTicks
    let dUser   = Int64(user)   - Int64(prev.user)
    let dSystem = Int64(system) - Int64(prev.system)
    let dNice   = Int64(nice)   - Int64(prev.nice)
    let dIdle   = Int64(idle)   - Int64(prev.idle)
    let dBusy   = dUser + dSystem + dNice
    let dTotal  = dBusy + dIdle

    // Persist for next call
    DispatchQueue.main.async {
        self.lastHostCPUTicks = (user, system, nice, idle)
    }

    // First call (no baseline yet) — return 0; the next call will report a real %.
    guard prev.user > 0 || prev.system > 0 else { return 0 }
    guard dTotal > 0 else { return 0 }

    return min(100.0, max(0.0, Double(dBusy) / Double(dTotal) * 100.0))
}


func getActualMemoryUsage() -> (usedMB: Double, totalMB: Double, percent: Double) {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    
    if kerr == KERN_SUCCESS {
        let usedMemoryMB = Double(info.resident_size) / 1_048_576
        let totalMemoryMB = Double(ProcessInfo.processInfo.physicalMemory) / 1_048_576
        let percent = (usedMemoryMB / totalMemoryMB) * 100.0
        
        return (usedMemoryMB, totalMemoryMB, percent)
    }
    
    return (0, 0, 0)
}


/// Real "app activity" % — total CPU+ANE time this process consumed since
/// the previous sample, divided by wall-clock elapsed and normalized to the
/// number of logical cores. Spikes hard when WhisperKit hammers the
/// Neural Engine because ANE work shows up as task thread time too.
func getGPUUsage() -> Double {
    var infoData = task_thread_times_info()
    var count = mach_msg_type_number_t(MemoryLayout<task_thread_times_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kerr = withUnsafeMutablePointer(to: &infoData) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_THREAD_TIMES_INFO), $0, &count)
        }
    }
    guard kerr == KERN_SUCCESS else { return 0 }

    // total user + system time for live threads, in seconds
    let userSecs = Double(infoData.user_time.seconds) + Double(infoData.user_time.microseconds) / 1_000_000
    let sysSecs  = Double(infoData.system_time.seconds) + Double(infoData.system_time.microseconds) / 1_000_000
    let currentTaskSecs = userSecs + sysSecs

    let now = Date()
    let prevSecs = lastTaskTimeSeconds
    let prevAt   = lastSampleAt

    // Update baseline for next call
    DispatchQueue.main.async {
        self.lastTaskTimeSeconds = currentTaskSecs
        self.lastSampleAt = now
    }

    guard let prevAt, prevSecs > 0 else { return 0 }
    let wallSecs = now.timeIntervalSince(prevAt)
    guard wallSecs > 0.05 else { return 0 } // ignore samples too close together

    let deltaTask = max(0, currentTaskSecs - prevSecs)
    // Normalize per logical core so an app pinning all cores reads ~100%.
    let cores = max(1.0, Double(ProcessInfo.processInfo.activeProcessorCount))
    let activity = (deltaTask / (wallSecs * cores)) * 100.0
    return min(100.0, max(0.0, activity))
}


func updateSystemStats() {
    let cpuUsage = getActualCPUUsage()
    let (memoryMB, _, memoryPercent) = getActualMemoryUsage()
    let gpuUsage = getGPUUsage()

    let snapshot = SystemStatsSnapshot(
        timestamp: Date(),
        cpuUsage: cpuUsage,
        memoryUsage: memoryMB,
        memoryPercent: memoryPercent,
        gpuUsage: gpuUsage,
        networkIO: 0.0
    )

    // Observation must be mutated on the main actor; the sampling itself is
    // cheap enough to stay inline.
    Task { @MainActor in
        self.history.append(snapshot)
        if self.history.count > 60 {
            self.history.removeFirst()
        }
        self.onSample?()
    }
}


func startSystemMonitoring() {
    stopSystemMonitoring()
    history.removeAll()

    // Push one snapshot immediately so the chart isn't blank for the
    // first second.
    updateSystemStats()

    // Attach to .common mode so the timer keeps firing while the user is
    // scrolling / pressing buttons (the default mode pauses during touch
    // tracking, which makes the chart look frozen on iPhone).
    let timer = Timer(timeInterval: 1.0, repeats: true) { _ in
        // Capture nothing problematic — read @State through the wrapper.
        self.updateSystemStats()
    }
    RunLoop.main.add(timer, forMode: .common)
    monitoringTimer = timer
}


func stopSystemMonitoring() {
    monitoringTimer?.invalidate()
    monitoringTimer = nil
}

    deinit {
        monitoringTimer?.invalidate()
    }
}

// MARK: - Performance meter

/// A separate View so that a 1 Hz sample only redraws this card.
struct PaperPerfMonitor: View {
    let monitor: SystemMonitor

    var body: some View {
        paperPerfMonitor
    }


var paperPerfMonitor: some View {
    let latest = monitor.history.last
    let elapsed = max(0, monitor.history.count - 1)
    return VStack(alignment: .leading, spacing: 12) {
        HStack(spacing: 6) {
            Rectangle()
                .fill(ContentView.paperInk.opacity(0.20))
                .frame(width: 18, height: 1)
            Text("Live performance")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .tracking(2)
                .foregroundColor(ContentView.paperMute)
            Spacer()
            Text("\(elapsed)s")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundColor(ContentView.paperMute.opacity(0.75))
        }

        paperPerfChannel(
            label: "Processor",
            accent: ContentView.perfCPU,
            value: latest?.cpuUsage ?? 0,
            samples: monitor.history.map { $0.cpuUsage },
            primaryUnit: "%",
            secondary: nil
        )

        Divider().background(ContentView.paperRule)

        paperPerfChannel(
            label: "Memory",
            accent: ContentView.perfMem,
            value: latest?.memoryPercent ?? 0,
            samples: monitor.history.map { $0.memoryPercent },
            primaryUnit: "%",
            secondary: String(format: "%.0f MB", latest?.memoryUsage ?? 0)
        )

        Divider().background(ContentView.paperRule)

        paperPerfChannel(
            label: "Neural Engine",
            accent: ContentView.perfNPU,
            value: latest?.gpuUsage ?? 0,
            samples: monitor.history.map { $0.gpuUsage },
            primaryUnit: "%",
            secondary: nil
        )
    }
    .padding(.horizontal, 16)
    .padding(.vertical, 14)
    .background(
        RoundedRectangle(cornerRadius: 14, style: .continuous)
            .fill(Color.white.opacity(0.62))
    )
    .overlay(
        RoundedRectangle(cornerRadius: 14, style: .continuous)
            .strokeBorder(ContentView.paperRule, lineWidth: 0.5)
    )
    .shadow(color: ContentView.paperInk.opacity(0.04), radius: 8, y: 2)
}


func paperPerfChannel(
    label: String,
    accent: Color,
    value: Double,
    samples: [Double],
    primaryUnit: String,
    secondary: String?
) -> some View {
    let peak = samples.max() ?? 0
    return VStack(alignment: .leading, spacing: 6) {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            // Channel dot
            Circle()
                .fill(accent)
                .frame(width: 7, height: 7)

            Text(label.uppercased())
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .tracking(1.5)
                .foregroundColor(ContentView.paperMute)

            Spacer()

            // Peak chip
            if samples.count > 1 {
                HStack(spacing: 3) {
                    Text("peak")
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundColor(ContentView.paperMute.opacity(0.75))
                    Text(String(format: "%.0f%@", peak, primaryUnit))
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundColor(accent.opacity(0.85))
                }
            }
        }

        HStack(alignment: .firstTextBaseline, spacing: 0) {
            // Hero value
            Text(String(format: "%.0f", value))
                .font(.system(size: 28, weight: .semibold, design: .serif))
                .foregroundColor(ContentView.paperInk)
                .monospacedDigit()
            Text(primaryUnit)
                .font(.system(size: 14, weight: .medium, design: .serif))
                .foregroundColor(ContentView.paperInk.opacity(0.55))
                .padding(.leading, 1)
            if let s = secondary {
                Text("· \(s)")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundColor(ContentView.paperMute.opacity(0.85))
                    .padding(.leading, 6)
            }
            Spacer()
        }

        // Sparkline with peak rule
        GeometryReader { geo in
            paperSparklineSmooth(samples: samples, accent: accent, in: geo.size)
        }
        .frame(height: 34)
    }
}

@ViewBuilder
func paperSparklineSmooth(samples: [Double], accent: Color, in size: CGSize) -> some View {
    let h = size.height
    let w = size.width
    if samples.count < 2 {
        // Empty baseline — dotted hairline so the chart has visual presence
        Path { p in
            p.move(to: CGPoint(x: 0, y: h - 1))
            p.addLine(to: CGPoint(x: w, y: h - 1))
        }
        .stroke(ContentView.paperRule, style: StrokeStyle(lineWidth: 1, dash: [2, 3]))
    } else {
        // Scale to a generous max so a 100% trace doesn't kiss the ceiling
        let maxVal = max(samples.max() ?? 100, 1) * 1.1
        let stepX = w / CGFloat(samples.count - 1)
        let points: [CGPoint] = samples.enumerated().map { (i, v) in
            let x = CGFloat(i) * stepX
            let y = h - CGFloat(v / maxVal) * (h - 2)
            return CGPoint(x: x, y: y)
        }
        ZStack {
            // Gradient fill below curve
            paperSmoothPath(points: points, closeAtY: h)
                .fill(
                    LinearGradient(
                        colors: [accent.opacity(0.22), accent.opacity(0.02)],
                        startPoint: .top, endPoint: .bottom
                    )
                )

            // Smooth curve stroke
            paperSmoothPath(points: points, closeAtY: nil)
                .stroke(accent, style: StrokeStyle(lineWidth: 1.4, lineCap: .round, lineJoin: .round))

            // Latest point indicator
            if let last = points.last {
                Circle()
                    .fill(accent)
                    .frame(width: 5, height: 5)
                    .position(last)
                Circle()
                    .stroke(accent.opacity(0.35), lineWidth: 4)
                    .frame(width: 12, height: 12)
                    .position(last)
            }
        }
    }
}

private func paperSmoothPath(points: [CGPoint], closeAtY: CGFloat?) -> Path {
    var path = Path()
    guard let first = points.first else { return path }

    if let baseY = closeAtY {
        path.move(to: CGPoint(x: first.x, y: baseY))
        path.addLine(to: first)
    } else {
        path.move(to: first)
    }

    for i in 1..<points.count {
        let p0 = points[i - 1]
        let p1 = points[i]
        let midX = (p0.x + p1.x) / 2
        let control1 = CGPoint(x: midX, y: p0.y)
        let control2 = CGPoint(x: midX, y: p1.y)
        path.addCurve(to: p1, control1: control1, control2: control2)
    }

    if let baseY = closeAtY, let last = points.last {
        path.addLine(to: CGPoint(x: last.x, y: baseY))
        path.closeSubpath()
    }
    return path
}
}
