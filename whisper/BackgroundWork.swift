import UIKit
import BackgroundTasks

/// Keeps work the user started running after they leave the app:
/// transcribing, downloading a model or a link, generating music.
///
/// On iOS 26 each piece of work is a continued-processing task. iOS shows
/// its progress in a system banner and keeps the app running for as long
/// as the progress moves — with the GPU too, where the device supports it
/// and the work asks for it. Before iOS 26 the app gets the usual short
/// background time and is suspended after it; the work then resumes when
/// the app is opened again.
///
/// This replaces a `BGProcessingTaskRequest`, which iOS runs later, at a
/// time of its choosing, and an audio session left active with nothing
/// playing: neither kept a transcription going once the app was left.
@MainActor
final class BackgroundWork {
    static let shared = BackgroundWork()

    /// Submitted identifiers are this plus a unique suffix; Info.plist's
    /// BGTaskSchedulerPermittedIdentifiers permits "euleryu.whisper.work.*".
    private static let prefix = "euleryu.whisper.work"

    /// One piece of work, from `begin` until `finish`.
    final class Job {
        fileprivate let id: String
        fileprivate let title: String
        fileprivate var subtitle: String
        fileprivate var fraction = 0.0
        fileprivate let wantsGPU: Bool
        fileprivate var task: AnyObject?
        fileprivate var legacy = UIBackgroundTaskIdentifier.invalid
        fileprivate var done = false
        /// Runs if iOS ends the work before it finishes: stop cleanly.
        var onExpire: (() -> Void)?

        fileprivate init(id: String, title: String, subtitle: String, wantsGPU: Bool) {
            self.id = id
            self.title = title
            self.subtitle = subtitle
            self.wantsGPU = wantsGPU
        }

        /// How far along, 0...1, and optionally what it is doing now.
        @MainActor func update(_ fraction: Double, _ subtitle: String? = nil) {
            BackgroundWork.shared.update(self, fraction: fraction, subtitle: subtitle)
        }

        @MainActor func finish(success: Bool = true) {
            BackgroundWork.shared.finish(self, success: success)
        }
    }

    private var jobs: [String: Job] = [:]
    private var registered = false

    private init() {}

    /// Call once at launch: follows the app on and off screen for `GPUGate`.
    func start() {
        let center = NotificationCenter.default
        // Resign-active comes first, before background, so GPU work stops at
        // its next step while the app can still finish what is in flight.
        center.addObserver(forName: UIApplication.willResignActiveNotification, object: nil, queue: .main) { _ in
            GPUGate.setOnScreen(false)
        }
        center.addObserver(forName: UIApplication.didBecomeActiveNotification, object: nil, queue: .main) { _ in
            GPUGate.setOnScreen(true)
        }
    }

    /// Whether iOS is showing the work's progress itself, so the app's own
    /// Live Activity would only repeat it.
    var systemShowsProgress: Bool { jobs.values.contains { $0.task != nil } }

    /// Starts a piece of work. `usesGPU` asks for background GPU time, which
    /// music generation and GPU transcription need.
    func begin(title: String, subtitle: String, usesGPU: Bool = false) -> Job {
        let job = Job(id: "\(Self.prefix).\(UUID().uuidString)", title: title, subtitle: subtitle, wantsGPU: usesGPU)
        jobs[job.id] = job
        if #available(iOS 26.0, *), submit(job) { return job }
        job.legacy = UIApplication.shared.beginBackgroundTask(withName: title) { [weak self, weak job] in
            // Background time is up; iOS suspends the app and the work
            // resumes when it is opened again.
            guard let self, let job else { return }
            self.endLegacy(job)
        }
        return job
    }

    fileprivate func update(_ job: Job, fraction: Double, subtitle: String?) {
        guard !job.done else { return }
        job.fraction = min(max(fraction, 0), 1)
        let newSubtitle = subtitle.flatMap { $0 == job.subtitle ? nil : $0 }
        if let newSubtitle { job.subtitle = newSubtitle }
        if #available(iOS 26.0, *), let task = job.task as? BGContinuedProcessingTask {
            task.progress.completedUnitCount = Int64(job.fraction * 1000)
            if newSubtitle != nil { task.updateTitle(job.title, subtitle: job.subtitle) }
        }
    }

    fileprivate func finish(_ job: Job, success: Bool) {
        guard !job.done else { return }
        job.done = true
        jobs[job.id] = nil
        if #available(iOS 26.0, *), let task = job.task as? BGContinuedProcessingTask {
            task.progress.completedUnitCount = task.progress.totalUnitCount
            task.setTaskCompleted(success: success)
        }
        job.task = nil
        Self.gpuGranted.remove(job.id)
        endLegacy(job)
        refreshGate()
    }

    private func endLegacy(_ job: Job) {
        guard job.legacy != .invalid else { return }
        UIApplication.shared.endBackgroundTask(job.legacy)
        job.legacy = .invalid
    }

    /// GPU work may run off screen only while a job holding background GPU
    /// time is running.
    private func refreshGate() {
        GPUGate.setBackgroundGPU(jobs.values.contains { $0.task != nil && $0.wantsGPU && Self.gpuGranted.contains($0.id) })
    }

    /// Jobs whose request carried the GPU resource and was accepted.
    private static var gpuGranted: Set<String> = []

    // MARK: - iOS 26 continued processing

    @available(iOS 26.0, *)
    private func submit(_ job: Job) -> Bool {
        registerIfNeeded()
        // Work that needs the GPU but can't have it in the background would
        // sit at `GPUGate` making no progress, and iOS ends stalled tasks. It
        // takes the short background task instead, pauses, and carries on
        // when the app is opened again.
        let gpuSupported = BGTaskScheduler.supportedResources.contains(.gpu)
        guard !job.wantsGPU || gpuSupported else { return false }
        let request = BGContinuedProcessingTaskRequest(identifier: job.id, title: job.title, subtitle: job.subtitle)
        // Fail rather than queue: queued work would sit suspended in the
        // background without anyone knowing. The short background task is
        // the fallback.
        request.strategy = .fail
        if job.wantsGPU { request.requiredResources = .gpu }
        do {
            try BGTaskScheduler.shared.submit(request)
            if job.wantsGPU { Self.gpuGranted.insert(job.id) }
            return true
        } catch {
            return false    // busy, or the GPU entitlement was refused
        }
    }

    @available(iOS 26.0, *)
    private func registerIfNeeded() {
        guard !registered else { return }
        registered = true
        BGTaskScheduler.shared.register(forTaskWithIdentifier: Self.prefix + ".*", using: nil) { task in
            guard let task = task as? BGContinuedProcessingTask else {
                task.setTaskCompleted(success: false)
                return
            }
            Task { @MainActor in BackgroundWork.shared.attach(task) }
        }
    }

    @available(iOS 26.0, *)
    private func attach(_ task: BGContinuedProcessingTask) {
        guard let job = jobs[task.identifier], !job.done else {
            task.setTaskCompleted(success: true)
            return
        }
        job.task = task
        task.progress.totalUnitCount = 1000
        task.progress.completedUnitCount = Int64(job.fraction * 1000)
        task.updateTitle(job.title, subtitle: job.subtitle)
        task.expirationHandler = { [weak job] in
            // iOS needs the resources back. Stop the work and say so.
            Task { @MainActor in
                guard let job, !job.done else { return }
                job.onExpire?()
                BackgroundWork.shared.finish(job, success: false)
            }
        }
        refreshGate()
    }
}

/// Holds GPU work while the app is off screen without background GPU time.
///
/// Metal refuses command buffers from an app in the background, and MLX
/// treats the refusal as fatal — generating music and leaving the app ended
/// the app. Work that uses the GPU calls `waitUntilOpen()` between steps;
/// off screen it waits there, using nothing, until the app is back (or a
/// job with background GPU time is running).
enum GPUGate {
    private static let condition = NSCondition()
    nonisolated(unsafe) private static var onScreen = true
    nonisolated(unsafe) private static var backgroundGPU = false

    static func setOnScreen(_ value: Bool) {
        condition.lock(); onScreen = value; condition.broadcast(); condition.unlock()
    }

    static func setBackgroundGPU(_ value: Bool) {
        condition.lock(); backgroundGPU = value; condition.broadcast(); condition.unlock()
    }

    /// Whether GPU work may run now.
    static var isOpen: Bool {
        condition.lock(); defer { condition.unlock() }
        return onScreen || backgroundGPU
    }

    /// Blocks the calling thread until GPU work may run. Never blocks the
    /// main thread, whose notifications are what reopen the gate.
    static func waitUntilOpen() {
        guard !Thread.isMainThread else { return }
        condition.lock()
        while !onScreen && !backgroundGPU { condition.wait() }
        condition.unlock()
    }
}
