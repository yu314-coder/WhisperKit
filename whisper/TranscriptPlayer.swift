import Foundation
import AVFoundation
import Combine

/// Plays back the audio file attached to a SavedTranscript and publishes
/// the current playhead time so SwiftUI views can highlight the active segment.
@MainActor
final class TranscriptPlayer: ObservableObject {
    @Published private(set) var isPlaying: Bool = false
    @Published private(set) var currentTime: Double = 0
    @Published private(set) var duration: Double = 0

    private var player: AVAudioPlayer?
    private var timer: Timer?

    func load(audioURL: URL) {
        stop()
        do {
            // Configure session for playback (not record), so we get to use the loudspeaker.
            let session = AVAudioSession.sharedInstance()
            try? session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try? session.setActive(true)

            let p = try AVAudioPlayer(contentsOf: audioURL)
            p.prepareToPlay()
            duration = p.duration
            player = p
            currentTime = 0
        } catch {
            print("TranscriptPlayer load error: \(error)")
            player = nil
            duration = 0
        }
    }

    func togglePlayPause() {
        guard let p = player else { return }
        if p.isPlaying {
            pause()
        } else {
            play()
        }
    }

    func play() {
        guard let p = player else { return }
        p.play()
        isPlaying = true
        startTimer()
    }

    func pause() {
        guard let p = player else { return }
        p.pause()
        isPlaying = false
        stopTimer()
    }

    func stop() {
        player?.stop()
        player = nil
        isPlaying = false
        currentTime = 0
        stopTimer()
    }

    /// Seek the audio playback to a specific time and begin playing.
    func seek(to seconds: Double) {
        guard let p = player else { return }
        p.currentTime = max(0, min(seconds, p.duration))
        currentTime = p.currentTime
        if !p.isPlaying {
            play()
        }
    }

    private func startTimer() {
        timer?.invalidate()
        // 10 Hz is indistinguishable from 20 Hz for a scrubber and a text
        // highlight, at half the publish traffic. Scheduled in .common mode so
        // the playhead keeps moving while the user scrolls the segment list —
        // the default mode stalls during touch tracking.
        let t = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            // Hop rather than assert. MainActor.assumeIsolated traps if the
            // assumption is ever wrong, which turns a timing quirk into a
            // crash; nothing here is hot enough to need the saved hop.
            Task { @MainActor in
                guard let self, let p = self.player else { return }
                // Only publish when the value actually moved, so a paused or
                // stalled player stops waking every observing view.
                if abs(p.currentTime - self.currentTime) > 0.001 {
                    self.currentTime = p.currentTime
                }
                if !p.isPlaying {
                    self.isPlaying = false
                    self.stopTimer()
                }
            }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    deinit {
        // `timer` is main-actor isolated and deinit is not, so it cannot be
        // touched here. The timer holds only a weak self and stops itself once
        // playback ends, and stop() invalidates it on every normal path.
    }
}
