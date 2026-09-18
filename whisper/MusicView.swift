import SwiftUI
import AVFoundation

/// The Music tab: a prompt, a model, and a generated clip.
///
/// The model list is deliberately honest about what cannot run yet. Three of
/// the five have no working Apple-silicon path at the time of writing, and
/// showing them with a Generate button that fails would waste the one thing
/// that actually settles these questions — a run on real hardware.
struct MusicView: View {
    @State private var prompt: String = ""
    @State private var selectedModel: MusicModel = .stableAudio3Small
    @State private var showModelPicker = false
    @State private var durationSeconds: Double = 10
    @State private var engine = MusicEngine()
    @State private var player: AVAudioPlayer?
    @State private var isPlaying = false

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Rectangle().fill(Studio.rule).frame(height: 1)
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    promptCard
                    lengthCard
                    if engine.isBusy || engine.lastResult != nil || engine.failureMessage != nil {
                        progressCard
                    }
                    statusCard
                }
                .padding(18)
            }
            generateBar
        }
        .background(Studio.bg.ignoresSafeArea())
        .sheet(isPresented: $showModelPicker) {
            MusicModelPicker(selected: $selectedModel)
                .presentationSizing(.page)
        }
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "music.note")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(Studio.accent)
                Text("music")
                    .font(Studio.mono(13, weight: .semibold))
                    .foregroundColor(Studio.ink)
            }
            Spacer()
            Button { showModelPicker = true } label: {
                HStack(spacing: 5) {
                    Circle()
                        .fill(selectedModel.isRunnable ? Studio.ok : Studio.mute.opacity(0.5))
                        .frame(width: 6, height: 6)
                    Text(selectedModel.displayName)
                        .font(Studio.mono(11))
                        .foregroundColor(Studio.ink.opacity(0.82))
                        .lineLimit(1)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .overlay(Capsule().strokeBorder(Studio.rule, lineWidth: 0.5))
            }
            .buttonStyle(PressableButtonStyle())
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }

    // MARK: - Cards

    private var promptCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            StudioLabel(text: "Prompt")
            ZStack(alignment: .topLeading) {
                if prompt.isEmpty {
                    Text("A slow lo-fi beat with warm bass and vinyl crackle")
                        .font(Studio.text(15))
                        .foregroundColor(Studio.mute.opacity(0.7))
                        .padding(.top, 8)
                        .padding(.horizontal, 12)
                        .allowsHitTesting(false)
                }
                TextEditor(text: $prompt)
                    .font(Studio.text(15))
                    .foregroundColor(Studio.ink)
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .frame(minHeight: 96)
            }
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Studio.sunk)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(Studio.rule, lineWidth: 0.5)
            )
        }
    }

    private var lengthCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                StudioLabel(text: "Length")
                Spacer()
                Text("\(Int(durationSeconds))s")
                    .font(Studio.mono(11, weight: .semibold))
                    .foregroundColor(Studio.accent)
            }
            Slider(value: $durationSeconds, in: 5...30, step: 1)
                .tint(Studio.accent)
        }
    }

    /// What the selected model can actually do on this device, stated plainly.
    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            StudioLabel(text: "Model")
            switch selectedModel.availability {
            case .ready:
                noteRow(icon: "checkmark.circle",
                        tint: Studio.ok,
                        text: "\(selectedModel.displayName) runs on this device. Weights are \(selectedModel.sizeLabel), downloaded once.")
            case .weightsPublished(let note):
                noteRow(icon: "exclamationmark.triangle", tint: Studio.hot, text: note)
            case .needsConversion(let note):
                noteRow(icon: "wrench.and.screwdriver", tint: Studio.mute, text: note)
            }
        }
    }

    /// Download and generation share one progress area: from the user's side
    /// it is a single wait, even though the first part only happens once.
    @ViewBuilder
    private var progressCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            switch engine.phase {
            case .downloading(let file, let completed, let total, let fraction, let received, let expected):
                HStack {
                    StudioLabel(text: "Downloading \(completed + 1) of \(total)")
                    Spacer()
                    // A number, not just a bar: a stalled download and a slow
                    // one look identical otherwise.
                    Text("\(Int(fraction * 100))%")
                        .font(Studio.mono(11, weight: .semibold))
                        .foregroundColor(Studio.accent)
                }
                ProgressView(value: fraction).tint(Studio.accent)
                HStack {
                    Text(file).font(Studio.mono(10)).foregroundColor(Studio.mute).lineLimit(1)
                    Spacer()
                    Text("\(received / 1_000_000) / \(expected / 1_000_000) MB")
                        .font(Studio.mono(10))
                        .foregroundColor(Studio.mute)
                }
            case .generating(let stage):
                StudioLabel(text: "Generating")
                ProgressView().tint(Studio.accent)
                Text(stage).font(Studio.mono(10)).foregroundColor(Studio.mute)
            case .failed(let message):
                StudioLabel(text: "Failed")
                Text(message)
                    .font(Studio.text(13))
                    .foregroundColor(Studio.hot)
                    .fixedSize(horizontal: false, vertical: true)
            case .idle:
                if let url = engine.lastResult {
                    HStack {
                        StudioLabel(text: "Result")
                        Spacer()
                        Text("\(engine.elapsedMilliseconds) ms")
                            .font(Studio.mono(10))
                            .foregroundColor(Studio.mute)
                    }
                    HStack(spacing: 12) {
                        Button { togglePlayback(url) } label: {
                            Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                                .font(.system(size: 38))
                                .foregroundColor(Studio.accent)
                        }
                        .buttonStyle(PressableButtonStyle())
                        ShareLink(item: url) {
                            Image(systemName: "square.and.arrow.up")
                                .font(.system(size: 17))
                                .foregroundColor(Studio.ink.opacity(0.75))
                        }
                        Spacer()
                    }
                }
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Studio.sunk))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
            .strokeBorder(Studio.rule, lineWidth: 0.5))
    }

    private func togglePlayback(_ url: URL) {
        if isPlaying { player?.pause(); isPlaying = false; return }
        if player?.url != url {
            try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try? AVAudioSession.sharedInstance().setActive(true)
            player = try? AVAudioPlayer(contentsOf: url)
        }
        player?.play()
        isPlaying = true
    }

    private func noteRow(icon: String, tint: Color, text: String) -> some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: icon)
                .font(.system(size: 13))
                .foregroundColor(tint)
                .padding(.top, 1)
            Text(text)
                .font(Studio.text(13))
                .foregroundColor(Studio.ink.opacity(0.75))
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }

    // MARK: - Generate

    private var canGenerate: Bool {
        engine.isBusy
            || (selectedModel.isRunnable && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    private var buttonTitle: String {
        if engine.isBusy { return "Cancel" }
        guard selectedModel.isRunnable else { return "\(selectedModel.displayName) is not runnable yet" }
        return "Generate"
    }

    private var generateBar: some View {
        VStack(spacing: 0) {
            Rectangle().fill(Studio.rule).frame(height: 1)
            Button {
                if engine.isBusy {
                    engine.cancel()
                } else {
                    isPlaying = false
                    player?.stop()
                    engine.generate(model: selectedModel, prompt: prompt, seconds: durationSeconds)
                }
            } label: {
                Text(buttonTitle)
                    .font(Studio.text(15, weight: .semibold))
                    .foregroundColor(canGenerate ? Studio.onAccent : Studio.mute)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 15)
                    .background(
                        RoundedRectangle(cornerRadius: 13, style: .continuous)
                            .fill(canGenerate ? Studio.accent : Studio.sunk)
                    )
            }
            .buttonStyle(PressableButtonStyle())
            .disabled(!canGenerate)
            .padding(16)
        }
    }
}

// MARK: - Model picker

struct MusicModelPicker: View {
    @Binding var selected: MusicModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 10) {
                    ForEach(MusicModel.allCases) { model in
                        Button { selected = model; dismiss() } label: { card(model) }
                            .buttonStyle(PressableButtonStyle())
                    }
                }
                .padding(16)
            }
            .background(Studio.bg.ignoresSafeArea())
            .navigationTitle("Music Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func card(_ model: MusicModel) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(alignment: .firstTextBaseline) {
                Text(model.displayName)
                    .font(Studio.text(16, weight: .semibold))
                    .foregroundColor(Studio.ink)
                Spacer()
                Text(model.parameterLabel)
                    .font(Studio.mono(10, weight: .semibold))
                    .foregroundColor(Studio.mute)
            }
            Text(model.tagline)
                .font(Studio.text(12))
                .foregroundColor(Studio.mute)

            HStack(spacing: 7) {
                chip(model.sizeLabel, tint: Studio.mute)
                switch model.availability {
                case .ready:               chip("RUNS ON DEVICE", tint: Studio.ok)
                case .weightsPublished:    chip("UNTESTED", tint: Studio.hot)
                case .needsConversion:     chip("NEEDS CONVERSION", tint: Studio.mute)
                }
                chip(model.engine == .coreML ? "CORE ML" : "MLX", tint: Studio.mute)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(selected == model ? Studio.panel : Studio.sunk)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(selected == model ? Studio.accent : Studio.rule,
                              lineWidth: selected == model ? 1.5 : 0.5)
        )
    }

    private func chip(_ text: String, tint: Color) -> some View {
        Text(text)
            .font(Studio.mono(9, weight: .semibold))
            .foregroundColor(tint)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Capsule().fill(tint.opacity(0.12)))
    }
}
