import SwiftUI

/// Paste a link to an audio or video file; it downloads here and is
/// transcribed like an imported file. See `MediaLink`.
struct LinkImportSheet: View {
    /// Gets the downloaded file, then owns it.
    let onFile: (URL) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var text = ""
    @State private var fraction: Double?
    @State private var downloading = false
    @State private var failure: String?
    @State private var task: Task<Void, Never>?
    @FocusState private var focused: Bool

    private var link: URL? { MediaLink.url(in: text) }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Paste a link to an audio or video file — one that ends in .mp3, .m4a, .wav or .mp4, for example. Pages that play a file, like many podcast episode pages, often work too.")
                        .font(Studio.text(14))
                        .foregroundColor(Studio.ink.opacity(0.8))
                        .fixedSize(horizontal: false, vertical: true)

                    HStack(spacing: 8) {
                        TextField("https://…", text: $text, axis: .vertical)
                            .font(Studio.text(15))
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .keyboardType(.URL)
                            .focused($focused)
                            .lineLimit(1 ... 4)
                            .padding(12)
                            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Studio.sunk))
                            .disabled(downloading)
                        PasteButton(payloadType: String.self) { strings in
                            if let first = strings.first { text = first; failure = nil }
                        }
                        .labelStyle(.iconOnly)
                        .buttonBorderShape(.roundedRectangle(radius: 12))
                        .disabled(downloading)
                    }

                    if downloading {
                        VStack(alignment: .leading, spacing: 6) {
                            if let fraction {
                                ProgressView(value: fraction).tint(Studio.accent)
                                Text("Downloading \(Int(fraction * 100))%")
                            } else {
                                ProgressView().tint(Studio.accent)
                                Text("Downloading…")
                            }
                        }
                        .font(Studio.mono(11))
                        .foregroundColor(Studio.mute)
                    }

                    if let failure {
                        Label(failure, systemImage: "exclamationmark.circle")
                            .font(Studio.text(13))
                            .foregroundColor(Studio.ink)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Studio.sunk))
                    }

                    Button {
                        if downloading { cancel() } else { start() }
                    } label: {
                        Label(downloading ? "Cancel" : "Download and transcribe",
                              systemImage: downloading ? "xmark" : "arrow.down.circle")
                            .font(Studio.text(15, weight: .semibold))
                            .foregroundColor(link == nil && !downloading ? Studio.mute : Studio.onAccent)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .background(RoundedRectangle(cornerRadius: 13, style: .continuous)
                                .fill(link == nil && !downloading ? Studio.sunk : (downloading ? Studio.hot : Studio.accent)))
                    }
                    .buttonStyle(PressableButtonStyle())
                    .disabled(link == nil && !downloading)

                    Label("Links to YouTube, Instagram, Facebook, TikTok and similar sites can't be downloaded. Save your own videos to Photos or Files and import them from there.",
                          systemImage: "info.circle")
                        .font(Studio.text(12))
                        .foregroundColor(Studio.mute)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(18)
            }
            .background(Studio.bg.ignoresSafeArea())
            .navigationTitle("Transcribe a Link")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Close") { cancel(); dismiss() }
                }
            }
            .onAppear { focused = true }
            .onDisappear { task?.cancel() }
        }
    }

    private func start() {
        guard let link else { return }
        if let refusal = MediaLink.refusal(for: link) {
            failure = refusal.errorDescription
            return
        }
        focused = false
        failure = nil
        fraction = nil
        downloading = true
        // Keeps downloading if the app is left; transcription then carries
        // on in its own background job.
        let job = BackgroundWork.shared.begin(title: "Downloading for transcription", subtitle: link.host ?? link.absoluteString)
        task = Task {
            do {
                let file = try await MediaLink.download(link) { value in
                    Task { @MainActor in
                        fraction = value
                        if let value { job.update(value) }
                    }
                }
                job.finish(success: true)
                downloading = false
                onFile(file)
                dismiss()
            } catch is CancellationError {
                job.finish(success: false)
                downloading = false
            } catch let error as URLError where error.code == .cancelled {
                job.finish(success: false)
                downloading = false
            } catch {
                job.finish(success: false)
                downloading = false
                failure = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            }
        }
    }

    private func cancel() {
        task?.cancel()
        task = nil
        downloading = false
    }
}
