import Charts
import SwiftUI

/// A recording looked at closely: how fast and how evenly it was said,
/// where the pauses fall, the voice's pitch and loudness, the background
/// noise, a spectrogram, who spoke for how long, and the words used most.
/// Measured on this device from the recording and its transcript, once;
/// tapping the spectrogram or a pause plays the recording from there.
struct SpeechAnalysisView: View {
    let transcript: SavedTranscript
    /// Plays the recording from a moment; nil where nothing can.
    var onSeek: ((Double) -> Void)?

    @Environment(\.dismiss) private var dismiss

    private enum Sound {
        case reading(Double)
        case done(AudioAnalysis)
        case missing
        case failed(String)
    }
    @State private var sound: Sound = .reading(0)
    @State private var words: SpeechTextAnalysis?

    private var audio: AudioAnalysis? {
        if case .done(let analysis) = sound { return analysis }
        return nil
    }

    var body: some View {
        NavigationStack {
            ScrollViewReader { scroller in
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        header
                        figures
                        switch sound {
                        case .reading(let fraction): readingCard(fraction)
                        case .missing:
                            note("The recording isn't kept with this transcript, so only its words can be measured.")
                        case .failed(let message):
                            note("The recording couldn't be measured: \(message)")
                        case .done: EmptyView()
                        }
                        if let audio {
                            spectrogramCard(audio).id("spectrogram")
                            levelCard(audio).id("loudness")
                            if audio.pitchMedian > 0 { pitchCard(audio).id("pitch") }
                            pausesCard(audio).id("pauses")
                        }
                        if let words {
                            if words.count > 0 { paceCard(words).id("pace") }
                            if !words.speakers.isEmpty { speakersCard(words).id("speakers") }
                            if let fillers = words.fillers { hesitationsCard(fillers).id("hesitations") }
                            if !words.frequent.isEmpty { frequentCard(words).id("words") }
                        }
                        footnote
                    }
                    .padding(18)
                    .frame(maxWidth: 760)
                    .frame(maxWidth: .infinity)
                }
                .background(Studio.bg.ignoresSafeArea())
                .navigationTitle("Speech analysis")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Done") { dismiss() }
                    }
                }
                .task(id: transcript.id) {
                    await analyze()
                    #if DEBUG
                    // `-DebugAnalysisScroll pitch` (or another section): there, for screenshots.
                    if let section = UserDefaults.standard.string(forKey: "DebugAnalysisScroll") {
                        try? await Task.sleep(for: .milliseconds(300))
                        scroller.scrollTo(section, anchor: .top)
                    }
                    #endif
                }
            }
        }
    }

    // MARK: - Work

    private func analyze() async {
        // Read here: the transcript's records belong to the main actor.
        var lines = transcript.segments.sorted { $0.startTime < $1.startTime }.map {
            SpeechTextAnalysis.Line(start: $0.startTime, end: $0.endTime,
                                    text: $0.originalText ?? $0.text, speaker: $0.speaker)
        }
        if lines.isEmpty, !transcript.fullText.isEmpty {
            lines = [.init(start: 0, end: transcript.duration,
                           text: transcript.originalFullText ?? transcript.fullText, speaker: nil)]
        }
        let language = transcript.originalLanguageCode ?? transcript.languageCode
        let duration = transcript.duration
        words = await Task.detached(priority: .userInitiated) {
            SpeechTextAnalysis(lines: lines, duration: duration, language: language)
        }.value

        guard let relative = transcript.audioFilePath else { sound = .missing; return }
        let url = AudioFiles.urlForRelativePath(relative)
        guard FileManager.default.fileExists(atPath: url.path) else { sound = .missing; return }
        let id = transcript.id
        if let saved = await Task.detached(operation: { AudioAnalysisCache.load(for: id, audio: url) }).value {
            sound = .done(saved)
            return
        }
        let job = Task.detached(priority: .userInitiated) { () throws -> AudioAnalysis in
            let analysis = try SpeechAnalyzer.analyze(url) { fraction in
                Task { @MainActor in
                    if case .reading = sound { sound = .reading(fraction) }
                }
            }
            AudioAnalysisCache.save(analysis, for: id, audio: url)
            return analysis
        }
        do {
            let analysis = try await withTaskCancellationHandler { try await job.value } onCancel: { job.cancel() }
            sound = .done(analysis)
        } catch is CancellationError {
        } catch {
            sound = .failed(error.localizedDescription)
        }
    }

    private func seek(_ seconds: Double) {
        onSeek?(max(0, seconds))
    }

    // MARK: - Header and figures

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(transcript.title)
                .font(Studio.text(20, weight: .semibold))
                .foregroundColor(Studio.ink)
                .lineLimit(2)
            Text("\(TranscriptExporter.formatTimestamp(transcript.duration)) · measured on this device")
                .font(Studio.mono(11))
                .foregroundColor(Studio.mute)
        }
    }

    private struct Figure: Identifiable {
        let title: String
        let value: String
        var unit = ""
        var detail = ""
        var id: String { title }
    }

    private var figureList: [Figure] {
        var list: [Figure] = []
        if let words, words.count > 0 {
            var detail = "\(words.count.formatted()) \(words.unit) in all"
            if let audio, audio.speakingTime > 30 {
                let speaking = Double(words.count) / (audio.speakingTime / 60)
                detail = "\(Int(speaking.rounded())) while speaking"
            }
            list.append(Figure(title: "Pace", value: "\(Int(words.perMinute.rounded()))",
                               unit: "\(words.unit)/min", detail: detail))
        }
        if let audio {
            let share = audio.duration > 0 ? Int((audio.speakingTime / audio.duration * 100).rounded()) : 0
            list.append(Figure(title: "Speaking", value: TranscriptExporter.formatTimestamp(audio.speakingTime),
                               detail: "\(share)% of \(TranscriptExporter.formatTimestamp(audio.duration))"))
            let longest = audio.pauses.map(\.length).max() ?? 0
            let minutes = max(audio.duration / 60, 1 / 60)
            list.append(Figure(title: "Pauses", value: "\(audio.pauses.count)",
                               detail: audio.pauses.isEmpty
                                   ? "none of half a second or more"
                                   : "longest \(Self.seconds(longest)) · \(Self.oneDecimal(Double(audio.pauses.count) / minutes))/min"))
            if audio.pitchMedian > 0 {
                list.append(Figure(title: "Pitch", value: "\(Int(audio.pitchMedian.rounded()))", unit: "Hz",
                                   detail: "mostly \(Int(audio.pitchLow.rounded()))–\(Int(audio.pitchHigh.rounded())) Hz"))
                list.append(Figure(title: "Intonation", value: "±\(Self.oneDecimal(Double(audio.pitchSpread)))",
                                   unit: "semitones", detail: Self.intonation(audio.pitchSpread)))
            }
            list.append(Figure(title: "Voice level", value: Self.decibels(audio.speechLevel), unit: "dBFS",
                               detail: "peak \(Self.decibels(audio.peak)) dBFS"))
            let gap = audio.speechLevel - audio.background
            list.append(Figure(title: "Background", value: Self.decibels(audio.background), unit: "dBFS",
                               detail: "\(Int(gap.rounded())) dB under the voice · \(Self.clarity(gap))"))
            if audio.clipping > 0 {
                list.append(Figure(title: "Clipping", value: "\(audio.clipping)",
                                   unit: audio.clipping == 1 ? "moment" : "moments",
                                   detail: "the sound hit the recording's limit"))
            }
        }
        if let words, words.count > 0 {
            list.append(Figure(title: words.inCharacters ? "Characters" : "Words", value: words.count.formatted(),
                               detail: "\(words.lines.formatted()) \(words.lines == 1 ? "line" : "lines")"))
        }
        return list
    }

    private var figures: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10) {
            ForEach(figureList) { figure in
                VStack(alignment: .leading, spacing: 6) {
                    StudioLabel(text: figure.title)
                    HStack(alignment: .firstTextBaseline, spacing: 4) {
                        Text(figure.value)
                            .font(Studio.text(22, weight: .semibold))
                            .foregroundColor(Studio.ink)
                            .monospacedDigit()
                        if !figure.unit.isEmpty {
                            Text(figure.unit)
                                .font(Studio.text(12))
                                .foregroundColor(Studio.mute)
                        }
                    }
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    if !figure.detail.isEmpty {
                        Text(figure.detail)
                            .font(Studio.mono(10))
                            .foregroundColor(Studio.mute)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, minHeight: 92, alignment: .topLeading)
                .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Studio.panel))
                .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(Studio.rule, lineWidth: 0.5))
                .accessibilityElement(children: .combine)
            }
        }
    }

    // MARK: - Cards

    private func card<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12, content: content)
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(Studio.panel))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(Studio.rule, lineWidth: 0.5))
    }

    private func explain(_ text: String) -> some View {
        Text(text)
            .font(Studio.text(12))
            .foregroundColor(Studio.mute)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func note(_ text: String) -> some View {
        card { explain(text) }
    }

    private func readingCard(_ fraction: Double) -> some View {
        card {
            HStack {
                StudioLabel(text: "Listening to the recording")
                Spacer()
                Text("\(Int(fraction * 100))%")
                    .font(Studio.mono(11, weight: .semibold))
                    .foregroundColor(Studio.accent)
            }
            ProgressView(value: fraction).tint(Studio.accent)
            explain("Pitch, loudness, pauses and the spectrogram are measured once and kept, so this is quick next time.")
        }
    }

    private func spectrogramCard(_ audio: AudioAnalysis) -> some View {
        card {
            HStack {
                StudioLabel(text: "Spectrogram")
                Spacer()
                if onSeek != nil {
                    Text("Tap to listen from there")
                        .font(Studio.mono(9))
                        .foregroundColor(Studio.mute)
                }
            }
            SpectrogramView(spectrogram: audio.spectrogram, duration: audio.duration,
                            onSeek: onSeek == nil ? nil : { seek($0) })
                .frame(height: 170)
            timeAxis(audio.duration)
            explain("Time runs left to right and pitch rises upward; brighter is louder. A voice shows as stacked bands, its harmonics, and steady background noise as an even haze.")
        }
    }

    private func timeAxis(_ duration: Double) -> some View {
        HStack {
            Text("0:00")
            Spacer()
            Text(TranscriptExporter.formatTimestamp(duration / 2))
            Spacer()
            Text(TranscriptExporter.formatTimestamp(duration))
        }
        .font(Studio.mono(9))
        .foregroundColor(Studio.mute)
    }

    private struct Point: Identifiable {
        let time: Double
        let value: Double
        var id: Double { time }
    }

    private func levelCard(_ audio: AudioAnalysis) -> some View {
        let points = Self.series(audio.loudness.map { Double($0) }, maxPoints: 360) { values in
            10 * log10(values.map { pow(10, $0 / 10) }.reduce(0, +) / Double(values.count))
        }
        let floor = Double(min(-60, audio.background - 6)).rounded(.down)
        return card {
            StudioLabel(text: "Loudness")
            Chart {
                ForEach(points) { point in
                    AreaMark(x: .value("Time", point.time),
                             yStart: .value("Floor", floor),
                             yEnd: .value("Level", max(floor, point.value)))
                        .foregroundStyle(Studio.accent.opacity(0.35))
                }
                RuleMark(y: .value("Voice", Double(audio.speechLevel)))
                    .foregroundStyle(Studio.accent)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .annotation(position: .top, alignment: .leading) {
                        Text("voice").font(Studio.mono(9)).foregroundColor(Studio.accent)
                    }
                RuleMark(y: .value("Background", Double(audio.background)))
                    .foregroundStyle(Studio.mute)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .annotation(position: .top, alignment: .leading) {
                        Text("background").font(Studio.mono(9)).foregroundColor(Studio.mute)
                    }
            }
            .chartYScale(domain: floor ... 0)
            .chartXScale(domain: 0 ... max(audio.duration, 1))
            .chartXAxis { timeMarks }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { value in
                    AxisGridLine().foregroundStyle(Studio.rule)
                    AxisValueLabel { Text("\(Self.decibels(Float(value.as(Double.self) ?? 0)))").font(Studio.mono(9)) }
                }
            }
            .frame(height: 150)
            explain("In dBFS, where 0 is the loudest a recording can hold. Speech usually sits 20 dB or more above the background; less than about 15 dB and words get harder to make out.")
        }
    }

    private func pitchCard(_ audio: AudioAnalysis) -> some View {
        let points = Self.series(audio.pitch.map { Double($0) }, maxPoints: 360, skipZero: true) { values in
            values.sorted()[values.count / 2]
        }
        let low = max(50, Double(audio.pitchLow) * 0.7)
        let high = min(600, Double(audio.pitchHigh) * 1.4)
        return card {
            StudioLabel(text: "Pitch")
            Chart {
                ForEach(points) { point in
                    PointMark(x: .value("Time", point.time), y: .value("Pitch", min(high, max(low, point.value))))
                        .symbolSize(10)
                        .foregroundStyle(Studio.accent)
                }
                RuleMark(y: .value("Median", Double(audio.pitchMedian)))
                    .foregroundStyle(Studio.mute)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .annotation(position: .top, alignment: .leading) {
                        Text("\(Int(audio.pitchMedian.rounded())) Hz").font(Studio.mono(9)).foregroundColor(Studio.mute)
                    }
            }
            .chartYScale(domain: low ... high, type: .log)
            .chartXScale(domain: 0 ... max(audio.duration, 1))
            .chartXAxis { timeMarks }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { value in
                    AxisGridLine().foregroundStyle(Studio.rule)
                    AxisValueLabel { Text("\(Int(value.as(Double.self) ?? 0))").font(Studio.mono(9)) }
                }
            }
            .frame(height: 150)
            explain("The voice's fundamental frequency, a second at a time. Most voices rise and fall by about 2 to 4 semitones; less sounds flatter, more sounds livelier. Lower voices usually sit around 85–180 Hz, higher ones around 165–255 Hz.")
        }
    }

    private func pausesCard(_ audio: AudioAnalysis) -> some View {
        let longest = audio.pauses.sorted { $0.length > $1.length }.prefix(6)
        let top = longest.first?.length ?? 1
        let total = audio.pauses.map(\.length).reduce(0, +)
        return card {
            StudioLabel(text: "Pauses")
            if audio.pauses.isEmpty {
                explain("No silences of half a second or more between the first and last words.")
            } else {
                explain("\(audio.pauses.count) silences of half a second or more, \(TranscriptExporter.formatTimestamp(total)) in all; on average \(Self.seconds(total / Double(audio.pauses.count))). The longest:")
                VStack(spacing: 2) {
                    ForEach(Array(longest), id: \.self) { pause in
                        Button { seek(pause.start - 2) } label: {
                            HStack(spacing: 12) {
                                Text(TranscriptExporter.formatTimestamp(pause.start))
                                    .font(Studio.mono(11))
                                    .foregroundColor(Studio.mute)
                                    .frame(width: 52, alignment: .leading)
                                GeometryReader { geo in
                                    Capsule()
                                        .fill(Studio.accent.opacity(0.55))
                                        .frame(width: max(4, geo.size.width * pause.length / top), height: 6)
                                        .frame(maxHeight: .infinity)
                                }
                                .frame(height: 20)
                                Text(Self.seconds(pause.length))
                                    .font(Studio.mono(11, weight: .semibold))
                                    .foregroundColor(Studio.ink)
                                    .frame(width: 50, alignment: .trailing)
                                if onSeek != nil {
                                    Image(systemName: "play.circle")
                                        .foregroundColor(Studio.accent)
                                }
                            }
                            .padding(.vertical, 6)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .disabled(onSeek == nil)
                        .accessibilityLabel("\(Self.seconds(pause.length)) pause at \(TranscriptExporter.formatTimestamp(pause.start))")
                    }
                }
            }
        }
    }

    private func paceCard(_ words: SpeechTextAnalysis) -> some View {
        card {
            StudioLabel(text: "Pace over time")
            Chart {
                ForEach(words.pace.filter { $0.start < paceEnd }, id: \.self) { bucket in
                    // Within the recording: the last bucket is often short.
                    RectangleMark(xStart: .value("From", bucket.start),
                                  xEnd: .value("To", min(bucket.start + words.bucketSeconds * 0.92, paceEnd)),
                                  yStart: .value("Zero", 0),
                                  yEnd: .value(words.unit, bucket.perMinute))
                        .foregroundStyle(Studio.accent.opacity(0.7))
                }
                RuleMark(y: .value("Average", words.perMinute))
                    .foregroundStyle(Studio.mute)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .annotation(position: .top, alignment: .leading) {
                        Text("average \(Int(words.perMinute.rounded()))").font(Studio.mono(9)).foregroundColor(Studio.mute)
                    }
            }
            .chartXScale(domain: 0 ... paceEnd)
            .chartXAxis { timeMarks }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { value in
                    AxisGridLine().foregroundStyle(Studio.rule)
                    AxisValueLabel { Text("\(Int(value.as(Double.self) ?? 0))").font(Studio.mono(9)) }
                }
            }
            .frame(height: 140)
            explain(words.inCharacters
                ? "Characters a minute, every \(Int(words.bucketSeconds)) seconds. Conversational Mandarin runs at roughly 200 to 300."
                : "Words a minute, every \(Int(words.bucketSeconds)) seconds. Conversation usually runs at 120 to 160; presentations a little slower.")
        }
    }

    private var paceEnd: Double { max(transcript.duration, 1) }

    private func speakersCard(_ words: SpeechTextAnalysis) -> some View {
        let total = max(words.speakers.map(\.seconds).reduce(0, +), 1)
        return card {
            StudioLabel(text: "Who spoke")
            ForEach(words.speakers, id: \.self) { speaker in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(speaker.name)
                            .font(Studio.text(13, weight: .medium))
                            .foregroundColor(Studio.ink)
                        Spacer()
                        Text("\(TranscriptExporter.formatTimestamp(speaker.seconds)) · \(Int((speaker.seconds / total * 100).rounded()))%")
                            .font(Studio.mono(11))
                            .foregroundColor(Studio.mute)
                    }
                    GeometryReader { geo in
                        Capsule().fill(Studio.sunk)
                            .overlay(alignment: .leading) {
                                Capsule().fill(Studio.accent.opacity(speaker.name == "No name" ? 0.3 : 0.7))
                                    .frame(width: geo.size.width * speaker.seconds / total)
                            }
                    }
                    .frame(height: 6)
                }
            }
            explain("From the speaker names added to the transcript's lines.")
        }
    }

    private func hesitationsCard(_ fillers: [SpeechTextAnalysis.Count]) -> some View {
        card {
            StudioLabel(text: "Hesitations")
            if fillers.isEmpty {
                Text("None written in the transcript.")
                    .font(Studio.text(14))
                    .foregroundColor(Studio.ink)
            } else {
                Chips(counts: fillers)
            }
            explain("Speech recognition often leaves out \"um\" and \"uh\", so the recording may hold more than the transcript shows.")
        }
    }

    private func frequentCard(_ words: SpeechTextAnalysis) -> some View {
        card {
            StudioLabel(text: "Words used most")
            Chips(counts: words.frequent)
            explain(words.frequentByWordClass
                ? "Nouns and adjectives, counted in their dictionary form."
                : "Words of four letters or more, said at least twice.")
        }
    }

    private var footnote: some View {
        Text("Everything here is worked out on this device from the recording and its transcript. Speech is told from silence by loudness, so a noisy room can blur pauses, and pitch is read only where the voice is steady enough to measure.")
            .font(Studio.text(11))
            .foregroundColor(Studio.mute)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 4)
    }

    private var timeMarks: some AxisContent {
        AxisMarks(values: .automatic(desiredCount: 4)) { value in
            AxisGridLine().foregroundStyle(Studio.rule)
            AxisValueLabel {
                Text(TranscriptExporter.formatTimestamp(value.as(Double.self) ?? 0)).font(Studio.mono(9))
            }
        }
    }

    // MARK: - Formatting

    /// Per-second values gathered into at most `maxPoints`, each at its
    /// group's middle.
    private static func series(_ values: [Double], maxPoints: Int, skipZero: Bool = false,
                               combine: ([Double]) -> Double) -> [Point] {
        guard !values.isEmpty else { return [] }
        let group = max(1, Int((Double(values.count) / Double(maxPoints)).rounded(.up)))
        return stride(from: 0, to: values.count, by: group).compactMap { start in
            let slice = values[start ..< min(values.count, start + group)].filter { !skipZero || $0 > 0 }
            guard !slice.isEmpty else { return nil }
            return Point(time: Double(start) + Double(group) / 2, value: combine(Array(slice)))
        }
    }

    private static func decibels(_ value: Float) -> String {
        let rounded = Int(value.rounded())
        return rounded < 0 ? "\u{2212}\(-rounded)" : "\(rounded)"
    }

    private static func oneDecimal(_ value: Double) -> String { String(format: "%.1f", value) }

    private static func seconds(_ value: Double) -> String { String(format: "%.1f s", value) }

    private static func intonation(_ spread: Float) -> String {
        spread < 2 ? "fairly even" : spread < 4 ? "varied" : "very varied"
    }

    private static func clarity(_ gap: Float) -> String {
        gap >= 30 ? "clean" : gap >= 15 ? "some noise" : "noisy"
    }
}

/// The spectrogram, drawn once into an image; a tap plays from that moment.
private struct SpectrogramView: View {
    let spectrogram: AudioAnalysis.Spectrogram
    let duration: Double
    let onSeek: ((Double) -> Void)?
    @State private var image: CGImage?

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                Color(red: 0.043, green: 0.039, blue: 0.078)
                if let image {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .interpolation(.medium)
                }
                ForEach([250.0, 1_000, 4_000], id: \.self) { hz in
                    Text(hz >= 1_000 ? "\(Int(hz / 1_000)) kHz" : "\(Int(hz)) Hz")
                        .font(Studio.mono(8))
                        .foregroundColor(.white.opacity(0.75))
                        .padding(.horizontal, 4)
                        .background(Color.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 3))
                        .offset(x: 4, y: geo.size.height * (1 - fraction(of: hz)) - 7)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .contentShape(Rectangle())
            .onTapGesture { location in
                guard geo.size.width > 0 else { return }
                onSeek?(Double(location.x / geo.size.width) * duration)
            }
        }
        .task(id: spectrogram.levels.count) { image = Self.render(spectrogram) }
        .accessibilityElement()
        .accessibilityLabel("Spectrogram of the recording")
    }

    /// Where a frequency sits on the mel-spaced bands, 0 at the bottom.
    private func fraction(of hz: Double) -> Double {
        func mel(_ hz: Double) -> Double { 2595 * log10(1 + hz / 700) }
        return (mel(hz) - mel(spectrogram.lowHz)) / (mel(spectrogram.highHz) - mel(spectrogram.lowHz))
    }

    /// Dark violet through the app's blue to near white.
    private static let palette: [(r: Double, g: Double, b: Double)] = {
        let stops: [(Double, (Double, Double, Double))] = [
            (0.00, (0.043, 0.039, 0.078)),
            (0.30, (0.118, 0.157, 0.353)),
            (0.55, (0.208, 0.443, 0.608)),
            (0.80, (0.498, 0.702, 0.835)),
            (1.00, (0.949, 0.945, 0.960)),
        ]
        return (0 ..< 256).map { index in
            let t = Double(index) / 255
            let upper = stops.firstIndex { $0.0 >= t } ?? stops.count - 1
            let lower = max(0, upper - 1)
            let (t0, c0) = stops[lower]
            let (t1, c1) = stops[upper]
            let f = t1 > t0 ? (t - t0) / (t1 - t0) : 0
            return (c0.0 + (c1.0 - c0.0) * f, c0.1 + (c1.1 - c0.1) * f, c0.2 + (c1.2 - c0.2) * f)
        }
    }()

    private static func render(_ spectrogram: AudioAnalysis.Spectrogram) -> CGImage? {
        let (width, height) = (spectrogram.columns, spectrogram.bands)
        guard width > 0, height > 0, spectrogram.levels.count == width * height else { return nil }
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        spectrogram.levels.withUnsafeBytes { (levels: UnsafeRawBufferPointer) in
            for x in 0 ..< width {
                for y in 0 ..< height {
                    // Lowest band at the bottom.
                    let colour = palette[Int(levels[x * height + (height - 1 - y)])]
                    let offset = (y * width + x) * 4
                    pixels[offset] = UInt8(colour.r * 255)
                    pixels[offset + 1] = UInt8(colour.g * 255)
                    pixels[offset + 2] = UInt8(colour.b * 255)
                }
            }
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

/// Words with their counts, wrapping onto as many lines as they need.
private struct Chips: View {
    let counts: [SpeechTextAnalysis.Count]

    var body: some View {
        FlowLayout(spacing: 6) {
            ForEach(counts, id: \.self) { item in
                HStack(spacing: 4) {
                    Text(item.word)
                        .font(Studio.text(13, weight: .medium))
                        .foregroundColor(Studio.ink)
                    Text("×\(item.count)")
                        .font(Studio.mono(10))
                        .foregroundColor(Studio.mute)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Capsule().fill(Studio.sunk))
            }
        }
    }
}

/// Lays its views out left to right, starting a new row when one is full.
private struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(subviews, width: proposal.width ?? .infinity)
        let height = rows.map(\.height).reduce(0, +) + spacing * CGFloat(max(0, rows.count - 1))
        return CGSize(width: proposal.width ?? rows.map(\.width).max() ?? 0, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for row in arrange(subviews, width: bounds.width) {
            var x = bounds.minX
            for index in row.indices {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private struct Row {
        var indices: [Int] = []
        var width: CGFloat = 0
        var height: CGFloat = 0
    }

    private func arrange(_ subviews: Subviews, width: CGFloat) -> [Row] {
        var rows: [Row] = []
        var row = Row()
        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            if !row.indices.isEmpty, row.width + spacing + size.width > width {
                rows.append(row)
                row = Row()
            }
            row.width += (row.indices.isEmpty ? 0 : spacing) + size.width
            row.height = max(row.height, size.height)
            row.indices.append(index)
        }
        if !row.indices.isEmpty { rows.append(row) }
        return rows
    }
}
