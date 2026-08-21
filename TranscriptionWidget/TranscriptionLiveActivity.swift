import ActivityKit
import WidgetKit
import SwiftUI

// MARK: - Attributes (Complete definition in widget)
@available(iOS 16.2, *)
struct TranscriptionAttributes: ActivityAttributes {
    public struct ContentState: Codable, Hashable {
        var progress: Double
        var currentSegment: Int
        var totalSegments: Int
        var status: String
    }
    
    var fileName: String
}

// MARK: - Live Activity Widget
@available(iOS 16.2, *)
struct TranscriptionLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: TranscriptionAttributes.self) { context in
            // Lock screen/banner UI
            HStack(spacing: 12) {
                // Icon
                Image(systemName: "waveform")
                    .font(.title2)
                    .foregroundColor(.blue)
                    .frame(width: 40, height: 40)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(8)
                
                VStack(alignment: .leading, spacing: 6) {
                    Text("Transcribing Audio")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text("\(context.state.currentSegment)/\(context.state.totalSegments) segments")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    // Progress Bar
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            Rectangle()
                                .fill(Color.gray.opacity(0.2))
                                .frame(height: 6)
                                .cornerRadius(3)
                            
                            Rectangle()
                                .fill(
                                    LinearGradient(
                                        colors: [.blue, .cyan],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: geometry.size.width * context.state.progress, height: 6)
                                .cornerRadius(3)
                        }
                    }
                    .frame(height: 6)
                }
                
                Spacer()
                
                // Progress Percentage
                Text("\(Int(context.state.progress * 100))%")
                    .font(.system(.title3, design: .monospaced))
                    .bold()
                    .foregroundColor(.blue)
            }
            .padding()
            .activityBackgroundTint(Color.white)
            .activitySystemActionForegroundColor(Color.blue)
            
        } dynamicIsland: { context in
            DynamicIsland {
                // Expanded UI
                DynamicIslandExpandedRegion(.leading) {
                    HStack(spacing: 8) {
                        Image(systemName: "waveform")
                            .foregroundColor(.blue)
                        Text("Transcribing")
                            .font(.caption)
                    }
                }
                
                DynamicIslandExpandedRegion(.trailing) {
                    Text("\(Int(context.state.progress * 100))%")
                        .font(.system(.body, design: .monospaced))
                        .bold()
                }
                
                DynamicIslandExpandedRegion(.bottom) {
                    VStack(spacing: 8) {
                        // Progress Bar
                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                Rectangle()
                                    .fill(Color.gray.opacity(0.3))
                                    .frame(height: 8)
                                    .cornerRadius(4)
                                
                                Rectangle()
                                    .fill(
                                        LinearGradient(
                                            colors: [.blue, .cyan],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        )
                                    )
                                    .frame(width: geometry.size.width * context.state.progress, height: 8)
                                    .cornerRadius(4)
                            }
                        }
                        .frame(height: 8)
                        
                        HStack {
                            Text("\(context.state.currentSegment)/\(context.state.totalSegments) segments")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            
                            Spacer()
                            
                            Text(context.state.status)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                                .lineLimit(1)
                        }
                    }
                    .padding(.top, 4)
                }
            } compactLeading: {
                // Compact Leading (left side of Dynamic Island)
                Image(systemName: "waveform")
                    .foregroundColor(.blue)
            } compactTrailing: {
                // Compact Trailing (right side of Dynamic Island)
                Text("\(Int(context.state.progress * 100))%")
                    .font(.system(.caption, design: .monospaced))
                    .bold()
            } minimal: {
                // Minimal view (when multiple activities)
                Image(systemName: "waveform")
                    .foregroundColor(.blue)
            }
            .keylineTint(Color.blue)
        }
    }
}

// MARK: - Widget Bundle (Main Entry Point)
@main
struct TranscriptionWidgetBundle: WidgetBundle {
    var body: some Widget {
        if #available(iOS 16.2, *) {
            TranscriptionLiveActivity()
        }
    }
}

// MARK: - Previews
@available(iOS 16.2, *)
struct TranscriptionLiveActivity_Previews: PreviewProvider {
    static let attributes = TranscriptionAttributes(fileName: "test_audio.mp3")
    static let contentState = TranscriptionAttributes.ContentState(
        progress: 0.45,
        currentSegment: 12,
        totalSegments: 27,
        status: "Processing audio..."
    )
    
    static var previews: some View {
        attributes
            .previewContext(contentState, viewKind: .content)
            .previewDisplayName("Notification")
        
        attributes
            .previewContext(contentState, viewKind: .dynamicIsland(.compact))
            .previewDisplayName("Compact")
        
        attributes
            .previewContext(contentState, viewKind: .dynamicIsland(.expanded))
            .previewDisplayName("Expanded")
        
        attributes
            .previewContext(contentState, viewKind: .dynamicIsland(.minimal))
            .previewDisplayName("Minimal")
    }
}
