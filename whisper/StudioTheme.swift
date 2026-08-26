import SwiftUI

/// The Studio design system.
///
/// Replaces the warm-paper editorial palette. Where paper was a calm notebook,
/// Studio is a tool: cool near-black panels, a single blue accent for anything
/// active or selected, and terracotta reserved exclusively for recording —
/// carried over from the old identity so the one destructive control still
/// reads the way it always has.
enum Studio {

    // MARK: - Palette

    /// Resolves per trait collection so every call site stays a plain token
    /// while still following the system appearance.
    private static func adaptive(
        light: (r: Double, g: Double, b: Double, a: Double),
        dark: (r: Double, g: Double, b: Double, a: Double)
    ) -> Color {
        Color(UIColor { traits in
            let c = traits.userInterfaceStyle == .dark ? dark : light
            return UIColor(red: c.r, green: c.g, blue: c.b, alpha: c.a)
        })
    }

    /// Page background.
    static let bg = adaptive(
        light: (0.949, 0.945, 0.960, 1.0),   // #F2F1F5
        dark:  (0.090, 0.086, 0.110, 1.0)    // #17161C
    )
    /// Raised surfaces: rails, toolbars, transport.
    static let panel = adaptive(
        light: (1.000, 1.000, 1.000, 1.0),
        dark:  (0.118, 0.114, 0.145, 1.0)    // #1E1D25
    )
    /// Recessed wells: the waveform trough, preview panes.
    static let sunk = adaptive(
        light: (0.914, 0.910, 0.929, 1.0),   // #E9E8ED
        dark:  (0.075, 0.071, 0.094, 1.0)    // #131218
    )
    static let ink = adaptive(
        light: (0.106, 0.102, 0.133, 1.0),   // #1B1A22
        dark:  (0.910, 0.902, 0.937, 1.0)    // #E8E6EF
    )
    static let mute = adaptive(
        light: (0.431, 0.420, 0.482, 1.0),   // #6E6B7B
        dark:  (0.545, 0.533, 0.600, 1.0)    // #8B8899
    )
    static let rule = adaptive(
        light: (0.000, 0.000, 0.000, 0.10),
        dark:  (1.000, 1.000, 1.000, 0.09)
    )
    /// Selection, playback, anything active.
    static let accent = adaptive(
        light: (0.208, 0.443, 0.608, 1.0),   // #35719B
        dark:  (0.498, 0.702, 0.835, 1.0)    // #7FB3D5
    )
    /// Recording only. Inherited from the paper palette's terracotta.
    static let hot = adaptive(
        light: (0.769, 0.314, 0.247, 1.0),   // #C4503F
        dark:  (0.878, 0.408, 0.353, 1.0)    // #E0685A
    )
    static let ok = adaptive(
        light: (0.243, 0.604, 0.392, 1.0),   // #3E9A64
        dark:  (0.420, 0.769, 0.557, 1.0)    // #6BC48E
    )
    /// Unplayed waveform bars.
    static let idle = adaptive(
        light: (0.000, 0.000, 0.000, 0.19),
        dark:  (1.000, 1.000, 1.000, 0.17)
    )
    /// Text drawn on top of `accent`.
    static let onAccent = adaptive(
        light: (1.000, 1.000, 1.000, 1.0),
        dark:  (0.090, 0.086, 0.110, 1.0)
    )

    // MARK: - Type
    //
    // SF Pro and SF Mono rather than the IBM Plex pairing the mockups used:
    // Plex would have to be bundled, and SF Mono already carries the same
    // technical register while matching every system control around it.

    static func label(_ size: CGFloat = 9) -> Font {
        .system(size: size, weight: .semibold, design: .monospaced)
    }
    static func mono(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
    static func text(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight)
    }

    // MARK: - Metrics

    static let corner: CGFloat = 8
    static let railWidth: CGFloat = 208
    static let hairline: CGFloat = 1
}

/// An uppercase monospaced section label — the recurring "TIME / SPEAKER /
/// TRANSCRIPT" register used throughout the redesign.
struct StudioLabel: View {
    let text: String
    var body: some View {
        Text(text.uppercased())
            .font(Studio.label())
            .tracking(1.4)
            .foregroundColor(Studio.mute)
    }
}
