import Foundation
import StoreKit

/// Whether the Music and Image tabs are offered: everywhere but the
/// China mainland App Store.
///
/// There, every feature that generates content needs a CAC filing for the
/// model behind it, cited in App Store Connect (its China mainland
/// compliance declarations, due 31 January 2027). On-device models have no
/// such filing, and a downloaded model counts as offered, so music and
/// pictures — and with them Enhance prompt and the lyrics finder — are
/// turned off there and the declarations answered "No". Transcription,
/// Live, translation and speech analysis stay: they don't generate content.
///
/// Decided by the App Store account's storefront, not the device's region
/// or language. The last one seen is kept, so the tabs are right from the
/// first frame of the next launch, offline included.
@MainActor @Observable
final class GenerationAvailability {
    static let shared = GenerationAvailability()

    /// ISO 3166-1 alpha-3, as StoreKit gives it.
    private static let blockedStorefronts: Set<String> = ["CHN"]
    private static let key = "StorefrontCountry"

    private(set) var offersGeneration: Bool

    private init() {
        offersGeneration = !Self.blockedStorefronts.contains(Self.override ?? UserDefaults.standard.string(forKey: Self.key) ?? "")
    }

    /// Debug builds: `-DebugStorefront CHN` pretends to be on that store.
    private static var override: String? {
        #if DEBUG
        return UserDefaults.standard.string(forKey: "DebugStorefront")
        #else
        return nil
        #endif
    }

    /// Follows the storefront for as long as the app runs (the account can
    /// change while it's open).
    func watch() async {
        guard Self.override == nil else { return }
        if let storefront = await Storefront.current { apply(storefront.countryCode) }
        for await storefront in Storefront.updates { apply(storefront.countryCode) }
    }

    private func apply(_ country: String) {
        UserDefaults.standard.set(country, forKey: Self.key)
        let offers = !Self.blockedStorefronts.contains(country)
        if offers != offersGeneration { offersGeneration = offers }
    }
}
