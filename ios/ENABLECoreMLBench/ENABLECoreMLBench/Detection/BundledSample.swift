import CoreGraphics
import Foundation
import ImageIO

/// The two upright validation images staged with the first-device model.
nonisolated enum BundledSample: String, CaseIterable, Identifiable, Sendable {
    case positive = "positive_5731"
    case negative = "negative_5812"

    var id: String { rawValue }
    var title: String { self == .positive ? "Basketball" : "Background" }

    /// Decodes the original JPEG on the calling worker actor, without resizing.
    func load(in bundle: Bundle = .main) throws -> CGImage {
        for directory in ["TestImages", "Resources/TestImages", nil] as [String?] {
            guard
                let url = bundle.url(
                    forResource: rawValue, withExtension: "jpg", subdirectory: directory
                )
            else { continue }
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(
                    source, 0,
                    [kCGImageSourceShouldCacheImmediately: true] as CFDictionary
                )
            else {
                throw DetectorError.invalidImage(
                    "The selected sample could not be decoded.")
            }
            return image
        }
        throw DetectorError.invalidImage(
            "The selected sample is missing. Rebuild the app with its test images."
        )
    }
}
