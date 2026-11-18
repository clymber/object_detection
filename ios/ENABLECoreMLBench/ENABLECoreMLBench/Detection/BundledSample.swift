import CoreGraphics
import Foundation
import ImageIO

/// The 25 test-split examples, retaining the notebook's original eight images.
nonisolated enum BundledSample: String, CaseIterable, Identifiable, Sendable {
    case testPositive6988 = "test_positive_6988"
    case testPositive6989 = "test_positive_6989"
    case testPositive6990 = "test_positive_6990"
    case testPositive6991 = "test_positive_6991"
    case testPositive6992 = "test_positive_6992"
    case testPositive6993 = "test_positive_6993"
    case testPositive6994 = "test_positive_6994"
    case testPositive6995 = "test_positive_6995"
    case testPositive6996 = "test_positive_6996"
    case testPositive6997 = "test_positive_6997"
    case testPositive6998 = "test_positive_6998"
    case testPositive6999 = "test_positive_6999"
    case testPositive7000 = "test_positive_7000"
    case testPositive7001 = "test_positive_7001"
    case testPositive7002 = "test_positive_7002"
    case testPositive7003 = "test_positive_7003"
    case testPositive7004 = "test_positive_7004"
    case testPositive7005 = "test_positive_7005"
    case testPositive7006 = "test_positive_7006"
    case testPositive7007 = "test_positive_7007"
    case testNegative7028 = "test_negative_7028"
    case testNegative7030 = "test_negative_7030"
    case testNegative7032 = "test_negative_7032"
    case testNegative7046 = "test_negative_7046"
    case testNegative7054 = "test_negative_7054"

    var id: String { rawValue }
    var isPositive: Bool { rawValue.contains("_positive_") }
    var title: String {
        "\(isPositive ? "Basketball" : "Background") · \(rawValue.split(separator: "_").last!)"
    }

    /// Decodes the original JPEG on the calling worker actor, without resizing.
    func load(in bundle: Bundle = .main) throws -> CGImage {
        let url = try resourceURL(in: bundle)
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

    /// Resolves the same original bytes for loading and export identity checks.
    func resourceURL(in bundle: Bundle = .main) throws -> URL {
        for directory in ["TestImages", "Resources/TestImages", nil] as [String?] {
            guard
                let url = bundle.url(
                    forResource: rawValue, withExtension: "jpg", subdirectory: directory
                )
            else { continue }
            return url
        }
        throw DetectorError.invalidImage(
            "The selected sample is missing. Rebuild the app with its test images."
        )
    }
}
