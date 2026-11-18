import CoreGraphics
import Foundation
import ImageIO

/// Decodes personal photos into upright source pixels for preview and inference.
nonisolated enum StillImagePhotoDecoder {
    /// Applies EXIF orientation while preserving the full source dimensions.
    static func load(_ url: URL) throws -> CGImage {
        try Task.checkCancellation()
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
            let height = properties[kCGImagePropertyPixelHeight] as? NSNumber,
            width.intValue > 0, height.intValue > 0
        else {
            throw DetectorError.invalidImage("The selected photo could not be read.")
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width.intValue, height.intValue),
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard
            let image = CGImageSourceCreateThumbnailAtIndex(
                source, 0, options as CFDictionary)
        else {
            throw DetectorError.invalidImage("The selected photo could not be decoded.")
        }
        try Task.checkCancellation()
        return image
    }
}
