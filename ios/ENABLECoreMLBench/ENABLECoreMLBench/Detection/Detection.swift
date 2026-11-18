import CoreGraphics

/// A model-independent detection in the upright source image's pixel coordinates.
///
/// `boundingBox` uses top-left-origin (x, y, width, height), matching the saved
/// Python COCO predictions. Coordinates remain unclipped and can extend outside
/// the image. Preview clipping and any export clipping are separate operations.
/// Here, normalized means a shared representation, not coordinates scaled to 0...1.
nonisolated struct Detection: Equatable, Sendable {
    /// A finite rectangle with positive dimensions in source pixels.
    let boundingBox: CGRect

    /// The model's confidence in 0...1, without thresholding or rounding.
    let confidence: Double

    /// The source dataset category ID, after model-specific class mapping.
    /// Basketball adapters map foreground index 0 to source category 1.
    let classID: Int

    /// Identifies malformed adapter output without silently repairing it.
    enum ValidationError: Error, Equatable {
        case invalidBoundingBox
        case invalidConfidence
        case invalidClassID
    }

    /// Validates decoded output while preserving its coordinates and score.
    init(boundingBox: CGRect, confidence: Double, classID: Int) throws {
        let components = [
            boundingBox.origin.x, boundingBox.origin.y,
            boundingBox.size.width, boundingBox.size.height,
            boundingBox.origin.x + boundingBox.size.width,
            boundingBox.origin.y + boundingBox.size.height,
        ]
        guard components.allSatisfy({ $0.isFinite }),
              boundingBox.size.width > 0, boundingBox.size.height > 0 else {
            throw ValidationError.invalidBoundingBox
        }
        guard confidence.isFinite, (0...1).contains(confidence) else {
            throw ValidationError.invalidConfidence
        }
        guard classID >= 0 else {
            throw ValidationError.invalidClassID
        }

        self.boundingBox = boundingBox
        self.confidence = confidence
        self.classID = classID
    }
}
