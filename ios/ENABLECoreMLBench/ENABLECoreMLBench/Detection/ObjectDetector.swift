import CoreGraphics

/// A model adapter whose mutable inference state belongs to its own actor.
///
/// Each adapter owns its preprocessing, inference, decoding and class mapping.
/// Implementations must keep that work off the UI's main actor and preserve
/// source-coordinate boxes for comparison with the Python reference.
nonisolated protocol ObjectDetector: Actor {
    /// Detects objects in one upright, decoded RGB source image.
    ///
    /// The caller resolves image orientation before submission. Return an empty
    /// array for a successful image with no detections; throw for inference or
    /// decoding failures. Order results by descending confidence, preserving
    /// candidate order for ties. Model adapters own filtering/NMS; this interface
    /// applies no additional thresholds, clipping or coordinate conversions.
    func detect(in image: CGImage) async throws -> [Detection]

    /// Reports backend timing when the adapter can measure its exact boundaries.
    func detectMeasured(in image: CGImage) async throws -> DetectorPrediction
}

extension ObjectDetector {
    /// Keeps existing adapters usable and marks unavailable model timing explicitly.
    func detectMeasured(in image: CGImage) async throws -> DetectorPrediction {
        DetectorPrediction(detections: try await detect(in: image), modelTiming: nil)
    }
}
