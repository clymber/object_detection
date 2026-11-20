import CoreGraphics
import Foundation

/// Reuses the selected detector's transforms/decoders for one upright video frame.
actor VideoFrameDetector {
    private let detector: any ObjectDetector

    /// Accepts a preloaded engine or adapter, keeping model load outside frame timing.
    init(detector: any ObjectDetector) { self.detector = detector }

    /// Applies the protocol selection without changing source boxes or category IDs.
    func detect(_ frame: UprightVideoFrame) async throws -> [Detection] {
        try Task.checkCancellation()
        let candidates = try await detector.detect(in: frame.image)
        try Task.checkCancellation()
        let retained = Self.retained(candidates)
        try Task.checkCancellation()
        return retained
    }

    /// Records success/empty/failure; cancellation propagates for the run to classify.
    func process(_ frame: UprightVideoFrame) async throws -> VideoFrameResult {
        try Task.checkCancellation()
        let detections: [Detection]
        let failure: VideoInferenceFailure?
        let modelTiming: ModelPredictionTiming?
        do {
            let prediction = try await detector.detectMeasured(in: frame.image)
            try Task.checkCancellation()
            detections = Self.retained(prediction.detections)
            modelTiming = prediction.modelTiming
            failure = nil
        } catch {
            if error is CancellationError || Task.isCancelled {
                throw CancellationError()
            }
            detections = []
            if let measured = error as? MeasuredPredictionFailure {
                modelTiming = measured.modelTiming
                failure = VideoInferenceFailure(measured.underlying)
            } else {
                modelTiming = nil
                failure = VideoInferenceFailure(error)
            }
        }
        try Task.checkCancellation()
        let timings = try VideoFrameTimings(
            readyMilliseconds: frame.readyMilliseconds, modelTiming: modelTiming,
            doneMilliseconds: InferenceClock.nowMilliseconds())
        return VideoFrameResult(
            metadata: frame.metadata,
            sourceWidth: frame.image.width, sourceHeight: frame.image.height,
            bufferToImage: frame.bufferToImage,
            status: failure == nil ? .completed : .failed,
            detections: detections, failure: failure, timings: timings)
    }

    /// Preserves candidate order for confidence ties and keeps unrounded geometry.
    private static func retained(_ candidates: [Detection]) -> [Detection] {
        candidates.enumerated().filter { $0.element.confidence >= 0.25 }
            .sorted {
                $0.element.confidence == $1.element.confidence
                    ? $0.offset < $1.offset
                    : $0.element.confidence > $1.element.confidence
            }.prefix(100).map(\.element)
    }
}
