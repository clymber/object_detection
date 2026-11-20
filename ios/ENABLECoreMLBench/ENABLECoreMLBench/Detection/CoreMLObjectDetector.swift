import CoreGraphics
import CoreML
import Foundation

/// Uses one shared inference pipeline with model-specific preprocessing/decoding.
actor CoreMLObjectDetector: ObjectDetector {
    private let definition: DetectorModel
    private let runtime: CoreMLRuntime

    /// Loads and verifies the selected artifact before accepting images.
    init(
        definition: DetectorModel, modelURL: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws {
        self.definition = definition
        self.runtime = try await CoreMLRuntime(
            definition: definition, modelURL: modelURL, computeUnits: computeUnits
        )
    }

    /// Preprocesses, predicts and decodes on worker actors without UI dependencies.
    func detect(in image: CGImage) async throws -> [Detection] {
        do {
            return try await detectMeasured(in: image).detections
        } catch let failure as MeasuredPredictionFailure {
            throw failure.underlying
        }
    }

    /// Excludes preprocessing/decoding from model time, including runtime readback.
    func detectMeasured(in image: CGImage) async throws -> DetectorPrediction {
        try Task.checkCancellation()
        let prepared = try ImagePreprocessor.prepare(RGBImage(image), for: definition)
        let submitted = InferenceClock.nowMilliseconds()
        var completed: Double?
        do {
            let outputs = try await runtime.predict(prepared.tensor)
            completed = InferenceClock.nowMilliseconds()
            try Task.checkCancellation()
            let detections = try decode(outputs, prepared: prepared)
            try Task.checkCancellation()
            return DetectorPrediction(
                detections: detections,
                modelTiming: ModelPredictionTiming(
                    submittedMilliseconds: submitted, completedMilliseconds: completed))
        } catch {
            if error is CancellationError || Task.isCancelled {
                throw CancellationError()
            }
            throw MeasuredPredictionFailure(
                underlying: error,
                modelTiming: ModelPredictionTiming(
                    submittedMilliseconds: submitted, completedMilliseconds: completed))
        }
    }

    /// Uses the same model-specific geometry and postprocessing as image detection.
    private func decode(
        _ outputs: [String: FloatTensor], prepared: PreparedImage
    ) throws -> [Detection] {
        if definition.isRFDETR {
            return try RFDETRDecoder.decode(
                outputs, model: definition,
                sourceWidth: prepared.sourceWidth, sourceHeight: prepared.sourceHeight
            )
        }
        guard let output = outputs["output"] else {
            throw DetectorError.invalidTensor("missing YOLOX output")
        }
        return try YOLOXDecoder.decode(output, scale: prepared.scale)
    }
}
