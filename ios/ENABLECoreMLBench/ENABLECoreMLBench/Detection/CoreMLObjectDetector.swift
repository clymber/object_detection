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
        try Task.checkCancellation()
        let prepared = try ImagePreprocessor.prepare(RGBImage(image), for: definition)
        let outputs = try await runtime.predict(prepared.tensor)
        try Task.checkCancellation()
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
