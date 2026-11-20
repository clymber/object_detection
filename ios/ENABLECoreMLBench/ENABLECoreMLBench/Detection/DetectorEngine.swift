import CoreGraphics
import CoreML
import Foundation

/// Switches the active adapter while preserving one image-to-detections API.
actor DetectorEngine: ObjectDetector {
    typealias Loader = @Sendable (DetectorModel) async throws -> any ObjectDetector
    private let loader: Loader
    private var detector: (any ObjectDetector)?
    private var generation: UInt64 = 0
    private(set) var selectedModel: DetectorModel?

    /// Resolves explicit artifact URLs or distinctly named app model resources.
    init(
        modelURLs: [DetectorModel: URL] = [:], bundle: Bundle = .main,
        computeUnits: MLComputeUnits = .all
    ) {
        self.loader = { definition in
            let url = try modelURLs[definition] ?? definition.bundledURL(in: bundle)
            return try await CoreMLObjectDetector(
                definition: definition, modelURL: url, computeUnits: computeUnits
            )
        }
    }

    /// Accepts an alternate loader for deterministic lifecycle checks.
    init(loader: @escaping Loader) {
        self.loader = loader
    }

    /// Replaces the active adapter only after the latest requested load succeeds.
    /// A failed switch leaves the previous working selection available.
    func selectModel(_ definition: DetectorModel) async throws {
        generation &+= 1
        let request = generation
        try Task.checkCancellation()
        if selectedModel == definition { return }
        let loaded = try await loader(definition)
        try Task.checkCancellation()
        guard request == generation else { throw DetectorError.selectionSuperseded }
        detector = loaded
        selectedModel = definition
    }

    /// Rejects stale in-flight results if the selection changes during inference.
    func detect(in image: CGImage) async throws -> [Detection] {
        do {
            return try await detectMeasured(in: image).detections
        } catch let failure as MeasuredPredictionFailure {
            throw failure.underlying
        }
    }

    /// Preserves adapter timing and the same generation/cancellation guarantees.
    func detectMeasured(in image: CGImage) async throws -> DetectorPrediction {
        try Task.checkCancellation()
        guard let detector else { throw DetectorError.noModelSelected }
        let request = generation
        let results = try await detector.detectMeasured(in: image)
        try Task.checkCancellation()
        guard request == generation else { throw DetectorError.selectionSuperseded }
        return results
    }
}
