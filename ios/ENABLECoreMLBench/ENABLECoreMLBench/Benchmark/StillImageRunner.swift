import CoreGraphics
import Foundation

/// One completed image's displayed detections and measured detection latency.
nonisolated struct StillImageResult: Sendable {
    let detections: [Detection]
    let latencyMilliseconds: Double
}

/// Allows the session to use worker actors for image loading and inference.
nonisolated protocol StillImageProcessing: Actor {
    /// Loads one upright sample outside the UI actor.
    func load(_ sample: BundledSample) async throws -> CGImage

    /// Loads an imported photo and resolves its orientation on the worker actor.
    func loadPhoto(_ url: URL) async throws -> CGImage

    /// Runs one image after loading the selected model.
    func run(_ image: CGImage, model: DetectorModel) async throws -> StillImageResult
}

extension StillImageProcessing {
    /// Shares the real photo decoder with actor-based workers and test doubles.
    func loadPhoto(_ url: URL) throws -> CGImage {
        try StillImagePhotoDecoder.load(url)
    }
}

/// Keeps image decoding, model loading and detection independent of SwiftUI.
actor StillImageRunner: StillImageProcessing {
    private let engine: DetectorEngine
    private let bundle: Bundle

    /// Uses app resources by default and permits explicit resources for host checks.
    init(engine: DetectorEngine = DetectorEngine(), bundle: Bundle = .main) {
        self.engine = engine
        self.bundle = bundle
    }

    /// Decodes one bundled JPEG without blocking the main actor.
    func load(_ sample: BundledSample) throws -> CGImage {
        try Task.checkCancellation()
        return try sample.load(in: bundle)
    }

    /// Times preprocessing/inference/decoding, excluding model load and rendering.
    func run(_ image: CGImage, model: DetectorModel) async throws -> StillImageResult {
        try await engine.selectModel(model)
        try Task.checkCancellation()
        let clock = ContinuousClock()
        let start = clock.now
        let candidates = try await engine.detect(in: image)
        try Task.checkCancellation()
        let retained = candidates.enumerated().filter { $0.element.confidence >= 0.25 }
            .sorted {
                $0.element.confidence == $1.element.confidence
                    ? $0.offset < $1.offset
                    : $0.element.confidence > $1.element.confidence
            }.prefix(100).map(\.element)
        let elapsed = start.duration(to: clock.now).components
        return StillImageResult(
            detections: retained,
            latencyMilliseconds: Double(elapsed.seconds) * 1000
                + Double(elapsed.attoseconds) / 1e15
        )
    }
}
