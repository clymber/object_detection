import CoreGraphics
import Foundation

/// Checks model-switch ordering and failure recovery without loading model weights.
@main
struct DetectorEngineSmokeTests {
    /// Completes loads out of order and verifies that the latest request wins.
    static func main() async throws {
        let gate = LoadGate()
        let engine = DetectorEngine(loader: { model in await gate.load(model) })
        let old = Task { try await engine.selectModel(.rfdetrSmall) }
        await gate.waitForRequest(.rfdetrSmall)
        let latest = Task { try await engine.selectModel(.rfdetrNano) }
        await gate.waitForRequest(.rfdetrNano)
        await gate.finish(.rfdetrNano)
        try await latest.value
        await gate.finish(.rfdetrSmall)
        do {
            try await old.value
            preconditionFailure("An outdated model load replaced the latest selection")
        } catch DetectorError.selectionSuperseded {}
        let selected = await engine.selectedModel
        precondition(selected == .rfdetrNano)

        let recoverable = DetectorEngine(loader: { model in
            if model == .yoloxNano { throw DetectorError.missingModel(model.rawValue) }
            return EmptyDetector()
        })
        try await recoverable.selectModel(.rfdetrSmall)
        do {
            try await recoverable.selectModel(.yoloxNano)
            preconditionFailure("Failed model load was accepted")
        } catch DetectorError.missingModel {}
        let recovered = await recoverable.selectedModel
        precondition(recovered == .rfdetrSmall)
        try await testStaleInference()
        try await testCancelledLoad()
        print(
            "Detector engine: load ordering, recovery, stale inference and cancellation passed."
        )
    }

    /// Discards results produced by a model that was replaced during inference.
    private static func testStaleInference() async throws {
        let slow = GatedDetector()
        let engine = DetectorEngine(loader: { model in
            if model == .rfdetrSmall { return slow }
            return EmptyDetector()
        })
        let context = CGContext(
            data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 8,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        let image = context.makeImage()!
        do {
            _ = try await engine.detect(in: image)
            preconditionFailure("Inference succeeded without a selected model")
        } catch DetectorError.noModelSelected {}
        try await engine.selectModel(.rfdetrSmall)
        let pending = Task { try await engine.detect(in: image) }
        await slow.waitForInference()
        try await engine.selectModel(.rfdetrNano)
        await slow.finish()
        do {
            _ = try await pending.value
            preconditionFailure("Stale inference was published")
        } catch DetectorError.selectionSuperseded {}
    }

    /// Prevents a cancelled load from becoming the active model.
    private static func testCancelledLoad() async throws {
        let gate = LoadGate()
        let engine = DetectorEngine(loader: { model in await gate.load(model) })
        let pending = Task { try await engine.selectModel(.rfdetrSmall) }
        await gate.waitForRequest(.rfdetrSmall)
        pending.cancel()
        await gate.finish(.rfdetrSmall)
        do {
            try await pending.value
            preconditionFailure("Cancelled model load was activated")
        } catch is CancellationError {}
        let selected = await engine.selectedModel
        precondition(selected == nil)
    }
}

/// Suspends loads so tests can control their completion order without sleeps.
private actor LoadGate {
    private var pending:
        [DetectorModel: CheckedContinuation<any ObjectDetector, Never>] = [:]
    private var notifications: [DetectorModel: CheckedContinuation<Void, Never>] = [:]

    /// Stores the load continuation and announces the pending request.
    func load(_ model: DetectorModel) async -> any ObjectDetector {
        await withCheckedContinuation { continuation in
            pending[model] = continuation
            notifications.removeValue(forKey: model)?.resume()
        }
    }

    /// Waits until a particular model load has started.
    func waitForRequest(_ model: DetectorModel) async {
        if pending[model] != nil { return }
        await withCheckedContinuation { notifications[model] = $0 }
    }

    /// Completes a pending load with an inert worker adapter.
    func finish(_ model: DetectorModel) {
        pending.removeValue(forKey: model)!.resume(returning: EmptyDetector())
    }
}

/// Supplies a successful empty-image response for engine lifecycle tests.
private actor EmptyDetector: ObjectDetector {
    /// Returns no objects without requiring a model runtime.
    func detect(in image: CGImage) async throws -> [Detection] { [] }
}

/// Holds a prediction while a test replaces the active model.
private actor GatedDetector: ObjectDetector {
    private var pending: CheckedContinuation<[Detection], Never>?
    private var notification: CheckedContinuation<Void, Never>?

    /// Suspends until the test explicitly completes the inference.
    func detect(in image: CGImage) async throws -> [Detection] {
        await withCheckedContinuation { continuation in
            pending = continuation
            notification?.resume()
            notification = nil
        }
    }

    /// Waits for a prediction to reach the controlled suspension point.
    func waitForInference() async {
        if pending != nil { return }
        await withCheckedContinuation { notification = $0 }
    }

    /// Completes the held prediction with an empty-image response.
    func finish() {
        pending?.resume(returning: [])
        pending = nil
    }
}
