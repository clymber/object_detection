import CoreGraphics
import CoreMedia
import Foundation

/// Checks timing boundaries, partial failures and publication after actor handoff.
@main
struct VideoFrameTimingSmokeTests {
    /// Verifies measured/default adapters, cancellation and malformed endpoints.
    static func main() async throws {
        let frame = try makeFrame()
        let engine = DetectorEngine(loader: { _ in TimingFixture(mode: .success) })
        try await engine.selectModel(.rfdetrSmall)
        let result = try await VideoFrameDetector(detector: engine).process(frame)
        precondition(result.status == .completed && result.detections.isEmpty)
        precondition(result.timings.readyMilliseconds == frame.readyMilliseconds)
        precondition(result.timings.modelLatencyMilliseconds! > 0)
        precondition(result.processingDurationMilliseconds
                     >= result.timings.modelLatencyMilliseconds!)
        precondition(result.timings.publishedMilliseconds == nil)
        // Stand in for a slow consumer/render path; detection latency stays fixed.
        try await Task.sleep(for: .milliseconds(5))
        let published = try result.published()
        precondition(published.timings.publishedMilliseconds!
                     >= result.timings.doneMilliseconds)
        precondition(published.processingDurationMilliseconds
                     == result.processingDurationMilliseconds)
        do {
            _ = try published.published()
            preconditionFailure("Publication was stamped twice")
        } catch VideoFrameTimings.ValidationError.invalidPublication {}
        let untimed = try await VideoFrameDetector(detector: UntimedTimingFixture())
            .process(frame)
        precondition(untimed.timings.modelTiming == nil)
        precondition(untimed.timings.modelLatencyMilliseconds == nil)
        for mode in [TimingFixture.Mode.runtimeFailure, .decodeFailure] {
            let failing = TimingFixture(mode: mode)
            let failed = try await VideoFrameDetector(detector: failing).process(frame)
            precondition(failed.status == .failed && failed.failure != nil)
            precondition(failed.timings.modelTiming != nil)
            precondition((failed.timings.modelTiming!.completedMilliseconds == nil)
                         == (mode == .runtimeFailure))
            precondition(failed.timings.publishedMilliseconds == nil)
            do {
                _ = try failed.published()
                preconditionFailure("Inference failure counted as a published result")
            } catch VideoFrameTimings.ValidationError.invalidPublication {}
            let failingEngine = DetectorEngine(loader: { _ in failing })
            try await failingEngine.selectModel(.rfdetrSmall)
            do {
                _ = try await failingEngine.detect(in: frame.image)
                preconditionFailure("Legacy API swallowed an inference error")
            } catch TimingFixtureError.inference {}
        }
        try testInvalidEndpoints()
        try await testInFlightCancellation(frame)
        print("Video timing: model/E2E, publication and partial errors/cancel passed.")
    }

    /// Rejects fabricated timing, including negative, nonfinite and reordered values.
    private static func testInvalidEndpoints() throws {
        for timing in [
            ModelPredictionTiming(submittedMilliseconds: 9, completedMilliseconds: 15),
            ModelPredictionTiming(submittedMilliseconds: 12, completedMilliseconds: 11),
            ModelPredictionTiming(submittedMilliseconds: 12, completedMilliseconds: 31),
            ModelPredictionTiming(
                submittedMilliseconds: .nan, completedMilliseconds: nil),
        ] {
            do {
                _ = try VideoFrameTimings(
                    readyMilliseconds: 10, modelTiming: timing, doneMilliseconds: 30)
                preconditionFailure("Invalid model timing was accepted")
            } catch VideoFrameTimings.ValidationError.invalidEndpoints {}
        }
        for publication in [-1.0, 29, .infinity] {
            do {
                _ = try VideoFrameTimings(
                    readyMilliseconds: 10, modelTiming: nil,
                    doneMilliseconds: 30, publishedMilliseconds: publication)
                preconditionFailure("Invalid publication time was accepted")
            } catch VideoFrameTimings.ValidationError.invalidEndpoints {}
        }
    }

    /// Cancels after inference starts and rejects its eventually completed result.
    private static func testInFlightCancellation(
        _ frame: UprightVideoFrame
    ) async throws {
        let gate = TimingGate()
        let detector = VideoFrameDetector(detector: gate)
        let pending = Task { try await detector.process(frame) }
        await gate.waitForRequest()
        pending.cancel()
        await gate.finish()
        do {
            _ = try await pending.value
            preconditionFailure("Cancelled in-flight result was retained")
        } catch is CancellationError {}
    }

    /// Creates an input whose host time is independent of its source PTS.
    private static func makeFrame() throws -> UprightVideoFrame {
        let image = CGContext(
            data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 8,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!.makeImage()!
        return UprightVideoFrame(
            metadata: try VideoFrameMetadata(
                sourceID: "timing-fixture", sourceFrameIndex: 100,
                presentationTime: CMTime(value: 100100, timescale: 30000),
                width: 2, height: 2), image: image, bufferToImage: .identity)
    }
}

/// Supplies known timing boundaries and partial failures through an engine actor.
private actor TimingFixture: ObjectDetector {
    enum Mode { case success, runtimeFailure, decodeFailure }
    let mode: Mode

    /// Selects whether model execution or subsequent decoding fails.
    init(mode: Mode) { self.mode = mode }

    /// Preserves the original unwrapped error contract for unmeasured callers.
    func detect(in image: CGImage) throws -> [Detection] {
        guard mode == .success else { throw TimingFixtureError.inference }
        return []
    }

    /// Marks execution/readback boundaries before any simulated decoding failure.
    func detectMeasured(in image: CGImage) async throws -> DetectorPrediction {
        let submitted = InferenceClock.nowMilliseconds()
        try await Task.sleep(for: .milliseconds(5))
        let timing = ModelPredictionTiming(
            submittedMilliseconds: submitted,
            completedMilliseconds: mode == .runtimeFailure
                ? nil : InferenceClock.nowMilliseconds())
        guard mode == .success else {
            throw MeasuredPredictionFailure(
                underlying: TimingFixtureError.inference, modelTiming: timing)
        }
        return DetectorPrediction(detections: [], modelTiming: timing)
    }
}

/// Verifies that the default protocol implementation reports missing model timing.
private actor UntimedTimingFixture: ObjectDetector {
    /// Returns a valid empty result without claiming to measure backend execution.
    func detect(in image: CGImage) -> [Detection] { [] }
}

/// Provides a typed error that must survive the legacy engine interface.
nonisolated private enum TimingFixtureError: Error { case inference }

/// Controls an in-flight prediction without racing a sleep against cancellation.
private actor TimingGate: ObjectDetector {
    private var pending: CheckedContinuation<[Detection], Never>?
    private var notification: CheckedContinuation<Void, Never>?

    /// Holds a prediction until the check has cancelled its caller.
    func detect(in image: CGImage) async -> [Detection] {
        await withCheckedContinuation {
            pending = $0
            notification?.resume()
            notification = nil
        }
    }

    /// Signals exactly when inference has begun.
    func waitForRequest() async {
        if pending != nil { return }
        await withCheckedContinuation { notification = $0 }
    }

    /// Releases inference so the caller can reject the cancelled result.
    func finish() {
        pending?.resume(returning: [])
        pending = nil
    }
}
