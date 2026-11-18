import CoreGraphics
import Foundation

/// Exercises actual session state transitions with a controllable worker actor.
@main
@MainActor
struct StillImageSessionSmokeTests {
    /// Checks successful/empty runs, retries, duplicate Start and stale completion.
    static func main() async throws {
        let worker = SessionWorker()
        let session = StillImageSession(worker: worker)
        session.start()
        precondition(session.status == .error && !session.canStart)
        session.load(.positive)
        try await waitForIdle(session)
        precondition(session.canStart && session.status == .idle)
        session.start()
        session.start()
        try await waitForIdle(session)
        precondition(session.status == .completed && session.detections.count == 1)
        precondition(
            session.processedImages == 1 && session.latencyMilliseconds == 12.5)
        let runCount = await worker.runCount
        precondition(runCount == 1, "Duplicate Start ran another prediction")

        session.load(.negative)
        precondition(session.detections.isEmpty && session.latencyMilliseconds == nil)
        precondition(session.processedImages == 0)
        try await waitForIdle(session)
        session.start()
        try await waitForIdle(session)
        precondition(session.status == .completed && session.detections.isEmpty)
        precondition(session.processedImages == 1)

        await worker.failNextPrediction()
        session.start()
        try await waitForIdle(session)
        precondition(session.status == .error && session.processedImages == 0)
        session.start()
        try await waitForIdle(session)
        precondition(session.status == .completed)

        session.load(.positive)
        try await waitForIdle(session)
        await worker.holdNextPrediction()
        session.start()
        await worker.waitForPrediction()
        session.stop()
        precondition(session.status == .idle && session.processedImages == 0)
        session.load(.negative)
        try await waitForIdle(session)
        session.start()
        try await waitForIdle(session)
        await worker.finishHeldPrediction()
        try await Task.sleep(for: .milliseconds(20))
        precondition(session.status == .completed && session.detections.isEmpty)
        precondition(session.image?.width == 3 && session.processedImages == 1)
        print(
            "Still-image session: success, empty image, retry, duplicate Start and stale cancellation passed."
        )
    }

    /// Bounds asynchronous checks so a broken session fails instead of hanging.
    private static func waitForIdle(_ session: StillImageSession) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while session.isBusy && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        precondition(!session.isBusy, "Session did not finish")
    }
}

/// Returns known image results and can fail or hold an inference on demand.
private actor SessionWorker: StillImageProcessing {
    private(set) var runCount = 0
    private var shouldFail = false
    private var shouldHold = false
    private var pending: CheckedContinuation<StillImageResult, Never>?
    private var heldResult: StillImageResult?
    private var notification: CheckedContinuation<Void, Never>?

    /// Uses image width to distinguish positive from negative input in checks.
    func load(_ sample: BundledSample) throws -> CGImage {
        let width = sample == .positive ? 2 : 3
        let context = CGContext(
            data: nil, width: width, height: 2, bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        return context.makeImage()!
    }

    /// Produces a result whose content remains tied to the submitted image.
    func run(_ image: CGImage, model: DetectorModel) async throws -> StillImageResult {
        runCount += 1
        if shouldFail {
            shouldFail = false
            throw DetectorError.invalidModel("Controlled load failure")
        }
        let rows =
            image.width == 2
            ? [
                try Detection(
                    boundingBox: CGRect(x: 0, y: 0, width: 1, height: 1),
                    confidence: 0.9, classID: 1
                )
            ] : []
        let result = StillImageResult(detections: rows, latencyMilliseconds: 12.5)
        if shouldHold {
            shouldHold = false
            heldResult = result
            return await withCheckedContinuation {
                pending = $0
                notification?.resume()
                notification = nil
            }
        }
        return result
    }

    /// Makes the next operation fail before publishing any image results.
    func failNextPrediction() { shouldFail = true }

    /// Suspends the next prediction until the test explicitly completes it.
    func holdNextPrediction() { shouldHold = true }

    /// Waits until the held prediction has started.
    func waitForPrediction() async {
        if pending != nil { return }
        await withCheckedContinuation { notification = $0 }
    }

    /// Returns the older positive result after a newer negative result completes.
    func finishHeldPrediction() {
        pending?.resume(returning: heldResult!)
        pending = nil
        heldResult = nil
    }
}
