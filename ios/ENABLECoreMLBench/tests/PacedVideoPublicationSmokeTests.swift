import CoreGraphics
import CoreMedia
import Foundation

/// Checks delivery-time freshness independently of inference completion and UI draws.
@main
@MainActor
struct PacedVideoPublicationSmokeTests {
    /// Exercises exact boundaries, slow consumers, one-shot receipts and cancellation.
    static func main() async throws {
        try await testReceiptBoundaries()
        try await testLateCompletion()
        try await testBlockedConsumerKeepsInferenceRunning()
        try await testPendingPublicationBecomesLate()
        try await testCancellationDuringDelivery(cancelTask: false)
        try await testCancellationDuringDelivery(cancelTask: true)
        try await testActiveInferenceFinishesAfterStop()
        try await testReceiptBeforeStopPersists()
        try await testTicketLifetimeAndConsumerErrors()
        try await testFailedAndSkippedFrames()
        try await testInvalidReceiptClock()
        try await testPreviewReceipt()
        print("Paced publication: 12 age, receipt, preview and cancellation checks passed.")
    }

    /// Rechecks on MainActor after its hop, including exactly 200 ms and empty results.
    private static func testReceiptBoundaries() async throws {
        for receiptTime in [Double(1200).nextDown, 1200, Double(1200).nextUp] {
            let clock = PublicationClock()
            let recorder = PublicationRecorder()
            let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) { frame in
                clock.set(1100)
                return try result(frame, done: 1100, modelTiming: ModelPredictionTiming(
                    submittedMilliseconds: 1010, completedMilliseconds: 1090))
            }
            try queue.offer(frame(index: 0, now: 1000))
            queue.finishInput()
            let snapshot = try await queue.run { ticket in
                try await MainActor.run {
                    clock.set(receiptTime)
                    try recorder.receive(ticket)
                }
            }
            let outcome = snapshot.outcomes[0]
            precondition(outcome.status == .completed
                         && outcome.completedAgeMilliseconds == 100)
            precondition(outcome.result?.timings.doneMilliseconds == 1100)
            precondition(outcome.result?.timings.modelLatencyMilliseconds == 80)
            precondition(outcome.result?.timings.endToEndMilliseconds == 100)
            precondition(outcome.result?.timings.publishedMilliseconds == nil)
            precondition(outcome.publication.checkedMilliseconds == receiptTime)
            if receiptTime <= 1200 {
                precondition(outcome.publication.status == .published)
                precondition(outcome.publishedAgeMilliseconds == receiptTime - 1000)
                precondition(recorder.received.count == 1)
                precondition(recorder.received[0].detections.isEmpty)
                precondition(recorder.received[0].timings.publishedMilliseconds
                             == receiptTime)
            } else {
                precondition(outcome.publication.status == .late)
                precondition(outcome.publishedAgeMilliseconds == nil)
                precondition(recorder.received.isEmpty)
            }
        }
    }

    /// A late active request completes normally but is never sent to a result consumer.
    private static func testLateCompletion() async throws {
        let clock = PublicationClock()
        let recorder = PublicationRecorder()
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) { frame in
            clock.set(Double(1200).nextUp)
            return try result(frame, done: clock.now)
        }
        try queue.offer(frame(index: 0, now: 1000))
        queue.finishInput()
        let snapshot = try await queue.run { ticket in
            try await recorder.receive(ticket)
        }
        precondition(snapshot.outcomes[0].status == .completed)
        precondition(snapshot.outcomes[0].completedAgeMilliseconds! > 200)
        precondition(snapshot.outcomes[0].publication.status == .late)
        precondition(snapshot.outcomes[0].publication.publishedMilliseconds == nil)
        precondition(recorder.calls == 0)
    }

    /// Model work continues while delivery is held, retaining only one pending result.
    private static func testBlockedConsumerKeepsInferenceRunning() async throws {
        let clock = PublicationClock()
        let receiver = HeldPublicationReceiver()
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try result($0, done: clock.now)
        }
        let worker = Task { try await queue.run { try await receiver.receive($0) } }
        try queue.offer(frame(index: 0, now: 1000))
        try await waitForReceiver(receiver)
        for index in 1...2 {
            clock.set(1000 + Double(index) * 50)
            try queue.offer(frame(index: index, now: clock.now))
            try await waitUntil { queue.snapshot.outcomes.count == index + 1 }
        }
        let held = queue.snapshot
        precondition(held.outcomes.count == 3 && held.activeFrames == 0)
        precondition(held.activePublications == 1 && held.pendingPublications == 1)
        precondition(held.outcomes[1].status == .completed)
        precondition(held.outcomes[1].publication.status == .superseded)
        precondition(held.outcomes[1].publication.publishedMilliseconds == nil)
        queue.finishInput()
        await receiver.release()
        let snapshot = try await worker.value
        let received = await receiver.received
        precondition(received.map { $0.metadata.sourceFrameIndex } == [0, 2])
        precondition(snapshot.outcomes.map { $0.publication.status }
                     == [.published, .superseded, .published])
        precondition(snapshot.activePublications == 0
                     && snapshot.pendingPublications == 0)
    }

    /// A waiting result is discarded if it becomes late behind a delayed delivery.
    private static func testPendingPublicationBecomesLate() async throws {
        let clock = PublicationClock()
        let receiver = HeldPublicationReceiver()
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try result($0, done: clock.now)
        }
        let worker = Task { try await queue.run { try await receiver.receive($0) } }
        try queue.offer(frame(index: 0, now: 1000))
        try await waitForReceiver(receiver)
        clock.set(1050)
        try queue.offer(frame(index: 1, now: 1050))
        try await waitUntil { queue.snapshot.outcomes.count == 2 }
        clock.set(1300)
        queue.finishInput()
        await receiver.release()
        let snapshot = try await worker.value
        let calls = await receiver.calls
        let received = await receiver.received
        precondition(calls == 1 && received.isEmpty)
        precondition(snapshot.outcomes.allSatisfy { $0.status == .completed })
        precondition(snapshot.outcomes.allSatisfy { $0.publication.status == .late })
        precondition(snapshot.outcomes.allSatisfy {
            $0.publishedAgeMilliseconds == nil
        })
    }

    /// Stop invalidates active and pending receipts without losing completed inference.
    private static func testCancellationDuringDelivery(cancelTask: Bool) async throws {
        let clock = PublicationClock()
        let receiver = HeldPublicationReceiver()
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try result($0, done: clock.now)
        }
        let worker = Task { try await queue.run(previewEnabled: true) {
            try await receiver.receive($0)
        } }
        try queue.offer(frame(index: 0, now: 1000))
        try await waitForReceiver(receiver)
        clock.set(1050)
        try queue.offer(frame(index: 1, now: 1050))
        try await waitUntil { queue.snapshot.outcomes.count == 2 }
        if cancelTask { worker.cancel() } else { queue.cancel() }
        try await waitUntil { queue.snapshot.state == .cancelled }
        precondition(queue.snapshot.outcomes.allSatisfy {
            $0.status == .completed && $0.publication.status == .cancelled
        })
        clock.set(1100)
        await receiver.release()
        do {
            _ = try await worker.value
            preconditionFailure("Cancelled delivery returned a completed run")
        } catch is CancellationError {}
        let received = await receiver.received
        precondition(received.isEmpty)
        precondition(queue.snapshot.pendingPublications == 0)
        precondition(queue.snapshot.activePublications == 0)
    }

    /// An active model request survives Stop, retaining completion without publication.
    private static func testActiveInferenceFinishesAfterStop() async throws {
        let clock = PublicationClock()
        let processor = HeldPublicationProcessor(clock: clock)
        let recorder = PublicationRecorder()
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try await processor.process($0)
        }
        let worker = Task { try await queue.run { try await recorder.receive($0) } }
        try queue.offer(frame(index: 0, now: 1000))
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while await !processor.started && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        let started = await processor.started
        precondition(started)
        queue.cancel()
        precondition(queue.snapshot.activeFrames == 1)
        clock.set(1100)
        await processor.release()
        do {
            _ = try await worker.value
            preconditionFailure("Stopped active inference returned a completed run")
        } catch is CancellationError {}
        let cancelled = await processor.wasCancelled
        precondition(!cancelled && recorder.calls == 0)
        let outcome = queue.snapshot.outcomes[0]
        precondition(outcome.status == .completed)
        precondition(outcome.publication.status == .cancelled)
        precondition(outcome.result?.timings.doneMilliseconds == 1100)
    }

    /// A delivery that actually occurred before Stop remains a valid publication.
    private static func testReceiptBeforeStopPersists() async throws {
        let clock = PublicationClock()
        let recorder = PublicationRecorder()
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try result($0, done: clock.now)
        }
        try queue.offer(frame(index: 0, now: 1000))
        queue.finishInput()
        do {
            _ = try await queue.run { ticket in
                try await recorder.receive(ticket)
                queue.cancel()
            }
            preconditionFailure("Stop after receipt returned a completed run")
        } catch is CancellationError {}
        let outcome = queue.snapshot.outcomes[0]
        precondition(outcome.status == .completed && recorder.received.count == 1)
        precondition(outcome.publication.status == .published)
        precondition(outcome.publication.publishedMilliseconds == 1000)
    }

    /// Receipts expire after callback return and cannot deliver twice or hide errors.
    private static func testTicketLifetimeAndConsumerErrors() async throws {
        for mode in 0..<4 {
            let clock = PublicationClock()
            let recorder = PublicationRecorder()
            let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
                try result($0, done: clock.now)
            }
            try queue.offer(frame(index: 0, now: 1000))
            queue.finishInput()
            let snapshot = try await queue.run { ticket in
                try await MainActor.run {
                    recorder.ticket = ticket
                    if mode == 0 { return }
                    if mode == 1 { throw PublicationFixtureFailure.expected }
                    try recorder.receive(ticket)
                    let duplicate = try ticket.receive()
                    precondition(duplicate == nil)
                    if mode == 3 { throw PublicationFixtureFailure.expected }
                }
            }
            let outcome = snapshot.outcomes[0]
            let expected: PacedVideoPublicationOutcome.Status = mode == 0
                ? .notReceived : (mode == 1 ? .failed : .published)
            precondition(outcome.status == .completed
                         && outcome.publication.status == expected)
            precondition(recorder.received.count == (mode >= 2 ? 1 : 0))
            let expired = try recorder.ticket!.receive()
            precondition(expired == nil)
            if mode == 1 || mode == 3 {
                precondition(outcome.publication.failure != nil)
            }
        }
    }

    /// Failed, skipped and unrequested results cannot become fresh publications.
    private static func testFailedAndSkippedFrames() async throws {
        for mode in 0..<4 {
            let clock = PublicationClock()
            let recorder = PublicationRecorder()
            let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
                if mode == 0 { throw PublicationFixtureFailure.expected }
                return try result($0, done: clock.now, failed: mode == 1)
            }
            try queue.offer(frame(index: 0, now: 1000))
            if mode == 2 { clock.set(1201) }
            queue.finishInput()
            let consume: PacedVideoScheduler.Consumer? = mode == 3 ? nil : {
                @Sendable ticket in
                try await recorder.receive(ticket)
            }
            let snapshot = try await queue.run(consume: consume)
            let outcome = snapshot.outcomes[0]
            precondition(recorder.calls == 0)
            precondition(outcome.publication.status
                         == (mode == 3 ? .notRequested : .notApplicable))
            precondition(outcome.publication.publishedMilliseconds == nil)
        }
    }

    /// Invalid or regressing clocks reject receipt while retaining completed inference.
    private static func testInvalidReceiptClock() async throws {
        for invalid in [Double.nan, 999] {
            let clock = PublicationClock()
            let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
                try result($0, done: clock.now)
            }
            try queue.offer(frame(index: 0, now: 1000))
            queue.finishInput()
            let snapshot = try await queue.run { ticket in
                clock.set(invalid)
                _ = try ticket.receive()
            }
            let outcome = snapshot.outcomes[0]
            precondition(outcome.status == .completed)
            precondition(outcome.publication.status == .failed)
            precondition(outcome.publication.failure != nil)
            precondition(outcome.publication.checkedMilliseconds == nil)
            precondition(outcome.publication.publishedMilliseconds == nil)
        }
    }

    /// Exact-frame preview uses the same gate; disabled preview leaves receipt available.
    private static func testPreviewReceipt() async throws {
        for enabled in [false, true] {
            let clock = PublicationClock()
            let recorder = PublicationPreviewRecorder()
            let input = try frame(index: 0, now: 1000)
            let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
                try result($0, done: clock.now)
            }
            try queue.offer(input)
            queue.finishInput()
            let snapshot = try await queue.run(previewEnabled: enabled) { ticket in
                try await recorder.receive(ticket)
            }
            precondition(snapshot.previewEnabled == enabled)
            precondition(snapshot.outcomes[0].publication.status == .published)
            precondition(recorder.previews.count == (enabled ? 1 : 0))
            if enabled {
                precondition(recorder.previews[0].image === input.frame.image)
                precondition(recorder.previews[0].result.metadata == input.frame.metadata)
                precondition(recorder.previews[0].result.timings.publishedMilliseconds == 1000)
            }
        }
        let clock = PublicationClock()
        let recorder = PublicationPreviewRecorder()
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try result($0, done: clock.now)
        }
        try queue.offer(frame(index: 0, now: 1000))
        queue.finishInput()
        let late = try await queue.run(previewEnabled: true) { ticket in
            clock.set(1201)
            try await recorder.receive(ticket)
        }
        precondition(recorder.previews.isEmpty)
        precondition(late.outcomes[0].publication.status == .late)
    }

    /// Waits for async scheduling independently of deterministic frame age.
    private static func waitUntil(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !condition() && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        precondition(condition(), "Publication worker did not reach the expected state")
    }

    /// Confirms the callback is held after dispatch and before freshness receipt.
    private static func waitForReceiver(
        _ receiver: HeldPublicationReceiver
    ) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while await receiver.calls == 0 && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        let calls = await receiver.calls
        precondition(calls == 1)
    }
}

/// Records numerical delivery on the consumer actor, without preview or rendering.
@MainActor
private final class PublicationRecorder {
    var calls = 0
    var received: [VideoFrameResult] = []
    var ticket: PacedVideoDelivery?

    /// Uses detections only after the freshness/cancellation gate grants receipt.
    func receive(_ ticket: PacedVideoDelivery) throws {
        calls += 1
        if let result = try ticket.receive() { received.append(result) }
    }
}

/// Tests preview receipt without retaining or drawing any unchecked frame.
@MainActor
private final class PublicationPreviewRecorder {
    var previews: [VideoFramePreview] = []

    /// Uses numerical receipt when no preview was configured on the worker.
    func receive(_ delivery: PacedVideoDelivery) throws {
        if let preview = try delivery.receivePreview() { previews.append(preview) }
        else { _ = try delivery.receive() }
    }
}

/// Models a delayed actor handoff without blocking the independent inference worker.
private actor HeldPublicationReceiver {
    private var waiting: CheckedContinuation<Void, Never>?
    private var released = false
    private(set) var calls = 0
    private(set) var received: [VideoFrameResult] = []

    /// Delays first receipt and accepts later callbacks directly.
    func receive(_ ticket: PacedVideoDelivery) async throws {
        calls += 1
        if !released { await withCheckedContinuation { waiting = $0 } }
        if let preview = try ticket.receivePreview() { received.append(preview.result) }
        else if let result = try ticket.receive() { received.append(result) }
    }

    /// Opens the receiving gate without changing the source clock or model work.
    func release() {
        released = true
        waiting?.resume()
        waiting = nil
    }
}

/// Lets a genuinely active inference complete after cancellation without cancelling it.
private actor HeldPublicationProcessor {
    let clock: PublicationClock
    private var waiting: CheckedContinuation<Void, Never>?
    private(set) var started = false
    private(set) var wasCancelled = false

    /// Uses the queue clock independently of consumer delays.
    init(clock: PublicationClock) { self.clock = clock }

    /// Holds backend completion until the caller explicitly permits it to finish.
    func process(_ frame: UprightVideoFrame) async throws -> VideoFrameResult {
        started = true
        await withCheckedContinuation { waiting = $0 }
        wasCancelled = Task.isCancelled
        return try result(frame, done: clock.now)
    }

    /// Releases active work while preserving its existing task cancellation state.
    func release() {
        waiting!.resume()
        waiting = nil
    }
}

/// Shares exact deterministic millisecond endpoints between both worker tasks.
nonisolated private final class PublicationClock: @unchecked Sendable {
    private let lock = NSLock()
    private var milliseconds = 1000.0
    var now: Double { lock.withLock { milliseconds } }

    /// Sets a test endpoint under the same lock used by clock reads.
    func set(_ milliseconds: Double) {
        lock.withLock { self.milliseconds = milliseconds }
    }
}

/// Builds a small immutable source frame with exact media identity and host readiness.
nonisolated private func frame(index: Int, now: Double) throws -> PacedVideoFrame {
    let image = CGContext(
        data: nil, width: 8, height: 4, bitsPerComponent: 8, bytesPerRow: 32,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!.makeImage()!
    return PacedVideoFrame(
        frame: UprightVideoFrame(
            metadata: try VideoFrameMetadata(
                sourceID: "publication", sourceFrameIndex: index,
                presentationTime: CMTime(value: Int64(index) * 1001, timescale: 30000),
                width: 8, height: 4, loopIndex: 2), image: image,
            bufferToImage: .identity, readyMilliseconds: now),
        dueMilliseconds: now, deliveredMilliseconds: now)
}

/// Keeps inference endpoints immutable while the publication clock advances later.
nonisolated private func result(
    _ frame: UprightVideoFrame, done: Double, failed: Bool = false,
    modelTiming: ModelPredictionTiming? = nil
) throws -> VideoFrameResult {
    VideoFrameResult(
        metadata: frame.metadata, sourceWidth: frame.image.width,
        sourceHeight: frame.image.height, bufferToImage: frame.bufferToImage,
        status: failed ? .failed : .completed, detections: [],
        failure: failed
            ? VideoInferenceFailure(PublicationFixtureFailure.expected) : nil,
        timings: try VideoFrameTimings(
            readyMilliseconds: frame.readyMilliseconds, modelTiming: modelTiming,
            doneMilliseconds: done))
}

/// Represents consumer or backend failures without altering completion data.
nonisolated private enum PublicationFixtureFailure: Error { case expected }
