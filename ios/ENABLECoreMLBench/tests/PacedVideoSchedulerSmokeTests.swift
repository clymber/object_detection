import CoreGraphics
import CoreMedia
import Foundation

/// Exercises bounded synchronous admission and one async inference worker.
@main
@MainActor
struct PacedVideoSchedulerSmokeTests {
    /// Checks replacement, exact stale boundaries, drain, cancellation and failures.
    static func main() async throws {
        try await testNewestPendingAndBoundedPixels()
        try await testStaleBoundary()
        try await testPendingAgesDuringInference()
        try await testReplacementBeforeWorkerStarts()
        try await testIdleWakeupsAndEmptyDrain()
        try await testCancellation(cancelTask: false)
        try await testCancellation(cancelTask: true)
        try await testIdleCancellation()
        try await testProcessingFailures()
        try await testRejectedInputAndReuse()
        try await testInvalidClock()
        print("Paced scheduler: 11 bounds, age, drain and cancellation checks passed.")
    }

    /// Hundreds of arrivals retain one pending image and cannot reenter inference.
    private static func testNewestPendingAndBoundedPixels() async throws {
        let clock = QueueTestClock()
        let processor = QueueTestProcessor(clock: clock)
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try await processor.process($0)
        }
        let worker = Task { try await queue.run() }
        try queue.offer(event(index: 0, now: clock.now))
        try await waitForStarts(processor, count: 1)
        for index in 1...500 {
            clock.advance(by: 1)
            try queue.offer(event(index: index, now: clock.now))
            let snapshot = queue.snapshot
            checkAccounting(snapshot)
            precondition(snapshot.activeFrames == 1 && snapshot.pendingFrames == 1)
        }
        precondition(queue.snapshot.outcomes.count == 499)
        precondition(queue.snapshot.outcomes.allSatisfy {
            $0.status == .skippedSuperseded && $0.result == nil
        })
        queue.finishInput()
        await processor.releaseNext()
        try await waitForStarts(processor, count: 2)
        let started = await processor.started
        let peak = await processor.peakActive
        precondition(started.map(\.sourceFrameIndex) == [0, 500])
        precondition(peak == 1)
        await processor.releaseNext()
        let result = try await worker.value
        checkAccounting(result)
        precondition(result.state == .completed && result.outcomes.count == 501)
        let completed = result.outcomes.filter { $0.status == .completed }
        precondition(completed.map { $0.metadata.sourceFrameIndex } == [0, 500])
        precondition(completed.allSatisfy { $0.result?.detections.isEmpty == true })
        precondition(completed.allSatisfy {
            $0.result?.timings.publishedMilliseconds == nil
        })
        // Source timestamp representation and loop identity survive queue decisions.
        let skipped = result.outcomes[0]
        precondition(skipped.metadata.presentationTime
                     == CMTime(value: 1001, timescale: 30000))
        precondition(skipped.metadata.loopIndex == 3)
        precondition(skipped.dueMilliseconds == 1001)
    }

    /// Exactly 200 ms remains eligible; newly decoded but older source PTS is stale.
    private static func testStaleBoundary() async throws {
        for now in [Double(1200).nextDown, 1200, Double(1200).nextUp] {
            let clock = QueueTestClock(now: now)
            let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
                try emptyResult($0, done: clock.now)
            }
            try queue.offer(event(index: 0, now: now, due: 1000))
            queue.finishInput()
            let result = try await queue.run()
            checkAccounting(result)
            let outcome = result.outcomes[0]
            precondition(outcome.readyMilliseconds == now)
            precondition(outcome.status == (now > 1200 ? .skippedStale : .completed))
            precondition(outcome.decidedMilliseconds == now)
        }
    }

    /// Fresh admission is rechecked after a slow active request, before model work.
    private static func testPendingAgesDuringInference() async throws {
        let clock = QueueTestClock()
        let processor = QueueTestProcessor(clock: clock)
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try await processor.process($0)
        }
        let worker = Task { try await queue.run() }
        try queue.offer(event(index: 0, now: clock.now))
        try await waitForStarts(processor, count: 1)
        clock.advance(by: 100)
        try queue.offer(event(index: 1, now: clock.now))
        clock.advance(by: 200.001)
        queue.finishInput()
        await processor.releaseNext()
        let result = try await worker.value
        let started = await processor.started
        precondition(started.map(\.sourceFrameIndex) == [0])
        precondition(result.outcomes.map(\.status) == [.completed, .skippedStale])
        precondition(result.outcomes[1].metadata.sourceFrameIndex == 1)
        precondition(result.outcomes[1].dueMilliseconds == 1100)
        precondition(result.outcomes[0].result!.timings.doneMilliseconds > 1200)
        checkAccounting(result)
    }

    /// Replacements before worker dispatch remain bounded and have explicit reasons.
    private static func testReplacementBeforeWorkerStarts() async throws {
        let clock = QueueTestClock()
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) { _ in
            preconditionFailure("Stale pending work entered inference")
        }
        try queue.offer(event(index: 0, now: clock.now))
        clock.advance(by: 300)
        try queue.offer(event(index: 1, now: clock.now))
        clock.advance(by: 201)
        queue.finishInput()
        let result = try await queue.run()
        precondition(result.outcomes.map(\.status)
                     == [.skippedSuperseded, .skippedStale])
        precondition(result.outcomes.map { $0.metadata.sourceFrameIndex } == [0, 1])
        checkAccounting(result)
    }

    /// Coalesced wakeups handle empty input, idle gaps and a final pending tail.
    private static func testIdleWakeupsAndEmptyDrain() async throws {
        let clock = QueueTestClock()
        let empty = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try emptyResult($0, done: clock.now)
        }
        empty.finishInput()
        let noFrames = try await empty.run()
        precondition(noFrames.state == .completed && noFrames.acceptedFrames == 0)
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try emptyResult($0, done: clock.now)
        }
        let worker = Task { try await queue.run() }
        for index in 0..<3 {
            clock.advance(by: 50)
            try queue.offer(event(index: index, now: clock.now))
            try await waitUntil { queue.snapshot.outcomes.count == index + 1 }
            checkAccounting(queue.snapshot)
        }
        queue.finishInput()
        let result = try await worker.value
        precondition(result.outcomes.count == 3)
    }

    /// Stop and task cancellation clear pending work while retaining active completion.
    private static func testCancellation(cancelTask: Bool) async throws {
        let clock = QueueTestClock()
        let processor = QueueTestProcessor(clock: clock)
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try await processor.process($0)
        }
        let worker = Task { try await queue.run() }
        try queue.offer(event(index: 0, now: clock.now))
        try await waitForStarts(processor, count: 1)
        clock.advance(by: 50)
        try queue.offer(event(index: 1, now: clock.now))
        if cancelTask { worker.cancel() } else { queue.cancel() }
        try await waitUntil { queue.snapshot.state == .cancelled }
        let cancelled = queue.snapshot
        checkAccounting(cancelled)
        precondition(cancelled.activeFrames == 1 && cancelled.pendingFrames == 0)
        precondition(cancelled.outcomes[0].status == .cancelled)
        precondition(cancelled.outcomes[0].metadata.sourceFrameIndex == 1)
        do {
            try queue.offer(event(index: 2, now: clock.now))
            preconditionFailure("Cancelled queue accepted input")
        } catch PacedVideoQueueError.inputClosed {}
        await processor.releaseNext()
        do {
            _ = try await worker.value
            preconditionFailure("Cancelled queue returned a successful run")
        } catch is CancellationError {}
        let result = queue.snapshot
        checkAccounting(result)
        precondition(result.activeFrames == 0 && result.pendingFrames == 0)
        precondition(result.outcomes.map(\.status) == [.cancelled, .completed])
        let activeWasCancelled = await processor.sawTaskCancellation
        precondition(!activeWasCancelled)
    }

    /// Cancelling before dispatch or during an idle wait cannot strand the drain task.
    private static func testIdleCancellation() async throws {
        for queued in [false, true] {
            let clock = QueueTestClock()
            let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
                try emptyResult($0, done: clock.now)
            }
            if queued { try queue.offer(event(index: 0, now: clock.now)) }
            let worker = Task { try await queue.run() }
            await Task.yield()
            worker.cancel()
            do {
                _ = try await worker.value
                preconditionFailure("Cancelled idle worker returned normally")
            } catch is CancellationError {}
            let result = queue.snapshot
            checkAccounting(result)
            precondition(result.state == .cancelled)
            precondition(result.pendingFrames == 0 && result.activeFrames == 0)
        }
    }

    /// Keeps thrown inference, explicit failure and wrong-frame results distinct.
    private static func testProcessingFailures() async throws {
        let clock = QueueTestClock()
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) { frame in
            switch frame.metadata.sourceFrameIndex {
            case 0: throw QueueFixtureFailure.expected
            case 1:
                return try emptyResult(frame, done: clock.now, failed: true)
            case 2:
                let wrong = try event(index: 99, now: clock.now)
                return try emptyResult(wrong.frame, done: clock.now)
            default: return try emptyResult(frame, done: clock.now)
            }
        }
        let worker = Task { try await queue.run() }
        for index in 0..<4 {
            try queue.offer(event(index: index, now: clock.now))
            try await waitUntil { queue.snapshot.outcomes.count == index + 1 }
        }
        queue.finishInput()
        let result = try await worker.value
        precondition(result.outcomes.map(\.status)
                     == [.failed, .failed, .failed, .completed])
        precondition(result.outcomes[0].result == nil)
        precondition(result.outcomes[0].failure?.type.contains(
            "QueueFixtureFailure") == true)
        precondition(result.outcomes[1].result?.status == .failed)
        precondition(result.outcomes[2].result == nil)
        precondition(result.outcomes[2].failure?.type.contains(
            "PacedVideoQueueError") == true)
        checkAccounting(result)
    }

    /// Closed/reused queues and malformed or reordered inputs cannot corrupt occupancy.
    private static func testRejectedInputAndReuse() async throws {
        let clock = QueueTestClock()
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try emptyResult($0, done: clock.now)
        }
        try queue.offer(event(index: 1, now: clock.now))
        for bad in [try event(index: 1, now: clock.now),
                    try event(index: 0, now: clock.now),
                    try event(index: 2, now: clock.now, source: "other"),
                    try event(index: 2, now: clock.now + 1),
                    try event(index: 2, now: clock.now, due: .nan)] {
            do {
                try queue.offer(bad)
                preconditionFailure("Invalid queue input was accepted")
            } catch PacedVideoQueueError.invalidInput {}
        }
        precondition(queue.snapshot.acceptedFrames == 1)
        let worker = Task { try await queue.run() }
        try await waitUntil { queue.snapshot.outcomes.count == 1 }
        do {
            _ = try await queue.run()
            preconditionFailure("Overlapping queue worker was accepted")
        } catch PacedVideoQueueError.alreadyStarted {}
        queue.finishInput()
        _ = try await worker.value
        do {
            _ = try await queue.run()
            preconditionFailure("Completed queue was reused")
        } catch PacedVideoQueueError.alreadyStarted {}
        do {
            try queue.offer(event(index: 2, now: clock.now))
            preconditionFailure("Finished queue accepted input")
        } catch PacedVideoQueueError.inputClosed {}
        let cancelled = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try emptyResult($0, done: clock.now)
        }
        try cancelled.offer(event(index: 0, now: clock.now))
        cancelled.cancel()
        do {
            _ = try await cancelled.run()
            preconditionFailure("Pre-cancelled queue completed")
        } catch is CancellationError {}
        checkAccounting(cancelled.snapshot)
    }

    /// Clock failures retain inputs and leave unavailable decision times unset.
    private static func testInvalidClock() async throws {
        let clock = QueueTestClock()
        let queue = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try emptyResult($0, done: clock.now)
        }
        try queue.offer(event(index: 0, now: clock.now))
        queue.finishInput()
        clock.set(.nan)
        do {
            _ = try await queue.run()
            preconditionFailure("Invalid clock admitted pending inference")
        } catch PacedVideoQueueError.invalidClock {}
        checkAccounting(queue.snapshot)
        precondition(queue.snapshot.outcomes[0].decidedMilliseconds == nil)
        clock.set(1000)
        let regressing = PacedVideoScheduler(nowMilliseconds: { clock.now }) {
            try emptyResult($0, done: clock.now)
        }
        try regressing.offer(event(index: 0, now: clock.now))
        clock.set(999)
        regressing.finishInput()
        do {
            _ = try await regressing.run()
            preconditionFailure("Regressing clock admitted pending inference")
        } catch PacedVideoQueueError.invalidClock {}
        checkAccounting(regressing.snapshot)
        precondition(regressing.snapshot.outcomes[0].decidedMilliseconds == nil)
        clock.set(1000)
        let completing = PacedVideoScheduler(nowMilliseconds: { clock.now }) { frame in
            let result = try emptyResult(frame, done: clock.now)
            clock.set(.nan)
            return result
        }
        try completing.offer(event(index: 0, now: clock.now))
        completing.finishInput()
        do {
            _ = try await completing.run()
            preconditionFailure("Invalid completion clock was hidden")
        } catch PacedVideoQueueError.invalidClock {}
        checkAccounting(completing.snapshot)
        precondition(completing.snapshot.outcomes[0].result != nil)
        precondition(completing.snapshot.outcomes[0].decidedMilliseconds == nil)
    }

    /// Requires every accepted frame to occupy a slot or have exactly one outcome.
    private static func checkAccounting(_ snapshot: PacedVideoQueueSnapshot) {
        precondition((0...1).contains(snapshot.activeFrames))
        precondition((0...1).contains(snapshot.pendingFrames))
        precondition(snapshot.acceptedFrames == snapshot.outcomes.count
                     + snapshot.activeFrames + snapshot.pendingFrames)
    }

    /// Waits for the real async worker with bounded wall time, independent of fake age.
    private static func waitUntil(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !condition() && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        precondition(condition(), "Queue did not reach the expected state")
    }

    /// Coordinates arrivals after inference has entered its gate.
    private static func waitForStarts(
        _ processor: QueueTestProcessor, count: Int
    ) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while await processor.started.count < count && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        let started = await processor.started.count
        precondition(started == count, "Wrong number of in-flight inference requests")
    }
}

/// Supplies valid exact source metadata with image storage separate from the queue.
nonisolated private func event(
    index: Int, now: Double, due: Double? = nil, source: String = "fixture"
) throws -> PacedVideoFrame {
    let image = CGContext(
        data: nil, width: 8, height: 4, bitsPerComponent: 8, bytesPerRow: 32,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!.makeImage()!
    let frame = UprightVideoFrame(
        metadata: try VideoFrameMetadata(
            sourceID: source, sourceFrameIndex: index,
            presentationTime: CMTime(value: Int64(index) * 1001, timescale: 30000),
            width: 8, height: 4, loopIndex: 3),
        image: image, bufferToImage: .identity, readyMilliseconds: now)
    return PacedVideoFrame(
        frame: frame, dueMilliseconds: due ?? now, deliveredMilliseconds: now)
}

/// Produces numerical outcomes with the injected clock and no fabricated publication.
nonisolated private func emptyResult(
    _ frame: UprightVideoFrame, done: Double, failed: Bool = false
) throws -> VideoFrameResult {
    VideoFrameResult(
        metadata: frame.metadata, sourceWidth: frame.image.width,
        sourceHeight: frame.image.height, bufferToImage: frame.bufferToImage,
        status: failed ? .failed : .completed, detections: [],
        failure: failed ? VideoInferenceFailure(QueueFixtureFailure.expected) : nil,
        timings: try VideoFrameTimings(
            readyMilliseconds: frame.readyMilliseconds, modelTiming: nil,
            doneMilliseconds: done))
}

/// Models elapsed age atomically across the synchronous producer and async worker.
nonisolated private final class QueueTestClock: @unchecked Sendable {
    private let lock = NSLock()
    private var milliseconds: Double
    var now: Double { lock.withLock { milliseconds } }

    /// Starts all endpoints on the same deterministic millisecond origin.
    init(now: Double = 1000) { milliseconds = now }

    /// Advances virtual time without sleeping or depending on thread scheduling.
    func advance(by milliseconds: Double) {
        lock.withLock { self.milliseconds += milliseconds }
    }

    /// Injects an unusable endpoint to check clock-failure cleanup paths.
    func set(_ milliseconds: Double) {
        lock.withLock { self.milliseconds = milliseconds }
    }
}

/// Lets source traffic proceed while inference waits on an explicit continuation.
private actor QueueTestProcessor {
    let clock: QueueTestClock
    private var waiting: CheckedContinuation<Void, Never>?
    private var active = 0
    private(set) var peakActive = 0
    private(set) var started: [VideoFrameMetadata] = []
    private(set) var sawTaskCancellation = false

    /// Shares the queue's clock so held inference includes its elapsed waiting time.
    init(clock: QueueTestClock) { self.clock = clock }

    /// Detects actor reentrancy and retains one deliberately slow active request.
    func process(_ frame: UprightVideoFrame) async throws -> VideoFrameResult {
        active += 1
        peakActive = max(peakActive, active)
        precondition(active == 1, "Scheduler allowed overlapping model requests")
        started.append(frame.metadata)
        await withCheckedContinuation { waiting = $0 }
        sawTaskCancellation = sawTaskCancellation || Task.isCancelled
        active -= 1
        return try emptyResult(frame, done: clock.now)
    }

    /// Completes exactly the current active request without cancelling its task.
    func releaseNext() {
        precondition(waiting != nil)
        waiting!.resume()
        waiting = nil
    }
}

/// Separates expected model failures from queue and timing validation errors.
nonisolated private enum QueueFixtureFailure: Error { case expected }
