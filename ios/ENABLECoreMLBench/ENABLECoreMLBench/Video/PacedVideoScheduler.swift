import CoreGraphics
import Foundation

/// Records queue decisions and inference completion without retaining frame pixels.
nonisolated struct PacedVideoFrameOutcome: Sendable {
    enum Status: String, Sendable {
        case completed, failed, cancelled
        case skippedSuperseded = "skipped_superseded"
        case skippedStale = "skipped_stale"
    }

    let metadata: VideoFrameMetadata
    let readyMilliseconds: Double
    let dueMilliseconds: Double
    let deliveredMilliseconds: Double
    let decidedMilliseconds: Double?
    let status: Status
    let result: VideoFrameResult?
    let failure: VideoInferenceFailure?
    var publication: PacedVideoPublicationOutcome

    var completedAgeMilliseconds: Double? {
        guard result?.status == .completed else { return nil }
        return result!.timings.doneMilliseconds - dueMilliseconds
    }

    var publishedAgeMilliseconds: Double? {
        publication.publishedMilliseconds.map { $0 - dueMilliseconds }
    }

    /// Copies only source identity, timings and numerical results into retained logs.
    init(
        frame: PacedVideoFrame, status: Status, decidedMilliseconds: Double?,
        result: VideoFrameResult? = nil, failure: VideoInferenceFailure? = nil,
        publication: PacedVideoPublicationOutcome = .init(status: .notApplicable)
    ) {
        metadata = frame.frame.metadata
        readyMilliseconds = frame.frame.readyMilliseconds
        dueMilliseconds = frame.dueMilliseconds
        deliveredMilliseconds = frame.deliveredMilliseconds
        self.decidedMilliseconds = decidedMilliseconds
        self.status = status
        self.result = result
        self.failure = failure
        self.publication = publication
    }
}

/// Provides an atomic queue snapshot; terminal outcomes plus occupancy equal inputs.
nonisolated struct PacedVideoQueueSnapshot: Sendable {
    enum State: Equatable, Sendable { case accepting, draining, completed, cancelled }

    let state: State
    let acceptedFrames: Int
    let activeFrames: Int
    let pendingFrames: Int
    let activePublications: Int
    let pendingPublications: Int
    let outcomes: [PacedVideoFrameOutcome]
}

/// Rejects reuse, closed input, invalid clock data and mismatched inference results.
nonisolated enum PacedVideoQueueError: Error, Equatable {
    case alreadyStarted, inputClosed, invalidInput, invalidClock, mismatchedResult
}

/// Runs one inference worker with synchronous admission to a bounded pixel inbox.
/// Construct one scheduler per run and preload its detector before offering input.
actor PacedVideoScheduler {
    typealias Processor = @Sendable (UprightVideoFrame) async throws -> VideoFrameResult
    typealias Consumer = @Sendable (PacedVideoDelivery) async throws -> Void
    private let process: Processor
    private nonisolated let inbox: PacedVideoInbox

    /// Injects processing and a shared host clock for deterministic queue checks.
    init(
        nowMilliseconds: @escaping @Sendable () -> Double = {
            InferenceClock.nowMilliseconds()
        },
        process: @escaping Processor
    ) {
        self.process = process
        inbox = PacedVideoInbox(nowMilliseconds: nowMilliseconds)
    }

    /// Reuses the existing video detector, including successful empty frame results.
    init(detector: VideoFrameDetector) {
        process = { try await detector.process($0) }
        inbox = PacedVideoInbox(nowMilliseconds: { InferenceClock.nowMilliseconds() })
    }

    nonisolated var snapshot: PacedVideoQueueSnapshot { inbox.snapshot }

    /// Admits on the producer's worker without awaiting inference or an actor hop.
    nonisolated func offer(_ frame: PacedVideoFrame) throws {
        try inbox.offer(frame)
    }

    /// Rejects future input and drains the active and newest pending frame normally.
    nonisolated func finishInput() { inbox.finishInput() }

    /// Drops pending work and wakes the worker; an active inference may finish.
    nonisolated func cancel() { inbox.cancel() }

    /// Drains inference and optional bounded delivery independently. A consumer must
    /// receive its ticket after the actor hop; nil consumer records no publication.
    func run(consume: Consumer? = nil) async throws -> PacedVideoQueueSnapshot {
        try inbox.begin(publicationEnabled: consume != nil)
        let inbox = self.inbox
        let publisher = consume.map { consume in
            Task.detached { try await Self.deliver(inbox: inbox, consume: consume) }
        }
        return try await withTaskCancellationHandler {
            do {
                for await _ in inbox.signals {
                    while let frame = try inbox.take() {
                        let process = self.process
                        // One task per admitted inference, never one per source frame.
                        // Stop cannot interrupt synchronous backend work or hide it.
                        let inference = Task.detached { try await process(frame.frame) }
                        let result: VideoFrameResult?
                        let failure: VideoInferenceFailure?
                        do {
                            let received = try await inference.value
                            guard received.metadata == frame.frame.metadata,
                                received.timings.readyMilliseconds
                                    == frame.frame.readyMilliseconds,
                                received.sourceWidth == frame.frame.image.width,
                                received.sourceHeight == frame.frame.image.height,
                                received.bufferToImage == frame.frame.bufferToImage,
                                received.timings.doneMilliseconds
                                    >= frame.deliveredMilliseconds,
                                received.timings.publishedMilliseconds == nil
                            else { throw PacedVideoQueueError.mismatchedResult }
                            result = received
                            failure = nil
                        } catch {
                            result = nil
                            failure = VideoInferenceFailure(error)
                        }
                        try inbox.complete(frame, result: result, failure: failure)
                    }
                }
                // Stream cancellation can wake iteration concurrently with its
                // cancellation handler. Close input before checking drain invariants.
                if Task.isCancelled { inbox.cancel() }
                inbox.finishPublications()
                try await publisher?.value
                inbox.end()
                if inbox.snapshot.state == .cancelled { throw CancellationError() }
                return inbox.snapshot
            } catch {
                inbox.cancel()
                _ = await publisher?.result
                inbox.end()
                throw error
            }
        } onCancel: {
            self.inbox.cancel()
        }
    }

    /// Runs one delivery callback plus one newest pending numerical result, never
    /// blocking model work on UI scheduling or accumulating per-result delivery tasks.
    private nonisolated static func deliver(
        inbox: PacedVideoInbox, consume: @escaping Consumer
    ) async throws {
        do {
            for await _ in inbox.publicationSignals {
                while let index = try inbox.takePublication() {
                    let ticket = PacedVideoDelivery { try inbox.receive(index: index) }
                    let failure: (any Error)?
                    do {
                        try await consume(ticket)
                        failure = nil
                    } catch {
                        failure = error
                    }
                    inbox.finishPublication(index: index, failure: failure)
                    try Task.checkCancellation()
                }
            }
            if Task.isCancelled {
                inbox.cancel()
                throw CancellationError()
            }
        } catch {
            inbox.cancel()
            throw error
        }
    }
}

/// Protects all mutable admission state with one lock. Signals carry no pixels and
/// coalesce to one wake-up; only pending and the worker's active frame retain images.
nonisolated private final class PacedVideoInbox: @unchecked Sendable {
    private let lock = NSLock()
    private let nowMilliseconds: @Sendable () -> Double
    let signals: AsyncStream<Void>
    private let wake: AsyncStream<Void>.Continuation
    let publicationSignals: AsyncStream<Void>
    private let publicationWake: AsyncStream<Void>.Continuation
    private var state = PacedVideoQueueSnapshot.State.accepting
    private var started = false
    private var pending: PacedVideoFrame?
    private var active = false
    private var accepted = 0
    private var lastMetadata: VideoFrameMetadata?
    private var lastDue: Double?
    private var lastClockTime = 0.0
    private var outcomes: [PacedVideoFrameOutcome] = []
    private var publicationEnabled = false
    private var activePublication: Int?
    private var pendingPublication: Int?

    /// Uses a bounded signal stream so producer traffic cannot create queued tasks.
    init(nowMilliseconds: @escaping @Sendable () -> Double) {
        self.nowMilliseconds = nowMilliseconds
        (signals, wake) = AsyncStream<Void>.makeStream(
            bufferingPolicy: .bufferingNewest(1))
        (publicationSignals, publicationWake) = AsyncStream<Void>.makeStream(
            bufferingPolicy: .bufferingNewest(1))
    }

    var snapshot: PacedVideoQueueSnapshot {
        lock.withLock {
            PacedVideoQueueSnapshot(
                state: state, acceptedFrames: accepted, activeFrames: active ? 1 : 0,
                pendingFrames: pending == nil ? 0 : 1,
                activePublications: activePublication == nil ? 0 : 1,
                pendingPublications: pendingPublication == nil ? 0 : 1,
                outcomes: outcomes)
        }
    }

    /// Keeps a newer frame in the one pending slot and classifies displaced metadata.
    func offer(_ frame: PacedVideoFrame) throws {
        try lock.withLock {
            guard state == .accepting else { throw PacedVideoQueueError.inputClosed }
            let now = try currentTime()
            let metadata = frame.frame.metadata
            guard frame.dueMilliseconds.isFinite, frame.dueMilliseconds >= 0,
                frame.deliveredMilliseconds.isFinite,
                frame.deliveredMilliseconds >= frame.dueMilliseconds,
                frame.deliveredMilliseconds <= now,
                frame.frame.readyMilliseconds.isFinite,
                frame.frame.readyMilliseconds >= 0,
                frame.frame.readyMilliseconds <= frame.deliveredMilliseconds
            else { throw PacedVideoQueueError.invalidInput }
            if let previous = lastMetadata {
                guard metadata.sourceID == previous.sourceID,
                    metadata.loopIndex > previous.loopIndex
                        || (metadata.loopIndex == previous.loopIndex
                            && metadata.sourceFrameIndex > previous.sourceFrameIndex),
                    frame.dueMilliseconds >= lastDue!
                else { throw PacedVideoQueueError.invalidInput }
            }
            if let pending {
                outcomes.append(PacedVideoFrameOutcome(
                    frame: pending, status: .skippedSuperseded,
                    decidedMilliseconds: now))
            }
            pending = frame
            accepted += 1
            lastMetadata = metadata
            lastDue = frame.dueMilliseconds
        }
        wake.yield(())
    }

    /// Claims the newest pending frame and checks its age at actual worker admission.
    func take() throws -> PacedVideoFrame? {
        try lock.withLock {
            guard state != .cancelled, let frame = pending else { return nil }
            let now = try currentTime(notBefore: frame.deliveredMilliseconds)
            pending = nil
            if now - frame.dueMilliseconds > 200 {
                outcomes.append(PacedVideoFrameOutcome(
                    frame: frame, status: .skippedStale, decidedMilliseconds: now))
                return nil
            }
            precondition(!active, "Only the scheduler's single worker may take frames")
            active = true
            return frame
        }
    }

    /// Appends a pixel-free result after backend completion and releases active state.
    func complete(
        _ frame: PacedVideoFrame, result: VideoFrameResult?,
        failure: VideoInferenceFailure?
    ) throws {
        try lock.withLock {
            precondition(active)
            let now = try? currentTime(notBefore: max(
                frame.deliveredMilliseconds,
                result?.timings.doneMilliseconds ?? frame.deliveredMilliseconds))
            outcomes.append(PacedVideoFrameOutcome(
                frame: frame,
                status: result?.status == .completed ? .completed : .failed,
                decidedMilliseconds: now, result: result,
                failure: failure ?? result?.failure))
            active = false
            let index = outcomes.count - 1
            guard now != nil else {
                if result?.status == .completed {
                    setPublication(
                        index, status: publicationEnabled ? .failed : .notRequested,
                        failure: PacedVideoQueueError.invalidClock)
                }
                throw PacedVideoQueueError.invalidClock
            }
            if result?.status == .completed {
                if !publicationEnabled {
                    setPublication(index, status: .notRequested)
                } else if state == .cancelled {
                    setPublication(index, status: .cancelled, at: now)
                } else if now! - frame.dueMilliseconds > 200 {
                    setPublication(index, status: .late, at: now)
                } else {
                    if let previous = pendingPublication {
                        setPublication(previous, status: .superseded, at: now)
                    }
                    setPublication(index, status: .pending)
                    pendingPublication = index
                }
            }
        }
        publicationWake.yield(())
    }

    /// Permits only one drain task for this run, including after it has completed.
    func begin(publicationEnabled: Bool) throws {
        try lock.withLock {
            guard !started else { throw PacedVideoQueueError.alreadyStarted }
            started = true
            self.publicationEnabled = publicationEnabled
        }
    }

    /// Wakes even a not-yet-started worker so a pending tail survives stream closure.
    func finishInput() {
        lock.withLock {
            if state == .accepting { state = .draining }
        }
        wake.yield(())
        wake.finish()
    }

    /// Classifies pending cancellation; already active work keeps its own outcome.
    func cancel() {
        lock.withLock {
            guard state != .completed, state != .cancelled else { return }
            state = .cancelled
            if let pending {
                outcomes.append(PacedVideoFrameOutcome(
                    frame: pending, status: .cancelled,
                    decidedMilliseconds: try? currentTime(
                        notBefore: pending.deliveredMilliseconds)))
                self.pending = nil
            }
            for index in [activePublication, pendingPublication].compactMap({ $0 }) {
                if outcomes[index].publication.status == .pending {
                    setPublication(index, status: .cancelled, at: try? currentTime())
                }
            }
            pendingPublication = nil
        }
        wake.finish()
        publicationWake.finish()
    }

    /// Closes delivery only after inference can no longer append a numerical result.
    func finishPublications() {
        lock.withLock { precondition(!active && pending == nil) }
        publicationWake.yield(())
        publicationWake.finish()
    }

    /// Claims a pending delivery without holding any source pixels or invoking UI.
    func takePublication() throws -> Int? {
        try lock.withLock {
            guard state != .cancelled, let index = pendingPublication else {
                return nil
            }
            pendingPublication = nil
            let now: Double
            do {
                now = try currentTime(
                    notBefore: outcomes[index].result!.timings.doneMilliseconds)
            } catch {
                setPublication(index, status: .failed, failure: error)
                throw error
            }
            if now - outcomes[index].dueMilliseconds > 200 {
                setPublication(index, status: .late, at: now)
                return nil
            }
            precondition(activePublication == nil)
            activePublication = index
            return index
        }
    }

    /// Checks age and cancellation atomically at actual consumer receipt after its hop.
    func receive(index: Int) throws -> VideoFrameResult? {
        try lock.withLock {
            guard activePublication == index,
                outcomes[index].publication.status == .pending else { return nil }
            guard state != .cancelled, !Task.isCancelled else {
                setPublication(index, status: .cancelled, at: try? currentTime())
                return nil
            }
            let result = outcomes[index].result!
            let now: Double
            do {
                now = try currentTime(notBefore: result.timings.doneMilliseconds)
            } catch {
                setPublication(index, status: .failed, failure: error)
                throw error
            }
            guard now - outcomes[index].dueMilliseconds <= 200 else {
                setPublication(index, status: .late, at: now)
                return nil
            }
            let received = try result.published(at: now)
            outcomes[index].publication = PacedVideoPublicationOutcome(
                status: .published, checkedMilliseconds: now,
                publishedMilliseconds: now)
            return received
        }
    }

    /// Expires unclaimed tickets and retains receipt if later consumer work fails.
    func finishPublication(index: Int, failure: (any Error)?) {
        lock.withLock {
            precondition(activePublication == index)
            if outcomes[index].publication.status == .pending {
                let status: PacedVideoPublicationOutcome.Status = failure == nil
                    ? .notReceived
                    : (failure is CancellationError ? .cancelled : .failed)
                setPublication(
                    index, status: status, at: try? currentTime(), failure: failure)
            } else if let failure {
                let previous = outcomes[index].publication
                outcomes[index].publication = PacedVideoPublicationOutcome(
                    status: previous.status,
                    checkedMilliseconds: previous.checkedMilliseconds,
                    publishedMilliseconds: previous.publishedMilliseconds,
                    failure: VideoInferenceFailure(failure))
            }
            activePublication = nil
        }
    }

    /// Updates delivery state while preserving completion and inference endpoints.
    private func setPublication(
        _ index: Int, status: PacedVideoPublicationOutcome.Status,
        at now: Double? = nil,
        failure: (any Error)? = nil
    ) {
        outcomes[index].publication = PacedVideoPublicationOutcome(
            status: status, checkedMilliseconds: now,
            failure: failure.map { VideoInferenceFailure($0) })
    }

    /// Finalizes an empty drain while leaving cancelled snapshots distinguishable.
    func end() {
        lock.withLock {
            precondition(!active && pending == nil)
            precondition(activePublication == nil && pendingPublication == nil)
            if state != .cancelled { state = .completed }
        }
    }

    /// Rejects unusable host time instead of admitting with an invented frame age.
    private func currentTime(notBefore minimum: Double = 0) throws -> Double {
        let now = nowMilliseconds()
        guard now.isFinite, now >= max(minimum, lastClockTime) else {
            throw PacedVideoQueueError.invalidClock
        }
        lastClockTime = now
        return now
    }
}
