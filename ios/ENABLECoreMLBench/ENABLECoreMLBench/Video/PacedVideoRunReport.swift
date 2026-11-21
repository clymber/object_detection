import Foundation

/// Publishes current counts without sending per-frame logs or pixels to a UI actor.
nonisolated struct PacedVideoRunProgress: Sendable {
    let source: PacedVideoSourceReport?
    let counts: PacedVideoRunCounts
    let metrics: PacedVideoRunMetrics?

    /// Allows controlled workers to omit live rates until a clock anchor exists.
    init(
        source: PacedVideoSourceReport?, counts: PacedVideoRunCounts,
        metrics: PacedVideoRunMetrics? = nil
    ) {
        self.source = source
        self.counts = counts
        self.metrics = metrics
    }
}

/// Combines source interruptions, skipped-frame logs and separate result publications.
nonisolated struct PacedVideoRunReport: Sendable {
    enum Status: String, Sendable { case running, completed, failed, cancelled }

    let id = UUID()
    let sourceID: String
    let model: DetectorModel
    let firstLoopIndex: Int
    let requestedLoopCount: Int?
    let previewEnabled: Bool
    let startedMilliseconds = InferenceClock.nowMilliseconds()
    private(set) var endedMilliseconds: Double?
    private(set) var status = Status.running
    private(set) var source: PacedVideoSourceReport?
    private(set) var queue: PacedVideoQueueSnapshot?
    private(set) var issue: VideoInferenceFailure?

    var counts: PacedVideoRunCounts? { queue.map { PacedVideoRunCounts(queue: $0) } }
    var decodeLosses: [PacedVideoSourceIssue] { source?.decodeLosses ?? [] }
    var hasUnreconstructedDecodeLosses: Bool { !decodeLosses.isEmpty }
    var decodedButNotDeliveredFrames: Int? {
        source.map { max(0, $0.decodedFrames - $0.deliveredFrames) }
    }
    var skippedFrames: [PacedVideoFrameOutcome] {
        queue?.outcomes.filter {
            $0.status == .skippedSuperseded || $0.status == .skippedStale
        } ?? []
    }

    /// Excludes loading and queue drain, using the actual paced source interval.
    var measurementWindow: PacedVideoMeasurementWindow? {
        guard let start = source?.timeline?.anchorMilliseconds,
            let end = source?.endedMilliseconds else { return nil }
        return try? PacedVideoMeasurementWindow(
            startedMilliseconds: start, endedMilliseconds: end)
    }

    var metrics: PacedVideoRunMetrics? {
        guard let window = measurementWindow, let queue else { return nil }
        return PacedVideoRunMetrics(queue: queue, window: window)
    }

    /// Binds run identity before model loading, preserving failures with no input yet.
    init(
        sourceID: String, model: DetectorModel, firstLoopIndex: Int, loopCount: Int?,
        previewEnabled: Bool = false
    ) {
        self.sourceID = sourceID
        self.model = model
        self.firstLoopIndex = firstLoopIndex
        requestedLoopCount = loopCount
        self.previewEnabled = previewEnabled
    }

    /// Finalizes all logs only after active inference and delivery have unwound.
    mutating func finish(
        status: Status, source: PacedVideoSourceReport?, queue: PacedVideoQueueSnapshot,
        error: (any Error)? = nil
    ) {
        self.status = status
        self.source = source
        self.queue = queue
        issue = error.map { VideoInferenceFailure($0) }
        endedMilliseconds = InferenceClock.nowMilliseconds()
    }
}
