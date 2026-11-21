import Foundation

/// Uses one explicit half-open host-clock interval for every paced rate numerator.
nonisolated struct PacedVideoMeasurementWindow: Equatable, Sendable {
    let startedMilliseconds: Double
    let endedMilliseconds: Double
    var durationMilliseconds: Double { endedMilliseconds - startedMilliseconds }

    enum ValidationError: Error { case invalidWindow }

    /// Allows zero duration as unmeasured; rejects invalid or reversed clock endpoints.
    init(startedMilliseconds: Double, endedMilliseconds: Double) throws {
        guard startedMilliseconds.isFinite, endedMilliseconds.isFinite,
            startedMilliseconds >= 0, endedMilliseconds >= startedMilliseconds,
            (endedMilliseconds - startedMilliseconds).isFinite
        else { throw ValidationError.invalidWindow }
        self.startedMilliseconds = startedMilliseconds
        self.endedMilliseconds = endedMilliseconds
    }

    /// Assigns events at the end boundary to the next interval, never both intervals.
    func contains(_ milliseconds: Double) -> Bool {
        milliseconds >= startedMilliseconds && milliseconds < endedMilliseconds
    }
}

/// Exposes processing, queue skips and delivery counts as independent quantities.
nonisolated struct PacedVideoRunCounts: Sendable {
    let inputFrames: Int
    let activeFrames: Int
    let pendingFrames: Int
    private(set) var completedFrames = 0
    private(set) var failedFrames = 0
    private(set) var cancelledFrames = 0
    private(set) var skippedSupersededFrames = 0
    private(set) var skippedStaleFrames = 0
    private(set) var publishedResults = 0
    private(set) var lateResults = 0
    private(set) var supersededPublications = 0
    private(set) var cancelledPublications = 0
    private(set) var failedPublications = 0
    private(set) var unreceivedPublications = 0

    var processedFrames: Int { completedFrames + failedFrames }
    var skippedFrames: Int { skippedSupersededFrames + skippedStaleFrames }
    var unfinishedFrames: Int { activeFrames + pendingFrames + cancelledFrames }

    /// Rebuilds counters from immutable logs off the UI actor, including empty results.
    init(queue: PacedVideoQueueSnapshot) {
        inputFrames = queue.acceptedFrames
        activeFrames = queue.activeFrames
        pendingFrames = queue.pendingFrames
        for outcome in queue.outcomes {
            switch outcome.status {
            case .completed: completedFrames += 1
            case .failed: failedFrames += 1
            case .cancelled: cancelledFrames += 1
            case .skippedSuperseded: skippedSupersededFrames += 1
            case .skippedStale: skippedStaleFrames += 1
            }
            switch outcome.publication.status {
            case .published: publishedResults += 1
            case .late: lateResults += 1
            case .superseded: supersededPublications += 1
            case .cancelled: cancelledPublications += 1
            case .failed: failedPublications += 1
            case .notReceived: unreceivedPublications += 1
            case .notApplicable, .notRequested, .pending: break
            }
        }
    }
}

/// An optional distinct-frame drawing receipt, independent of consumer publication.
nonisolated struct PacedVideoDrawSample: Sendable {
    let id: UUID
    let drawnMilliseconds: Double
}

/// Counts events by source delivery, model completion and actual fresh receipt time.
/// The same denominator includes source gaps, skips and idle time, excluding drain.
nonisolated struct PacedVideoRunMetrics: Sendable {
    let window: PacedVideoMeasurementWindow
    let totals: PacedVideoRunCounts
    let inputFrames: Int
    let completedFrames: Int
    let freshResults: Int
    let postWindowCompletions: Int
    let postWindowPublications: Int
    let invalidPublications: Int
    let modelLatency: VideoLatencyStatistics
    let frameLatency: VideoLatencyStatistics
    let completedAge: VideoLatencyStatistics
    let publishedAge: VideoLatencyStatistics
    let displayFrames: Int?
    private let publicationEnabled: Bool

    var inputFramesPerSecond: Double? { rate(inputFrames) }
    var completedFramesPerSecond: Double? { rate(completedFrames) }
    var freshResultsPerSecond: Double? {
        publicationEnabled ? rate(freshResults) : nil
    }
    var displayFramesPerSecond: Double? { displayFrames.flatMap { rate($0) } }

    /// Supports a full source interval or a fixed 60/600-second protocol window.
    /// Total counters retain drain work; window rates exclude post-window outcomes.
    init(
        queue: PacedVideoQueueSnapshot, window: PacedVideoMeasurementWindow,
        displaySamples: [PacedVideoDrawSample]? = nil
    ) {
        self.window = window
        totals = PacedVideoRunCounts(queue: queue)
        publicationEnabled = queue.publicationEnabled
        inputFrames = queue.inputDeliveryMilliseconds.filter {
            window.contains($0)
        }.count
        let completed = queue.outcomes.filter {
            $0.status == .completed && $0.result != nil
        }
        let inWindow = completed.filter {
            window.contains($0.result!.timings.doneMilliseconds)
        }
        completedFrames = inWindow.count
        postWindowCompletions = completed.filter {
            $0.result!.timings.doneMilliseconds >= window.endedMilliseconds
        }.count
        let published = completed.filter {
            $0.publication.status == .published
                && $0.publication.publishedMilliseconds != nil
        }
        let received = published.filter {
            window.contains($0.publication.publishedMilliseconds!)
        }
        freshResults = received.filter {
            let age = $0.publishedAgeMilliseconds!
            return age >= 0 && age <= 200
                && $0.publication.publishedMilliseconds!
                    >= $0.result!.timings.doneMilliseconds
        }.count
        invalidPublications = received.count - freshResults
        postWindowPublications = published.filter {
            $0.publication.publishedMilliseconds! >= window.endedMilliseconds
        }.count
        modelLatency = VideoLatencyStatistics(samples: inWindow.compactMap {
            $0.result!.timings.modelLatencyMilliseconds
        })
        frameLatency = VideoLatencyStatistics(samples: inWindow.map {
            $0.result!.timings.endToEndMilliseconds
        })
        completedAge = VideoLatencyStatistics(samples: inWindow.compactMap {
            $0.completedAgeMilliseconds
        })
        publishedAge = VideoLatencyStatistics(samples: received.compactMap {
            $0.publishedAgeMilliseconds
        })
        displayFrames = displaySamples.map { samples in
            var firstDraws: [UUID: Double] = [:]
            for sample in samples where sample.drawnMilliseconds.isFinite
                && sample.drawnMilliseconds >= 0
            {
                firstDraws[sample.id] = min(
                    firstDraws[sample.id] ?? sample.drawnMilliseconds,
                    sample.drawnMilliseconds)
            }
            return firstDraws.values.filter { window.contains($0) }.count
        }
    }

    /// Uses total events over elapsed time, never a mean of instantaneous frame rates.
    private func rate(_ count: Int) -> Double? {
        guard window.durationMilliseconds > 0 else { return nil }
        let value = Double(count) * 1000 / window.durationMilliseconds
        return value.isFinite ? value : nil
    }
}
