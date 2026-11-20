import Foundation

/// Summarizes successful samples; percentiles are finalized on the worker.
nonisolated struct VideoLatencyStatistics: Sendable {
    private(set) var sampleCount = 0
    private(set) var totalMilliseconds = 0.0
    private(set) var latestMilliseconds: Double?
    private(set) var p50Milliseconds: Double?
    private(set) var p95Milliseconds: Double?

    var meanMilliseconds: Double? {
        sampleCount > 0 ? totalMilliseconds / Double(sampleCount) : nil
    }

    /// Starts an empty live summary without allocating sample storage on the UI actor.
    init() {}

    /// Applies the protocol's nearest-rank ceil(p * N) rule to retained samples.
    init(samples: [Double]) {
        for sample in samples { record(sample) }
        let sorted = samples.sorted()
        guard !sorted.isEmpty else { return }
        p50Milliseconds = sorted[Int(ceil(0.50 * Double(sorted.count))) - 1]
        p95Milliseconds = sorted[Int(ceil(0.95 * Double(sorted.count))) - 1]
    }

    /// Updates live means in constant time; final percentiles require all samples.
    mutating func record(_ milliseconds: Double) {
        sampleCount += 1
        totalMilliseconds += milliseconds
        latestMilliseconds = milliseconds
        p50Milliseconds = nil
        p95Milliseconds = nil
    }
}

/// Diagnostic streamed-run measures, separate from preloaded/warmed benchmarks.
nonisolated struct OfflineVideoRunMetrics: Sendable {
    private(set) var modelLatency = VideoLatencyStatistics()
    private(set) var frameLatency = VideoLatencyStatistics()
    private(set) var runElapsedMilliseconds: Double?
    private(set) var decodedFrames: Int?
    private(set) var completedFrames: Int?
    private var firstReadyMilliseconds: Double?
    private var lastDoneMilliseconds: Double?

    /// Uses successful model samples without averaging instantaneous FPS.
    var modelFramesPerSecond: Double? {
        guard modelLatency.sampleCount > 0 else { return nil }
        return Self.rate(
            modelLatency.sampleCount, over: modelLatency.totalMilliseconds)
    }

    /// Includes inter-request gaps, failed attempts and inter-frame decode delays.
    var endToEndFramesPerSecond: Double? {
        guard let firstReadyMilliseconds, let lastDoneMilliseconds else { return nil }
        return Self.rate(
            frameLatency.sampleCount,
            over: lastDoneMilliseconds - firstReadyMilliseconds)
    }

    /// Counts decoded inputs over the full worker run, including setup and cleanup.
    var inputFramesPerSecond: Double? {
        guard let decodedFrames, let runElapsedMilliseconds else { return nil }
        return Self.rate(decodedFrames, over: runElapsedMilliseconds)
    }

    /// Counts successful frames over the same full-run denominator as input FPS.
    var completedRunFramesPerSecond: Double? {
        guard let completedFrames, let runElapsedMilliseconds else { return nil }
        return Self.rate(completedFrames, over: runElapsedMilliseconds)
    }

    /// Creates a live summary; full-run rates remain unknown until the worker ends.
    init() {}

    /// Rebuilds final metrics and percentiles off the UI actor from the retained log.
    init(report: OfflineVideoRunReport) {
        let results = report.outcomes.compactMap(\.result)
        for result in results { record(result) }
        let successful = results.filter { $0.status == .completed }
        modelLatency = VideoLatencyStatistics(samples: successful.compactMap {
            $0.timings.modelLatencyMilliseconds
        })
        frameLatency = VideoLatencyStatistics(samples: successful.map {
            $0.timings.endToEndMilliseconds
        })
        if report.status != .running, let ended = report.endedMilliseconds {
            runElapsedMilliseconds = ended - report.startedMilliseconds
            decodedFrames = report.decodedFrames
            completedFrames = report.completedFrames
        }
    }

    /// Updates live timing without collecting samples or sorting on the UI actor.
    mutating func record(_ result: VideoFrameResult) {
        let timing = result.timings
        firstReadyMilliseconds = min(
            firstReadyMilliseconds ?? timing.readyMilliseconds,
            timing.readyMilliseconds)
        lastDoneMilliseconds = max(
            lastDoneMilliseconds ?? timing.doneMilliseconds, timing.doneMilliseconds)
        // Failed attempts widen the throughput window but never count as successes.
        guard result.status == .completed else { return }
        frameLatency.record(timing.endToEndMilliseconds)
        if let modelMilliseconds = timing.modelLatencyMilliseconds {
            modelLatency.record(modelMilliseconds)
        }
    }

    /// Leaves absent/zero-width measurements unavailable instead of inventing FPS.
    private static func rate(_ count: Int, over milliseconds: Double) -> Double? {
        guard milliseconds.isFinite, milliseconds > 0 else { return nil }
        return Double(count) * 1000 / milliseconds
    }
}
