import CoreGraphics
import CoreMedia
import Foundation

/// Verifies numerical metrics against independent synthetic host-clock endpoints.
@main
struct OfflineVideoMetricsSmokeTests {
    /// Checks percentile rules, denominators, missing timing and partial attempts.
    static func main() throws {
        testNearestRank()
        try testLiveRates()
        try testFailedAndUntimedSamples()
        try testFinalReport()
        try testCancelledReport()
        try testEmptyAndZeroTime()
        print("Video metrics: 6 numerical timing, rate and partial-run checks passed.")
    }

    /// Uses protocol nearest-rank percentiles rather than interpolated quantiles.
    private static func testNearestRank() {
        let stats = VideoLatencyStatistics(samples: (1...20).map(Double.init))
        precondition(stats.sampleCount == 20 && stats.meanMilliseconds == 10.5)
        precondition(stats.p50Milliseconds == 10 && stats.p95Milliseconds == 19)
        let singleton = VideoLatencyStatistics(samples: [7])
        precondition(singleton.meanMilliseconds == 7)
        precondition(singleton.p50Milliseconds == 7 && singleton.p95Milliseconds == 7)
        let pair = VideoLatencyStatistics(samples: [40, 10])
        precondition(pair.p50Milliseconds == 10 && pair.p95Milliseconds == 40)
    }

    /// Distinguishes processing-window FPS from inverse mean latency and source PTS.
    private static func testLiveRates() throws {
        var metrics = OfflineVideoRunMetrics()
        metrics.record(try result(0, ready: 100, done: 140, model: 110...120))
        metrics.record(try result(1, ready: 200, done: 280, model: 210...250))
        precondition(metrics.frameLatency.meanMilliseconds == 60)
        precondition(metrics.modelLatency.meanMilliseconds == 25)
        near(metrics.modelFramesPerSecond!, 40)
        near(metrics.endToEndFramesPerSecond!, 2000 / 180)
        precondition(metrics.endToEndFramesPerSecond != 1000 / 60)
        precondition(metrics.frameLatency.p95Milliseconds == nil)
        precondition(metrics.runElapsedMilliseconds == nil)
        precondition(metrics.inputFramesPerSecond == nil)
        precondition(metrics.completedRunFramesPerSecond == nil)
    }

    /// Failed time stays in throughput, while absent model timing is never invented.
    private static func testFailedAndUntimedSamples() throws {
        var metrics = OfflineVideoRunMetrics()
        metrics.record(try result(0, ready: 100, done: 140, model: 110...120))
        metrics.record(try result(
            1, ready: 200, done: 300, model: 210...290, failed: true))
        precondition(metrics.frameLatency.sampleCount == 1)
        precondition(metrics.modelLatency.sampleCount == 1)
        precondition(metrics.frameLatency.meanMilliseconds == 40)
        near(metrics.endToEndFramesPerSecond!, 5)
        near(metrics.modelFramesPerSecond!, 100)
        metrics.record(try result(2, ready: 400, done: 420))
        precondition(metrics.frameLatency.sampleCount == 2)
        precondition(metrics.modelLatency.sampleCount == 1)
        precondition(metrics.frameLatency.meanMilliseconds == 30)
        near(metrics.endToEndFramesPerSecond!, 2000 / 320)
    }

    /// Reconciles successful/empty samples and separates the full-run denominator.
    private static func testFinalReport() throws {
        var report = makeReport()
        for frame in [
            try result(0, ready: 100, done: 140, model: 110...120),
            try result(1, ready: 200, done: 280, model: 210...250),
            try result(2, ready: 300, done: 600, model: 310...590, failed: true),
        ] {
            report.append(OfflineFrameOutcome(result: frame))
        }
        report.recordDecodedFrames(3)
        report.recordEndOfSource()
        let live = OfflineVideoRunMetrics(report: report)
        precondition(live.runElapsedMilliseconds == nil)
        report.finish(status: .completed)
        let metrics = OfflineVideoRunMetrics(report: report)
        precondition(metrics.completedFrames == 2 && metrics.decodedFrames == 3)
        precondition(metrics.frameLatency.meanMilliseconds == 60)
        precondition(metrics.frameLatency.p50Milliseconds == 40)
        precondition(metrics.frameLatency.p95Milliseconds == 80)
        precondition(metrics.modelLatency.p50Milliseconds == 10)
        precondition(metrics.modelLatency.p95Milliseconds == 40)
        near(metrics.endToEndFramesPerSecond!, 4)
        let elapsed = report.endedMilliseconds! - report.startedMilliseconds
        near(metrics.runElapsedMilliseconds!, elapsed)
        near(metrics.inputFramesPerSecond!, 3000 / elapsed)
        near(metrics.completedRunFramesPerSecond!, 2000 / elapsed)
    }

    /// Keeps accepted work after cancellation and excludes unfinished work from FPS.
    private static func testCancelledReport() throws {
        var report = makeReport()
        let success = try result(0, ready: 100, done: 150, model: 110...130)
        report.append(OfflineFrameOutcome(result: success))
        let partial = try result(1, ready: 200, done: 250)
        report.append(OfflineFrameOutcome(
            sourceFrameIndex: 1, metadata: partial.metadata,
            readyMilliseconds: 200, cancelled: true,
            issue: OfflineVideoIssue(stage: .inference, error: CancellationError())))
        report.recordDecodedFrames(2)
        report.finish(status: .cancelled)
        let metrics = OfflineVideoRunMetrics(report: report)
        precondition(metrics.completedFrames == 1 && metrics.decodedFrames == 2)
        precondition(metrics.frameLatency.sampleCount == 1)
        near(metrics.endToEndFramesPerSecond!, 20)
        let elapsed = report.endedMilliseconds! - report.startedMilliseconds
        near(metrics.completedRunFramesPerSecond!, 1000 / elapsed)
        near(metrics.inputFramesPerSecond!, 2000 / elapsed)
    }

    /// Unavailable data stays nil; zero latency never creates infinite throughput.
    private static func testEmptyAndZeroTime() throws {
        var metrics = OfflineVideoRunMetrics()
        precondition(metrics.frameLatency.meanMilliseconds == nil)
        precondition(metrics.modelFramesPerSecond == nil)
        precondition(metrics.endToEndFramesPerSecond == nil)
        metrics.record(try result(0, ready: 100, done: 100, model: 100...100))
        precondition(metrics.frameLatency.meanMilliseconds == 0)
        precondition(metrics.modelLatency.meanMilliseconds == 0)
        precondition(metrics.modelFramesPerSecond == nil)
        precondition(metrics.endToEndFramesPerSecond == nil)
        var report = makeReport()
        report.finish(status: .failed)
        let empty = OfflineVideoRunMetrics(report: report)
        precondition(empty.frameLatency.p95Milliseconds == nil)
        precondition(empty.modelFramesPerSecond == nil)
        precondition(empty.endToEndFramesPerSecond == nil)
        precondition(empty.completedRunFramesPerSecond == 0)
        precondition(empty.inputFramesPerSecond == 0)
    }

    /// Creates an explicit host-clock result whose distant source PTS cannot set FPS.
    private static func result(
        _ index: Int, ready: Double, done: Double,
        model: ClosedRange<Double>? = nil, failed: Bool = false
    ) throws -> VideoFrameResult {
        VideoFrameResult(
            metadata: try VideoFrameMetadata(
                sourceID: "metrics", sourceFrameIndex: index,
                presentationTime: CMTime(value: Int64(index * 999), timescale: 1),
                width: 8, height: 8),
            sourceWidth: 8, sourceHeight: 8, bufferToImage: .identity,
            status: failed ? .failed : .completed, detections: [],
            failure: failed
                ? VideoInferenceFailure(DetectorError.noModelSelected) : nil,
            timings: try VideoFrameTimings(
                readyMilliseconds: ready,
                modelTiming: model.map { ModelPredictionTiming(
                    submittedMilliseconds: $0.lowerBound,
                    completedMilliseconds: $0.upperBound) },
                doneMilliseconds: done))
    }

    /// Uses a real monotonic run envelope independently of synthetic frame samples.
    private static func makeReport() -> OfflineVideoRunReport {
        OfflineVideoRunReport(
            sourceID: "metrics", model: .rfdetrSmall, loopIndex: 0, indices: nil)
    }

    /// Checks rate arithmetic with tolerance for floating-point division.
    private static func near(_ actual: Double, _ expected: Double) {
        precondition(abs(actual - expected) <= max(1e-9, abs(expected) * 1e-12))
    }
}
