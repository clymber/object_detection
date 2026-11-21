import CoreGraphics
import CoreMedia
import Foundation

/// Verifies event-scoped paced rates, queue counters and unreconstructed decode losses.
@main
struct PacedVideoMetricsSmokeTests {
    /// Runs deterministic arithmetic independently of actual detector performance.
    static func main() throws {
        try testCountsRatesAndTiming()
        try testWindowBoundariesAndLiveInputs()
        try testDisplayReceipts()
        try testUnavailableRatesAndInvalidWindows()
        try testInvalidPublication()
        try testDecodeLossDiagnostics()
        print("Paced metrics: 6 count, window, rate and decode-loss checks passed.")
    }

    /// Uses one denominator while retaining all late/drain work in total counters.
    private static func testCountsRatesAndTiming() throws {
        let queue = try fixture()
        let metrics = PacedVideoRunMetrics(queue: queue, window: window(1000, 2000))
        precondition(metrics.totals.inputFrames == 7)
        precondition(metrics.totals.completedFrames == 3)
        precondition(metrics.totals.failedFrames == 1)
        precondition(metrics.totals.processedFrames == 4)
        precondition(metrics.totals.skippedFrames == 2)
        precondition(metrics.totals.skippedSupersededFrames == 1)
        precondition(metrics.totals.skippedStaleFrames == 1)
        precondition(metrics.totals.cancelledFrames == 1)
        precondition(metrics.totals.unfinishedFrames == 1)
        precondition(metrics.totals.publishedResults == 2)
        precondition(metrics.totals.lateResults == 1)
        precondition(metrics.inputFramesPerSecond == 7)
        precondition(metrics.completedFramesPerSecond == 2)
        precondition(metrics.freshResultsPerSecond == 1)
        precondition(metrics.postWindowCompletions == 1)
        precondition(metrics.postWindowPublications == 1)
        precondition(metrics.modelLatency.sampleCount == 2)
        precondition(metrics.modelLatency.meanMilliseconds == 40)
        precondition(metrics.modelLatency.p50Milliseconds == 30)
        precondition(metrics.modelLatency.p95Milliseconds == 50)
        precondition(metrics.frameLatency.meanMilliseconds == 125)
        precondition(metrics.completedAge.p95Milliseconds == 150)
        precondition(metrics.publishedAge.meanMilliseconds == 150)
        precondition(metrics.displayFramesPerSecond == nil)
        let skips = queue.outcomes.filter {
            $0.status == .skippedSuperseded || $0.status == .skippedStale
        }
        precondition(skips.map { $0.metadata.sourceFrameIndex } == [3, 4])
        precondition(skips[0].metadata.loopIndex == 4)
        precondition(skips[0].metadata.presentationTime
                     == CMTime(value: 3003, timescale: 30000))
        precondition(skips[0].dueMilliseconds == 1300)
        precondition(skips[0].decidedMilliseconds == 1400)
    }

    /// Counts completion/receipt time and inputs still occupying queue slots.
    private static func testWindowBoundariesAndLiveInputs() throws {
        let queue = try fixture()
        let next = PacedVideoRunMetrics(queue: queue, window: window(2000, 3000))
        precondition(next.inputFrames == 0)
        precondition(next.completedFrames == 1 && next.freshResults == 1)
        precondition(next.completedFramesPerSecond == 1)
        let live = snapshot(
            outcomes: [], inputs: [1000, 1900], active: 1, pending: 1)
        let metrics = PacedVideoRunMetrics(queue: live, window: window(1000, 2000))
        precondition(metrics.inputFrames == 2)
        precondition(metrics.completedFrames == 0)
        precondition(metrics.totals.unfinishedFrames == 2)
        let long = PacedVideoRunMetrics(queue: queue, window: window(1000, 61000))
        precondition(abs(long.inputFramesPerSecond! - 7.0 / 60) < 1e-12)
        precondition(abs(long.completedFramesPerSecond! - 3.0 / 60) < 1e-12)
        precondition(abs(long.freshResultsPerSecond! - 2.0 / 60) < 1e-12)
    }

    /// Distinct first draws, including delayed receipt order, never stand in for FPS.
    private static func testDisplayReceipts() throws {
        let old = UUID(), a = UUID(), b = UUID(), c = UUID()
        let samples = [
            PacedVideoDrawSample(id: old, drawnMilliseconds: 1100),
            PacedVideoDrawSample(id: old, drawnMilliseconds: 900),
            PacedVideoDrawSample(id: a, drawnMilliseconds: 1200),
            PacedVideoDrawSample(id: a, drawnMilliseconds: 1250),
            PacedVideoDrawSample(id: b, drawnMilliseconds: 1300),
            PacedVideoDrawSample(id: c, drawnMilliseconds: 1500),
            PacedVideoDrawSample(id: c, drawnMilliseconds: .nan),
        ]
        let metrics = PacedVideoRunMetrics(
            queue: try fixture(), window: window(1000, 2000), displaySamples: samples)
        precondition(metrics.displayFrames == 3 && metrics.displayFramesPerSecond == 3)
        precondition(metrics.freshResultsPerSecond == 1)
        precondition(metrics.completedFramesPerSecond == 2)
    }

    /// Disabled receipt collection and zero spans remain visibly unmeasured.
    private static func testUnavailableRatesAndInvalidWindows() throws {
        let queue = snapshot(outcomes: [], inputs: [], publicationEnabled: false)
        let metrics = PacedVideoRunMetrics(queue: queue, window: window(1000, 2000))
        precondition(metrics.inputFramesPerSecond == 0)
        precondition(metrics.completedFramesPerSecond == 0)
        precondition(metrics.freshResultsPerSecond == nil)
        let zero = PacedVideoRunMetrics(queue: queue, window: window(1000, 1000))
        precondition(zero.inputFramesPerSecond == nil
                     && zero.completedFramesPerSecond == nil)
        for (start, end) in [
            (Double.nan, 1000.0), (0, .infinity), (1000, 999), (-1, 1000)
        ] {
            do {
                _ = try PacedVideoMeasurementWindow(
                    startedMilliseconds: start, endedMilliseconds: end)
                preconditionFailure("Invalid measurement window succeeded")
            } catch PacedVideoMeasurementWindow.ValidationError.invalidWindow {}
        }
    }

    /// Invalid published ages remain visible and cannot inflate fresh counts.
    private static func testInvalidPublication() throws {
        let bad = try outcome(index: 0, due: 1000, done: 1100, published: 1210)
        let metrics = PacedVideoRunMetrics(
            queue: snapshot(outcomes: [bad], inputs: [1000]),
            window: window(1000, 2000))
        precondition(metrics.totals.publishedResults == 1)
        precondition(metrics.freshResultsPerSecond == 0)
        precondition(metrics.invalidPublications == 1)
        precondition(metrics.publishedAge.meanMilliseconds == 210)
    }

    /// An unknown decoder tail stays unknown and retains exact known sample timestamps.
    private static func testDecodeLossDiagnostics() throws {
        let pts = CMTime(
            value: 1001, timescale: 30000, flags: [.valid, .hasBeenRounded], epoch: 2)
        let issue = PacedVideoSourceIssue(
            stage: .decoding, sourceID: "fixture", loopIndex: 4,
            attemptedFrameIndex: 2,
            presentationTime: pts, read: nil, dueMilliseconds: 1100,
            observedMilliseconds: 1150, cancelled: false, decodeFailure: true,
            error: VideoReaderError.decodeFailed(frameIndex: 2, reason: "damaged"))
        var source = PacedVideoSourceReport(
            sourceID: "fixture", firstLoopIndex: 4, loopCount: 1, started: 900)
        source.decodedFrames = 2
        source.finish(
            status: .failed, at: 1150,
            summary: PacedVideoSourceSummary(deliveredFrames: 2), issue: issue)
        precondition(source.decodeLosses.count == 1)
        precondition(source.decodeLosses[0].lostFrameCount == nil)
        precondition(source.decodeLosses[0].metadata == nil)
        precondition(source.decodeLosses[0].presentationTime == pts)
        let conversion = PacedVideoSourceIssue(
            stage: .conversion, sourceID: "fixture", loopIndex: 4,
            attemptedFrameIndex: 2,
            presentationTime: nil, read: nil, dueMilliseconds: nil,
            observedMilliseconds: 1150, cancelled: false,
            error: VideoReaderError.decodeFailed(frameIndex: 2, reason: "adapter"))
        precondition(!conversion.hasUnreconstructedDecodeLoss)
        var report = PacedVideoRunReport(
            sourceID: "fixture", model: .rfdetrSmall, firstLoopIndex: 4, loopCount: 1)
        report.finish(status: .failed, source: source,
                      queue: snapshot(outcomes: [], inputs: []))
        precondition(report.hasUnreconstructedDecodeLosses)
        precondition(report.decodeLosses[0].lostFrameCount == nil)
        let timeline = try VideoPlaybackTimeline(
            sourceID: "fixture", sourceTimeRange: CMTimeRange(
                start: .zero, duration: CMTime(value: 2, timescale: 1)),
            anchorMilliseconds: 1000)
        let due = try timeline.dueMilliseconds(
            presentationTime: CMTime(value: 1001, timescale: 30000), loopIndex: 0)
        precondition(abs(due - (1000 + 1001.0 / 30)) < 1e-9)
    }

    /// Uses completed, failed, skipped and cancelled records with a boundary drain.
    private static func fixture() throws -> PacedVideoQueueSnapshot {
        snapshot(outcomes: [
            try outcome(index: 0, due: 1000, done: 1100, published: 1150, model: 50),
            try outcome(index: 1, due: 1100, done: 1250, model: 30, publication: .late),
            try outcome(index: 2, due: 1200, done: 1300, status: .failed),
            try outcome(index: 3, due: 1300, status: .skippedSuperseded, decided: 1400),
            try outcome(index: 4, due: 1400, status: .skippedStale, decided: 1700),
            try outcome(index: 5, due: 1800, status: .cancelled, decided: 1900),
            try outcome(index: 6, due: 1900, done: 2000, published: 2010),
        ], inputs: [1000, 1100, 1200, 1300, 1400, 1800, 1900])
    }

    /// Builds an explicit interval with simple millisecond endpoints for arithmetic.
    private static func window(
        _ start: Double, _ end: Double
    ) -> PacedVideoMeasurementWindow {
        try! PacedVideoMeasurementWindow(
            startedMilliseconds: start, endedMilliseconds: end)
    }

    /// Builds final or live immutable snapshots without invoking an inference engine.
    private static func snapshot(
        outcomes: [PacedVideoFrameOutcome], inputs: [Double], active: Int = 0,
        pending: Int = 0, publicationEnabled: Bool = true
    ) -> PacedVideoQueueSnapshot {
        PacedVideoQueueSnapshot(
            state: active + pending > 0 ? .accepting : .completed,
            acceptedFrames: inputs.count, inputDeliveryMilliseconds: inputs,
            activeFrames: active, pendingFrames: pending,
            activePublications: 0, pendingPublications: 0,
            publicationEnabled: publicationEnabled, previewEnabled: false,
            outcomes: outcomes)
    }

    /// Preserves raw source identity while varying completion and publication timings.
    private static func outcome(
        index: Int, due: Double, done: Double? = nil, published: Double? = nil,
        model: Double? = nil, status: PacedVideoFrameOutcome.Status = .completed,
        decided: Double? = nil,
        publication: PacedVideoPublicationOutcome.Status = .notRequested
    ) throws -> PacedVideoFrameOutcome {
        let metadata = try VideoFrameMetadata(
            sourceID: "fixture", sourceFrameIndex: index,
            presentationTime: CMTime(value: Int64(index) * 1001, timescale: 30000),
            width: 8, height: 4, loopIndex: 4)
        let image = CGContext(
            data: nil, width: 8, height: 4, bitsPerComponent: 8, bytesPerRow: 32,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!.makeImage()!
        let frame = PacedVideoFrame(
            frame: UprightVideoFrame(metadata: metadata, image: image,
                                     bufferToImage: .identity, readyMilliseconds: due),
            dueMilliseconds: due, deliveredMilliseconds: due)
        let result = try done.map { done in
            VideoFrameResult(
                metadata: metadata, sourceWidth: 8, sourceHeight: 4,
                bufferToImage: .identity,
                status: status == .failed ? .failed : .completed,
                detections: [], failure: nil,
                timings: try VideoFrameTimings(
                    readyMilliseconds: due,
                    modelTiming: model.map {
                        ModelPredictionTiming(
                            submittedMilliseconds: done - $0,
                            completedMilliseconds: done)
                    },
                    doneMilliseconds: done))
        }
        return PacedVideoFrameOutcome(
            frame: frame, status: status, decidedMilliseconds: decided ?? done,
            result: result,
            publication: PacedVideoPublicationOutcome(
                status: published != nil ? .published : publication,
                checkedMilliseconds: published, publishedMilliseconds: published))
    }
}
