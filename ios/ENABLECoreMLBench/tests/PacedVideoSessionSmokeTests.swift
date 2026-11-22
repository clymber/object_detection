import CoreGraphics
import CoreMedia
import Foundation

/// Exercises mode routing and UI rejection of late receipts, polls and final reports.
@main
@MainActor
struct PacedVideoSessionSmokeTests {
    private static let video = URL(fileURLWithPath: "/fixture/video.mov")

    /// Checks numeric/preview receipt, controls, source replacement, errors and modes.
    static func main() async throws {
        try await testSelectionProgressAndFinish()
        try await testStopAndLateReceipt()
        try await testReplacementRejectsOldProgress()
        try await testPreviewAndDrawing()
        try await testFailureAndRetry()
        try await testModeRouting()
        print("Paced UI: 6 receipt, progress, mode and cancellation checks passed.")
    }

    /// Freezes configuration, polls counts/rates and consumes preview-off empty results.
    private static func testSelectionProgressAndFinish() async throws {
        let worker = PacedUIWorker()
        let session = PacedVideoSession(worker: worker, pollInterval: .milliseconds(1))
        session.start(model: .rfdetrSmall)
        precondition(session.status == .error && !session.canStart)
        session.select(video, sourceID: "stable-source")
        session.start(model: .yoloxNano)
        session.start(model: .rfdetrSmall)
        try await wait { await worker.runCount == 1 }
        let model = await worker.model
        let source = await worker.sourceID
        precondition(model == .yoloxNano && source == "stable-source")
        let probe = UIReceiptProbe()
        try await worker.emit(index: 0, probe: probe)
        try await wait { session.counts.completedFrames == 1 }
        precondition(session.status == .running && !session.canStart)
        precondition(session.latestReceipt?.detections.isEmpty == true)
        precondition(session.previewFrame == nil && probe.count == 1)
        precondition(session.metrics?.freshResultsPerSecond == 1)
        precondition(session.metrics?.frameLatency.p95Milliseconds == nil)
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.status == .completed && session.canStart)
        precondition(session.latestReport?.sourceID == "stable-source")
        precondition(session.metrics?.frameLatency.p95Milliseconds == 1)
        precondition(session.displayFramesPerSecond == nil)
    }

    /// Stop rejects receipt before claim and blocks restart during cleanup.
    private static func testStopAndLateReceipt() async throws {
        let worker = PacedUIWorker()
        let session = PacedVideoSession(worker: worker, pollInterval: .milliseconds(1))
        session.select(video)
        session.start(model: .rfdetrSmall)
        session.stop()
        try await wait { await worker.runCount == 1 }
        session.start(model: .yoloxNano)
        precondition(session.status == .stopping && !session.canStart)
        let probe = UIReceiptProbe()
        do {
            try await worker.emit(index: 0, probe: probe)
            preconditionFailure("Receipt entered stopped UI")
        } catch is CancellationError {}
        precondition(probe.count == 0 && session.latestReceipt == nil)
        await worker.finish(.cancelled)
        try await wait { !session.isBusy }
        precondition(session.status == .cancelled && session.canStart)
        session.start(model: .yoloxNano)
        try await wait { await worker.runCount == 2 }
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.status == .completed)
    }

    /// Input/model replacement rejects old snapshots and uncooperative final success.
    private static func testReplacementRejectsOldProgress() async throws {
        let worker = PacedUIWorker()
        let session = PacedVideoSession(worker: worker, pollInterval: .milliseconds(1))
        session.select(video)
        session.setPreviewEnabled(true)
        session.start(model: .rfdetrSmall)
        try await wait { await worker.runCount == 1 }
        try await worker.emit(index: 0, probe: UIReceiptProbe())
        try await wait { session.counts.completedFrames == 1 }
        await worker.pauseProgress()
        try await wait { await worker.hasHeldProgress }
        let replacement = video.appendingPathExtension("new")
        session.select(replacement, sourceID: "replacement")
        precondition(session.status == .stopping && session.latestReceipt == nil)
        await worker.releaseProgress()
        let probe = UIReceiptProbe()
        do {
            try await worker.emit(index: 1, probe: probe)
            preconditionFailure("Old input published a receipt")
        } catch is CancellationError {}
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.status == .ready && session.sourceURL == replacement)
        precondition(session.latestReport == nil && session.metrics == nil)
        precondition(session.counts.inputFrames == 0 && probe.count == 0)
    }

    /// Empty results clear boxes and drawing remains independent and unique.
    private static func testPreviewAndDrawing() async throws {
        let worker = PacedUIWorker()
        let session = PacedVideoSession(worker: worker, pollInterval: .milliseconds(1))
        session.select(video)
        session.setPreviewEnabled(true)
        session.start(model: .rfdetrSmall)
        try await wait { await worker.runCount == 1 }
        session.setPreviewEnabled(false)
        precondition(session.isPreviewEnabled)
        try await worker.emit(index: 0, probe: UIReceiptProbe(), withBox: true)
        precondition(session.previewFrame?.result.detections.count == 1)
        try await worker.emit(index: 1, probe: UIReceiptProbe())
        precondition(session.previewFrame?.result.detections.isEmpty == true)
        let token = session.previewFrame!.drawToken
        let generation = session.previewGeneration
        session.recordPreviewDraw(token, generation: generation, at: 1100)
        session.recordPreviewDraw(token, generation: generation, at: 1101)
        precondition(session.previewStatistics.drawnFrames == 1)
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.counts.publishedResults == 2)
        precondition(session.displayFramesPerSecond == 1)
        session.select(video.appendingPathExtension("new"))
        session.recordPreviewDraw(token, generation: generation, at: 1200)
        precondition(session.previewFrame == nil
                     && session.previewStatistics.drawnFrames == 0)
    }

    /// Error reports remain reviewable and retry starts with clear counters and rates.
    private static func testFailureAndRetry() async throws {
        let worker = PacedUIWorker()
        let session = PacedVideoSession(worker: worker, pollInterval: .milliseconds(1))
        session.select(video)
        session.start(model: .rfdetrSmall)
        try await wait { await worker.runCount == 1 }
        try await worker.emit(index: 0, probe: UIReceiptProbe())
        await worker.finish(.failed)
        try await wait { !session.isBusy }
        precondition(session.status == .error && session.latestReport != nil)
        precondition(session.counts.completedFrames == 1 && session.canStart)
        session.start(model: .rfdetrNano)
        precondition(session.counts.inputFrames == 0 && session.metrics == nil)
        try await wait { await worker.runCount == 2 }
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.status == .completed)
    }

    /// Controls default offline, freeze mode while busy and clear cross-mode data.
    private static func testModeRouting() async throws {
        let off = OfflineUIWorker(), paced = PacedUIWorker()
        let session = VideoSession(
            offline: OfflineVideoSession(worker: off),
            paced: PacedVideoSession(worker: paced, pollInterval: .milliseconds(1)))
        session.select(video, sourceID: "mode-source")
        precondition(session.mode == .offline && session.canStart)
        session.start(model: .yoloxNano)
        session.setMode(.paced)
        precondition(session.mode == .offline)
        try await wait { !session.isBusy }
        let offRuns = await off.runs
        precondition(offRuns == 1 && session.hasReport)
        session.setMode(.paced)
        precondition(session.mode == .paced && !session.hasReport && session.canStart)
        session.setPreviewEnabled(true)
        session.start(model: .rfdetrNano)
        session.setMode(.offline)
        precondition(session.mode == .paced)
        try await wait { await paced.runCount == 1 }
        let model = await paced.model
        precondition(model == .rfdetrNano)
        session.suspendDetection()
        await paced.finish(.cancelled)
        try await wait { !session.isBusy }
        session.setMode(.offline)
        precondition(session.isPreviewEnabled && session.offline.latestReport == nil)
        precondition(session.paced.latestReport == nil && session.canStart)
    }

    /// Polls bounded async state while permitting the UI and worker actors to advance.
    private static func wait(_ condition: () async -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !(await condition()) && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        let satisfied = await condition()
        precondition(satisfied, "Paced UI did not reach the expected state")
    }
}

/// Counts whether the UI claimed a receipt after its lifecycle guards passed.
nonisolated private final class UIReceiptProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    var count: Int { lock.withLock { calls } }
    /// Records both numerical and preview claims under a lock.
    func claim() { lock.withLock { calls += 1 } }
}

/// Holds a fake run and delayed progress to exercise UI generation checks explicitly.
private actor PacedUIWorker: PacedVideoProcessing {
    private(set) var latestReport: PacedVideoRunReport?
    var latestMetrics: PacedVideoRunMetrics? { latestReport?.metrics }
    private(set) var runCount = 0
    private(set) var model = DetectorModel.rfdetrSmall
    private(set) var sourceID = ""
    private var previewEnabled = false
    private var consumer: PacedVideoScheduler.Consumer?
    private var completion: CheckedContinuation<Void, Error>?
    private var outcomes: [PacedVideoFrameOutcome] = []
    private var holdProgress = false
    private var heldProgress: CheckedContinuation<Void, Never>?
    var hasHeldProgress: Bool { heldProgress != nil }

    /// Captures configuration and deliberately lets tests finish after cancellation.
    func run(
        _ url: URL, sourceID: String, model: DetectorModel,
        firstLoopIndex: Int, loopCount: Int?, trackID: CMPersistentTrackID?,
        previewEnabled: Bool, consume: PacedVideoScheduler.Consumer?
    ) async throws -> PacedVideoRunReport {
        runCount += 1
        self.model = model
        self.sourceID = sourceID
        self.previewEnabled = previewEnabled
        consumer = consume
        outcomes = []
        holdProgress = false
        latestReport = PacedVideoRunReport(
            sourceID: sourceID, model: model, firstLoopIndex: firstLoopIndex,
            loopCount: loopCount, previewEnabled: previewEnabled)
        try await withCheckedThrowingContinuation { completion = $0 }
        return latestReport!
    }

    /// Emits a controlled receipt while keeping original inference endpoints immutable.
    func emit(index: Int, probe: UIReceiptProbe, withBox: Bool = false) async throws {
        let frame = try uiFrame(sourceID: sourceID, index: index)
        let ready = frame.readyMilliseconds
        let detections = withBox ? [try Detection(
            boundingBox: CGRect(x: 1, y: 1, width: 2, height: 2),
            confidence: 0.9, classID: 0)] : []
        let result = VideoFrameResult(
            metadata: frame.metadata, sourceWidth: 8, sourceHeight: 4,
            bufferToImage: .identity, status: .completed, detections: detections,
            failure: nil, timings: try VideoFrameTimings(
                readyMilliseconds: ready, modelTiming: nil, doneMilliseconds: ready + 1))
        let preview: (@Sendable () throws -> VideoFramePreview?)? = previewEnabled
            ? { @Sendable in
                probe.claim()
                return try VideoFramePreview(
                    frame: frame, result: result.published(at: ready + 2))
            } : nil
        let ticket = PacedVideoDelivery(accept: {
            probe.claim()
            return try result.published(at: ready + 2)
        }, preview: preview)
        try await consumer?(ticket)
        outcomes.append(PacedVideoFrameOutcome(
            frame: PacedVideoFrame(frame: frame, dueMilliseconds: ready,
                                   deliveredMilliseconds: ready),
            status: .completed, decidedMilliseconds: ready + 1, result: result,
            publication: PacedVideoPublicationOutcome(
                status: .published, publishedMilliseconds: ready + 2)))
    }

    /// Returns old snapshots even if cancellation occurs during their actor hop.
    func progress() async -> PacedVideoRunProgress? {
        guard latestReport != nil else { return nil }
        let queue = snapshot()
        let progress = PacedVideoRunProgress(
            source: sourceReport(.running), counts: PacedVideoRunCounts(queue: queue),
            metrics: PacedVideoRunMetrics(queue: queue, window: try! .init(
                startedMilliseconds: 1000, endedMilliseconds: 2000),
                finalizeStatistics: false))
        if holdProgress { await withCheckedContinuation { heldProgress = $0 } }
        return progress
    }

    /// Makes the next polling callback wait so tests can replace the selected input.
    func pauseProgress() { holdProgress = true }
    /// Releases the stale callback without cooperating with task cancellation.
    func releaseProgress() {
        holdProgress = false
        heldProgress?.resume()
        heldProgress = nil
    }

    /// Finalizes with success, failure or cancellation while keeping completed records.
    func finish(_ status: PacedVideoRunReport.Status) {
        let sourceStatus: PacedVideoSourceReport.Status = status == .completed
            ? .completed : (status == .cancelled ? .cancelled : .failed)
        latestReport?.finish(
            status: status, source: sourceReport(sourceStatus), queue: snapshot(),
            error: status == .failed ? UIWorkerFailure.expected : nil)
        let pending = completion
        completion = nil
        if status == .failed { pending?.resume(throwing: UIWorkerFailure.expected) }
        else if status == .cancelled { pending?.resume(throwing: CancellationError()) }
        else { pending?.resume() }
    }

    /// Supplies metadata only; every test interval is exactly one second.
    private func sourceReport(_ status: PacedVideoSourceReport.Status)
        -> PacedVideoSourceReport
    {
        var source = PacedVideoSourceReport(
            sourceID: sourceID, firstLoopIndex: 0, loopCount: 1, started: 900)
        source.timeline = try! VideoPlaybackTimeline(
            sourceID: sourceID, sourceTimeRange: CMTimeRange(
                start: .zero, duration: CMTime(value: 1, timescale: 1)),
            anchorMilliseconds: 1000)
        source.decodedFrames = outcomes.count
        source.deliveredFrames = outcomes.count
        source.status = status
        if status != .running { source.endedMilliseconds = 2000 }
        return source
    }

    /// Builds terminal queue data without sending frame pixels through progress.
    private func snapshot() -> PacedVideoQueueSnapshot {
        PacedVideoQueueSnapshot(
            state: .completed, acceptedFrames: outcomes.count,
            inputDeliveryMilliseconds: outcomes.map(\.deliveredMilliseconds),
            activeFrames: 0, pendingFrames: 0, activePublications: 0,
            pendingPublications: 0, publicationEnabled: true,
            previewEnabled: previewEnabled, outcomes: outcomes)
    }
}

/// Makes offline mode routing observable while preserving its existing session API.
private actor OfflineUIWorker: OfflineVideoProcessing {
    var latestReport: OfflineVideoRunReport?
    private(set) var runs = 0

    /// Completes an empty input without asynchronous callbacks or real model loading.
    func run(
        _ url: URL, sourceID: String, model: DetectorModel,
        selection: VideoFrameSelection, loopIndex: Int, trackID: CMPersistentTrackID?,
        preview: (@Sendable (VideoFramePreview) async throws -> Void)?,
        consume: @escaping @Sendable (VideoFrameResult) async throws -> Void
    ) async throws -> OfflineVideoRunSummary {
        runs += 1
        latestReport = OfflineVideoRunReport(
            sourceID: sourceID, model: model, loopIndex: loopIndex, indices: [])
        latestReport?.recordEndOfSource()
        latestReport?.finish(status: .completed)
        return latestReport!.summary
    }
}

/// Builds a controlled immutable input for session-only lifecycle tests.
nonisolated private func uiFrame(
    sourceID: String, index: Int
) throws -> UprightVideoFrame {
    let image = CGContext(
        data: nil, width: 8, height: 4, bitsPerComponent: 8, bytesPerRow: 32,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!.makeImage()!
    return UprightVideoFrame(
        metadata: try VideoFrameMetadata(
            sourceID: sourceID, sourceFrameIndex: index,
            presentationTime: CMTime(value: Int64(index) * 1001, timescale: 30000),
            width: 8, height: 4), image: image, bufferToImage: .identity,
        readyMilliseconds: 1000 + Double(index) * 10)
}

/// Keeps deliberate UI-worker failures distinct from decoding/model issues.
nonisolated private enum UIWorkerFailure: Error { case expected }
