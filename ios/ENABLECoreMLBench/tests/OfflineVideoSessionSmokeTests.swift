import CoreGraphics
import CoreMedia
import Foundation
import Observation

/// Exercises UI state with callbacks deliberately delivered after cancellation.
@main
@MainActor
struct OfflineVideoSessionSmokeTests {
    private static let video = URL(fileURLWithPath: "/fixture/selected.mov")

    /// Checks selection binding, progress, errors, Stop and replacement races.
    static func main() async throws {
        try await testSelectionAndProgress()
        try await testCompletedFailures()
        try await testStopAndRestart()
        try await testStopDuringPreparation()
        try await testInputReplacement()
        try await testErrorAndRetry()
        try await testPreviewLifecycle()
        try await testPreviewReplacement()
        print("Video session: 8 control, preview and stale-callback checks passed.")
    }

    /// Freezes URL/model and counts a successful empty frame as processed.
    private static func testSelectionAndProgress() async throws {
        let worker = VideoSessionWorker()
        let session = OfflineVideoSession(worker: worker)
        precondition(!session.canStart && session.status == .ready)
        session.start(model: .rfdetrSmall)
        precondition(session.status == .error && !session.isBusy)
        session.select(video, sourceID: "selected-source")
        precondition(session.canStart && session.status == .ready)
        session.start(model: .yoloxNano)
        session.start(model: .rfdetrSmall)
        precondition(session.status == .preparing && !session.canStart)
        try await wait { await worker.runCount == 1 }
        let input = await worker.selectedURL
        let model = await worker.selectedModel
        precondition(input == video && model == .yoloxNano)
        try await worker.emit(index: 0)
        precondition(session.status == .running && session.processedFrames == 1)
        precondition(session.completedFrames == 1 && session.failedFrames == 0)
        precondition(session.progressText == "1 frame processed")
        precondition(session.metrics.frameLatency.meanMilliseconds == 1)
        precondition(session.metrics.modelLatency.meanMilliseconds == nil)
        precondition(session.metrics.runElapsedMilliseconds == nil)
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.status == .completed && session.canStart)
        precondition(session.latestReport?.sourceID == "selected-source")
        precondition(session.latestReport?.model == .yoloxNano)
        precondition(
            session.latestReport?.outcomes[0].result?.detections.isEmpty == true)
        precondition(session.latestReport?.coverage.isFullyProcessed == true)
        precondition(session.metrics.frameLatency.p95Milliseconds == 1)
        precondition(session.metrics.runElapsedMilliseconds != nil)
        precondition(session.metrics.completedRunFramesPerSecond != nil)
        precondition(session.metrics.inputFramesPerSecond
                     == session.metrics.completedRunFramesPerSecond)
    }

    /// Keeps per-frame inference failures distinct from a clean completed run.
    private static func testCompletedFailures() async throws {
        let worker = VideoSessionWorker()
        let session = OfflineVideoSession(worker: worker)
        session.select(video)
        session.start(model: .rfdetrNano)
        try await wait { await worker.runCount == 1 }
        try await worker.emit(index: 0)
        try await worker.emit(index: 1, failed: true)
        precondition(session.processedFrames == 2 && session.failedFrames == 1)
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.status == .completedWithFailures)
        precondition(session.message.contains("1 failed"))
        precondition(session.latestReport?.coverage.isFullyAccounted == true)
        precondition(session.latestReport?.coverage.isFullyProcessed == false)
        precondition(session.metrics.frameLatency.sampleCount == 1)
    }

    /// Blocks restart while stopping and retains the accepted prefix/unfinished row.
    private static func testStopAndRestart() async throws {
        let worker = VideoSessionWorker()
        let session = OfflineVideoSession(worker: worker)
        session.select(video)
        session.start(model: .rfdetrSmall)
        try await wait { await worker.runCount == 1 }
        try await worker.emit(index: 0)
        session.stop()
        session.stop()
        session.start(model: .yoloxNano)
        precondition(session.status == .stopping && !session.canStart)
        do {
            try await worker.emit(index: 1, record: false)
            preconditionFailure("A callback published after Stop")
        } catch is CancellationError {}
        precondition(session.processedFrames == 1)
        precondition(session.metrics.frameLatency.sampleCount == 1)
        await worker.finish(.cancelled, unfinishedIndex: 1)
        try await wait { !session.isBusy }
        precondition(session.status == .cancelled && session.canStart)
        precondition(session.processedFrames == 1 && session.unfinishedFrames == 1)
        precondition(session.latestReport?.status == .cancelled)
        precondition(session.metrics.frameLatency.sampleCount == 1)
        precondition(session.metrics.decodedFrames == 2)
        session.start(model: .yoloxNano)
        precondition(session.processedFrames == 0 && session.latestReport == nil)
        precondition(session.metrics.frameLatency.meanMilliseconds == nil)
        precondition(session.metrics.runElapsedMilliseconds == nil)
        try await wait { await worker.runCount == 2 }
        try await worker.emit(index: 0)
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.status == .completed)
        precondition(session.latestReport?.model == .yoloxNano)
    }

    /// Cancels even before the worker's first frame without allowing overlap.
    private static func testStopDuringPreparation() async throws {
        let worker = VideoSessionWorker()
        let session = OfflineVideoSession(worker: worker)
        session.select(video)
        session.start(model: .rfdetrSmall)
        session.stop()
        try await wait { await worker.runCount == 1 }
        precondition(session.status == .stopping && session.processedFrames == 0)
        await worker.finish(.cancelled)
        try await wait { !session.isBusy }
        precondition(session.status == .cancelled && session.canStart)
    }

    /// Clears old progress and rejects both queued callbacks and late final reports.
    private static func testInputReplacement() async throws {
        let worker = VideoSessionWorker()
        let session = OfflineVideoSession(worker: worker)
        session.select(video)
        session.start(model: .rfdetrSmall)
        try await wait { await worker.runCount == 1 }
        try await worker.emit(index: 0)
        let replacement = video.deletingLastPathComponent()
            .appendingPathComponent("replacement.mov")
        session.select(replacement)
        precondition(session.isBusy && session.processedFrames == 0)
        do {
            try await worker.emit(index: 1, record: false)
            preconditionFailure("An old input published progress")
        } catch is CancellationError {}
        // An uncooperative worker can return success after cancellation.
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.status == .ready && session.sourceURL == replacement)
        precondition(session.latestReport == nil && session.processedFrames == 0)
        precondition(session.metrics.frameLatency.meanMilliseconds == nil)
        precondition(session.metrics.runElapsedMilliseconds == nil)
        session.select(nil)
        precondition(!session.canStart && session.sourceURL == nil)
    }

    /// Publishes a setup failure and allows a fresh attempt with cleared state.
    private static func testErrorAndRetry() async throws {
        let worker = VideoSessionWorker()
        let session = OfflineVideoSession(worker: worker)
        session.select(video)
        session.start(model: .rfdetrSmall)
        try await wait { await worker.runCount == 1 }
        await worker.finish(.failed)
        try await wait { !session.isBusy }
        precondition(session.status == .error && session.canStart)
        precondition(session.message.contains("Controlled model failure"))
        precondition(session.latestReport?.issue?.stage == .modelLoading)
        precondition(session.metrics.frameLatency.meanMilliseconds == nil)
        precondition(session.metrics.runElapsedMilliseconds != nil)
        session.start(model: .rfdetrNano)
        precondition(session.latestReport == nil && session.processedFrames == 0)
        try await wait { await worker.runCount == 2 }
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.status == .completed && session.completedFrames == 0)
    }

    /// Shows paired frames, clears old boxes on empty/failure, and ignores late draws.
    private static func testPreviewLifecycle() async throws {
        let worker = VideoSessionWorker()
        let session = OfflineVideoSession(worker: worker)
        session.select(video)
        precondition(!session.isPreviewEnabled && session.previewFrame == nil)
        session.setPreviewEnabled(true)
        session.start(model: .rfdetrSmall)
        session.setPreviewEnabled(false)
        precondition(session.isPreviewEnabled, "Preview setting changed mid-run")
        try await wait { await worker.runCount == 1 }
        try await worker.emit(index: 0, withDetection: true)
        let first = session.previewFrame!
        precondition(first.result.detections.count == 1)
        precondition(session.previewStatistics.drawnFrames == 0)
        let drawGeneration = session.previewGeneration
        session.recordPreviewDraw(first.drawToken, generation: drawGeneration, at: 10)
        let observation = PreviewObservation()
        withObservationTracking {
            _ = session.previewStatistics.drawnFrames
        } onChange: {
            MainActor.assumeIsolated { observation.changed = true }
        }
        session.recordPreviewDraw(first.drawToken, generation: drawGeneration, at: 20)
        precondition(session.previewStatistics.drawnFrames == 1)
        precondition(!observation.changed, "Duplicate draws invalidated the preview")
        try await worker.emit(index: 1)
        precondition(session.previewFrame!.result.metadata.sourceFrameIndex == 1)
        precondition(session.previewFrame!.result.detections.isEmpty)
        try await worker.emit(index: 2, failed: true)
        let last = session.previewFrame!
        precondition(last.result.status == .failed && last.result.detections.isEmpty)
        session.stop()
        do {
            try await worker.emitPreviewOnly(index: 3)
            preconditionFailure("Stopped run published a preview")
        } catch is CancellationError {}
        precondition(session.previewFrame?.id == last.id)
        await worker.finish(.cancelled)
        try await wait { !session.isBusy }
        precondition(session.latestReport?.previewEnabled == true)
        session.setPreviewEnabled(false)
        precondition(session.previewFrame == nil)
        session.recordPreviewDraw(last.drawToken, generation: drawGeneration, at: 30)
        precondition(session.previewStatistics.drawnFrames == 1)
        session.start(model: .yoloxNano)
        precondition(session.previewStatistics.drawnFrames == 0)
        try await wait { await worker.runCount == 2 }
        try await worker.emit(index: 0)
        precondition(session.previewFrame == nil)
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.latestReport?.previewEnabled == false)
    }

    /// Invalidates renderer callbacks and queued previews when input/model resets.
    private static func testPreviewReplacement() async throws {
        let worker = VideoSessionWorker()
        let session = OfflineVideoSession(worker: worker)
        session.select(video)
        session.setPreviewEnabled(true)
        session.start(model: .rfdetrSmall)
        try await wait { await worker.runCount == 1 }
        try await worker.emit(index: 0)
        let old = session.previewFrame!
        let oldGeneration = session.previewGeneration
        session.select(video, sourceID: "new-selection")
        precondition(session.previewFrame == nil)
        do {
            try await worker.emitPreviewOnly(index: 1)
            preconditionFailure("Replaced input published a preview")
        } catch is CancellationError {}
        session.recordPreviewDraw(old.drawToken, generation: oldGeneration, at: 50)
        precondition(session.previewStatistics.drawnFrames == 0)
        await worker.finish(.completed)
        try await wait { !session.isBusy }
        precondition(session.previewFrame == nil && session.latestReport == nil)
        precondition(session.previewEnabledForRun == nil)
    }

    /// Bounds all asynchronous transitions so regressions fail instead of hanging.
    private static func wait(
        _ condition: @MainActor () async -> Bool
    ) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while !(await condition()) {
            precondition(ContinuousClock.now < deadline, "Video session timed out")
            try await Task.sleep(for: .milliseconds(2))
        }
    }
}

/// Records whether a repeated paint pass would trigger another SwiftUI update.
@MainActor
private final class PreviewObservation {
    var changed = false
}

/// Holds a run independently of cancellation to expose stale UI publication races.
private actor VideoSessionWorker: OfflineVideoProcessing {
    private(set) var latestReport: OfflineVideoRunReport?
    private(set) var selectedURL: URL?
    private(set) var selectedModel: DetectorModel?
    private(set) var runCount = 0
    private var consumer: OfflineVideoRunner.Consumer?
    private var previewConsumer: OfflineVideoRunner.PreviewConsumer?
    private var pending: CheckedContinuation<OfflineVideoRunSummary, any Error>?

    /// Retains the submitted configuration and waits for explicit test completion.
    func run(
        _ url: URL, sourceID: String, model: DetectorModel,
        selection: VideoFrameSelection, loopIndex: Int,
        trackID: CMPersistentTrackID?, preview: OfflineVideoRunner.PreviewConsumer?,
        consume: @escaping OfflineVideoRunner.Consumer
    ) async throws -> OfflineVideoRunSummary {
        precondition(pending == nil, "Session overlapped worker runs")
        guard case .all = selection else {
            preconditionFailure("Unexpected selection")
        }
        precondition(loopIndex == 0 && trackID == nil)
        runCount += 1
        selectedURL = url
        selectedModel = model
        latestReport = OfflineVideoRunReport(
            sourceID: sourceID, model: model, loopIndex: loopIndex, indices: nil,
            previewEnabled: preview != nil)
        consumer = consume
        previewConsumer = preview
        return try await withCheckedThrowingContinuation { pending = $0 }
    }

    /// Delivers a valid success/empty or failed row, optionally simulating a late hop.
    func emit(
        index: Int, failed: Bool = false, record: Bool = true,
        withDetection: Bool = false
    ) async throws {
        let result = try result(
            index: index, failed: failed, withDetection: withDetection)
        if record {
            latestReport?.recordDecodedFrames(index + 1)
            latestReport?.append(OfflineFrameOutcome(result: result))
        }
        try await consumer!(result)
        if let previewConsumer { try await previewConsumer(makePreview(result)) }
    }

    /// Simulates a queued preview without another progress callback first.
    func emitPreviewOnly(index: Int) async throws {
        try await previewConsumer!(makePreview(result(index: index)))
    }

    /// Builds the same upright input without copying mutable decoder storage.
    private func makePreview(_ result: VideoFrameResult) throws -> VideoFramePreview {
        let context = CGContext(
            data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 32,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        return try VideoFramePreview(frame: UprightVideoFrame(
            metadata: result.metadata, image: context.makeImage()!,
            bufferToImage: .identity, readyMilliseconds: 1), result: result)
    }

    /// Keeps success/empty/failure construction independent of preview publication.
    private func result(
        index: Int, failed: Bool = false, withDetection: Bool = false
    ) throws -> VideoFrameResult {
        let metadata = try metadata(index)
        return VideoFrameResult(
            metadata: metadata, sourceWidth: 8, sourceHeight: 8,
            bufferToImage: .identity, status: failed ? .failed : .completed,
            detections: withDetection ? [try Detection(
                boundingBox: CGRect(x: 1, y: 1, width: 2, height: 2),
                confidence: 0.9, classID: 1)] : [],
            failure: failed
                ? VideoInferenceFailure(DetectorError.noModelSelected) : nil,
            timings: try VideoFrameTimings(
                readyMilliseconds: 1, modelTiming: nil, doneMilliseconds: 2))
    }

    /// Finalizes the report even when the caller task has already been cancelled.
    func finish(
        _ status: OfflineVideoRunReport.Status, unfinishedIndex: Int? = nil
    ) {
        let error: any Error = status == .cancelled
            ? CancellationError() : DetectorError.invalidModel("Controlled model failure")
        let issue = OfflineVideoIssue(stage: .modelLoading, error: error)
        if let index = unfinishedIndex {
            let frameMetadata = try! metadata(index)
            latestReport?.recordDecodedFrames(index + 1)
            latestReport?.append(OfflineFrameOutcome(
                sourceFrameIndex: index, metadata: frameMetadata,
                readyMilliseconds: 1, cancelled: true,
                issue: OfflineVideoIssue(stage: .inference, error: error)))
        }
        if status == .completed { latestReport?.recordEndOfSource() }
        latestReport?.finish(
            status: status, issue: status == .completed ? nil : issue)
        if status == .completed {
            pending?.resume(returning: latestReport!.summary)
        } else {
            pending?.resume(throwing: error)
        }
        pending = nil
        consumer = nil
        previewConsumer = nil
    }

    /// Keeps callback identity tied to the input supplied at run creation.
    private func metadata(_ index: Int) throws -> VideoFrameMetadata {
        try VideoFrameMetadata(
            sourceID: latestReport!.sourceID, sourceFrameIndex: index,
            presentationTime: CMTime(value: Int64(index), timescale: 30),
            width: 8, height: 8)
    }
}
