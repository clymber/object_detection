import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation

/// Exercises serial offline orchestration against a real decoded movie.
@main
@MainActor
struct OfflineVideoRunnerSmokeTests {
    private static let timestamps: [Int64] = [0, 1001, 4004, 6006]

    /// Verifies selections, slow inference, backpressure, cancellation and errors.
    static func main() async throws {
        let directory = URL(
            fileURLWithPath: "outputs/ios_builds/video_pipeline/offline_fixtures")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let video = directory.appendingPathComponent("offline.mov")
        try await writeVideo(to: video)
        try await testAllFrames(video)
        try await testSelection(video)
        try await testConsumerBackpressure(video)
        try await testCancellation(video, explicit: false)
        try await testCancellation(video, explicit: true)
        try await testConsumerFailure(video)
        try await testInvalidRequests(video)
        try await testPreCancelledRun(video)
        try await testInterruptedConversion(video)
        try await testConversionFailure(video)
        try await testPrefixCancellation(video, explicit: false)
        try await testPrefixCancellation(video, explicit: true)
        try await testConsumerCancellation(video)
        try await testDecodeFailure(video, directory: directory)
        try await testFrozenCoverage(video)
        try await testCoverageGuard(video)
        try await testSessionProgress(video)
        try await testSessionStop(video)
        try await testSessionInputError(video)
        print("Offline video: 19 sequencing, coverage and session checks passed.")
    }

    /// Processes frames exceeding 200 ms without age-based dropping or overlap.
    private static func testAllFrames(_ video: URL) async throws {
        let backend = OfflineFixtureDetector(delay: .milliseconds(220), failFrame: 2)
        let runner = makeRunner(backend)
        let collector = OfflineCollector()
        let summary = try await runner.run(
            video, sourceID: "offline-fixture", model: .rfdetrSmall, loopIndex: 2,
            consume: { await collector.append($0) })
        precondition(summary == OfflineVideoRunSummary(
            decodedFrames: 4, processedFrames: 4, completedFrames: 3, failedFrames: 1))
        let records = await collector.records
        precondition(records.map { $0.metadata.sourceFrameIndex } == [0, 1, 2, 3])
        precondition(records.map { $0.metadata.presentationTime.value } == timestamps)
        precondition(records.allSatisfy {
            $0.metadata.sourceID == "offline-fixture" && $0.metadata.loopIndex == 2
                && $0.metadata.presentationTime.timescale == 30000
                && $0.sourceWidth == 32 && $0.sourceHeight == 64
                && $0.processingDurationMilliseconds >= 200
        })
        precondition(records[0].detections.count == 1)
        precondition(records[1].status == .completed && records[1].detections.isEmpty)
        precondition(records[2].status == .failed && records[2].failure != nil)
        precondition(records[3].detections.count == 1)
        let maximum = await backend.maximumActive
        let loads = await backend.loads
        precondition(maximum == 1 && loads == 1)
        let running = await runner.isRunning
        precondition(!running)
        let report = await report(runner)
        precondition(report.status == .completed && report.summary == summary)
        precondition(report.outcomes.map(\.status) == [.completed, .completed,
                                                     .failed, .completed])
        precondition(report.outcomes[1].result!.detections.isEmpty)
        precondition(report.outcomes[2].issue?.stage == .inference)
        precondition(report.outcomes[2].result!.status == .failed)
        precondition(report.unfinishedFrames == 0 && report.issue == nil)
        precondition(report.endedMilliseconds! >= report.startedMilliseconds)
        precondition(report.reachedEndOfSource && report.coverage.isFullyAccounted)
        precondition(!report.coverage.isFullyProcessed)
    }

    /// Preserves sparse source indices and stops after the last selected frame.
    private static func testSelection(_ video: URL) async throws {
        let backend = OfflineFixtureDetector()
        let runner = makeRunner(backend)
        let collector = OfflineCollector()
        let summary = try await runner.run(
            video, sourceID: "selection", model: .rfdetrSmall,
            selection: .indices([0, 2]), consume: { await collector.append($0) })
        let records = await collector.records
        precondition(summary.decodedFrames == 3 && summary.processedFrames == 2)
        precondition(records.map { $0.metadata.sourceFrameIndex } == [0, 2])
        precondition(records.map { $0.metadata.presentationTime.value } == [0, 4004])
        let received = await backend.received
        precondition(received == [0, 2])
        let previous = await report(runner)
        precondition(previous.selectedFrameIndices == [0, 2])
        precondition(!previous.reachedEndOfSource && previous.coverage.isFullyProcessed)
        let empty = try await runner.run(
            video, sourceID: "empty", model: .rfdetrSmall,
            selection: .indices([]), consume: { _ in
                preconditionFailure("Empty selection produced a result")
            })
        precondition(empty.processedFrames == 0 && empty.decodedFrames == 0)
        let latest = await report(runner)
        precondition(latest.id != previous.id && latest.outcomes.isEmpty)
        precondition(previous.outcomes.count == 2)
    }

    /// Holds consumption to prove no next inference starts and overlapping runs reject.
    private static func testConsumerBackpressure(_ video: URL) async throws {
        let backend = OfflineFixtureDetector()
        let runner = makeRunner(backend)
        let gate = OfflineGate()
        let pending = Task {
            try await runner.run(
                video, sourceID: "backpressure", model: .rfdetrSmall,
                consume: { _ in await gate.holdOnce() })
        }
        await gate.waitForRequest()
        let received = await backend.received
        precondition(received == [0])
        let active = await report(runner)
        precondition(active.status == .running && active.outcomes.count == 1)
        do {
            _ = try await runner.run(
                video, sourceID: "overlap", model: .rfdetrSmall, consume: { _ in })
            preconditionFailure("Overlapping runs were accepted")
        } catch OfflineVideoError.alreadyRunning {}
        let unchanged = await report(runner)
        precondition(unchanged.id == active.id)
        await gate.finish()
        let summary = try await pending.value
        precondition(summary.processedFrames == 4)
        let maximum = await backend.maximumActive
        precondition(maximum == 1)
    }

    /// Rejects an in-flight result after task or explicit cancellation, then restarts.
    private static func testCancellation(_ video: URL, explicit: Bool) async throws {
        let gate = OfflineGate()
        let backend = OfflineFixtureDetector(gate: gate)
        let runner = makeRunner(backend)
        let collector = OfflineCollector()
        let pending = Task {
            try await runner.run(
                video, sourceID: "cancel", model: .rfdetrSmall,
                consume: { await collector.append($0) })
        }
        await gate.waitForRequest()
        if explicit { await runner.cancel() } else { pending.cancel() }
        await gate.finish()
        do {
            _ = try await pending.value
            preconditionFailure("Cancelled run completed normally")
        } catch is CancellationError {}
        let records = await collector.records
        let running = await runner.isRunning
        precondition(records.isEmpty && !running)
        let interrupted = await report(runner)
        precondition(interrupted.status == .cancelled)
        precondition(interrupted.outcomes.count == 1)
        precondition(interrupted.unfinishedFrames == 1)
        precondition(interrupted.outcomes[0].status == .unfinished)
        precondition(interrupted.outcomes[0].metadata?.sourceFrameIndex == 0)
        precondition(interrupted.outcomes[0].result == nil)
        precondition(interrupted.outcomes[0].issue?.stage == .inference)
        precondition(interrupted.summary.processedFrames == 0)
        let restarted = try await runner.run(
            video, sourceID: "restart", model: .rfdetrSmall,
            selection: .indices([0]), consume: { await collector.append($0) })
        precondition(restarted.processedFrames == 1)
        let retained = await collector.records
        precondition(retained.count == 1 && retained[0].metadata.sourceID == "restart")
        let latest = await report(runner)
        precondition(latest.status == .completed && latest.id != interrupted.id)
        precondition(latest.outcomes.count == 1 && latest.unfinishedFrames == 0)
        precondition(interrupted.status == .cancelled)
        precondition(interrupted.unfinishedFrames == 1)
    }

    /// Stops on a consumer failure without continuing decode and permits another run.
    private static func testConsumerFailure(_ video: URL) async throws {
        let backend = OfflineFixtureDetector()
        let runner = makeRunner(backend)
        do {
            _ = try await runner.run(
                video, sourceID: "consumer-failure", model: .rfdetrSmall,
                consume: { _ in throw OfflineFixtureError.consumer })
            preconditionFailure("Consumer error was lost")
        } catch OfflineFixtureError.consumer {}
        let received = await backend.received
        let running = await runner.isRunning
        precondition(received == [0] && !running)
        let failed = await report(runner)
        precondition(failed.status == .failed && failed.issue?.stage == .consumption)
        precondition(failed.outcomes.count == 1)
        precondition(failed.outcomes[0].status == .completed)
        let restarted = try await runner.run(
            video, sourceID: "restart", model: .rfdetrSmall,
            selection: .indices([0]), consume: { _ in })
        precondition(restarted.processedFrames == 1)
    }

    /// Rejects bad selections and releases the runner after model-load failures.
    private static func testInvalidRequests(_ video: URL) async throws {
        let backend = OfflineFixtureDetector()
        let runner = makeRunner(backend)
        for indices in [[-1], [1, 1], [2, 1]] {
            do {
                _ = try await runner.run(
                    video, sourceID: "invalid", model: .rfdetrSmall,
                    selection: .indices(indices), consume: { _ in })
                preconditionFailure("Invalid selection was accepted")
            } catch OfflineVideoError.invalidSelection {}
        }
        let loads = await backend.loads
        precondition(loads == 0)
        let rejected = await runner.latestReport
        precondition(rejected == nil)
        let collector = OfflineCollector()
        do {
            _ = try await runner.run(
                video, sourceID: "missing", model: .rfdetrSmall,
                selection: .indices([0, 8]), consume: { await collector.append($0) })
            preconditionFailure("Incomplete selection was reported as completed")
        } catch OfflineVideoError.missingSelectedFrame(8) {}
        let partial = await collector.records
        precondition(partial.map { $0.metadata.sourceFrameIndex } == [0])
        let missing = await report(runner)
        precondition(missing.status == .failed && missing.decodedFrames == 4)
        precondition(missing.outcomes.map(\.sourceFrameIndex) == [0, 8])
        precondition(missing.outcomes[1].status == .failed)
        precondition(missing.outcomes[1].metadata == nil)
        precondition(missing.outcomes[1].presentationTime == nil)
        precondition(missing.outcomes[1].readyMilliseconds == nil)
        precondition(missing.outcomes[1].issue?.stage == .decoding)
        let failing = OfflineVideoRunner(engine: DetectorEngine(loader: { _ in
            throw DetectorError.missingModel("fixture model")
        }))
        do {
            _ = try await failing.run(
                video, sourceID: "load-failure", model: .rfdetrSmall, consume: { _ in })
            preconditionFailure("Model-load failure was lost")
        } catch DetectorError.missingModel {}
        let running = await failing.isRunning
        precondition(!running)
        let failure = await report(failing)
        precondition(failure.status == .failed && failure.issue?.stage == .modelLoading)
        precondition(failure.outcomes.isEmpty)
    }

    /// Retains a cancelled run header without fabricating a frame before decoding.
    private static func testPreCancelledRun(_ video: URL) async throws {
        let runner = makeRunner(OfflineFixtureDetector())
        let pending = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await runner.run(
                video, sourceID: "pre-cancelled", model: .rfdetrSmall,
                consume: { _ in })
        }
        do {
            _ = try await pending.value
            preconditionFailure("Pre-cancelled run succeeded")
        } catch is CancellationError {}
        let snapshot = await report(runner)
        precondition(snapshot.status == .cancelled && snapshot.outcomes.isEmpty)
        precondition(snapshot.decodedFrames == 0)
        precondition(snapshot.issue?.stage == .modelLoading)
    }

    /// Keeps actual admission metadata when reader cancellation discards its callback.
    private static func testInterruptedConversion(_ video: URL) async throws {
        let backend = OfflineFixtureDetector()
        let runner = makeRunner(backend, converter: { frame in
            withUnsafeCurrentTask { $0?.cancel() }
            return try VideoFrameConverter.convert(frame)
        })
        let pending = Task {
            try await runner.run(
                video, sourceID: "conversion-cancel", model: .rfdetrSmall,
                consume: { _ in preconditionFailure("Cancelled conversion delivered") })
        }
        do {
            _ = try await pending.value
            preconditionFailure("Cancelled conversion succeeded")
        } catch is CancellationError {}
        let snapshot = await report(runner)
        precondition(snapshot.status == .cancelled && snapshot.decodedFrames == 1)
        precondition(snapshot.outcomes.count == 1 && snapshot.unfinishedFrames == 1)
        let frame = snapshot.outcomes[0]
        precondition(frame.issue?.stage == .conversion && frame.result == nil)
        precondition(frame.metadata?.sourceFrameIndex == 0)
        precondition(frame.presentationTime?.value == 0)
        precondition(frame.presentationTime?.timescale == 30000)
        precondition(frame.readyMilliseconds != nil)
        let received = await backend.received
        precondition(received.isEmpty)
    }

    /// Retains the completed prefix and exact metadata for a failed conversion.
    private static func testConversionFailure(_ video: URL) async throws {
        let runner = makeRunner(OfflineFixtureDetector(), converter: { frame in
            if frame.metadata.sourceFrameIndex == 1 {
                throw OfflineFixtureError.conversion
            }
            return try VideoFrameConverter.convert(frame)
        })
        do {
            _ = try await runner.run(
                video, sourceID: "conversion-failure", model: .rfdetrSmall,
                consume: { _ in })
            preconditionFailure("Conversion failure was lost")
        } catch OfflineFixtureError.conversion {}
        let snapshot = await report(runner)
        precondition(snapshot.status == .failed && snapshot.decodedFrames == 2)
        precondition(snapshot.outcomes.map(\.status) == [.completed, .failed])
        let failed = snapshot.outcomes[1]
        precondition(failed.sourceFrameIndex == 1)
        precondition(failed.presentationTime?.value == 1001)
        precondition(failed.issue?.stage == .conversion && failed.result == nil)
    }

    /// Preserves a completed prefix and marks the interrupted inference unfinished.
    private static func testPrefixCancellation(
        _ video: URL, explicit: Bool
    ) async throws {
        let gate = OfflineGate(request: 2)
        let runner = makeRunner(OfflineFixtureDetector(gate: gate))
        let collector = OfflineCollector()
        let pending = Task {
            try await runner.run(
                video, sourceID: "prefix-cancel", model: .rfdetrSmall,
                consume: { await collector.append($0) })
        }
        await gate.waitForRequest()
        let prefix = await report(runner)
        precondition(prefix.status == .running && prefix.outcomes.count == 1)
        if explicit { await runner.cancel() } else { pending.cancel() }
        await gate.finish()
        do {
            _ = try await pending.value
            preconditionFailure("Cancelled prefix run succeeded")
        } catch is CancellationError {}
        let snapshot = await report(runner)
        precondition(snapshot.status == .cancelled)
        precondition(snapshot.outcomes.map(\.sourceFrameIndex) == [0, 1])
        precondition(snapshot.outcomes.map(\.status) == [.completed, .unfinished])
        precondition(snapshot.outcomes[1].presentationTime?.value == 1001)
        precondition(snapshot.outcomes[1].issue?.stage == .inference)
        precondition(snapshot.summary.completedFrames == 1)
        precondition(snapshot.unfinishedFrames == 1)
        precondition(snapshot.coverage.sourceRemainderUnknown)
        precondition(!snapshot.coverage.isFullyAccounted)
        precondition(prefix.status == .running && prefix.outcomes.count == 1)
        let delivered = await collector.records
        precondition(delivered.map { $0.metadata.sourceFrameIndex } == [0])
    }

    /// Does not relabel completed inference when cancellation interrupts consumption.
    private static func testConsumerCancellation(_ video: URL) async throws {
        let gate = OfflineGate(request: 2)
        let runner = makeRunner(OfflineFixtureDetector())
        let pending = Task {
            try await runner.run(
                video, sourceID: "consumer-cancel", model: .rfdetrSmall,
                consume: { _ in await gate.holdOnce() })
        }
        await gate.waitForRequest()
        pending.cancel()
        await gate.finish()
        do {
            _ = try await pending.value
            preconditionFailure("Cancelled consumer run succeeded")
        } catch is CancellationError {}
        let snapshot = await report(runner)
        precondition(snapshot.status == .cancelled)
        precondition(snapshot.issue?.stage == .consumption)
        precondition(snapshot.outcomes.map(\.status) == [.completed, .completed])
        precondition(snapshot.outcomes[1].result!.detections.isEmpty)
        precondition(snapshot.unfinishedFrames == 0)
    }

    /// Leaves source time unknown for a decoder failure that yielded no valid pixels.
    private static func testDecodeFailure(_ video: URL, directory: URL) async throws {
        let damaged = directory.appendingPathComponent("report_damaged.mov")
        try corruptMedia(video, to: damaged)
        let runner = makeRunner(OfflineFixtureDetector())
        var failedIndex: Int?
        do {
            _ = try await runner.run(
                damaged, sourceID: "decode-failure", model: .rfdetrSmall,
                consume: { _ in })
            preconditionFailure("Damaged movie succeeded")
        } catch let error as VideoReaderError {
            switch error {
            case .decodeFailed(let index, _), .invalidFrame(let index, _):
                failedIndex = index
            case .cannotStart: break
            default: preconditionFailure("Unexpected damaged movie error: \(error)")
            }
        }
        let snapshot = await report(runner)
        precondition(snapshot.status == .failed)
        if let failedIndex {
            let failed = snapshot.outcomes.last!
            precondition(failed.status == .failed && failed.issue?.stage == .decoding)
            precondition(failed.sourceFrameIndex == failedIndex)
            precondition(failed.metadata == nil && failed.presentationTime == nil)
            precondition(failed.readyMilliseconds == nil)
        } else {
            precondition(snapshot.issue?.stage == .sourceOpening)
            precondition(snapshot.outcomes.isEmpty)
        }
    }

    /// Uses independent exact frame references for an unpaced frozen selection.
    private static func testFrozenCoverage(_ video: URL) async throws {
        let plan = try OfflineVideoCoveragePlan(
            sourceID: "frozen-video-fixture", loopIndex: 1, frames: [0, 2, 3].map {
                OfflineVideoFrameReference(sourceFrameIndex: $0,
                    presentationTime: CMTime(value: timestamps[$0], timescale: 30000))
            })
        let runner = makeRunner(OfflineFixtureDetector())
        _ = try await runner.run(
            video, sourceID: plan.sourceID, model: .rfdetrSmall,
            selection: plan.selection, loopIndex: plan.loopIndex, consume: { _ in })
        let snapshot = await report(runner)
        let coverage = snapshot.coverage(using: plan)
        precondition(coverage.basis == .referenceSequence)
        precondition(coverage.isFullyAccounted && coverage.isFullyProcessed)
        precondition(coverage.expectedFrameCount == 3)
        precondition(coverage.observedFrameCount == 3)
        precondition(coverage.timestampMismatchIndices.isEmpty)
    }

    /// Rejects a malformed converter's incorrect source identity at completion.
    private static func testCoverageGuard(_ video: URL) async throws {
        let runner = makeRunner(OfflineFixtureDetector(), converter: { raw in
            let frame = try VideoFrameConverter.convert(raw)
            let metadata = try VideoFrameMetadata(
                sourceID: "wrong-source",
                sourceFrameIndex: frame.metadata.sourceFrameIndex,
                presentationTime: frame.metadata.presentationTime,
                width: frame.metadata.width, height: frame.metadata.height)
            return UprightVideoFrame(
                metadata: metadata, image: frame.image,
                bufferToImage: frame.bufferToImage,
                readyMilliseconds: frame.readyMilliseconds)
        })
        do {
            _ = try await runner.run(
                video, sourceID: "coverage-guard", model: .rfdetrSmall,
                selection: .indices([0]), consume: { _ in })
            preconditionFailure("Incorrect identity passed the coverage gate")
        } catch OfflineVideoError.coverageMismatch {}
        let snapshot = await report(runner)
        precondition(snapshot.status == .failed)
        precondition(snapshot.issue?.stage == .reconciliation)
        precondition(snapshot.coverage.invalidOutcomeIndices == [0])
        precondition(!snapshot.coverage.isFullyAccounted)
    }

    /// Publishes live progress from real decoding and retains mixed frame outcomes.
    private static func testSessionProgress(_ video: URL) async throws {
        let gate = OfflineGate(request: 2)
        let backend = OfflineFixtureDetector(failFrame: 2, gate: gate)
        let session = OfflineVideoSession(worker: makeRunner(backend))
        session.select(video, sourceID: "session-progress")
        session.start(model: .rfdetrNano)
        precondition(session.status == .preparing && !session.canStart)
        await gate.waitForRequest()
        precondition(session.status == .running && session.processedFrames == 1)
        precondition(session.metrics.frameLatency.sampleCount == 1)
        precondition(session.metrics.frameLatency.meanMilliseconds! > 0)
        precondition(session.metrics.runElapsedMilliseconds == nil)
        await gate.finish()
        try await waitForSession(session)
        precondition(session.status == .completedWithFailures && session.canStart)
        precondition(session.processedFrames == 4 && session.completedFrames == 3)
        precondition(session.failedFrames == 1 && session.unfinishedFrames == 0)
        let snapshot = session.latestReport!
        precondition(
            snapshot.model == .rfdetrNano && snapshot.sourceID == "session-progress")
        precondition(snapshot.coverage.isFullyAccounted)
        precondition(snapshot.outcomes[1].result!.detections.isEmpty)
        precondition(session.metrics.frameLatency.sampleCount == 3)
        precondition(session.metrics.frameLatency.p95Milliseconds != nil)
        precondition(session.metrics.modelLatency.meanMilliseconds == nil)
        let runMilliseconds = snapshot.endedMilliseconds! - snapshot.startedMilliseconds
        precondition(session.metrics.runElapsedMilliseconds == runMilliseconds)
        precondition(session.metrics.inputFramesPerSecond == 4000 / runMilliseconds)
        precondition(
            session.metrics.completedRunFramesPerSecond == 3000 / runMilliseconds)
    }

    /// Keeps Stop pending during an active prediction, then permits a fresh run.
    private static func testSessionStop(_ video: URL) async throws {
        let gate = OfflineGate()
        let backend = OfflineFixtureDetector(gate: gate)
        let session = OfflineVideoSession(worker: makeRunner(backend))
        session.select(video)
        session.start(model: .rfdetrSmall)
        await gate.waitForRequest()
        session.stop()
        session.start(model: .yoloxNano)
        precondition(session.status == .stopping && !session.canStart)
        await gate.finish()
        try await waitForSession(session)
        precondition(session.status == .cancelled && session.processedFrames == 0)
        precondition(session.unfinishedFrames == 1 && session.canStart)
        let cancelled = session.latestReport!
        precondition(cancelled.status == .cancelled && cancelled.model == .rfdetrSmall)
        precondition(cancelled.outcomes[0].sourceFrameIndex == 0)
        precondition(session.metrics.frameLatency.sampleCount == 0)
        precondition(session.metrics.completedRunFramesPerSecond == 0)
        session.start(model: .yoloxNano)
        try await waitForSession(session)
        precondition(session.status == .completed && session.completedFrames == 4)
        precondition(session.latestReport!.id != cancelled.id)
        precondition(session.latestReport!.model == .yoloxNano)
        precondition(session.latestReport!.coverage.isFullyProcessed)
        precondition(session.metrics.frameLatency.sampleCount == 4)
    }

    /// Surfaces a real reader startup failure without leaving controls busy.
    private static func testSessionInputError(_ video: URL) async throws {
        let session = OfflineVideoSession(worker: makeRunner(OfflineFixtureDetector()))
        session.select(video.appendingPathExtension("missing"))
        session.start(model: .rfdetrSmall)
        try await waitForSession(session)
        precondition(session.status == .error && session.canStart)
        precondition(session.processedFrames == 0 && !session.message.isEmpty)
        precondition(session.latestReport?.status == .failed)
        precondition(session.latestReport?.issue?.stage == .sourceOpening)
        precondition(session.metrics.runElapsedMilliseconds != nil)
        precondition(session.metrics.inputFramesPerSecond == 0)
        precondition(session.metrics.completedRunFramesPerSecond == 0)
    }

    /// Bounds session completion checks while allowing actor handoffs to progress.
    private static func waitForSession(_ session: OfflineVideoSession) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while session.isBusy && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(2))
        }
        precondition(!session.isBusy, "Offline video session did not finish")
    }

    /// Requires a retained value snapshot after both normal and exceptional exits.
    private static func report(
        _ runner: OfflineVideoRunner
    ) async -> OfflineVideoRunReport {
        guard let report = await runner.latestReport else {
            preconditionFailure("Run report was not retained")
        }
        return report
    }

    /// Counts model loads independently of per-frame inference calls.
    private static func makeRunner(
        _ backend: OfflineFixtureDetector,
        converter: @escaping OfflineVideoRunner.Converter = {
            try VideoFrameConverter.convert($0)
        }
    ) -> OfflineVideoRunner {
        OfflineVideoRunner(engine: DetectorEngine(loader: { _ in
            await backend.loaded()
            return backend
        }), converter: converter)
    }

    /// Generates an independent movie with irregular PTS, rotation and known pixels.
    private static func writeVideo(to url: URL) async throws {
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 64, AVVideoHeightKey: 32,
        ])
        input.mediaTimeScale = 30000
        input.transform = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 32, ty: 0)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ])
        writer.add(input)
        guard writer.startWriting() else {
            throw OfflineFixtureError.encoding(String(describing: writer.error))
        }
        writer.startSession(atSourceTime: .zero)
        for index in timestamps.indices {
            let deadline = ContinuousClock.now.advanced(by: .seconds(10))
            while !input.isReadyForMoreMediaData && ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(5))
            }
            precondition(input.isReadyForMoreMediaData, "Fixture encoder timed out")
            var buffer: CVPixelBuffer?
            precondition(CVPixelBufferCreate(
                kCFAllocatorDefault, 64, 32, kCVPixelFormatType_32BGRA,
                nil, &buffer) == kCVReturnSuccess)
            let pixels = buffer!
            CVPixelBufferLockBaseAddress(pixels, [])
            let base = CVPixelBufferGetBaseAddress(pixels)!
                .assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(pixels)
            for y in 0..<32 {
                for x in 0..<64 {
                    let offset = y * stride + x * 4
                    for channel in 0..<3 {
                        base[offset + channel] = UInt8(16 + 64 * index)
                    }
                    base[offset + 3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(pixels, [])
            precondition(adaptor.append(
                pixels, withPresentationTime: CMTime(
                    value: timestamps[index], timescale: 30000)))
        }
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else {
            throw OfflineFixtureError.encoding(String(describing: writer.error))
        }
    }

    /// Damages compressed samples while preserving track metadata and sample tables.
    private static func corruptMedia(_ source: URL, to target: URL) throws {
        var bytes = try Data(contentsOf: source)
        var offset = 0
        var corrupted = false
        while offset + 8 <= bytes.count {
            let shortSize = bytes[offset..<offset + 4].reduce(UInt64(0)) {
                ($0 << 8) | UInt64($1)
            }
            let type = String(decoding: bytes[offset + 4..<offset + 8], as: UTF8.self)
            let header = shortSize == 1 ? 16 : 8
            precondition(offset + header <= bytes.count)
            let size = shortSize == 1
                ? bytes[offset + 8..<offset + 16].reduce(UInt64(0)) {
                    ($0 << 8) | UInt64($1)
                } : (shortSize == 0 ? UInt64(bytes.count - offset) : shortSize)
            precondition(size >= UInt64(header) && size <= UInt64(bytes.count - offset))
            let end = offset + Int(size)
            if type == "mdat" {
                bytes.replaceSubrange(
                    offset + header..<end,
                    with: repeatElement(UInt8(255), count: Int(size) - header))
                corrupted = true
            }
            offset = end
        }
        precondition(corrupted)
        try bytes.write(to: target)
    }
}

/// Supplies measurable delays, controlled failures and overlap detection.
private actor OfflineFixtureDetector: ObjectDetector {
    let delay: Duration
    let failFrame: Int?
    let gate: OfflineGate?
    private var active = 0
    private(set) var maximumActive = 0
    private(set) var received: [Int] = []
    private(set) var loads = 0

    /// Configures a slow/failing or manually suspended inference operation.
    init(
        delay: Duration = .milliseconds(1), failFrame: Int? = nil,
        gate: OfflineGate? = nil
    ) {
        self.delay = delay
        self.failFrame = failFrame
        self.gate = gate
    }

    /// Counts adapter loading so frame requests cannot conceal repeated model loads.
    func loaded() { loads += 1 }

    /// Identifies frames by real decoded pixels, independently of their source index.
    func detect(in image: CGImage) async throws -> [Detection] {
        active += 1
        maximumActive = max(maximumActive, active)
        defer { active -= 1 }
        let rgb = try RGBImage(image)
        let index = (Int(rgb.pixels[0]) - 16 + 32) / 64
        received.append(index)
        await gate?.holdOnce()
        try await Task.sleep(for: delay)
        if index == failFrame { throw OfflineFixtureError.inference }
        if index == 1 { return [] }
        return [try Detection(
            boundingBox: CGRect(x: 0, y: 0, width: image.width, height: image.height),
            confidence: 0.9, classID: 1)]
    }
}

/// Stores immutable results at an actual consumer boundary.
private actor OfflineCollector {
    private(set) var records: [VideoFrameResult] = []

    /// Buffers successful, empty and failed outcomes for order and identity assertions.
    func append(_ result: VideoFrameResult) {
        records.append(result)
    }
}

/// Suspends exactly the first request without racing wall-clock sleeps.
private actor OfflineGate {
    private let request: Int
    private var calls = 0
    private var held = false
    private var pending: CheckedContinuation<Void, Never>?
    private var notification: CheckedContinuation<Void, Never>?

    /// Selects the exact call to suspend, including cancellation after a prefix.
    init(request: Int = 1) { self.request = request }

    /// Blocks one inference/consumer call until the check releases it.
    func holdOnce() async {
        guard !held else { return }
        calls += 1
        guard calls == request else { return }
        held = true
        await withCheckedContinuation {
            pending = $0
            notification?.resume()
            notification = nil
        }
    }

    /// Signals exactly when the protected operation is in flight.
    func waitForRequest() async {
        if held { return }
        await withCheckedContinuation { notification = $0 }
    }

    /// Allows the active operation and its caller to unwind.
    func finish() {
        pending?.resume()
        pending = nil
    }
}

/// Separates test encoder, model and consumer failures from reader failures.
nonisolated private enum OfflineFixtureError: Error {
    case inference, consumer, conversion
    case encoding(String)
}
