import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation

/// Exercises the actual source/queue/consumer lifecycle with pixel-free run reports.
@main
@MainActor
struct PacedVideoRunnerSmokeTests {
    /// Builds a small real clip and validates final metrics, progress and failures.
    static func main() async throws {
        let directory = URL(fileURLWithPath:
            "outputs/ios_builds/video_pipeline/paced_runner_fixtures")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let video = directory.appendingPathComponent("runner.mov")
        try await writeVideo(to: video)
        try await testCompletedRun(video)
        try await testStopProgressAndRestart(video)
        try await testTaskCancellation(video)
        try await testModelAndSourceFailures(video)
        try await testConversionAndConsumerFailures(video)
        try await testDecodeFailure(video)
        print("Paced runner: 6 source, progress, report and lifecycle checks passed.")
    }

    /// Defaults to numerical receipt with preview off and measures the source interval.
    private static func testCompletedRun(_ url: URL) async throws {
        let detector = RunnerDetector()
        let runner = PacedVideoRunner(engine: DetectorEngine(loader: { _ in detector }))
        let report = try await runner.run(
            url, sourceID: "runner", model: .rfdetrSmall,
            firstLoopIndex: 6, loopCount: 2)
        precondition(report.status == .completed && report.issue == nil)
        precondition(report.source?.decodedFrames == 8)
        precondition(report.source?.deliveredFrames == 8
                     && report.source?.completedLoops == 2)
        precondition(report.counts!.inputFrames == 8)
        precondition(report.counts!.processedFrames + report.counts!.skippedFrames == 8)
        precondition(report.counts!.unfinishedFrames == 0)
        precondition(report.decodedButNotDeliveredFrames == 0)
        precondition(report.decodeLosses.isEmpty)
        let metrics = report.metrics!
        precondition(metrics.window.durationMilliseconds >= 600)
        precondition(metrics.window.endedMilliseconds
                     == report.source!.endedMilliseconds)
        precondition(metrics.window.endedMilliseconds <= report.endedMilliseconds!)
        precondition(metrics.inputFramesPerSecond! > 0)
        precondition(metrics.freshResultsPerSecond! > 0)
        precondition(metrics.invalidPublications == 0)
        precondition(metrics.displayFramesPerSecond == nil)
        for outcome in report.queue!.outcomes {
            precondition((6...7).contains(outcome.metadata.loopIndex))
            precondition(outcome.metadata.sourceID == "runner")
            if outcome.status == .completed {
                precondition(outcome.result?.detections.isEmpty == true)
            }
            if outcome.publication.status == .published {
                precondition(outcome.publishedAgeMilliseconds! <= 200)
            }
        }
        let running = await runner.isRunning
        let progress = await runner.progress()
        precondition(!running && progress == nil)
    }

    /// Current counts reconcile while work waits; Stop retains post-window completion.
    private static func testStopProgressAndRestart(_ url: URL) async throws {
        let detector = RunnerDetector(hold: true)
        let runner = PacedVideoRunner(engine: DetectorEngine(loader: { _ in detector }))
        let worker = Task { try await runner.run(
            url, sourceID: "held", model: .rfdetrSmall) }
        try await waitForActive(detector)
        do {
            _ = try await runner.run(url, sourceID: "overlap", model: .yoloxNano)
            preconditionFailure("Overlapping paced run succeeded")
        } catch VideoPlaybackError.alreadyRunning {}
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        var progress = await runner.progress()
        while progress?.source?.status != .completed && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
            progress = await runner.progress()
        }
        precondition(progress?.source?.status == .completed)
        let counts = progress!.counts
        precondition(counts.inputFrames == 4 && counts.processedFrames == 0)
        precondition(counts.inputFrames == counts.skippedFrames
                     + counts.activeFrames + counts.pendingFrames)
        await runner.cancel()
        await detector.release()
        do {
            _ = try await worker.value
            preconditionFailure("Stopped paced run returned success")
        } catch is CancellationError {}
        let report = await runner.latestReport!
        precondition(report.status == .cancelled)
        precondition(report.counts?.completedFrames == 1)
        precondition(report.metrics?.completedFrames == 0)
        precondition(report.metrics?.postWindowCompletions == 1)
        precondition(report.metrics?.freshResults == 0)
        precondition(report.decodeLosses.isEmpty)
        let next = try await runner.run(url, sourceID: "new-run", model: .rfdetrNano)
        precondition(next.id != report.id && next.status == .completed)
        precondition(next.sourceID == "new-run" && next.counts?.inputFrames == 4)
        precondition(next.queue!.outcomes.allSatisfy {
            $0.metadata.sourceID == "new-run"
        })
    }

    /// Task cancellation wakes pacing while preserving active backend work.
    private static func testTaskCancellation(_ url: URL) async throws {
        let detector = RunnerDetector(hold: true)
        let runner = PacedVideoRunner(engine: DetectorEngine(loader: { _ in detector }))
        let worker = Task { try await runner.run(
            url, sourceID: "task-cancel", model: .rfdetrSmall) }
        try await waitForActive(detector)
        worker.cancel()
        await detector.release()
        do {
            _ = try await worker.value
            preconditionFailure("Cancelled run returned success")
        } catch is CancellationError {}
        let report = await runner.latestReport!
        precondition(report.status == .cancelled)
        precondition(report.counts?.completedFrames == 1)
        precondition(report.source?.status == .cancelled)
        precondition(report.decodeLosses.isEmpty)
        precondition(report.counts!.inputFrames == report.counts!.processedFrames
                     + report.counts!.skippedFrames + report.counts!.cancelledFrames)
    }

    /// Failures before source admission cannot invent frame loss counts or FPS.
    private static func testModelAndSourceFailures(_ url: URL) async throws {
        let failedLoad = PacedVideoRunner(engine: DetectorEngine(loader: { _ in
            throw RunnerFixtureFailure.expected
        }))
        do {
            _ = try await failedLoad.run(url, sourceID: "load", model: .rfdetrSmall)
            preconditionFailure("Model-loading failure was hidden")
        } catch RunnerFixtureFailure.expected {}
        let load = await failedLoad.latestReport!
        precondition(load.status == .failed
                     && load.source == nil && load.metrics == nil)
        precondition(load.counts?.inputFrames == 0 && load.decodeLosses.isEmpty)
        let detector = RunnerDetector()
        let missing = PacedVideoRunner(
            engine: DetectorEngine(loader: { _ in detector }))
        do {
            _ = try await missing.run(
                url.appendingPathExtension("missing"), sourceID: "missing",
                model: .rfdetrSmall)
            preconditionFailure("Missing video source succeeded")
        } catch is VideoReaderError {}
        let report = await missing.latestReport!
        precondition(report.status == .failed && report.metrics == nil)
        precondition(report.source?.issue?.stage == .opening)
        precondition(report.decodeLosses.isEmpty)
    }

    /// Distinguishes converter/consumer failures from decoder losses.
    private static func testConversionAndConsumerFailures(_ url: URL) async throws {
        let detector = RunnerDetector()
        let runner = PacedVideoRunner(
            engine: DetectorEngine(loader: { _ in detector }), converter: { frame in
                if frame.metadata.sourceFrameIndex == 1 {
                    throw VideoReaderError.decodeFailed(
                        frameIndex: 1, reason: "adapter")
                }
                return try VideoFrameConverter.convert(frame)
            })
        do {
            _ = try await runner.run(url, sourceID: "conversion", model: .rfdetrSmall)
            preconditionFailure("Converter failure was hidden")
        } catch is VideoReaderError {}
        let report = await runner.latestReport!
        precondition(report.status == .failed && report.decodeLosses.isEmpty)
        precondition(report.source?.issue?.stage == .conversion)
        precondition(report.source?.issue?.metadata?.sourceFrameIndex == 1)
        precondition(report.source?.issue?.presentationTime
                     == CMTime(value: 1000, timescale: 30000))
        precondition(report.source?.issue?.dueMilliseconds != nil)
        precondition(report.source?.decodedFrames == 2
                     && report.counts?.inputFrames == 1)
        precondition(report.decodedButNotDeliveredFrames == 1)
        let normal = PacedVideoRunner(engine: DetectorEngine(loader: { _ in detector }))
        let completed = try await normal.run(
            url, sourceID: "consumer", model: .rfdetrSmall, consume: { _ in
                throw RunnerFixtureFailure.expected
            })
        precondition(completed.status == .completed)
        precondition(completed.counts!.failedFrames == 0)
        precondition(completed.counts!.failedPublications > 0)
        precondition(completed.metrics?.freshResultsPerSecond == 0)
    }

    /// Decoder interruptions retain known timing and leave unknown tails uncounted.
    private static func testDecodeFailure(_ url: URL) async throws {
        for knownPTS in [false, true] {
            let detector = RunnerDetector()
            let runner = PacedVideoRunner(
                engine: DetectorEngine(loader: { _ in detector }),
                readerFactory: { FailingRunnerReader(knownPTS: knownPTS) })
            do {
                _ = try await runner.run(
                    url, sourceID: "decode-failure", model: .rfdetrSmall,
                    firstLoopIndex: 4)
                preconditionFailure("Decoder failure was hidden")
            } catch is VideoReaderError {}
            let report = await runner.latestReport!
            precondition(report.status == .failed
                         && report.hasUnreconstructedDecodeLosses)
            precondition(report.source?.decodedFrames == 1
                         && report.counts?.inputFrames == 1)
            precondition(report.decodeLosses.count == 1)
            let loss = report.decodeLosses[0]
            precondition(loss.stage == .decoding && loss.sourceID == "decode-failure")
            precondition(loss.loopIndex == 4 && loss.attemptedFrameIndex == 1)
            let pts = knownPTS ? CMTime(value: 1001, timescale: 30000) : nil
            precondition(loss.presentationTime == pts)
            precondition(loss.lostFrameCount == nil && loss.metadata == nil)
            precondition((loss.dueMilliseconds != nil) == knownPTS)
            precondition(loss.observedMilliseconds != nil && !loss.cancelled)
        }
    }

    /// Synchronizes cancellation only after inference has actually entered the backend.
    private static func waitForActive(_ detector: RunnerDetector) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while await detector.calls == 0 && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        let calls = await detector.calls
        precondition(calls == 1)
    }

    /// Writes four rotated frames with irregular PTS and an explicit final-frame span.
    private static func writeVideo(to url: URL) async throws {
        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 64, AVVideoHeightKey: 32,
        ])
        input.mediaTimeScale = 30000
        input.transform = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 32, ty: 0)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input, sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: 64,
                kCVPixelBufferHeightKey as String: 32,
            ])
        writer.add(input)
        guard writer.startWriting() else { throw RunnerFixtureFailure.encoding }
        writer.startSession(atSourceTime: .zero)
        for pts in [Int64(0), 1000, 4000, 6000] {
            let deadline = ContinuousClock.now.advanced(by: .seconds(10))
            while !input.isReadyForMoreMediaData && ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(2))
            }
            precondition(input.isReadyForMoreMediaData)
            var buffer: CVPixelBuffer?
            precondition(CVPixelBufferCreate(
                kCFAllocatorDefault, 64, 32, kCVPixelFormatType_32BGRA,
                nil, &buffer) == kCVReturnSuccess)
            let pixels = buffer!
            precondition(CVPixelBufferLockBaseAddress(pixels, []) == kCVReturnSuccess)
            memset(CVPixelBufferGetBaseAddress(pixels), 80,
                   CVPixelBufferGetBytesPerRow(pixels) * 32)
            CVPixelBufferUnlockBaseAddress(pixels, [])
            precondition(adaptor.append(
                pixels, withPresentationTime: CMTime(value: pts, timescale: 30000)))
        }
        writer.endSession(atSourceTime: CMTime(value: 9000, timescale: 30000))
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else { throw RunnerFixtureFailure.encoding }
    }
}

/// Controls the detector adapter independently of real source decoding.
private actor RunnerDetector: ObjectDetector {
    private var hold: Bool
    private var waiting: CheckedContinuation<Void, Never>?
    private(set) var calls = 0

    /// Holds the first request when checking progress, cancellation and drain.
    init(hold: Bool = false) { self.hold = hold }

    /// Returns successful empty results, validating upright image geometry off the UI.
    func detect(in image: CGImage) async throws -> [Detection] {
        precondition(!Thread.isMainThread && image.width == 32 && image.height == 64)
        calls += 1
        if hold { await withCheckedContinuation { waiting = $0 } }
        return []
    }

    /// Releases the current request and lets later runs complete without a held gate.
    func release() {
        hold = false
        waiting?.resume()
        waiting = nil
    }
}

/// Distinguishes expected pipeline failures from codec fixture setup failures.
nonisolated private enum RunnerFixtureFailure: Error { case expected, encoding }

/// Emits one real pixel buffer, then a failed sample whose remaining extent is unknown.
private actor FailingRunnerReader: VideoFrameReading {
    var selectedTrackID: CMPersistentTrackID? = 1
    var selectedTimeRange: CMTimeRange? = CMTimeRange(
        start: .zero, duration: CMTime(value: 1, timescale: 1))
    var framesRead = 0
    var lastFrameInfo: VideoFrameReadInfo?
    var decodeFailureInfo: VideoFrameDecodeFailureInfo?
    var failure: VideoReaderError?
    private var sourceID = ""
    private var loopIndex = 0
    private let knownPTS: Bool

    /// Models a failed sample with a known PTS or a decoder failure without a sample.
    init(knownPTS: Bool) { self.knownPTS = knownPTS }

    /// Binds the fake decoder to the same run/loop identity as a real reader.
    func open(
        _ url: URL, sourceID: String, loopIndex: Int, trackID: CMPersistentTrackID?
    ) async throws {
        self.sourceID = sourceID
        self.loopIndex = loopIndex
        framesRead = 0
    }

    /// Exercises the real conversion callback before simulating a malformed sample.
    func nextFrame<Result: Sendable>(
        _ process: @Sendable (VideoFrame) throws -> Result
    ) throws -> Result? {
        if framesRead == 1 {
            failure = knownPTS
                ? .invalidFrame(frameIndex: 1, reason: "Decoder fixture interruption")
                : .decodeFailed(frameIndex: 1, reason: "Decoder fixture interruption")
            decodeFailureInfo = VideoFrameDecodeFailureInfo(
                sourceFrameIndex: 1,
                presentationTime: knownPTS
                    ? CMTime(value: 1001, timescale: 30000) : nil)
            throw failure!
        }
        var pixels: CVPixelBuffer?
        precondition(CVPixelBufferCreate(
            kCFAllocatorDefault, 64, 32, kCVPixelFormatType_32BGRA,
            nil, &pixels) == kCVReturnSuccess)
        precondition(CVPixelBufferLockBaseAddress(pixels!, []) == kCVReturnSuccess)
        memset(CVPixelBufferGetBaseAddress(pixels!), 80,
               CVPixelBufferGetBytesPerRow(pixels!) * 32)
        CVPixelBufferUnlockBaseAddress(pixels!, [])
        let frame = try VideoFrame(
            sourceID: sourceID, sourceFrameIndex: 0, presentationTime: .zero,
            preferredTransform: CGAffineTransform(
                a: 0, b: 1, c: -1, d: 0, tx: 32, ty: 0),
            loopIndex: loopIndex, pixelBuffer: pixels!)
        framesRead = 1
        lastFrameInfo = VideoFrameReadInfo(
            metadata: frame.metadata, readyMilliseconds: frame.readyMilliseconds)
        return try process(frame)
    }

    /// Preserves the diagnostic after cleanup, matching the production reader contract.
    func cancel() {}
}
