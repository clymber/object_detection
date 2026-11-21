import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation

/// Exercises real decoding with virtual and production monotonic playback clocks.
@main
@MainActor
struct PacedVideoSourceSmokeTests {
    private static let timestamps: [Int64] = [0, 1000, 4000, 6000]

    /// Writes private H.264 fixtures and validates the source without model inference.
    static func main() async throws {
        let directory = URL(fileURLWithPath:
            "outputs/ios_builds/video_pipeline/paced_fixtures")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let video = directory.appendingPathComponent("irregular.mov")
        let sparse = directory.appendingPathComponent("sparse.mov")
        try await writeVideo(to: video, times: timestamps, end: 9000)
        try await writeVideo(to: sparse, times: [0, 30000], end: 60000)
        try await testCadenceAndLoops(video)
        try await testLateSourceDoesNotDrift(video)
        try await testRestart(video)
        try await testExplicitCancellation(sparse, restart: video)
        try await testTaskCancellation(sparse)
        try await testFailures(video)
        try await testContinuousLoopCancellation(video)
        try await testRealClock(video)
        try await testBoundedDetectorHandoff(video)
        print("Paced video source: all 9 checks passed.")
    }

    /// Checks actual PTS, rotation, frame identity and the complete final-frame span.
    private static func testCadenceAndLoops(_ url: URL) async throws {
        let clock = TestPlaybackClock()
        let recorder = PacedFrameRecorder()
        let source = PacedVideoSource(clock: clock.playbackClock)
        let result = try await source.run(
            url, sourceID: "fixture", firstLoopIndex: 4, loopCount: 3) { frame in
                precondition(!Thread.isMainThread)
                recorder.append(frame)
            }
        precondition(result == PacedVideoSourceSummary(
            deliveredFrames: 12, completedLoops: 3))
        let frames = recorder.frames
        precondition(frames.count == 12)
        let timeline = await source.timeline!
        precondition(timeline.sourceTimeRange.duration == CMTime(value: 9000,
                                                                timescale: 30000))
        for (offset, event) in frames.enumerated() {
            let index = offset % 4
            let loop = offset / 4
            let metadata = event.frame.metadata
            precondition(metadata.sourceID == "fixture")
            precondition(metadata.sourceFrameIndex == index
                         && metadata.loopIndex == loop + 4)
            precondition(metadata.presentationTime == CMTime(
                value: timestamps[index], timescale: 30000))
            precondition(event.frame.image.width == 32
                         && event.frame.image.height == 64)
            let expected = 1000 + Double(loop * 9000 + Int(timestamps[index])) / 30
            precondition(abs(event.dueMilliseconds - expected) < 1e-9)
            precondition(event.deliveredMilliseconds == event.dueMilliseconds)
        }
        precondition(abs(clock.now - 1900) < 1e-9)
        let state = await source.state
        precondition(state == .completed)
    }

    /// Decode, callback and wake-up delays keep fixed deadlines and retain late frames.
    private static func testLateSourceDoesNotDrift(_ url: URL) async throws {
        let clock = TestPlaybackClock(oversleep: 7)
        let recorder = PacedFrameRecorder()
        let source = PacedVideoSource(clock: clock.playbackClock) { frame in
            clock.advance(by: 5)
            return try VideoFrameConverter.convert(frame)
        }
        _ = try await source.run(url, sourceID: "fixture", loopCount: 2) { event in
            recorder.append(event)
            if event.frame.metadata.sourceFrameIndex == 0 {
                clock.advance(by: 150)
            }
        }
        let frames = recorder.frames
        precondition(frames.count == 8, "Pacing silently dropped late source frames")
        for (offset, event) in frames.enumerated() {
            let expected = 1005 + Double((offset / 4) * 9000
                + Int(timestamps[offset % 4])) / 30
            precondition(abs(event.dueMilliseconds - expected) < 1e-9)
            precondition(event.deliveredMilliseconds >= event.dueMilliseconds)
        }
        precondition(frames[1].deliveredMilliseconds > frames[1].dueMilliseconds + 100)
        precondition(frames[4].dueMilliseconds == 1305)
    }

    /// A new run gets a new anchor and identity without retaining previous loop state.
    private static func testRestart(_ url: URL) async throws {
        let clock = TestPlaybackClock()
        let source = PacedVideoSource(clock: clock.playbackClock)
        _ = try await source.run(url, sourceID: "old", firstLoopIndex: 2) { _ in }
        let recorder = PacedFrameRecorder()
        _ = try await source.run(url, sourceID: "new", firstLoopIndex: 8) {
            recorder.append($0)
        }
        let first = recorder.frames[0]
        precondition(first.dueMilliseconds == 1300)
        precondition(first.frame.metadata.sourceID == "new")
        precondition(first.frame.metadata.loopIndex == 8)
        let timeline = await source.timeline
        precondition(timeline?.firstLoopIndex == 8)
    }

    /// Explicit Stop interrupts deadlines, rejects overlap and permits restart.
    private static func testExplicitCancellation(
        _ url: URL, restart: URL
    ) async throws {
        let source = PacedVideoSource()
        let recorder = PacedFrameRecorder()
        let task = Task { try await source.run(url, sourceID: "cancel") {
            recorder.append($0)
        } }
        try await waitForFrames(recorder, count: 1)
        do {
            _ = try await source.run(url, sourceID: "overlap") { _ in }
            preconditionFailure("Overlapping source run succeeded")
        } catch VideoPlaybackError.alreadyRunning {}
        let stopped = ContinuousClock.now
        await source.cancel()
        do {
            _ = try await task.value
            preconditionFailure("Stopped source completed")
        } catch is CancellationError {}
        precondition(stopped.duration(to: .now) < .milliseconds(500))
        precondition(recorder.frames.count == 1)
        let state = await source.state
        precondition(state == .cancelled)
        _ = try await source.run(restart, sourceID: "restarted") { _ in }
    }

    /// Parent-task cancellation also interrupts the independently retained sleep task.
    private static func testTaskCancellation(_ url: URL) async throws {
        let source = PacedVideoSource()
        let recorder = PacedFrameRecorder()
        let task = Task { try await source.run(url, sourceID: "task-cancel") {
            recorder.append($0)
        } }
        try await waitForFrames(recorder, count: 1)
        task.cancel()
        do {
            _ = try await task.value
            preconditionFailure("Cancelled task completed")
        } catch is CancellationError {}
        precondition(recorder.frames.count == 1)
        let state = await source.state
        let running = await source.isRunning
        precondition(state == .cancelled && !running)
    }

    /// Conversion, callback and source errors terminate without hiding accepted frames.
    private static func testFailures(_ url: URL) async throws {
        let clock = TestPlaybackClock()
        let source = PacedVideoSource(clock: clock.playbackClock) { frame in
            if frame.metadata.sourceFrameIndex == 1 { throw FixtureFailure.expected }
            return try VideoFrameConverter.convert(frame)
        }
        do {
            _ = try await source.run(url, sourceID: "conversion") { _ in }
            preconditionFailure("Conversion error was hidden")
        } catch FixtureFailure.expected {}
        let summary = await source.summary
        let state = await source.state
        precondition(summary.deliveredFrames == 1 && state == .failed)
        let normal = PacedVideoSource(clock: clock.playbackClock)
        do {
            _ = try await normal.run(url, sourceID: "consumer") { _ in
                throw FixtureFailure.expected
            }
            preconditionFailure("Consumer error was hidden")
        } catch FixtureFailure.expected {}
        for count in [0, -1] {
            do {
                _ = try await normal.run(
                    url, sourceID: "invalid", loopCount: count) { _ in }
                preconditionFailure("Invalid loop count succeeded")
            } catch VideoPlaybackError.invalidLoop {}
        }
        do {
            _ = try await normal.run(
                url.appendingPathExtension("missing"), sourceID: "missing") { _ in }
            preconditionFailure("Missing source succeeded")
        } catch is VideoReaderError {}
        _ = try await normal.run(url, sourceID: "recovered") { _ in }
    }

    /// Unbounded looping retains source identity and remains cancellable at handoff.
    private static func testContinuousLoopCancellation(_ url: URL) async throws {
        let clock = TestPlaybackClock()
        let source = PacedVideoSource(clock: clock.playbackClock)
        let recorder = PacedFrameRecorder()
        let task = Task {
            try await source.run(url, sourceID: "continuous", loopCount: nil) { event in
                recorder.append(event)
                if event.frame.metadata.loopIndex == 2 {
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            }
        }
        do {
            _ = try await task.value
            preconditionFailure("Continuous playback ignored cancellation")
        } catch is CancellationError {}
        precondition(recorder.frames.count == 9)
        precondition(recorder.frames.last?.frame.metadata.loopIndex == 2)
    }

    /// Uses cancellable production sleeps and the inference clock across two loops.
    private static func testRealClock(_ url: URL) async throws {
        let source = PacedVideoSource()
        let recorder = PacedFrameRecorder()
        let (stream, continuation) = AsyncStream<PacedVideoFrame>.makeStream(
            bufferingPolicy: .bufferingNewest(1))
        let detector = PacedFrameRecorder()
        let started = PacedFrameRecorder()
        let processing = Task {
            for await event in stream {
                started.append(event)
                try await Task.sleep(for: .seconds(3))
                detector.append(event)
            }
        }
        defer {
            continuation.finish()
            processing.cancel()
        }
        _ = try await source.run(url, sourceID: "real-clock", loopCount: 2) {
            recorder.append($0)
            continuation.yield($0)
        }
        let timeline = await source.timeline!
        let elapsed = InferenceClock.nowMilliseconds() - timeline.anchorMilliseconds
        precondition(elapsed >= 600 && elapsed < 1500)
        for event in recorder.frames {
            precondition(event.deliveredMilliseconds >= event.dueMilliseconds)
            precondition(event.deliveredMilliseconds - event.dueMilliseconds < 500)
            precondition(event.frame.readyMilliseconds <= event.deliveredMilliseconds)
        }
        precondition(recorder.frames[4].dueMilliseconds
                     == timeline.anchorMilliseconds + 300)
        precondition(recorder.frames.count == 8 && detector.frames.isEmpty,
                     "The source waited for slow asynchronous inference")
        precondition(started.frames.count == 1, "Slow inference was never in flight")
    }

    /// Decoded frames enter the bounded queue while a real detector adapter is held.
    private static func testBoundedDetectorHandoff(_ url: URL) async throws {
        let detector = PacedQueueDetectorFixture()
        let queue = PacedVideoScheduler(
            detector: VideoFrameDetector(detector: detector))
        let published = PacedResultRecorder()
        let worker = Task {
            try await queue.run { delivery in
                try await published.receive(delivery)
            }
        }
        let source = PacedVideoSource()
        let supplied = try await source.run(
            url, sourceID: "queue-integration", firstLoopIndex: 6, loopCount: 2) {
                try queue.offer($0)
            }
        precondition(supplied.deliveredFrames == timestamps.count * 2)
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while await detector.calls == 0 && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        let held = queue.snapshot
        precondition(held.acceptedFrames == timestamps.count * 2
                     && held.activeFrames == 1)
        precondition(held.pendingFrames <= 1)
        precondition(held.outcomes.allSatisfy { $0.status == .skippedSuperseded })
        queue.finishInput()
        await detector.release()
        let result = try await worker.value
        precondition(!published.received.isEmpty)
        precondition(result.state == .completed
                     && result.outcomes.count == timestamps.count * 2)
        precondition(result.activeFrames == 0 && result.pendingFrames == 0)
        let sorted = result.outcomes.sorted {
            ($0.metadata.loopIndex, $0.metadata.sourceFrameIndex)
                < ($1.metadata.loopIndex, $1.metadata.sourceFrameIndex)
        }
        for (offset, outcome) in sorted.enumerated() {
            let index = offset % timestamps.count
            precondition(outcome.metadata.sourceID == "queue-integration")
            precondition(outcome.metadata.loopIndex == 6 + offset / timestamps.count)
            precondition(outcome.metadata.sourceFrameIndex == index)
            precondition(outcome.metadata.presentationTime
                         == CMTime(value: timestamps[index], timescale: 30000))
            if let inference = outcome.result {
                precondition(inference.sourceWidth == 32
                             && inference.sourceHeight == 64)
                precondition(inference.status == .completed
                             && inference.detections.isEmpty)
                precondition(inference.timings.publishedMilliseconds == nil)
                precondition(outcome.publication.status == .published
                             || outcome.publication.status == .late
                             || outcome.publication.status == .superseded)
            }
            if outcome.publication.status == .published {
                precondition(outcome.publishedAgeMilliseconds! <= 200)
                let receipt = published.received.first {
                    $0.metadata == outcome.metadata
                }
                precondition(receipt?.timings.publishedMilliseconds
                             == outcome.publication.publishedMilliseconds)
            }
        }
    }

    /// Polls a locked observation with a bounded timeout while the source awaits PTS.
    private static func waitForFrames(
        _ recorder: PacedFrameRecorder, count: Int
    ) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while recorder.frames.count < count && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        precondition(recorder.frames.count >= count,
                     "Source did not deliver its first frame")
    }

    /// Writes distinguishable rotated frames and an explicit tail presentation span.
    private static func writeVideo(
        to url: URL, times: [Int64], end: Int64
    ) async throws {
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
        guard writer.startWriting() else { throw FixtureFailure.encoding }
        writer.startSession(atSourceTime: .zero)
        for (index, pts) in times.enumerated() {
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
            memset(CVPixelBufferGetBaseAddress(pixels), Int32(16 + index * 40),
                   CVPixelBufferGetBytesPerRow(pixels) * 32)
            CVPixelBufferUnlockBaseAddress(pixels, [])
            precondition(adaptor.append(
                pixels, withPresentationTime: CMTime(value: pts, timescale: 30000)))
        }
        writer.endSession(atSourceTime: CMTime(value: end, timescale: 30000))
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else { throw FixtureFailure.encoding }
    }
}

/// Protects virtual time used by the decoder, pacing task and synchronous consumer.
nonisolated private final class TestPlaybackClock: @unchecked Sendable {
    private let lock = NSLock()
    private var milliseconds = 1000.0
    private let oversleep: Double

    /// Optionally models a scheduler that wakes later than the requested deadline.
    init(oversleep: Double = 0) { self.oversleep = oversleep }

    var now: Double { lock.withLock { milliseconds } }

    var playbackClock: VideoPlaybackClock {
        VideoPlaybackClock(nowMilliseconds: { self.now }, sleepUntil: { deadline in
            try Task.checkCancellation()
            self.lock.withLock {
                self.milliseconds = max(self.milliseconds, deadline) + self.oversleep
            }
        })
    }

    /// Models work without making deterministic scheduling checks wait in real time.
    func advance(by milliseconds: Double) {
        lock.withLock { self.milliseconds += milliseconds }
    }
}

/// Retains test events under a lock without requiring a UI or async consumer hop.
nonisolated private final class PacedFrameRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [PacedVideoFrame] = []
    var frames: [PacedVideoFrame] { lock.withLock { stored } }

    /// Snapshots accepted events to inspect source identity and host deadlines.
    func append(_ frame: PacedVideoFrame) { lock.withLock { stored.append(frame) } }
}

/// Separates deliberate downstream failures from fixture creation errors.
nonisolated private enum FixtureFailure: Error { case expected, encoding }

/// Keeps inference active while real source decoding and cadence continue to EOF.
private actor PacedQueueDetectorFixture: ObjectDetector {
    private var waiting: CheckedContinuation<Void, Never>?
    private var released = false
    private var active = false
    private(set) var calls = 0

    /// Returns empty detections only after the test releases its first active request.
    func detect(in image: CGImage) async throws -> [Detection] {
        precondition(!active, "Real video handoff overlapped inference")
        active = true
        calls += 1
        if !released { await withCheckedContinuation { waiting = $0 } }
        active = false
        return []
    }

    /// Lets the held inference finish and permits the queue to drain its pending tail.
    func release() {
        released = true
        waiting?.resume()
        waiting = nil
    }
}

/// Receives numerical results with preview off after the real MainActor handoff.
@MainActor
private final class PacedResultRecorder {
    var received: [VideoFrameResult] = []

    /// Grants receipt only while the scheduler still considers this result fresh.
    func receive(_ delivery: PacedVideoDelivery) throws {
        if let result = try delivery.receive() { received.append(result) }
    }
}
