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
        print("Offline video: 7 selection, sequencing and error checks passed.")
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
        let empty = try await runner.run(
            video, sourceID: "empty", model: .rfdetrSmall,
            selection: .indices([]), consume: { _ in
                preconditionFailure("Empty selection produced a result")
            })
        precondition(empty.processedFrames == 0 && empty.decodedFrames == 0)
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
        do {
            _ = try await runner.run(
                video, sourceID: "overlap", model: .rfdetrSmall, consume: { _ in })
            preconditionFailure("Overlapping runs were accepted")
        } catch OfflineVideoError.alreadyRunning {}
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
        let restarted = try await runner.run(
            video, sourceID: "restart", model: .rfdetrSmall,
            selection: .indices([0]), consume: { await collector.append($0) })
        precondition(restarted.processedFrames == 1)
        let retained = await collector.records
        precondition(retained.count == 1 && retained[0].metadata.sourceID == "restart")
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
        let collector = OfflineCollector()
        do {
            _ = try await runner.run(
                video, sourceID: "missing", model: .rfdetrSmall,
                selection: .indices([0, 8]), consume: { await collector.append($0) })
            preconditionFailure("Incomplete selection was reported as completed")
        } catch OfflineVideoError.missingSelectedFrame(8) {}
        let partial = await collector.records
        precondition(partial.map { $0.metadata.sourceFrameIndex } == [0])
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
    }

    /// Counts model loads independently of per-frame inference calls.
    private static func makeRunner(
        _ backend: OfflineFixtureDetector
    ) -> OfflineVideoRunner {
        OfflineVideoRunner(engine: DetectorEngine(loader: { _ in
            await backend.loaded()
            return backend
        }))
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
    private var held = false
    private var pending: CheckedContinuation<Void, Never>?
    private var notification: CheckedContinuation<Void, Never>?

    /// Blocks one inference/consumer call until the check releases it.
    func holdOnce() async {
        guard !held else { return }
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
    case inference, consumer
    case encoding(String)
}
