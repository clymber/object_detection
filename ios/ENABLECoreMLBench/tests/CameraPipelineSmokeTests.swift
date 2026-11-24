import CoreGraphics
import CoreMedia
import Foundation

/// Tests live capture contracts, permissions, lifecycle, bounded work and export without hardware.
@main
@MainActor
struct CameraPipelineSmokeTests {
    /// Leaves real camera-schema artifacts for independent Python readback.
    static func main() async throws {
        try testClock()
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let output = root.appendingPathComponent(
            "outputs/ios_builds/video_pipeline/camera_exports")
        try? FileManager.default.removeItem(at: output)
        let exporter = VideoRunExporter(directory: output)
        var paths: [String: String] = [:]
        let session = try await testPermissionAndPreview(exporter: exporter)
        paths["session"] = session.export.files!.directory.path
        let denied = try await testDenied(exporter: exporter)
        paths["denied"] = denied.directory.path
        paths["recorded"] = try await testRecording(
            exporter: exporter, directory: output).directory.path
        paths["recording_failure"] = try await testRecording(
            exporter: exporter, directory: output, failRecording: true).directory.path
        try await testBackgroundPermissionCancellation()
        for (name, report) in [
            "drops": try await testBoundedCapture(),
            "stale": try await testStaleCapture(),
            "source_failure": try await testSourceFailure(),
            "stopped": try await testStopDuringInference(),
        ] {
            let files = try await exporter.export(.camera(report), sourceURL: nil)
            let saved = output.appendingPathComponent("copy_\(name)")
            try FileManager.default.copyItem(at: files.directory, to: saved)
            _ = try VideoExportVerifier.verify(saved, expected: files.fingerprints)
            paths[name] = saved.path
        }
        try JSONSerialization.data(withJSONObject: paths).write(
            to: output.appendingPathComponent("fixture_index.json"))
        print(
            "Camera pipeline: clock, permissions, lifecycle, exact previews, drops, bounded work and exports passed"
        )
    }

    /// Ensures mapping uses actual host capture time rather than callback arrival or FPS.
    static func testClock() throws {
        let mapping = try CameraClockMapping(
            host: CMTime(value: 10, timescale: 1), before: 5000, after: 5002)
        let mapped = try mapping.map(CMTime(value: 10033, timescale: 1000))
        precondition(abs(mapped - 5034) < 1e-8 && mapping.uncertaintyMilliseconds == 1)
        for invalid in [
            CMTime.invalid, CMTime(value: 10, timescale: 1, flags: .valid, epoch: 2),
        ] {
            do {
                _ = try mapping.map(invalid)
                preconditionFailure("Invalid clock accepted")
            } catch CameraError.invalidTimestamp {}
        }
        do {
            _ = try CameraClockMapping(host: .zero, before: 2, after: 1)
            preconditionFailure("Reversed calibration accepted")
        } catch CameraError.invalidTimestamp {}
    }

    /// Waits for permission/foreground, draws paired frames and blocks publication after Stop.
    static func testPermissionAndPreview(exporter: VideoRunExporter) async throws
        -> CameraSession
    {
        let source = FakeCameraCapture(authorization: .unknown)
        let detector = CameraTestDetector()
        let worker = CameraRunner(
            engine: DetectorEngine(loader: { _ in detector }), capture: source)
        let export = VideoExportSession(write: { run, url, _ in
            try await exporter.export(run, sourceURL: url)
        })
        let session = CameraSession(worker: worker, export: export)
        var config = CameraConfiguration()
        config.detectionPreview = true
        config.sceneNotes = "Gym / lumière, moving basketball"
        session.start(configuration: config, model: .yoloxNano)
        session.start(configuration: config, model: .rfdetrSmall)
        try await wait { source.permissionPending }
        precondition(session.status == .requestingPermission)
        session.setActivity(active: false)
        source.resolvePermission(true)
        try await wait { session.status == .waitingForActive }
        precondition(source.runs == 0)
        session.setActivity(active: true)
        try await wait { source.running }
        try source.emit()
        try await wait { session.previewFrame != nil }
        let positive = session.previewFrame!
        precondition(
            positive.result.metadata.sourceFrameIndex == 0
                && positive.result.metadata.presentationTime.epoch == 7)
        precondition(positive.result.detections.count == 1)
        let drawTime = InferenceClock.nowMilliseconds()
        session.recordDraw(
            positive.drawToken, generation: session.previewGeneration, at: drawTime)
        session.recordDraw(
            positive.drawToken, generation: session.previewGeneration, at: drawTime)
        precondition(session.previewStatistics.drawnFrames == 1)
        try source.emit()
        try await wait { session.previewFrame?.result.metadata.sourceFrameIndex == 1 }
        precondition(session.previewFrame!.result.detections.isEmpty)
        session.setActivity(active: false, background: true)
        try await wait { !session.isCapturing }
        precondition(!source.running && session.status == .stopped)
        precondition(
            session.counts.completedFrames == 2
                && session.latestReport?.status == "cancelled")
        session.exportResults()
        try await wait { !session.export.isBusy }
        precondition(
            session.export.files != nil, session.export.message ?? "No export state")
        return session
    }

    /// Preserves denied/zero-frame attempts and supports later settings-based permission retry.
    static func testDenied(exporter: VideoRunExporter) async throws -> VideoExportFiles {
        let source = FakeCameraCapture(authorization: .denied)
        let detector = CameraTestDetector()
        let worker = CameraRunner(engine: DetectorEngine(loader: { _ in detector }), capture: source)
        let session = CameraSession(
            worker: worker,
            export: VideoExportSession(
                write: { run, url, accuracy in
                    try await exporter.export(run, sourceURL: url, accuracy: accuracy) }))
        session.start(configuration: .init(), model: .rfdetrSmall)
        try await wait { !session.isCapturing }
        precondition(
            session.status == .error && session.authorization == .denied
                && source.runs == 0)
        session.exportResults()
        try await wait { !session.export.isBusy }
        precondition(
            session.export.files != nil, session.export.message ?? "No export state")
        let files = session.export.files!
        source.resolvePermission(true)
        session.setActivity(active: true)
        session.start(configuration: .init(), model: .yoloxNano)
        try await wait { source.running }
        try source.emit()
        try await wait { await detector.calls == 1 }
        session.stop()
        try await wait { !session.isCapturing }
        precondition(source.runs == 1 && session.authorization == .authorized)
        return files
    }

    /// A late permission grant cannot start the camera after the app entered background.
    static func testBackgroundPermissionCancellation() async throws {
        let source = FakeCameraCapture(authorization: .unknown)
        let session = CameraSession(worker: CameraRunner(capture: source))
        session.start(configuration: .init(), model: .yoloxNano)
        try await wait { source.permissionPending }
        session.setActivity(active: false, background: true)
        source.resolvePermission(true)
        try await wait { !session.isCapturing }
        precondition(source.runs == 0 && session.latestReport?.status == "cancelled")
    }

    /// Records with previews disabled, finalizes on Stop, exports evidence and clears files.
    static func testRecording(
        exporter: VideoRunExporter, directory: URL, failRecording: Bool = false
    ) async throws -> VideoExportFiles {
        let source = FakeCameraCapture()
        let detector = CameraTestDetector()
        let recordingDirectory = directory.appendingPathComponent(
            failRecording ? "blocked_recording" : "recordings")
        if failRecording { try Data([0]).write(to: recordingDirectory) }
        let worker = CameraRunner(
            engine: DetectorEngine(loader: { _ in detector }), capture: source,
            recordingDirectory: recordingDirectory)
        let session = CameraSession(worker: worker)
        var config = CameraConfiguration()
        config.livePreview = false
        config.detectionPreview = false
        config.recordAnnotatedVideo = true
        session.start(configuration: config, model: .yoloxNano)
        try await wait { source.running }
        try source.emit()
        try await wait { await worker.progress()?.counts.completedFrames == 1 }
        try await Task.sleep(for: .milliseconds(150))
        try source.emit()
        try await wait { await worker.progress()?.counts.completedFrames == 2 }
        session.stop()
        try await wait { !session.isCapturing }
        precondition(session.status == .stopped && session.previewFrame == nil)
        let report = session.latestReport!
        let recording = report.recording!
        precondition(recording.status == (failRecording ? .failed : .completed),
                     recording.issue?.message ?? "Recording was not finalized")
        precondition(recording.encodedFrames == (failRecording ? 0 : 2))
        if failRecording {
            precondition(recording.url == nil && recording.issue != nil)
            precondition(session.counts.completedFrames == 2)
        } else {
            precondition(FileManager.default.fileExists(atPath: recording.url!.path))
        }
        let files = try await exporter.export(.camera(report), sourceURL: nil)
        let metadata = try JSONSerialization.jsonObject(with: Data(
            contentsOf: files.directory.appendingPathComponent("mobile_run.json")))
            as! [String: Any]
        let capture = metadata["capture"] as! [String: Any]
        precondition((capture["configuration"] as! [String: Any])[
            "annotated_video_enabled"] as! Bool)
        let exportedRecording = capture["recording"] as! [String: Any]
        precondition(exportedRecording["encoded_frames"] as! Int
                     == recording.encodedFrames)
        session.reset()
        if let url = recording.url {
            precondition(!FileManager.default.fileExists(atPath: url.path))
        }
        return files
    }

    /// Stop rejects late UI publication, cancels pending input and retains active completion.
    static func testStopDuringInference() async throws -> CameraRunReport {
        let source = FakeCameraCapture()
        let detector = CameraTestDetector(held: true)
        let worker = CameraRunner(engine: DetectorEngine(loader: { _ in detector }), capture: source)
        let session = CameraSession(worker: worker)
        var config = CameraConfiguration()
        config.detectionPreview = true
        session.start(configuration: config, model: .yoloxNano)
        try await wait { source.running }
        try source.emit()
        try await wait { await detector.active == 1 }
        try source.emit()
        session.stop()
        try await wait { !source.running }
        await detector.release()
        try await wait { !session.isCapturing }
        precondition(session.previewFrame == nil && session.counts.publishedResults == 0)
        precondition(session.counts.completedFrames == 1 && session.counts.cancelledFrames == 1)
        precondition(session.metrics?.postWindowCompletions == 1)
        return session.latestReport!
    }

    /// Delivers bursts while inference is held; camera drops never become queue skips.
    static func testBoundedCapture() async throws -> CameraRunReport {
        let source = FakeCameraCapture()
        let detector = CameraTestDetector(held: true)
        let worker = CameraRunner(
            engine: DetectorEngine(loader: { _ in detector }), capture: source)
        let task = Task {
            try await worker.run(configuration: .init(), model: .rfdetrNano)
        }
        try await wait { source.running }
        try source.emit()
        try await wait { await detector.active == 1 }
        source.drop(unknown: false)
        for _ in 0..<40 { try source.emit() }
        source.drop(unknown: true)
        let progress = await worker.progress()!
        precondition(
            progress.counts.inputFrames == 41 && progress.counts.pendingFrames == 1)
        precondition(
            progress.counts.skippedSupersededFrames == 39 && progress.dropEvents == 2)
        precondition(progress.knownLostFrames == nil)
        source.stop()
        await detector.release()
        try await task.value
        let report = await worker.latestReport!
        let peak = await detector.maximumActive
        precondition(peak == 1 && report.counts.completedFrames == 2)
        precondition(report.capture?.events.count == 43)
        precondition(
            CameraExportRecord.coverage(report)["status"] as! String == "incomplete")
        return report
    }

    /// A capture already older than 200 ms is skipped even when its callback just arrived.
    static func testStaleCapture() async throws -> CameraRunReport {
        let source = FakeCameraCapture()
        let detector = CameraTestDetector()
        let worker = CameraRunner(
            engine: DetectorEngine(loader: { _ in detector }), capture: source)
        let task = Task {
            try await worker.run(configuration: .init(), model: .yoloxNano)
        }
        try await wait { source.running }
        try await Task.sleep(for: .milliseconds(260))
        try source.emit(age: 250)
        source.stop()
        try await task.value
        let report = await worker.latestReport!
        let calls = await detector.calls
        precondition(calls == 0 && report.counts.skippedStaleFrames == 1)
        return report
    }

    /// Source conversion errors retain a callback record and an exportable partial report.
    static func testSourceFailure() async throws -> CameraRunReport {
        let source = FakeCameraCapture()
        let detector = CameraTestDetector()
        let worker = CameraRunner(
            engine: DetectorEngine(loader: { _ in detector }), capture: source)
        let task = Task {
            try await worker.run(configuration: .init(), model: .yoloxNano)
        }
        try await wait { source.running }
        source.failCallback()
        do {
            try await task.value
            preconditionFailure("Source failure was hidden")
        } catch {}
        let report = await worker.latestReport!
        precondition(report.status == "failed" && report.capture?.events.count == 1)
        precondition(report.counts.inputFrames == 0)
        return report
    }

    /// Bounds async checks so a lifecycle deadlock produces a useful failure.
    static func wait(_ condition: () async -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !(await condition()) && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        let reached = await condition()
        precondition(reached, "Camera test did not reach expected state")
    }
}

/// Serializes synthetic sample callbacks and exposes permission decisions to tests.
nonisolated final class FakeCameraCapture: CameraCapturing, @unchecked Sendable {
    private let lock = NSLock()
    private var permission: CameraAuthorization
    private var permissionContinuation: CheckedContinuation<Bool, Never>?
    private var continuation: CheckedContinuation<Void, Error>?
    private var report: CameraCaptureReport?
    private var deliver: (@Sendable (PacedVideoFrame) throws -> Void)?
    private var stopped = true
    private var runCount = 0
    var preview: CameraPreviewHandle? { nil }
    var snapshot: CameraCaptureReport? { lock.withLock { report } }
    var authorization: CameraAuthorization { lock.withLock { permission } }
    var permissionPending: Bool { lock.withLock { permissionContinuation != nil } }
    var running: Bool {
        lock.withLock { (report?.isRunning ?? false) && continuation != nil }
    }
    var runs: Int { lock.withLock { runCount } }

    /// Selects an initial authorization state without accessing physical hardware.
    init(authorization: CameraAuthorization = .authorized) {
        permission = authorization
    }

    /// Holds the first permission request so lifecycle transitions can be tested.
    func authorize() async -> Bool {
        await withCheckedContinuation { pending in
            let result = lock.withLock { () -> Bool? in
                if permission == .unknown {
                    permissionContinuation = pending
                    return nil
                }
                return permission == .authorized
            }
            if let result { pending.resume(returning: result) }
        }
    }

    /// Resolves a simulated operating-system permission prompt exactly once.
    func resolvePermission(_ allowed: Bool) {
        let pending = lock.withLock {
            permission = allowed ? .authorized : .denied
            let value = permissionContinuation
            permissionContinuation = nil
            return value
        }
        pending?.resume(returning: allowed)
    }

    /// Waits for Stop while accepting synchronous frame offers from the test.
    func run(
        configuration: CameraConfiguration, sourceID: String,
        deliver: @escaping @Sendable (PacedVideoFrame) throws -> Void
    ) async throws {
        try Task.checkCancellation()
        lock.withLock {
            report = CameraCaptureReport(
                sourceID: sourceID, configuration: configuration)
            report?.startedMilliseconds = InferenceClock.nowMilliseconds()
            report?.mapping = try! CameraClockMapping(host: .zero, before: 0, after: 0)
            report?.actual = [
                "buffer_width": "16", "buffer_height": "16", "pixel_format": "BGRA32",
            ]
            report?.isRunning = true
            stopped = false
            runCount += 1
            self.deliver = deliver
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { pending in
                let alreadyStopped = lock.withLock { () -> Bool in
                    if stopped { return true }
                    continuation = pending
                    return false
                }
                if alreadyStopped { pending.resume() }
            }
        } onCancel: {
            self.stop()
        }
    }

    /// Produces a real immutable image with separate original and synchronized timestamps.
    func emit(age: Double = 5) throws {
        let context = lock.withLock {
            () -> (String, Int, @Sendable (PacedVideoFrame) throws -> Void)? in
            guard !stopped, let deliver, let report else { return nil }
            return (report.sourceID, report.events.count, deliver)
        }
        guard let (sourceID, index, deliver) = context else { return }
        let now = InferenceClock.nowMilliseconds()
        let host = CMTime(
            seconds: max(0, now - age) / 1000, preferredTimescale: 1_000_000_000)
        let due = try CameraClockMapping(host: .zero, before: 0, after: 0).map(host)
        let pts = CMTime(
            value: Int64(index * 1001), timescale: 30000, flags: .valid, epoch: 7)
        let metadata = try VideoFrameMetadata(
            sourceID: sourceID, sourceFrameIndex: index,
            presentationTime: pts, width: 16, height: 16)
        let image = CGContext(
            data: nil, width: 16, height: 16, bitsPerComponent: 8, bytesPerRow: 64,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        let frame = UprightVideoFrame(
            metadata: metadata, image: image, bufferToImage: .identity,
            readyMilliseconds: now)
        let delivered = InferenceClock.nowMilliseconds()
        try deliver(
            .init(frame: frame, dueMilliseconds: due, deliveredMilliseconds: delivered))
        var event = CameraCaptureEvent(
            index: index, kind: .delivered, presentationTime: pts,
            hostPresentationTime: host, capturedMilliseconds: due,
            receivedMilliseconds: now)
        event.admitted = true
        event.deliveredMilliseconds = delivered
        lock.withLock { report?.events.append(event) }
    }

    /// Records one known late-frame loss or an unknown discontinuity without inference input.
    func drop(unknown: Bool) {
        lock.withLock {
            let now = InferenceClock.nowMilliseconds()
            let index = report!.events.count
            let time = CMTime(seconds: now / 1000, preferredTimescale: 1_000_000_000)
            var event = CameraCaptureEvent(
                index: index, kind: .dropped, presentationTime: time,
                hostPresentationTime: time, capturedMilliseconds: time.seconds * 1000,
                receivedMilliseconds: now)
            event.reason = unknown ? "Discontinuity" : "FrameWasLate"
            event.knownLostFrames = unknown ? nil : 1
            report?.events.append(event)
        }
    }

    /// Emits an identified input failure before any queue admission.
    func failCallback() {
        lock.withLock {
            let now = InferenceClock.nowMilliseconds()
            let time = CMTime(seconds: now / 1000, preferredTimescale: 1_000_000_000)
            var event = CameraCaptureEvent(
                index: report!.events.count, kind: .delivered,
                presentationTime: time, hostPresentationTime: time,
                capturedMilliseconds: time.seconds * 1000, receivedMilliseconds: now)
            event.failure = VideoInferenceFailure(
                CameraError.configuration("Simulated conversion failure"))
            report?.events.append(event)
            report?.issue = event.failure
        }
        stop(error: CameraError.configuration("Simulated conversion failure"))
    }

    /// Closes capture while preserving its clock window and observed prefix.
    func stop() { stop(error: nil) }

    /// Completes a held run outside the lock so continuations cannot reenter shared state.
    private func stop(error: (any Error)?) {
        let pending = lock.withLock {
            stopped = true
            report?.isRunning = false
            if report?.endedMilliseconds == nil {
                report?.endedMilliseconds = InferenceClock.nowMilliseconds()
            }
            let value = continuation
            continuation = nil
            deliver = nil
            return value
        }
        if let error { pending?.resume(throwing: error) } else { pending?.resume() }
    }
}

/// Exposes active concurrency while producing a positive first frame and empty later frames.
private actor CameraTestDetector: ObjectDetector {
    private var held: Bool
    private var continuation: CheckedContinuation<Void, Never>?
    private(set) var active = 0
    private(set) var maximumActive = 0
    private(set) var calls = 0

    /// Optionally pauses the first inference to stress the source/queue boundary.
    init(held: Bool = false) { self.held = held }
    /// Returns the same measured numerical output used by the frame detector.
    func detect(in image: CGImage) async throws -> [Detection] {
        try await detectMeasured(in: image).detections
    }
    /// Keeps model timings valid across an intentional suspension.
    func detectMeasured(in image: CGImage) async throws -> DetectorPrediction {
        let submitted = InferenceClock.nowMilliseconds()
        active += 1
        maximumActive = max(active, maximumActive)
        calls += 1
        if held { await withCheckedContinuation { continuation = $0 } }
        active -= 1
        let detections =
            calls == 1
            ? [
                try Detection(
                    boundingBox: CGRect(x: 2, y: 3, width: 4, height: 4),
                    confidence: 0.9, classID: 1)
            ] : []
        return DetectorPrediction(
            detections: detections,
            modelTiming: .init(
                submittedMilliseconds: submitted,
                completedMilliseconds: InferenceClock.nowMilliseconds()))
    }
    /// Releases active inference and lets the newest pending frame proceed normally.
    func release() {
        held = false
        continuation?.resume()
        continuation = nil
    }
}
