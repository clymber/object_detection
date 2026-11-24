import Foundation

/// Retains camera-system callbacks separately from the shared inference queue log.
nonisolated struct CameraRunReport: Sendable {
    let id = UUID()
    let sourceID: String
    let model: DetectorModel
    let configuration: CameraConfiguration
    let startedAt = Date()
    let startedMilliseconds = InferenceClock.nowMilliseconds()
    var endedAt: Date?
    var endedMilliseconds: Double?
    var status = "running"
    var capture: CameraCaptureReport?
    var recording: CameraRecordingReport?
    var queue: PacedVideoQueueSnapshot?
    var issue: VideoInferenceFailure?

    var counts: PacedVideoRunCounts {
        queue.map { PacedVideoRunCounts(queue: $0) } ?? .init()
    }
    var window: PacedVideoMeasurementWindow? {
        guard let start = capture?.startedMilliseconds,
            let end = capture?.endedMilliseconds
        else { return nil }
        return try? PacedVideoMeasurementWindow(
            startedMilliseconds: start, endedMilliseconds: end)
    }
    var metrics: PacedVideoRunMetrics? {
        guard let window, let queue else { return nil }
        return PacedVideoRunMetrics(queue: queue, window: window)
    }

    /// Assigns a new source identity to every capture attempt, including permission failures.
    init(model: DetectorModel, configuration: CameraConfiguration) {
        self.model = model
        self.configuration = configuration
        sourceID = "camera:\(UUID().uuidString)"
    }
}

/// Sends only numerical progress and rates to the UI at a modest polling frequency.
nonisolated struct CameraRunProgress: Sendable {
    let counts: PacedVideoRunCounts
    let metrics: PacedVideoRunMetrics?
    let delivered: Int
    let dropEvents: Int
    let knownLostFrames: Int?
    let sourceFailures: Int
    let captureFramesPerSecond: Double?
    let actual: [String: String]
    let isCapturing: Bool

    /// Reduces retained source logs on the worker before crossing to MainActor.
    init(
        capture: CameraCaptureReport?, queue: PacedVideoQueueSnapshot,
        window: PacedVideoMeasurementWindow?, final: Bool
    ) {
        counts = PacedVideoRunCounts(queue: queue)
        metrics = window.map {
            PacedVideoRunMetrics(queue: queue, window: $0, finalizeStatistics: final)
        }
        delivered = capture?.delivered ?? 0
        dropEvents = capture?.dropEvents ?? 0
        let drops = capture?.events.filter { $0.kind == .dropped } ?? []
        knownLostFrames =
            drops.allSatisfy { $0.knownLostFrames != nil }
            ? drops.reduce(0) { $0 + $1.knownLostFrames! } : nil
        sourceFailures =
            capture?.events.filter { $0.failure != nil && !$0.cancelled }.count ?? 0
        actual = capture?.actual ?? [:]
        isCapturing = capture?.isRunning ?? false
        if let window, window.durationMilliseconds > 0 {
            let count =
                capture?.events.filter {
                    $0.kind == .delivered && window.contains($0.receivedMilliseconds)
                }.count ?? 0
            captureFramesPerSecond = Double(count) * 1000 / window.durationMilliseconds
        } else {
            captureFramesPerSecond = nil
        }
    }
}

/// Allows deterministic permission, capture and lifecycle workers in tests.
nonisolated protocol CameraRunning: Actor {
    nonisolated var preview: CameraPreviewHandle? { get }
    nonisolated var authorization: CameraAuthorization { get }
    var latestReport: CameraRunReport? { get }
    /// Requests permission before model loading and hardware setup.
    func authorize() async -> Bool
    /// Captures one immutable settings configuration until Stop or failure.
    func run(
        configuration: CameraConfiguration, model: DetectorModel,
        consume: PacedVideoScheduler.Consumer?) async throws
    /// Computes live or final numerical summaries off the UI actor.
    func progress() -> CameraRunProgress?
}

/// Reuses model loading, single-flight inference, bounded pending work and fresh receipts.
actor CameraRunner: CameraRunning {
    private let engine: DetectorEngine
    private let capture: any CameraCapturing
    private let recordingDirectory: URL
    private var queue: PacedVideoScheduler?
    private(set) var latestReport: CameraRunReport?
    private var isRunning = false
    nonisolated var preview: CameraPreviewHandle? { capture.preview }
    nonisolated var authorization: CameraAuthorization { capture.authorization }

    /// Keeps the same engine available to recorded-video and live-camera modes.
    init(
        engine: DetectorEngine = DetectorEngine(),
        capture: any CameraCapturing = CameraCaptureSource(),
        recordingDirectory: URL = URL.cachesDirectory.appendingPathComponent(
            "CameraRecordings")
    ) {
        self.engine = engine
        self.capture = capture
        self.recordingDirectory = recordingDirectory
    }

    /// Runs the operating-system permission request before capture/model work begins.
    func authorize() async -> Bool { await capture.authorize() }

    /// Runs until Stop, preserving completed active work and source-drop evidence.
    func run(
        configuration: CameraConfiguration, model: DetectorModel,
        consume: PacedVideoScheduler.Consumer? = { _ = try $0.receive() }
    ) async throws {
        guard !isRunning else { throw CameraError.alreadyRunning }
        isRunning = true
        latestReport = CameraRunReport(model: model, configuration: configuration)
        let sourceID = latestReport!.sourceID
        let capture = self.capture
        let recorder = configuration.recordAnnotatedVideo
            ? CameraVideoRecorder(
                framesPerSecond: configuration.framesPerSecond,
                directory: recordingDirectory) : nil
        let detector = VideoFrameDetector(detector: engine)
        let queue = PacedVideoScheduler { frame in
            let result = try await detector.process(frame)
            if result.status == .completed, let recorder {
                recorder.offer(try VideoFramePreview(frame: frame, result: result))
            }
            return result
        }
        self.queue = queue
        defer {
            self.queue = nil
            isRunning = false
        }
        var processing: Task<PacedVideoQueueSnapshot, Error>?
        var failure: (any Error)?
        do {
            try await withTaskCancellationHandler {
                try Task.checkCancellation()
                try configuration.validate()
                guard capture.authorization == .authorized else {
                    throw CameraError.denied
                }
                try await engine.selectModel(model)
                try Task.checkCancellation()
                processing = Task {
                    try await queue.run(
                        previewEnabled: configuration.detectionPreview, consume: consume
                    )
                }
                try await capture.run(configuration: configuration, sourceID: sourceID)
                {
                    try queue.offer($0)
                }
                if Task.isCancelled { queue.cancel() } else { queue.finishInput() }
                _ = try await processing!.value
                try Task.checkCancellation()
            } onCancel: {
                queue.cancel()
                capture.stop()
            }
        } catch {
            failure = error
            queue.cancel()
            capture.stop()
            if let processing { _ = await processing.result }
        }
        latestReport?.capture = capture.snapshot.flatMap {
            $0.sourceID == sourceID ? $0 : nil
        }
        latestReport?.queue = queue.snapshot
        latestReport?.endedAt = Date()
        latestReport?.endedMilliseconds = InferenceClock.nowMilliseconds()
        latestReport?.issue = failure.map { VideoInferenceFailure($0) }
        latestReport?.status =
            failure == nil
            ? "completed"
            : (failure is CancellationError || Task.isCancelled
                ? "cancelled" : "failed")
        latestReport?.recording = await recorder?.finish()
        if let failure { throw failure }
    }

    /// Uses a capture-only window and avoids percentile sorting during live polling.
    func progress() -> CameraRunProgress? {
        guard let report = latestReport, let snapshot = queue?.snapshot ?? report.queue
        else { return nil }
        let source =
            isRunning
            ? capture.snapshot.flatMap { $0.sourceID == report.sourceID ? $0 : nil }
            : report.capture
        let window = source?.startedMilliseconds.flatMap {
            try? PacedVideoMeasurementWindow(
                startedMilliseconds: $0,
                endedMilliseconds: source?.endedMilliseconds
                    ?? InferenceClock.nowMilliseconds())
        }
        return CameraRunProgress(
            capture: source, queue: snapshot, window: window, final: !isRunning)
    }
}
