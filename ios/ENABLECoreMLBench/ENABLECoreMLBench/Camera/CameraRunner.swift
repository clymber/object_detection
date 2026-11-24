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
    var durationLimitReached = false
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
    let durationLimitReached: Bool

    /// Reduces retained source logs on the worker before crossing to MainActor.
    init(
        capture: CameraCaptureReport?, queue: PacedVideoQueueSnapshot,
        window: PacedVideoMeasurementWindow?, final: Bool,
        durationLimitReached: Bool
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
        self.durationLimitReached = durationLimitReached
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
    private var durationTimer: CameraDurationTimer?
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
        let timer = CameraDurationTimer(
            seconds: configuration.durationLimitEnabled
                ? configuration.durationLimitSeconds : nil,
            stop: { capture.stop() })
        durationTimer = timer
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
            timer.cancel()
            durationTimer = nil
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
                try await capture.run(
                    configuration: configuration, sourceID: sourceID,
                    started: { timer.start(at: $0) },
                    deliver: { try queue.offer($0) })
                timer.cancel()
                if Task.isCancelled { queue.cancel() } else { queue.finishInput() }
                _ = try await processing!.value
                try Task.checkCancellation()
            } onCancel: {
                timer.cancel()
                queue.cancel()
                capture.stop()
            }
        } catch {
            timer.cancel()
            failure = error
            queue.cancel()
            capture.stop()
            if let processing { _ = await processing.result }
        }
        latestReport?.capture = capture.snapshot.flatMap {
            $0.sourceID == sourceID ? $0 : nil
        }
        latestReport?.queue = queue.snapshot
        latestReport?.durationLimitReached = timer.hasExpired
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
            capture: source, queue: snapshot, window: window, final: !isRunning,
            durationLimitReached: durationTimer?.hasExpired
                ?? report.durationLimitReached)
    }
}

/// Stops one capture window independently of model inference and UI polling.
nonisolated private final class CameraDurationTimer: @unchecked Sendable {
    private let lock = NSLock()
    private let seconds: Double?
    private let stop: @Sendable () -> Void
    private var task: Task<Void, Never>?
    private var cancelled = false
    private var expired = false

    var hasExpired: Bool { lock.withLock { expired } }

    /// Leaves unlimited runs unarmed and binds one timer to one capture source.
    init(seconds: Double?, stop: @escaping @Sendable () -> Void) {
        self.seconds = seconds
        self.stop = stop
    }

    /// Anchors to the same monotonic capture start used by the FPS measurement window.
    func start(at milliseconds: Double) {
        lock.withLock {
            guard !cancelled, task == nil, let seconds else { return }
            let deadline = milliseconds + seconds * 1000
            task = Task.detached { [weak self] in
                do {
                    try await InferenceClock.sleep(untilMilliseconds: deadline)
                } catch { return }
                self?.expire()
            }
        }
    }

    /// Cancels on every terminal path so an old deadline cannot stop a later run.
    func cancel() {
        lock.withLock {
            cancelled = true
            task?.cancel()
            task = nil
        }
    }

    /// Closes capture promptly, while the run drains inference and finalizes its MP4.
    private func expire() {
        lock.withLock {
            guard !cancelled, !expired else { return }
            expired = true
            stop()
        }
    }
}
