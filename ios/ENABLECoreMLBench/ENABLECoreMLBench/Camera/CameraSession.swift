import Foundation
import Observation

/// Controls permissions, foreground state and generation-checked camera publications.
@MainActor
@Observable
final class CameraSession {
    enum Status: String {
        case ready = "Ready"
        case requestingPermission = "Requesting camera access"
        case waitingForActive = "Waiting for app"
        case preparing = "Preparing"
        case running = "Running"
        case stopping = "Stopping"
        case stopped = "Stopped"
        case error = "Error"
    }
    private(set) var status = Status.ready
    private(set) var message = "Tap Start to use the camera."
    private(set) var authorization: CameraAuthorization
    private(set) var configurationForRun: CameraConfiguration?
    private(set) var latestReport: CameraRunReport?
    private(set) var counts = PacedVideoRunCounts()
    private(set) var metrics: PacedVideoRunMetrics?
    private(set) var deliveredFrames = 0
    private(set) var captureDrops = 0
    private(set) var knownLostFrames: Int? = 0
    private(set) var sourceFailures = 0
    private(set) var captureFPS: Double?
    private(set) var elapsedSeconds: Double?
    private(set) var actual: [String: String] = [:]
    private(set) var previewFrame: VideoFramePreview?
    private(set) var previewGeneration = UUID()
    private(set) var previewStatistics = VideoPreviewStatistics()
    private var draws: [PacedVideoDrawSample] = []
    let export: VideoExportSession
    @ObservationIgnored private let worker: any CameraRunning
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var poll: Task<Void, Never>?
    @ObservationIgnored private var generation = UUID()
    @ObservationIgnored private var activation: CheckedContinuation<Void, Never>?
    @ObservationIgnored private var isActive = true

    var preview: CameraPreviewHandle? { worker.preview }
    var isCapturing: Bool {
        [.requestingPermission, .waitingForActive, .preparing, .running, .stopping]
            .contains(status)
    }
    var isBusy: Bool { isCapturing || export.isBusy }
    var displayFPS: Double? {
        guard let window = metrics?.window, window.durationMilliseconds > 0,
            configurationForRun?.detectionPreview == true
        else { return nil }
        return Double(draws.filter { window.contains($0.drawnMilliseconds) }.count)
            * 1000 / window.durationMilliseconds
    }

    /// Uses an injectable actor while retaining the production capture preview handle.
    init(worker: any CameraRunning = CameraRunner(), export: VideoExportSession? = nil)
    {
        self.worker = worker
        self.export = export ?? VideoExportSession()
        authorization = worker.authorization
    }

    /// Freezes settings and defers capture until permission and foreground activation.
    func start(configuration: CameraConfiguration, model: DetectorModel) {
        guard !isBusy else { return }
        resetResults()
        let request = generation
        configurationForRun = configuration
        authorization = worker.authorization
        status = authorization == .unknown ? .requestingPermission : .preparing
        message =
            status == .requestingPermission
            ? "Allow camera access to continue."
            : "Loading \(model.displayName)…"
        let worker = self.worker
        let attempt = CameraRunReport(model: model, configuration: configuration)
        task = Task { [weak self] in
            var didRun = false
            var failure: (any Error)?
            do {
                let granted = await worker.authorize()
                try Task.checkCancellation()
                guard let self, self.generation == request else {
                    throw CancellationError()
                }
                self.authorization = worker.authorization
                guard granted else { throw CameraError.denied }
                if !self.isActive {
                    self.status = .waitingForActive
                    await self.waitForActivation()
                }
                try Task.checkCancellation()
                guard self.generation == request else { throw CancellationError() }
                self.status = .preparing
                self.message = "Starting camera…"
                self.startPolling(request: request)
                didRun = true
                try await worker.run(
                    configuration: configuration, model: model,
                    consume: {
                        [weak self] delivery in
                        guard let self else { throw CancellationError() }
                        try await self.receive(
                            delivery, request: request,
                            preview: configuration.detectionPreview)
                    })
            } catch { failure = error }
            self?.poll?.cancel()
            var report = didRun ? await worker.latestReport : attempt
            let progress = didRun ? await worker.progress() : nil
            if !didRun {
                report?.status = failure is CancellationError ? "cancelled" : "failed"
                report?.issue = failure.map { VideoInferenceFailure($0) }
                report?.endedAt = Date()
                report?.endedMilliseconds = InferenceClock.nowMilliseconds()
            }
            self?.finish(report, progress: progress, error: failure, request: request)
        }
    }

    /// Invalidates receipts synchronously; hardware and active inference then unwind.
    func stop() {
        if isCapturing, status != .stopping {
            status = .stopping
            message = configurationForRun?.recordAnnotatedVideo == true
                ? "Stopping capture, finishing inference and finalizing MP4…"
                : "Stopping capture and finishing active inference…"
            poll?.cancel()
            task?.cancel()
            activation?.resume()
            activation = nil
        }
    }

    /// Ignores permission-dialog inactivity but always stops when actually backgrounded.
    func setActivity(active: Bool, background: Bool = false) {
        isActive = active
        authorization = worker.authorization
        if active {
            activation?.resume()
            activation = nil
        } else if background || status != .requestingPermission {
            stop()
        }
    }

    /// Clears old source/model results without publishing any cancelled run's callbacks.
    func reset() {
        stop()
        generation = UUID()
        resetResults(changeGeneration: false)
        if task == nil {
            status = .ready
            message = "Tap Start to use the camera."
        }
    }

    /// Exports stopped, failed or completed attempts using the same three-file handoff.
    func exportResults() {
        guard !isBusy, let latestReport else { return }
        export.start(.camera(latestReport))
    }

    /// Deduplicates exact processed-frame draws while keeping native preview unmeasured.
    func recordDraw(
        _ token: VideoPreviewDrawToken, generation: UUID, at milliseconds: Double
    ) {
        guard generation == previewGeneration,
            configurationForRun?.detectionPreview == true,
            previewStatistics.canRecord(token, at: milliseconds)
        else { return }
        previewStatistics.record(token, at: milliseconds)
        draws.append(.init(id: token.id, drawnMilliseconds: milliseconds))
    }

    /// Claims only on the receiving actor, after every actor-hop freshness check.
    private func receive(_ delivery: PacedVideoDelivery, request: UUID, preview: Bool)
        throws
    {
        try Task.checkCancellation()
        guard generation == request, status == .preparing || status == .running else {
            throw CancellationError()
        }
        if preview {
            guard let frame = try delivery.receivePreview() else { return }
            previewFrame = frame
            previewStatistics.admit(frame.drawToken)
        } else {
            _ = try delivery.receive()
        }
        status = .running
        message = "Capturing live frames…"
    }

    /// Waits without starting a camera behind the permission prompt or another app.
    private func waitForActivation() async {
        await withCheckedContinuation { activation = $0 }
    }

    /// Polls numerical worker summaries without touching camera pixels on MainActor.
    private func startPolling(request: UUID) {
        let worker = self.worker
        poll = Task { [weak self] in
            while !Task.isCancelled {
                guard self != nil else { break }
                if let progress = await worker.progress(), let self,
                    self.generation == request,
                    self.status == .preparing || self.status == .running
                {
                    self.apply(progress)
                    if progress.durationLimitReached {
                        self.status = .stopping
                        let recording = self.configurationForRun?.recordAnnotatedVideo
                            == true
                        self.message = recording
                            ? "Time limit reached. Finishing inference and MP4…"
                            : "Time limit reached. Finishing active inference…"
                    } else if progress.isCapturing {
                        self.status = .running
                        self.message = "Capturing live frames…"
                    }
                }
                do { try await Task.sleep(for: .milliseconds(500)) } catch { break }
            }
        }
    }

    /// Assigns only scalar statistics prepared on the worker actor.
    private func apply(_ progress: CameraRunProgress) {
        counts = progress.counts
        metrics = progress.metrics
        deliveredFrames = progress.delivered
        captureDrops = progress.dropEvents
        knownLostFrames = progress.knownLostFrames
        sourceFailures = progress.sourceFailures
        captureFPS = progress.captureFramesPerSecond
        elapsedSeconds = (progress.metrics?.window.durationMilliseconds).map {
            $0 / 1000
        }
        actual = progress.actual
    }

    /// Keeps interrupted runs exportable while ignoring superseded input generations.
    private func finish(
        _ report: CameraRunReport?, progress: CameraRunProgress?,
        error: (any Error)?, request: UUID
    ) {
        task = nil
        poll?.cancel()
        poll = nil
        guard generation == request else {
            if let url = report?.recording?.url {
                try? FileManager.default.removeItem(at: url)
            }
            status = .ready
            return
        }
        latestReport = report
        if let progress { apply(progress) }
        status = report?.status == "failed" ? .error : .stopped
        message =
            report?.status == "failed"
            ? (report?.issue?.message ?? error?.localizedDescription ?? "Camera failed.")
            : "Stopped. \(counts.processedFrames) processed; \(captureDrops) capture drops; \(counts.skippedFrames) queue skips."
        if report?.durationLimitReached == true, status != .error {
            message = "Time limit reached. " + message
        }
    }

    /// Releases previous images and export baselines before a different run can begin.
    private func resetResults(changeGeneration: Bool = true) {
        if changeGeneration { generation = UUID() }
        if let url = latestReport?.recording?.url {
            try? FileManager.default.removeItem(at: url)
        }
        export.reset()
        latestReport = nil
        counts = .init()
        metrics = nil
        deliveredFrames = 0
        captureDrops = 0
        knownLostFrames = 0
        sourceFailures = 0
        captureFPS = nil
        elapsedSeconds = nil
        actual = [:]
        previewFrame = nil
        previewStatistics = .init()
        previewGeneration = UUID()
        draws = []
        configurationForRun = nil
    }
}
