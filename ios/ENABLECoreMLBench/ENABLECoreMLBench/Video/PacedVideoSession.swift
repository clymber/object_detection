import Foundation
import Observation

/// Binds paced controls to worker snapshots and claims fresh receipts on MainActor.
@MainActor
@Observable
final class PacedVideoSession {
    private(set) var status = OfflineVideoSession.Status.ready
    private(set) var message = "Choose a video, then tap Start."
    private(set) var sourceURL: URL?
    private(set) var sourceID: String?
    private(set) var counts = PacedVideoRunCounts()
    private(set) var latestReport: PacedVideoRunReport?
    private(set) var metrics: PacedVideoRunMetrics?
    private(set) var isPreviewEnabled = false
    private(set) var previewEnabledForRun: Bool?
    private(set) var previewFrame: VideoFramePreview?
    private(set) var previewStatistics = VideoPreviewStatistics()
    private(set) var previewGeneration = UUID()
    private(set) var latestReceipt: VideoFrameResult?
    private(set) var decodedFrames = 0
    private(set) var decodeErrorEvents = 0
    private var drawSamples: [PacedVideoDrawSample] = []
    @ObservationIgnored private let worker: any PacedVideoProcessing
    @ObservationIgnored private let pollInterval: Duration
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var pollTask: Task<Void, Never>?
    @ObservationIgnored private var generation: UInt64 = 0
    @ObservationIgnored private var activeRequest: UInt64?
    @ObservationIgnored var onFinished: ((VideoRunExport) -> Void)?

    var isBusy: Bool {
        status == .preparing || status == .running || status == .stopping
    }
    var canStart: Bool { sourceURL != nil && !isBusy }
    var progressText: String {
        "\(counts.inputFrames) inputs · \(counts.processedFrames) processed · "
            + "\(counts.skippedFrames) skipped · \(counts.publishedResults) fresh"
    }
    var displayFramesPerSecond: Double? {
        guard previewEnabledForRun == true, let window = metrics?.window,
            window.durationMilliseconds > 0 else { return nil }
        let count = drawSamples.filter { window.contains($0.drawnMilliseconds) }.count
        return Double(count) * 1000 / window.durationMilliseconds
    }

    /// Uses worker-side polling at a fixed modest cadence, independent of inference.
    init(
        worker: any PacedVideoProcessing = PacedVideoRunner(),
        pollInterval: Duration = .milliseconds(500)
    ) {
        self.worker = worker
        self.pollInterval = pollInterval
    }

    /// Freezes preview configuration per run and releases the previous accepted image.
    func setPreviewEnabled(_ enabled: Bool) {
        guard !isBusy, enabled != isPreviewEnabled else { return }
        isPreviewEnabled = enabled
        previewFrame = nil
        previewGeneration = UUID()
    }

    /// Invalidates old receipts/snapshots before accepting a new input or model.
    func select(_ url: URL?, sourceID: String? = nil) {
        generation &+= 1
        task?.cancel()
        pollTask?.cancel()
        self.sourceURL = url
        self.sourceID = url.map { sourceID ?? $0.standardizedFileURL.absoluteString }
        clearResults()
        if task != nil {
            status = .stopping
            message = "Stopping the previous run…"
        } else { showReady() }
    }

    /// Runs one source pass from the beginning with an optional exact-frame preview.
    func start(model: DetectorModel) {
        guard !isBusy else { return }
        guard let sourceURL, let sourceID else {
            status = .error
            message = "Choose a video before starting."
            return
        }
        generation &+= 1
        let request = generation
        activeRequest = request
        clearResults()
        let preview = isPreviewEnabled
        previewEnabledForRun = preview
        status = .preparing
        message = "Loading \(model.displayName) and opening the video…"
        let worker = worker
        task = Task { [weak self] in
            var failure: (any Error)?
            do {
                _ = try await worker.run(
                    sourceURL, sourceID: sourceID, model: model,
                    firstLoopIndex: 0, loopCount: 1, trackID: nil,
                    previewEnabled: preview,
                    consume: { [weak self] delivery in
                        guard let self else { throw CancellationError() }
                        try await self.receive(
                            delivery, preview: preview, request: request)
                    })
            } catch { failure = error }
            let report = await worker.latestReport
            let metrics = await worker.latestMetrics
            self?.finish(report, metrics: metrics, error: failure, request: request)
        }
        let interval = pollInterval
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard self != nil else { break }
                if let progress = await worker.progress() {
                    self?.receiveProgress(progress, request: request)
                }
                do { try await Task.sleep(for: interval) } catch { break }
            }
        }
    }

    /// Stops UI admission synchronously, then lets the worker retain active completion.
    func stop() {
        guard isBusy, status != .stopping else { return }
        status = .stopping
        message = "Stopping… The active operation may need to finish."
        pollTask?.cancel()
        task?.cancel()
    }

    /// Claims on the final actor and updates receiving state without another await.
    private func receive(
        _ delivery: PacedVideoDelivery, preview: Bool, request: UInt64
    ) throws {
        try Task.checkCancellation()
        guard generation == request, activeRequest == request,
            status == .preparing || status == .running else {
                throw CancellationError()
            }
        if preview {
            guard let frame = try delivery.receivePreview() else { return }
            previewFrame = frame
            previewStatistics.admit(frame.drawToken)
            latestReceipt = frame.result
        } else {
            guard let result = try delivery.receive() else { return }
            latestReceipt = result
        }
        status = .running
        message = "Processing at source cadence…"
    }

    /// Ignores late snapshots after Stop, input/model replacement or finalization.
    private func receiveProgress(_ progress: PacedVideoRunProgress, request: UInt64) {
        guard generation == request, activeRequest == request,
            status == .preparing || status == .running else { return }
        counts = progress.counts
        metrics = progress.metrics
        decodedFrames = progress.source?.decodedFrames ?? 0
        decodeErrorEvents = progress.source?.decodeLosses.count ?? 0
        if progress.source != nil {
            status = .running
            message = progress.source?.status == .completed
                ? "Finishing active processing and delivery…"
                : "Processing at source cadence…"
        }
    }

    /// Deduplicates actual drawing receipts and rejects previous-run render callbacks.
    func recordPreviewDraw(
        _ token: VideoPreviewDrawToken, generation: UUID, at milliseconds: Double
    ) {
        guard generation == previewGeneration, isPreviewEnabled,
            previewStatistics.canRecord(token, at: milliseconds) else { return }
        previewStatistics.record(token, at: milliseconds)
        drawSamples.append(PacedVideoDrawSample(
            id: token.id, drawnMilliseconds: milliseconds))
    }

    /// Final counts include retained late work; rates retain their source-window scope.
    private func finish(
        _ report: PacedVideoRunReport?, metrics: PacedVideoRunMetrics?,
        error: (any Error)?, request: UInt64
    ) {
        guard activeRequest == request else { return }
        pollTask?.cancel()
        pollTask = nil
        task = nil
        activeRequest = nil
        guard generation == request else {
            showReady()
            return
        }
        latestReport = report
        self.metrics = metrics
        if let report {
            counts = report.counts ?? PacedVideoRunCounts()
            decodedFrames = report.source?.decodedFrames ?? 0
            decodeErrorEvents = report.decodeLosses.count
            previewEnabledForRun = report.previewEnabled
        }
        switch report?.status {
        case .completed:
            let failures = counts.failedFrames + counts.failedPublications
            status = failures > 0 ? .completedWithFailures : .completed
            message = "Completed. \(progressText)."
        case .cancelled:
            status = .cancelled
            message = "Stopped. \(progressText)."
        case .failed:
            status = .error
            message = report?.issue?.message ?? error?.localizedDescription
                ?? "Paced processing failed."
        case .running, nil:
            status = error is CancellationError ? .cancelled : .error
            message = error?.localizedDescription ?? "The paced worker did not finish."
        }
        if let report { onFinished?(.paced(report)) }
    }

    /// Clears values before a new run or input, preserving the chosen preview setting.
    private func clearResults() {
        counts = PacedVideoRunCounts()
        latestReport = nil
        metrics = nil
        latestReceipt = nil
        decodedFrames = 0
        decodeErrorEvents = 0
        previewEnabledForRun = nil
        previewFrame = nil
        previewStatistics = VideoPreviewStatistics()
        previewGeneration = UUID()
        drawSamples = []
    }

    /// Presents the current input only after cancelled work has fully unwound.
    private func showReady() {
        status = .ready
        message = sourceURL == nil ? "Choose a video, then tap Start."
            : "Video ready. Tap Start for one pass at source cadence."
    }
}
