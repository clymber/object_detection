import Foundation
import Observation

/// Publishes video controls/progress while the worker owns decoding and frame logs.
@MainActor
@Observable
final class OfflineVideoSession {
    enum Status: String {
        case ready = "Ready"
        case preparing = "Preparing"
        case running = "Running"
        case stopping = "Stopping"
        case completed = "Completed"
        case completedWithFailures = "Completed with failures"
        case cancelled = "Cancelled"
        case error = "Error"
    }

    private(set) var status = Status.ready
    private(set) var message = "Choose a video, then tap Start."
    private(set) var sourceURL: URL?
    private(set) var sourceID: String?
    private(set) var completedFrames = 0
    private(set) var failedFrames = 0
    private(set) var unfinishedFrames = 0
    private(set) var latestReport: OfflineVideoRunReport?

    @ObservationIgnored private let worker: any OfflineVideoProcessing
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var generation: UInt64 = 0
    @ObservationIgnored private var activeRequest: UInt64?

    var isBusy: Bool {
        status == .preparing || status == .running || status == .stopping
    }
    var canStart: Bool { sourceURL != nil && !isBusy }
    var processedFrames: Int { completedFrames + failedFrames }
    // Full-video frame count is unknown until EOF; never estimate it from FPS.
    var progressText: String {
        "\(processedFrames) \(processedFrames == 1 ? "frame" : "frames") processed"
    }

    /// Accepts a real offline runner or a controlled worker for lifecycle checks.
    init(worker: any OfflineVideoProcessing = OfflineVideoRunner()) {
        self.worker = worker
    }

    /// Replaces input/results, waiting for cancelled work before enabling Start.
    /// Interactive imports use their app-owned URL identity until manifest binding.
    func select(_ url: URL?, sourceID: String? = nil) {
        generation &+= 1
        task?.cancel()
        sourceURL = url
        self.sourceID = url.map { sourceID ?? $0.standardizedFileURL.absoluteString }
        clearResults()
        if task != nil {
            status = .stopping
            message = "Stopping the previous run…"
        } else {
            showReady()
        }
    }

    /// Freezes the current input/model and runs every frame with one active worker.
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
        status = .preparing
        message = "Loading \(model.displayName) and opening the video…"
        let worker = worker
        task = Task { [weak self] in
            var failure: (any Error)?
            do {
                _ = try await worker.run(
                    sourceURL, sourceID: sourceID, model: model,
                    selection: .all, loopIndex: 0, trackID: nil,
                    consume: { [weak self] result in
                        guard let self else { throw CancellationError() }
                        try await self.receive(result, request: request)
                    })
            } catch {
                failure = error
            }
            let report = await worker.latestReport
            self?.finish(report: report, error: failure, request: request)
        }
    }

    /// Requests cooperative cancellation and keeps Start disabled until unwinding.
    func stop() {
        guard isBusy, status != .stopping else { return }
        status = .stopping
        message = "Stopping… The active operation may need to finish."
        task?.cancel()
    }

    /// Rejects queued callbacks after Stop or selection changes, including empty rows.
    private func receive(_ result: VideoFrameResult, request: UInt64) throws {
        try Task.checkCancellation()
        guard generation == request, activeRequest == request,
            status == .preparing || status == .running
        else { throw CancellationError() }
        if result.status == .completed {
            completedFrames += 1
        } else {
            failedFrames += 1
        }
        status = .running
        message = "Processing every video frame in source order…"
    }

    /// Retains partial reports and prevents an old run from replacing a new input.
    private func finish(
        report: OfflineVideoRunReport?, error: (any Error)?, request: UInt64
    ) {
        guard activeRequest == request else { return }
        task = nil
        activeRequest = nil
        guard generation == request else {
            showReady()
            return
        }
        latestReport = report
        if let report {
            completedFrames = report.completedFrames
            failedFrames = report.failedFrames
            unfinishedFrames = report.unfinishedFrames
        }
        switch report?.status {
        case .completed:
            status = failedFrames == 0 ? .completed : .completedWithFailures
            message = failedFrames == 0
                ? "Completed. \(progressText)."
                : "Completed. \(progressText); \(failedFrames) failed."
        case .cancelled:
            showCancelled()
        case .failed:
            status = .error
            message = report?.issue?.details.message
                ?? error?.localizedDescription ?? "Video processing failed."
        case .running, nil:
            if error is CancellationError || status == .stopping {
                showCancelled()
            } else {
                status = .error
                message = error?.localizedDescription
                    ?? "The video worker did not produce a final report."
            }
        }
    }

    /// Resets progress without discarding the current input.
    private func clearResults() {
        completedFrames = 0
        failedFrames = 0
        unfinishedFrames = 0
        latestReport = nil
    }

    /// Presents the currently selected input after old work has fully stopped.
    private func showReady() {
        status = .ready
        message = sourceURL == nil
            ? "Choose a video, then tap Start."
            : "Video ready. Tap Start to process every frame."
    }

    /// Distinguishes a stopped attempt from both success and processing errors.
    private func showCancelled() {
        status = .cancelled
        message = "Stopped. \(progressText); \(unfinishedFrames) unfinished."
    }
}
