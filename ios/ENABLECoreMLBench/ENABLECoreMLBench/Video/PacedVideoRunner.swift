import CoreMedia
import Foundation

/// Exposes the paced lifecycle to a UI session and controlled lifecycle workers.
nonisolated protocol PacedVideoProcessing: Actor {
    var latestReport: PacedVideoRunReport? { get }
    var latestMetrics: PacedVideoRunMetrics? { get }

    /// Runs a frozen configuration with gated numerical or preview receipt.
    func run(
        _ url: URL, sourceID: String, model: DetectorModel,
        firstLoopIndex: Int, loopCount: Int?, trackID: CMPersistentTrackID?,
        previewEnabled: Bool, consume: PacedVideoScheduler.Consumer?
    ) async throws -> PacedVideoRunReport

    /// Returns pixel-free counts and live rates without final percentile sorting.
    func progress() async -> PacedVideoRunProgress?
}

/// Loads once and combines source-clock, bounded queue and receipt logs for one run.
actor PacedVideoRunner: PacedVideoProcessing {
    private let engine: DetectorEngine
    private let converter: PacedVideoSource.Converter
    private let readerFactory: PacedVideoSource.ReaderFactory
    private var activeSource: PacedVideoSource?
    private var activeQueue: PacedVideoScheduler?
    private var generation: UInt64 = 0
    private(set) var isRunning = false
    private(set) var latestReport: PacedVideoRunReport?

    var latestMetrics: PacedVideoRunMetrics? { latestReport?.metrics }

    /// Reuses real reading/detection with controlled engines or conversion in checks.
    init(
        engine: DetectorEngine = DetectorEngine(),
        readerFactory: @escaping PacedVideoSource.ReaderFactory = {
            VideoFrameReader()
        },
        converter: @escaping PacedVideoSource.Converter = {
            try VideoFrameConverter.convert($0)
        }
    ) {
        self.engine = engine
        self.converter = converter
        self.readerFactory = readerFactory
    }

    /// Defaults to one source pass and a numerical consumer with preview off. Counts
    /// survive failures/cancellation; this diagnostic run removes no warm-up frames.
    func run(
        _ url: URL, sourceID: String, model: DetectorModel,
        firstLoopIndex: Int = 0, loopCount: Int? = 1,
        trackID: CMPersistentTrackID? = nil,
        previewEnabled: Bool = false,
        consume: PacedVideoScheduler.Consumer? = { _ = try $0.receive() }
    ) async throws -> PacedVideoRunReport {
        guard !isRunning else { throw VideoPlaybackError.alreadyRunning }
        isRunning = true
        generation &+= 1
        let request = generation
        latestReport = PacedVideoRunReport(
            sourceID: sourceID, model: model,
            firstLoopIndex: firstLoopIndex, loopCount: loopCount,
            previewEnabled: previewEnabled)
        let source = PacedVideoSource(
            readerFactory: readerFactory, converter: converter)
        let queue = PacedVideoScheduler(detector: VideoFrameDetector(detector: engine))
        activeSource = source
        activeQueue = queue
        defer {
            isRunning = false
            activeSource = nil
            activeQueue = nil
        }
        do {
            try checkRequest(request)
            try await engine.selectModel(model)
            try checkRequest(request)
            let processing = Task {
                try await queue.run(previewEnabled: previewEnabled, consume: consume)
            }
            do {
                try await withTaskCancellationHandler {
                    _ = try await source.run(
                        url, sourceID: sourceID, firstLoopIndex: firstLoopIndex,
                        loopCount: loopCount, trackID: trackID) {
                            try queue.offer($0)
                        }
                    queue.finishInput()
                    _ = try await processing.value
                    try checkRequest(request)
                } onCancel: {
                    queue.cancel()
                }
            } catch {
                queue.cancel()
                await source.cancel()
                let completed = await processing.result
                if error as? PacedVideoQueueError == .inputClosed,
                    case .failure(let processingError) = completed,
                    !(processingError is CancellationError)
                {
                    throw processingError
                }
                throw error
            }
            let sourceReport = await source.latestReport
            try checkRequest(request)
            latestReport?.finish(
                status: .completed, source: sourceReport, queue: queue.snapshot)
            return latestReport!
        } catch {
            let cancelled = error is CancellationError || Task.isCancelled
                || request != generation
            queue.cancel()
            await source.cancel()
            latestReport?.finish(
                status: cancelled ? .cancelled : .failed,
                source: await source.latestReport, queue: queue.snapshot,
                error: cancelled ? CancellationError() : error)
            if cancelled { throw CancellationError() }
            throw error
        }
    }

    /// Invalidates receipt immediately, then wakes the source's current clock wait.
    func cancel() async {
        guard isRunning else { return }
        generation &+= 1
        activeQueue?.cancel()
        await activeSource?.cancel()
    }

    /// Builds live counts on the worker, rejecting snapshots from a superseded run.
    func progress() async -> PacedVideoRunProgress? {
        guard isRunning, let source = activeSource, let queue = activeQueue else {
            return nil
        }
        let request = generation
        let sourceReport = await source.latestReport
        guard isRunning, request == generation else { return nil }
        let snapshot = queue.snapshot
        let window = sourceReport?.timeline.flatMap { timeline in
            try? PacedVideoMeasurementWindow(
                startedMilliseconds: timeline.anchorMilliseconds,
                endedMilliseconds: sourceReport?.endedMilliseconds
                    ?? InferenceClock.nowMilliseconds())
        }
        let metrics = window.map {
            PacedVideoRunMetrics(queue: snapshot, window: $0, finalizeStatistics: false)
        }
        return PacedVideoRunProgress(
            source: sourceReport, counts: PacedVideoRunCounts(queue: snapshot),
            metrics: metrics)
    }

    /// Rejects work completed after Stop while retaining the finalized diagnostic log.
    private func checkRequest(_ request: UInt64) throws {
        try Task.checkCancellation()
        guard request == generation else { throw CancellationError() }
    }
}
