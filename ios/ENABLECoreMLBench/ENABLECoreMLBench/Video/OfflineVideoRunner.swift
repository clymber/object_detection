import CoreMedia
import Foundation

/// Selects original presentation-order indices without renumbering selected frames.
nonisolated enum VideoFrameSelection: Sendable {
    case all
    case indices([Int])
}

/// Counts decoded inputs and terminal selected-frame attempts, excluding unfinished.
nonisolated struct OfflineVideoRunSummary: Equatable, Sendable {
    let decodedFrames: Int
    let processedFrames: Int
    let completedFrames: Int
    let failedFrames: Int
}

/// Rejects ambiguous selections and requests that cannot be completed from the file.
nonisolated enum OfflineVideoError: Error, Equatable, LocalizedError {
    case alreadyRunning
    case invalidSelection
    case missingSelectedFrame(Int)
    case coverageMismatch

    var errorDescription: String? {
        switch self {
        case .alreadyRunning: return "An offline video run is already active."
        case .invalidSelection:
            return "Frame indices must be nonnegative, unique and increasing."
        case .missingSelectedFrame(let index):
            return "Video ended before selected source frame \(index)."
        case .coverageMismatch:
            return "Offline outcomes do not reconcile with the selected frames."
        }
    }
}

/// Exposes worker-owned processing and reports to the UI session and test workers.
nonisolated protocol OfflineVideoProcessing: Actor {
    var latestReport: OfflineVideoRunReport? { get }
    var latestMetrics: OfflineVideoRunMetrics? { get }

    /// Delivers ordered outcomes while retaining the final report on every exit.
    func run(
        _ url: URL, sourceID: String, model: DetectorModel,
        selection: VideoFrameSelection, loopIndex: Int,
        trackID: CMPersistentTrackID?,
        preview: (@Sendable (VideoFramePreview) async throws -> Void)?,
        consume: @escaping @Sendable (VideoFrameResult) async throws -> Void
    ) async throws -> OfflineVideoRunSummary
}

extension OfflineVideoProcessing {
    /// Aggregates retained results on the worker actor after its run has ended.
    var latestMetrics: OfflineVideoRunMetrics? {
        latestReport.map { OfflineVideoRunMetrics(report: $0) }
    }
}

/// Processes selected frames serially, without a playback clock or age-based skips.
actor OfflineVideoRunner: OfflineVideoProcessing {
    typealias Consumer = @Sendable (VideoFrameResult) async throws -> Void
    typealias PreviewConsumer = @Sendable (VideoFramePreview) async throws -> Void
    typealias Converter = @Sendable (VideoFrame) throws -> UprightVideoFrame

    private let engine: DetectorEngine
    private let detector: VideoFrameDetector
    private let converter: Converter
    private var activeReader: VideoFrameReader?
    private var generation: UInt64 = 0
    private(set) var isRunning = false
    private(set) var latestReport: OfflineVideoRunReport?

    /// Permits engine/conversion injection while preserving the real reader pipeline.
    init(
        engine: DetectorEngine = DetectorEngine(),
        converter: @escaping Converter = { try VideoFrameConverter.convert($0) }
    ) {
        self.engine = engine
        detector = VideoFrameDetector(detector: engine)
        self.converter = converter
    }

    /// Loads once, then awaits detection and consumption before reading another frame.
    /// The consumer receives success/empty/failure records and owns publication time.
    /// Cancellation/input/consumer errors throw after finalizing latestReport.
    func run(
        _ url: URL, sourceID: String, model: DetectorModel,
        selection: VideoFrameSelection = .all, loopIndex: Int = 0,
        trackID: CMPersistentTrackID? = nil, preview: PreviewConsumer? = nil,
        consume: @escaping Consumer
    ) async throws -> OfflineVideoRunSummary {
        guard !isRunning else { throw OfflineVideoError.alreadyRunning }
        let indices = try validatedIndices(selection)
        let wanted = indices.map { Set($0) }
        let convert = converter
        isRunning = true
        generation &+= 1
        let request = generation
        let reader = VideoFrameReader()
        activeReader = reader
        latestReport = OfflineVideoRunReport(
            sourceID: sourceID, model: model, loopIndex: loopIndex, indices: indices,
            previewEnabled: preview != nil)
        var stage = OfflineVideoStage.modelLoading
        var activeFrame: VideoFrameReadInfo?
        var decoded = 0
        defer {
            activeReader = nil
            isRunning = false
        }
        do {
            try checkRequest(request)
            // Loading/compilation completes before source-frame readiness is measured.
            try await engine.selectModel(model)
            try checkRequest(request)
            stage = .sourceOpening
            try await reader.open(
                url, sourceID: sourceID, loopIndex: loopIndex, trackID: trackID)
            try checkRequest(request)
            var processed = 0
            while indices == nil || processed < indices!.count {
                stage = .decoding
                try checkRequest(request)
                let next = try await reader.nextFrame { frame -> SelectedFrame in
                    if let wanted, !wanted.contains(frame.metadata.sourceFrameIndex) {
                        return .unselected
                    }
                    return .selected(try convert(frame))
                }
                // Retain admission before observing cancellation after an actor hop.
                if case .selected(let frame) = next {
                    activeFrame = VideoFrameReadInfo(
                        metadata: frame.metadata,
                        readyMilliseconds: frame.readyMilliseconds)
                    stage = .inference
                } else if next == nil {
                    latestReport?.recordEndOfSource()
                }
                try checkRequest(request)
                guard let next else {
                    if let indices, processed < indices.count {
                        throw OfflineVideoError.missingSelectedFrame(indices[processed])
                    }
                    break
                }
                decoded += 1
                latestReport?.recordDecodedFrames(decoded)
                guard case .selected(let frame) = next else { continue }
                let result = try await detector.process(frame)
                try checkRequest(request)
                processed += 1
                latestReport?.append(OfflineFrameOutcome(result: result))
                activeFrame = nil
                stage = .consumption
                try await consume(result)
                try checkRequest(request)
                if let preview {
                    try await preview(VideoFramePreview(frame: frame, result: result))
                    try checkRequest(request)
                }
            }
            await reader.cancel()
            try checkRequest(request)
            stage = .reconciliation
            let coverage = latestReport!.coverage
            try checkRequest(request)
            guard coverage.hasCompleteOutcomeSet else {
                throw OfflineVideoError.coverageMismatch
            }
            latestReport?.finish(status: .completed)
            return latestReport!.summary
        } catch {
            let cancelled = error is CancellationError || Task.isCancelled
                || request != generation
            // The reader preserves metadata when cancellation hides a callback result.
            if activeFrame == nil, stage == .decoding,
                let info = await reader.lastFrameInfo,
                info.metadata.sourceFrameIndex >= decoded,
                wanted == nil || wanted!.contains(info.metadata.sourceFrameIndex)
            {
                activeFrame = info
                stage = .conversion
            }
            let reported: any Error = cancelled ? CancellationError() : error
            let issue = OfflineVideoIssue(stage: stage, error: reported)
            if let activeFrame {
                latestReport?.append(OfflineFrameOutcome(
                    sourceFrameIndex: activeFrame.metadata.sourceFrameIndex,
                    metadata: activeFrame.metadata,
                    readyMilliseconds: activeFrame.readyMilliseconds,
                    cancelled: cancelled, issue: issue))
            } else if !cancelled, stage == .decoding,
                let index = failedSourceIndex(error),
                wanted == nil || wanted!.contains(index)
            {
                latestReport?.append(OfflineFrameOutcome(
                    sourceFrameIndex: index, metadata: nil, readyMilliseconds: nil,
                    cancelled: false, issue: issue))
            }
            let decodedCount = await reader.framesRead
            latestReport?.recordDecodedFrames(decodedCount)
            let readerState = await reader.state
            if readerState == .completed { latestReport?.recordEndOfSource() }
            await reader.cancel()
            latestReport?.finish(status: cancelled ? .cancelled : .failed, issue: issue)
            if cancelled {
                throw CancellationError()
            }
            throw error
        }
    }

    /// Prevents delivery after cancellation; wait for the active run before restarting.
    /// A model load or synchronous prediction may finish before the run can unwind.
    func cancel() async {
        guard isRunning else { return }
        generation &+= 1
        await activeReader?.cancel()
    }

    /// Keeps unselected decoded inputs distinct from selected immutable images.
    private enum SelectedFrame: Sendable {
        case unselected
        case selected(UprightVideoFrame)
    }

    /// Requires presentation order rather than silently sorting or deduplicating input.
    private func validatedIndices(_ selection: VideoFrameSelection) throws -> [Int]? {
        guard case .indices(let indices) = selection else { return nil }
        guard indices.allSatisfy({ $0 >= 0 }),
            zip(indices, indices.dropFirst()).allSatisfy({ $0 < $1 })
        else { throw OfflineVideoError.invalidSelection }
        return indices
    }

    /// Discards work completed after task cancellation or an explicit stop request.
    private func checkRequest(_ request: UInt64) throws {
        try Task.checkCancellation()
        guard request == generation else { throw CancellationError() }
    }

    /// Uses explicit source indices from reader errors, never a guessed FPS timestamp.
    private func failedSourceIndex(_ error: any Error) -> Int? {
        switch error {
        case VideoReaderError.decodeFailed(let index, _),
             VideoReaderError.invalidFrame(let index, _),
             OfflineVideoError.missingSelectedFrame(let index): return index
        default: return nil
        }
    }
}
