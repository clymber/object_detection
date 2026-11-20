import CoreMedia
import Foundation

/// Selects original presentation-order indices without renumbering selected frames.
nonisolated enum VideoFrameSelection: Sendable {
    case all
    case indices([Int])
}

/// Counts decoded inputs and attempted selected frames for a normally finished run.
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

    var errorDescription: String? {
        switch self {
        case .alreadyRunning: return "An offline video run is already active."
        case .invalidSelection:
            return "Frame indices must be nonnegative, unique and increasing."
        case .missingSelectedFrame(let index):
            return "Video ended before selected source frame \(index)."
        }
    }
}

/// Processes selected frames serially, without a playback clock or age-based skips.
actor OfflineVideoRunner {
    typealias Consumer = @Sendable (VideoFrameResult) async throws -> Void

    private let engine: DetectorEngine
    private let detector: VideoFrameDetector
    private var activeReader: VideoFrameReader?
    private var generation: UInt64 = 0
    private(set) var isRunning = false

    /// Owns model selection and permits an injected engine for controlled checks.
    init(engine: DetectorEngine = DetectorEngine()) {
        self.engine = engine
        detector = VideoFrameDetector(detector: engine)
    }

    /// Loads once, then awaits detection and consumption before reading another frame.
    /// The consumer receives success/empty/failure records and owns publication time.
    /// Cancellation and input/consumer errors throw; partial accounting is separate.
    func run(
        _ url: URL, sourceID: String, model: DetectorModel,
        selection: VideoFrameSelection = .all, loopIndex: Int = 0,
        trackID: CMPersistentTrackID? = nil, consume: @escaping Consumer
    ) async throws -> OfflineVideoRunSummary {
        guard !isRunning else { throw OfflineVideoError.alreadyRunning }
        let indices = try validatedIndices(selection)
        let wanted = indices.map { Set($0) }
        isRunning = true
        generation &+= 1
        let request = generation
        let reader = VideoFrameReader()
        activeReader = reader
        defer {
            activeReader = nil
            isRunning = false
        }
        do {
            try checkRequest(request)
            // Loading/compilation completes before source-frame readiness is measured.
            try await engine.selectModel(model)
            try checkRequest(request)
            try await reader.open(
                url, sourceID: sourceID, loopIndex: loopIndex, trackID: trackID)
            try checkRequest(request)
            var decoded = 0
            var processed = 0
            var completed = 0
            var failed = 0
            while indices == nil || processed < indices!.count {
                try checkRequest(request)
                let next = try await reader.nextFrame { frame -> SelectedFrame in
                    if let wanted, !wanted.contains(frame.metadata.sourceFrameIndex) {
                        return .unselected
                    }
                    return .selected(try VideoFrameConverter.convert(frame))
                }
                try checkRequest(request)
                guard let next else {
                    if let indices, processed < indices.count {
                        throw OfflineVideoError.missingSelectedFrame(indices[processed])
                    }
                    break
                }
                decoded += 1
                guard case .selected(let frame) = next else { continue }
                let result = try await detector.process(frame)
                try checkRequest(request)
                processed += 1
                if result.status == .completed { completed += 1 } else { failed += 1 }
                try await consume(result)
                try checkRequest(request)
            }
            await reader.cancel()
            try checkRequest(request)
            return OfflineVideoRunSummary(
                decodedFrames: decoded, processedFrames: processed,
                completedFrames: completed, failedFrames: failed)
        } catch {
            await reader.cancel()
            if error is CancellationError || Task.isCancelled || request != generation {
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
}
