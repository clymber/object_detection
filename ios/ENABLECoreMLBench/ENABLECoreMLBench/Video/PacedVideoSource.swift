import CoreMedia
import Foundation

/// One immutable frame, its scheduled media instant and actual host delivery time.
nonisolated struct PacedVideoFrame: Sendable {
    let frame: UprightVideoFrame
    let dueMilliseconds: Double
    let deliveredMilliseconds: Double
}

/// Counts source handoffs and fully elapsed playback iterations, without inference.
nonisolated struct PacedVideoSourceSummary: Equatable, Sendable {
    var deliveredFrames = 0
    var completedLoops = 0
}

/// Decodes on a worker and releases frames at absolute PTS deadlines independently
/// of inference. The synchronous consumer must only hand off work, never run or
/// await detection. Late source frames keep their original deadline and identity.
actor PacedVideoSource {
    typealias Consumer = @Sendable (PacedVideoFrame) throws -> Void
    typealias Converter = @Sendable (VideoFrame) throws -> UprightVideoFrame
    typealias ReaderFactory = @Sendable () -> any VideoFrameReading

    enum State: Equatable, Sendable {
        case idle, running, completed, cancelled, failed
    }

    private(set) var state = State.idle
    private(set) var isRunning = false
    private(set) var timeline: VideoPlaybackTimeline?
    private(set) var summary = PacedVideoSourceSummary()
    private(set) var latestReport: PacedVideoSourceReport?
    private let clock: VideoPlaybackClock
    private let converter: Converter
    private let readerFactory: ReaderFactory
    private var activeReader: (any VideoFrameReading)?
    private var sleeper: Task<Void, Error>?
    private var generation: UInt64 = 0

    /// Keeps real decoding/conversion while allowing controlled clocks in checks.
    init(
        clock: VideoPlaybackClock = .continuous,
        readerFactory: @escaping ReaderFactory = { VideoFrameReader() },
        converter: @escaping Converter = { try VideoFrameConverter.convert($0) }
    ) {
        self.clock = clock
        self.converter = converter
        self.readerFactory = readerFactory
    }

    /// Runs once by default; nil loopCount repeats until cancellation. No frames are
    /// skipped here: PacedVideoScheduler classifies admission; freshness is separate.
    func run(
        _ url: URL, sourceID: String, firstLoopIndex: Int = 0,
        loopCount: Int? = 1, trackID: CMPersistentTrackID? = nil,
        consume: @escaping Consumer
    ) async throws -> PacedVideoSourceSummary {
        guard !isRunning else { throw VideoPlaybackError.alreadyRunning }
        guard firstLoopIndex >= 0, loopCount == nil || loopCount! > 0 else {
            throw VideoPlaybackError.invalidLoop
        }
        if let loopCount {
            let last = firstLoopIndex.addingReportingOverflow(loopCount - 1)
            guard !last.overflow, Int32(exactly: loopCount - 1) != nil else {
                throw VideoPlaybackError.invalidLoop
            }
        }
        isRunning = true
        state = .running
        timeline = nil
        summary = PacedVideoSourceSummary()
        latestReport = PacedVideoSourceReport(
            sourceID: sourceID, firstLoopIndex: firstLoopIndex, loopCount: loopCount,
            started: clock.nowMilliseconds())
        generation &+= 1
        let request = generation
        let reader = readerFactory()
        activeReader = reader
        let convert = converter
        var loopIndex = firstLoopIndex
        var selectedTrackID = trackID
        var stage = PacedVideoSourceStage.opening
        var activeRead: VideoFrameReadInfo?
        var activeDue: Double?
        var decodedBase = 0
        var framesInLoop = 0
        defer {
            activeReader = nil
            sleeper = nil
            isRunning = false
        }
        do {
            while true {
                try checkRequest(request)
                stage = .opening
                decodedBase = latestReport!.decodedFrames
                try await reader.open(
                    url, sourceID: sourceID, loopIndex: loopIndex,
                    trackID: selectedTrackID)
                try checkRequest(request)
                selectedTrackID = await reader.selectedTrackID
                latestReport?.selectedTrackID = selectedTrackID
                guard let timeRange = await reader.selectedTimeRange else {
                    throw VideoPlaybackError.invalidTimeRange
                }
                if let timeline, timeline.sourceTimeRange != timeRange {
                    throw VideoPlaybackError.sourceChanged
                }
                framesInLoop = 0
                while true {
                    stage = .decoding
                    activeRead = nil
                    activeDue = nil
                    let next = try await reader.nextFrame({ try convert($0) })
                    guard let frame = next else { break }
                    activeRead = VideoFrameReadInfo(
                        metadata: frame.metadata,
                        readyMilliseconds: frame.readyMilliseconds)
                    latestReport?.decodedFrames = decodedBase
                        + frame.metadata.sourceFrameIndex + 1
                    try checkRequest(request)
                    stage = .scheduling
                    if timeline == nil {
                        timeline = try VideoPlaybackTimeline(
                            sourceID: sourceID, sourceTimeRange: timeRange,
                            anchorMilliseconds: clock.nowMilliseconds(),
                            firstLoopIndex: firstLoopIndex)
                    }
                    latestReport?.timeline = timeline
                    let due = try timeline!.dueMilliseconds(for: frame.metadata)
                    activeDue = due
                    if framesInLoop == 0, loopIndex > firstLoopIndex {
                        try await wait(
                            until: timeline!.endMilliseconds(loopIndex: loopIndex - 1),
                            request: request)
                        summary.completedLoops += 1
                        latestReport?.completedLoops = summary.completedLoops
                    }
                    try await wait(until: due, request: request)
                    let delivered = clock.nowMilliseconds()
                    guard delivered.isFinite, delivered >= due else {
                        throw VideoPlaybackError.invalidHostTime
                    }
                    stage = .handoff
                    try consume(PacedVideoFrame(
                        frame: frame, dueMilliseconds: due,
                        deliveredMilliseconds: delivered))
                    summary.deliveredFrames += 1
                    latestReport?.deliveredFrames = summary.deliveredFrames
                    framesInLoop += 1
                    activeRead = nil
                    activeDue = nil
                    try checkRequest(request)
                }
                try checkRequest(request)
                guard framesInLoop > 0, let timeline else {
                    throw VideoPlaybackError.emptySource
                }
                // Prepare the next loop before its boundary; reopening costs never
                // move the anchor. The final loop also retains its last sample span.
                if let loopCount, summary.completedLoops + 1 == loopCount {
                    stage = .scheduling
                    try await wait(
                        until: timeline.endMilliseconds(loopIndex: loopIndex),
                        request: request)
                    summary.completedLoops += 1
                    state = .completed
                    latestReport?.finish(
                        status: .completed, at: clock.nowMilliseconds(),
                        summary: summary)
                    return summary
                }
                guard loopIndex < Int.max else { throw VideoPlaybackError.invalidLoop }
                loopIndex += 1
            }
        } catch {
            let cancelled = error is CancellationError || Task.isCancelled
                || request != generation
            let decoded = await reader.framesRead
            latestReport?.decodedFrames = decodedBase + decoded
            let decoderFailure = await reader.decodeFailureInfo
            if stage == .decoding, decoderFailure == nil, activeRead == nil,
                let info = await reader.lastFrameInfo,
                info.metadata.loopIndex == loopIndex,
                info.metadata.sourceFrameIndex >= framesInLoop
            {
                activeRead = info
                stage = .conversion
            }
            let now = clock.nowMilliseconds()
            if activeDue == nil,
                let pts = decoderFailure?.presentationTime
                    ?? activeRead?.metadata.presentationTime,
                let timeline
            {
                activeDue = try? timeline.dueMilliseconds(
                    presentationTime: pts, loopIndex: loopIndex)
            }
            let sourceError: any Error
            if decoderFailure != nil, let failure = await reader.failure {
                sourceError = failure
            } else {
                sourceError = cancelled ? CancellationError() : error
            }
            let issue = PacedVideoSourceIssue(
                stage: stage, sourceID: sourceID, loopIndex: loopIndex,
                attemptedFrameIndex: decoderFailure?.sourceFrameIndex
                    ?? activeRead?.metadata.sourceFrameIndex,
                presentationTime: decoderFailure?.presentationTime,
                read: activeRead, dueMilliseconds: activeDue,
                observedMilliseconds: now.isFinite && now >= 0 ? now : nil,
                cancelled: cancelled, decodeFailure: decoderFailure != nil,
                error: sourceError)
            await reader.cancel()
            state = cancelled ? .cancelled : .failed
            latestReport?.timeline = timeline
            latestReport?.finish(
                status: cancelled ? .cancelled : .failed,
                at: now, summary: summary, issue: issue)
            if cancelled { throw CancellationError() }
            throw error
        }
    }

    /// Interrupts source-clock sleeps as well as decoding; wait for run to unwind.
    func cancel() async {
        guard isRunning else { return }
        generation &+= 1
        sleeper?.cancel()
        await activeReader?.cancel()
    }

    /// Waits against one absolute clock, with task and explicit-stop cancellation.
    private func wait(until deadline: Double, request: UInt64) async throws {
        try checkRequest(request)
        while true {
            let now = clock.nowMilliseconds()
            guard now.isFinite, now >= 0 else {
                throw VideoPlaybackError.invalidHostTime
            }
            if now >= deadline { break }
            let sleepUntil = clock.sleepUntil
            let task = Task { try await sleepUntil(deadline) }
            sleeper = task
            defer { sleeper = nil }
            try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
            try checkRequest(request)
        }
        try checkRequest(request)
    }

    /// Rejects frames decoded or awakened after a stop, cancellation or old run.
    private func checkRequest(_ request: UInt64) throws {
        try Task.checkCancellation()
        guard request == generation else { throw CancellationError() }
    }
}
