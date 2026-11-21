import CoreMedia
import Foundation

/// Distinguishes reader losses from conversion, scheduling and handoff interruptions.
nonisolated enum PacedVideoSourceStage: String, Sendable {
    case opening, decoding, conversion, scheduling, handoff
}

/// Preserves known source identity without inventing missing media timestamps/counts.
nonisolated struct PacedVideoSourceIssue: Sendable {
    let stage: PacedVideoSourceStage
    let sourceID: String
    let loopIndex: Int
    let attemptedFrameIndex: Int?
    let presentationTime: CMTime?
    let metadata: VideoFrameMetadata?
    let readyMilliseconds: Double?
    let dueMilliseconds: Double?
    let observedMilliseconds: Double?
    let cancelled: Bool
    let decodeFailure: Bool
    let details: VideoInferenceFailure

    /// A decoder failure can hide an unknown tail; one error is not one lost frame.
    var hasUnreconstructedDecodeLoss: Bool { decodeFailure }
    var lostFrameCount: Int? { nil }

    /// Keeps the actual sample time when available, including invalid CMTime flags.
    init(
        stage: PacedVideoSourceStage, sourceID: String, loopIndex: Int,
        attemptedFrameIndex: Int?, presentationTime: CMTime?,
        read: VideoFrameReadInfo?, dueMilliseconds: Double?,
        observedMilliseconds: Double?, cancelled: Bool, decodeFailure: Bool = false,
        error: any Error
    ) {
        self.stage = stage
        self.sourceID = sourceID
        self.loopIndex = loopIndex
        self.attemptedFrameIndex = attemptedFrameIndex
        self.presentationTime = presentationTime ?? read?.metadata.presentationTime
        metadata = read?.metadata
        readyMilliseconds = read?.readyMilliseconds
        self.dueMilliseconds = dueMilliseconds
        self.observedMilliseconds = observedMilliseconds
        self.cancelled = cancelled
        self.decodeFailure = decodeFailure
        details = VideoInferenceFailure(error)
    }
}

/// A pixel-free source log retained after success, cancellation and read failures.
nonisolated struct PacedVideoSourceReport: Sendable {
    enum Status: String, Sendable { case running, completed, cancelled, failed }

    let sourceID: String
    let firstLoopIndex: Int
    let requestedLoopCount: Int?
    let startedMilliseconds: Double?
    var selectedTrackID: CMPersistentTrackID?
    var timeline: VideoPlaybackTimeline?
    var decodedFrames = 0
    var deliveredFrames = 0
    var completedLoops = 0
    var endedMilliseconds: Double?
    var status = Status.running
    var issue: PacedVideoSourceIssue?

    var decodeLosses: [PacedVideoSourceIssue] {
        issue.map { $0.hasUnreconstructedDecodeLoss ? [$0] : [] } ?? []
    }

    /// Records setup separately from the playback anchor used for rate denominators.
    init(sourceID: String, firstLoopIndex: Int, loopCount: Int?, started: Double) {
        self.sourceID = sourceID
        self.firstLoopIndex = firstLoopIndex
        requestedLoopCount = loopCount
        startedMilliseconds = started.isFinite && started >= 0 ? started : nil
    }

    /// Does not call clean EOF verified coverage without a frozen source reference.
    mutating func finish(
        status: Status, at milliseconds: Double, summary: PacedVideoSourceSummary,
        issue: PacedVideoSourceIssue? = nil
    ) {
        self.status = status
        endedMilliseconds = milliseconds.isFinite && milliseconds >= 0
            ? milliseconds : nil
        deliveredFrames = summary.deliveredFrames
        completedLoops = summary.completedLoops
        self.issue = issue
    }
}
