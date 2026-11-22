import CoreMedia
import Foundation

/// Identifies the stage at which a frame or run stopped.
nonisolated enum OfflineVideoStage: String, Sendable {
    case modelLoading, sourceOpening, decoding, conversion, inference, consumption
    case reconciliation
}

/// Keeps a stage and diagnostic without retaining a backend error or image buffer.
nonisolated struct OfflineVideoIssue: Sendable {
    let stage: OfflineVideoStage
    let details: VideoInferenceFailure

    /// Preserves the original error type/message for a specific pipeline stage.
    init(stage: OfflineVideoStage, error: any Error) {
        self.stage = stage
        details = VideoInferenceFailure(error)
    }

    /// Reuses diagnostics already captured by the single-frame detector.
    fileprivate init(stage: OfflineVideoStage, details: VideoInferenceFailure) {
        self.stage = stage
        self.details = details
    }
}

/// A terminal selected-frame outcome; unavailable source times remain nil.
nonisolated struct OfflineFrameOutcome: Sendable {
    enum Status: String, Sendable { case completed, failed, unfinished }

    let sourceFrameIndex: Int
    let metadata: VideoFrameMetadata?
    let readyMilliseconds: Double?
    let status: Status
    let result: VideoFrameResult?
    let issue: OfflineVideoIssue?
    let recordedMilliseconds: Double

    var presentationTime: CMTime? { metadata?.presentationTime }

    /// Records accepted inference output, including successful empty detections.
    init(result: VideoFrameResult) {
        sourceFrameIndex = result.metadata.sourceFrameIndex
        metadata = result.metadata
        readyMilliseconds = result.timings.readyMilliseconds
        status = result.status == .completed ? .completed : .failed
        self.result = result
        issue = result.failure.map {
            OfflineVideoIssue(stage: .inference, details: $0)
        }
        recordedMilliseconds = InferenceClock.nowMilliseconds()
    }

    /// Records interruption/input failure without inventing detections or timestamps.
    init(
        sourceFrameIndex: Int, metadata: VideoFrameMetadata?,
        readyMilliseconds: Double?, cancelled: Bool, issue: OfflineVideoIssue
    ) {
        self.sourceFrameIndex = sourceFrameIndex
        self.metadata = metadata
        self.readyMilliseconds = readyMilliseconds
        status = cancelled ? .unfinished : .failed
        result = nil
        self.issue = issue
        recordedMilliseconds = InferenceClock.nowMilliseconds()
    }
}

/// A value snapshot of the latest accepted run, surviving its cancellation/errors.
nonisolated struct OfflineVideoRunReport: Sendable {
    enum Status: String, Sendable { case running, completed, failed, cancelled }

    /// Internal execution identity; the handoff/export run ID is assigned separately.
    let id = UUID()
    let sourceID: String
    let model: DetectorModel
    let loopIndex: Int
    let selectedFrameIndices: [Int]?
    let previewEnabled: Bool
    let startedMilliseconds = InferenceClock.nowMilliseconds()
    let startedAt = Date()
    private(set) var endedAt: Date?
    private(set) var endedMilliseconds: Double?
    private(set) var status = Status.running
    private(set) var decodedFrames = 0
    private(set) var reachedEndOfSource = false
    private(set) var completedFrames = 0
    private(set) var failedFrames = 0
    private(set) var unfinishedFrames = 0
    private(set) var outcomes: [OfflineFrameOutcome] = []
    private(set) var issue: OfflineVideoIssue?

    var summary: OfflineVideoRunSummary {
        OfflineVideoRunSummary(
            decodedFrames: decodedFrames,
            processedFrames: completedFrames + failedFrames,
            completedFrames: completedFrames, failedFrames: failedFrames)
    }

    var coverage: OfflineVideoCoverage { OfflineVideoCoverage(report: self) }

    /// Checks a complete frozen selection, with exact source timestamps when available.
    func coverage(using plan: OfflineVideoCoveragePlan) -> OfflineVideoCoverage {
        OfflineVideoCoverage(report: self, plan: plan)
    }

    /// Binds a fresh report to the validated selection before loading any model.
    init(
        sourceID: String, model: DetectorModel, loopIndex: Int, indices: [Int]?,
        previewEnabled: Bool = false
    ) {
        self.sourceID = sourceID
        self.model = model
        self.loopIndex = loopIndex
        selectedFrameIndices = indices
        self.previewEnabled = previewEnabled
    }

    /// Appends numerical outcomes in source order without retaining frame pixels.
    mutating func append(_ outcome: OfflineFrameOutcome) {
        outcomes.append(outcome)
        switch outcome.status {
        case .completed: completedFrames += 1
        case .failed: failedFrames += 1
        case .unfinished: unfinishedFrames += 1
        }
    }

    /// Includes a decoded frame even if its conversion/callback was interrupted.
    mutating func recordDecodedFrames(_ count: Int) { decodedFrames = count }

    /// Records actual reader EOF instead of guessing source length from a partial run.
    mutating func recordEndOfSource() { reachedEndOfSource = true }

    /// Finalizes the snapshot before the runner returns or rethrows its error.
    mutating func finish(status: Status, issue: OfflineVideoIssue? = nil) {
        self.status = status
        self.issue = issue
        endedMilliseconds = InferenceClock.nowMilliseconds()
        endedAt = Date()
    }
}
