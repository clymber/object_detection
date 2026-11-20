import CoreMedia
import Foundation

/// Identifies a selected source frame; an optional PTS binds a frozen sequence.
nonisolated struct OfflineVideoFrameReference: Sendable {
    let sourceFrameIndex: Int
    let presentationTime: CMTime?
}

/// Describes the entire expected selection, not just a subset of a run's output.
nonisolated struct OfflineVideoCoveragePlan: Sendable {
    enum ValidationError: Error { case invalidSource, invalidFrames }

    let sourceID: String
    let loopIndex: Int
    let frames: [OfflineVideoFrameReference]

    var selection: VideoFrameSelection { .indices(frames.map(\.sourceFrameIndex)) }

    /// Requires unique increasing indices and valid exact source PTS when supplied.
    init(
        sourceID: String, loopIndex: Int = 0, frames: [OfflineVideoFrameReference]
    ) throws {
        guard !sourceID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            loopIndex >= 0
        else { throw ValidationError.invalidSource }
        guard frames.allSatisfy({ reference in
            reference.sourceFrameIndex >= 0 && (reference.presentationTime.map {
                $0.isNumeric && $0.timescale > 0
            } ?? true)
        }), zip(frames, frames.dropFirst()).allSatisfy({
            $0.sourceFrameIndex < $1.sourceFrameIndex
        }) else { throw ValidationError.invalidFrames }
        self.sourceID = sourceID
        self.loopIndex = loopIndex
        self.frames = frames
    }
}

/// Reconciles expected identities against raw outcomes, independently of cached counts.
nonisolated struct OfflineVideoCoverage: Sendable {
    enum Basis: String, Sendable {
        case selection, decodedSource, referenceSequence, unknownSource
    }

    let basis: Basis
    let expectedFrameCount: Int?
    let observedFrameCount: Int
    let missingFrameIndices: [Int]
    let duplicateFrameIndices: [Int]
    let unexpectedFrameIndices: [Int]
    let unfinishedFrameIndices: [Int]
    let failedFrameIndices: [Int]
    let invalidOutcomeIndices: [Int]
    let timestampMismatchIndices: [Int]
    let unknownTimestampIndices: [Int]
    let outOfOrderFrameIndices: [Int]
    let expectationMatchesRun: Bool
    let countsAreConsistent: Bool
    let isFinalized: Bool
    let sourceRemainderUnknown: Bool
    private let runCompleted: Bool

    /// Checks identities/counts before the runner seals its final report.
    var hasCompleteOutcomeSet: Bool {
        expectedFrameCount != nil && !sourceRemainderUnknown
            && expectationMatchesRun && countsAreConsistent
            && missingFrameIndices.isEmpty && duplicateFrameIndices.isEmpty
            && unexpectedFrameIndices.isEmpty && invalidOutcomeIndices.isEmpty
            && timestampMismatchIndices.isEmpty && outOfOrderFrameIndices.isEmpty
    }

    /// Counts failed/unfinished statuses as accounted, without claiming success.
    var isFullyAccounted: Bool { isFinalized && hasCompleteOutcomeSet }

    /// Requires accepted successful inference for every frame, including empty results.
    var isFullyProcessed: Bool {
        isFullyAccounted && runCompleted && failedFrameIndices.isEmpty
            && unfinishedFrameIndices.isEmpty && unknownTimestampIndices.isEmpty
    }

    /// Uses explicit selection, observed EOF extent or a full reference sequence.
    init(report: OfflineVideoRunReport, plan: OfflineVideoCoveragePlan? = nil) {
        let expected: [Int]?
        if let plan {
            basis = .referenceSequence
            expected = plan.frames.map(\.sourceFrameIndex)
        } else if let selected = report.selectedFrameIndices {
            basis = .selection
            expected = selected
        } else if report.reachedEndOfSource && report.decodedFrames >= 0 {
            basis = .decodedSource
            expected = Array(0..<report.decodedFrames)
        } else {
            basis = .unknownSource
            expected = nil
        }
        expectedFrameCount = expected?.count
        sourceRemainderUnknown = expected == nil
        let expectedSet = expected.map { Set($0) }
        let references = Dictionary(uniqueKeysWithValues:
            (plan?.frames ?? []).map { ($0.sourceFrameIndex, $0) })
        let validExpected = expected.map {
            $0.allSatisfy { $0 >= 0 }
                && zip($0, $0.dropFirst()).allSatisfy { $0 < $1 }
        } ?? true
        expectationMatchesRun = validExpected && (plan.map {
            $0.sourceID == report.sourceID && $0.loopIndex == report.loopIndex
                && (report.selectedFrameIndices.map { $0 == expected } ?? true)
        } ?? true)

        var occurrences: [Int: Int] = [:]
        var invalid = Set<Int>()
        var timestampMismatch = Set<Int>()
        var unknownTimestamp = Set<Int>()
        var outOfOrder = Set<Int>()
        var failed = Set<Int>()
        var unfinished = Set<Int>()
        var completedCount = 0
        var failedCount = 0
        var unfinishedCount = 0
        var decodedExtentValid = report.decodedFrames >= 0
        var previousIndex: Int?
        for outcome in report.outcomes {
            let index = outcome.sourceFrameIndex
            occurrences[index, default: 0] += 1
            if index < 0 { invalid.insert(index) }
            if let previousIndex, index < previousIndex { outOfOrder.insert(index) }
            previousIndex = index
            if let metadata = outcome.metadata {
                if metadata.sourceID != report.sourceID
                    || metadata.loopIndex != report.loopIndex
                    || metadata.sourceFrameIndex != index
                    || index >= report.decodedFrames {
                    invalid.insert(index)
                }
                if index < 0 || index >= report.decodedFrames {
                    decodedExtentValid = false
                }
                if let expectedTime = references[index]?.presentationTime,
                    !Self.sameTimestamp(metadata.presentationTime, expectedTime) {
                    timestampMismatch.insert(index)
                }
            } else {
                unknownTimestamp.insert(index)
            }
            switch outcome.status {
            case .completed:
                completedCount += 1
                guard let result = outcome.result, result.status == .completed,
                    result.failure == nil, outcome.issue == nil
                else { invalid.insert(index); continue }
            case .failed:
                failedCount += 1
                failed.insert(index)
                if outcome.issue == nil { invalid.insert(index) }
                if let result = outcome.result {
                    if result.status != .failed || result.failure == nil {
                        invalid.insert(index)
                    }
                } else if outcome.issue?.stage != .decoding
                    && outcome.issue?.stage != .conversion {
                    invalid.insert(index)
                }
            case .unfinished:
                unfinishedCount += 1
                unfinished.insert(index)
                if outcome.result != nil || outcome.metadata == nil
                    || outcome.issue == nil { invalid.insert(index) }
            }
            if let result = outcome.result {
                if result.metadata != outcome.metadata || outcome.metadata == nil
                    || result.metadata.sourceFrameIndex != index {
                    invalid.insert(index)
                }
            } else if outcome.metadata == nil && outcome.issue?.stage != .decoding {
                invalid.insert(index)
            }
        }
        let observed = Set(occurrences.keys)
        observedFrameCount = report.outcomes.count
        // With an unknown source extent, only missing decoded-prefix frames are known.
        let required = expectedSet ?? Set(0..<max(0, report.decodedFrames))
        missingFrameIndices = required.subtracting(observed).sorted()
        duplicateFrameIndices = occurrences.filter { $0.value > 1 }.keys.sorted()
        unexpectedFrameIndices = expectedSet.map {
            observed.subtracting($0).sorted()
        } ?? []
        failedFrameIndices = failed.sorted()
        unfinishedFrameIndices = unfinished.sorted()
        invalidOutcomeIndices = invalid.sorted()
        timestampMismatchIndices = timestampMismatch.sorted()
        unknownTimestampIndices = unknownTimestamp.sorted()
        outOfOrderFrameIndices = outOfOrder.sorted()
        countsAreConsistent = decodedExtentValid
            && report.completedFrames == completedCount
            && report.failedFrames == failedCount
            && report.unfinishedFrames == unfinishedCount
        isFinalized = report.status != .running && report.endedMilliseconds != nil
        runCompleted = report.status == .completed
    }

    /// Compares raw PTS/time-base identity rather than silently normalizing timestamps.
    private static func sameTimestamp(_ actual: CMTime, _ expected: CMTime) -> Bool {
        actual.value == expected.value && actual.timescale == expected.timescale
            && actual.epoch == expected.epoch && actual.flags == expected.flags
    }
}
