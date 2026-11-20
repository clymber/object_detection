import CoreGraphics
import CoreMedia
import Foundation

/// Exercises coverage false-pass protections independently of the runner counters.
@main
struct OfflineVideoCoverageSmokeTests {
    /// Covers extents, bad identities/order, frozen PTS and unfinished work.
    static func main() throws {
        try testSparseAndEmpty()
        try testUnknownSource()
        try testDuplicatesAndOrder()
        try testFailedAndUnfinished()
        try testFrozenSequence()
        try testBadMetadataAndCounts()
        try testRunningSnapshot()
        try testInvalidPlans()
        print("Video coverage: all 8 reconciliation and false-pass checks passed.")
    }

    /// Includes empty detections and a deliberately empty selection as valid coverage.
    private static func testSparseAndEmpty() throws {
        let report = makeReport(indices: [0, 2], decoded: 3,
                                outcomes: [try completed(0), try completed(2)])
        let coverage = report.coverage
        precondition(coverage.basis == .selection && coverage.expectedFrameCount == 2)
        precondition(coverage.isFullyAccounted && coverage.isFullyProcessed)
        precondition(coverage.missingFrameIndices.isEmpty)
        let empty = makeReport(indices: [], decoded: 0, outcomes: []).coverage
        precondition(empty.expectedFrameCount == 0 && empty.isFullyAccounted)
    }

    /// Does not infer full source length from an interrupted decoded prefix.
    private static func testUnknownSource() throws {
        let outcomes = [try completed(0), try completed(1)]
        let drained = makeReport(
            indices: nil, decoded: 2, outcomes: outcomes, eof: true)
        precondition(drained.coverage.basis == .decodedSource)
        precondition(drained.coverage.isFullyProcessed)
        let partial = makeReport(indices: nil, decoded: 2, outcomes: outcomes,
                                 status: .cancelled)
        precondition(partial.coverage.basis == .unknownSource)
        precondition(partial.coverage.sourceRemainderUnknown)
        precondition(partial.coverage.expectedFrameCount == nil)
        precondition(!partial.coverage.isFullyAccounted)
        let plan = try makePlan([0, 1, 2, 3])
        let known = partial.coverage(using: plan)
        precondition(!known.sourceRemainderUnknown && known.expectedFrameCount == 4)
        precondition(known.missingFrameIndices == [2, 3] && !known.isFullyAccounted)
    }

    /// Rejects equal aggregate counts when identities are duplicated or reordered.
    private static func testDuplicatesAndOrder() throws {
        let report = makeReport(indices: [0, 1], decoded: 4, outcomes: [
            try completed(1), try completed(0), try completed(0), try completed(3),
        ])
        let coverage = report.coverage
        precondition(coverage.observedFrameCount == 4)
        precondition(coverage.duplicateFrameIndices == [0])
        precondition(coverage.unexpectedFrameIndices == [3])
        precondition(coverage.outOfOrderFrameIndices == [0])
        precondition(!coverage.isFullyAccounted && !coverage.isFullyProcessed)
        let missing = makeReport(
            indices: [0, 1], decoded: 2,
            outcomes: [try completed(0), try completed(0)]).coverage
        precondition(missing.missingFrameIndices == [1])
        precondition(missing.duplicateFrameIndices == [0])
        precondition(!missing.isFullyAccounted)
        let badSelection = makeReport(indices: [0, 0], decoded: 1,
                                      outcomes: [try completed(0)]).coverage
        precondition(!badSelection.expectationMatchesRun)
        precondition(!badSelection.isFullyAccounted)
    }

    /// Accounted terminal failures are never mistaken for successful full processing.
    private static func testFailedAndUnfinished() throws {
        let unfinished = OfflineFrameOutcome(
            sourceFrameIndex: 1, metadata: try metadata(1), readyMilliseconds: 0,
            cancelled: true,
            issue: OfflineVideoIssue(stage: .inference, error: CancellationError()))
        let failed = OfflineFrameOutcome(
            sourceFrameIndex: 2, metadata: nil, readyMilliseconds: nil,
            cancelled: false,
            issue: OfflineVideoIssue(
                stage: .decoding, error: CoverageFixtureError.failed))
        let report = makeReport(indices: [0, 1, 2], decoded: 2,
                                outcomes: [try completed(0), unfinished, failed],
                                status: .cancelled)
        precondition(report.coverage.isFullyAccounted)
        precondition(!report.coverage.isFullyProcessed)
        precondition(report.coverage.failedFrameIndices == [2])
        precondition(report.coverage.unfinishedFrameIndices == [1])
        precondition(report.coverage.unknownTimestampIndices == [2])
        let partial = makeReport(
            indices: [0, 1, 2, 3], decoded: 2,
            outcomes: [try completed(0), unfinished], status: .cancelled)
        precondition(partial.coverage.missingFrameIndices == [2, 3])
        precondition(!partial.coverage.isFullyAccounted)
    }

    /// Matches exact PTS/time bases and binds references to source/loop/selection.
    private static func testFrozenSequence() throws {
        let report = makeReport(indices: [0, 2], decoded: 3,
                                outcomes: [try completed(0), try completed(2)])
        let valid = try makePlan([0, 2])
        precondition(report.coverage(using: valid).isFullyProcessed)
        let wrongTime = try OfflineVideoCoveragePlan(
            sourceID: "coverage-source", frames: [
                OfflineVideoFrameReference(sourceFrameIndex: 0,
                    presentationTime: CMTime(value: 0, timescale: 30000)),
                OfflineVideoFrameReference(sourceFrameIndex: 2,
                    presentationTime: CMTime(value: 4004, timescale: 60000)),
            ])
        precondition(report.coverage(using: wrongTime).timestampMismatchIndices == [2])
        precondition(!report.coverage(using: wrongTime).isFullyAccounted)
        let wrongSource = try OfflineVideoCoveragePlan(
            sourceID: "other-source", frames: valid.frames)
        precondition(!report.coverage(using: wrongSource).expectationMatchesRun)
        let wrongLoop = try OfflineVideoCoveragePlan(
            sourceID: "coverage-source", loopIndex: 1, frames: valid.frames)
        precondition(!report.coverage(using: wrongLoop).expectationMatchesRun)
        let wrongSelection = try makePlan([0, 1, 2])
        precondition(!report.coverage(using: wrongSelection).expectationMatchesRun)
    }

    /// Detects wrong source/loop identities and impossible decoded-count claims.
    private static func testBadMetadataAndCounts() throws {
        let wrongSource = makeReport(
            indices: [0], decoded: 1,
            outcomes: [try completed(0, source: "wrong-source")])
        precondition(wrongSource.coverage.invalidOutcomeIndices == [0])
        let wrongLoop = makeReport(indices: [0], decoded: 1,
                                   outcomes: [try completed(0, loop: 2)])
        precondition(wrongLoop.coverage.invalidOutcomeIndices == [0])
        let wrongCount = makeReport(
            indices: [0], decoded: 0, outcomes: [try completed(0)])
        precondition(!wrongCount.coverage.countsAreConsistent)
        precondition(!wrongCount.coverage.isFullyAccounted)
        let wrongIndex = OfflineFrameOutcome(
            sourceFrameIndex: 0, metadata: try metadata(1), readyMilliseconds: 0,
            cancelled: true,
            issue: OfflineVideoIssue(stage: .inference, error: CancellationError()))
        let bad = makeReport(indices: [0], decoded: 2, outcomes: [wrongIndex],
                             status: .cancelled).coverage
        precondition(bad.invalidOutcomeIndices == [0] && !bad.isFullyAccounted)
    }

    /// Does not certify an unsealed snapshot even when all requested rows are present.
    private static func testRunningSnapshot() throws {
        var report = OfflineVideoRunReport(
            sourceID: "coverage-source", model: .rfdetrSmall,
            loopIndex: 0, indices: [0])
        report.recordDecodedFrames(1)
        report.append(try completed(0))
        precondition(report.coverage.hasCompleteOutcomeSet)
        precondition(!report.coverage.isFinalized && !report.coverage.isFullyAccounted)
    }

    /// Rejects ambiguous manifests rather than silently sorting or repairing them.
    private static func testInvalidPlans() throws {
        for indices in [[-1], [0, 0], [2, 1]] {
            do {
                _ = try makePlan(indices)
                preconditionFailure("Invalid frozen selection was accepted")
            } catch OfflineVideoCoveragePlan.ValidationError.invalidFrames {}
        }
        do {
            _ = try OfflineVideoCoveragePlan(sourceID: " ", frames: [])
            preconditionFailure("Empty source identity was accepted")
        } catch OfflineVideoCoveragePlan.ValidationError.invalidSource {}
        do {
            _ = try OfflineVideoCoveragePlan(sourceID: "coverage-source", frames: [
                OfflineVideoFrameReference(
                    sourceFrameIndex: 0, presentationTime: .indefinite)
            ])
            preconditionFailure("Invalid reference timestamp was accepted")
        } catch OfflineVideoCoveragePlan.ValidationError.invalidFrames {}
    }

    /// Creates a finished value report without relying on the real decoder's counters.
    private static func makeReport(
        indices: [Int]?, decoded: Int, outcomes: [OfflineFrameOutcome],
        eof: Bool = false, status: OfflineVideoRunReport.Status = .completed
    ) -> OfflineVideoRunReport {
        var report = OfflineVideoRunReport(
            sourceID: "coverage-source", model: .rfdetrSmall, loopIndex: 0,
            indices: indices)
        report.recordDecodedFrames(decoded)
        for outcome in outcomes { report.append(outcome) }
        if eof { report.recordEndOfSource() }
        report.finish(status: status)
        return report
    }

    /// Supplies exact reference identities independently of frame-result construction.
    private static func makePlan(_ indices: [Int]) throws -> OfflineVideoCoveragePlan {
        try OfflineVideoCoveragePlan(sourceID: "coverage-source", frames: indices.map {
            OfflineVideoFrameReference(sourceFrameIndex: $0,
                presentationTime: CMTime(value: Int64($0) * 1001, timescale: 30000))
        })
    }

    /// Supplies valid source metadata while allowing identity-corruption assertions.
    private static func metadata(
        _ index: Int, source: String = "coverage-source", loop: Int = 0
    ) throws -> VideoFrameMetadata {
        try VideoFrameMetadata(
            sourceID: source, sourceFrameIndex: index,
            presentationTime: CMTime(value: Int64(index) * 1001, timescale: 30000),
            width: 32, height: 16, loopIndex: loop)
    }

    /// Treats a successful empty prediction as a genuine completed frame.
    private static func completed(
        _ index: Int, source: String = "coverage-source", loop: Int = 0
    ) throws -> OfflineFrameOutcome {
        let ready = InferenceClock.nowMilliseconds()
        return OfflineFrameOutcome(result: VideoFrameResult(
            metadata: try metadata(index, source: source, loop: loop),
            sourceWidth: 32, sourceHeight: 16, bufferToImage: .identity,
            status: .completed, detections: [], failure: nil,
            timings: try VideoFrameTimings(
                readyMilliseconds: ready, modelTiming: nil,
                doneMilliseconds: InferenceClock.nowMilliseconds())))
    }
}

/// Supplies a nonempty diagnostic for a source frame whose timestamp is unavailable.
nonisolated private enum CoverageFixtureError: Error { case failed }
