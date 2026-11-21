import CoreGraphics
import CoreMedia
import Foundation

/// Checks exact media identities, irregular cadence, looping and invalid mappings.
@main
struct VideoPlaybackTimelineSmokeTests {
    /// Exercises arithmetic independently of decoders, sleeps and detector speed.
    static func main() throws {
        try testIrregularTimesAndLoops()
        try testNonzeroStartAndTimeBases()
        try testInvalidMappings()
        print("Video playback timeline: all 3 checks passed.")
    }

    /// Retains source indices and exact PTS while mapping repeated variable gaps.
    private static func testIrregularTimesAndLoops() throws {
        let range = CMTimeRange(
            start: .zero, duration: CMTime(value: 9009, timescale: 30000))
        let timeline = try VideoPlaybackTimeline(
            sourceID: "fixture", sourceTimeRange: range,
            anchorMilliseconds: 500, firstLoopIndex: 7)
        for loop in 7...9 {
            for (index, value) in [Int64(0), 1001, 4004, 6006].enumerated() {
                let pts = CMTime(value: value, timescale: 30000)
                let metadata = try frame(index: index, pts: pts, loop: loop)
                let due = try timeline.dueMilliseconds(for: metadata)
                let expected = 500 + Double((loop - 7) * 9009 + Int(value)) / 30
                precondition(abs(due - expected) < 1e-9)
                precondition(metadata.presentationTime == pts)
                precondition(metadata.loopIndex == loop)
                precondition(metadata.sourceFrameIndex == index)
            }
        }
        let end = try timeline.endMilliseconds(loopIndex: 9)
        precondition(abs(end - 1400.9) < 1e-9)
    }

    /// Handles negative source time, nonzero epochs and equivalent media time bases.
    private static func testNonzeroStartAndTimeBases() throws {
        let range = CMTimeRange(
            start: CMTime(value: -120, timescale: 60, flags: .valid, epoch: 4),
            duration: CMTime(value: 300, timescale: 60))
        let timeline = try VideoPlaybackTimeline(
            sourceID: "fixture", sourceTimeRange: range, anchorMilliseconds: 2000)
        let first = try frame(pts: range.start)
        let a = try frame(pts: CMTime(
            value: -1500, timescale: 1000, flags: [.valid, .hasBeenRounded], epoch: 4))
        let b = try frame(pts: CMTime(
            value: -90, timescale: 60, flags: .valid, epoch: 4))
        let firstDue = try timeline.dueMilliseconds(for: first)
        let aDue = try timeline.dueMilliseconds(for: a)
        let bDue = try timeline.dueMilliseconds(for: b)
        precondition(firstDue == 2000 && aDue == 2500 && bDue == 2500)
        precondition(a.presentationTime.value == -1500)
        precondition(a.presentationTime.timescale == 1000)
        precondition(a.presentationTime.flags == [.valid, .hasBeenRounded])
        precondition(a.presentationTime.epoch == 4)
        let end = try timeline.endMilliseconds(loopIndex: 2)
        precondition(end == 17000)
    }

    /// Rejects indefinite spans, foreign frames, invalid clocks and unsupported loops.
    private static func testInvalidMappings() throws {
        let range = CMTimeRange(start: .zero, duration: CMTime(value: 1, timescale: 1))
        for duration in [CMTime.zero, .invalid, .indefinite, .positiveInfinity,
                         CMTime(value: -1, timescale: 1)] {
            try expect(.invalidTimeRange) {
                _ = try VideoPlaybackTimeline(
                    sourceID: "fixture", sourceTimeRange: CMTimeRange(
                        start: .zero, duration: duration), anchorMilliseconds: 0)
            }
        }
        for time in [Double.nan, .infinity, -1] {
            try expect(.invalidHostTime) {
                _ = try VideoPlaybackTimeline(
                    sourceID: "fixture", sourceTimeRange: range,
                    anchorMilliseconds: time)
            }
        }
        let timeline = try VideoPlaybackTimeline(
            sourceID: "fixture", sourceTimeRange: range,
            anchorMilliseconds: 0, firstLoopIndex: 2)
        try expect(.mismatchedSource) {
            _ = try timeline.dueMilliseconds(for: frame(sourceID: "other", loop: 2))
        }
        try expect(.invalidLoop) {
            _ = try timeline.dueMilliseconds(for: frame(loop: 1))
        }
        try expect(.invalidLoop) {
            _ = try timeline.endMilliseconds(loopIndex: Int.max)
        }
        for pts in [CMTime(value: -1, timescale: 1000), CMTime(value: 1, timescale: 1),
                    CMTime(value: 0, timescale: 1, flags: .valid, epoch: 1)] {
            try expect(.timestampOutsideTrack) {
                _ = try timeline.dueMilliseconds(for: frame(pts: pts, loop: 2))
            }
        }
    }

    /// Builds validated metadata without deriving timestamps from frame indices.
    private static func frame(
        sourceID: String = "fixture", index: Int = 0, pts: CMTime = .zero,
        loop: Int = 0
    ) throws -> VideoFrameMetadata {
        try VideoFrameMetadata(
            sourceID: sourceID, sourceFrameIndex: index, presentationTime: pts,
            width: 64, height: 32, loopIndex: loop)
    }

    /// Requires a specific validation failure instead of accepting any thrown error.
    private static func expect(
        _ expected: VideoPlaybackError, operation: () throws -> Void
    ) throws {
        do {
            try operation()
            preconditionFailure("Invalid playback mapping succeeded")
        } catch let error as VideoPlaybackError {
            precondition(error == expected, "Expected \(expected), received \(error)")
        }
    }
}
