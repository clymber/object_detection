import CoreMedia
import Foundation

/// Rejects unusable playback mappings rather than guessing cadence from nominal FPS.
nonisolated enum VideoPlaybackError: Error, Equatable, LocalizedError {
    case invalidTimeRange
    case invalidHostTime
    case invalidLoop
    case mismatchedSource
    case timestampOutsideTrack
    case sourceChanged
    case emptySource
    case alreadyRunning

    var errorDescription: String? {
        switch self {
        case .invalidTimeRange: return "Video playback needs a finite track duration."
        case .invalidHostTime: return "The playback clock returned an invalid time."
        case .invalidLoop: return "Playback loop indices and count are invalid."
        case .mismatchedSource: return "The frame belongs to another playback source."
        case .timestampOutsideTrack: return "The frame PTS is outside its track range."
        case .sourceChanged: return "The video track changed while looping."
        case .emptySource: return "The video track contains no decoded frames."
        case .alreadyRunning: return "A paced video source is already running."
        }
    }
}

/// Maps original media PTS to one host-clock anchor across every playback iteration.
nonisolated struct VideoPlaybackTimeline: Sendable {
    let sourceID: String
    let firstLoopIndex: Int
    let sourceTimeRange: CMTimeRange
    let anchorMilliseconds: Double

    /// Anchors the selected track's start after startup and first-frame preparation.
    init(
        sourceID: String, sourceTimeRange: CMTimeRange,
        anchorMilliseconds: Double, firstLoopIndex: Int = 0
    ) throws {
        guard !sourceID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw VideoPlaybackError.mismatchedSource
        }
        guard firstLoopIndex >= 0 else { throw VideoPlaybackError.invalidLoop }
        guard anchorMilliseconds.isFinite, anchorMilliseconds >= 0 else {
            throw VideoPlaybackError.invalidHostTime
        }
        guard sourceTimeRange.isValid, sourceTimeRange.start.isNumeric,
            sourceTimeRange.duration.isNumeric,
            sourceTimeRange.duration.epoch == 0,
            CMTimeCompare(sourceTimeRange.duration, .zero) > 0,
            sourceTimeRange.end.isNumeric
        else { throw VideoPlaybackError.invalidTimeRange }
        self.sourceID = sourceID
        self.firstLoopIndex = firstLoopIndex
        self.sourceTimeRange = sourceTimeRange
        self.anchorMilliseconds = anchorMilliseconds
    }

    /// Preserves irregular gaps and source time bases; never rewrites frame metadata.
    func dueMilliseconds(for metadata: VideoFrameMetadata) throws -> Double {
        guard metadata.sourceID == sourceID else {
            throw VideoPlaybackError.mismatchedSource
        }
        let pts = metadata.presentationTime
        guard pts.epoch == sourceTimeRange.start.epoch,
            CMTimeRangeContainsTime(sourceTimeRange, time: pts)
        else { throw VideoPlaybackError.timestampOutsideTrack }
        return try hostTime(
            loopIndex: metadata.loopIndex,
            offset: CMTimeSubtract(pts, sourceTimeRange.start))
    }

    /// Includes the final frame's presentation duration when crossing a loop boundary.
    func endMilliseconds(loopIndex: Int) throws -> Double {
        try hostTime(loopIndex: loopIndex, offset: sourceTimeRange.duration)
    }

    /// Computes from the original anchor so decode or dispatch delays cannot add drift.
    private func hostTime(loopIndex: Int, offset: CMTime) throws -> Double {
        guard loopIndex >= firstLoopIndex,
            let iterations = Int32(exactly: loopIndex - firstLoopIndex)
        else { throw VideoPlaybackError.invalidLoop }
        let elapsed = CMTimeAdd(
            CMTimeMultiply(sourceTimeRange.duration, multiplier: iterations), offset)
        let due = anchorMilliseconds + CMTimeGetSeconds(elapsed) * 1000
        guard elapsed.isNumeric, due.isFinite, due >= anchorMilliseconds else {
            throw VideoPlaybackError.invalidHostTime
        }
        return due
    }
}

/// Injects deterministic time in checks while production shares the inference clock.
nonisolated struct VideoPlaybackClock: Sendable {
    let nowMilliseconds: @Sendable () -> Double
    let sleepUntil: @Sendable (Double) async throws -> Void

    static let continuous = VideoPlaybackClock(
        nowMilliseconds: { InferenceClock.nowMilliseconds() },
        sleepUntil: { try await InferenceClock.sleep(untilMilliseconds: $0) })
}
