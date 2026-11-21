import CoreGraphics
import Foundation

/// Pairs the immutable upright input with the result produced from that exact frame.
nonisolated struct VideoFramePreview: Identifiable, Sendable {
    enum ValidationError: Error { case mismatchedFrame }

    let id = UUID()
    let image: CGImage
    let result: VideoFrameResult

    /// Rejects mismatched identity, orientation mapping or upright image dimensions.
    init(frame: UprightVideoFrame, result: VideoFrameResult) throws {
        guard frame.metadata == result.metadata,
            frame.readyMilliseconds == result.timings.readyMilliseconds,
            frame.image.width == result.sourceWidth,
            frame.image.height == result.sourceHeight,
            frame.bufferToImage == result.bufferToImage
        else { throw ValidationError.mismatchedFrame }
        image = frame.image
        self.result = result
    }

    /// Sends only identity and clock endpoints back from the renderer, never pixels.
    var drawToken: VideoPreviewDrawToken {
        VideoPreviewDrawToken(
            id: id, readyMilliseconds: result.timings.readyMilliseconds,
            doneMilliseconds: result.timings.doneMilliseconds)
    }
}

/// Carries enough information to count a distinct drawn frame without retaining it.
nonisolated struct VideoPreviewDrawToken: Sendable {
    let id: UUID
    let readyMilliseconds: Double
    let doneMilliseconds: Double
}

/// Counts unique preview draws separately from detector completion or screen refresh.
nonisolated struct VideoPreviewStatistics: Sendable {
    private(set) var drawnFrames = 0
    private var firstReadyMilliseconds: Double?
    private var lastDrawnMilliseconds: Double?
    private var drawnIDs: Set<UUID> = []

    var framesPerSecond: Double? {
        guard let firstReadyMilliseconds, let lastDrawnMilliseconds,
            lastDrawnMilliseconds > firstReadyMilliseconds
        else { return nil }
        return Double(drawnFrames) * 1000
            / (lastDrawnMilliseconds - firstReadyMilliseconds)
    }

    /// Includes frames delivered while the preview is offscreen in the time window.
    @discardableResult
    mutating func admit(_ token: VideoPreviewDrawToken) -> Bool {
        guard token.readyMilliseconds.isFinite, token.doneMilliseconds.isFinite,
            token.readyMilliseconds >= 0,
            token.doneMilliseconds >= token.readyMilliseconds
        else { return false }
        firstReadyMilliseconds = min(
            firstReadyMilliseconds ?? token.readyMilliseconds, token.readyMilliseconds)
        return true
    }

    /// Ignores frozen-frame redraws and invalid samples; supports delayed callbacks.
    mutating func record(_ token: VideoPreviewDrawToken, at milliseconds: Double) {
        guard canRecord(token, at: milliseconds) else { return }
        admit(token)
        drawnIDs.insert(token.id)
        drawnFrames += 1
        lastDrawnMilliseconds = max(lastDrawnMilliseconds ?? milliseconds, milliseconds)
    }

    /// Checks before observable mutation, preventing a draw/state-update feedback loop.
    func canRecord(_ token: VideoPreviewDrawToken, at milliseconds: Double) -> Bool {
        milliseconds.isFinite && milliseconds >= token.doneMilliseconds
            && token.readyMilliseconds.isFinite && token.doneMilliseconds.isFinite
            && token.readyMilliseconds >= 0
            && token.doneMilliseconds >= token.readyMilliseconds
            && !drawnIDs.contains(token.id)
    }
}
