import CoreGraphics
import CoreMedia
import Foundation

/// Checks exact frame pairing and independent renderer acknowledgements.
@main
struct VideoPreviewSmokeTests {
    /// Verifies geometry/identity guards, immutable pixels and draw-rate arithmetic.
    static func main() throws {
        let frame = try makeFrame()
        let good = try result(frame)
        let preview = try VideoFramePreview(frame: frame, result: good)
        precondition(preview.image === frame.image)
        precondition(preview.result.metadata == frame.metadata)
        precondition(preview.result.detections == good.detections)
        precondition(preview.result.timings == good.timings)
        let wrongMetadata = try VideoFrameMetadata(
            sourceID: "another-video", sourceFrameIndex: 0,
            presentationTime: CMTime(value: 7, timescale: 30), width: 16, height: 8)
        for invalid in [
            try result(frame, metadata: wrongMetadata),
            try result(frame, width: 16),
            try result(frame, transform: .identity),
            try result(frame, ready: 99),
        ] {
            do {
                _ = try VideoFramePreview(frame: frame, result: invalid)
                preconditionFailure("Mismatched pixels/results were paired")
            } catch VideoFramePreview.ValidationError.mismatchedFrame {}
        }
        let failed = try VideoFramePreview(
            frame: frame, result: result(frame, failed: true))
        precondition(
            failed.result.detections.isEmpty && failed.result.status == .failed)
        try testDrawStatistics()
        print("Video preview: frame pairing, timing isolation and draw counts passed.")
    }

    /// Counts unique drawn frames rather than admitted results or repeat paint passes.
    private static func testDrawStatistics() throws {
        var stats = VideoPreviewStatistics()
        let first = VideoPreviewDrawToken(
            id: UUID(), readyMilliseconds: 100, doneMilliseconds: 120)
        let second = VideoPreviewDrawToken(
            id: UUID(), readyMilliseconds: 200, doneMilliseconds: 220)
        stats.admit(first)
        stats.admit(second)
        precondition(stats.drawnFrames == 0 && stats.framesPerSecond == nil)
        stats.record(first, at: 119)
        stats.record(first, at: .infinity)
        stats.record(first, at: .nan)
        precondition(stats.drawnFrames == 0)
        stats.record(first, at: 150)
        precondition(stats.drawnFrames == 1 && stats.framesPerSecond == 20)
        stats.record(second, at: 250)
        let rate = 2000.0 / 150
        precondition(stats.drawnFrames == 2 && stats.framesPerSecond == rate)
        stats.record(first, at: 1000)
        stats.record(second, at: 1100)
        precondition(stats.drawnFrames == 2 && stats.framesPerSecond == rate)
        let invalid = VideoPreviewDrawToken(
            id: UUID(), readyMilliseconds: .nan, doneMilliseconds: 100)
        stats.record(invalid, at: 2000)
        precondition(stats.drawnFrames == 2 && stats.framesPerSecond == rate)
        // Another accepted frame that never draws does not inflate display counts.
        stats.admit(VideoPreviewDrawToken(
            id: UUID(), readyMilliseconds: 300, doneMilliseconds: 400))
        precondition(stats.drawnFrames == 2 && stats.framesPerSecond == rate)
    }

    /// Uses a rotated frame whose upright dimensions differ from the encoded buffer.
    private static func makeFrame() throws -> UprightVideoFrame {
        let transform = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 8, ty: 0)
        let metadata = try VideoFrameMetadata(
            sourceID: "preview", sourceFrameIndex: 0,
            presentationTime: CMTime(value: 7, timescale: 30),
            width: 16, height: 8, preferredTransform: transform)
        let context = CGContext(
            data: nil, width: 8, height: 16, bitsPerComponent: 8, bytesPerRow: 32,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        return UprightVideoFrame(
            metadata: metadata, image: context.makeImage()!, bufferToImage: transform,
            readyMilliseconds: 100)
    }

    /// Produces independent copies with deliberate identity/geometry/timing errors.
    private static func result(
        _ frame: UprightVideoFrame, metadata: VideoFrameMetadata? = nil,
        width: Int? = nil, transform: CGAffineTransform? = nil,
        ready: Double = 100, failed: Bool = false
    ) throws -> VideoFrameResult {
        VideoFrameResult(
            metadata: metadata ?? frame.metadata,
            sourceWidth: width ?? frame.image.width,
            sourceHeight: frame.image.height,
            bufferToImage: transform ?? frame.bufferToImage,
            status: failed ? .failed : .completed,
            detections: failed ? [] : [try Detection(
                boundingBox: CGRect(x: 1, y: 2, width: 4, height: 5),
                confidence: 0.9, classID: 1)],
            failure: failed
                ? VideoInferenceFailure(DetectorError.noModelSelected) : nil,
            timings: try VideoFrameTimings(
                readyMilliseconds: ready,
                modelTiming: ModelPredictionTiming(
                    submittedMilliseconds: 110, completedMilliseconds: 120),
                doneMilliseconds: 130))
    }
}
