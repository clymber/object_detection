import CoreGraphics
import CoreMedia
import Foundation

/// Checks per-frame identity, source boxes, empty success and explicit failures.
@main
struct VideoFrameResultSmokeTests {
    /// Verifies complete records and prevents cancellation from becoming a false empty.
    static func main() async throws {
        let transform = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 8, ty: 0)
        let metadata = try VideoFrameMetadata(
            sourceID: "result-fixture", sourceFrameIndex: 37,
            presentationTime: CMTime(value: -1001, timescale: 30000),
            width: 16, height: 8, preferredTransform: transform, loopIndex: 2)
        let image = CGContext(
            data: nil, width: 8, height: 16, bitsPerComponent: 8, bytesPerRow: 32,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!.makeImage()!
        let frame = UprightVideoFrame(
            metadata: metadata, image: image, bufferToImage: transform)
        let detection = try Detection(
            boundingBox: CGRect(x: -2.5, y: 3.25, width: 7.125, height: 20),
            confidence: 0.8734567, classID: 1)
        let positive = try await VideoFrameDetector(
            detector: ResultFixture(detections: [detection])).process(frame)
        precondition(positive.metadata == metadata)
        precondition(positive.metadata.presentationTime.value == -1001)
        precondition(positive.metadata.presentationTime.timescale == 30000)
        precondition(positive.sourceWidth == 8 && positive.sourceHeight == 16)
        precondition(positive.bufferToImage == transform)
        precondition(positive.status == .completed && positive.failure == nil)
        precondition(positive.detections == [detection])
        precondition(positive.processingDurationMilliseconds >= 0)
        let empty = try await VideoFrameDetector(
            detector: ResultFixture(detections: [])).process(frame)
        precondition(empty.status == .completed && empty.detections.isEmpty)
        precondition(empty.failure == nil)
        let failed = try await VideoFrameDetector(
            detector: ResultFixture(detections: nil)).process(frame)
        precondition(failed.status == .failed && failed.detections.isEmpty)
        precondition(failed.failure?.type.contains("ResultFixtureError") == true)
        precondition(failed.failure?.message == "fixture inference failure")
        precondition(failed.metadata == metadata)
        precondition(failed.processingDurationMilliseconds >= 0)
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await VideoFrameDetector(
                detector: ResultFixture(detections: [])).process(frame)
        }
        do {
            _ = try await cancelled.value
            preconditionFailure("Cancelled inference produced a completed record")
        } catch is CancellationError {}
        print("Video results: identity, geometry, timing, empty/failure/cancel passed.")
    }
}

/// Supplies independent successful, empty and failing inference outcomes.
private actor ResultFixture: ObjectDetector {
    let detections: [Detection]?

    /// Uses nil to request an explicit inference failure.
    init(detections: [Detection]?) { self.detections = detections }

    /// Exercises the same shared detector interface as the real Core ML adapters.
    func detect(in image: CGImage) throws -> [Detection] {
        guard let detections else { throw ResultFixtureError.inference }
        return detections
    }
}

/// Provides a stable diagnostic for failure-record assertions.
nonisolated private enum ResultFixtureError: LocalizedError {
    case inference
    var errorDescription: String? { "fixture inference failure" }
}
