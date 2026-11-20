import CoreGraphics
import CoreMedia
import Foundation

/// Verifies protocol filtering and engine reuse without model-specific video paths.
@main
struct VideoFrameDetectionSmokeTests {
    /// Checks threshold boundaries, stable ordering, caps, source geometry and errors.
    static func main() async throws {
        let frame = try makeFrame()
        let box = CGRect(x: -1.25, y: 2.5, width: 17.75, height: 21.125)
        let scores = [Double(0.25).nextDown, 0.25, 0.9, 0.9, Double(0.25).nextUp]
        let candidates = try scores.enumerated().map {
            try Detection(boundingBox: box, confidence: $0.element, classID: $0.offset)
        }
        let fixture = VideoDetectorFixture(detections: candidates)
        let detector = VideoFrameDetector(detector: fixture)
        let results = try await detector.detect(frame)
        precondition(results.map(\.classID) == [2, 3, 4, 1])
        precondition(results.allSatisfy { $0.boundingBox == box })
        let received = await fixture.receivedSize
        precondition(received == CGSize(width: 8, height: 16))
        let tied = try (0..<150).map {
            try Detection(boundingBox: box, confidence: 0.5, classID: $0)
        }
        let capped = try await VideoFrameDetector(
            detector: VideoDetectorFixture(detections: tied)).detect(frame)
        precondition(capped.map(\.classID) == Array(0..<100))
        let empty = try await VideoFrameDetector(
            detector: VideoDetectorFixture(detections: [])).detect(frame)
        precondition(empty.isEmpty)
        let engine = DetectorEngine(loader: { _ in fixture })
        for model in DetectorModel.allCases {
            try await engine.selectModel(model)
            let actual = try await VideoFrameDetector(detector: engine).detect(frame)
            precondition(actual == results)
        }
        do {
            _ = try await VideoFrameDetector(detector: FailingVideoDetector())
                .detect(frame)
            preconditionFailure("Inference failure was replaced with empty detections")
        } catch DetectorError.invalidTensor {}
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await detector.detect(frame)
        }
        do {
            _ = try await cancelled.value
            preconditionFailure("Cancelled detection succeeded")
        } catch is CancellationError {}
        print("Video detection: thresholds, stable cap and engine reuse/errors passed.")
    }

    /// Supplies upright portrait geometry with independent encoded source dimensions.
    private static func makeFrame() throws -> UprightVideoFrame {
        let image = CGContext(
            data: nil, width: 8, height: 16, bitsPerComponent: 8, bytesPerRow: 32,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!.makeImage()!
        return UprightVideoFrame(
            metadata: try VideoFrameMetadata(
                sourceID: "detector-fixture", sourceFrameIndex: 17,
                presentationTime: CMTime(value: 1001, timescale: 30000),
                width: 16, height: 8),
            image: image, bufferToImage: .identity)
    }
}

/// Records the actual image dimensions received by the shared adapter interface.
private actor VideoDetectorFixture: ObjectDetector {
    let detections: [Detection]
    private(set) var receivedSize: CGSize?

    /// Supplies ordered candidates including threshold and ranking boundary cases.
    init(detections: [Detection]) { self.detections = detections }

    /// Returns source-coordinate detections from a worker actor.
    func detect(in image: CGImage) throws -> [Detection] {
        receivedSize = CGSize(width: image.width, height: image.height)
        return detections
    }
}

/// Exercises an actual thrown inference failure through the video detector.
private actor FailingVideoDetector: ObjectDetector {
    /// Rejects the request with an explicit backend/tensor error.
    func detect(in image: CGImage) throws -> [Detection] {
        throw DetectorError.invalidTensor("fixture backend failure")
    }
}
