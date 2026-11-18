import CoreGraphics

/// Checks the shared detection contract without loading a model or signing an app.
@main
struct DetectionContractSmokeTests {
    /// Runs geometry, confidence, category and actor-boundary checks.
    static func main() async throws {
        try testSourceCoordinates()
        try testInvalidGeometry()
        try testConfidenceBounds()
        try testCategoryIDs()
        try await testActorBoundary()
        print("Detection contract: all 5 checks passed.")
    }

    /// Keeps fractional, out-of-image boxes unchanged for Python parity checks.
    private static func testSourceCoordinates() throws {
        let box = CGRect(x: -3.25, y: 10.5, width: 700.75, height: 40.125)
        let detection = try Detection(boundingBox: box, confidence: 0.001, classID: 1)
        precondition(detection.boundingBox == box)
        precondition(detection.confidence == 0.001)
    }

    /// Rejects invalid dimensions, nonfinite coordinates and overflowing corners.
    private static func testInvalidGeometry() throws {
        let invalidBoxes = [
            CGRect(x: 0, y: 0, width: 0, height: 1),
            CGRect(x: 0, y: 0, width: 1, height: -1),
            CGRect(x: 0, y: 0, width: -1, height: 1),
            CGRect(x: 0, y: 0, width: 1, height: 0),
            CGRect(x: CGFloat.nan, y: 0, width: 1, height: 1),
            CGRect(x: 0, y: CGFloat.infinity, width: 1, height: 1),
            CGRect(x: 0, y: 0, width: CGFloat.infinity, height: 1),
            CGRect(x: 0, y: 0, width: 1, height: CGFloat.nan),
            CGRect(
                x: CGFloat.greatestFiniteMagnitude, y: 0,
                width: CGFloat.greatestFiniteMagnitude, height: 1
            ),
        ]
        for box in invalidBoxes {
            try expectFailure(.invalidBoundingBox, box: box)
        }
    }

    /// Accepts inclusive confidence endpoints and rejects out-of-range scores.
    private static func testConfidenceBounds() throws {
        for score in [0.0, 1.0] {
            _ = try Detection(boundingBox: validBox, confidence: score, classID: 1)
        }
        for score in [-0.001, 1.001, Double.nan, Double.infinity] {
            try expectFailure(.invalidConfidence, confidence: score)
        }
    }

    /// Preserves source category IDs without applying a second class-index shift.
    private static func testCategoryIDs() throws {
        for category in [0, 1, 42] {
            let detection = try Detection(
                boundingBox: validBox, confidence: 0.5, classID: category
            )
            precondition(detection.classID == category)
        }
        try expectFailure(.invalidClassID, classID: -1)
    }

    /// Passes a decoded image and immutable results across a detector actor.
    @MainActor
    private static func testActorBoundary() async throws {
        let context = CGContext(
            data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 8,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        let image = context.makeImage()!
        let expected = try Detection(
            boundingBox: validBox, confidence: 0.9, classID: 1
        )
        let detector: any ObjectDetector = FixtureDetector(results: [expected])
        let results = try await detector.detect(in: image)
        precondition(results == [expected])
        let emptyDetector: any ObjectDetector = FixtureDetector(results: [])
        let emptyResults = try await emptyDetector.detect(in: image)
        precondition(emptyResults.isEmpty)
    }

    /// Supplies valid geometry when checking unrelated contract fields.
    private static var validBox: CGRect {
        CGRect(x: 1, y: 2, width: 3, height: 4)
    }

    /// Requires the expected validation failure rather than a silent correction.
    private static func expectFailure(
        _ expected: Detection.ValidationError,
        box: CGRect = CGRect(x: 1, y: 2, width: 3, height: 4),
        confidence: Double = 0.5,
        classID: Int = 1
    ) throws {
        do {
            _ = try Detection(boundingBox: box, confidence: confidence, classID: classID)
            preconditionFailure("Invalid detection was accepted: \(expected)")
        } catch let error as Detection.ValidationError {
            precondition(error == expected)
        }
    }
}

/// Supplies fixed outputs to exercise the protocol without model dependencies.
private actor FixtureDetector: ObjectDetector {
    private let results: [Detection]

    /// Stores the response for a positive-image or empty-image scenario.
    init(results: [Detection]) {
        self.results = results
    }

    /// Returns immutable detections from a worker actor.
    func detect(in image: CGImage) async throws -> [Detection] {
        precondition(image.width == 2 && image.height == 2)
        return results
    }
}
