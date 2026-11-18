import Foundation

/// Checks grid strides, score products, stable NMS and inverse letterboxing.
@main
struct YOLOXDecoderSmokeTests {
    /// Runs overlapping, multi-class, empty and malformed-grid scenarios.
    static func main() throws {
        var values = [Float](repeating: 0, count: 8400 * 6)
        setCandidate(&values, slot: 0, x: 2, y: 2)
        setCandidate(&values, slot: 1, x: 1, y: 2)
        setCandidate(&values, slot: 6400, x: 2, y: 2)
        setCandidate(&values, slot: 8000, x: 2, y: 2)
        let results = try YOLOXDecoder.decode(
            FloatTensor(shape: [1, 8400, 6], values: values), scale: 0.5
        )
        precondition(
            results.count == 3, "Overlapping same-class boxes must be suppressed")
        precondition(results.map { $0.boundingBox.origin.x } == [24, 48, 96])
        precondition(results.allSatisfy { abs($0.confidence - 0.4) < 0.000001 })
        precondition(results.allSatisfy { $0.classID == 1 })

        var multi = [Float](repeating: 0, count: 8400 * 7)
        multi.replaceSubrange(0..<7, with: [2, 2, 0, 0, 1, 0.8, 0.1])
        multi.replaceSubrange(7..<14, with: [1, 2, 0, 0, 1, 0.1, 0.8])
        let separateClasses = try YOLOXDecoder.decode(
            FloatTensor(shape: [1, 8400, 7], values: multi),
            scale: 1, categoryIDs: [1, 42]
        )
        precondition(
            separateClasses.map(\.classID) == [1, 42], "NMS crossed categories")
        let empty = try YOLOXDecoder.decode(
            FloatTensor(
                shape: [1, 8400, 6], values: Array(repeating: 0, count: 8400 * 6)),
            scale: 1
        )
        precondition(empty.isEmpty)
        do {
            _ = try YOLOXDecoder.decode(FloatTensor(shape: [1], values: [0]), scale: 1)
            preconditionFailure("Wrong grid was accepted")
        } catch DetectorError.invalidTensor {}
        print(
            "YOLOX decoder: strides, score products, stable/class-aware NMS, scale and empty checks passed."
        )
    }

    /// Writes a raw candidate with score 0.8 times 0.5 and an eight-pixel base size.
    private static func setCandidate(
        _ values: inout [Float], slot: Int, x: Float, y: Float
    ) {
        values.replaceSubrange(slot * 6..<(slot * 6 + 6), with: [x, y, 0, 0, 0.8, 0.5])
    }
}
