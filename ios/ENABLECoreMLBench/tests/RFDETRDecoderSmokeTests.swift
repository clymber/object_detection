import Foundation

/// Exercises RF-DETR background selection, stable ties, score floor and geometry.
@main
struct RFDETRDecoderSmokeTests {
    /// Runs semantic decoder checks for both exported output-name contracts.
    static func main() throws {
        for model in [DetectorModel.rfdetrSmall, .rfdetrNano] {
            var logits = [Float](repeating: -20, count: 600)
            logits[0] = 0
            logits[2] = 0
            let outputs = try fixture(logits, model: model)
            let detections = try RFDETRDecoder.decode(
                outputs, model: model, sourceWidth: 1000, sourceHeight: 500
            )
            precondition(detections.count == 2, "RF-DETR must not apply NMS")
            precondition(
                detections.allSatisfy { $0.confidence == 0.5 && $0.classID == 1 })
            precondition(abs(detections[0].boundingBox.origin.x + 100) < 0.001)
            precondition(abs(detections[0].boundingBox.size.width - 200) < 0.001)
            precondition(
                detections[1].boundingBox.origin.x > 0, "Ties lost candidate order")
            for index in 0..<300 { logits[index * 2 + 1] = 10 }
            let background = try RFDETRDecoder.decode(
                fixture(logits, model: model), model: model,
                sourceWidth: 1000, sourceHeight: 500
            )
            precondition(
                background.isEmpty, "Background must participate in top-300 ranking")
            logits[0] = .nan
            do {
                _ = try RFDETRDecoder.decode(
                    fixture(logits, model: model), model: model,
                    sourceWidth: 1000, sourceHeight: 500
                )
                preconditionFailure("Nonfinite logits were accepted")
            } catch DetectorError.invalidTensor {}
        }
        print(
            "Both RF-DETR decoders: background, ties, no-NMS, geometry and invalid output checks passed."
        )
    }

    /// Builds query boxes with two distinct leading candidates and raw logits.
    private static func fixture(
        _ logits: [Float], model: DetectorModel
    ) throws -> [String: FloatTensor] {
        var boxes = [Float](repeating: 0.2, count: 1200)
        boxes[0] = 0
        boxes[4] = 0.5
        return [
            "concat_2": try FloatTensor(shape: [1, 300, 4], values: boxes),
            model.logitsName: try FloatTensor(shape: [1, 300, 2], values: logits),
        ]
    }
}
