import CoreGraphics
import Foundation

/// Decodes both RF-DETR variants using the checkpoint reference's selection order.
nonisolated enum RFDETRDecoder {
    /// Applies sigmoid, ranks all slots, drops background and restores source boxes.
    static func decode(
        _ outputs: [String: FloatTensor], model: DetectorModel,
        sourceWidth: Int, sourceHeight: Int
    ) throws -> [Detection] {
        guard model.isRFDETR, sourceWidth > 0, sourceHeight > 0,
            let boxes = outputs["concat_2"], boxes.shape == [1, 300, 4],
            let logits = outputs[model.logitsName], logits.shape == [1, 300, 2],
            boxes.values.allSatisfy(\.isFinite), logits.values.allSatisfy(\.isFinite)
        else {
            throw DetectorError.invalidTensor(
                "RF-DETR boxes/logits or source dimensions")
        }
        let scores: [Float] = logits.values.map { value in
            1 / (1 + exp(-min(max(value, -88), 88)))
        }
        let order = scores.indices.sorted {
            scores[$0] == scores[$1] ? $0 < $1 : scores[$0] > scores[$1]
        }.prefix(300)
        var results: [Detection] = []
        for index in order where index % 2 == 0 && scores[index] >= 0.001 {
            let base = (index / 2) * 4
            let left = boxes.values[base] - boxes.values[base + 2] / 2
            let top = boxes.values[base + 1] - boxes.values[base + 3] / 2
            let x1 = left * Float(sourceWidth)
            let y1 = top * Float(sourceHeight)
            let x2 = (left + boxes.values[base + 2]) * Float(sourceWidth)
            let y2 = (top + boxes.values[base + 3]) * Float(sourceHeight)
            guard [x1, y1, x2, y2].allSatisfy(\.isFinite) else {
                throw DetectorError.invalidTensor("nonfinite RF-DETR source box")
            }
            if x2 > x1 && y2 > y1 {
                results.append(
                    try Detection(
                        boundingBox: CGRect(
                            x: Double(x1), y: Double(y1),
                            width: Double(x2) - Double(x1),
                            height: Double(y2) - Double(y1)
                        ), confidence: Double(scores[index]), classID: 1
                    ))
            }
        }
        return results
    }
}
