import CoreGraphics
import Foundation

/// Decodes raw YOLOX grids and applies stable, class-aware CPU NMS.
nonisolated enum YOLOXDecoder {
    /// Keeps model-pixel corners until NMS is complete, matching Python arithmetic.
    private struct Candidate {
        let slot: Int
        let corners: [Float]
        let score: Float
        let classID: Int
    }

    /// Restores raw-grid outputs to unclipped source boxes after NMS at IoU 0.65.
    static func decode(
        _ output: FloatTensor, scale: Float, categoryIDs: [Int] = [1]
    ) throws -> [Detection] {
        let columns = 5 + categoryIDs.count
        guard !categoryIDs.isEmpty, categoryIDs.allSatisfy({ $0 >= 0 }),
            output.shape == [1, 8400, columns], scale.isFinite, scale > 0,
            output.values.allSatisfy(\.isFinite)
        else {
            throw DetectorError.invalidTensor("YOLOX grid, categories or source scale")
        }
        var candidates: [Candidate] = []
        var slot = 0
        for stride in [8, 16, 32] {
            let side = 640 / stride
            for y in 0..<side {
                for x in 0..<side {
                    let base = slot * columns
                    var label = 0
                    var score = output.values[base + 4] * output.values[base + 5]
                    for category in 1..<categoryIDs.count {
                        let other =
                            output.values[base + 4] * output.values[base + 5 + category]
                        if other > score {
                            label = category
                            score = other
                        }
                    }
                    if score >= 0.001 {
                        let cx = (output.values[base] + Float(x)) * Float(stride)
                        let cy = (output.values[base + 1] + Float(y)) * Float(stride)
                        let width = exp(output.values[base + 2]) * Float(stride)
                        let height = exp(output.values[base + 3]) * Float(stride)
                        let left = cx - width / 2
                        let top = cy - height / 2
                        let box = [left, top, left + width, top + height]
                        guard box.allSatisfy(\.isFinite) else {
                            throw DetectorError.invalidTensor(
                                "nonfinite YOLOX decoded box")
                        }
                        candidates.append(
                            Candidate(
                                slot: slot, corners: box, score: score,
                                classID: categoryIDs[label]
                            ))
                    }
                    slot += 1
                }
            }
        }
        candidates.sort {
            $0.score == $1.score ? $0.slot < $1.slot : $0.score > $1.score
        }
        var kept: [Candidate] = []
        for candidate in candidates {
            if !kept.contains(where: {
                $0.classID == candidate.classID
                    && overlap($0.corners, candidate.corners) > 0.65
            }) {
                kept.append(candidate)
            }
        }
        return try kept.compactMap { candidate in
            let box = candidate.corners.map { $0 / scale }
            guard box.allSatisfy(\.isFinite) else {
                throw DetectorError.invalidTensor("nonfinite YOLOX source box")
            }
            guard box[2] > box[0], box[3] > box[1] else { return nil }
            return try Detection(
                boundingBox: CGRect(
                    x: Double(box[0]), y: Double(box[1]),
                    width: Double(box[2]) - Double(box[0]),
                    height: Double(box[3]) - Double(box[1])
                ), confidence: Double(candidate.score), classID: candidate.classID
            )
        }
    }

    /// Calculates Float32 IoU in model pixels before inverse letterboxing.
    private static func overlap(_ first: [Float], _ second: [Float]) -> Float {
        let width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
        let height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
        let intersection = width * height
        let areaA = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        let areaB = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        let union = areaA + areaB - intersection
        return union > 0 ? intersection / union : 0
    }
}
