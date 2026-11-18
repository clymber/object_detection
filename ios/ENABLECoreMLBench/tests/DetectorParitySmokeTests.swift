import CoreGraphics
import Foundation

/// Runs all three actual exports through the shared Swift engine on the Mac.
@main
struct DetectorParitySmokeTests {
    /// Compares each development image with its saved checkpoint reference.
    static func main() async throws {
        let directory = URL(fileURLWithPath: CommandLine.arguments[1])
        let document =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("parity.json"))
            ) as! [String: Any]
        let selections = document["models"] as! [[String: Any]]
        var urls: [DetectorModel: URL] = [:]
        for entry in selections {
            urls[DetectorModel(rawValue: entry["model"] as! String)!] = URL(
                fileURLWithPath: entry["package"] as! String
            )
        }
        let engine = DetectorEngine(modelURLs: urls, computeUnits: .cpuOnly)
        for entry in selections {
            let model = DetectorModel(rawValue: entry["model"] as! String)!
            try await engine.selectModel(model)
            let selected = await engine.selectedModel
            precondition(selected == model)
            let cases = entry["cases"] as! [[String: Any]]
            for (index, record) in cases.enumerated() {
                let width = record["width"] as! Int
                let height = record["height"] as! Int
                let bytes = try Data(
                    contentsOf: directory.appendingPathComponent(
                        record["rgb"] as! String))
                let image = CGImage(
                    width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                    bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                    provider: CGDataProvider(data: bytes as CFData)!, decode: nil,
                    shouldInterpolate: false, intent: .defaultIntent
                )!
                let actual = try await engine.detect(in: image)
                let expected = try (record["expected"] as! [[String: Any]]).map { row in
                    let box = row["bbox"] as! [Double]
                    return try Detection(
                        boundingBox: CGRect(
                            x: box[0], y: box[1], width: box[2], height: box[3]),
                        confidence: row["score"] as! Double,
                        classID: row["category_id"] as! Int
                    )
                }
                try compare(
                    actual, expected, width: width, height: height,
                    label: "\(model.rawValue)/\(record["id"]!)")
                if (index + 1) % 5 == 0 {
                    FileHandle.standardOutput.write(
                        Data(
                            "\(model.rawValue): \(index + 1)/\(cases.count) parity images passed\n"
                                .utf8
                        ))
                }
            }
        }
        print(
            "All three models: checkpoint parity passed through the Swift switching engine."
        )
    }

    /// Applies the inclusive confidence threshold and stable top-100 cap.
    private static func retained(_ detections: [Detection]) -> [Detection] {
        Array(
            detections.enumerated().filter { $0.element.confidence >= 0.25 }.sorted {
                $0.element.confidence == $1.element.confidence
                    ? $0.offset < $1.offset
                    : $0.element.confidence > $1.element.confidence
            }.prefix(100).map(\.element))
    }

    /// Enforces the unchanged count, category, corner and score acceptance rules.
    private static func compare(
        _ actual: [Detection], _ expected: [Detection],
        width: Int, height: Int, label: String
    ) throws {
        var available = retained(actual)
        let reference = retained(expected)
        guard available.count == reference.count else {
            throw DetectorError.invalidTensor(
                "\(label): count \(available.count) != \(reference.count)")
        }
        for detection in reference {
            let candidates = available.indices.filter {
                available[$0].classID == detection.classID
            }
            guard
                let selected = candidates.max(by: {
                    iou(detection.boundingBox, available[$0].boundingBox)
                        < iou(detection.boundingBox, available[$1].boundingBox)
                })
            else { throw DetectorError.invalidTensor("\(label): category mismatch") }
            let other = available.remove(at: selected)
            let first = detection.boundingBox
            let second = other.boundingBox
            let errors = [
                abs(first.minX - second.minX), abs(first.minY - second.minY),
                abs(first.maxX - second.maxX), abs(first.maxY - second.maxY),
            ]
            let limits = [
                max(2, Double(width) * 0.002), max(2, Double(height) * 0.002),
            ]
            guard
                errors.enumerated().allSatisfy({ $0.element <= limits[$0.offset % 2] }),
                abs(detection.confidence - other.confidence) <= 0.02
            else {
                throw DetectorError.invalidTensor(
                    "\(label): box/score tolerance exceeded: \(errors)")
            }
        }
    }

    /// Calculates source-pixel IoU for score-ordered same-class matching.
    private static func iou(_ first: CGRect, _ second: CGRect) -> Double {
        let intersection = first.intersection(second)
        let overlap = intersection.isNull ? 0 : intersection.width * intersection.height
        return overlap
            / max(
                first.width * first.height + second.width * second.height - overlap,
                1e-12)
    }
}
