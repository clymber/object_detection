import CoreGraphics
import CoreML
import Foundation
import ImageIO
import SwiftUI

/// Runs the UI session with actual bundled JPEGs and the selected Small package.
@main
@MainActor
struct StillImageIntegrationSmokeTests {
    /// Checks every bundled sample against its saved checkpoint predictions.
    static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let appURL = URL(fileURLWithPath: CommandLine.arguments[2])
        let bundle = Bundle(url: appURL)!
        let manifest =
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf:
                        bundle.url(
                            forResource: "sample_manifest", withExtension: "json")!
                )) as! [String: Any]
        let references = manifest["samples"] as! [[String: Any]]
        precondition(
            manifest["split"] as! String == "test"
                && references.count == BundledSample.allCases.count)
        precondition(references.allSatisfy { $0["split"] as! String == "test" })
        precondition(
            Set(references.map { $0["file_name"] as! String })
                == Set(BundledSample.allCases.map { $0.rawValue + ".jpg" }))
        let engine = DetectorEngine(
            modelURLs: [
                .rfdetrSmall: root.appendingPathComponent(
                    "ios/ENABLECoreMLBench/ENABLECoreMLBench/Models/rfdetr_small.mlpackage"
                )
            ], computeUnits: .cpuOnly)
        let worker = StillImageRunner(engine: engine, bundle: bundle)
        let session = StillImageSession(worker: worker)
        let output = root.appendingPathComponent("outputs/ios_builds/still_image")
        try FileManager.default.createDirectory(
            at: output, withIntermediateDirectories: true)
        for sample in BundledSample.allCases {
            session.load(sample)
            try await waitForIdle(session)
            precondition(session.status == .idle && session.image?.width == 640)
            session.start()
            try await waitForIdle(session)
            guard session.status == .completed else {
                throw DetectorError.invalidImage(session.message)
            }
            precondition(session.processedImages == 1)
            precondition(session.latencyMilliseconds! > 0)
            let reference = references.first {
                $0["file_name"] as! String == sample.rawValue + ".jpg"
            }!
            let expected =
                reference["expected_checkpoint_detections"] as! [[String: Any]]
            precondition(
                session.detections.count == expected.count,
                "Detection count differs for \(sample.rawValue)")
            var unmatched = session.detections
            for row in expected {
                let box = row["bbox"] as! [Double]
                let referenceBox = CGRect(
                    x: box[0], y: box[1], width: box[2], height: box[3])
                let candidates = unmatched.indices.filter {
                    unmatched[$0].classID == row["category_id"] as! Int
                }.sorted {
                    let first = iou(unmatched[$0].boundingBox, referenceBox)
                    let second = iou(unmatched[$1].boundingBox, referenceBox)
                    return first == second ? $0 < $1 : first > second
                }
                precondition(
                    !candidates.isEmpty, "Category differs for \(sample.rawValue)")
                let detection = unmatched.remove(at: candidates.first!)
                let corners = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                let actual = [
                    detection.boundingBox.minX, detection.boundingBox.minY,
                    detection.boundingBox.maxX, detection.boundingBox.maxY,
                ]
                precondition(
                    zip(actual, corners).allSatisfy { abs($0 - $1) <= 2 },
                    "Source box differs for \(sample.rawValue)")
                precondition(detection.classID == row["category_id"] as! Int)
                precondition(
                    abs(detection.confidence - (row["score"] as! Double)) <= 0.02)
            }
            let preview = FramePreview(
                image: session.image, detections: session.detections,
                imageTitle: sample.title
            ).frame(width: 360).background(.white)
                .environment(\.colorScheme, .light)
            let renderer = ImageRenderer(content: preview)
            renderer.scale = 2
            guard let rendered = renderer.cgImage else {
                throw DetectorError.invalidImage("Preview renderer returned no image")
            }
            if !session.detections.isEmpty {
                precondition(
                    rendered.height > 560,
                    "Detection details must sit below the full-height image")
            }
            let destination = CGImageDestinationCreateWithURL(
                output.appendingPathComponent("\(sample.rawValue)_preview.png")
                    as CFURL,
                "public.png" as CFString, 1, nil
            )!
            CGImageDestinationAddImage(destination, rendered, nil)
            precondition(CGImageDestinationFinalize(destination))
            print(
                "\(sample.title): \(session.detections.count) detections, measured latency and preview saved."
            )
        }
        let batch = StillImageBatchSession(
            runner: StillImageBatchRunner(
                worker: worker, bundle: bundle,
                runtime: .current(computeUnits: "cpuOnly"),
                exportDirectory: output.appendingPathComponent("batch_exports")))
        batch.start()
        let deadline = ContinuousClock.now.advanced(by: .seconds(90))
        while batch.isBusy && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        precondition(
            !batch.isBusy && batch.completed == references.count, "Real batch timed out"
        )
        let exportURL = batch.exportURL!
        let report = try JSONDecoder().decode(
            StillImageBatchReport.self, from: Data(contentsOf: exportURL))
        precondition(
            report.status == "completed" && report.images.count == references.count)
        let expectedEmptyCount = references.filter {
            ($0["expected_checkpoint_detections"] as! [[String: Any]]).isEmpty
        }.count
        precondition(
            report.images.filter { $0.detections.isEmpty }.count == expectedEmptyCount)
        precondition(report.runtime.platform == "macos")
        try exportURL.path.write(
            to: output.appendingPathComponent("batch_export_path.txt"),
            atomically: true, encoding: .utf8)
        print("Batch prediction export: \(exportURL.path)")
    }

    /// Matches multi-object examples without assuming an identical score order.
    private static func iou(_ first: CGRect, _ second: CGRect) -> Double {
        let intersection = first.intersection(second)
        let overlap = intersection.isNull ? 0 : intersection.width * intersection.height
        return overlap
            / (first.width * first.height + second.width * second.height - overlap)
    }

    /// Allows bounded model compilation while keeping host checks deterministic.
    private static func waitForIdle(_ session: StillImageSession) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(45))
        while session.isBusy && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        precondition(!session.isBusy, "Real session timed out")
    }
}
