import CoreGraphics
import CoreML
import Darwin
import Foundation
import ImageIO
import SwiftUI

/// Runs every selectable model through the UI session and batch exporter.
@main
@MainActor
struct StillImageIntegrationSmokeTests {
    /// Uses the built app's inputs/manifests and the original local model packages.
    static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let bundle = Bundle(url: URL(fileURLWithPath: CommandLine.arguments[2]))!
        let urls = Dictionary(
            uniqueKeysWithValues: DetectorModel.allCases.map {
                (
                    $0,
                    root.appendingPathComponent(
                        "ios/ENABLECoreMLBench/ENABLECoreMLBench/Models/\($0.rawValue).mlpackage"
                    )
                )
            })
        let engine = DetectorEngine(modelURLs: urls, computeUnits: .cpuOnly)
        let worker = StillImageRunner(engine: engine, bundle: bundle)
        let output = root.appendingPathComponent("outputs/ios_builds/still_image")
        try FileManager.default.createDirectory(
            at: output, withIntermediateDirectories: true)
        var exports: [String: String] = [:]
        var parityFailures: [String] = []
        let models = DetectorModel.allCases.filter {
            CommandLine.arguments.count < 4 || $0.rawValue == CommandLine.arguments[3]
        }
        for model in models {
            _ = try model.bundledURL(in: bundle)
            let result = try await checkModel(
                model, worker: worker, bundle: bundle, output: output)
            exports[model.rawValue] = result.url.path
            parityFailures.append(contentsOf: result.failures)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(exports).write(
            to: output.appendingPathComponent("batch_export_paths.json"))
        if !parityFailures.isEmpty {
            FileHandle.standardError.write(
                Data(
                    ("Reference parity failed:\n"
                        + parityFailures.joined(separator: "\n") + "\n").utf8))
            exit(1)
        }
    }

    /// Checks one model's source geometry, real UI results and exported provenance.
    private static func checkModel(
        _ model: DetectorModel, worker: StillImageRunner, bundle: Bundle, output: URL
    ) async throws -> (url: URL, failures: [String]) {
        let data = try Data(
            contentsOf: bundle.url(
                forResource: model.sampleManifestName, withExtension: "json")!)
        let manifest = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let references = manifest["samples"] as! [[String: Any]]
        precondition(manifest["model"] as! String == model.rawValue)
        precondition(manifest["split"] as! String == "test")
        precondition(references.count == BundledSample.allCases.count)
        precondition(references.allSatisfy { $0["split"] as! String == "test" })
        precondition(
            Set(references.map { $0["file_name"] as! String })
                == Set(BundledSample.allCases.map { $0.rawValue + ".jpg" }))
        let batch = StillImageBatchSession(
            runner: StillImageBatchRunner(
                worker: worker, bundle: bundle,
                runtime: .current(computeUnits: "cpuOnly"),
                exportDirectory: output.appendingPathComponent("batch_exports")))
        batch.start(model: model)
        let deadline = ContinuousClock.now.advanced(by: .seconds(90))
        while batch.isBusy && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        precondition(
            !batch.isBusy && batch.completed == references.count,
            "Real batch timed out for \(model.rawValue)")
        let exportURL = batch.exportURL!
        let report = try JSONDecoder().decode(
            StillImageBatchReport.self, from: Data(contentsOf: exportURL))
        precondition(
            report.status == "completed" && report.images.count == references.count)
        precondition(report.model == model.rawValue)
        precondition(report.model_sha256 == manifest["model_sha256"] as! String)
        precondition(report.sample_manifest_sha256 == StillImageCatalog.digest(data))
        let expectedEmptyCount = references.filter {
            ($0["expected_checkpoint_detections"] as! [[String: Any]]).isEmpty
        }.count
        precondition(
            report.images.filter { $0.detections.isEmpty }.count == expectedEmptyCount)
        precondition(report.runtime.platform == "macos")
        print("\(model.displayName) batch export: \(exportURL.path)")
        fflush(nil)
        var failures: [String] = []
        let session = StillImageSession(worker: worker)
        for sample in BundledSample.allCases {
            session.load(sample)
            try await waitForIdle(session)
            precondition(session.status == .idle && session.image?.width == 640)
            session.start(model: model)
            try await waitForIdle(session)
            guard session.status == .completed else {
                throw DetectorError.invalidImage(session.message)
            }
            precondition(
                session.processedImages == 1 && session.latencyMilliseconds! > 0)
            let reference = references.first {
                $0["file_name"] as! String == sample.rawValue + ".jpg"
            }!
            let expected =
                reference["expected_checkpoint_detections"] as! [[String: Any]]
            let label = "\(model.rawValue) / \(sample.rawValue)"
            if session.detections.count != expected.count {
                failures.append(
                    "\(label): count \(session.detections.count) != \(expected.count)")
            }
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
                guard let index = candidates.first else {
                    failures.append("\(label): unmatched reference category")
                    continue
                }
                let detection = unmatched.remove(at: index)
                let corners = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                let actual = [
                    detection.boundingBox.minX, detection.boundingBox.minY,
                    detection.boundingBox.maxX, detection.boundingBox.maxY,
                ]
                if !zip(actual, corners).allSatisfy({ abs($0 - $1) <= 2 }) {
                    failures.append("\(label): source box exceeds 2 px tolerance")
                }
                let scoreError = abs(detection.confidence - (row["score"] as! Double))
                if scoreError > 0.02 {
                    failures.append(
                        "\(label): confidence difference \(scoreError) exceeds 0.02")
                }
            }
            let preview = FramePreview(
                image: session.image, detections: session.detections,
                imageTitle: sample.title
            )
            .frame(width: 360).background(.white).environment(\.colorScheme, .light)
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
                output.appendingPathComponent(
                    "\(model.rawValue)_\(sample.rawValue)_preview.png")
                    as CFURL, "public.png" as CFString, 1, nil)!
            CGImageDestinationAddImage(destination, rendered, nil)
            precondition(CGImageDestinationFinalize(destination))
            print(
                "\(model.displayName) / \(sample.title): \(session.detections.count) detections; preview saved."
            )
            fflush(nil)
        }
        return (exportURL, failures)
    }

    /// Matches multi-object examples without assuming an identical score order.
    private static func iou(_ first: CGRect, _ second: CGRect) -> Double {
        let intersection = first.intersection(second)
        let overlap = intersection.isNull ? 0 : intersection.width * intersection.height
        return overlap
            / (first.width * first.height + second.width * second.height - overlap)
    }

    /// Bounds model compilation and image inference without blocking the UI actor.
    private static func waitForIdle(_ session: StillImageSession) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(45))
        while session.isBusy && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        precondition(!session.isBusy, "Real session timed out")
    }
}
