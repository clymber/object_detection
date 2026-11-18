import CoreGraphics
import CoreML
import Foundation
import ImageIO
import SwiftUI

/// Runs the UI session with actual bundled JPEGs and the selected Small package.
@main
@MainActor
struct StillImageIntegrationSmokeTests {
    /// Accepts the workspace/app bundle and verifies both real inference outcomes.
    static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let appURL = URL(fileURLWithPath: CommandLine.arguments[2])
        let bundle = Bundle(url: appURL)!
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
            precondition(session.detections.count == (sample == .positive ? 1 : 0))
            if let detection = session.detections.first {
                precondition(detection.classID == 1 && detection.confidence > 0.95)
                precondition(abs(detection.boundingBox.minX - 48.1788) <= 2)
                precondition(abs(detection.boundingBox.minY - 103.8894) <= 2)
            }
            let preview = FramePreview(
                image: session.image, detections: session.detections,
                imageTitle: sample.title
            ).frame(width: 360, height: 280).environment(\.colorScheme, .light)
            let renderer = ImageRenderer(content: preview)
            renderer.scale = 2
            guard let rendered = renderer.cgImage else {
                throw DetectorError.invalidImage("Preview renderer returned no image")
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
