import CoreGraphics
import Foundation

/// Exercises coverage, precision, error exports and stale batch cancellation.
@main
@MainActor
struct StillImageBatchSmokeTests {
    /// Uses synthetic inputs to test batch behavior independently of Core ML.
    static func main() async throws {
        let directory = URL(
            fileURLWithPath: "outputs/ios_builds/still_image/batch_checks")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let bundle = try fixtureBundle(in: directory)
        let worker = BatchWorker()
        let runner = StillImageBatchRunner(
            worker: worker, bundle: bundle,
            runtime: .current(computeUnits: "test_worker"),
            exportDirectory: directory.appendingPathComponent("exports"))
        let session = StillImageBatchSession(runner: runner)
        session.start()
        session.start()
        try await waitForIdle(session)
        let firstURL = session.exportURL!
        let first = try JSONDecoder().decode(
            StillImageBatchReport.self, from: Data(contentsOf: firstURL))
        let count = BundledSample.allCases.count
        let negativeCount = BundledSample.allCases.filter { !$0.isPositive }.count
        precondition(first.status == "completed" && first.images.count == count)
        precondition(session.completed == count && session.total == count)
        precondition(
            first.images.filter { $0.detections.isEmpty }.count == negativeCount)
        precondition(first.images[0].detections[0].score == 0.9123456789012345)
        precondition(first.images[0].detections[0].bbox[0] == 1.123456789012345)
        let runCount = await worker.runCount
        precondition(runCount == count, "Duplicate Start ran the batch twice")

        await worker.failSomeImages()
        session.start()
        precondition(session.exportURL == nil)
        try await waitForIdle(session)
        let failed = try JSONDecoder().decode(
            StillImageBatchReport.self, from: Data(contentsOf: session.exportURL!))
        precondition(failed.status == "incomplete" && failed.images.count == count)
        precondition(failed.images.filter { $0.status == "error" }.count == 2)
        precondition(
            failed.images.filter { $0.status == "error" }.allSatisfy {
                $0.error != nil && $0.latency_ms == nil
            })

        await worker.holdNextRun()
        session.start()
        let heldDeadline = ContinuousClock.now.advanced(by: .seconds(3))
        while !(await worker.isHeld) && ContinuousClock.now < heldDeadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        let isHeld = await worker.isHeld
        precondition(isHeld, "Controlled inference did not start")
        session.stop()
        precondition(!session.isBusy && session.exportURL == nil)
        session.start()
        try await waitForIdle(session)
        let newestURL = session.exportURL
        precondition(newestURL != nil && newestURL != firstURL)
        await worker.release()
        try await Task.sleep(for: .milliseconds(20))
        precondition(session.exportURL == newestURL && !session.isBusy)
        print(
            "Batch: \(count)-image coverage, precision, empty results, read/inference errors and stale cancellation passed."
        )
    }

    /// Builds deterministic hashed resource bytes for a controlled image worker.
    private static func fixtureBundle(in directory: URL) throws -> Bundle {
        let url = directory.appendingPathComponent("Inputs.bundle")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        let plist = try PropertyListSerialization.data(
            fromPropertyList: ["CFBundleIdentifier": "test.batch.inputs"],
            format: .xml, options: 0)
        try plist.write(to: url.appendingPathComponent("Info.plist"))
        var samples: [[String: Any]] = []
        for sample in BundledSample.allCases {
            let bytes = Data(sample.rawValue.utf8)
            let name = sample.rawValue + ".jpg"
            try bytes.write(to: url.appendingPathComponent(name))
            samples.append([
                "image_id": Int(sample.rawValue.split(separator: "_").last!)!,
                "file_name": name, "sha256": StillImageCatalog.digest(bytes),
                "width": 640, "height": 640, "split": "test",
            ])
        }
        let manifest: [String: Any] = [
            "schema_version": 3, "scope": "test_examples",
            "model": "rfdetr_small", "run_id": "test",
            "model_sha256": "test", "checkpoint_sha256": "test", "split": "test",
            "confidence_threshold": 0.25, "max_detections": 100,
            "box_convention": "unclipped source-pixel top-left xywh",
            "samples": samples,
        ]
        try JSONSerialization.data(withJSONObject: manifest).write(
            to: url.appendingPathComponent("sample_manifest.json"))
        return Bundle(url: url)!
    }

    /// Bounds async failures instead of letting a broken batch hang the suite.
    private static func waitForIdle(_ session: StillImageBatchSession) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while session.isBusy && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        precondition(!session.isBusy, "Batch timed out")
    }
}

/// Supplies detections independently of the catalog's unused expected predictions.
private actor BatchWorker: StillImageProcessing {
    private var sample = BundledSample.testPositive6988
    private var failing = false
    private var holdNext = false
    private var pending: CheckedContinuation<Void, Never>?
    private(set) var runCount = 0
    var isHeld: Bool { pending != nil }

    /// Can fail a read while retaining the remaining catalog coverage.
    func load(_ sample: BundledSample) throws -> CGImage {
        self.sample = sample
        if failing && sample == .testPositive6990 {
            throw DetectorError.invalidImage("Controlled image read failure")
        }
        return CGContext(
            data: nil, width: 640, height: 640, bitsPerComponent: 8, bytesPerRow: 2560,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!.makeImage()!
    }

    /// Can suspend an older inference until a newer session has finished.
    func run(_ image: CGImage, model: DetectorModel) async throws -> StillImageResult {
        runCount += 1
        let input = sample
        if holdNext {
            holdNext = false
            await withCheckedContinuation { pending = $0 }
        }
        if failing && input == .testPositive6991 {
            throw DetectorError.invalidModel("Controlled inference failure")
        }
        return StillImageResult(
            detections: input.isPositive
                ? [
                    try Detection(
                        boundingBox: CGRect(
                            x: 1.123456789012345, y: 2, width: 3, height: 4),
                        confidence: 0.9123456789012345, classID: 1)
                ] : [],
            latencyMilliseconds: 12.3456789)
    }

    /// Enables separate read and inference errors for a subsequent run.
    func failSomeImages() { failing = true }

    /// Holds only the next inference, allowing a newer batch to complete.
    func holdNextRun() { holdNext = true }

    /// Completes an old inference after its originating session was cancelled.
    func release() {
        pending?.resume()
        pending = nil
    }
}
