import CoreGraphics
import Foundation

/// Runs the fixed test catalog serially and writes precise outputs off the UI actor.
actor StillImageBatchRunner {
    typealias Progress = @Sendable (Int, Int) async -> Void
    private let worker: any StillImageProcessing
    private let bundle: Bundle
    private let runtime: StillImageRuntime
    private let exportDirectory: URL

    /// Uses the real app model by default; tests can supply an explicit worker.
    init(
        worker: any StillImageProcessing = StillImageRunner(),
        bundle: Bundle = .main,
        runtime: StillImageRuntime = .current(computeUnits: "all"),
        exportDirectory: URL = URL.cachesDirectory.appendingPathComponent(
            "ParityExports")
    ) {
        self.worker = worker
        self.bundle = bundle
        self.runtime = runtime
        self.exportDirectory = exportDirectory
    }

    /// Processes every catalog entry; individual failures make the export incomplete.
    func run(
        model: DetectorModel = .rfdetrSmall, progress: Progress
    ) async throws -> StillImageBatchReport {
        let manifestURL = ["TestImages", "Resources/TestImages", nil]
            .compactMap {
                bundle.url(
                    forResource: model.sampleManifestName, withExtension: "json",
                    subdirectory: $0)
            }.first
        guard let manifestURL else {
            throw DetectorError.invalidImage("The test sample manifest is missing.")
        }
        let data = try Data(contentsOf: manifestURL)
        let catalog = try JSONDecoder().decode(StillImageCatalog.self, from: data)
        let names = Set(BundledSample.allCases.map { $0.rawValue + ".jpg" })
        guard catalog.schema_version == 3, catalog.scope == "test_examples",
            catalog.model == model.rawValue,
            catalog.split == "test", catalog.confidence_threshold == 0.25,
            catalog.max_detections == 100, catalog.samples.count == names.count,
            Set(catalog.samples.map(\.file_name)) == names,
            Set(catalog.samples.map(\.image_id)).count == names.count,
            catalog.samples.allSatisfy({ $0.split == "test" })
        else {
            throw DetectorError.invalidImage(
                "The test catalog does not match this app.")
        }
        var rows: [StillImageBatchRow] = []
        await progress(0, catalog.samples.count)
        for sample in catalog.samples {
            try Task.checkCancellation()
            var hash = ""
            do {
                let name = String(sample.file_name.dropLast(4))
                guard let input = BundledSample(rawValue: name) else {
                    throw DetectorError.invalidImage("Unknown test image: \(name)")
                }
                hash = StillImageCatalog.digest(
                    try Data(contentsOf: input.resourceURL(in: bundle)))
                guard hash == sample.sha256 else {
                    throw DetectorError.invalidImage(
                        "Image hash differs from the manifest.")
                }
                let image = try await worker.load(input)
                guard image.width == sample.width, image.height == sample.height else {
                    throw DetectorError.invalidImage("Decoded image dimensions differ.")
                }
                let result = try await worker.run(image, model: model)
                try Task.checkCancellation()
                rows.append(
                    StillImageBatchRow(
                        image_id: sample.image_id, file_name: sample.file_name,
                        image_sha256: hash, width: image.width, height: image.height,
                        status: "completed", error: nil,
                        latency_ms: result.latencyMilliseconds,
                        detections: result.detections.map(StillImagePrediction.init)
                    ))
            } catch {
                try Task.checkCancellation()
                if error is CancellationError { throw error }
                rows.append(
                    StillImageBatchRow(
                        image_id: sample.image_id, file_name: sample.file_name,
                        image_sha256: hash, width: sample.width, height: sample.height,
                        status: "error", error: error.localizedDescription,
                        latency_ms: nil, detections: []
                    ))
            }
            await progress(rows.count, catalog.samples.count)
        }
        try Task.checkCancellation()
        return StillImageBatchReport(
            schema_version: 1, scope: catalog.scope,
            created_at: ISO8601DateFormatter().string(from: Date()),
            status: rows.allSatisfy { $0.status == "completed" }
                ? "completed" : "incomplete",
            model: catalog.model, run_id: catalog.run_id,
            model_sha256: catalog.model_sha256,
            checkpoint_sha256: catalog.checkpoint_sha256,
            sample_manifest_sha256: StillImageCatalog.digest(data),
            split: catalog.split,
            confidence_threshold: catalog.confidence_threshold,
            max_detections: catalog.max_detections,
            box_convention: catalog.box_convention,
            runtime: runtime, images: rows
        )
    }

    /// Writes a unique atomic shareable JSON file, retaining full numeric precision.
    func export(_ report: StillImageBatchReport) throws -> URL {
        try Task.checkCancellation()
        try FileManager.default.createDirectory(
            at: exportDirectory, withIntermediateDirectories: true)
        let url = exportDirectory.appendingPathComponent(
            "\(report.model)_test_\(UUID().uuidString)_predictions.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: url, options: .atomic)
        return url
    }
}
