import CoreGraphics
import CoreML
import Foundation

/// Restricts diagnostic launches to explicit, reproducible configurations.
nonisolated struct CoreMLDiagnosticConfiguration: Sendable {
    let model: DetectorModel
    let computeUnitsName: String
    let warmupCount: Int
    let measuredCount: Int
    let revision: String
    let toolchain: String
    let conditions: String

    var computeUnits: MLComputeUnits {
        switch computeUnitsName {
        case "cpuAndGPU": .cpuAndGPU
        case "cpuOnly": .cpuOnly
        default: .all
        }
    }

    /// Rejects missing provenance, unknown switches and invalid sample counts.
    init(arguments: [String]) throws {
        let allowed = Set([
            "--model", "--compute-units", "--warmup", "--samples",
            "--revision", "--toolchain", "--conditions",
        ])
        var values: [String: String] = [:]
        guard arguments.count.isMultiple(of: 2) else {
            throw DetectorError.invalidModel("Diagnostic options need values.")
        }
        for index in stride(from: 0, to: arguments.count, by: 2) {
            let key = arguments[index]
            guard allowed.contains(key), values[key] == nil else {
                throw DetectorError.invalidModel("Invalid diagnostic option: \(key)")
            }
            values[key] = arguments[index + 1]
        }
        guard let model = DetectorModel(rawValue: values["--model"] ?? "rfdetr_small"),
            let units = values["--compute-units"],
            ["all", "cpuAndGPU", "cpuOnly"].contains(units),
            let warmup = Int(values["--warmup"] ?? "30"), (1...1000).contains(warmup),
            let count = Int(values["--samples"] ?? "30"), (1...10000).contains(count),
            let revision = values["--revision"], !revision.isEmpty,
            let toolchain = values["--toolchain"], !toolchain.isEmpty,
            let conditions = values["--conditions"], !conditions.isEmpty
        else {
            throw DetectorError.invalidModel("Invalid diagnostic configuration.")
        }
        self.model = model
        computeUnitsName = units
        warmupCount = warmup
        measuredCount = count
        self.revision = revision
        self.toolchain = toolchain
        self.conditions = conditions
    }
}

/// Keeps every prediction and its monotonic pipeline boundaries for recalculation.
nonisolated struct CoreMLDiagnosticSample: Codable, Sendable {
    let phase: String
    let index: Int
    let ready_ms: Double
    let submit_ms: Double
    let model_done_ms: Double
    let done_ms: Double
    let detections: [StillImagePrediction]
}

/// Describes planned placement; this is not an execution trace or fallback proof.
nonisolated struct CoreMLDiagnosticOperation: Codable, Sendable {
    let path: String
    let operator_name: String
    let outputs: [String]
    let preferred_device: String?
    let supported_devices: [String]?
    let estimated_cost_weight: Double?
}

/// Separates one-image diagnostic timing from formal frozen-video benchmarks.
nonisolated struct CoreMLDiagnosticReport: Encodable, Sendable {
    let schema_version = 1
    let scope = "coreml_still_image_diagnostics"
    let formal_benchmark = false
    let created_at: String
    let model: String
    let model_sha256: String
    let checkpoint_sha256: String
    let sample_manifest_sha256: String
    let image_sha256: String
    let image_id = 6988
    let runtime: StillImageRuntime
    let app_revision: String
    let toolchain: String
    let build_configuration: String
    let conditions: String
    let warmup_count: Int
    let measured_count: Int
    let thermal_start: Int
    let low_power_mode: Bool
    var thermal_end: Int?
    var status = "running"
    var error: String?
    var load_ms: Double?
    var samples: [CoreMLDiagnosticSample] = []
    var compute_plan: [CoreMLDiagnosticOperation] = []
    var compute_plan_error: String?
    var parity_export: String?
}

/// Repeats the same decoded pixels and exports timings, placement and parity data.
actor CoreMLDiagnosticRunner {
    /// Runs only when explicitly requested by launch arguments, off the UI actor.
    func run(_ configuration: CoreMLDiagnosticConfiguration) async throws -> URL {
        let model = configuration.model
        let bundle = Bundle.main
        guard let manifestURL = ["TestImages", "Resources/TestImages", nil]
            .compactMap({ bundle.url(forResource: model.sampleManifestName,
                withExtension: "json", subdirectory: $0) }).first
        else { throw DetectorError.invalidImage("Missing diagnostic sample manifest.") }
        let manifestData = try Data(contentsOf: manifestURL)
        let catalog = try JSONDecoder().decode(StillImageCatalog.self, from: manifestData)
        let sample = BundledSample.testPositive6988
        let hash = StillImageCatalog.digest(
            try Data(contentsOf: sample.resourceURL(in: bundle)))
        let image = try sample.load(in: bundle)
        guard catalog.model == model.rawValue,
            let record = catalog.samples.first(where: { $0.image_id == 6988 }),
            record.sha256 == hash, record.width == image.width,
            record.height == image.height
        else { throw DetectorError.invalidImage("Diagnostic sample identity differs.") }
        let directory = URL.documentsDirectory.appendingPathComponent(
            "CoreMLDiagnostics/\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("diagnostic.json")
        #if DEBUG
            let build = "Debug"
        #else
            let build = "Release"
        #endif
        var report = CoreMLDiagnosticReport(
            created_at: ISO8601DateFormatter().string(from: Date()),
            model: model.rawValue, model_sha256: catalog.model_sha256,
            checkpoint_sha256: catalog.checkpoint_sha256,
            sample_manifest_sha256: StillImageCatalog.digest(manifestData),
            image_sha256: hash,
            runtime: .current(computeUnits: configuration.computeUnitsName),
            app_revision: configuration.revision, toolchain: configuration.toolchain,
            build_configuration: build, conditions: configuration.conditions,
            warmup_count: configuration.warmupCount,
            measured_count: configuration.measuredCount,
            thermal_start: ProcessInfo.processInfo.thermalState.rawValue,
            low_power_mode: ProcessInfo.processInfo.isLowPowerModeEnabled)
        do {
            let engine = DetectorEngine(computeUnits: configuration.computeUnits)
            Self.log("load_begin \(model.rawValue) \(configuration.computeUnitsName)")
            let loadStart = InferenceClock.nowMilliseconds()
            try await engine.selectModel(model)
            report.load_ms = InferenceClock.nowMilliseconds() - loadStart
            Self.log("load_end")
            Self.log("first_prediction_begin")
            report.samples.append(try await measure(engine, image: image,
                phase: "first_prediction", index: 0))
            Self.log("first_prediction_end warmup_begin")
            for index in 0..<configuration.warmupCount {
                report.samples.append(try await measure(engine, image: image,
                    phase: "warmup", index: index))
            }
            Self.log("warmup_end measured_begin")
            // All pixels are in memory. No console or disk writes between requests.
            for index in 0..<configuration.measuredCount {
                report.samples.append(try await measure(engine, image: image,
                    phase: "measured", index: index))
            }
            report.thermal_end = ProcessInfo.processInfo.thermalState.rawValue
            Self.log("measured_end parity_begin")
            let batch = StillImageBatchRunner(
                worker: StillImageRunner(engine: engine),
                runtime: report.runtime, exportDirectory: directory)
            let parity = try await batch.run(model: model, progress: { _, _ in })
            report.parity_export = try await batch.export(parity).lastPathComponent
            guard parity.status == "completed" else {
                throw DetectorError.invalidTensor("Diagnostic parity batch incomplete.")
            }
            Self.log("parity_end compute_plan_begin")
            do {
                report.compute_plan = try await computePlan(
                    model: model, units: configuration.computeUnits)
            } catch {
                report.compute_plan_error = error.localizedDescription
            }
            Self.log("compute_plan_end")
            try Task.checkCancellation()
            report.status = report.compute_plan_error == nil
                ? "completed" : "completed_without_compute_plan"
        } catch {
            report.status = "failed"
            report.error = error.localizedDescription
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: url, options: .atomic)
        Self.log("report_saved \(url.path) status=\(report.status)")
        guard report.status == "completed" else {
            throw DetectorError.invalidModel(report.error
                ?? report.compute_plan_error ?? "Diagnostic run incomplete.")
        }
        return url
    }

    /// Includes filtering in postprocessing and excludes image decode/model load.
    private func measure(
        _ engine: DetectorEngine, image: CGImage, phase: String, index: Int
    ) async throws -> CoreMLDiagnosticSample {
        try Task.checkCancellation()
        let ready = InferenceClock.nowMilliseconds()
        let prediction = try await engine.detectMeasured(in: image)
        let retained = prediction.detections.enumerated()
            .filter { $0.element.confidence >= 0.25 }.sorted {
                $0.element.confidence == $1.element.confidence
                    ? $0.offset < $1.offset
                    : $0.element.confidence > $1.element.confidence
            }.prefix(100).map(\.element)
        let done = InferenceClock.nowMilliseconds()
        guard let timing = prediction.modelTiming,
            let modelDone = timing.completedMilliseconds,
            ready <= timing.submittedMilliseconds,
            timing.submittedMilliseconds <= modelDone, modelDone <= done
        else { throw DetectorError.invalidTensor("Invalid diagnostic timing.") }
        return CoreMLDiagnosticSample(phase: phase, index: index, ready_ms: ready,
            submit_ms: timing.submittedMilliseconds, model_done_ms: modelDone,
            done_ms: done, detections: retained.map(StillImagePrediction.init))
    }

    /// Loads placement after timing so inspection cannot prewarm measured loading.
    private func computePlan(
        model: DetectorModel, units: MLComputeUnits
    ) async throws -> [CoreMLDiagnosticOperation] {
        let url = try model.bundledURL()
        let compiled = url.pathExtension == "mlmodelc"
            ? url : try await MLModel.compileModel(at: url)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        let plan = try await MLComputePlan.load(
            contentsOf: compiled, configuration: configuration)
        guard case let .program(program) = plan.modelStructure else {
            throw DetectorError.invalidModel("Expected an ML Program compute plan.")
        }
        var rows: [CoreMLDiagnosticOperation] = []
        for name in program.functions.keys.sorted() {
            if let function = program.functions[name] {
                Self.visit(function.block, path: name, plan: plan, rows: &rows)
            }
        }
        return rows
    }

    /// Preserves nested operation paths and unknown placement/cost as null.
    private static func visit(
        _ block: MLModelStructure.Program.Block, path: String,
        plan: MLComputePlan, rows: inout [CoreMLDiagnosticOperation]
    ) {
        for (index, operation) in block.operations.enumerated() {
            let location = "\(path)/\(index)"
            let usage = plan.deviceUsage(for: operation)
            rows.append(CoreMLDiagnosticOperation(path: location,
                operator_name: operation.operatorName,
                outputs: operation.outputs.map(\.name),
                preferred_device: usage.map { deviceName($0.preferred) },
                supported_devices: usage.map { $0.supported.map(deviceName) },
                estimated_cost_weight: plan.estimatedCost(of: operation)?.weight))
            for (child, nested) in operation.blocks.enumerated() {
                visit(nested, path: "\(location)/block\(child)", plan: plan, rows: &rows)
            }
        }
    }

    /// Uses stable names instead of hardware object descriptions in the export.
    private static func deviceName(_ device: MLComputeDevice) -> String {
        switch device {
        case .cpu: "cpu"
        case .gpu: "gpu"
        case .neuralEngine: "neuralEngine"
        @unknown default: "unknown"
        }
    }

    /// Flushes phase markers immediately for correlation with Core ML messages.
    private static func log(_ message: String) {
        let time = InferenceClock.nowMilliseconds()
        FileHandle.standardOutput.write(Data("COREML_DIAGNOSTIC \(time) \(message)\n".utf8))
    }
}
