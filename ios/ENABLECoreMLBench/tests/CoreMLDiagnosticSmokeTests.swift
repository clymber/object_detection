import CoreML
import Foundation

/// Checks launch validation before allowing expensive or mislabelled device runs.
@main
struct CoreMLDiagnosticSmokeTests {
    /// Exercises each execution policy and rejects ambiguous/missing provenance.
    static func main() throws {
        let provenance = [
            "--revision", "test-revision", "--toolchain", "test-toolchain",
            "--conditions", "test conditions",
        ]
        for (name, expected) in [
            ("all", MLComputeUnits.all), ("cpuAndGPU", .cpuAndGPU),
            ("cpuOnly", .cpuOnly),
        ] {
            let config = try CoreMLDiagnosticConfiguration(
                arguments: ["--compute-units", name] + provenance)
            precondition(config.computeUnits == expected)
            precondition(config.warmupCount == 30 && config.measuredCount == 30)
            precondition(config.model == .rfdetrSmall)
        }
        let valid = ["--compute-units", "all"] + provenance
        for invalid in [
            [], ["--compute-units"], ["--compute-units", "unknown"] + provenance,
            valid + ["--samples", "0"], valid + ["--warmup", "-1"],
            valid + ["--compute-units", "cpuOnly"], valid + ["--unknown", "x"],
            valid + ["--model", "invalid"],
        ] {
            do {
                _ = try CoreMLDiagnosticConfiguration(arguments: invalid)
                preconditionFailure("Invalid diagnostic launch was accepted")
            } catch DetectorError.invalidModel {}
        }
        print("Core ML diagnostic launch validation passed.")
    }
}
