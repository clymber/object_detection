import CoreML
import Foundation

/// Loads an actual export and exercises its fixed tensor contract on the Mac.
@main
struct CoreMLRuntimeSmokeTests {
    /// Accepts a model ID and artifact URL, then runs one CPU prediction.
    static func main() async throws {
        precondition(CommandLine.arguments.count == 3, "Pass model ID and package path")
        let definition = DetectorModel(rawValue: CommandLine.arguments[1])!
        let url = URL(fileURLWithPath: CommandLine.arguments[2])
        let runtime = try await CoreMLRuntime(
            definition: definition, modelURL: url, computeUnits: .cpuOnly
        )
        let input = try FloatTensor(
            shape: definition.inputShape,
            values: Array(repeating: 0, count: 3 * 640 * 640)
        )
        let outputs = try await runtime.predict(input)
        precondition(Set(outputs.keys) == Set(definition.outputShapes.keys))
        for (name, tensor) in outputs {
            precondition(tensor.shape == definition.outputShapes[name])
            precondition(tensor.values.allSatisfy(\.isFinite))
        }
        do {
            _ = try await runtime.predict(FloatTensor(shape: [1], values: [0]))
            preconditionFailure("Wrong input shape was accepted")
        } catch DetectorError.invalidTensor {}
        print(
            "\(definition.rawValue): real Core ML load/prediction and input rejection passed."
        )
    }
}
