import CoreML
import Foundation

/// Owns one Core ML model and keeps raw predictions on a worker actor.
actor CoreMLRuntime {
    private let model: MLModel
    let definition: DetectorModel

    /// Loads a compiled resource or compiles an explicit package URL before loading.
    init(
        definition: DetectorModel, modelURL: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws {
        let compiledURL: URL
        switch modelURL.pathExtension {
        case "mlmodelc": compiledURL = modelURL
        case "mlpackage", "mlmodel":
            compiledURL = try await MLModel.compileModel(at: modelURL)
        default:
            throw DetectorError.invalidModel(
                "unsupported file: \(modelURL.lastPathComponent)")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let loaded = try await MLModel.load(
            contentsOf: compiledURL, configuration: configuration)
        try Self.validate(loaded.modelDescription, definition: definition)
        self.definition = definition
        self.model = loaded
    }

    /// Executes one fixed-shape input and returns validated, named FP32 tensors.
    func predict(_ input: FloatTensor) throws -> [String: FloatTensor] {
        try Task.checkCancellation()
        guard input.shape == definition.inputShape, input.values.allSatisfy(\.isFinite)
        else {
            throw DetectorError.invalidTensor(
                "input shape or values differ from contract")
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            definition.inputName: MLFeatureValue(multiArray: input.multiArray())
        ])
        let prediction = try model.prediction(from: provider)
        try Task.checkCancellation()
        var outputs: [String: FloatTensor] = [:]
        for (name, shape) in definition.outputShapes {
            guard let array = prediction.featureValue(for: name)?.multiArrayValue,
                array.shape.map(\.intValue) == shape,
                array.dataType == definition.outputDataType
            else {
                throw DetectorError.invalidTensor("unexpected output: \(name)")
            }
            let tensor = try FloatTensor(array)
            guard tensor.values.allSatisfy(\.isFinite) else {
                throw DetectorError.invalidTensor("nonfinite output: \(name)")
            }
            outputs[name] = tensor
        }
        return outputs
    }

    /// Rejects incompatible artifacts before accepting them for inference.
    private static func validate(
        _ description: MLModelDescription, definition: DetectorModel
    ) throws {
        let inputs = description.inputDescriptionsByName
        let outputs = description.outputDescriptionsByName
        guard Set(inputs.keys) == [definition.inputName],
            let input = inputs[definition.inputName]?.multiArrayConstraint,
            input.shape.map(\.intValue) == definition.inputShape,
            input.dataType == .float32,
            Set(outputs.keys) == Set(definition.outputShapes.keys)
        else {
            throw DetectorError.invalidModel(
                "input/output names, input shape or FP32 input")
        }
        for (name, shape) in definition.outputShapes {
            guard let constraint = outputs[name]?.multiArrayConstraint,
                constraint.shape.map(\.intValue) == shape,
                constraint.dataType == definition.outputDataType
            else {
                throw DetectorError.invalidModel("output contract differs: \(name)")
            }
        }
    }
}
