import CoreML
import Foundation

/// The three selected basketball exports and their explicit tensor contracts.
nonisolated enum DetectorModel: String, CaseIterable, Sendable, Identifiable {
    case rfdetrSmall = "rfdetr_small"
    case rfdetrNano = "rfdetr_nano"
    case yoloxNano = "yolox_nano"

    var id: String { rawValue }

    var sampleManifestName: String {
        "test_images_\(rawValue)_manifest"
    }

    var displayName: String {
        switch self {
        case .rfdetrSmall: "RF-DETR Small"
        case .rfdetrNano: "RF-DETR Nano"
        case .yoloxNano: "YOLOX Nano"
        }
    }

    var inputName: String { self == .yoloxNano ? "images" : "tensors" }
    var inputShape: [Int] { [1, 3, 640, 640] }
    var isRFDETR: Bool { self != .yoloxNano }

    var outputShapes: [String: [Int]] {
        switch self {
        case .rfdetrSmall: ["concat_2": [1, 300, 4], "linear_112": [1, 300, 2]]
        case .rfdetrNano: ["concat_2": [1, 300, 4], "linear_102": [1, 300, 2]]
        case .yoloxNano: ["output": [1, 8400, 6]]
        }
    }

    var outputDataType: MLMultiArrayDataType { isRFDETR ? .float16 : .float32 }
    var logitsName: String { self == .rfdetrSmall ? "linear_112" : "linear_102" }

    /// Finds a uniquely named compiled model or package in the app resources.
    func bundledURL(in bundle: Bundle = .main) throws -> URL {
        for directory in ["Models", nil] as [String?] {
            for extensionName in ["mlmodelc", "mlpackage"] {
                if let url = bundle.url(
                    forResource: rawValue, withExtension: extensionName,
                    subdirectory: directory
                ) {
                    return url
                }
            }
        }
        throw DetectorError.missingModel(rawValue)
    }
}

/// Errors shared by model loading, tensor conversion and detector adapters.
nonisolated enum DetectorError: LocalizedError {
    case missingModel(String)
    case invalidModel(String)
    case invalidTensor(String)
    case invalidImage(String)
    case noModelSelected
    case selectionSuperseded

    var errorDescription: String? {
        switch self {
        case .missingModel(let name): "Model resource is missing: \(name)"
        case .invalidModel(let reason): "Invalid model contract: \(reason)"
        case .invalidTensor(let reason): "Invalid tensor: \(reason)"
        case .invalidImage(let reason): "Invalid image: \(reason)"
        case .noModelSelected: "Select and load a detector before running inference."
        case .selectionSuperseded: "A newer model selection superseded this operation."
        }
    }
}
