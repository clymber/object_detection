import CoreGraphics
import CryptoKit
import Darwin
import Foundation

// Wire names intentionally match the Python comparison schema.

/// Records the actual execution platform separately from the model provenance.
nonisolated struct StillImageRuntime: Codable, Sendable {
    let platform: String
    let hardware: String
    let os_version: String
    let compute_units: String

    /// Identifies a physical phone, simulator or host without device-name PII.
    static func current(computeUnits: String) -> Self {
        #if targetEnvironment(macCatalyst)
            let platform = "mac_catalyst"
        #elseif os(iOS) && !targetEnvironment(simulator)
            let platform = "ios"
        #elseif targetEnvironment(simulator)
            let platform = "ios_simulator"
        #else
            let platform = "macos"
        #endif
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: max(size, 1))
        _ = machine.withUnsafeMutableBufferPointer {
            sysctlbyname("hw.machine", $0.baseAddress, &size, nil, 0)
        }
        return Self(
            platform: platform, hardware: String(cString: machine),
            os_version: ProcessInfo.processInfo.operatingSystemVersionString,
            compute_units: computeUnits
        )
    }
}

/// Preserves unrounded confidence and unclipped source-pixel xywh coordinates.
nonisolated struct StillImagePrediction: Codable, Sendable {
    let category_id: Int
    let bbox: [Double]
    let score: Double

    /// Copies native output directly without applying presentation formatting.
    init(_ detection: Detection) {
        category_id = detection.classID
        bbox = [
            detection.boundingBox.minX, detection.boundingBox.minY,
            detection.boundingBox.width, detection.boundingBox.height,
        ]
        score = detection.confidence
    }
}

/// Includes failed reads/inferences and successful empty results explicitly.
nonisolated struct StillImageBatchRow: Codable, Sendable {
    let image_id: Int
    let file_name: String
    let image_sha256: String
    let width: Int
    let height: Int
    let status: String
    let error: String?
    let latency_ms: Double?
    let detections: [StillImagePrediction]
}

/// Is a prediction export, not a claim that device parity has passed.
nonisolated struct StillImageBatchReport: Codable, Sendable {
    let schema_version: Int
    let scope: String
    let created_at: String
    let status: String
    let model: String
    let run_id: String
    let model_sha256: String
    let checkpoint_sha256: String
    let sample_manifest_sha256: String
    let split: String
    let confidence_threshold: Double
    let max_detections: Int
    let box_convention: String
    let runtime: StillImageRuntime
    let images: [StillImageBatchRow]
}

/// Reads only the catalog identity and inputs, never its expected predictions.
nonisolated struct StillImageCatalog: Decodable, Sendable {
    let schema_version: Int
    let scope: String
    let model: String
    let run_id: String
    let model_sha256: String
    let checkpoint_sha256: String
    let split: String
    let confidence_threshold: Double
    let max_detections: Int
    let box_convention: String
    let samples: [Sample]

    /// Describes an original input whose bytes are verified before inference.
    nonisolated struct Sample: Decodable, Sendable {
        let image_id: Int
        let file_name: String
        let sha256: String
        let width: Int
        let height: Int
        let split: String
    }

    /// Hashes the exact manifest/image bytes shared with the Mac comparator.
    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
