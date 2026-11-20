import Foundation

/// One process-wide monotonic millisecond clock shared by all pipeline actors.
nonisolated enum InferenceClock {
    private static let origin = ContinuousClock.now

    /// Uses one origin rather than resetting the clock for individual requests.
    static func nowMilliseconds() -> Double {
        let reference = origin
        let elapsed = reference.duration(to: ContinuousClock.now).components
        return Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15
    }
}

/// Measures prepared-input submission through required CPU output readback.
nonisolated struct ModelPredictionTiming: Equatable, Sendable {
    let submittedMilliseconds: Double
    let completedMilliseconds: Double?
}

/// Supports untimed adapters without inventing a model-only duration for them.
nonisolated struct DetectorPrediction: Sendable {
    let detections: [Detection]
    let modelTiming: ModelPredictionTiming?
}

/// Carries partial timing through runtime or postprocessing failures.
nonisolated struct MeasuredPredictionFailure: Error {
    let underlying: any Error
    let modelTiming: ModelPredictionTiming
}
