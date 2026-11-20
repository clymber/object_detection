import Foundation

/// Validated host-clock endpoints; source PTS stays in the separate frame metadata.
nonisolated struct VideoFrameTimings: Equatable, Sendable {
    let readyMilliseconds: Double
    let modelTiming: ModelPredictionTiming?
    let doneMilliseconds: Double
    let publishedMilliseconds: Double?

    enum ValidationError: Error { case invalidEndpoints, invalidPublication }

    var modelLatencyMilliseconds: Double? {
        guard let modelTiming, let completed = modelTiming.completedMilliseconds
        else { return nil }
        return completed - modelTiming.submittedMilliseconds
    }

    var endToEndMilliseconds: Double { doneMilliseconds - readyMilliseconds }

    /// Rejects nonfinite, negative or out-of-order endpoints, including partial models.
    init(
        readyMilliseconds: Double, modelTiming: ModelPredictionTiming?,
        doneMilliseconds: Double, publishedMilliseconds: Double? = nil
    ) throws {
        var endpoints = [readyMilliseconds]
        if let modelTiming {
            endpoints.append(modelTiming.submittedMilliseconds)
            if let completed = modelTiming.completedMilliseconds {
                endpoints.append(completed)
            }
        }
        endpoints.append(doneMilliseconds)
        if let publishedMilliseconds { endpoints.append(publishedMilliseconds) }
        guard endpoints.allSatisfy({ $0.isFinite && $0 >= 0 }),
            zip(endpoints, endpoints.dropFirst()).allSatisfy({ $0 <= $1 })
        else { throw ValidationError.invalidEndpoints }
        self.readyMilliseconds = readyMilliseconds
        self.modelTiming = modelTiming
        self.doneMilliseconds = doneMilliseconds
        self.publishedMilliseconds = publishedMilliseconds
    }
}
