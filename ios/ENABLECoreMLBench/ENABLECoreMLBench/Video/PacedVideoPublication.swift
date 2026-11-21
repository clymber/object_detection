import Foundation

/// Describes result delivery separately from inference completion and queue skips.
nonisolated struct PacedVideoPublicationOutcome: Sendable {
    enum Status: String, Sendable {
        case notApplicable = "not_applicable"
        case notRequested = "not_requested"
        case pending, published, late, superseded, cancelled, failed
        case notReceived = "not_received"
    }

    let status: Status
    let checkedMilliseconds: Double?
    let publishedMilliseconds: Double?
    let failure: VideoInferenceFailure?

    /// Leaves receipt time unset unless the consumer actually accepts a fresh result.
    init(
        status: Status, checkedMilliseconds: Double? = nil,
        publishedMilliseconds: Double? = nil, failure: VideoInferenceFailure? = nil
    ) {
        self.status = status
        self.checkedMilliseconds = checkedMilliseconds
        self.publishedMilliseconds = publishedMilliseconds
        self.failure = failure
    }
}

/// An opaque, single-use receipt; raw detections stay hidden until freshness passes.
nonisolated struct PacedVideoDelivery: Sendable {
    private let accept: @Sendable () throws -> VideoFrameResult?

    /// Binds this receipt to the scheduler's cancellation and monotonic-clock gate.
    init(accept: @escaping @Sendable () throws -> VideoFrameResult?) {
        self.accept = accept
    }

    /// Call on the receiving actor, then use the result synchronously without another
    /// await. Nil means late, cancelled, already received or an expired callback.
    func receive() throws -> VideoFrameResult? { try accept() }
}
