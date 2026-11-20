import CoreGraphics
import Foundation

/// Describes an inference failure independently of a successful empty prediction.
nonisolated struct VideoInferenceFailure: Equatable, Sendable {
    let type: String
    let message: String

    /// Retains diagnostic context without carrying backend objects into result logs.
    init(_ error: any Error) {
        type = String(reflecting: Swift.type(of: error))
        message = error.localizedDescription
    }
}

/// One immutable frame outcome before CSV/JSON export or optional preview rendering.
nonisolated struct VideoFrameResult: Sendable {
    enum Status: String, Sendable { case completed, failed }

    /// Includes source identity, loop/index and the exact original media timestamp.
    let metadata: VideoFrameMetadata

    /// Boxes use top-left pixels in this upright source size, without display clipping.
    let sourceWidth: Int
    let sourceHeight: Int
    let bufferToImage: CGAffineTransform
    let status: Status
    let detections: [Detection]
    let failure: VideoInferenceFailure?
    let timings: VideoFrameTimings

    var processingDurationMilliseconds: Double { timings.endToEndMilliseconds }

    /// Stamps actual consumer receipt after the async handoff, including preview off.
    func published(at milliseconds: Double = InferenceClock.nowMilliseconds()) throws
        -> VideoFrameResult
    {
        guard status == .completed, timings.publishedMilliseconds == nil else {
            throw VideoFrameTimings.ValidationError.invalidPublication
        }
        return VideoFrameResult(
            metadata: metadata, sourceWidth: sourceWidth, sourceHeight: sourceHeight,
            bufferToImage: bufferToImage, status: status, detections: detections,
            failure: failure,
            timings: try VideoFrameTimings(
                readyMilliseconds: timings.readyMilliseconds,
                modelTiming: timings.modelTiming,
                doneMilliseconds: timings.doneMilliseconds,
                publishedMilliseconds: milliseconds))
    }
}
