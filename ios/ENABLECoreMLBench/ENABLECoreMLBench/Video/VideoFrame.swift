import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation

/// Immutable source metadata that can cross actors without retaining frame pixels.
nonisolated struct VideoFrameMetadata: Equatable, Sendable {
    /// Stable input-manifest source/track ID, resolved there to the clip's hash.
    /// Use the same ID after copying a clip; an import-cache URL is not an ID.
    let sourceID: String

    /// Zero-based playback iteration; the original source timestamp stays unchanged.
    let loopIndex: Int

    /// Zero-based presentation-order index in the source, before frame selection.
    /// Selected or skipped frames never cause later frames to be renumbered.
    let sourceFrameIndex: Int

    /// Actual sample PTS, preserving value, timescale, flags and epoch exactly.
    /// This is source media time, not a host-clock timestamp or index/FPS estimate.
    let presentationTime: CMTime

    /// Decoded pixel-buffer dimensions before applying the orientation transform.
    let width: Int
    let height: Int

    /// Source presentation transform, including rotation, mirroring and translation.
    /// Pixels remain unmodified. Use identity if orientation was already applied.
    let preferredTransform: CGAffineTransform

    /// Rejects unusable metadata without replacing timestamps or geometry.
    enum ValidationError: Error, Equatable {
        case invalidSourceID
        case invalidLoopIndex
        case invalidFrameIndex
        case invalidPresentationTime
        case invalidDimensions
        case invalidTransform
    }

    /// Validates source metadata while retaining the original time and transform.
    init(
        sourceID: String, sourceFrameIndex: Int, presentationTime: CMTime,
        width: Int, height: Int, preferredTransform: CGAffineTransform = .identity,
        loopIndex: Int = 0
    ) throws {
        guard !sourceID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError.invalidSourceID
        }
        guard loopIndex >= 0 else { throw ValidationError.invalidLoopIndex }
        guard sourceFrameIndex >= 0 else { throw ValidationError.invalidFrameIndex }
        guard presentationTime.isNumeric, presentationTime.timescale > 0 else {
            throw ValidationError.invalidPresentationTime
        }
        guard width > 0, height > 0 else { throw ValidationError.invalidDimensions }
        let transform = preferredTransform
        let components = [transform.a, transform.b, transform.c, transform.d,
                          transform.tx, transform.ty]
        let determinant = transform.a * transform.d - transform.b * transform.c
        guard components.allSatisfy({ $0.isFinite }),
            determinant.isFinite, determinant != 0
        else { throw ValidationError.invalidTransform }

        self.sourceID = sourceID
        self.loopIndex = loopIndex
        self.sourceFrameIndex = sourceFrameIndex
        self.presentationTime = presentationTime
        self.width = width
        self.height = height
        self.preferredTransform = preferredTransform
    }
}

/// Retains one decoded frame and the metadata needed to identify its results.
///
/// Identity within a run is (sourceID, loopIndex, sourceFrameIndex). The buffer is
/// retained after its sample buffer is released; no pixels are copied or rotated.
/// This type deliberately does not conform to Sendable: CVPixelBuffer is mutable.
/// Keep it inside the owning worker and treat pixels/attachments as read-only while
/// it is in use. Pass metadata or a decoded immutable image across actor boundaries.
nonisolated struct VideoFrame {
    let metadata: VideoFrameMetadata
    let pixelBuffer: CVPixelBuffer

    /// Derives dimensions from the retained buffer to prevent mismatched metadata.
    init(
        sourceID: String, sourceFrameIndex: Int, presentationTime: CMTime,
        preferredTransform: CGAffineTransform = .identity, loopIndex: Int = 0,
        pixelBuffer: CVPixelBuffer
    ) throws {
        metadata = try VideoFrameMetadata(
            sourceID: sourceID, sourceFrameIndex: sourceFrameIndex,
            presentationTime: presentationTime,
            width: CVPixelBufferGetWidth(pixelBuffer),
            height: CVPixelBufferGetHeight(pixelBuffer),
            preferredTransform: preferredTransform, loopIndex: loopIndex)
        self.pixelBuffer = pixelBuffer
    }
}
