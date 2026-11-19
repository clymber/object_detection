import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation

/// Checks video frame identity, exact PTS, orientation and pixel-buffer ownership.
@main
struct VideoFrameContractSmokeTests {
    /// Exercises valid frames, malformed metadata and metadata actor handoff.
    static func main() async throws {
        try testBufferOwnership()
        try testExactTimestamp()
        try testSourceIdentity()
        try testOrientation()
        try testInvalidMetadata()
        try await testMetadataActorBoundary()
        print("Video frame contract: all 6 checks passed.")
    }

    /// Keeps decoded pixels alive after releasing the originating sample buffer.
    private static func testBufferOwnership() throws {
        let frame = try makeFrame()
        precondition(frame.metadata.width == 16 && frame.metadata.height == 8)
        precondition(CVPixelBufferGetWidth(frame.pixelBuffer) == 16)
        precondition(CVPixelBufferGetHeight(frame.pixelBuffer) == 8)
        precondition(
            CVPixelBufferGetPixelFormatType(frame.pixelBuffer)
                == kCVPixelFormatType_32BGRA)
        precondition(
            CVPixelBufferLockBaseAddress(frame.pixelBuffer, .readOnly)
                == kCVReturnSuccess)
        defer { CVPixelBufferUnlockBaseAddress(frame.pixelBuffer, .readOnly) }
        let bytes = CVPixelBufferGetBaseAddress(frame.pixelBuffer)!
            .assumingMemoryBound(to: UInt8.self)
        precondition(bytes[0] == 73)
    }

    /// Preserves fractional, rounded, negative and nonzero-epoch source times.
    private static func testExactTimestamp() throws {
        let times = [
            CMTime(value: 1001, timescale: 30000),
            CMTime(value: -1001, timescale: 30000),
            CMTime(value: 2002, timescale: 60000,
                   flags: [.valid, .hasBeenRounded], epoch: 7),
            .zero,
        ]
        for time in times {
            let metadata = try makeMetadata(time: time)
            precondition(metadata.presentationTime.value == time.value)
            precondition(metadata.presentationTime.timescale == time.timescale)
            precondition(metadata.presentationTime.flags == time.flags)
            precondition(metadata.presentationTime.epoch == time.epoch)
        }
    }

    /// Keeps source indices across selection gaps and distinguishes playback loops.
    private static func testSourceIdentity() throws {
        let first = try makeMetadata(index: 60)
        let selected = try makeMetadata(index: 120)
        let repeated = try makeMetadata(index: 60, loop: 1)
        precondition(first.sourceID == selected.sourceID)
        precondition(first.sourceFrameIndex == 60 && selected.sourceFrameIndex == 120)
        precondition(first.loopIndex == 0 && repeated.loopIndex == 1)
        precondition(first.presentationTime == repeated.presentationTime)
        precondition(first != repeated)
    }

    /// Preserves presentation rotation and mirroring without swapping buffer sizes.
    private static func testOrientation() throws {
        let transforms = [
            CGAffineTransform.identity,
            CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 8, ty: 0),
            CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: 16, ty: 0),
        ]
        for transform in transforms {
            let frame = try makeFrame(transform: transform)
            precondition(frame.metadata.preferredTransform == transform)
            precondition(frame.metadata.width == 16 && frame.metadata.height == 8)
        }
    }

    /// Rejects missing identities, invalid times, dimensions and singular transforms.
    private static func testInvalidMetadata() throws {
        try expectFailure(.invalidSourceID) { _ = try makeMetadata(sourceID: " \n") }
        try expectFailure(.invalidFrameIndex) { _ = try makeMetadata(index: -1) }
        try expectFailure(.invalidLoopIndex) { _ = try makeMetadata(loop: -1) }
        for time in [CMTime.invalid, .indefinite, .positiveInfinity, .negativeInfinity,
                     CMTime(value: 1, timescale: 0),
                     CMTime(value: 1, timescale: -1)] {
            try expectFailure(.invalidPresentationTime) {
                _ = try makeMetadata(time: time)
            }
        }
        for size in [(0, 8), (16, 0), (-1, 8)] {
            try expectFailure(.invalidDimensions) {
                _ = try makeMetadata(width: size.0, height: size.1)
            }
        }
        for transform in [
            CGAffineTransform(a: 1, b: 0, c: 0, d: 0, tx: 0, ty: 0),
            CGAffineTransform(a: 1, b: 0, c: 0, d: 1, tx: .nan, ty: 0),
            CGAffineTransform(a: .infinity, b: 0, c: 0, d: 1, tx: 0, ty: 0),
        ] {
            try expectFailure(.invalidTransform) {
                _ = try makeMetadata(transform: transform)
            }
        }
    }

    /// Transfers metadata between actors while each worker keeps its pixels local.
    @MainActor
    private static func testMetadataActorBoundary() async throws {
        let expected = try makeMetadata()
        let worker = FrameMetadataWorker()
        let returned = try await worker.inspect(expected)
        precondition(returned == expected)
    }

    /// Supplies valid defaults while varying individual source metadata fields.
    private static func makeMetadata(
        sourceID: String = "fixture-video-track", index: Int = 60, loop: Int = 0,
        time: CMTime = CMTime(value: 1001, timescale: 30000),
        width: Int = 16, height: Int = 8,
        transform: CGAffineTransform = .identity
    ) throws -> VideoFrameMetadata {
        try VideoFrameMetadata(
            sourceID: sourceID, sourceFrameIndex: index, presentationTime: time,
            width: width, height: height, preferredTransform: transform,
            loopIndex: loop)
    }

    /// Requires the expected contract failure rather than any unrelated error.
    private static func expectFailure(
        _ expected: VideoFrameMetadata.ValidationError, operation: () throws -> Void
    ) throws {
        do {
            try operation()
            preconditionFailure("Invalid video metadata was accepted: \(expected)")
        } catch let error as VideoFrameMetadata.ValidationError {
            precondition(error == expected)
        }
    }
}

/// Builds a frame on its owning worker without sending the mutable buffer.
private actor FrameMetadataWorker {
    /// Verifies the received metadata against a newly created worker-local frame.
    func inspect(_ metadata: VideoFrameMetadata) throws -> VideoFrameMetadata {
        let frame = try makeFrame()
        precondition(frame.metadata == metadata)
        return frame.metadata
    }
}

/// Releases the original sample and local buffer references before returning a frame.
nonisolated private func makeFrame(
    transform: CGAffineTransform = .identity
) throws -> VideoFrame {
    var buffer: CVPixelBuffer?
    precondition(
        CVPixelBufferCreate(kCFAllocatorDefault, 16, 8, kCVPixelFormatType_32BGRA,
                            nil, &buffer) == kCVReturnSuccess)
    let pixels = buffer!
    precondition(CVPixelBufferLockBaseAddress(pixels, []) == kCVReturnSuccess)
    CVPixelBufferGetBaseAddress(pixels)!.initializeMemory(
        as: UInt8.self, repeating: 73,
        count: CVPixelBufferGetBytesPerRow(pixels) * CVPixelBufferGetHeight(pixels))
    CVPixelBufferUnlockBaseAddress(pixels, [])
    var format: CMVideoFormatDescription?
    precondition(
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault, imageBuffer: pixels,
            formatDescriptionOut: &format) == noErr)
    var timing = CMSampleTimingInfo(
        duration: CMTime(value: 1001, timescale: 30000),
        presentationTimeStamp: CMTime(value: 1001, timescale: 30000),
        decodeTimeStamp: .invalid)
    var sample: CMSampleBuffer?
    precondition(
        CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault, imageBuffer: pixels,
            formatDescription: format!, sampleTiming: &timing,
            sampleBufferOut: &sample) == noErr)
    return try VideoFrame(
        sourceID: "fixture-video-track", sourceFrameIndex: 60,
        presentationTime: CMSampleBufferGetPresentationTimeStamp(sample!),
        preferredTransform: transform,
        pixelBuffer: CMSampleBufferGetImageBuffer(sample!)!)
}
