import CoreGraphics
import CoreVideo
import Foundation

/// An immutable upright source image, independent of the decoder's mutable pixels.
nonisolated struct UprightVideoFrame: Sendable {
    let metadata: VideoFrameMetadata
    let image: CGImage

    /// Maps encoded buffer coordinates to top-left upright source-image coordinates.
    let bufferToImage: CGAffineTransform
    let readyMilliseconds: Double

    /// Allows preloaded inputs to receive a new ready time for each timed submission.
    init(
        metadata: VideoFrameMetadata, image: CGImage, bufferToImage: CGAffineTransform,
        readyMilliseconds: Double = InferenceClock.nowMilliseconds()
    ) {
        self.metadata = metadata
        self.image = image
        self.bufferToImage = bufferToImage
        self.readyMilliseconds = readyMilliseconds
    }
}

/// Snapshots decoded BGRA and applies source presentation geometry on the worker.
nonisolated enum VideoFrameConverter {
    /// Produces upright RGB-compatible pixels without model resizing or filtering.
    static func convert(_ frame: VideoFrame) throws -> UprightVideoFrame {
        try Task.checkCancellation()
        let pixels = frame.pixelBuffer
        guard CVPixelBufferGetPixelFormatType(pixels) == kCVPixelFormatType_32BGRA,
            !CVPixelBufferIsPlanar(pixels)
        else {
            throw DetectorError.invalidImage("Video conversion requires BGRA pixels")
        }
        let width = CVPixelBufferGetWidth(pixels)
        let height = CVPixelBufferGetHeight(pixels)
        let rowBytes = CVPixelBufferGetBytesPerRow(pixels)
        let storage = rowBytes.multipliedReportingOverflow(by: height)
        guard !storage.overflow,
            CVPixelBufferLockBaseAddress(pixels, .readOnly) == kCVReturnSuccess
        else { throw DetectorError.invalidImage("Cannot lock decoded video pixels") }
        let snapshot: Data
        do {
            defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(pixels) else {
                throw DetectorError.invalidImage("Missing decoded video pixels")
            }
            snapshot = Data(bytes: base, count: storage.partialValue)
        }
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        let transformed = bounds.applying(frame.metadata.preferredTransform)
        guard transformed.width.isFinite, transformed.height.isFinite,
            transformed.width > 0, transformed.height > 0,
            transformed.width < CGFloat(Int.max), transformed.height < CGFloat(Int.max)
        else { throw DetectorError.invalidImage("Invalid upright video dimensions") }
        let uprightWidth = pixelDimension(transformed.width)
        let uprightHeight = pixelDimension(transformed.height)
        let transform = frame.metadata.preferredTransform.concatenating(
            CGAffineTransform(translationX: -transformed.minX, y: -transformed.minY))
        let space = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: snapshot as CFData),
            let source = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: rowBytes, space: space,
                bitmapInfo: CGBitmapInfo(
                    rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue)
                    .union(.byteOrder32Little),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent),
            let context = CGContext(
                data: nil, width: uprightWidth, height: uprightHeight,
                bitsPerComponent: 8, bytesPerRow: 0, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue)
        else { throw DetectorError.invalidImage("Cannot prepare upright video image") }
        context.interpolationQuality = .none
        context.setShouldAntialias(false)
        context.setBlendMode(.copy)
        // Track transforms use top-left image coordinates; Quartz draws bottom-up.
        let sourceFlip = CGAffineTransform(a: 1, b: 0, c: 0, d: -1,
                                          tx: 0, ty: CGFloat(height))
        let destinationFlip = CGAffineTransform(a: 1, b: 0, c: 0, d: -1,
                                               tx: 0, ty: CGFloat(uprightHeight))
        context.concatenate(
            sourceFlip.concatenating(transform).concatenating(destinationFlip))
        context.draw(source, in: bounds)
        guard let image = context.makeImage() else {
            throw DetectorError.invalidImage("Cannot snapshot upright video image")
        }
        try Task.checkCancellation()
        return UprightVideoFrame(
            metadata: frame.metadata, image: image, bufferToImage: transform,
            readyMilliseconds: frame.readyMilliseconds)
    }

    /// Avoids an extra pixel from floating-point residue at exact quarter turns.
    private static func pixelDimension(_ extent: CGFloat) -> Int {
        let nearest = extent.rounded()
        return Int(abs(extent - nearest) <= 1e-7 ? max(1, nearest) : ceil(extent))
    }
}
