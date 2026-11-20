import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation

/// Checks real asymmetric pixels rather than only transformed image dimensions.
@main
struct VideoFrameConversionSmokeTests {
    /// Covers rotations, mirrors, translations, snapshot lifetime and box mapping.
    static func main() async throws {
        let worker = ConversionWorker()
        let cases: [(CGAffineTransform, [UInt8], Int, Int)] = [
            (.identity, [10, 20, 30, 40, 50, 60], 3, 2),
            (CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 2, ty: 0),
             [40, 10, 50, 20, 60, 30], 2, 3),
            (CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: 3, ty: 2),
             [60, 50, 40, 30, 20, 10], 3, 2),
            (CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: 3),
             [30, 60, 20, 50, 10, 40], 2, 3),
            (CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: 3, ty: 0),
             [30, 20, 10, 60, 50, 40], 3, 2),
            (CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: 2),
             [40, 50, 60, 10, 20, 30], 3, 2),
            (CGAffineTransform(a: 0, b: 1, c: 1, d: 0, tx: 0, ty: 0),
             [10, 40, 20, 50, 30, 60], 2, 3),
            (CGAffineTransform(a: 0, b: -1, c: -1, d: 0, tx: 2, ty: 3),
             [60, 30, 50, 20, 40, 10], 2, 3),
            (CGAffineTransform(translationX: -17, y: 9),
             [10, 20, 30, 40, 50, 60], 3, 2),
            (CGAffineTransform(rotationAngle: .pi / 2),
             [40, 10, 50, 20, 60, 30], 2, 3),
            (CGAffineTransform(rotationAngle: -.pi / 2),
             [30, 60, 20, 50, 10, 40], 2, 3),
        ]
        for (transform, expected, width, height) in cases {
            let frame = try await worker.convert(transform)
            let rgb = try RGBImage(frame.image)
            precondition(rgb.width == width && rgb.height == height)
            for index in expected.indices {
                precondition(rgb.pixels[index * 3] == expected[index],
                             "Wrong orientation: \(transform), \(rgb.pixels)")
                precondition(rgb.pixels[index * 3 + 1] == 100)
                precondition(rgb.pixels[index * 3 + 2] == 200)
            }
            let box = CGRect(x: 0, y: 0, width: 1, height: 1)
            let upright = box.applying(frame.bufferToImage)
            let restored = upright.applying(frame.bufferToImage.inverted())
            precondition(abs(restored.minX - box.minX) < 1e-9)
            precondition(abs(restored.minY - box.minY) < 1e-9)
            precondition(abs(restored.width - box.width) < 1e-9)
            precondition(abs(restored.height - box.height) < 1e-9)
            precondition(frame.metadata.sourceID == "conversion-fixture")
            precondition(frame.metadata.presentationTime.value == 1001)
            precondition(frame.metadata.presentationTime.timescale == 30000)
        }
        try await worker.rejectUnsupportedPixels()
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try await worker.convert(.identity)
        }
        do {
            try await cancelled.value
            preconditionFailure("Cancelled conversion succeeded")
        } catch is CancellationError {}
        print("Video conversion: \(cases.count) orientations, RGB and geometry passed.")
    }
}

/// Keeps test pixel buffers on a worker while immutable images cross to the caller.
private actor ConversionWorker {
    /// Converts padded BGRA rows and then overwrites their original storage.
    func convert(_ transform: CGAffineTransform) throws -> UprightVideoFrame {
        let buffer = makeBuffer(kCVPixelFormatType_32BGRA)
        precondition(CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess)
        let base = CVPixelBufferGetBaseAddress(buffer)!
            .assumingMemoryBound(to: UInt8.self)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<2 {
            for x in 0..<3 {
                let offset = y * rowBytes + x * 4
                base[offset] = 200
                base[offset + 1] = 100
                base[offset + 2] = UInt8(10 * (y * 3 + x + 1))
                base[offset + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        let frame = try VideoFrame(
            sourceID: "conversion-fixture", sourceFrameIndex: 17,
            presentationTime: CMTime(value: 1001, timescale: 30000),
            preferredTransform: transform, pixelBuffer: buffer)
        let upright = try VideoFrameConverter.convert(frame)
        CVPixelBufferLockBaseAddress(buffer, [])
        base.initialize(repeating: 0, count: rowBytes * 2)
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return upright
    }

    /// Requires an explicit error when a producer bypasses the BGRA reader contract.
    func rejectUnsupportedPixels() throws {
        let frame = try VideoFrame(
            sourceID: "bad-format", sourceFrameIndex: 0, presentationTime: .zero,
            pixelBuffer: makeBuffer(kCVPixelFormatType_32ARGB))
        do {
            _ = try VideoFrameConverter.convert(frame)
            preconditionFailure("Unsupported pixel format succeeded")
        } catch DetectorError.invalidImage {}
    }

    /// Creates a small buffer whose row padding must not become image pixels.
    private func makeBuffer(_ format: OSType) -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        precondition(CVPixelBufferCreate(
            kCFAllocatorDefault, 3, 2, format, nil, &buffer) == kCVReturnSuccess)
        return buffer!
    }
}
