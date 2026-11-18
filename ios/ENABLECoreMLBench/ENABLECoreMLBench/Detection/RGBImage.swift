import CoreGraphics

/// Upright, interleaved eight-bit RGB pixels before model-specific transforms.
nonisolated struct RGBImage: Sendable {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    /// Validates explicitly decoded RGB storage.
    init(width: Int, height: Int, pixels: [UInt8]) throws {
        let area = width.multipliedReportingOverflow(by: height)
        let count = area.partialValue.multipliedReportingOverflow(by: 3)
        guard width > 0, height > 0, !area.overflow, !count.overflow,
            pixels.count == count.partialValue
        else {
            throw DetectorError.invalidImage("RGB dimensions and byte count differ")
        }
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    /// Renders an orientation-resolved CGImage into opaque RGB pixels.
    init(_ image: CGImage) throws {
        let width = image.width
        let height = image.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        try rgba.withUnsafeMutableBytes { buffer in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height,
                    bitsPerComponent: 8, bytesPerRow: width * 4,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                        | CGBitmapInfo.byteOrder32Big.rawValue
                )
            else { throw DetectorError.invalidImage("cannot decode RGB pixels") }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            for channel in 0..<3 {
                rgb[pixel * 3 + channel] = rgba[pixel * 4 + channel]
            }
        }
        try self.init(width: width, height: height, pixels: rgb)
    }
}
