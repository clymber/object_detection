import Foundation

/// Tensor input plus the source geometry needed to restore decoded boxes.
nonisolated struct PreparedImage: Sendable {
    let tensor: FloatTensor
    let sourceWidth: Int
    let sourceHeight: Int
    let scale: Float
}

/// Reproduces the model-owned Python preprocessing without UI dependencies.
nonisolated enum ImagePreprocessor {
    /// Converts upright RGB pixels into a fixed batch-one NCHW FP32 tensor.
    static func prepare(_ image: RGBImage, for model: DetectorModel) throws
        -> PreparedImage
    {
        let size = model.inputShape[2]
        let scale = min(
            Double(size) / Double(image.width), Double(size) / Double(image.height))
        let width = model.isRFDETR ? size : max(1, Int(Double(image.width) * scale))
        let height = model.isRFDETR ? size : max(1, Int(Double(image.height) * scale))
        var values = [Float](
            repeating: model.isRFDETR ? 0 : 114, count: 3 * size * size)
        let mean: [Float] = [0.485, 0.456, 0.406]
        let deviation: [Float] = [0.229, 0.224, 0.225]
        for y in 0..<height {
            if y % 32 == 0 { try Task.checkCancellation() }
            for x in 0..<width {
                for channel in 0..<3 {
                    let value: Float
                    if model.isRFDETR {
                        value =
                            (rfdetrPixel(
                                image, x: x, y: y, size: size, channel: channel)
                                - mean[channel]) / deviation[channel]
                    } else {
                        value = Float(
                            yoloxPixel(
                                image, x: x, y: y, width: width, height: height,
                                channel: 2 - channel
                            ))
                    }
                    values[channel * size * size + y * size + x] = value
                }
            }
        }
        return PreparedImage(
            tensor: try FloatTensor(shape: model.inputShape, values: values),
            sourceWidth: image.width, sourceHeight: image.height, scale: Float(scale)
        )
    }

    /// Matches the FP32 half-pixel coordinates and bilinear RF-DETR reference.
    private static func rfdetrPixel(
        _ image: RGBImage, x: Int, y: Int, size: Int, channel: Int
    ) -> Float {
        let sx = min(
            max((Float(x) + 0.5) * (Float(image.width) / Float(size)) - 0.5, 0),
            Float(image.width - 1))
        let sy = min(
            max((Float(y) + 0.5) * (Float(image.height) / Float(size)) - 0.5, 0),
            Float(image.height - 1))
        let x0 = Int(sx)
        let y0 = Int(sy)
        let x1 = min(x0 + 1, image.width - 1)
        let y1 = min(y0 + 1, image.height - 1)
        let dx = Double(sx) - Double(x0)
        let dy = Double(sy) - Double(y0)

        /// Normalizes source bytes before interpolation, matching the reference.
        func sample(_ px: Int, _ py: Int) -> Double {
            Double(Float(image.pixels[(py * image.width + px) * 3 + channel]) / 255)
        }

        let top = (1 - dx) * sample(x0, y0) + dx * sample(x1, y0)
        let bottom = (1 - dx) * sample(x0, y1) + dx * sample(x1, y1)
        return Float((1 - dy) * top + dy * bottom)
    }

    /// Uses portable OpenCV uint8 bilinear arithmetic with fixed-point weights.
    /// OpenCV 5's ARM HAL can differ by two byte levels when downscaling; retain
    /// prediction-parity checks rather than assuming platform-independent pixels.
    private static func yoloxPixel(
        _ image: RGBImage, x: Int, y: Int, width: Int, height: Int, channel: Int
    ) -> UInt8 {
        let sx = max(
            0,
            min(
                Float(
                    (Double(x) + 0.5) * Double(image.width) / Double(width) - 0.5
                ), Float(image.width - 1)))
        let sy = Float((Double(y) + 0.5) * Double(image.height) / Double(height) - 0.5)
        let baseY = Int(floor(sy))
        let x0 = Int(sx)
        let y0 = min(max(baseY, 0), image.height - 1)
        let x1 = min(x0 + 1, image.width - 1)
        let y1 = min(max(baseY + 1, 0), image.height - 1)
        let dx = sx - Float(x0)
        let dy = sy - Float(baseY)
        let ax = Int((dx * 2048).rounded(.toNearestOrEven))
        let bx = Int(((1 - dx) * 2048).rounded(.toNearestOrEven))
        let ay = Int((dy * 2048).rounded(.toNearestOrEven))
        let by = Int(((1 - dy) * 2048).rounded(.toNearestOrEven))

        /// Reads one source channel before the BGR reorder.
        func sample(_ px: Int, _ py: Int) -> Int {
            Int(image.pixels[(py * image.width + px) * 3 + channel])
        }

        let top = bx * sample(x0, y0) + ax * sample(x1, y0)
        let bottom = bx * sample(x0, y1) + ax * sample(x1, y1)
        // Match OpenCV's two-stage uint8 vertical rounding, including borders.
        let topQuarter = (by * (top >> 4)) >> 16
        let bottomQuarter = (ay * (bottom >> 4)) >> 16
        let value = (topQuarter + bottomQuarter + 2) >> 2
        return UInt8(clamping: value)
    }
}
