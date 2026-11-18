import CoreGraphics
import Foundation

/// Compares all tensor elements against independently generated Python fixtures.
@main
struct PreprocessingSmokeTests {
    /// Accepts the fixture directory and checks channel order, resize and padding.
    static func main() throws {
        let directory = URL(fileURLWithPath: CommandLine.arguments[1])
        let cases =
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: directory.appendingPathComponent("preprocessing.json"))
            ) as! [[String: Any]]
        for entry in cases {
            let width = entry["width"] as! Int
            let height = entry["height"] as! Int
            let bytes = try Data(
                contentsOf: directory.appendingPathComponent(entry["rgb"] as! String))
            let source = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: CGDataProvider(data: bytes as CFData)!, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent
            )!
            let image = try RGBImage(source)
            precondition(image.pixels == Array(bytes), "RGB decoding changed pixels")
            let model = DetectorModel(rawValue: entry["model"] as! String)!
            let prepared = try ImagePreprocessor.prepare(image, for: model)
            let expected = try Data(
                contentsOf: directory.appendingPathComponent(entry["tensor"] as! String)
            )
            let maximum = expected.withUnsafeBytes { buffer in
                prepared.tensor.values.enumerated().reduce(Float(0)) { error, pair in
                    max(
                        error,
                        abs(
                            pair.element
                                - buffer.loadUnaligned(
                                    fromByteOffset: pair.offset * 4, as: Float.self
                                )))
                }
            }
            let message =
                "\(model.rawValue), \(width)x\(height): maximum tensor error \(maximum)\n"
            FileHandle.standardOutput.write(Data(message.utf8))
            try prepared.tensor.values.withUnsafeBytes { buffer in
                try Data(buffer).write(
                    to: directory.appendingPathComponent(
                        (entry["tensor"] as! String) + ".swift"
                    ))
            }
            // Allow the observed ARM HAL difference only for downscaling.
            // This byte-level bound does not relax detection box/score tolerances.
            let downscaling = width > 640 || height > 640
            let tolerance: Float = model.isRFDETR ? 0.000001 : (downscaling ? 2 : 0)
            precondition(maximum <= tolerance, "Python preprocessing differs")
            if !model.isRFDETR && width != height {
                for channel in 0..<3 {
                    precondition(
                        prepared.tensor.values[(channel + 1) * 640 * 640 - 1] == 114)
                }
            }
        }
    }
}
