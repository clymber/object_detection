import CoreGraphics
import CoreMedia
import CoreML
import CoreVideo
import Foundation
import ImageIO

/// Exercises converted BGRA through all three actual Core ML adapters on Mac CPU.
@main
struct VideoFrameModelsSmokeTests {
    /// Compares video detection to direct shared inference on identical upright pixels.
    static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let source = root.appendingPathComponent(
            "ios/ENABLECoreMLBench/ENABLECoreMLBench")
        let frame = try await ModelFrameFixture().load(
            source.appendingPathComponent(
                "Resources/TestImages/test_positive_6988.jpg"))
        let urls = Dictionary(uniqueKeysWithValues: DetectorModel.allCases.map {
            ($0, source.appendingPathComponent("Models/\($0.rawValue).mlpackage"))
        })
        let engine = DetectorEngine(modelURLs: urls, computeUnits: .cpuOnly)
        let detector = VideoFrameDetector(detector: engine)
        for model in DetectorModel.allCases {
            try await engine.selectModel(model)
            let candidates = try await engine.detect(in: frame.image)
            let expected = candidates.enumerated()
                .filter { $0.element.confidence >= 0.25 }
                .sorted {
                    $0.element.confidence == $1.element.confidence
                        ? $0.offset < $1.offset
                        : $0.element.confidence > $1.element.confidence
                }.prefix(100).map(\.element)
            let actual = try await detector.detect(frame)
            precondition(!actual.isEmpty && actual == expected)
            precondition(actual.allSatisfy { $0.classID == 1 })
            // Preloaded input becomes ready for this request after model selection.
            let ready = UprightVideoFrame(
                metadata: frame.metadata, image: frame.image,
                bufferToImage: frame.bufferToImage)
            let record = try await detector.process(ready)
            precondition(record.status == .completed && record.detections == expected)
            precondition(record.timings.modelLatencyMilliseconds! > 0)
            precondition(record.processingDurationMilliseconds
                         >= record.timings.modelLatencyMilliseconds!)
            let delivered = try record.published()
            precondition(delivered.timings.publishedMilliseconds != nil)
            print("Video model integration: \(model.rawValue) passed on CPU.")
        }
    }
}

/// Creates worker-local BGRA from the frozen image before crossing actor boundaries.
private actor ModelFrameFixture {
    /// Reproduces decoded RGB bytes and passes them through the actual converter.
    func load(_ url: URL) throws -> UprightVideoFrame {
        let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)!
        let rgb = try RGBImage(image)
        var optional: CVPixelBuffer?
        precondition(CVPixelBufferCreate(
            kCFAllocatorDefault, rgb.width, rgb.height, kCVPixelFormatType_32BGRA,
            nil, &optional) == kCVReturnSuccess)
        let pixels = optional!
        CVPixelBufferLockBaseAddress(pixels, [])
        let base = CVPixelBufferGetBaseAddress(pixels)!
            .assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(pixels)
        for y in 0..<rgb.height {
            for x in 0..<rgb.width {
                let input = (y * rgb.width + x) * 3
                let output = y * stride + x * 4
                base[output] = rgb.pixels[input + 2]
                base[output + 1] = rgb.pixels[input + 1]
                base[output + 2] = rgb.pixels[input]
                base[output + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(pixels, [])
        let frame = try VideoFrame(
            sourceID: "frozen-6988", sourceFrameIndex: 0, presentationTime: .zero,
            pixelBuffer: pixels)
        let upright = try VideoFrameConverter.convert(frame)
        let decoded = try RGBImage(upright.image)
        precondition(decoded.pixels == rgb.pixels)
        return upright
    }
}
