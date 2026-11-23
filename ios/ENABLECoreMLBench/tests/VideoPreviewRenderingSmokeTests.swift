import CoreGraphics
import CoreMedia
import Foundation
import SwiftUI

/// Rasterizes the real preview to verify box/image alignment and draw callbacks.
@main
@MainActor
struct VideoPreviewRenderingSmokeTests {
    /// Exercises both letterbox axes and confirms only actual drawing is counted.
    static func main() async throws {
        try await check(
            width: 160, height: 80,
            box: CGRect(x: 40, y: 20, width: 40, height: 20),
            border: CGPoint(x: 80, y: 110), interior: CGPoint(x: 120, y: 120),
            padding: CGPoint(x: 100, y: 30))
        try await check(
            width: 80, height: 160,
            box: CGRect(x: 20, y: 40, width: 20, height: 40),
            border: CGPoint(x: 125, y: 85), interior: CGPoint(x: 140, y: 100),
            padding: CGPoint(x: 20, y: 100))
        try checkFullScreen(
            width: 160, height: 80, viewport: CGSize(width: 320, height: 480),
            box: CGRect(x: 40, y: 20, width: 40, height: 20),
            border: CGPoint(x: 80, y: 210), interior: CGPoint(x: 120, y: 220),
            labelRegion: CGRect(x: 80, y: 160, width: 160, height: 40))
        try checkFullScreen(
            width: 80, height: 160, viewport: CGSize(width: 480, height: 320),
            box: CGRect(x: 20, y: 40, width: 20, height: 40),
            border: CGPoint(x: 200, y: 100), interior: CGPoint(x: 220, y: 120),
            labelRegion: CGRect(x: 160, y: 0, width: 160, height: 80))
        print(
            "Video rendering: inline/full-screen geometry, labels and draw receipts passed.")
    }

    /// Verifies full-size rendering and white-on-red text above boxes in both orientations.
    private static func checkFullScreen(
        width: Int, height: Int, viewport: CGSize, box: CGRect,
        border: CGPoint, interior: CGPoint, labelRegion: CGRect
    ) throws {
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.setFillColor(CGColor(red: 0, green: 0, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let detection = try Detection(boundingBox: box, confidence: 0.9, classID: 1)
        let view = FramePreview(
            image: context.makeImage()!, detections: [detection], imageTitle: "Fixture",
            presentation: .fullScreen
        ).frame(width: viewport.width, height: viewport.height)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 1
        guard let image = renderer.cgImage else {
            preconditionFailure("Full-screen preview could not be rasterized")
        }
        let rgb = try RGBImage(image)
        let red = pixel(rgb, at: border)
        let blue = pixel(rgb, at: interior)
        precondition(red[0] > 180 && red[2] < 80, "Full-screen box is misaligned")
        precondition(blue[2] > 180 && blue[0] < 80, "Full-screen image was resized wrongly")
        var whitePixels = 0
        var redPixels = 0
        for y in Int(labelRegion.minY)..<Int(labelRegion.maxY) {
            for x in Int(labelRegion.minX)..<Int(labelRegion.maxX) {
                let value = pixel(rgb, at: CGPoint(x: x, y: y))
                if value.allSatisfy({ $0 > 200 }) { whitePixels += 1 }
                if value[0] > 180 && value[1] < 80 && value[2] < 80 { redPixels += 1 }
            }
        }
        precondition(whitePixels > 20, "Class/confidence text is missing above the box")
        precondition(redPixels > 100, "Label background is missing above the box")
    }

    /// Checks independently calculated display coordinates in a 320-by-280 image area.
    private static func check(
        width: Int, height: Int, box: CGRect, border: CGPoint,
        interior: CGPoint, padding: CGPoint
    ) async throws {
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.setFillColor(CGColor(red: 0, green: 0, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let received = DrawReceipt()
        let detection = try Detection(boundingBox: box, confidence: 0.9, classID: 1)
        let view = FramePreview(
            image: context.makeImage()!, detections: [detection], imageTitle: "Fixture",
            didDraw: { received.times.append($0) }).frame(width: 320)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 1
        guard let image = renderer.cgImage else {
            preconditionFailure("SwiftUI preview could not be rasterized")
        }
        let rgb = try RGBImage(image)
        let red = pixel(rgb, at: border)
        let blue = pixel(rgb, at: interior)
        let blank = pixel(rgb, at: padding)
        precondition(red[0] > 180 && red[2] < 80, "Box outline is misaligned")
        precondition(blue[2] > 180 && blue[0] < 80, "Source pixels changed")
        precondition(blank[2] <= blank[0] + 30, "Aspect fit lost its padding")
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while received.times.isEmpty && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(2))
        }
        precondition(!received.times.isEmpty, "Drawing never acknowledged the frame")
    }

    /// Reads top-left RGB coordinates without reusing production preview geometry.
    private static func pixel(_ image: RGBImage, at point: CGPoint) -> [Int] {
        let offset = (Int(point.y) * image.width + Int(point.x)) * 3
        return image.pixels[offset..<offset + 3].map(Int.init)
    }
}

/// Captures deferred renderer callbacks on the same UI actor as the session.
@MainActor
private final class DrawReceipt {
    var times: [Double] = []
}
