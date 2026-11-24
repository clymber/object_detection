import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Exercises real MP4 encoding, annotation pixels, source timing and bounded overload.
@main
@MainActor
struct CameraRecordingSmokeTests {
    /// Writes diagnostic artifacts outside the source tree and decodes them again.
    static func main() async throws {
        let directory = URL(fileURLWithPath:
            "outputs/ios_builds/video_pipeline/camera_recordings")
        try? FileManager.default.removeItem(at: directory)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try testAnnotationPixels(directory)
        try await testMovie(directory)
        try await testEmptyAndFailure(directory)
        try await testBoundedAdmission(directory)
        try await testCancelledFinish(directory)
        print("Camera recording: annotations, MP4 readback, timing, overload "
              + "and cancellation passed")
    }

    /// Checks top-left box coordinates, unchanged source pixels and visible label text.
    private static func testAnnotationPixels(_ directory: URL) throws {
        let frame = try fixture(index: 0, pts: 100)
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 640, 360, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferCGImageCompatibilityKey: true,
             kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary,
            &buffer)
        precondition(status == kCVReturnSuccess)
        try CameraAnnotationRenderer.draw(frame, into: buffer!)
        let image = try VideoFrameConverter.convert(VideoFrame(
            sourceID: "rendered", sourceFrameIndex: 0,
            presentationTime: .zero, pixelBuffer: buffer!)).image
        let rgb = try RGBImage(image)
        precondition(isRed(rgb, x: 80, y: 110), "Box was not at its source coordinates")
        precondition(!isRed(rgb, x: 80, y: 250), "Box was vertically mirrored")
        precondition(channel(rgb, x: 400, y: 20, c: 2) > 240)
        precondition(channel(rgb, x: 400, y: 340, c: 1) > 240)
        var whiteTextPixels = 0
        for y in 50..<78 {
            for x in 80..<250 {
                if (0..<3).allSatisfy({ channel(rgb, x: x, y: y, c: $0) > 200 }) {
                    whiteTextPixels += 1
                }
            }
        }
        precondition(whiteTextPixels > 30, "Class/confidence label is missing")
        let output = directory.appendingPathComponent("annotations.png")
        let destination = CGImageDestinationCreateWithURL(
            output as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image, nil)
        precondition(CGImageDestinationFinalize(destination))
    }

    /// Preserves capture gaps and empty detections in a silent H.264 movie.
    private static func testMovie(_ directory: URL) async throws {
        let recorder = CameraVideoRecorder(framesPerSecond: 30, directory: directory)
        for (index, time) in [100, 105, 120].enumerated() {
            recorder.offer(try fixture(index: index, pts: Int64(time)))
            // Pace this encoding check; overload is covered independently below.
            try await Task.sleep(for: .milliseconds(150))
        }
        let report = await recorder.finish()
        precondition(report.status == .completed, report.issue?.message ?? "No MP4")
        precondition(report.encodedFrames == 3 && report.skippedFrames == 0)
        precondition(report.firstSourceTime?.epoch == 7)
        let url = report.url!
        precondition(url.pathExtension == "mp4")
        let asset = AVURLAsset(url: url)
        let audio = try await asset.loadTracks(withMediaType: .audio)
        precondition(audio.isEmpty)
        let video = try await asset.loadTracks(withMediaType: .video).first!
        let formats = try await video.load(.formatDescriptions)
        precondition(
            CMFormatDescriptionGetMediaSubType(formats[0]) == kCMVideoCodecType_H264)
        let duration = try await asset.load(.duration)
        precondition(abs(duration.seconds - 21.0 / 30) < 0.01)
        let reader = VideoFrameReader()
        try await reader.open(url, sourceID: "recorded")
        for (index, expected) in [0.0, 5.0 / 30, 20.0 / 30].enumerated() {
            let decoded = try await reader.nextFrame { frame in
                (frame.metadata, try RGBImage(VideoFrameConverter.convert(frame).image))
            }!
            precondition(decoded.0.width == 640 && decoded.0.height == 360)
            precondition(abs(decoded.0.presentationTime.seconds - expected) < 0.001)
            precondition(isRed(decoded.1, x: 80, y: 110) == (index != 1))
            precondition(channel(decoded.1, x: 400, y: 20, c: 2) > 200)
        }
        let eof = try await reader.nextFrame { $0.metadata }
        precondition(eof == nil)
    }

    /// Empty and unwritable runs expose no partial MP4 and retain failure details.
    private static func testEmptyAndFailure(_ directory: URL) async throws {
        let empty = CameraVideoRecorder(framesPerSecond: 30, directory: directory)
        let emptyReport = await empty.finish()
        precondition(emptyReport.status == .empty && emptyReport.url == nil)
        let blocked = directory.appendingPathComponent("regular_file")
        try Data([0]).write(to: blocked)
        let failed = CameraVideoRecorder(framesPerSecond: 30, directory: blocked)
        failed.offer(try fixture(index: 0, pts: 100))
        let failure = await failed.finish()
        precondition(failure.status == .failed && failure.issue != nil)
        precondition(failure.url == nil && failure.skippedFrames == 1)
    }

    /// A stalled encoder retains two frames and accounts for every rejected input.
    private static func testBoundedAdmission(_ directory: URL) async throws {
        let queue = DispatchQueue(label: "test.camera.recording.stalled")
        queue.suspend()
        let recorder = CameraVideoRecorder(
            framesPerSecond: 30, directory: directory, queue: queue)
        for index in 0..<100 {
            recorder.offer(try fixture(index: index, pts: Int64(100 + index)))
        }
        queue.resume()
        let report = await recorder.finish()
        precondition(
            report.status == .completed, report.issue?.message ?? "Overload failed")
        precondition(report.encodedFrames <= 2 && report.skippedFrames >= 98)
        precondition(report.encodedFrames + report.skippedFrames == 100)
        recorder.offer(try fixture(index: 101, pts: 201))
    }

    /// Stop's cancelled task drains accepted frames and finishes a playable file.
    private static func testCancelledFinish(_ directory: URL) async throws {
        let recorder = CameraVideoRecorder(framesPerSecond: 30, directory: directory)
        recorder.offer(try fixture(index: 0, pts: 100))
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return await recorder.finish()
        }
        let report = await task.value
        precondition(report.status == .completed && report.encodedFrames == 1)
        let asset = AVURLAsset(url: report.url!)
        let duration = try await asset.load(.duration)
        precondition(abs(duration.seconds - 1.0 / 30) < 0.01)
    }

    /// Builds an asymmetric upright image with one basketball box or no detections.
    private static func fixture(index: Int, pts: Int64) throws -> VideoFramePreview {
        let width = 640
        let height = 360
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                bytes[offset + (y < height / 2 ? 2 : 1)] = 255
                bytes[offset + 3] = 255
            }
        }
        let image = CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: CGDataProvider(data: Data(bytes) as CFData)!, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent)!
        let metadata = try VideoFrameMetadata(
            sourceID: "camera-fixture", sourceFrameIndex: index,
            presentationTime: CMTime(
                value: pts, timescale: 30, flags: .valid, epoch: 7),
            width: width, height: height)
        let frame = UprightVideoFrame(
            metadata: metadata, image: image, bufferToImage: .identity,
            readyMilliseconds: 0)
        let detections = index == 1 ? [] : [try Detection(
            boundingBox: CGRect(x: 80, y: 80, width: 120, height: 60),
            confidence: 0.9, classID: 1)]
        return try VideoFramePreview(frame: frame, result: VideoFrameResult(
            metadata: metadata, sourceWidth: width, sourceHeight: height,
            bufferToImage: .identity, status: .completed, detections: detections,
            failure: nil, timings: VideoFrameTimings(
                readyMilliseconds: 0, modelTiming: nil, doneMilliseconds: 1,
                publishedMilliseconds: nil)))
    }

    /// Reads one top-left RGB component for lossless and compressed pixel checks.
    private static func channel(_ image: RGBImage, x: Int, y: Int, c: Int) -> UInt8 {
        image.pixels[(y * image.width + x) * 3 + c]
    }

    /// Allows compression variation while requiring a clearly red outline pixel.
    private static func isRed(_ image: RGBImage, x: Int, y: Int) -> Bool {
        channel(image, x: x, y: y, c: 0) > 150
            && channel(image, x: x, y: y, c: 1) < 100
            && channel(image, x: x, y: y, c: 2) < 100
    }
}
