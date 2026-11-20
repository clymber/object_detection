import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Exercises real file imports, image orientation and video metadata without pickers.
@main
@MainActor
struct MediaImportSmokeTests {
    /// Generates private fixtures and checks imports, failures and input replacement.
    static func main() async throws {
        let directory = URL(
            fileURLWithPath: "outputs/ios_builds/still_image/media_import")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let uprightURL = directory.appendingPathComponent("upright.jpg")
        let rotatedURL = directory.appendingPathComponent("rotated.jpg")
        try writePhoto(to: uprightURL, orientation: 1)
        try writePhoto(to: rotatedURL, orientation: 6)
        let upright = try StillImagePhotoDecoder.load(uprightURL)
        let rotated = try StillImagePhotoDecoder.load(rotatedURL)
        precondition(upright.width == 32 && upright.height == 16)
        precondition(rotated.width == 16 && rotated.height == 32)
        let rgb = try RGBImage(rotated)
        let top = (4 * rgb.width + 8) * 3
        let bottom = (28 * rgb.width + 8) * 3
        precondition(rgb.pixels[top] > 200 && rgb.pixels[top + 2] < 50)
        precondition(rgb.pixels[bottom] < 50 && rgb.pixels[bottom + 2] > 200)

        let videoURL = directory.appendingPathComponent("fixture.mov")
        try await writeVideo(to: videoURL)
        let worker = MediaImportWorker(
            directory: directory.appendingPathComponent("cache"))
        let session = MediaImportSession(worker: worker)
        session.importFile(rotatedURL, as: .photo)
        try await waitForIdle(session)
        let photo = session.photo!
        precondition(photo.url != rotatedURL && session.video == nil)
        let copiedBytes = try Data(contentsOf: photo.url)
        let originalBytes = try Data(contentsOf: rotatedURL)
        precondition(copiedBytes == originalBytes)
        await worker.remove(ImportedMediaFile(url: rotatedURL))
        precondition(FileManager.default.fileExists(atPath: rotatedURL.path))

        let detector = StillImageSession(worker: PhotoWorker())
        detector.loadPhoto(photo.url)
        try await waitForImage(detector)
        precondition(detector.image?.width == 16 && detector.image?.height == 32)
        detector.start()
        try await waitForImage(detector)
        precondition(detector.status == .completed && detector.processedImages == 1)
        precondition(
            detector.detections.first?.boundingBox.size == CGSize(width: 16, height: 32)
        )
        detector.clear()
        precondition(
            detector.image == nil && detector.detections.isEmpty && !detector.canStart)

        session.importFile(videoURL, as: .video)
        try await waitForIdle(session)
        precondition(session.photo == nil && session.video!.durationSeconds > 0)
        precondition(session.video!.file.displayName == "fixture.mov")
        precondition(session.video!.file.url != videoURL)
        let copiedVideo = session.video!.file

        let invalid = directory.appendingPathComponent("invalid.txt")
        try Data("not a media file".utf8).write(to: invalid)
        detector.loadPhoto(invalid)
        try await waitForImage(detector)
        precondition(
            detector.status == .error && detector.image == nil && !detector.canStart)
        session.importFile(invalid, as: .video)
        try await waitForIdle(session)
        precondition(
            session.errorMessage != nil && session.video == nil && session.photo == nil)
        precondition(FileManager.default.fileExists(atPath: videoURL.path))

        session.importFile(videoURL, as: .video)
        session.cancel()
        session.importFile(uprightURL, as: .photo)
        try await waitForIdle(session)
        precondition(
            session.photo != nil && session.video == nil && session.errorMessage == nil)
        let newest = session.photo!
        try await Task.sleep(for: .milliseconds(30))
        precondition(session.photo == newest && session.video == nil)
        session.reset()
        try await waitForRemoval(newest.url)
        try await waitForRemoval(copiedVideo.url)
        precondition(FileManager.default.fileExists(atPath: uprightURL.path))
        try await testBundledSelection(
            directory: directory, movie: videoURL, worker: worker)
        print(
            "Media import: upright/rotated photos, inference handoff, video metadata, invalid inputs, cancellation and cache ownership passed."
        )
    }

    /// Retains bundled bytes across selection changes, cancellation and import cleanup.
    private static func testBundledSelection(
        directory: URL, movie: URL, worker: MediaImportWorker
    ) async throws {
        let bundleURL = directory.appendingPathComponent("videos.bundle")
        try FileManager.default.createDirectory(
            at: bundleURL, withIntermediateDirectories: true)
        let info = try PropertyListSerialization.data(
            fromPropertyList: ["CFBundleIdentifier": "test.bundled.videos"],
            format: .xml, options: 0)
        try info.write(to: bundleURL.appendingPathComponent("Info.plist"))
        let bundledURL = bundleURL.appendingPathComponent(
            BundledVideo.nba.rawValue + ".mp4")
        try Data(contentsOf: movie).write(to: bundledURL)
        let bundle = Bundle(url: bundleURL)!
        let session = MediaImportSession(worker: worker)
        session.selectTestVideo(.nba, bundle: bundle)
        try await waitForIdle(session)
        precondition(
            session.video?.file.url.standardizedFileURL.path
                == bundledURL.standardizedFileURL.path,
            "Bundled selection failed: \(session.errorMessage ?? "URL mismatch")")
        precondition(session.video?.sourceID == BundledVideo.nba.sourceID)
        precondition(session.video!.durationSeconds > 0)
        session.selectTestVideo(.dribbling, bundle: bundle)
        try await waitForIdle(session)
        precondition(session.video == nil && session.errorMessage != nil)
        session.selectTestVideo(.nba, bundle: bundle)
        session.cancel()
        session.importFile(movie, as: .video)
        try await waitForIdle(session)
        precondition(session.video?.sourceID == nil)
        precondition(session.video?.file.url != bundledURL)
        let imported = session.video!.file
        session.reset()
        try await waitForRemoval(imported.url)
        let bundledBytes = try Data(contentsOf: bundledURL)
        let originalBytes = try Data(contentsOf: movie)
        precondition(bundledBytes == originalBytes)
    }

    /// Encodes an asymmetric image with independent EXIF orientation metadata.
    private static func writePhoto(to url: URL, orientation: Int) throws {
        var pixels = [UInt8](repeating: 0, count: 32 * 16 * 3)
        for y in 0..<16 {
            for x in 0..<32 { pixels[(y * 32 + x) * 3 + (x < 16 ? 0 : 2)] = 255 }
        }
        let image = CGImage(
            width: 32, height: 16, bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: 96, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: CGDataProvider(data: Data(pixels) as CFData)!,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.jpeg.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(
            destination, image,
            [
                kCGImagePropertyOrientation: orientation,
                kCGImageDestinationLossyCompressionQuality: 1.0,
            ] as CFDictionary)
        precondition(CGImageDestinationFinalize(destination))
    }

    /// Produces a short playable local movie for real AVFoundation validation.
    private static func writeVideo(to url: URL) async throws {
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: 32, AVVideoHeightKey: 32,
            ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: 32,
                kCVPixelBufferHeightKey as String: 32,
            ])
        writer.add(input)
        precondition(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        for frame in 0..<3 {
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while !input.isReadyForMoreMediaData && ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(5))
            }
            precondition(input.isReadyForMoreMediaData)
            var buffer: CVPixelBuffer?
            precondition(
                CVPixelBufferCreate(
                    kCFAllocatorDefault, 32, 32,
                    kCVPixelFormatType_32BGRA, nil, &buffer) == kCVReturnSuccess)
            CVPixelBufferLockBaseAddress(buffer!, [])
            let bytes = CVPixelBufferGetBytesPerRow(buffer!) * 32
            CVPixelBufferGetBaseAddress(buffer!)!.initializeMemory(
                as: UInt8.self, repeating: 0, count: bytes)
            CVPixelBufferUnlockBaseAddress(buffer!, [])
            precondition(
                adaptor.append(
                    buffer!,
                    withPresentationTime: CMTime(value: Int64(frame), timescale: 5)))
        }
        input.markAsFinished()
        await writer.finishWriting()
        precondition(writer.status == .completed, "Video fixture encoding failed")
    }

    /// Bounds import/metadata waits so failures do not hang the check process.
    private static func waitForIdle(_ session: MediaImportSession) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while session.isBusy && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        precondition(!session.isBusy, "Import timed out")
    }

    /// Waits for the shared still-image session to publish its imported result.
    private static func waitForImage(_ session: StillImageSession) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while session.isBusy && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        precondition(!session.isBusy, "Photo session timed out")
    }

    /// Confirms asynchronous cache cleanup without deleting the original fixture.
    private static func waitForRemoval(_ url: URL) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while FileManager.default.fileExists(atPath: url.path)
            && ContinuousClock.now < deadline
        {
            try await Task.sleep(for: .milliseconds(5))
        }
        precondition(
            !FileManager.default.fileExists(atPath: url.path),
            "Imported copy was not released")
    }
}

/// Verifies that the detector receives upright dimensions from the real decoder.
private actor PhotoWorker: StillImageProcessing {
    /// Keeps this fixture worker restricted to imported-photo loading.
    func load(_ sample: BundledSample) throws -> CGImage {
        throw DetectorError.invalidImage("This check expects an imported photo")
    }

    /// Returns a source-coordinate box covering the imported upright image.
    func run(_ image: CGImage, model: DetectorModel) throws -> StillImageResult {
        StillImageResult(
            detections: [
                try Detection(
                    boundingBox: CGRect(
                        x: 0, y: 0, width: image.width, height: image.height),
                    confidence: 0.9, classID: 1)
            ], latencyMilliseconds: 1)
    }
}
