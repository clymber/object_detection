import AVFoundation
import CoreGraphics
import CoreMedia
import CoreText
import CoreVideo
import Foundation

/// Describes a finalized recording without retaining any camera images.
nonisolated struct CameraRecordingReport: Sendable {
    enum Status: String, Sendable { case completed, empty, failed }
    let status: Status
    let url: URL?
    let encodedFrames: Int
    let skippedFrames: Int
    let firstSourceTime: CMTime?
    let lastSourceTime: CMTime?
    let issue: VideoInferenceFailure?
}

/// Bounds admission to two copied images; AVFoundation stays on one serial queue.
/// Each instance belongs to one run and is finished once after inference unwinds.
nonisolated final class CameraVideoRecorder: @unchecked Sendable {
    private let queue: DispatchQueue
    private let lock = NSLock()
    private var accepting = true
    private var outstanding = 0
    private var skipped = 0
    private let outputURL: URL
    private let frameDuration: CMTime
    // Only the recording queue accesses the following state.
    private var writer: AVAssetWriter?
    private var receiver: AVAssetWriterInput.PixelBufferReceiver?
    private var dimensions: CGSize?
    private var encoded = 0
    private var firstTime: CMTime?
    private var lastTime: CMTime?
    private var failure: VideoInferenceFailure?

    /// Creates a unique file lazily so denied or empty camera runs leave no files.
    init(
        framesPerSecond: Int,
        directory: URL = URL.cachesDirectory.appendingPathComponent("CameraRecordings"),
        queue: DispatchQueue = DispatchQueue(
            label: "bench.camera.recording", qos: .utility)
    ) {
        self.queue = queue
        outputURL = directory.appendingPathComponent(
            "camera_\(UUID().uuidString).mp4")
        frameDuration = CMTime(
            value: 1, timescale: Int32(clamping: max(1, framesPerSecond)))
    }

    /// Returns immediately; overload skips recording frames without blocking inference.
    func offer(_ frame: VideoFramePreview) {
        lock.withLock {
            guard accepting else { return }
            guard outstanding < 2 else {
                skipped += 1
                return
            }
            outstanding += 1
            queue.async { [self] in
                autoreleasepool {
                    if failure == nil {
                        do { try append(frame) } catch {
                            failure = VideoInferenceFailure(error)
                            lock.withLock { skipped += 1 }
                        }
                    } else {
                        lock.withLock { skipped += 1 }
                    }
                }
                lock.withLock { outstanding -= 1 }
            }
        }
    }

    /// Drains accepted images and finishes even when the camera task was cancelled.
    func finish() async -> CameraRecordingReport {
        await withCheckedContinuation { continuation in
            lock.withLock {
                accepting = false
                queue.async { [self] in
                    guard failure == nil, encoded > 0, let writer, let lastTime,
                        let firstTime
                    else {
                        writer?.cancelWriting()
                        continuation.resume(returning: completedReport(success: false))
                        return
                    }
                    let end = CMTimeAdd(
                        CMTimeSubtract(lastTime, firstTime), frameDuration)
                    writer.endSession(atSourceTime: end)
                    receiver?.finish()
                    writer.finishWriting { [self] in
                        queue.async { [self] in
                            guard let writer = self.writer else { return }
                            if writer.status != .completed {
                                failure = VideoInferenceFailure(
                                    writer.error ?? CameraError.configuration(
                                        "Could not finish the MP4 recording."))
                            }
                            continuation.resume(
                                returning: completedReport(
                                    success: writer.status == .completed))
                        }
                    }
                }
            }
        }
    }

    /// Preserves capture gaps and uses the upright image dimensions and exact result.
    private func append(_ frame: VideoFramePreview) throws {
        let time = frame.result.metadata.presentationTime
        if let lastTime {
            guard time.epoch == lastTime.epoch,
                CMTimeCompare(time, lastTime) > 0
            else {
                throw CameraError.configuration(
                    "Recording timestamps are not increasing.")
            }
        }
        if writer == nil { try prepare(frame.image) }
        let size = CGSize(width: frame.image.width, height: frame.image.height)
        guard dimensions == size,
            let receiver, let pool = receiver.pixelBufferPool
        else {
            throw CameraError.configuration("Recording frame dimensions changed.")
        }
        let pixels: CVMutablePixelBuffer
        do {
            pixels = try pool.makeMutablePixelBuffer(.init(allocationThreshold: 4))
        } catch let error as CVError where error == .wouldExceedAllocationThreshold {
            lock.withLock { skipped += 1 }
            return
        }
        try pixels.withUnsafeBuffer { buffer in
            try CameraAnnotationRenderer.draw(frame, into: buffer)
        }
        let origin = firstTime ?? time
        let accepted = try receiver.appendImmediately(
            CVReadOnlyPixelBuffer(pixels), with: CMTimeSubtract(time, origin))
        if accepted {
            firstTime = origin
            lastTime = time
            encoded += 1
        } else {
            lock.withLock { skipped += 1 }
        }
    }

    /// Configures H.264 MP4 and a bounded pixel pool using the real-time receiver.
    private func prepare(_ image: CGImage) throws {
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        self.writer = writer
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: image.width, AVVideoHeightKey: image.height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAllowFrameReorderingKey: false,
                AVVideoExpectedSourceFrameRateKey: frameDuration.timescale,
            ],
        ])
        guard writer.canAdd(input) else {
            throw CameraError.configuration("H.264 recording is unavailable.")
        }
        receiver = writer.inputPixelBufferReceiver(
            for: input,
            pixelBufferAttributes: CVPixelBufferCreationAttributes(
                pixelFormatType: .init(rawValue: kCVPixelFormatType_32BGRA),
                size: .init(width: image.width, height: image.height),
                compatibility: [.cgImage, .cgBitmapContext]))
        dimensions = CGSize(width: image.width, height: image.height)
        try writer.start()
        writer.startSession(atSourceTime: .zero)
    }

    /// Deletes partial files and exposes only successfully finalized movies.
    private func completedReport(success: Bool) -> CameraRecordingReport {
        if !success { try? FileManager.default.removeItem(at: outputURL) }
        let report = CameraRecordingReport(
            status: success ? .completed : failure == nil ? .empty : .failed,
            url: success ? outputURL : nil, encodedFrames: encoded,
            skippedFrames: lock.withLock { skipped }, firstSourceTime: firstTime,
            lastSourceTime: lastTime, issue: failure)
        receiver = nil
        writer = nil
        return report
    }
}

/// Burns source-pixel boxes and readable class/confidence labels into copied pixels.
nonisolated enum CameraAnnotationRenderer {
    /// Renders off MainActor, including empty detections and clipped edge labels.
    static func draw(_ frame: VideoFramePreview, into pixels: CVPixelBuffer) throws {
        guard CVPixelBufferGetWidth(pixels) == frame.image.width,
            CVPixelBufferGetHeight(pixels) == frame.image.height,
            CVPixelBufferGetPixelFormatType(pixels) == kCVPixelFormatType_32BGRA
        else {
            throw CameraError.configuration("Invalid recording pixel format or size.")
        }
        guard CVPixelBufferLockBaseAddress(pixels, []) == kCVReturnSuccess else {
            throw CameraError.configuration("Could not lock recording pixels.")
        }
        defer { CVPixelBufferUnlockBaseAddress(pixels, []) }
        let width = frame.image.width
        let height = frame.image.height
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(pixels), width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixels),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue)
        else { throw CameraError.configuration("Could not draw recording pixels.") }
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        context.draw(frame.image, in: bounds)
        // Quartz uses a bottom-left origin; detections use top-left source pixels.
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        let layout = PreviewGeometry(sourceSize: bounds.size, viewportSize: bounds.size)
        let fontSize = max(13, CGFloat(width) / 64)
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, fontSize, nil)
        let red = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
        context.setStrokeColor(red)
        context.setLineWidth(max(2, CGFloat(width) / 400))
        for detection in frame.result.detections {
            guard let box = layout.displayedBox(detection.boundingBox) else { continue }
            context.stroke(box)
            let name = detection.classID == 1
                ? "Basketball" : "Class \(detection.classID)"
            let text = name + String(format: " %.1f%%", detection.confidence * 100)
            let attributes: [NSAttributedString.Key: Any] = [
                NSAttributedString.Key(kCTFontAttributeName as String): font,
                NSAttributedString.Key(kCTForegroundColorAttributeName as String):
                    CGColor(gray: 1, alpha: 1),
            ]
            let line = CTLineCreateWithAttributedString(
                NSAttributedString(string: text, attributes: attributes))
            let rect = layout.labelRect(
                above: box,
                size: CGSize(
                    width: CTLineGetTypographicBounds(line, nil, nil, nil) + 12,
                    height: fontSize + 10))
            context.setFillColor(red)
            context.fill(rect)
            context.saveGState()
            context.clip(to: rect)
            context.textMatrix = CGAffineTransform(scaleX: 1, y: -1)
            context.textPosition = CGPoint(x: rect.minX + 6, y: rect.maxY - 5)
            CTLineDraw(line, context)
            context.restoreGState()
        }
    }
}
