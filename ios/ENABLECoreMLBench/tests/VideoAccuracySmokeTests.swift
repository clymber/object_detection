import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation

/// Checks actual video import, scoring policies, model warm-up and shared exports.
@main
@MainActor
struct VideoAccuracySmokeTests {
    static let sourceID = "accuracy-fixture"

    /// Runs fixed matching cases and a complete native pipeline with a fake detector.
    static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let folder = root.appendingPathComponent(
            "outputs/ios_builds/video_pipeline/accuracy_checks")
        try? FileManager.default.removeItem(at: folder)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let video = folder.appendingPathComponent("test.mp4")
        try await makeVideo(video)
        let data = try annotationData(video: video)
        let annotationURL = folder.appendingPathComponent("annotations.json")
        try data.write(to: annotationURL)
        let annotations = try VideoAnnotations.decode(data)
        try testInvalidInputs(data)
        let imported = try ImportedVideoAnnotations.load(annotationURL, videoURL: video)
        let wrong = folder.appendingPathComponent("wrong.mp4")
        try Data("different video".utf8).write(to: wrong)
        try rejects { _ = try ImportedVideoAnnotations.load(annotationURL, videoURL: wrong) }
        let offline = try offlineReport()
        let metrics = try VideoAccuracyEvaluator.evaluate(
            .offline(offline), annotations: annotations)
        precondition(metrics.truePositives == 2 && metrics.falsePositives == 2)
        precondition(metrics.falseNegatives == 1 && metrics.scoredFrames == 4)
        precondition(abs(metrics.f1 - 4.0 / 7) < 1e-12 && metrics.frameCoverage == 1)
        precondition(metrics.scope == "all_video_frames")
        let failed = try VideoAccuracyEvaluator.evaluate(
            .offline(offlineReport(failedFirst: true)), annotations: annotations)
        precondition(failed.failedFrames == 1 && failed.falseNegatives == 2)
        precondition(failed.completedFrames == 3 && failed.frameCoverage == 0.75)
        try rejects {
            _ = try VideoAccuracyEvaluator.evaluate(
                .offline(offlineReport(cancelled: true)), annotations: annotations)
        }
        try rejects {
            _ = try VideoAccuracyEvaluator.evaluate(
                .offline(offlineReport(duplicate: true)), annotations: annotations)
        }
        try rejects {
            _ = try VideoAccuracyEvaluator.evaluate(
                .offline(offlineReport(wrongPTS: true)), annotations: annotations)
        }
        let paced = try VideoAccuracyEvaluator.evaluate(
            .paced(pacedReport()), annotations: annotations)
        precondition(paced.truePositives == 1 && paced.falsePositives == 1)
        precondition(paced.falseNegatives == 0 && paced.scoredFrames == 2)
        precondition(paced.frameCoverage == 0.5 && paced.skippedFrames == 1)
        precondition(paced.postWindowFrames == 1 && paced.detectionFPS == 5)
        precondition(abs(paced.f1 - 2.0 / 3) < 1e-12)
        let exporter = VideoRunExporter(directory: folder)
        let exportAccuracy = VideoAccuracyExport(
            runID: offline.id, annotations: imported, report: metrics,
            unavailableReason: nil, warmupFrames: 30)
        let files = try await exporter.export(.offline(offline), sourceURL: video,
                                               accuracy: exportAccuracy)
        let saved = try JSONSerialization.jsonObject(
            with: Data(contentsOf: files.directory.appendingPathComponent("mobile_run.json")))
            as! [String: Any]
        let accuracy = saved["accuracy"] as! [String: Any]
        precondition(accuracy["annotation_sha256"] as! String == imported.sha256)
        precondition(accuracy["warmup_frames"] as! Int == 30)
        precondition((accuracy["metrics"] as! [String: Any])["f1"] as! Double == metrics.f1)
        let verified = try VideoExportVerifier.verify(files.directory)
        precondition(verified.records == 4)
        do {
            _ = try await exporter.export(.offline(offlineReport()), sourceURL: video,
                                          accuracy: exportAccuracy)
            preconditionFailure("Accepted accuracy from another run")
        } catch is VideoBenchmarkError {}
        // Retain a small oracle fixture for checking the shared Python evaluator.
        var predictions: [[String: Any]] = []
        for outcome in offline.outcomes {
            for detection in outcome.result?.detections ?? [] {
                let box = detection.boundingBox
                predictions.append(["image_id": outcome.sourceFrameIndex + 1,
                    "category_id": detection.classID, "score": detection.confidence,
                    "bbox": [box.minX, box.minY, box.width, box.height]])
            }
        }
        try JSONSerialization.data(withJSONObject: predictions).write(
            to: folder.appendingPathComponent("predictions.json"))
        try JSONSerialization.data(withJSONObject: metrics.json()).write(
            to: folder.appendingPathComponent("swift_metrics.json"))
        try await testSession(video: video, annotationURL: annotationURL, folder: folder)
        let full = root.appendingPathComponent(
            "data/processed/ios_benchmark/annotated_test_video_v2/basketball_test_annotations.json")
        if FileManager.default.fileExists(atPath: full.path) {
            let actual = try VideoAnnotations.decode(Data(contentsOf: full))
            precondition(actual.images.count == 5400 && actual.annotations.count == 5460)
        }
        print("Video accuracy: parsing, matching, failures, paced coverage/window, hashes,")
        print("export round trip, warm-up/replay and input/model lifecycle checks passed.")
    }

    /// Builds a small ground-truth file with three positive frames and one background.
    static func annotationData(video: URL) throws -> Data {
        let boxes: [[String: Any]] = [
            ["id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20],
             "area": 400, "iscrowd": 0],
            ["id": 2, "image_id": 2, "category_id": 1, "bbox": [50, 10, 10, 10],
             "area": 100, "iscrowd": 0],
            ["id": 3, "image_id": 4, "category_id": 1, "bbox": [10, 10, 20, 20],
             "area": 400, "iscrowd": 0],
        ]
        let frames: [[String: Any]] = (0..<4).map {
            ["id": $0 + 1, "video_id": 1, "frame_index": $0,
             "timestamp_seconds": Double($0) / 10, "width": 64, "height": 64]
        }
        return try JSONSerialization.data(withJSONObject: [
            "schema_version": 2, "type": "coco_video_boxes",
            "info": ["bbox_format": "xywh", "coordinate_origin": "top_left",
                     "frame_index_base": 0],
            "videos": [["id": 1, "file_name": "test.mp4", "width": 64, "height": 64,
                        "fps": 10, "frame_count": 4, "duration_seconds": 0.4,
                        "sha256": try VideoAnnotations.fileHash(video)]],
            "images": frames, "annotations": boxes,
            "categories": [["id": 1, "name": "basketball"]],
        ])
    }

    /// Rejects incomplete frame coverage, duplicate IDs and invalid box references.
    static func testInvalidInputs(_ data: Data) throws {
        let original = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        for mutation in 0..<4 {
            var object = original
            var frames = object["images"] as! [[String: Any]]
            var boxes = object["annotations"] as! [[String: Any]]
            switch mutation {
            case 0: frames.removeLast()
            case 1: frames[1]["frame_index"] = 0
            case 2: boxes[0]["image_id"] = 999
            default: boxes[0]["iscrowd"] = 1
            }
            object["images"] = frames; object["annotations"] = boxes
            try rejects {
                _ = try VideoAnnotations.decode(JSONSerialization.data(withJSONObject: object))
            }
        }
    }

    /// Requires a failure instead of silently accepting invalid benchmark inputs.
    static func rejects(_ operation: () throws -> Void) throws {
        do { try operation(); preconditionFailure("Invalid benchmark input was accepted") }
        catch is VideoBenchmarkError {}
    }

    /// Includes duplicates, misses, background false positives and a score exactly 0.25.
    static func result(_ index: Int, failed: Bool = false, wrongPTS: Bool = false)
        throws -> VideoFrameResult {
        let metadata = try VideoFrameMetadata(sourceID: sourceID, sourceFrameIndex: index,
            presentationTime: CMTime(value: Int64(index + (wrongPTS ? 1 : 0)), timescale: 10),
            width: 64, height: 64)
        let scores: [[Double]] = [[0.9, 0.8], [], [0.7], [0.25]]
        let detections = try scores[index].map {
            try Detection(boundingBox: CGRect(
                x: 10, y: 10, width: index == 3 ? 10 : 20, height: 20),
                          confidence: $0, classID: 1)
        }
        return VideoFrameResult(metadata: metadata, sourceWidth: 64, sourceHeight: 64,
            bufferToImage: .identity, status: failed ? .failed : .completed,
            detections: failed ? [] : detections,
            failure: failed ? VideoInferenceFailure(VideoBenchmarkError("fixture")) : nil,
            timings: try VideoFrameTimings(readyMilliseconds: 1000 + Double(index) * 100,
                modelTiming: nil, doneMilliseconds: 1050 + Double(index) * 100))
    }

    /// Preserves frame indices even when inference fails or results arrive malformed.
    static func offlineReport(
        failedFirst: Bool = false, cancelled: Bool = false, duplicate: Bool = false,
        wrongPTS: Bool = false
    ) throws -> OfflineVideoRunReport {
        var report = OfflineVideoRunReport(sourceID: sourceID, model: .rfdetrSmall,
                                           loopIndex: 0, indices: nil)
        for index in 0..<4 {
            report.append(OfflineFrameOutcome(result: try result(
                duplicate && index == 3 ? 0 : index,
                failed: index == 0 && failedFirst, wrongPTS: wrongPTS)))
        }
        report.recordDecodedFrames(4)
        report.recordEndOfSource()
        report.finish(status: cancelled ? .cancelled : .completed)
        return report
    }

    /// Makes processed indices 0 and 2 non-contiguous; index 3 completes after playback.
    static func pacedReport() throws -> PacedVideoRunReport {
        let image = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8,
            bytesPerRow: 256, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!.makeImage()!
        var outcomes: [PacedVideoFrameOutcome] = []
        for index in 0..<4 {
            let base = try result(index)
            let done = index == 3 ? 1450.0 : base.timings.doneMilliseconds
            let detection = VideoFrameResult(metadata: base.metadata, sourceWidth: 64,
                sourceHeight: 64, bufferToImage: .identity, status: .completed,
                detections: index == 0 ? [base.detections[0]] : base.detections,
                failure: nil, timings: try VideoFrameTimings(
                    readyMilliseconds: base.timings.readyMilliseconds,
                    modelTiming: nil, doneMilliseconds: done))
            let frame = UprightVideoFrame(metadata: base.metadata, image: image,
                bufferToImage: .identity, readyMilliseconds: base.timings.readyMilliseconds)
            outcomes.append(PacedVideoFrameOutcome(
                frame: PacedVideoFrame(frame: frame,
                    dueMilliseconds: base.timings.readyMilliseconds,
                    deliveredMilliseconds: base.timings.readyMilliseconds),
                status: index == 1 ? .skippedSuperseded : .completed,
                decidedMilliseconds: 1400, result: index == 1 ? nil : detection))
        }
        let queue = PacedVideoQueueSnapshot(state: .completed, acceptedFrames: 4,
            inputDeliveryMilliseconds: [1000, 1100, 1200, 1300], activeFrames: 0,
            pendingFrames: 0, activePublications: 0, pendingPublications: 0,
            publicationEnabled: false, previewEnabled: false, outcomes: outcomes)
        var source = PacedVideoSourceReport(sourceID: sourceID,
            firstLoopIndex: 0, loopCount: 1, started: 900)
        source.timeline = try VideoPlaybackTimeline(sourceID: sourceID,
            sourceTimeRange: CMTimeRange(start: .zero, duration: CMTime(value: 4, timescale: 10)),
            anchorMilliseconds: 1000)
        source.decodedFrames = 4
        source.finish(status: .completed, at: 1400,
                      summary: PacedVideoSourceSummary(deliveredFrames: 4, completedLoops: 1))
        var report = PacedVideoRunReport(sourceID: sourceID, model: .rfdetrSmall,
            firstLoopIndex: 0, loopCount: 1, previewEnabled: false, trackID: nil)
        report.finish(status: .completed, source: source, queue: queue)
        return report
    }

    /// Verifies actual warm-up/replay, model reuse, paced scoring and stale import removal.
    static func testSession(video: URL, annotationURL: URL, folder: URL) async throws {
        let detector = AccuracyFixtureDetector()
        let engine = DetectorEngine(loader: { _ in detector })
        let exporter = VideoRunExporter(directory: folder.appendingPathComponent("session_exports"))
        let export = VideoExportSession(write: { run, url, accuracy in
            try await exporter.export(run, sourceURL: url, accuracy: accuracy)
        })
        let session = VideoSession(export: export, engine: engine)
        session.select(video, sourceID: sourceID)
        session.importAnnotations(annotationURL)
        try await wait { !session.isBusy }
        precondition(session.accuracy.imported != nil)
        session.start(model: .rfdetrSmall)
        try await wait { !session.isBusy }
        precondition(session.accuracy.result?.report?.scoredFrames == 4)
        precondition(session.accuracy.result?.warmupFrames == 30)
        let firstCount = await detector.calls
        precondition(firstCount == 34)
        session.exportResults()
        try await wait { !session.isBusy }
        let saved = try JSONSerialization.jsonObject(with: Data(contentsOf:
            session.export.files!.directory.appendingPathComponent("mobile_run.json")))
            as! [String: Any]
        precondition((saved["accuracy"] as! [String: Any])["warmup_frames"] as! Int == 30)
        session.select(video, sourceID: sourceID)
        precondition(session.accuracy.imported != nil && session.accuracy.result == nil)
        session.setMode(.paced)
        session.start(model: .yoloxNano)
        try await wait { !session.isBusy }
        precondition(session.accuracy.result?.report?.scoredFrames == 4)
        precondition(session.accuracy.result?.report?.scope == "processed_frames_within_playback")
        let nextCount = await detector.calls
        precondition(nextCount == 68)
        session.exportResults()
        try await wait { !session.isBusy }
        let pacedSaved = try JSONSerialization.jsonObject(with: Data(contentsOf:
            session.export.files!.directory.appendingPathComponent("mobile_run.json")))
            as! [String: Any]
        let pacedAccuracy = pacedSaved["accuracy"] as! [String: Any]
        precondition(pacedAccuracy["status"] as! String == "completed")
        precondition((pacedAccuracy["metrics"] as! [String: Any])["scope"] as! String
            == "processed_frames_within_playback")
        session.accuracy.clearResult()
        session.exportResults()
        try await wait { !session.isBusy }
        let unscored = try JSONSerialization.jsonObject(with: Data(contentsOf:
            session.export.files!.directory.appendingPathComponent("mobile_run.json")))
            as! [String: Any]
        precondition((unscored["accuracy"] as! [String: Any])["status"] as! String == "unavailable")
        await detector.delayNextCall()
        session.start(model: .rfdetrNano)
        for _ in 0..<1000 {
            if await detector.calls > 68 { break }
            try await Task.sleep(for: .milliseconds(2))
        }
        let warmingCount = await detector.calls
        precondition(warmingCount == 69)
        session.stop()
        precondition(session.isBusy)
        session.start(model: .rfdetrSmall)
        try await wait { !session.isBusy }
        let stoppedCount = await detector.calls
        precondition(stoppedCount == 69 && !session.hasReport)
        precondition(session.accuracy.result == nil)
        session.importAnnotations(annotationURL)
        session.select(folder.appendingPathComponent("another.mp4"))
        try await Task.sleep(for: .milliseconds(30))
        precondition(session.accuracy.imported == nil && session.accuracy.result == nil)
    }

    /// Bounds asynchronous UI tests without depending on exact scheduling order.
    static func wait(_ condition: () -> Bool) async throws {
        for _ in 0..<1000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        preconditionFailure("Timed out waiting for video accuracy state")
    }

    /// Encodes four real frames so warm-up uses the same reader as the actual app.
    static func makeVideo(_ url: URL) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 64, AVVideoHeightKey: 64,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
            sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String:
                kCVPixelFormatType_32BGRA, kCVPixelBufferWidthKey as String: 64,
                kCVPixelBufferHeightKey as String: 64])
        writer.add(input)
        precondition(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        for index in 0..<4 {
            while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(5)) }
            var buffer: CVPixelBuffer?
            precondition(CVPixelBufferCreate(kCFAllocatorDefault, 64, 64,
                kCVPixelFormatType_32BGRA, nil, &buffer) == kCVReturnSuccess)
            CVPixelBufferLockBaseAddress(buffer!, [])
            memset(CVPixelBufferGetBaseAddress(buffer!), 128,
                   CVPixelBufferGetBytesPerRow(buffer!) * 64)
            CVPixelBufferUnlockBaseAddress(buffer!, [])
            precondition(adaptor.append(buffer!, withPresentationTime:
                CMTime(value: Int64(index), timescale: 10)))
        }
        input.markAsFinished()
        await writer.finishWriting()
        precondition(writer.status == .completed)
    }
}

/// Counts untimed and measured calls independently of any actual model framework.
actor AccuracyFixtureDetector: ObjectDetector {
    private(set) var calls = 0
    private var delayNext = false

    /// Holds a genuine warm-up prediction long enough to stop it during execution.
    func delayNextCall() { delayNext = true }

    /// Returns one source-pixel box quickly enough to process all four paced frames.
    func detect(in image: CGImage) async throws -> [Detection] {
        try Task.checkCancellation()
        calls += 1
        if delayNext {
            delayNext = false
            try await Task.sleep(for: .milliseconds(200))
        }
        return [try Detection(boundingBox: CGRect(x: 10, y: 10, width: 20, height: 20),
                              confidence: 0.9, classID: 1)]
    }
}
