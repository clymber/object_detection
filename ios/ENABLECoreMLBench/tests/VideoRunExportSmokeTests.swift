import CoreGraphics
import CoreMedia
import CryptoKit
import Foundation

/// Exercises the actual disk writer with source identities and partial-run records.
@main
struct VideoRunExportSmokeTests {
    static let sourceID = "fixture, \"quoted\"\nvideo"

    /// Leaves real Swift-produced artifacts for the independent Python reader.
    static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let folder = root.appendingPathComponent("outputs/ios_builds/video_pipeline/export_checks")
        try? FileManager.default.removeItem(at: folder)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let source = folder.appendingPathComponent("fixture.mp4")
        try Data("fixture video bytes".utf8).write(to: source)
        let bundle = try testBundle(root: root, folder: folder)
        let exporter = VideoRunExporter(bundle: bundle, directory: folder)
        let offline = try offlineReport()
        let first = try await exporter.export(.offline(offline), sourceURL: source)
        try verifyOffline(first)
        try await verifySavedCopy(first, folder: folder)
        let paced = try pacedReport()
        let second = try await exporter.export(.paced(paced), sourceURL: source)
        try verifyPaced(second)
        let repeated = try await exporter.export(.offline(offline), sourceURL: source)
        precondition(repeated.directory != first.directory)
        var failed = OfflineVideoRunReport(sourceID: sourceID, model: .rfdetrSmall,
                                           loopIndex: 0, indices: nil)
        failed.finish(status: .failed, issue: OfflineVideoIssue(
            stage: .modelLoading, error: DetectorError.missingModel("fixture")))
        let zero = try await exporter.export(.offline(failed), sourceURL: source)
        let zeroRows = try records(zero)
        precondition(zeroRows.isEmpty)
        let zeroMetrics = try mobile(zero)["metrics"] as! [String: Any]
        precondition((zeroMetrics["model_latency_ms"] as! [String: Any])["mean"] is NSNull)
        try Data("changed fixture bytes".utf8).write(to: source)
        let changed = try await exporter.export(.offline(offline), sourceURL: source)
        let input = try mobile(changed)["input"] as! [String: Any]
        precondition((input["manifest"] as! [String: Any])["resource_hash_matches"] as! Bool == false)
        let missing = try await exporter.export(.offline(failed),
            sourceURL: folder.appendingPathComponent("missing.mp4"))
        let missingInput = try mobile(missing)["input"] as! [String: Any]
        precondition(missingInput["sha256"] is NSNull)
        precondition(missingInput["sha256_unavailable_reason"] is String)
        let brokenBundleURL = folder.appendingPathComponent("Broken.bundle")
        try FileManager.default.createDirectory(at: brokenBundleURL, withIntermediateDirectories: true)
        try Data("invalid manifest".utf8).write(to: brokenBundleURL.appendingPathComponent(
            "test_images_rfdetr_small_manifest.json"))
        let broken = VideoRunExporter(bundle: Bundle(url: brokenBundleURL)!, directory: folder)
        do {
            _ = try await broken.export(.offline(offline), sourceURL: source)
            preconditionFailure("Corrupt provenance manifest was accepted")
        } catch {}
        let items = try FileManager.default.contentsOfDirectory(atPath: folder.path)
        precondition(items.filter { $0.hasPrefix(".") }.isEmpty)
        do {
            _ = try await exporter.export(.offline(OfflineVideoRunReport(
                sourceID: sourceID, model: .rfdetrSmall, loopIndex: 0, indices: nil)),
                sourceURL: source)
            preconditionFailure("Running report was exported")
        } catch {}
        try await verifyExportSession(run: .offline(offline), files: first, source: source)
        try await verifyVerificationLifecycle(run: .offline(offline), files: first)
        let index: [String: String] = ["offline": first.directory.path,
            "paced": second.directory.path, "failed_before_frames": zero.directory.path]
        try JSONSerialization.data(withJSONObject: index).write(
            to: folder.appendingPathComponent("fixture_index.json"))
        print("VideoRunExport: offline, paced, provenance, integrity, partial runs and lifecycle passed")
    }

    /// Uses exact PTS fields beyond double's integer precision and rotated geometry.
    static func metadata(_ index: Int) throws -> VideoFrameMetadata {
        try VideoFrameMetadata(sourceID: sourceID, sourceFrameIndex: index,
            presentationTime: CMTime(value: 9_007_199_254_740_993 + Int64(index),
                                     timescale: 30000, flags: [.valid, .hasBeenRounded], epoch: 2),
            width: 64, height: 32,
            preferredTransform: CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 32, ty: 0),
            loopIndex: 4)
    }

    /// Creates either a positive or successful empty model result with raw timings.
    static func result(_ index: Int, empty: Bool = false) throws -> VideoFrameResult {
        try VideoFrameResult(metadata: metadata(index), sourceWidth: 32, sourceHeight: 64,
            bufferToImage: metadata(index).preferredTransform, status: .completed,
            detections: empty ? [] : [Detection(boundingBox: CGRect(x: -2.125, y: 3.5, width: 8, height: 9),
                                               confidence: 0.912345678901, classID: 1)],
            failure: nil, timings: VideoFrameTimings(readyMilliseconds: 1000,
                modelTiming: ModelPredictionTiming(submittedMilliseconds: 1010,
                                                   completedMilliseconds: 1040),
                doneMilliseconds: 1060))
    }

    /// Includes empty success, missing decode metadata and unfinished cancellation.
    static func offlineReport() throws -> OfflineVideoRunReport {
        var r = OfflineVideoRunReport(sourceID: sourceID, model: .rfdetrSmall,
                                      loopIndex: 4, indices: [0, 1, 2, 3])
        r.append(OfflineFrameOutcome(result: try result(0)))
        r.append(OfflineFrameOutcome(result: try result(1, empty: true)))
        r.append(OfflineFrameOutcome(sourceFrameIndex: 2, metadata: nil,
            readyMilliseconds: nil, cancelled: false,
            issue: OfflineVideoIssue(stage: .decoding, error: CocoaError(.fileReadCorruptFile))))
        r.append(OfflineFrameOutcome(sourceFrameIndex: 3, metadata: try metadata(3),
            readyMilliseconds: 1100, cancelled: true,
            issue: OfflineVideoIssue(stage: .inference, error: CancellationError())))
        r.recordDecodedFrames(3)
        r.finish(status: .cancelled)
        return r
    }

    /// Records publication separately from completed/empty/skipped/cancelled inference.
    static func pacedReport() throws -> PacedVideoRunReport {
        let image = CGContext(data: nil, width: 32, height: 64, bitsPerComponent: 8,
            bytesPerRow: 128, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        var outcomes: [PacedVideoFrameOutcome] = []
        let statuses: [PacedVideoFrameOutcome.Status] = [
            .completed, .completed, .skippedSuperseded, .skippedStale, .cancelled]
        for index in 0..<5 {
            let frame = try UprightVideoFrame(metadata: metadata(index), image: image,
                bufferToImage: metadata(index).preferredTransform, readyMilliseconds: 1000)
            outcomes.append(PacedVideoFrameOutcome(
                frame: PacedVideoFrame(frame: frame, dueMilliseconds: 1000,
                                       deliveredMilliseconds: 1000),
                status: statuses[index], decidedMilliseconds: 1060,
                result: index < 2 ? try result(index, empty: index == 1) : nil,
                publication: .init(status: index == 0 ? .published : index == 1 ? .late : .notApplicable,
                                   checkedMilliseconds: index < 2 ? 1070 : nil,
                                   publishedMilliseconds: index == 0 ? 1070 : nil)))
        }
        let queue = PacedVideoQueueSnapshot(state: .cancelled, acceptedFrames: 5,
            inputDeliveryMilliseconds: Array(repeating: 1000, count: 5),
            activeFrames: 0, pendingFrames: 0, activePublications: 0, pendingPublications: 0,
            publicationEnabled: true, previewEnabled: true, outcomes: outcomes)
        var source = PacedVideoSourceReport(sourceID: sourceID, firstLoopIndex: 4,
                                            loopCount: 1, started: 900)
        source.timeline = try VideoPlaybackTimeline(sourceID: sourceID,
            sourceTimeRange: CMTimeRange(start: .zero, duration: CMTime(value: 1, timescale: 1)),
            anchorMilliseconds: 1000, firstLoopIndex: 4)
        source.decodedFrames = 5
        source.finish(status: .failed, at: 2000,
            summary: PacedVideoSourceSummary(deliveredFrames: 5),
            issue: PacedVideoSourceIssue(stage: .decoding, sourceID: sourceID, loopIndex: 4,
                attemptedFrameIndex: 5, presentationTime: .invalid, read: nil,
                dueMilliseconds: nil, observedMilliseconds: 1900, cancelled: false,
                decodeFailure: true, error: CocoaError(.fileReadCorruptFile)))
        var r = PacedVideoRunReport(sourceID: sourceID, model: .yoloxNano,
                                    firstLoopIndex: 4, loopCount: 1, previewEnabled: true)
        r.finish(status: .failed, source: source, queue: queue,
                 error: CocoaError(.fileReadCorruptFile))
        return r
    }

    /// Checks numerical precision, explicit absence and exact returned file hashes.
    static func verifyOffline(_ files: VideoExportFiles) throws {
        let rows = try records(files)
        precondition(rows.count == 4)
        let pts = rows[0]["pts"] as! [String: Any]
        precondition((pts["value"] as! NSNumber).int64Value == 9_007_199_254_740_993)
        precondition(pts["epoch"] as! Int == 2 && pts["flags"] as! Int == 3)
        let predictions = rows[0]["detections"] as! [[String: Any]]
        precondition(predictions[0]["score"] as! Double == 0.912345678901)
        precondition((predictions[0]["bbox"] as! [Double])[0] == -2.125)
        precondition((rows[1]["detections"] as! [Any]).isEmpty)
        precondition(rows[2]["pts"] is NSNull && rows[2]["detections"] is NSNull)
        precondition(rows[3]["status"] as! String == "unfinished")
        let metadata = try mobile(files)
        let validity = metadata["validity"] as! [String: Any]
        precondition(validity["coverage_status"] as! String == "incomplete")
        let model = metadata["model_provenance"] as! [String: Any]
        precondition(model["runtime_model_sha256"] is NSNull)
        precondition(model["manifest"] is [String: Any])
        let input = metadata["input"] as! [String: Any]
        let reference = input["manifest"] as! [String: Any]
        precondition(reference["resource_hash_matches"] as! Bool)
        let entry = reference["entry"] as! [String: Any]
        precondition((entry["source_path"] as! String).hasSuffix(".webm"))
        precondition((entry["resource_path"] as! String).hasSuffix(".mp4"))
        for name in ["frames.jsonl", "timings.csv"] {
            let reference = (metadata["files"] as! [String: [String: Any]])[name]!
            let data = try Data(contentsOf: files.directory.appendingPathComponent(name))
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            precondition(reference["sha256"] as! String == digest)
        }
    }

    /// Ensures late delivery does not erase inference success or unknown decode tails.
    static func verifyPaced(_ files: VideoExportFiles) throws {
        let rows = try records(files)
        precondition(rows[1]["status"] as! String == "completed")
        precondition(rows[1]["publication_status"] as! String == "late")
        precondition(rows[2]["status"] as! String == "skipped_superseded")
        precondition((rows[2]["timing_ms"] as! [String: Any])["model_submitted"] is NSNull)
        let metadata = try mobile(files)
        let issue = (metadata["source"] as! [String: Any])["issue"] as! [String: Any]
        precondition(issue["lost_frame_count"] is NSNull)
        precondition((issue["pts"] as! [String: Any])["flags"] as! Int == 0)
        let metrics = metadata["metrics"] as! [String: Any]
        precondition(metrics["input_fps"] as! Double == 5)
        precondition(metrics["fresh_fps"] as! Double == 1)
    }

    /// Reads the actual line-delimited JSON artifact without a shared serializer.
    static func records(_ files: VideoExportFiles) throws -> [[String: Any]] {
        try Data(contentsOf: files.urls[0]).split(separator: 0x0A).map {
            try JSONSerialization.jsonObject(with: Data($0)) as! [String: Any]
        }
    }

    /// Exercises a transferred folder, fresh UI readback and byte-level tamper rejection.
    @MainActor
    static func verifySavedCopy(_ files: VideoExportFiles, folder: URL) async throws {
        let saved = folder.appendingPathComponent("SavedCopy")
        try FileManager.default.copyItem(at: files.directory, to: saved)
        let verified = try VideoExportVerifier.verify(saved, expected: files.fingerprints)
        precondition(verified.records == 4 && verified.fingerprints.count == 3)
        let session = VideoExportSession(write: { _, _ in files })
        session.start(.offline(try offlineReport()), sourceURL: folder)
        while session.isBusy { await Task.yield() }
        session.verifySaved(saved)
        while session.isBusy { await Task.yield() }
        precondition(session.verification?.records == 4)
        precondition(session.verificationStatus == .verified)
        let csv = saved.appendingPathComponent("timings.csv")
        var bytes = try Data(contentsOf: csv)
        bytes.append(0x20)
        try bytes.write(to: csv)
        session.verifySaved(saved)
        while session.isBusy { await Task.yield() }
        precondition(session.verification == nil)
        precondition(session.verificationStatus == .failed)
        precondition(session.verificationMessage!.contains("differs from the prepared file"))
        do {
            _ = try VideoExportVerifier.verify(saved)
            preconditionFailure("Changed CSV passed its run-file hash")
        } catch {}
        session.reset()
        precondition(session.verification == nil && session.files == nil)
        try FileManager.default.removeItem(at: saved)
    }

    /// Reads the exported run metadata from disk.
    static func mobile(_ files: VideoExportFiles) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: Data(contentsOf: files.urls[2])) as! [String: Any]
    }

    /// Stages real provenance resources into a temporary test bundle.
    static func testBundle(root: URL, folder: URL) throws -> Bundle {
        let url = folder.appendingPathComponent("Fixtures.bundle")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let resources = root.appendingPathComponent("ios/ENABLECoreMLBench/ENABLECoreMLBench/Resources")
        for (subfolder, name) in [("TestImages", "test_images_rfdetr_small_manifest.json"),
                                  ("TestVideos", "test_videos_manifest.json")] {
            try FileManager.default.copyItem(at: resources.appendingPathComponent(subfolder)
                .appendingPathComponent(name), to: url.appendingPathComponent(name))
        }
        let manifestURL = url.appendingPathComponent("test_videos_manifest.json")
        var manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        var samples = manifest["samples"] as! [[String: Any]]
        let digest = SHA256.hash(data: Data("fixture video bytes".utf8))
            .map { String(format: "%02x", $0) }.joined()
        samples.append(["id": sourceID, "sha256": digest,
                        "source_path": "fixtures/original.webm",
                        "resource_path": "fixtures/converted.mp4"])
        manifest["samples"] = samples
        try JSONSerialization.data(withJSONObject: manifest).write(to: manifestURL)
        return Bundle(url: url)!
    }

    /// Rejects stale uncooperative writer completion and permits retry after failure.
    @MainActor
    static func verifyExportSession(run: VideoRunExport, files: VideoExportFiles, source: URL) async throws {
        let gate = ExportTestGate()
        let session = VideoExportSession(write: { _, _ in try await gate.wait() })
        session.start(run, sourceURL: source)
        while !(await gate.isWaiting) { await Task.yield() }
        session.reset()
        await gate.resume(.success(files))
        for _ in 0..<20 { await Task.yield() }
        precondition(session.files == nil && !session.isBusy)
        session.start(run, sourceURL: source)
        while !(await gate.isWaiting) { await Task.yield() }
        await gate.resume(.failure(CocoaError(.fileWriteNoPermission)))
        while session.isBusy { await Task.yield() }
        precondition(session.files == nil && session.message!.hasPrefix("Export failed:"))
        session.start(run, sourceURL: source)
        while !(await gate.isWaiting) { await Task.yield() }
        await gate.resume(.success(files))
        while session.isBusy { await Task.yield() }
        precondition(session.files?.directory == files.directory)
    }

    /// Keeps results through picker inactivity, then checks Stop and stale-reader rejection.
    @MainActor
    static func verifyVerificationLifecycle(run: VideoRunExport, files: VideoExportFiles) async throws {
        let gate = VerificationTestGate()
        let session = VideoExportSession(
            read: { _, _ in try await gate.wait() }, write: { _, _ in files })
        let coordinator = VideoSession(export: session)
        let report = VideoExportVerification(
            runID: "held-read", records: 4, fingerprints: files.fingerprints)
        session.start(run, sourceURL: files.directory)
        while session.isBusy { await Task.yield() }

        session.verifySaved(files.directory)
        while !(await gate.isWaiting) { await Task.yield() }
        // ContentView uses this path for both picker-driven inactivity and disappearance.
        coordinator.suspendDetection()
        precondition(session.isBusy && session.files != nil)
        precondition(session.verificationStatus == .checking)
        await gate.resume(.success(report))
        while session.isBusy { await Task.yield() }
        coordinator.suspendDetection()
        precondition(session.verificationStatus == .verified && session.verification != nil)
        precondition(session.verificationMessage!.contains("4 frame records match"))
        session.reportVerificationFailure(CocoaError(.userCancelled))
        precondition(session.verificationStatus == .verified)

        session.verifySaved(files.directory)
        while !(await gate.isWaiting) { await Task.yield() }
        coordinator.suspendDetection()
        await gate.resume(.failure(CocoaError(.fileReadNoPermission)))
        while session.isBusy { await Task.yield() }
        coordinator.suspendDetection()
        precondition(session.verificationStatus == .failed && session.verificationMessage != nil)
        precondition(session.files?.directory == files.directory)

        session.verifySaved(files.directory)
        while !(await gate.isWaiting) { await Task.yield() }
        coordinator.stop()
        precondition(session.verificationStatus == .cancelled && !session.isBusy)
        precondition(session.files?.directory == files.directory)
        await gate.resume(.success(report))
        for _ in 0..<20 { await Task.yield() }
        precondition(session.verificationStatus == .cancelled && session.verification == nil)

        session.verifySaved(files.directory)
        while !(await gate.isWaiting) { await Task.yield() }
        coordinator.select(nil)
        await gate.resume(.failure(CocoaError(.fileReadNoPermission)))
        for _ in 0..<20 { await Task.yield() }
        precondition(session.verificationStatus == .idle && session.files == nil)
        precondition(session.verificationMessage == nil)
    }
}

/// Retains a read completion across inactivity and cancellation for deterministic checks.
actor VerificationTestGate {
    private var continuation: CheckedContinuation<VideoExportVerification, Error>?
    var isWaiting: Bool { continuation != nil }

    /// Suspends verification until the test injects a success or a provider error.
    func wait() async throws -> VideoExportVerification {
        try await withCheckedThrowingContinuation { continuation = $0 }
    }

    /// Releases the reader even if its UI request has already been cancelled or reset.
    func resume(_ result: Result<VideoExportVerification, Error>) {
        continuation?.resume(with: result)
        continuation = nil
    }
}

/// Holds an export completion independently of cancellation for lifecycle checks.
actor ExportTestGate {
    private var continuation: CheckedContinuation<VideoExportFiles, Error>?
    var isWaiting: Bool { continuation != nil }

    /// Suspends until the test chooses success or failure.
    func wait() async throws -> VideoExportFiles {
        try await withCheckedThrowingContinuation { continuation = $0 }
    }

    /// Releases exactly one held completion, even after its UI generation expired.
    func resume(_ result: Result<VideoExportFiles, Error>) {
        continuation?.resume(with: result)
        continuation = nil
    }
}
