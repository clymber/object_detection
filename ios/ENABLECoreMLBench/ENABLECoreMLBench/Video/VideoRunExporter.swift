import CoreMedia
import CryptoKit
import Darwin
import Foundation

/// Writes diagnostic handoff files on a worker actor after all measured work ends.
actor VideoRunExporter {
    private let bundle: Bundle
    private let directory: URL

    /// Allows tests to export into an isolated directory with a controlled bundle.
    init(
        bundle: Bundle = .main,
        directory: URL = URL.cachesDirectory.appendingPathComponent("VideoRunExports")
    ) {
        self.bundle = bundle
        self.directory = directory
    }

    /// Publishes a unique directory only after every file has been written and hashed.
    func export(_ run: VideoRunExport, sourceURL: URL?) throws -> VideoExportFiles {
        guard run.status != "running", run.endedAt != nil else {
            throw CocoaError(.fileWriteUnknown, userInfo: [
                NSLocalizedDescriptionKey: "Wait for the video worker to finish."])
        }
        try Task.checkCancellation()
        let stamp = Self.utc(run.startedAt).filter { $0.isNumber || $0 == "T" || $0 == "Z" }
        let runID = "\(stamp)_native_\(run.model.rawValue)_\(run.mode)_\(UUID().uuidString)"
        let target = directory.appendingPathComponent(runID, isDirectory: true)
        let staging = directory.appendingPathComponent(".\(runID)", isDirectory: true)
        let manager = FileManager.default
        try manager.createDirectory(at: staging, withIntermediateDirectories: true)
        do {
            let count = try writeRecords(run, runID: runID, to: staging)
            var metadata = mobileRun(run, runID: runID, recordCount: count)
            let coverageStatus: String
            let reasons: [String]
            if case .camera(let r) = run {
                let manifest: [String: Any] = ["source_type": "camera", "source_id": r.sourceID,
                    "configuration": CameraExportRecord.configuration(r.configuration),
                    "actual_capture_settings": r.capture?.actual ?? [:], "category_mapping": ["0": 1],
                    "timestamp_mapping": "CMSyncConvertTime to host; calibration in mobile_run.capture"]
                let manifestData = try Self.json(manifest)
                metadata["input"] = ["source_id": r.sourceID, "source_type": "camera",
                    "sha256": NSNull(), "sha256_unavailable_reason": "Live camera has no fixed input file",
                    "manifest": manifest, "manifest_json": String(data: manifestData, encoding: .utf8)!,
                    "manifest_sha256": Self.digest(manifestData)]
                let coverage = CameraExportRecord.coverage(r)
                metadata["coverage"] = coverage
                coverageStatus = coverage["status"] as! String
                reasons = coverage["reasons"] as! [String]
            } else {
                guard let sourceURL else { throw CameraError.configuration("Video export needs its source URL.") }
                let input = try inputReference(run, url: sourceURL)
                metadata["input"] = input
                let coverage = VideoRunCoverage(run: run, reference: VideoCoverageReference(input: input))
                metadata["coverage"] = coverage.json
                coverageStatus = coverage.status; reasons = coverage.reasons
            }
            var validity = metadata["validity"] as! [String: Any]
            validity["coverage_status"] = coverageStatus
            validity["reasons"] = reasons + [
                "Model/video parity and camera ground-truth evaluation are not established by this diagnostic run",
                "Diagnostic run without benchmark warm-up or repetitions",
                "Sustained-run conditions have not been verified"]
            metadata["validity"] = validity
            metadata["model_provenance"] = try modelReference(run.model)
            metadata["files"] = [
                "frames.jsonl": [
                    "sha256": try Self.fileDigest(staging.appendingPathComponent("frames.jsonl")),
                    "records": count],
                "timings.csv": [
                    "sha256": try Self.fileDigest(staging.appendingPathComponent("timings.csv")),
                    "records": count],
            ]
            try Self.json(metadata, pretty: true).write(
                to: staging.appendingPathComponent("mobile_run.json"))
            let verified = try VideoExportVerifier.verify(staging)
            try Task.checkCancellation()
            try manager.moveItem(at: staging, to: target)
            return VideoExportFiles(directory: target, fingerprints: verified.fingerprints)
        } catch {
            try? manager.removeItem(at: staging)
            throw error
        }
    }

    /// Streams one numerical record at a time without copying a whole run's JSON.
    private func writeRecords(_ run: VideoRunExport, runID: String, to folder: URL) throws -> Int {
        let jsonURL = folder.appendingPathComponent("frames.jsonl")
        let csvURL = folder.appendingPathComponent("timings.csv")
        try Data().write(to: jsonURL)
        try Data((VideoExportRecord.csvColumns.joined(separator: ",") + "\r\n").utf8)
            .write(to: csvURL)
        let jsonFile = try FileHandle(forWritingTo: jsonURL)
        defer { try? jsonFile.close() }
        let csvFile = try FileHandle(forWritingTo: csvURL)
        defer { try? csvFile.close() }
        _ = try csvFile.seekToEnd()
        var count = 0
        try VideoExportRecord.forEachFrame(run) { record in
            try Task.checkCancellation()
            var row = record
            row["run_id"] = runID
            var data = try Self.json(row)
            data.append(0x0A)
            try jsonFile.write(contentsOf: data)
            try csvFile.write(contentsOf: Data(VideoExportRecord.csvRow(row, runID: runID).utf8))
            count += 1
        }
        return count
    }

    /// Uses explicit unavailable annotations instead of guessing uncollected conditions.
    private func mobileRun(
        _ run: VideoRunExport, runID: String, recordCount: Int
    ) -> [String: Any] {
        let model = run.model
        var value: [String: Any] = [
            "schema_version": 1, "scope": run.mode == "camera" ? "camera_diagnostic" : "video_diagnostic", "run_id": runID,
            "execution_id": run.id.uuidString, "protocol_id": "enable_ios_detection",
            "protocol_version": "0.3-draft", "mode": run.mode, "status": run.status,
            "started_at": Self.utc(run.startedAt),
            "ended_at": VideoExportRecord.nullable(run.endedAt.map { Self.utc($0) }),
            "exported_at": Self.utc(Date()), "runtime": runtime(),
            "settings": [
                "model": model.rawValue, "input_shape": model.inputShape,
                "precision": model.isRFDETR ? "float16" : "float32",
                "compute_units": "all", "compute_placement": NSNull(),
                "compute_placement_unavailable_reason": "Hardware placement was not profiled",
                "confidence_threshold": 0.25, "score_floor": 0.001,
                "max_detections": 100, "nms": model.isRFDETR ? "none" : "class_aware_cpu",
                "nms_iou": model.isRFDETR ? NSNull() : 0.65,
                "box_convention": "unclipped upright source-pixel top-left xywh",
                "preview_enabled": run.previewEnabled, "max_in_flight": 1,
                "max_pending": run.mode != "offline" ? 1 : 0,
                "max_result_age_ms": run.mode != "offline" ? 200 : NSNull(),
                "warmup_frames": 0, "preloaded_frames": false,
                "repetition": NSNull(), "repetition_unavailable_reason": "Ad hoc diagnostic run",
            ],
            "timing": [
                "clock": "process-local monotonic InferenceClock", "unit": "ms",
                "source_time": "exact CMTime value/timescale/flags/epoch",
                "frame_latency": "done - ready; conversion, preprocessing, model, postprocessing",
                "model_latency": "model_completed - model_submitted",
                "excluded_from_frame_latency": ["decoding", "rendering", "export"],
                "null_endpoint_reason": "Stage not reached or endpoint not collected",
                "percentile_rule": "nearest rank ceil(p * N)",
            ],
            "record_order": run.mode == "offline" ? "source order" : "queue outcome log order",
            "frame_records": recordCount,
            "conditions": ["value": NSNull(), "unavailable_reason": "Not collected"],
            "thermal_observations": ["value": NSNull(), "unavailable_reason": "Not collected"],
            "validity": [
                "benchmark_status": "not_validated", "coverage_status": "unverified",
                "parity_status": "pending", "f1": NSNull(),
                "reasons": ["Frozen input coverage has not been reconciled",
                            "Video parity has not been verified",
                            "Diagnostic run without benchmark warm-up or repetitions",
                            "Sustained-run conditions have not been verified"],
                "f1_unavailable_reason": "No video ground truth evaluation",
            ],
        ]
        switch run {
        case .camera(let r):
            value["selection"] = ["first_loop_index": 0, "loop_count": 1,
                "frame_selection": "live capture callbacks with bounded latest-frame queue"]
            value["host_window_ms"] = ["started": r.startedMilliseconds,
                "ended": VideoExportRecord.nullable(r.endedMilliseconds)]
            value["capture"] = CameraExportRecord.capture(r)
            value["counts"] = CameraExportRecord.counts(r)
            value["issue"] = VideoExportRecord.failure(r.issue)
            value["metrics"] = pacedMetrics(r.metrics)
            let conditions: [String: Any] = [
                "scene_notes": r.configuration.sceneNotes.isEmpty ? NSNull() : r.configuration.sceneNotes,
                "unavailable_reason": r.configuration.sceneNotes.isEmpty ? "Not supplied" : NSNull()]
            value["conditions"] = conditions
            value["record_order"] = "queue outcomes followed by source-only capture callbacks"
            value["capture_preview_enabled"] = r.configuration.livePreview
            value["detection_preview_enabled"] = r.configuration.detectionPreview
            value["validity"] = ["benchmark_status": "not_validated", "coverage_status": "unverified",
                "parity_status": "pending", "f1": NSNull(),
                "f1_unavailable_reason": "No live-camera ground truth"]
        case .offline(let r):
            value["selection"] = [
                "first_loop_index": r.loopIndex, "loop_count": 1,
                "frame_indices": VideoExportRecord.nullable(r.selectedFrameIndices),
                "frame_selection": r.selectedFrameIndices == nil ? "all" : "indices",
                "selected_track_id": NSNull(),
                "requested_track_id": VideoExportRecord.nullable(r.requestedTrackID),
                "track_unavailable_reason": "Offline report does not retain track ID",
            ]
            value["host_window_ms"] = [
                "started": r.startedMilliseconds,
                "ended": VideoExportRecord.nullable(r.endedMilliseconds)]
            value["counts"] = ["decoded": r.decodedFrames, "completed": r.completedFrames,
                "failed": r.failedFrames, "unfinished": r.unfinishedFrames,
                "reached_end_of_source": r.reachedEndOfSource]
            value["issue"] = r.issue.map {
                ["stage": $0.stage.rawValue, "details": VideoExportRecord.failure($0.details)]
            } as Any? ?? NSNull()
            value["decode_loss"] = r.decodeFailure.map {
                ["attempted_frame_index": $0.sourceFrameIndex,
                 "pts": VideoExportRecord.time($0.presentationTime),
                 "details": VideoExportRecord.failure(r.decodeFailureDetails),
                 "lost_frame_count": NSNull()] as [String: Any]
            } as Any? ?? NSNull()
            value["metrics"] = offlineMetrics(r)
        case .paced(let r):
            value["selection"] = [
                "first_loop_index": r.firstLoopIndex,
                "loop_count": VideoExportRecord.nullable(r.requestedLoopCount),
                "frame_selection": "source cadence with bounded latest-frame queue",
                "selected_track_id": VideoExportRecord.nullable(r.source?.selectedTrackID),
                "requested_track_id": VideoExportRecord.nullable(r.requestedTrackID),
            ]
            value["host_window_ms"] = ["started": r.startedMilliseconds,
                "ended": VideoExportRecord.nullable(r.endedMilliseconds)]
            value["source"] = pacedSource(r.source)
            value["counts"] = pacedCounts(r)
            value["issue"] = VideoExportRecord.failure(r.issue)
            value["metrics"] = pacedMetrics(r)
        }
        return value
    }

    /// Rebuilds offline rates using their original full-run and ready-to-done scopes.
    private func offlineMetrics(_ r: OfflineVideoRunReport) -> [String: Any] {
        let m = OfflineVideoRunMetrics(report: r)
        return ["model_latency_ms": statistics(m.modelLatency),
            "frame_latency_ms": statistics(m.frameLatency),
            "model_fps": VideoExportRecord.nullable(m.modelFramesPerSecond),
            "end_to_end_fps": VideoExportRecord.nullable(m.endToEndFramesPerSecond),
            "input_fps": VideoExportRecord.nullable(m.inputFramesPerSecond),
            "completed_fps": VideoExportRecord.nullable(m.completedRunFramesPerSecond),
            "rate_scope": "input/completed: whole run; end-to-end: first ready to last done",
            "fresh_fps": NSNull(), "fresh_fps_unavailable_reason": "Offline run is unpaced",
            "display_fps": NSNull(),
            "display_fps_unavailable_reason": "Draw receipts are not retained in run reports"]
    }

    /// Preserves the half-open source window and post-window drain counters.
    private func pacedMetrics(_ r: PacedVideoRunReport) -> Any {
        pacedMetrics(r.metrics)
    }

    /// Shares the same event window and latency calculations with live camera.
    private func pacedMetrics(_ metrics: PacedVideoRunMetrics?) -> Any {
        guard let m = metrics else {
            return ["value": NSNull(), "unavailable_reason": "No paced measurement window"]
        }
        return ["window_start_ms": m.window.startedMilliseconds,
            "window_end_ms": m.window.endedMilliseconds, "window_bounds": "[start,end)",
            "input_frames": m.inputFrames, "completed_frames": m.completedFrames,
            "fresh_results": m.freshResults,
            "post_window_completions": m.postWindowCompletions,
            "post_window_publications": m.postWindowPublications,
            "invalid_publications": m.invalidPublications,
            "input_fps": VideoExportRecord.nullable(m.inputFramesPerSecond),
            "completed_fps": VideoExportRecord.nullable(m.completedFramesPerSecond),
            "fresh_fps": VideoExportRecord.nullable(m.freshResultsPerSecond),
            "model_latency_ms": statistics(m.modelLatency),
            "frame_latency_ms": statistics(m.frameLatency),
            "completed_age_ms": statistics(m.completedAge),
            "published_age_ms": statistics(m.publishedAge),
            "display_fps": NSNull(),
            "display_fps_unavailable_reason": "Draw receipts are not retained in run reports"]
    }

    /// Exports counts already reported by the runner without claiming frozen coverage.
    private func pacedCounts(_ r: PacedVideoRunReport) -> [String: Any] {
        guard let c = r.counts else {
            return ["value": NSNull(), "unavailable_reason": "Queue did not start"]
        }
        return ["decoded": VideoExportRecord.nullable(r.source?.decodedFrames),
            "input": c.inputFrames, "completed": c.completedFrames, "failed": c.failedFrames,
            "unfinished": c.unfinishedFrames, "active": c.activeFrames,
            "pending": c.pendingFrames, "cancelled": c.cancelledFrames,
            "skipped_superseded": c.skippedSupersededFrames,
            "skipped_stale": c.skippedStaleFrames, "published": c.publishedResults,
            "late": c.lateResults, "superseded_publications": c.supersededPublications,
            "cancelled_publications": c.cancelledPublications,
            "failed_publications": c.failedPublications,
            "unreceived_publications": c.unreceivedPublications]
    }

    /// Retains known decoder diagnostics while leaving unknown lost-frame counts null.
    private func pacedSource(_ source: PacedVideoSourceReport?) -> Any {
        guard let s = source else {
            return ["value": NSNull(), "unavailable_reason": "Source did not start"]
        }
        var value: [String: Any] = ["status": s.status.rawValue,
            "decoded_frames": s.decodedFrames, "delivered_frames": s.deliveredFrames,
            "completed_loops": s.completedLoops,
            "anchor_ms": VideoExportRecord.nullable(s.timeline?.anchorMilliseconds),
            "track_start": VideoExportRecord.time(s.timeline?.sourceTimeRange.start),
            "track_duration": VideoExportRecord.time(s.timeline?.sourceTimeRange.duration),
            "ended_ms": VideoExportRecord.nullable(s.endedMilliseconds), "issue": NSNull()]
        if let i = s.issue {
            value["issue"] = ["stage": i.stage.rawValue, "source_id": i.sourceID,
                "loop_index": i.loopIndex,
                "attempted_frame_index": VideoExportRecord.nullable(i.attemptedFrameIndex),
                "pts": VideoExportRecord.time(i.presentationTime),
                "ready_ms": VideoExportRecord.nullable(i.readyMilliseconds),
                "due_ms": VideoExportRecord.nullable(i.dueMilliseconds),
                "observed_ms": VideoExportRecord.nullable(i.observedMilliseconds),
                "cancelled": i.cancelled, "decode_failure": i.decodeFailure,
                "lost_frame_count": NSNull(),
                "lost_frame_count_unavailable_reason": "No frozen frame reconciliation",
                "details": VideoExportRecord.failure(i.details)]
        }
        return value
    }

    /// Uses null percentiles/means for empty samples rather than exporting fake zeros.
    private func statistics(_ s: VideoLatencyStatistics) -> [String: Any] {
        ["sample_count": s.sampleCount, "total": s.totalMilliseconds,
         "mean": VideoExportRecord.nullable(s.meanMilliseconds),
         "p50": VideoExportRecord.nullable(s.p50Milliseconds),
         "p95": VideoExportRecord.nullable(s.p95Milliseconds),
         "unavailable_reason": s.sampleCount == 0 ? "No eligible samples" : NSNull()]
    }

    /// Records the actual binary/runtime without inferring accelerator placement.
    private func runtime() -> [String: Any] {
        #if DEBUG
            let build = "Debug"
        #else
            let build = "Release"
        #endif
        #if targetEnvironment(simulator)
            let platform = "ios_simulator"
        #elseif os(iOS)
            let platform = "ios"
        #else
            let platform = "macos"
        #endif
        var machine = utsname()
        uname(&machine)
        let machineSize = MemoryLayout.size(ofValue: machine.machine)
        let hardware = withUnsafePointer(to: &machine.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: machineSize) {
                String(cString: $0)
            }
        }
        return ["platform": platform, "hardware": hardware,
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
            "backend": "Core ML", "backend_version": NSNull(),
            "backend_version_unavailable_reason": "Core ML is supplied by the recorded OS",
            "build_configuration": build,
            "app_version": VideoExportRecord.nullable(bundle.infoDictionary?["CFBundleShortVersionString"]),
            "app_build": VideoExportRecord.nullable(bundle.infoDictionary?["CFBundleVersion"]),
            "app_revision": VideoExportRecord.nullable(bundle.infoDictionary?["AppRevision"]),
            "app_revision_unavailable_reason": "Available only when AppRevision is embedded at build time",
            "toolchain": VideoExportRecord.nullable(bundle.infoDictionary?["DTXcodeBuild"]),
            "toolchain_unavailable_reason": "Available only when DTXcodeBuild is embedded"]
    }

    /// Hashes the exact input bytes after the run, independently of manifest claims.
    private func inputReference(_ run: VideoRunExport, url: URL) throws -> [String: Any] {
        let actual: String?
        let hashUnavailableReason: String?
        do {
            actual = try Self.fileDigest(url)
            hashUnavailableReason = nil
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // A missing input must not prevent exporting its source-opening failure.
            actual = nil
            hashUnavailableReason = error.localizedDescription
        }
        var value: [String: Any] = ["source_id": run.sourceID,
            "file_name": url.lastPathComponent,
            "sha256": VideoExportRecord.nullable(actual),
            "sha256_unavailable_reason": VideoExportRecord.nullable(hashUnavailableReason),
            "hash_observed": "export time; input immutability during run not verified",
            "manifest": NSNull(), "manifest_unavailable_reason": "No matching bundled input manifest"]
        if let manifest = try manifest("test_videos_manifest", folder: "TestVideos"),
            let samples = manifest.object["samples"] as? [[String: Any]],
            let sample = samples.first(where: { $0["id"] as? String == run.sourceID })
        {
            value["manifest"] = ["resource": manifest.name, "sha256": manifest.hash,
                "sample_id": run.sourceID, "entry": sample,
                "resource_hash_matches": VideoExportRecord.nullable(actual.map {
                    sample["sha256"] as? String == $0
                }),
                "resource_hash_comparison_unavailable_reason":
                    VideoExportRecord.nullable(hashUnavailableReason)]
            value["manifest_unavailable_reason"] = NSNull()
        }
        return value
    }

    /// Labels original package/checkpoint hashes as manifest claims, not runtime hashes.
    private func modelReference(_ model: DetectorModel) throws -> [String: Any] {
        var value: [String: Any] = ["runtime_model_sha256": NSNull(),
            "runtime_model_sha256_unavailable_reason": "Compiled model bytes were not fingerprinted",
            "manifest": NSNull(), "manifest_unavailable_reason": "Model provenance manifest is absent"]
        if let m = try manifest(model.sampleManifestName, folder: "TestImages"),
            m.object["model"] as? String == model.rawValue
        {
            value["manifest"] = ["resource": m.name, "sha256": m.hash,
                "model_sha256": VideoExportRecord.nullable(m.object["model_sha256"]),
                "checkpoint_sha256": VideoExportRecord.nullable(m.object["checkpoint_sha256"]),
                "export_manifest_path": VideoExportRecord.nullable(m.object["export_manifest_path"]),
                "hash_verification": "Provenance claims; runtime model hash not verified"]
            value["manifest_unavailable_reason"] = NSNull()
        }
        return value
    }

    /// Locates either flattened Xcode resources or preserved resource folders.
    private func manifest(_ name: String, folder: String) throws
        -> (name: String, hash: String, object: [String: Any])?
    {
        for subdirectory in [folder, "Resources/\(folder)", nil] as [String?] {
            if let url = bundle.url(forResource: name, withExtension: "json", subdirectory: subdirectory) {
                let data = try Data(contentsOf: url)
                guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                else { throw CocoaError(.propertyListReadCorrupt) }
                return (name + ".json", Self.digest(data), object)
            }
        }
        return nil
    }

    /// Encodes finite JSON without rounding numbers or omitting null-valued fields.
    private static func json(_ object: [String: Any], pretty: Bool = false) throws -> Data {
        try JSONSerialization.data(withJSONObject: object,
                                   options: pretty ? [.sortedKeys, .prettyPrinted] : [.sortedKeys])
    }

    /// Formats UTC wall time separately from every monotonic duration.
    private static func utc(_ date: Date) -> String {
        date.ISO8601Format(.init(includingFractionalSeconds: true))
    }

    /// Hashes exact returned file bytes for integrity checks on the Mac.
    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Streams source hashing with bounded memory and cooperative cancellation.
    private static func fileDigest(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
            try Task.checkCancellation()
            hash.update(data: data)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
