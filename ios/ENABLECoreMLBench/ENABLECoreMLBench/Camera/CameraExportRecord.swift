import CoreMedia
import Foundation

/// Projects camera callbacks and inference outcomes without merging drop/skip meanings.
nonisolated enum CameraExportRecord {
    /// Serializes frozen choices independently of actual negotiated device settings.
    static func configuration(_ c: CameraConfiguration) -> [String: Any] {
        [
            "lens": c.lens.rawValue, "width": Int(c.resolution.width),
            "height": Int(c.resolution.height), "target_fps": c.framesPerSecond,
            "rotation_degrees": c.orientation.rawValue, "mirrored": c.mirrored,
            "live_preview_enabled": c.livePreview,
            "detection_preview_enabled": c.detectionPreview,
            "scene_notes": c.sceneNotes.isEmpty ? NSNull() : c.sceneNotes,
            "scene_notes_unavailable_reason": c.sceneNotes.isEmpty
                ? "Not supplied" : NSNull(),
        ]
    }

    /// Keeps the actual synchronization calibration, never a first-callback time guess.
    static func capture(_ r: CameraRunReport) -> [String: Any] {
        let s = r.capture
        let fps = r.window.flatMap { window -> Double? in
            guard window.durationMilliseconds > 0 else { return nil }
            let count =
                s?.events.filter {
                    $0.kind == .delivered && window.contains($0.receivedMilliseconds)
                }.count ?? 0
            return Double(count) * 1000 / window.durationMilliseconds
        }
        return [
            "configuration": configuration(r.configuration),
            "actual": s?.actual ?? [:], "source_id": r.sourceID,
            "frame_index_kind":
                "observed capture callback ordinal, including drop callbacks",
            "started_ms": VideoExportRecord.nullable(s?.startedMilliseconds),
            "ended_ms": VideoExportRecord.nullable(s?.endedMilliseconds),
            "timestamp_mapping": s?.mapping.map { m in
                [
                    "method":
                        "CMSyncConvertTime(session synchronizationClock -> host clock), then per-run host anchor",
                    "host_anchor": VideoExportRecord.time(m.hostAnchor),
                    "inference_anchor_ms": m.inferenceAnchorMilliseconds,
                    "calibration_uncertainty_ms": m.uncertaintyMilliseconds,
                ] as [String: Any]
            } as Any? ?? NSNull(),
            "mapping_unavailable_reason": s?.mapping == nil
                ? "Capture did not establish a clock" : NSNull(),
            "issue": VideoExportRecord.failure(s?.issue),
            "device_observation_semantics":
                "Exposure/ISO/focus observed at callback time, not sensor exposure metadata",
            "delivered_callbacks": s?.delivered ?? 0, "drop_events": s?.dropEvents ?? 0,
            "actual_delivered_fps": VideoExportRecord.nullable(fps),
            "observed_callbacks": s?.events.count ?? 0,
        ]
    }

    /// Copies actual callback identity, clock endpoints and automatic camera observations.
    private static func attach(
        _ event: CameraCaptureEvent?, to row: inout [String: Any]
    ) {
        row["capture_admitted"] = event?.admitted ?? false
        row["capture_cancelled"] = event?.cancelled ?? false
        row["capture_admitted_ms"] = VideoExportRecord.nullable(
            event?.deliveredMilliseconds)
        row["capture_kind"] = event?.kind.rawValue ?? "missing_callback"
        row["capture_pts"] = VideoExportRecord.time(event?.presentationTime)
        row["capture_host_pts"] = VideoExportRecord.time(event?.hostPresentationTime)
        row["capture_received_ms"] = VideoExportRecord.nullable(
            event?.receivedMilliseconds)
        row["capture_drop_reason"] = VideoExportRecord.nullable(event?.reason)
        row["capture_lost_frames"] = VideoExportRecord.nullable(event?.knownLostFrames)
        row["capture_observation"] = event?.observation ?? [:]
    }

    /// Writes queue outcomes first, followed by source-only drops/failures/unfinished inputs.
    static func forEachFrame(_ r: CameraRunReport, body: ([String: Any]) throws -> Void)
        rethrows
    {
        let events = r.capture?.events ?? []
        let byIndex = Dictionary(
            events.map { ($0.index, $0) }, uniquingKeysWith: { first, _ in first })
        var recorded = Set<Int>()
        for o in r.queue?.outcomes ?? [] {
            let index = o.metadata.sourceFrameIndex
            var row = VideoExportRecord.frame(
                sourceID: r.sourceID, loop: 0, index: index,
                metadata: o.metadata, status: o.status.rawValue, result: o.result,
                ready: o.readyMilliseconds, due: o.dueMilliseconds,
                delivered: o.deliveredMilliseconds,
                decided: o.decidedMilliseconds, issue: o.failure,
                publication: o.publication)
            row["record_type"] = "inference"
            attach(byIndex[index], to: &row)
            try body(row)
            recorded.insert(index)
        }
        for event in events where !recorded.contains(event.index) {
            let status =
                event.kind == .dropped
                ? "capture_dropped"
                : event.failure == nil || event.admitted || event.cancelled
                    ? "unfinished" : "capture_failed"
            var row = VideoExportRecord.frame(
                sourceID: r.sourceID, loop: 0, index: event.index,
                metadata: nil, status: status, result: nil,
                ready: event.kind == .delivered ? event.receivedMilliseconds : nil,
                due: event.capturedMilliseconds, delivered: event.deliveredMilliseconds,
                issue: event.failure, stage: "capture")
            row["record_type"] = "capture"
            row["publication_status"] = "not_applicable"
            if event.presentationTime.isNumeric {
                row["pts"] = VideoExportRecord.time(event.presentationTime)
                row["pts_unavailable_reason"] = NSNull()
            }
            attach(event, to: &row)
            try body(row)
        }
    }

    /// Requires every observed callback and every accepted queue input to reconcile.
    static func coverage(_ r: CameraRunReport) -> [String: Any] {
        let events = r.capture?.events ?? []
        let outcomes = r.queue?.outcomes ?? []
        let admitted = events.filter(\.admitted).map(\.index)
        let completed = outcomes.map { $0.metadata.sourceFrameIndex }
        let c = r.counts
        let balanced =
            Set(admitted) == Set(completed) && admitted.count == completed.count
            && Set(completed).count == completed.count
            && c.inputFrames == admitted.count
            && c.inputFrames == c.completedFrames + c.failedFrames + c.skippedFrames
                + c.unfinishedFrames
            && events.enumerated().allSatisfy { $0.offset == $0.element.index }
        var reasons: [String] = []
        if !balanced {
            reasons.append("Capture callbacks and queue outcomes do not reconcile")
        }
        if r.capture?.mapping == nil {
            reasons.append("Capture clock mapping is unavailable")
        }
        if r.capture?.hasUnknownLoss == true {
            reasons.append("Capture loss or source failure cannot be reconstructed")
        }
        if c.unfinishedFrames > 0 {
            reasons.append("Inference work was unfinished at Stop")
        }
        if events.contains(where: { $0.kind == .delivered && !$0.admitted }) {
            reasons.append("Delivered capture work was not admitted to the queue")
        }
        if events.isEmpty { reasons.append("No capture callbacks were observed") }
        if r.capture?.isRunning == true { reasons.append("Capture has not finalized") }
        return [
            "scope": "observed_capture_callbacks",
            "status": reasons.isEmpty ? "complete" : "incomplete",
            "sensor_coverage_status": "unverified", "expected_sensor_frames": NSNull(),
            "sensor_coverage_reason":
                "No independent sensor sequence exists; unreported losses cannot be inferred",
            "accounting_balanced": balanced, "capture_callbacks": events.count,
            "delivered_callbacks": r.capture?.delivered ?? 0,
            "capture_drop_events": r.capture?.dropEvents ?? 0,
            "queue_inputs": c.inputFrames, "completed": c.completedFrames,
            "failed": c.failedFrames, "skipped": c.skippedFrames,
            "unfinished": c.unfinishedFrames,
            "reasons": reasons,
        ]
    }

    /// Keeps all queue categories alongside camera-system drop counters.
    static func counts(_ r: CameraRunReport) -> [String: Any] {
        let c = r.counts
        return [
            "input": c.inputFrames, "completed": c.completedFrames,
            "failed": c.failedFrames,
            "unfinished": c.unfinishedFrames, "active": c.activeFrames,
            "pending": c.pendingFrames,
            "cancelled": c.cancelledFrames,
            "skipped_superseded": c.skippedSupersededFrames,
            "skipped_stale": c.skippedStaleFrames, "published": c.publishedResults,
            "late": c.lateResults, "superseded_publications": c.supersededPublications,
            "cancelled_publications": c.cancelledPublications,
            "failed_publications": c.failedPublications,
            "unreceived_publications": c.unreceivedPublications,
            "capture_callbacks": r.capture?.events.count ?? 0,
            "capture_delivered": r.capture?.delivered ?? 0,
            "capture_drop_events": r.capture?.dropEvents ?? 0,
        ]
    }
}
