import CoreGraphics
import CoreMedia
import Foundation

/// Contains the same fixed-threshold counts used by the shared Python evaluator.
nonisolated struct VideoAccuracyReport: Encodable, Sendable {
    let scope: String
    let expectedFrames: Int
    let scoredFrames: Int
    let completedFrames: Int
    let failedFrames: Int
    let skippedFrames: Int
    let postWindowFrames: Int
    let truePositives: Int
    let falsePositives: Int
    let falseNegatives: Int
    let detectionFPS: Double?
    let fpsScope: String
    let processingSeconds: Double?
    let confidenceThreshold: Double = 0.25
    let iouThreshold: Double = 0.5
    let maximumDetections: Int = 100

    var frameCoverage: Double { Double(completedFrames) / Double(expectedFrames) }
    var precision: Double {
        Double(truePositives) / Double(max(1, truePositives + falsePositives))
    }
    var recall: Double {
        Double(truePositives) / Double(max(1, truePositives + falseNegatives))
    }
    var f1: Double {
        let denominator = 2 * truePositives + falsePositives + falseNegatives
        return Double(2 * truePositives) / Double(max(1, denominator))
    }

    /// Includes computed metrics as well as raw counts in the export.
    func json() throws -> [String: Any] {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        var value = try JSONSerialization.jsonObject(with: encoder.encode(self))
            as! [String: Any]
        value["f1"] = f1
        value["precision"] = precision
        value["recall"] = recall
        value["frame_coverage"] = frameCoverage
        value["failed_frame_policy"] = scope == "all_video_frames"
            ? "empty_predictions" : "excluded_from_f1"
        value["matching"] = "score-descending greedy, same category, unmatched maximum IoU"
        return value
    }
}

/// Keeps both successful scoring and unavailable reasons linked to one finished run.
nonisolated struct VideoAccuracyExport: Sendable {
    let runID: UUID
    let annotations: ImportedVideoAnnotations
    let report: VideoAccuracyReport?
    let unavailableReason: String?
    let warmupFrames: Int

    /// Keeps generation details out of evaluation reports while retaining input hashes.
    func json() throws -> [String: Any] {
        ["schema_version": 1, "status": report == nil ? "unavailable" : "completed",
         "annotation_file": annotations.fileName, "annotation_sha256": annotations.sha256,
         "video_sha256": annotations.annotations.video.sha256,
         "warmup_frames": warmupFrames,
         "unavailable_reason": VideoExportRecord.nullable(unavailableReason),
         "metrics": try report.map { try $0.json() } ?? NSNull()]
    }
}

/// Scores source-frame identities after inference; never uses preview/delivery order.
nonisolated enum VideoAccuracyEvaluator {
    /// Requires a completed source pass; paced F1 shares its FPS playback window.
    static func evaluate(
        _ run: VideoRunExport, annotations: VideoAnnotations
    ) throws -> VideoAccuracyReport {
        guard run.status == "completed" else {
            throw VideoBenchmarkError(
                "F1 requires a completed video run; this run was interrupted.")
        }
        let expected = annotations.video.frame_count
        var seen = Set<Int>()
        var results: [Int: VideoFrameResult] = [:]
        var failures = 0, skipped = 0, postWindow = 0
        let scope: String
        let fps: Double?
        let seconds: Double?
        let fpsScope: String
        switch run {
        case .offline(let report):
            guard report.reachedEndOfSource, report.selectedFrameIndices == nil,
                report.loopIndex == 0, report.decodedFrames == expected,
                report.unfinishedFrames == 0 else {
                throw VideoBenchmarkError("Offline F1 requires every annotated video frame.")
            }
            for outcome in report.outcomes {
                guard seen.insert(outcome.sourceFrameIndex).inserted else {
                    throw VideoBenchmarkError("Duplicate source frame in the video results.")
                }
                if let result = outcome.result, outcome.status == .completed {
                    results[outcome.sourceFrameIndex] = result
                } else { failures += 1 }
            }
            let metrics = OfflineVideoRunMetrics(report: report)
            fps = metrics.endToEndFramesPerSecond
            seconds = metrics.endToEndDurationMilliseconds.map { $0 / 1000 }
            scope = "all_video_frames"
            fpsScope = "first_frame_ready_to_last_completed_detection"
        case .paced(let report):
            guard let source = report.source, source.status == .completed,
                source.completedLoops == 1, report.firstLoopIndex == 0,
                report.requestedLoopCount == 1, source.decodedFrames == expected,
                source.deliveredFrames == expected, report.decodeLosses.isEmpty,
                let queue = report.queue, queue.state == .completed,
                let window = report.measurementWindow else {
                throw VideoBenchmarkError("Paced F1 requires a complete, matched video pass.")
            }
            for outcome in queue.outcomes {
                guard outcome.metadata.loopIndex == 0,
                    seen.insert(outcome.metadata.sourceFrameIndex).inserted else {
                    throw VideoBenchmarkError("Duplicate or looped source frame in paced results.")
                }
                switch outcome.status {
                case .completed:
                    guard let result = outcome.result, result.status == .completed else {
                        throw VideoBenchmarkError("Completed frame has no successful result.")
                    }
                    if window.contains(result.timings.doneMilliseconds) {
                        results[outcome.metadata.sourceFrameIndex] = result
                    } else { postWindow += 1 }
                case .failed: failures += 1
                case .skippedStale, .skippedSuperseded: skipped += 1
                case .cancelled:
                    throw VideoBenchmarkError("Paced run contains unfinished frames.")
                }
            }
            fps = report.metrics?.completedFramesPerSecond
            seconds = window.durationMilliseconds / 1000
            scope = "processed_frames_within_playback"
            fpsScope = "completed_detections_within_source_playback_window"
        case .camera:
            throw VideoBenchmarkError("Video annotations cannot score live camera input.")
        }
        guard seen == Set(0..<expected) else {
            throw VideoBenchmarkError("Video results have missing or unexpected frame indices.")
        }
        let frames = Dictionary(uniqueKeysWithValues:
            annotations.images.map { ($0.frame_index, $0) })
        let boxes = Dictionary(grouping: annotations.annotations, by: \.image_id)
        var tp = 0, fp = 0, fn = 0, scored = 0
        for index in 0..<expected {
            try Task.checkCancellation()
            let result = results[index]
            if scope != "all_video_frames" && result == nil { continue }
            if let result {
                guard result.sourceWidth == annotations.video.width,
                    result.sourceHeight == annotations.video.height,
                    result.metadata.sourceFrameIndex == index,
                    result.metadata.sourceID == run.sourceID,
                    abs(result.metadata.presentationTime.seconds
                        - frames[index]!.timestamp_seconds) < 0.0001 else {
                    throw VideoBenchmarkError("Frame dimensions, identity or timestamps differ.")
                }
            }
            let actual = boxes[frames[index]!.id] ?? []
            let predicted = (result?.detections ?? []).enumerated()
                .filter { $0.element.confidence >= 0.25 }
                .sorted { a, b in
                    a.element.confidence == b.element.confidence
                        ? a.offset < b.offset : a.element.confidence > b.element.confidence
                }.prefix(100).map(\.element)
            var matched = Set<Int>()
            for detection in predicted {
                var best = -1, overlap = -1.0
                for (targetIndex, target) in actual.enumerated()
                    where !matched.contains(targetIndex)
                        && target.category_id == detection.classID {
                    let candidate = iou(detection.boundingBox, target.bbox)
                    // Match the Python evaluator's last-index tie break.
                    if candidate >= overlap { best = targetIndex; overlap = candidate }
                }
                if best >= 0 && overlap >= 0.5 { matched.insert(best); tp += 1 }
                else { fp += 1 }
            }
            fn += actual.count - matched.count
            scored += 1
        }
        guard scored > 0 else {
            throw VideoBenchmarkError("No completed frames are available for F1 scoring.")
        }
        return VideoAccuracyReport(
            scope: scope, expectedFrames: expected, scoredFrames: scored,
            completedFrames: results.count, failedFrames: failures,
            skippedFrames: skipped, postWindowFrames: postWindow,
            truePositives: tp, falsePositives: fp, falseNegatives: fn,
            detectionFPS: fps, fpsScope: fpsScope, processingSeconds: seconds)
    }

    /// Uses continuous source-pixel xywh geometry without display clipping.
    private static func iou(_ first: CGRect, _ second: [Double]) -> Double {
        let x = max(Double(first.minX), second[0])
        let y = max(Double(first.minY), second[1])
        let right = min(Double(first.maxX), second[0] + second[2])
        let bottom = min(Double(first.maxY), second[1] + second[3])
        let intersection = max(0, right - x) * max(0, bottom - y)
        let union = Double(first.width * first.height) + second[2] * second[3] - intersection
        return union > 0 ? intersection / union : 0
    }
}
