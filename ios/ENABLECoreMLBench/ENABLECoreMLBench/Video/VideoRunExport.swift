import CoreGraphics
import CoreMedia
import Foundation

/// Freezes one finished numerical report without retaining decoded images.
nonisolated enum VideoRunExport: Sendable {
    case offline(OfflineVideoRunReport)
    case paced(PacedVideoRunReport)

    var sourceID: String {
        switch self { case .offline(let r): r.sourceID; case .paced(let r): r.sourceID }
    }
    var model: DetectorModel {
        switch self { case .offline(let r): r.model; case .paced(let r): r.model }
    }
    var mode: String { switch self { case .offline: "offline"; case .paced: "paced" } }
    var id: UUID {
        switch self { case .offline(let r): r.id; case .paced(let r): r.id }
    }
    var startedAt: Date {
        switch self { case .offline(let r): r.startedAt; case .paced(let r): r.startedAt }
    }
    var endedAt: Date? {
        switch self { case .offline(let r): r.endedAt; case .paced(let r): r.endedAt }
    }
    var status: String {
        switch self {
        case .offline(let r): r.status.rawValue
        case .paced(let r): r.status.rawValue
        }
    }
    var previewEnabled: Bool {
        switch self {
        case .offline(let r): r.previewEnabled
        case .paced(let r): r.previewEnabled
        }
    }
}

/// Holds only the three completed files; a failed export never exposes partial files.
nonisolated struct VideoExportFiles: Sendable {
    let directory: URL
    var urls: [URL] {
        ["frames.jsonl", "timings.csv", "mobile_run.json"].map {
            directory.appendingPathComponent($0)
        }
    }
}

/// Builds schema-v1 JSON/CSV using explicit nulls and exact CMTime integers.
nonisolated enum VideoExportRecord {
    /// Keeps absent values distinct from zero, empty predictions and invalid PTS.
    static func nullable<T>(_ value: T?) -> Any { value.map { $0 as Any } ?? NSNull() }

    /// Preserves all CMTime fields rather than rounding to floating-point seconds.
    static func time(_ value: CMTime?) -> Any {
        guard let value else { return NSNull() }
        return ["value": value.value, "timescale": Int64(value.timescale),
                "flags": Int64(value.flags.rawValue), "epoch": value.epoch]
    }

    /// Exports the original affine mapping in Core Graphics component order.
    static func transform(_ value: CGAffineTransform?) -> Any {
        guard let value else { return NSNull() }
        return [value.a, value.b, value.c, value.d, value.tx, value.ty]
    }

    /// Copies error diagnostics without backend objects or image buffers.
    static func failure(_ value: VideoInferenceFailure?) -> Any {
        guard let value else { return NSNull() }
        return ["type": value.type, "message": value.message]
    }

    /// Uses the same upright, unclipped source-pixel xywh convention as still images.
    static func frame(
        sourceID: String, loop: Int, index: Int, metadata: VideoFrameMetadata?,
        status: String, result: VideoFrameResult?, ready: Double?,
        due: Double? = nil, delivered: Double? = nil, decided: Double? = nil,
        recorded: Double? = nil, issue: VideoInferenceFailure? = nil,
        stage: String? = nil, publication: PacedVideoPublicationOutcome? = nil
    ) -> [String: Any] {
        let timing = result?.timings
        let published = publication?.publishedMilliseconds ?? timing?.publishedMilliseconds
        return [
            "schema_version": 1, "source_id": sourceID, "loop_index": loop,
            "source_frame_index": index, "pts": time(metadata?.presentationTime),
            "pts_unavailable_reason": metadata == nil ? "No decoded sample metadata" : NSNull(),
            "encoded_width": nullable(metadata?.width),
            "encoded_height": nullable(metadata?.height),
            "preferred_transform": transform(metadata?.preferredTransform),
            "width": nullable(result?.sourceWidth), "height": nullable(result?.sourceHeight),
            "buffer_to_image": transform(result?.bufferToImage),
            "status": status, "failure": failure(issue ?? result?.failure),
            "failure_stage": nullable(stage),
            "detections": result.map { r in
                r.detections.map { d -> [String: Any] in
                    ["category_id": d.classID, "score": d.confidence,
                     "bbox": [d.boundingBox.minX, d.boundingBox.minY,
                              d.boundingBox.width, d.boundingBox.height]]
                }
            } as Any? ?? NSNull(),
            "detections_unavailable_reason": result == nil ? "Inference did not return a result" : NSNull(),
            "publication_status": publication?.status.rawValue
                ?? (published == nil ? "not_recorded" : "published"),
            "publication_failure": failure(publication?.failure),
            "timing_ms": [
                "ready": nullable(ready), "due": nullable(due),
                "delivered": nullable(delivered), "decided": nullable(decided),
                "model_submitted": nullable(timing?.modelTiming?.submittedMilliseconds),
                "model_completed": nullable(timing?.modelTiming?.completedMilliseconds),
                "done": nullable(timing?.doneMilliseconds), "recorded": nullable(recorded),
                "publication_checked": nullable(publication?.checkedMilliseconds),
                "published": nullable(published),
                "model_latency": nullable(timing?.modelLatencyMilliseconds),
                "frame_latency": nullable(timing?.endToEndMilliseconds),
                "completed_age": nullable(due.flatMap { due in
                    result?.status == .completed ? timing!.doneMilliseconds - due : nil
                }),
                "published_age": nullable(due.flatMap { due in published.map { $0 - due } }),
            ],
        ]
    }

    /// Preserves source order offline and logged decision/completion order when paced.
    static func forEachFrame(
        _ run: VideoRunExport, body: ([String: Any]) throws -> Void
    ) rethrows {
        switch run {
        case .offline(let report):
            for outcome in report.outcomes {
                try body(frame(sourceID: report.sourceID, loop: report.loopIndex,
                    index: outcome.sourceFrameIndex, metadata: outcome.metadata,
                    status: outcome.status.rawValue, result: outcome.result,
                    ready: outcome.readyMilliseconds, recorded: outcome.recordedMilliseconds,
                    issue: outcome.issue?.details, stage: outcome.issue?.stage.rawValue))
            }
        case .paced(let report):
            for outcome in report.queue?.outcomes ?? [] {
                try body(frame(sourceID: outcome.metadata.sourceID,
                    loop: outcome.metadata.loopIndex, index: outcome.metadata.sourceFrameIndex,
                    metadata: outcome.metadata, status: outcome.status.rawValue,
                    result: outcome.result, ready: outcome.readyMilliseconds,
                    due: outcome.dueMilliseconds, delivered: outcome.deliveredMilliseconds,
                    decided: outcome.decidedMilliseconds,
                    issue: outcome.failure, publication: outcome.publication))
            }
        }
    }

    static let timingNames = [
        "ready", "due", "delivered", "decided", "model_submitted", "model_completed",
        "done", "recorded", "publication_checked", "published", "model_latency",
        "frame_latency", "completed_age", "published_age",
    ]
    static let csvColumns = [
        "run_id", "source_id", "loop_index", "source_frame_index", "pts_value",
        "pts_timescale", "pts_flags", "pts_epoch", "status", "publication_status",
    ] + timingNames.map { "\($0)_ms" }

    /// Quotes RFC 4180 fields, including commas, quotes and embedded newlines.
    static func csvField(_ value: Any?) -> String {
        guard let value, !(value is NSNull) else { return "" }
        let string = String(describing: value)
        if string.contains(where: { ",\"\r\n".contains($0) }) {
            return "\"" + string.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        return string
    }

    /// Leaves missing timing endpoints blank; no missing duration becomes zero.
    static func csvRow(_ row: [String: Any], runID: String) -> String {
        let pts = row["pts"] as? [String: Any] ?? [:]
        let timing = row["timing_ms"] as? [String: Any] ?? [:]
        let values: [Any?] = [runID, row["source_id"], row["loop_index"],
            row["source_frame_index"], pts["value"], pts["timescale"], pts["flags"],
            pts["epoch"], row["status"], row["publication_status"]]
            + timingNames.map { timing[$0] }
        return values.map { csvField($0) }.joined(separator: ",") + "\r\n"
    }
}
