import SwiftUI

/// Keeps detector accuracy and measured processing speed visible in one summary.
struct VideoAccuracyView: View {
    let session: VideoAccuracySession
    let disabled: Bool
    let chooseAnnotations: () -> Void
    let removeAnnotations: () -> Void

    var body: some View {
        Section {
            Button("Import annotations JSON", systemImage: "doc.badge.plus",
                   action: chooseAnnotations)
                .disabled(disabled)
                .accessibilityIdentifier("importVideoAnnotations")
            if let imported = session.imported {
                Text(imported.fileName).font(.footnote)
                LabeledContent("Annotated frames",
                               value: String(imported.annotations.video.frame_count))
                Button("Remove annotations", role: .destructive, action: removeAnnotations)
                    .disabled(disabled)
            }
            if session.isBusy { ProgressView(session.message ?? "Preparing accuracy…") }
            else if let message = session.message {
                Text(message).font(.footnote).foregroundStyle(.secondary)
            }
            if let result = session.result, let report = result.report {
                LabeledContent(report.scope == "all_video_frames" ? "F1 (all frames)"
                    : "F1 (processed frames)", value: number(report.f1))
                    .accessibilityIdentifier("videoF1")
                LabeledContent("Precision", value: number(report.precision))
                LabeledContent("Recall", value: number(report.recall))
                LabeledContent("Detection FPS", value: report.detectionFPS.map {
                    $0.formatted(.number.precision(.fractionLength(2)))
                } ?? "—")
                    .accessibilityIdentifier("videoBenchmarkFPS")
                LabeledContent(report.scope == "all_video_frames"
                    ? "Measured processing time" : "Playback window",
                    value: report.processingSeconds.map {
                        $0.formatted(.number.precision(.fractionLength(2))) + " s"
                    } ?? "—")
                LabeledContent("Frame coverage", value: report.frameCoverage.formatted(
                    .percent.precision(.fractionLength(1))))
                    .accessibilityIdentifier("videoAccuracyCoverage")
                LabeledContent("Scored frames",
                               value: "\(report.scoredFrames) / \(report.expectedFrames)")
                LabeledContent("True / false positives",
                               value: "\(report.truePositives) / \(report.falsePositives)")
                LabeledContent("False negatives", value: String(report.falseNegatives))
                LabeledContent("Failed / skipped frames",
                               value: "\(report.failedFrames) / \(report.skippedFrames)")
                if report.postWindowFrames > 0 {
                    LabeledContent("Completed after playback",
                                   value: String(report.postWindowFrames))
                }
                LabeledContent("Untimed warm-up", value: "\(result.warmupFrames) predictions")
            }
        } header: { Text("Video accuracy") } footer: {
            Text("Import the JSON paired with the clean MP4 from Files. F1 uses confidence "
                + "≥25% and IoU ≥0.5. Offline scores every frame, counting failed inference "
                + "as empty predictions. Paced scores completed frames within playback; "
                + "read F1 together with coverage. Scoring and export happen after timing.")
        }
        .monospacedDigit()
    }

    /// Displays unrounded counts alongside consistently formatted accuracy ratios.
    private func number(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(4)))
    }
}
