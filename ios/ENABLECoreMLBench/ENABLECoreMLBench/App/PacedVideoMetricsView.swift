import SwiftUI

/// Separates paced input, completion, receipt and drawing scopes in the video UI.
struct PacedVideoMetricsView: View {
    let session: PacedVideoSession
    let accuracy: VideoAccuracyReport?

    var body: some View {
        Group {
            Section {
                LabeledContent("Processing mode", value: "Paced")
                LabeledContent("Build configuration", value: buildConfiguration)
                LabeledContent("Preview during run", value: session.previewEnabledForRun
                    .map { $0 ? "On" : "Off" } ?? "—")
                LabeledContent("Source window", value: number(
                    session.metrics.map { $0.window.durationMilliseconds / 1000 },
                    unit: "s"))
                LabeledContent("Decoded frames", value: String(session.decodedFrames))
                LabeledContent("Input frames", value: String(session.counts.inputFrames))
                LabeledContent("Processed frames",
                               value: String(session.counts.processedFrames))
                LabeledContent("Completed frames",
                               value: String(session.counts.completedFrames))
                LabeledContent("Failed frames", value: String(session.counts.failedFrames))
                LabeledContent("Unfinished work",
                               value: String(session.counts.unfinishedFrames))
            } header: { Text("Paced video run") } footer: {
                Text("One pass at source cadence. Annotated benchmarks warm up separately. "
                    + "Counts and rates update during the run.")
            }
            Section {
                LabeledContent("Skipped frames",
                               value: String(session.counts.skippedFrames))
                    .accessibilityIdentifier("pacedSkippedFrames")
                LabeledContent("Replaced by newer input",
                               value: String(session.counts.skippedSupersededFrames))
                LabeledContent("Stale before inference",
                               value: String(session.counts.skippedStaleFrames))
                LabeledContent("Fresh receipts (total)",
                               value: String(session.counts.publishedResults))
                LabeledContent("Late results", value: String(session.counts.lateResults))
                LabeledContent("Superseded results",
                               value: String(session.counts.supersededPublications))
                LabeledContent("Delivery failures",
                               value: String(session.counts.failedPublications))
                LabeledContent("Observed decoder errors",
                               value: String(session.decodeErrorEvents))
                if session.decodeErrorEvents > 0 {
                    LabeledContent("Lost-frame count", value: "Unknown")
                }
                LabeledContent("Source coverage", value: accuracy == nil
                    ? "Unverified" : "Matched to annotations")
            } header: { Text("Paced outcomes") } footer: {
                Text("Skipped frames never entered inference. Late or superseded results "
                    + "completed inference without fresh delivery. Published empty results "
                    + "count as fresh. Decode errors may hide an unknown number of frames; "
                    + "annotated runs verify source indices against the imported labels.")
            }
            Section {
                LabeledContent("Input FPS", value: number(
                    session.metrics?.inputFramesPerSecond, unit: "FPS"))
                    .accessibilityIdentifier("pacedInputFPS")
                LabeledContent("Completed FPS", value: number(
                    session.metrics?.completedFramesPerSecond, unit: "FPS"))
                    .accessibilityIdentifier("pacedCompletedFPS")
                LabeledContent("Fresh-result FPS", value: number(
                    session.metrics?.freshResultsPerSecond, unit: "FPS"))
                    .accessibilityIdentifier("pacedFreshFPS")
                LabeledContent("Preview frames drawn",
                               value: String(session.previewStatistics.drawnFrames))
                LabeledContent("Preview FPS (drawn)", value: previewRate)
                    .accessibilityIdentifier("pacedPreviewFPS")
            } header: { Text("Paced video rates") } footer: {
                Text("Input, completed, fresh-result and preview rates use the source window, "
                    + "excluding model loading and post-window drain work. Preview FPS uses "
                    + "distinct drawing receipts; physical screen refresh is unmeasured.")
            }
            Section {
                LabeledContent("Model mean", value: number(
                    session.metrics?.modelLatency.meanMilliseconds, unit: "ms"))
                LabeledContent("Model p50 / p95", value: percentiles(
                    session.metrics?.modelLatency))
                LabeledContent("Frame mean", value: number(
                    session.metrics?.frameLatency.meanMilliseconds, unit: "ms"))
                LabeledContent("Frame p50 / p95", value: percentiles(
                    session.metrics?.frameLatency))
                LabeledContent("Completed age p50 / p95", value: percentiles(
                    session.metrics?.completedAge))
                LabeledContent("Published age p50 / p95", value: percentiles(
                    session.metrics?.publishedAge))
            } header: { Text("Paced video latency") } footer: {
                Text("Latency statistics appear when the run ends. Frame latency includes "
                    + "conversion, queueing and detection. Result age starts at the scheduled "
                    + "source time. Drawing is separate from inference timing.")
            }
        }
        .monospacedDigit()
    }

    private var previewRate: String {
        guard let enabled = session.previewEnabledForRun else { return "—" }
        return enabled ? number(session.displayFramesPerSecond, unit: "FPS") : "Off"
    }
    private var buildConfiguration: String {
        #if DEBUG
        "Debug"
        #else
        "Release"
        #endif
    }

    /// Keeps unavailable measurements distinct from measured zero throughput.
    private func number(_ value: Double?, unit: String) -> String {
        guard let value, value.isFinite else { return "—" }
        return value.formatted(.number.precision(.fractionLength(2))) + " \(unit)"
    }

    /// Shows finalized nearest-rank percentiles without sorting on the UI actor.
    private func percentiles(_ value: VideoLatencyStatistics?) -> String {
        guard let p50 = value?.p50Milliseconds, let p95 = value?.p95Milliseconds else {
            return "—"
        }
        return number(p50, unit: "") + "/ " + number(p95, unit: "ms")
    }
}
