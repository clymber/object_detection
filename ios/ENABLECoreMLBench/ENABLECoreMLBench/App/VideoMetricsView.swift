import SwiftUI

/// Labels streamed-run timing scopes without presenting them as formal benchmarks.
struct VideoMetricsView: View {
    let metrics: OfflineVideoRunMetrics
    let completedFrames: Int
    let failedFrames: Int
    let unfinishedFrames: Int

    var body: some View {
        Group {
            Section {
                LabeledContent(
                    "Build configuration", value: Self.buildConfiguration)
                LabeledContent(
                    "Run time", value: number(
                        metrics.runElapsedMilliseconds.map { $0 / 1000 }, unit: "s"))
                    .accessibilityIdentifier("videoRunTime")
                LabeledContent("Completed frames", value: String(completedFrames))
                LabeledContent("Failed frames", value: String(failedFrames))
                LabeledContent("Unfinished frames", value: String(unfinishedFrames))
            } header: {
                Text("Video run")
            } footer: {
                Text(
                    "Diagnostic run: every frame is included, with no warm-up removed. "
                        + "Run time includes setup, model loading, decoding "
                        + "and cleanup."
                )
            }
            Section {
                LabeledContent(
                    "Model mean", value: number(
                        metrics.modelLatency.meanMilliseconds, unit: "ms"))
                    .accessibilityIdentifier("videoModelLatency")
                LabeledContent(
                    "Model p50 / p95", value: percentiles(metrics.modelLatency))
                LabeledContent(
                    "Frame mean", value: number(
                        metrics.frameLatency.meanMilliseconds, unit: "ms"))
                    .accessibilityIdentifier("videoFrameLatency")
                LabeledContent(
                    "Frame p50 / p95", value: percentiles(metrics.frameLatency))
                LabeledContent(
                    "Frames with model timing",
                    value: "\(metrics.modelLatency.sampleCount) / \(completedFrames)")
            } header: {
                Text("Video latency")
            } footer: {
                Text(
                    "Successful frames include empty detections. Frame latency covers "
                        + "pixel conversion, preprocessing, prediction and "
                        + "postprocessing; video decoding and drawing are excluded. "
                        + "Percentiles appear when the run ends."
                )
            }
            Section {
                LabeledContent(
                    "Model FPS",
                    value: number(metrics.modelFramesPerSecond, unit: "FPS"))
                LabeledContent(
                    "End-to-end FPS",
                    value: number(metrics.endToEndFramesPerSecond, unit: "FPS"))
                    .accessibilityIdentifier("videoEndToEndFPS")
                LabeledContent(
                    "Input FPS (decoded)",
                    value: number(metrics.inputFramesPerSecond, unit: "FPS"))
                LabeledContent(
                    "Completed FPS (run)",
                    value: number(metrics.completedRunFramesPerSecond, unit: "FPS"))
                    .accessibilityIdentifier("videoCompletedFPS")
                LabeledContent("Display FPS", value: "Unmeasured")
            } header: {
                Text("Video throughput")
            } footer: {
                Text(
                    "Model FPS uses total measured model time. End-to-end FPS includes "
                        + "gaps and decoding between frames. Input and completed run "
                        + "rates share the full run time and appear at run end. "
                        + "Playback display FPS is separate and unmeasured."
                )
            }
        }
        .monospacedDigit()
    }

    /// Shows the build being measured so Debug/Release comparisons stay explicit.
    private static var buildConfiguration: String {
        #if DEBUG
        "Debug"
        #else
        "Release"
        #endif
    }

    /// Keeps missing or invalid measurements visibly unavailable rather than zero.
    private func number(_ value: Double?, unit: String) -> String {
        guard let value, value.isFinite else { return "—" }
        return value.formatted(.number.precision(.fractionLength(2))) + " \(unit)"
    }

    /// Presents protocol nearest-rank percentiles only after worker finalization.
    private func percentiles(_ statistics: VideoLatencyStatistics) -> String {
        guard let p50 = statistics.p50Milliseconds, let p95 = statistics.p95Milliseconds
        else { return "—" }
        return number(p50, unit: "") + "/ " + number(p95, unit: "ms")
    }
}
