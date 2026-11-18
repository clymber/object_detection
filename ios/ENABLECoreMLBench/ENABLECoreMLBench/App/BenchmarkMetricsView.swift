import SwiftUI

/// Reports one image's detection latency without inventing benchmark FPS.
struct BenchmarkMetricsView: View {
    let latencyMilliseconds: Double?
    let processedImages: Int
    let detectionCount: Int

    var body: some View {
        Section {
            LabeledContent("Detection latency", value: latencyText)
            LabeledContent("Processed images", value: String(processedImages))
            LabeledContent("Detections", value: String(detectionCount))
            LabeledContent("Model-only FPS", value: "—")
            LabeledContent("End-to-end FPS", value: "—")
        } header: {
            Text("Metrics")
        } footer: {
            Text(
                "Latency includes preprocessing, inference and decoding. Image/model loading and rendering are excluded. FPS needs a timed benchmark run."
            )
        }
        .monospacedDigit()
    }

    private var latencyText: String {
        guard let latencyMilliseconds else { return "—" }
        return latencyMilliseconds.formatted(.number.precision(.fractionLength(1)))
            + " ms"
    }
}
