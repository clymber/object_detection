import SwiftUI

/// Displays unmeasured timings and zero processed frames for the UI demo.
struct BenchmarkMetricsView: View {
    var body: some View {
        Section {
            LabeledContent("Model-only FPS", value: "—")
            LabeledContent("End-to-end FPS", value: "—")
            LabeledContent("Inference latency", value: "— ms")
            LabeledContent("Processed frames", value: "0")
        } header: {
            Text("Metrics")
        } footer: {
            Text("Demo only: no frames are processed and no timings are measured.")
        }
        .monospacedDigit()
    }
}

#Preview {
    Form {
        BenchmarkMetricsView()
    }
}
