import SwiftUI

/// Runs the first-device RF-DETR Small check on the bundled validation images.
struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var session = StillImageSession()
    @State private var selectedSample = BundledSample.positive

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    LabeledContent(
                        "Detector", value: DetectorModel.rfdetrSmall.displayName)
                }
                Section {
                    Picker("Test image", selection: $selectedSample) {
                        ForEach(BundledSample.allCases) { sample in
                            Text(sample.title).tag(sample)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(session.isBusy)
                    .accessibilityIdentifier("samplePicker")
                    if session.image == nil && !session.isBusy {
                        Button("Load image") { session.load(selectedSample) }
                    }
                } header: {
                    Text("Input")
                } footer: {
                    Text(
                        "Bundled validation images: one basketball and one background image."
                    )
                }
                Section {
                    FramePreview(
                        image: session.image, detections: session.detections,
                        imageTitle: selectedSample.title
                    )
                } header: {
                    Text("Preview")
                } footer: {
                    Text("Boxes show detections at confidence ≥ 25%.")
                }
                Section {
                    LabeledContent("Status") {
                        HStack {
                            if session.isBusy { ProgressView().controlSize(.small) }
                            Text(session.status.rawValue)
                                .foregroundStyle(
                                    session.status == .error ? Color.red : .primary
                                )
                                .accessibilityIdentifier("detectionStatus")
                        }
                    }
                    Text(session.message)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("detectionMessage")
                    HStack {
                        Button("Start", systemImage: "play.fill") { session.start() }
                            .buttonStyle(.borderedProminent)
                            .disabled(!session.canStart)
                            .accessibilityIdentifier("startDetection")
                        Spacer()
                        Button("Stop", systemImage: "stop.fill") { session.stop() }
                            .buttonStyle(.bordered)
                            .disabled(!session.isBusy)
                            .accessibilityIdentifier("stopDetection")
                    }
                } header: {
                    Text("Detection")
                }
                BenchmarkMetricsView(
                    latencyMilliseconds: session.latencyMilliseconds,
                    processedImages: session.processedImages,
                    detectionCount: session.detections.count
                )
            }
            .navigationTitle("ENABLE Core ML Bench")
            .toolbarTitleDisplayMode(.inline)
        }
        .task { session.load(selectedSample) }
        .onChange(of: selectedSample) { _, sample in session.load(sample) }
        .onDisappear { session.stop() }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active {
                session.stop()
            } else if session.image == nil && !session.isBusy {
                session.load(selectedSample)
            }
        }
    }
}

#Preview {
    ContentView()
}
