import SwiftUI

/// Runs the selected detector on the bundled test-split examples.
struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var session = StillImageSession()
    @State private var batch = StillImageBatchSession()
    @State private var selectedModel = DetectorModel.rfdetrSmall
    @State private var selectedSample = BundledSample.testPositive6988

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    Picker("Detector", selection: $selectedModel) {
                        ForEach(DetectorModel.allCases) { model in
                            Text(model.displayName).tag(model)
                        }
                    }
                    .pickerStyle(.menu)
                    .disabled(session.isBusy || batch.isBusy)
                    .accessibilityIdentifier("modelPicker")
                }
                Section {
                    Picker("Test image", selection: $selectedSample) {
                        ForEach(BundledSample.allCases) { sample in
                            Text(sample.title).tag(sample)
                        }
                    }
                    .pickerStyle(.menu)
                    .disabled(session.isBusy || batch.isBusy)
                    .accessibilityIdentifier("samplePicker")
                    if session.image == nil && !session.isBusy {
                        Button("Load image") { session.load(selectedSample) }
                            .disabled(batch.isBusy)
                    }
                } header: {
                    Text("Input")
                } footer: {
                    Text(
                        "\(BundledSample.allCases.count) test-split images, including the notebook comparison grid."
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
                        Button("Start", systemImage: "play.fill") {
                            session.start(model: selectedModel)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!session.canStart || batch.isBusy)
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
                Section {
                    Button(
                        "Run all test images",
                        systemImage: "play.rectangle.on.rectangle"
                    ) {
                        batch.start(model: selectedModel)
                    }
                    .disabled(session.isBusy || batch.isBusy)
                    .accessibilityIdentifier("runAllTestImages")
                    if batch.isBusy {
                        ProgressView(
                            value: Double(batch.completed), total: Double(batch.total)
                        ) {
                            Text("\(batch.completed) of \(batch.total) images")
                        }
                        Button("Stop batch", role: .cancel) { batch.stop() }
                    }
                    Text(batch.message).font(.footnote).foregroundStyle(.secondary)
                    if let url = batch.exportURL {
                        ShareLink("Export results", item: url)
                            .accessibilityIdentifier("exportTestResults")
                    }
                } header: {
                    Text("Test-set comparison")
                } footer: {
                    Text("Export exact boxes and scores to compare on your Mac.")
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
        .onChange(of: selectedModel) { _, _ in
            session.load(selectedSample)
            batch.reset()
        }
        .onDisappear {
            session.stop()
            batch.stop()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active {
                session.stop()
                batch.stop()
            } else if session.image == nil && !session.isBusy {
                session.load(selectedSample)
            }
        }
    }
}

#Preview {
    ContentView()
}
