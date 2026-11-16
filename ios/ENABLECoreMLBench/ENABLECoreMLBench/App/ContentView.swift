import SwiftUI

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var session = DemoSession()
    @State private var selectedModel = "RF-DETR Small"
    @State private var selectedInput = "Static image"
    @State private var simulateError = false
    @State private var showingExportNotice = false

    private let modelNames = [
        "RF-DETR Small",
        "RF-DETR Nano",
        "YOLOX Nano",
    ]

    private let inputNames = [
        "Static image",
        "Prerecorded video",
        "Live camera",
    ]

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Detector", selection: $selectedModel) {
                        ForEach(modelNames, id: \.self) { modelName in
                            Text(modelName).tag(modelName)
                        }
                    }
                    .pickerStyle(.menu)
                } header: {
                    Text("Model")
                } footer: {
                    Text("Demo mode: selecting a model does not load it yet.")
                }
                .disabled(session.isRunning)

                Section {
                    Picker("Source", selection: $selectedInput) {
                        ForEach(inputNames, id: \.self) { inputName in
                            Text(inputName).tag(inputName)
                        }
                    }
                    .pickerStyle(.menu)
                } header: {
                    Text("Input")
                } footer: {
                    Text("Demo mode: media loading and camera capture come later.")
                }
                .disabled(session.isRunning)

                Section {
                    FramePreview(inputName: selectedInput)
                } header: {
                    Text("Preview")
                } footer: {
                    Text("Placeholder for future frames and detection boxes.")
                }

                BenchmarkMetricsView()

                Section {
                    LabeledContent("Status") {
                        HStack {
                            if session.isRunning {
                                ProgressView()
                                    .controlSize(.small)
                            }
                            Text(session.status.rawValue)
                                .foregroundStyle(
                                    session.status == .error ? Color.red : Color.primary
                                )
                        }
                    }

                    Text(session.message)
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    HStack {
                        Button("Start", systemImage: "play.fill") {
                            session.start(
                                model: selectedModel,
                                input: selectedInput,
                                simulateError: simulateError
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(session.isRunning)

                        Spacer()

                        Button("Stop", systemImage: "stop.fill") {
                            session.stop()
                        }
                        .buttonStyle(.bordered)
                        .disabled(!session.isRunning)
                    }

                    #if DEBUG
                    Toggle("Simulate an error", isOn: $simulateError)
                        .disabled(session.isRunning)
                    #endif
                } header: {
                    Text("Demo session")
                } footer: {
                    Text("Finishes after 5 seconds. No model or media is loaded.")
                }

                Section {
                    Button("Export Results", systemImage: "square.and.arrow.up") {
                        showingExportNotice = true
                    }
                    .buttonStyle(.bordered)
                    .disabled(session.isRunning)
                } header: {
                    Text("Results")
                } footer: {
                    Text("Demo placeholder. Export will be added with real results.")
                }
            }
            .navigationTitle("ENABLE Core ML Bench")
            .toolbarTitleDisplayMode(.inline)
        }
        .alert("Export not available yet", isPresented: $showingExportNotice) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("Export is not implemented yet. This demo has no results to save.")
        }
        .onDisappear {
            session.stop()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active {
                session.stop()
            }
        }
    }
}

#Preview {
    ContentView()
}
