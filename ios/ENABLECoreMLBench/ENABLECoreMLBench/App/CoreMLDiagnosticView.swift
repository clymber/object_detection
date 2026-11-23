import SwiftUI

/// Provides a foreground surface only for explicitly requested diagnostic launches.
struct CoreMLDiagnosticView: View {
    let arguments: [String]
    @Environment(\.scenePhase) private var scenePhase
    @State private var started = false
    @State private var task: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
            Text("Core ML diagnostics")
            Text("Keep the app in the foreground. Results save automatically.")
                .font(.footnote)
        }
        .padding()
        .task {
            guard !started else { return }
            started = true
            task = Task {
                do {
                    let configuration = try CoreMLDiagnosticConfiguration(
                        arguments: arguments)
                    _ = try await CoreMLDiagnosticRunner().run(configuration)
                    // Only the command-line diagnostic launch terminates itself.
                    exit(0)
                } catch {
                    FileHandle.standardError.write(
                        Data("COREML_DIAGNOSTIC failed: \(error)\n".utf8))
                    exit(1)
                }
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { task?.cancel() }
        }
    }
}
