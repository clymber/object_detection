import SwiftUI

@main struct ENABLECoreMLBenchApp: App {
    var body: some Scene {
        WindowGroup {
            if let index = CommandLine.arguments.firstIndex(of: "--coreml-diagnostics") {
                CoreMLDiagnosticView(
                    arguments: Array(CommandLine.arguments.dropFirst(index + 1)))
            } else {
                ContentView()
            }
        }
    }
}
