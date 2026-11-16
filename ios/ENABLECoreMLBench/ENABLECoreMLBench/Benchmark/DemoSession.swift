import Foundation
import Observation

/// Controls the demo lifecycle without loading media or running inference.
@MainActor
@Observable
final class DemoSession {
    enum Status: String {
        case idle = "Idle"
        case running = "Running"
        case error = "Error"
        case completed = "Completed"
    }

    private(set) var status: Status = .idle
    private(set) var message = "Choose a model and input, then tap Start."

    @ObservationIgnored private var runTask: Task<Void, Never>?
    private let completionDelay: Duration

    var isRunning: Bool {
        status == .running
    }

    /// Sets the demo duration, allowing fast lifecycle checks without inference.
    init(completionDelay: Duration = .seconds(5)) {
        self.completionDelay = completionDelay
    }

    /// Starts one demo session and optionally simulates a failure at completion.
    func start(model: String, input: String, simulateError: Bool = false) {
        guard !isRunning else { return }

        status = .running
        message = "Demo: \(model) with \(input). No inference is running."
        let delay = completionDelay

        runTask = Task { [weak self] in
            do {
                try await Task.sleep(for: delay)
            } catch {
                return
            }

            // A cancelled run must never overwrite a newer session's state.
            guard !Task.isCancelled, let self else { return }
            self.runTask = nil

            if simulateError {
                self.status = .error
                self.message = "Simulated error. Turn off the error switch and retry."
            } else {
                self.status = .completed
                self.message = "Demo completed. No frames were processed."
            }
        }
    }

    /// Cancels an active demo and returns to Idle without reporting completion.
    func stop() {
        guard isRunning else { return }

        runTask?.cancel()
        runTask = nil
        status = .idle
        message = "Demo stopped. Tap Start to run again."
    }
}
