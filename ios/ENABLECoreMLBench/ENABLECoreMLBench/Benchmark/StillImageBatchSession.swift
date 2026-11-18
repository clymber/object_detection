import Foundation
import Observation

/// Publishes batch progress and the current share URL without blocking the UI.
@MainActor
@Observable
final class StillImageBatchSession {
    private(set) var isBusy = false
    private(set) var completed = 0
    private(set) var total = BundledSample.allCases.count
    private(set) var message =
        "Run all test images, then export results for comparison."
    private(set) var exportURL: URL?
    @ObservationIgnored private let runner: StillImageBatchRunner
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var generation: UInt64 = 0

    /// Accepts an alternate batch runner for host and cancellation checks.
    init(runner: StillImageBatchRunner = StillImageBatchRunner()) {
        self.runner = runner
    }

    /// Starts a fresh run and prevents sharing a previous run while it is replaced.
    func start(model: DetectorModel = .rfdetrSmall) {
        guard !isBusy else { return }
        generation &+= 1
        let request = generation
        completed = 0
        exportURL = nil
        isBusy = true
        message = "Running \(model.displayName) on all test images…"
        let runner = runner
        task = Task { [weak self] in
            do {
                let report = try await runner.run(model: model) {
                    [weak self] completed, total in
                    await self?.update(
                        completed: completed, total: total, request: request)
                }
                try Task.checkCancellation()
                let url = try await runner.export(report)
                try Task.checkCancellation()
                guard let self, self.generation == request else { return }
                self.exportURL = url
                let errors = report.images.filter { $0.status != "completed" }.count
                self.message =
                    errors == 0
                    ? "All \(report.images.count) images completed. Export results to check parity."
                    : "Incomplete: \(errors) image(s) failed. Export results for details."
                self.isBusy = false
                self.task = nil
            } catch {
                guard let self, self.generation == request else { return }
                self.isBusy = false
                self.message = "Batch failed: \(error.localizedDescription)"
                self.task = nil
            }
        }
    }

    /// Clears the previous model's results when the model selector changes.
    func reset() {
        generation &+= 1
        task?.cancel()
        task = nil
        isBusy = false
        completed = 0
        exportURL = nil
        message = "Run all test images, then export results for comparison."
    }

    /// Cancels publication so an older batch cannot replace a newer export.
    func stop() {
        guard isBusy else { return }
        generation &+= 1
        task?.cancel()
        task = nil
        isBusy = false
        exportURL = nil
        message = "Stopped. Run all images again to export a complete attempt."
    }

    /// Rejects progress callbacks from stopped or superseded operations.
    private func update(completed: Int, total: Int, request: UInt64) {
        guard generation == request, isBusy else { return }
        self.completed = completed
        self.total = total
    }
}
