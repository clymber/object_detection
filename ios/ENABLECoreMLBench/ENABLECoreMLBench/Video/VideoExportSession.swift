import Foundation
import Observation

/// Owns export progress separately from the detector's measured run status.
@MainActor
@Observable
final class VideoExportSession {
    typealias Writer = @Sendable (VideoRunExport, URL) async throws -> VideoExportFiles
    private let write: Writer
    private var task: Task<Void, Never>?
    private var generation = UUID()
    private(set) var isBusy = false
    private(set) var files: VideoExportFiles?
    private(set) var message: String?

    /// Accepts an isolated writer for failure, cancellation and stale-result checks.
    init(write: Writer? = nil) {
        let exporter = VideoRunExporter()
        self.write = write ?? { try await exporter.export($0, sourceURL: $1) }
    }

    /// Freezes a finished report while disk work runs outside the UI actor.
    func start(_ run: VideoRunExport, sourceURL: URL) {
        guard !isBusy else { return }
        reset()
        let request = generation
        let write = self.write
        isBusy = true
        message = "Preparing export files…"
        task = Task { [weak self] in
            do {
                let files = try await write(run, sourceURL)
                guard !Task.isCancelled, let self, self.generation == request else { return }
                self.files = files
                self.message = "Three files ready to save or share."
                self.isBusy = false
                self.task = nil
            } catch {
                guard let self, self.generation == request else { return }
                self.message = error is CancellationError
                    ? "Export cancelled." : "Export failed: \(error.localizedDescription)"
                self.isBusy = false
                self.task = nil
            }
        }
    }

    /// Invalidates old receipts without deleting files an active share sheet may use.
    func reset() {
        generation = UUID()
        task?.cancel()
        task = nil
        isBusy = false
        files = nil
        message = nil
    }
}
