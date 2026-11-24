import Foundation
import Observation

/// Imports matched labels and computes accuracy on a worker after inference ends.
@MainActor
@Observable
final class VideoAccuracySession {
    private(set) var imported: ImportedVideoAnnotations?
    private(set) var result: VideoAccuracyExport?
    private(set) var isBusy = false
    private(set) var message: String?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var generation = UUID()

    /// Reads security-scoped JSON and verifies video bytes without blocking the UI.
    func load(_ url: URL, videoURL: URL) {
        reset()
        let request = generation
        isBusy = true
        message = "Reading annotations and checking the selected video…"
        task = Task { [weak self] in
            let worker = Task.detached {
                try ImportedVideoAnnotations.load(url, videoURL: videoURL)
            }
            do {
                let imported = try await withTaskCancellationHandler {
                    try await worker.value
                } onCancel: { worker.cancel() }
                guard !Task.isCancelled, let self, self.generation == request else { return }
                self.imported = imported
                self.message = "\(imported.annotations.video.frame_count) annotated frames ready."
                self.isBusy = false
                self.task = nil
            } catch {
                guard let self, self.generation == request else { return }
                self.message = error.localizedDescription
                self.isBusy = false
                self.task = nil
            }
        }
    }

    /// Rechecks video identity and rejects stale scores after model/input changes.
    func score(_ run: VideoRunExport, videoURL: URL, warmupFrames: Int) {
        clearResult()
        guard let imported else { return }
        let request = generation
        isBusy = true
        message = "Calculating F1…"
        task = Task { [weak self] in
            let worker = Task.detached {
                guard try VideoAnnotations.fileHash(videoURL)
                    == imported.annotations.video.sha256 else {
                    throw VideoBenchmarkError("Video bytes changed since annotation import.")
                }
                return try VideoAccuracyEvaluator.evaluate(run, annotations: imported.annotations)
            }
            do {
                let report = try await withTaskCancellationHandler {
                    try await worker.value
                } onCancel: { worker.cancel() }
                guard !Task.isCancelled, let self, self.generation == request else { return }
                self.result = VideoAccuracyExport(
                    runID: run.id, annotations: imported, report: report,
                    unavailableReason: nil, warmupFrames: warmupFrames)
                self.message = nil
                self.isBusy = false
                self.task = nil
            } catch {
                guard let self, self.generation == request else { return }
                self.result = VideoAccuracyExport(
                    runID: run.id, annotations: imported, report: nil,
                    unavailableReason: error.localizedDescription, warmupFrames: warmupFrames)
                self.message = "F1 unavailable: \(error.localizedDescription)"
                self.isBusy = false
                self.task = nil
            }
        }
    }

    /// Invalidates pending disk/scoring work while keeping labels for the next model.
    func clearResult() {
        generation = UUID()
        task?.cancel()
        task = nil
        isBusy = false
        result = nil
        message = imported == nil ? nil : "Annotations ready for the next run."
    }

    /// Preserves an unavailable reason if scoring was cancelled before export.
    func exportResult(for run: VideoRunExport, warmupFrames: Int) -> VideoAccuracyExport? {
        if let result, result.runID == run.id { return result }
        guard let imported else { return nil }
        return VideoAccuracyExport(runID: run.id, annotations: imported, report: nil,
            unavailableReason: "Accuracy scoring did not complete for this run.",
            warmupFrames: warmupFrames)
    }

    /// Drops the selected labels when the video changes or the user removes them.
    func reset() {
        clearResult()
        imported = nil
        message = nil
    }
}

/// Warms the shared model before reopening the full video for the measured pass.
actor VideoBenchmarkPreparation {
    static let warmupFrames = 30

    /// Uses a separate reader and discards predictions; scored indices still start at zero.
    static func warmup(_ url: URL, model: DetectorModel, engine: DetectorEngine) async throws {
        try await engine.selectModel(model)
        let reader = VideoFrameReader()
        do {
            try await reader.open(url, sourceID: "benchmark-warmup")
            guard let image = try await reader.nextFrame({
                try VideoFrameConverter.convert($0)
            }) else { throw VideoBenchmarkError("The video has no frame for model warm-up.") }
            for _ in 0..<warmupFrames {
                try Task.checkCancellation()
                _ = try await engine.detect(in: image.image)
            }
            await reader.cancel()
        } catch {
            await reader.cancel()
            throw error
        }
    }
}
