import CoreGraphics
import Foundation
import Observation

/// Publishes one still-image session while worker actors own all expensive work.
@MainActor
@Observable
final class StillImageSession {
    enum Status: String {
        case idle = "Ready"
        case loading = "Loading image"
        case running = "Running"
        case completed = "Completed"
        case error = "Error"
    }

    private(set) var status: Status = .idle
    private(set) var message = "Choose a test image, then tap Start."
    private(set) var image: CGImage?
    private(set) var detections: [Detection] = []
    private(set) var latencyMilliseconds: Double?
    private(set) var processedImages = 0

    @ObservationIgnored private let worker: any StillImageProcessing
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var generation: UInt64 = 0

    var isBusy: Bool { status == .loading || status == .running }
    var canStart: Bool { image != nil && !isBusy }

    /// Supplies the real worker by default or a controlled worker for lifecycle checks.
    init(worker: any StillImageProcessing = StillImageRunner()) {
        self.worker = worker
    }

    /// Replaces the selected image and clears results from the previous input.
    func load(_ sample: BundledSample) {
        invalidateTask()
        let request = generation
        image = nil
        clearResults()
        status = .loading
        message = "Loading \(sample.title.lowercased()) image…"
        let worker = worker
        task = Task { [weak self] in
            do {
                let loaded = try await worker.load(sample)
                try Task.checkCancellation()
                guard let self, self.generation == request else { return }
                self.image = loaded
                self.status = .idle
                self.message = "Image ready. Tap Start to detect basketballs."
                self.task = nil
            } catch {
                guard let self, self.generation == request else { return }
                self.fail(error)
            }
        }
    }

    /// Starts one real inference and ignores duplicate Start calls while busy.
    func start(model: DetectorModel = .rfdetrSmall) {
        guard !isBusy else { return }
        guard let image else {
            fail(DetectorError.invalidImage("Choose a test image before starting."))
            return
        }
        invalidateTask()
        let request = generation
        clearResults()
        status = .running
        message = "Loading \(model.displayName) and running detection…"
        let worker = worker
        task = Task { [weak self] in
            do {
                let result = try await worker.run(image, model: model)
                try Task.checkCancellation()
                guard let self, self.generation == request else { return }
                self.detections = result.detections
                self.latencyMilliseconds = result.latencyMilliseconds
                self.processedImages = 1
                self.status = .completed
                switch result.detections.count {
                case 0: self.message = "No basketball detected."
                case 1: self.message = "Detected 1 basketball."
                default:
                    self.message = "Detected \(result.detections.count) basketballs."
                }
                self.task = nil
            } catch {
                guard let self, self.generation == request else { return }
                self.fail(error)
            }
        }
    }

    /// Stops the active operation so late image or inference results cannot publish.
    func stop() {
        guard isBusy else { return }
        invalidateTask()
        status = .idle
        message = "Stopped. Choose an image or tap Start to try again."
    }

    /// Cancels an operation and invalidates any pending callbacks.
    private func invalidateTask() {
        generation &+= 1
        task?.cancel()
        task = nil
    }

    /// Resets measurements without discarding the currently decoded image.
    private func clearResults() {
        detections = []
        latencyMilliseconds = nil
        processedImages = 0
    }

    /// Reports actionable failures while preserving the image for a retry.
    private func fail(_ error: Error) {
        status = .error
        message = error.localizedDescription
        task = nil
    }
}
