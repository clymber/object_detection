import Foundation
import Observation

/// Keeps explicit offline/paced mode selection separate from each run's state.
nonisolated enum VideoProcessingMode: String, CaseIterable, Identifiable, Sendable {
    case offline = "Offline"
    case paced = "Paced"
    var id: Self { self }
}

/// Routes shared controls while preserving each worker's lifecycle and timing scopes.
@MainActor
@Observable
final class VideoSession {
    let offline: OfflineVideoSession
    let paced: PacedVideoSession
    let export: VideoExportSession
    let accuracy = VideoAccuracySession()
    private(set) var mode = VideoProcessingMode.offline
    private(set) var isPreparingBenchmark = false
    private(set) var preparationError: String?
    private var sourceURL: URL?
    private var sourceID: String?
    private let engine: DetectorEngine
    private var preparationTask: Task<Void, Never>?
    private var preparationGeneration = UUID()
    private var activePreparation: UUID?
    private var isStoppingPreparation = false
    private var warmupFramesForRun = 0

    var isDetecting: Bool { offline.isBusy || paced.isBusy }
    var isBusy: Bool {
        isDetecting || isPreparingBenchmark || accuracy.isBusy || export.isBusy
    }
    var canStart: Bool { !isBusy && (mode == .offline ? offline.canStart : paced.canStart) }
    var status: OfflineVideoSession.Status {
        if isPreparingBenchmark { return isStoppingPreparation ? .stopping : .preparing }
        if preparationError != nil { return .error }
        return mode == .offline ? offline.status : paced.status
    }
    var message: String {
        if isStoppingPreparation { return "Stopping benchmark preparation…" }
        if isPreparingBenchmark { return "Loading the model and warming up 30 predictions…" }
        if let preparationError { return preparationError }
        return mode == .offline ? offline.message : paced.message
    }
    var progressText: String {
        mode == .offline ? offline.progressText : paced.progressText
    }
    var hasReport: Bool {
        mode == .offline ? offline.latestReport != nil : paced.latestReport != nil
    }
    var isPreviewEnabled: Bool {
        mode == .offline ? offline.isPreviewEnabled : paced.isPreviewEnabled
    }
    var previewFrame: VideoFramePreview? {
        mode == .offline ? offline.previewFrame : paced.previewFrame
    }
    var previewGeneration: UUID {
        mode == .offline ? offline.previewGeneration : paced.previewGeneration
    }

    /// Permits controlled child sessions without changing default app behaviour.
    init(
        offline: OfflineVideoSession? = nil, paced: PacedVideoSession? = nil,
        export: VideoExportSession? = nil, engine: DetectorEngine = DetectorEngine()
    ) {
        self.offline = offline ?? OfflineVideoSession(
            worker: OfflineVideoRunner(engine: engine))
        self.paced = paced ?? PacedVideoSession(
            worker: PacedVideoRunner(engine: engine))
        self.export = export ?? VideoExportSession()
        self.engine = engine
        self.offline.onFinished = { [weak self] in self?.score($0) }
        self.paced.onFinished = { [weak self] in self?.score($0) }
    }

    /// Clears old mode results and rejects interactive switches while work is active.
    func setMode(_ mode: VideoProcessingMode) {
        guard !isBusy, self.mode != mode else { return }
        self.mode = mode
        preparationError = nil
        warmupFramesForRun = 0
        accuracy.clearResult()
        export.reset()
        offline.select(sourceURL, sourceID: sourceID)
        paced.select(sourceURL, sourceID: sourceID)
    }

    /// Applies preview choice consistently when switching between processing modes.
    func setPreviewEnabled(_ enabled: Bool) {
        guard !isBusy else { return }
        offline.setPreviewEnabled(enabled)
        paced.setPreviewEnabled(enabled)
    }

    /// Invalidates both modes on input/model changes, preserving the mode choice.
    func select(_ url: URL?, sourceID: String? = nil) {
        cancelPreparation()
        if sourceURL != url { accuracy.reset() } else { accuracy.clearResult() }
        export.reset()
        sourceURL = url
        self.sourceID = sourceID
        offline.select(url, sourceID: sourceID)
        paced.select(url, sourceID: sourceID)
    }

    /// Starts only the selected mode with a frozen model and one source pass.
    func start(model: DetectorModel) {
        guard !isBusy else { return }
        export.reset()
        accuracy.clearResult()
        preparationError = nil
        warmupFramesForRun = 0
        offline.select(sourceURL, sourceID: sourceID)
        paced.select(sourceURL, sourceID: sourceID)
        guard accuracy.imported != nil, let sourceURL else {
            beginRun(model: model)
            return
        }
        isPreparingBenchmark = true
        let request = UUID()
        preparationGeneration = request
        activePreparation = request
        isStoppingPreparation = false
        let engine = engine
        preparationTask = Task { [weak self] in
            do {
                try await VideoBenchmarkPreparation.warmup(sourceURL, model: model, engine: engine)
                self?.finishPreparation(request, model: model, error: nil)
            } catch {
                self?.finishPreparation(request, model: model, error: error)
            }
        }
    }

    /// Waits for cancelled warm-up to unwind before another detector can begin.
    private func finishPreparation(_ request: UUID, model: DetectorModel, error: Error?) {
        guard activePreparation == request else { return }
        preparationTask = nil
        activePreparation = nil
        isPreparingBenchmark = false
        isStoppingPreparation = false
        guard preparationGeneration == request else { return }
        if let error {
            preparationError = error is CancellationError
                ? "Benchmark preparation stopped." : error.localizedDescription
        } else {
            warmupFramesForRun = VideoBenchmarkPreparation.warmupFrames
            beginRun(model: model)
        }
    }

    /// Starts timing only after optional annotated-benchmark preparation has ended.
    private func beginRun(model: DetectorModel) {
        if mode == .offline { offline.start(model: model) }
        else { paced.start(model: model) }
    }

    /// Calculates accuracy after the worker has finalized its timing report.
    private func score(_ run: VideoRunExport) {
        guard let sourceURL, run.mode == mode.rawValue.lowercased() else { return }
        accuracy.score(run, videoURL: sourceURL, warmupFrames: warmupFramesForRun)
    }

    /// Imports optional ground truth without rerunning or altering an existing video.
    func importAnnotations(_ url: URL) {
        guard !isBusy, let sourceURL else { return }
        preparationError = nil
        export.reset()
        offline.select(sourceURL, sourceID: sourceID)
        paced.select(sourceURL, sourceID: sourceID)
        accuracy.load(url, videoURL: sourceURL)
    }

    /// Removes labels and their accuracy result while retaining the selected video.
    func removeAnnotations() {
        guard !isBusy else { return }
        preparationError = nil
        accuracy.reset()
        export.reset()
    }

    /// Cancels preparation before permitting another source/model selection.
    private func cancelPreparation() {
        preparationGeneration = UUID()
        preparationTask?.cancel()
        isStoppingPreparation = preparationTask != nil
        isPreparingBenchmark = preparationTask != nil
        preparationError = nil
    }

    /// Stops inference during system UI transitions without erasing file verification.
    func suspendDetection() {
        if isPreparingBenchmark {
            cancelPreparation()
            preparationError = "Benchmark preparation stopped."
        }
        offline.stop()
        paced.stop()
    }

    /// Cancels explicit Stop while retaining prepared files and verification feedback.
    func stop() {
        suspendDetection()
        if accuracy.isBusy { accuracy.clearResult() }
        export.cancel()
    }

    /// Exports either final report, including failed/cancelled attempts, on demand.
    func exportResults() {
        guard !isBusy, let sourceURL else { return }
        switch mode {
        case .offline:
            if let report = offline.latestReport {
                export.start(.offline(report), sourceURL: sourceURL,
                    accuracy: accuracy.exportResult(
                        for: .offline(report), warmupFrames: warmupFramesForRun))
            }
        case .paced:
            if let report = paced.latestReport {
                export.start(.paced(report), sourceURL: sourceURL,
                    accuracy: accuracy.exportResult(
                        for: .paced(report), warmupFrames: warmupFramesForRun))
            }
        }
    }

    /// Routes only a current generation's renderer acknowledgement to the active mode.
    func recordPreviewDraw(
        _ token: VideoPreviewDrawToken, generation: UUID, at milliseconds: Double
    ) {
        if mode == .offline {
            offline.recordPreviewDraw(token, generation: generation, at: milliseconds)
        } else {
            paced.recordPreviewDraw(token, generation: generation, at: milliseconds)
        }
    }
}
