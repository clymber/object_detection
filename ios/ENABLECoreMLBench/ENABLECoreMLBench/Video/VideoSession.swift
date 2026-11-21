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
    private(set) var mode = VideoProcessingMode.offline
    private var sourceURL: URL?
    private var sourceID: String?

    var isBusy: Bool { offline.isBusy || paced.isBusy }
    var canStart: Bool { !isBusy && (mode == .offline ? offline.canStart : paced.canStart) }
    var status: OfflineVideoSession.Status {
        mode == .offline ? offline.status : paced.status
    }
    var message: String { mode == .offline ? offline.message : paced.message }
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
        offline: OfflineVideoSession? = nil, paced: PacedVideoSession? = nil
    ) {
        let engine = DetectorEngine()
        self.offline = offline ?? OfflineVideoSession(
            worker: OfflineVideoRunner(engine: engine))
        self.paced = paced ?? PacedVideoSession(
            worker: PacedVideoRunner(engine: engine))
    }

    /// Clears old mode results and rejects interactive switches while work is active.
    func setMode(_ mode: VideoProcessingMode) {
        guard !isBusy, self.mode != mode else { return }
        self.mode = mode
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
        sourceURL = url
        self.sourceID = sourceID
        offline.select(url, sourceID: sourceID)
        paced.select(url, sourceID: sourceID)
    }

    /// Starts only the selected mode with a frozen model and one source pass.
    func start(model: DetectorModel) {
        guard !isBusy else { return }
        if mode == .offline { offline.start(model: model) }
        else { paced.start(model: model) }
    }

    /// Cancels on Stop, disappearance or backgrounding before old results can publish.
    func stop() {
        offline.stop()
        paced.stop()
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
