import CoreMedia
import Foundation

/// Freezes capture and presentation choices for one diagnostic run.
nonisolated struct CameraConfiguration: Sendable, Equatable {
    enum Lens: String, CaseIterable, Identifiable, Sendable {
        case rear = "Rear wide"
        case front = "Front wide"
        var id: Self { self }
    }
    enum Resolution: String, CaseIterable, Identifiable, Sendable {
        case hd = "1280 × 720"
        case fullHD = "1920 × 1080"
        var id: Self { self }
        var width: Int32 { self == .hd ? 1280 : 1920 }
        var height: Int32 { self == .hd ? 720 : 1080 }
    }
    enum Orientation: Int, CaseIterable, Identifiable, Sendable {
        case portrait = 90
        case landscape = 0
        case invertedLandscape = 180
        case invertedPortrait = 270
        var id: Self { self }
        var title: String {
            switch self {
            case .portrait: "Portrait"
            case .landscape: "Landscape (0°)"
            case .invertedLandscape: "Landscape (180°)"
            case .invertedPortrait: "Upside down"
            }
        }
    }
    var lens = Lens.rear
    var resolution = Resolution.hd
    var framesPerSecond = 30
    var orientation = Orientation.portrait
    var mirrored = false
    var livePreview = true
    var detectionPreview = false
    var sceneNotes = ""

    /// Rejects unsupported diagnostic requests before opening the camera.
    func validate() throws {
        guard [15, 30].contains(framesPerSecond) else {
            throw CameraError.configuration("Choose 15 or 30 FPS.")
        }
    }
}

/// Keeps permission outcomes testable without opening physical hardware.
nonisolated enum CameraAuthorization: Sendable {
    case unknown, authorized, denied, restricted
}

/// Reports hardware, permission and capture failures without mutable backend objects.
nonisolated enum CameraError: LocalizedError {
    case unavailable, denied, alreadyRunning, invalidTimestamp
    case interrupted(String)
    case configuration(String)
    var errorDescription: String? {
        switch self {
        case .unavailable: "This camera is unavailable on this device."
        case .denied:
            "Camera access is not allowed. Enable it in Settings, then try again."
        case .alreadyRunning: "A camera run is already active."
        case .invalidTimestamp:
            "A camera timestamp could not be mapped to the host clock."
        case .interrupted(let reason):
            "Camera interrupted: \(reason). Tap Start to retry."
        case .configuration(let reason): reason
        }
    }
}

/// Anchors Core Media host time to this run's process-local inference clock.
nonisolated struct CameraClockMapping: Sendable {
    let hostAnchor: CMTime
    let inferenceAnchorMilliseconds: Double
    let uncertaintyMilliseconds: Double

    /// Records calibration bounds; callbacks never substitute for sample capture time.
    init(host: CMTime, before: Double, after: Double) throws {
        guard host.isNumeric, before.isFinite, before >= 0, after.isFinite,
            after >= before
        else {
            throw CameraError.invalidTimestamp
        }
        hostAnchor = host
        inferenceAnchorMilliseconds = before + (after - before) / 2
        uncertaintyMilliseconds = (after - before) / 2
    }

    /// Converts an already synchronized host timestamp without rewriting original PTS.
    func map(_ hostTime: CMTime) throws -> Double {
        guard hostTime.isNumeric, hostTime.epoch == hostAnchor.epoch else {
            throw CameraError.invalidTimestamp
        }
        let value =
            inferenceAnchorMilliseconds
            + CMTimeGetSeconds(CMTimeSubtract(hostTime, hostAnchor)) * 1000
        guard value.isFinite, value >= 0 else { throw CameraError.invalidTimestamp }
        return value
    }
}

/// Describes observed capture callbacks; indices do not claim a sensor frame count.
nonisolated struct CameraCaptureEvent: Sendable {
    enum Kind: String, Sendable { case delivered, dropped }
    let index: Int
    let kind: Kind
    let presentationTime: CMTime
    let hostPresentationTime: CMTime?
    let capturedMilliseconds: Double?
    let receivedMilliseconds: Double
    var admitted = false
    var deliveredMilliseconds: Double?
    var cancelled = false
    var reason: String?
    var knownLostFrames: Int?
    var failure: VideoInferenceFailure?
    var observation: [String: Double] = [:]
}

/// A pixel-free value snapshot of capture configuration, drops and interruptions.
nonisolated struct CameraCaptureReport: Sendable {
    let sourceID: String
    let configuration: CameraConfiguration
    var actual: [String: String] = [:]
    var mapping: CameraClockMapping?
    var startedMilliseconds: Double?
    var endedMilliseconds: Double?
    var events: [CameraCaptureEvent] = []
    var issue: VideoInferenceFailure?
    var isRunning = false
    var delivered: Int { events.filter { $0.kind == .delivered }.count }
    var admitted: Int { events.filter(\.admitted).count }
    var dropEvents: Int { events.filter { $0.kind == .dropped }.count }
    var hasUnknownLoss: Bool {
        events.contains {
            ($0.kind == .dropped && $0.knownLostFrames == nil)
                || $0.capturedMilliseconds == nil
                || ($0.failure != nil && !$0.cancelled)
        }
    }
}

/// Separates synchronous capture admission from asynchronous inference processing.
nonisolated protocol CameraCapturing: Sendable {
    var authorization: CameraAuthorization { get }
    var preview: CameraPreviewHandle? { get }
    var snapshot: CameraCaptureReport? { get }
    /// Requests video permission without starting capture.
    func authorize() async -> Bool
    /// Delivers bounded synchronous callbacks until capture stops or fails.
    func run(
        configuration: CameraConfiguration, sourceID: String,
        deliver: @escaping @Sendable (PacedVideoFrame) throws -> Void) async throws
    func stop()
}
