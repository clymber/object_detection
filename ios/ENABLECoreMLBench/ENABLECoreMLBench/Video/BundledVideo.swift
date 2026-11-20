import Foundation

/// The three repeatable developer clips staged separately from benchmark manifests.
nonisolated enum BundledVideo: String, CaseIterable, Identifiable, Sendable {
    case nba = "basketball-nba_clip"
    case uhd = "basketball-uhd_4096_2160_30fps"
    case dribbling = "basketball_dribbling_h264"

    var id: String { rawValue }
    var sourceID: String { "bundled-video:\(rawValue)" }
    var title: String {
        switch self {
        case .nba: "NBA game · 720p"
        case .uhd: "UHD basketball · 4K"
        case .dribbling: "Dribbling · 540p"
        }
    }

    /// Supports Xcode's flattened resources and preserved test-video folders.
    func resourceURL(in bundle: Bundle = .main) throws -> URL {
        for directory in ["TestVideos", "Resources/TestVideos", nil] as [String?] {
            if let url = bundle.url(
                forResource: rawValue, withExtension: "mp4", subdirectory: directory
            ) { return url }
        }
        throw DetectorError.invalidImage(
            "The test video is missing. Stage test videos and rebuild the app.")
    }
}
