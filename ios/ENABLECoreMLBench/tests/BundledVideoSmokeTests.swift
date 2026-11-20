import Foundation

/// Exercises resource discovery independently of downloaded videos or model packages.
@main
struct BundledVideoSmokeTests {
    /// Checks all supported Xcode layouts and explicit missing-resource errors.
    static func main() throws {
        precondition(BundledVideo.allCases.count == 3)
        precondition(Set(BundledVideo.allCases.map(\.sourceID)).count == 3)
        let directory = URL(
            fileURLWithPath: "outputs/ios_builds/video_pipeline/catalog")
        for (index, subdirectory) in ["", "TestVideos", "Resources/TestVideos"]
            .enumerated() {
            let bundleURL = directory.appendingPathComponent("layout_\(index).bundle")
            let resources = bundleURL.appendingPathComponent(subdirectory)
            try FileManager.default.createDirectory(
                at: resources, withIntermediateDirectories: true)
            let info = try PropertyListSerialization.data(
                fromPropertyList: ["CFBundleIdentifier": "test.videos.layout\(index)"],
                format: .xml, options: 0)
            try info.write(to: bundleURL.appendingPathComponent("Info.plist"))
            for sample in BundledVideo.allCases {
                let url = resources.appendingPathComponent(sample.rawValue + ".mp4")
                try Data("resource lookup fixture".utf8).write(to: url)
            }
            let bundle = Bundle(url: bundleURL)!
            for sample in BundledVideo.allCases {
                let url = try sample.resourceURL(in: bundle)
                precondition(url.lastPathComponent == sample.rawValue + ".mp4")
                precondition(url.deletingLastPathComponent().standardizedFileURL.path
                             == resources.standardizedFileURL.path)
            }
        }
        do {
            _ = try BundledVideo.nba.resourceURL(in: Bundle(for: MissingVideo.self))
            preconditionFailure("Missing test video was silently accepted")
        } catch DetectorError.invalidImage(let message) {
            precondition(message.contains("Stage test videos"))
        }
        print("Bundled videos: three resource layouts and missing-file errors passed.")
    }
}

/// Supplies a resource-free bundle for the missing-clip check.
private final class MissingVideo: NSObject {}
