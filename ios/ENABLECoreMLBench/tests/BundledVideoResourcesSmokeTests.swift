import CoreGraphics
import CoreMedia
import CryptoKit
import Foundation

/// Decodes staged clips from the built iOS app; run explicitly after resource staging.
@main
struct BundledVideoResourcesSmokeTests {
    /// Verifies packaged bytes, frame counts and every original presentation timestamp.
    static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let app = root.appendingPathComponent(
            "outputs/ios_builds/benchmark_resources/Build/Products/Release-iphoneos/"
                + "ENABLECoreMLBench.app")
        guard let bundle = Bundle(url: app),
            let manifestURL = bundle.url(
                forResource: "test_videos_manifest", withExtension: "json")
        else {
            preconditionFailure("Stage test videos and build the Release app first")
        }
        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        let samples = manifest["samples"] as! [[String: Any]]
        precondition(samples.count == BundledVideo.allCases.count)
        for sample in BundledVideo.allCases {
            let entry = samples.first { $0["id"] as? String == sample.sourceID }!
            let url = try sample.resourceURL(in: bundle)
            let bytes = try Data(contentsOf: url, options: .mappedIfSafe)
            precondition(hash(bytes) == entry["sha256"] as! String)
            let reference = VideoCoverageReference(input: [
                "source_id": sample.sourceID, "sha256": hash(bytes),
                "manifest": ["entry": entry],
            ])!
            let video = entry["video"] as! [String: Any]
            let base = video["time_base"] as! String
            let parts = base.split(separator: "/").map { Int32($0)! }
            precondition(parts[0] == 1)
            let reader = VideoFrameReader()
            try await reader.open(url, sourceID: sample.sourceID)
            var pts: [Int64] = []
            while let metadata = try await reader.nextFrame({ $0.metadata }) {
                precondition(metadata.sourceID == sample.sourceID)
                precondition(metadata.sourceFrameIndex == pts.count)
                precondition(reference.matches(metadata.presentationTime, index: pts.count))
                precondition(metadata.width == video["width"] as! Int)
                precondition(metadata.height == video["height"] as! Int)
                // Core Media can express the same instant with a different time base.
                // Convert only this comparison copy; reader metadata stays untouched.
                let comparable = CMTimeConvertScale(
                    metadata.presentationTime, timescale: parts[1],
                    method: .roundHalfAwayFromZero)
                precondition(CMTimeCompare(comparable, metadata.presentationTime) == 0)
                pts.append(comparable.value)
            }
            precondition(pts.count == video["frame_count"] as! Int)
            let serialized = "{\"time_base\":\"\(base)\",\"pts\":["
                + pts.map(String.init).joined(separator: ",") + "]}"
            precondition(
                hash(Data(serialized.utf8)) == video["pts_sha256"] as! String)
            let runner = OfflineVideoRunner(engine: DetectorEngine(loader: { _ in
                BundledResourceDetector()
            }))
            let summary = try await runner.run(
                url, sourceID: sample.sourceID, model: .yoloxNano,
                selection: .indices([0]), consume: { result in
                    precondition(result.metadata.sourceID == sample.sourceID)
                    precondition(
                        result.status == .completed && result.detections.isEmpty)
                })
            precondition(summary.completedFrames == 1)
            let report = await runner.latestReport
            precondition(report?.coverage.isFullyProcessed == true)
            let coverage = VideoRunCoverage(run: .offline(report!), reference: reference)
            precondition(coverage.status == "complete" && coverage.expected == 1)
            print(
                "Bundled video \(sample.title): \(pts.count) frames, exact times "
                    + "and detection handoff passed.")
        }
    }

    /// Matches Python's SHA-256 resource and ordered-timestamp identities.
    private static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

/// Exercises real conversion and orchestration without requiring model execution.
private actor BundledResourceDetector: ObjectDetector {
    /// Treats the immutable upright frame as a successful empty detection.
    func detect(in image: CGImage) -> [Detection] { [] }
}
