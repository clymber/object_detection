import CoreGraphics
import CoreMedia
import CryptoKit
import Foundation

/// Checks source proof separately from input balance and model success.
@main
struct VideoRunCoverageSmokeTests {
  static let sourceID = "coverage-fixture"
  static let times: [Int64] = [0, 33, 67]

  /// Exercises silent EOF, exact timestamps, skips, loops, failure and corruption.
  static func main() async throws {
    let input = inputReference()
    let reference = VideoCoverageReference(input: input)!
    let complete = try offline()
    precondition(
      VideoRunCoverage(run: .offline(complete), reference: reference).status == "complete")
    let sparse = try offline(indices: [0, 2])
    precondition(VideoRunCoverage(run: .offline(sparse), reference: reference).status == "complete")
    let failedInference = try offline(failedIndex: 1)
    let failedCoverage = VideoRunCoverage(run: .offline(failedInference), reference: reference)
    precondition(failedCoverage.status == "complete" && failedCoverage.failed == 1)
    let short = try offline(count: 2)
    let loss = VideoRunCoverage(run: .offline(short), reference: reference)
    precondition(loss.status == "incomplete" && loss.accountingBalanced)
    precondition(loss.missing == ["0:2"])
    let unknown = VideoRunCoverage(run: .offline(complete), reference: nil)
    precondition(unknown.status == "incomplete" && unknown.expected == nil)
    var duplicate = complete
    duplicate.append(OfflineFrameOutcome(result: try result(index: 0)))
    let repeated = VideoRunCoverage(run: .offline(duplicate), reference: reference)
    precondition(repeated.duplicates == ["0:0"] && !repeated.accountingBalanced)
    let drift = try offline(drift: true)
    precondition(
      VideoRunCoverage(run: .offline(drift), reference: reference)
        .timestampMismatches == ["0:1"])
    let explicitTrack = try offline(trackID: 2)
    precondition(
      VideoRunCoverage(run: .offline(explicitTrack), reference: reference).expected == nil)
    var decode = short
    decode.recordDecodeFailure(
      .init(sourceFrameIndex: 2, presentationTime: nil),
      error: .decodeFailed(frameIndex: 2, reason: "damaged"))
    let decodeLoss = VideoRunCoverage(run: .offline(decode), reference: reference)
    precondition(decodeLoss.decodeLossEvents == 1 && decodeLoss.status == "incomplete")
    let paced = try paced()
    let pacedCoverage = VideoRunCoverage(run: .paced(paced), reference: reference)
    precondition(pacedCoverage.status == "complete" && pacedCoverage.skipped == 4)
    precondition(pacedCoverage.inputFrames == 6 && pacedCoverage.expected == 6)
    let handoff = VideoRunCoverage(
      run: .paced(try self.paced(extraDecoded: true)), reference: reference)
    precondition(handoff.status == "incomplete" && handoff.decodedNotDelivered == 1)
    let cancelled = VideoRunCoverage(
      run: .paced(try self.paced(cancelled: true)), reference: reference)
    precondition(cancelled.status == "incomplete" && cancelled.unfinished == 2)
    precondition(cancelled.accountingBalanced)
    var badInput = input
    badInput["sha256"] = "changed"
    precondition(VideoCoverageReference(input: badInput) == nil)
    var entry = ((input["manifest"] as! [String: Any])["entry"] as! [String: Any])
    entry["frame_reference"] = ["time_base": "1/1000", "pts": [0, 34, 67]]
    badInput = input
    badInput["manifest"] = ["entry": entry]
    precondition(VideoCoverageReference(input: badInput) == nil)
    try await exports([
      "offline": .offline(complete), "sparse": .offline(sparse),
      "paced": .paced(paced), "short": .offline(short),
      "failed_inference": .offline(failedInference),
    ])
    print("Video run coverage: source proof, silent losses, loops, skips and partial work passed")
  }

  /// Supplies an independently fingerprinted irregular presentation sequence.
  static func inputReference() -> [String: Any] {
    let hash = SHA256.hash(data: Data("source".utf8))
      .map { String(format: "%02x", $0) }.joined()
    let sequenceHash = SHA256.hash(data: Data("{\"time_base\":\"1/1000\",\"pts\":[0,33,67]}".utf8))
      .map { String(format: "%02x", $0) }.joined()
    return [
      "source_id": sourceID, "sha256": hash,
      "manifest": [
        "entry": [
          "id": sourceID, "sha256": hash,
          "video": ["frame_count": 3, "pts_sha256": sequenceHash],
          "frame_reference": ["time_base": "1/1000", "pts": times],
        ]
      ],
    ]
  }

  /// Uses an equivalent time base to test exact rational comparison.
  static func metadata(index: Int, loop: Int = 0, drift: Bool = false) throws -> VideoFrameMetadata
  {
    try VideoFrameMetadata(
      sourceID: sourceID, sourceFrameIndex: index,
      presentationTime: CMTime(
        value: (times[index] + (drift ? 1 : 0)) * 30,
        timescale: 30000), width: 32, height: 32, loopIndex: loop)
  }

  /// Produces successful empty results, which still count as completed frames.
  static func result(
    index: Int, loop: Int = 0, drift: Bool = false,
    failed: Bool = false
  ) throws -> VideoFrameResult {
    let ready = 1000 + Double(loop >= 4 ? loop - 4 : 0) * 1000 + Double(times[index])
    return try VideoFrameResult(
      metadata: metadata(index: index, loop: loop, drift: drift),
      sourceWidth: 32, sourceHeight: 32, bufferToImage: .identity,
      status: failed ? .failed : .completed, detections: [],
      failure: failed ? VideoInferenceFailure(DetectorError.invalidTensor("fixture")) : nil,
      timings: VideoFrameTimings(
        readyMilliseconds: ready,
        modelTiming: ModelPredictionTiming(
          submittedMilliseconds: ready + 2,
          completedMilliseconds: ready + 18),
        doneMilliseconds: ready + 20))
  }

  /// Makes a finalized all-frame or sparse run, optionally with silent omissions.
  static func offline(
    count: Int = 3, indices: [Int]? = nil, drift: Bool = false,
    trackID: Int32? = nil, failedIndex: Int? = nil
  ) throws -> OfflineVideoRunReport {
    var r = OfflineVideoRunReport(
      sourceID: sourceID, model: .yoloxNano,
      loopIndex: 0, indices: indices, trackID: trackID)
    for index in 0..<count where indices == nil || indices!.contains(index) {
      r.append(
        OfflineFrameOutcome(
          result: try result(
            index: index,
            drift: drift && index == 1,
            failed: index == failedIndex)))
    }
    r.recordDecodedFrames(count)
    r.recordEndOfSource()
    r.finish(status: .completed)
    return r
  }

  /// Balances two loops with successful inference, both skip reasons and cancellation.
  static func paced(cancelled: Bool = false, extraDecoded: Bool = false) throws
    -> PacedVideoRunReport
  {
    let image = CGContext(
      data: nil, width: 32, height: 32, bitsPerComponent: 8,
      bytesPerRow: 128, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
    var outcomes: [PacedVideoFrameOutcome] = []
    for loop in 4..<6 {
      for index in 0..<3 {
        let result = try result(index: index, loop: loop)
        let ready = result.timings.readyMilliseconds
        let frame = UprightVideoFrame(
          metadata: result.metadata, image: image,
          bufferToImage: .identity, readyMilliseconds: ready)
        let status: PacedVideoFrameOutcome.Status =
          index == 0
          ? .completed
          : index == 1 ? .skippedSuperseded : cancelled ? .cancelled : .skippedStale
        outcomes.append(
          PacedVideoFrameOutcome(
            frame: PacedVideoFrame(
              frame: frame, dueMilliseconds: ready,
              deliveredMilliseconds: ready),
            status: status, decidedMilliseconds: ready + 20,
            result: index == 0 ? result : nil,
            publication: .init(
              status: index == 0 ? .published : .notApplicable,
              publishedMilliseconds: index == 0 ? ready + 25 : nil)))
      }
    }
    let queue = PacedVideoQueueSnapshot(
      state: .completed, acceptedFrames: 6,
      inputDeliveryMilliseconds: outcomes.map(\.deliveredMilliseconds), activeFrames: 0,
      pendingFrames: 0, activePublications: 0, pendingPublications: 0,
      publicationEnabled: true, previewEnabled: false, outcomes: outcomes)
    var source = PacedVideoSourceReport(
      sourceID: sourceID, firstLoopIndex: 4,
      loopCount: 2, started: 900)
    source.timeline = try VideoPlaybackTimeline(
      sourceID: sourceID,
      sourceTimeRange: CMTimeRange(start: .zero, duration: CMTime(value: 1, timescale: 1)),
      anchorMilliseconds: 1000, firstLoopIndex: 4)
    source.decodedFrames = extraDecoded ? 7 : 6
    source.finish(
      status: .completed, at: 3000,
      summary: PacedVideoSourceSummary(deliveredFrames: 6, completedLoops: 2))
    var report = PacedVideoRunReport(
      sourceID: sourceID, model: .yoloxNano,
      firstLoopIndex: 4, loopCount: 2)
    report.finish(status: cancelled ? .cancelled : .completed, source: source, queue: queue)
    return report
  }

  /// Saves native artifacts for independent Python coverage and metric reconstruction.
  static func exports(_ runs: [String: VideoRunExport]) async throws {
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    let folder = root.appendingPathComponent("outputs/ios_builds/video_pipeline/coverage_exports")
    try? FileManager.default.removeItem(at: folder)
    let bundleURL = folder.appendingPathComponent("Reference.bundle")
    try FileManager.default.createDirectory(at: bundleURL, withIntermediateDirectories: true)
    let entry = (inputReference()["manifest"] as! [String: Any])["entry"]!
    try JSONSerialization.data(withJSONObject: ["samples": [entry]]).write(
      to: bundleURL.appendingPathComponent("test_videos_manifest.json"))
    let source = folder.appendingPathComponent("source.mp4")
    try Data("source".utf8).write(to: source)
    let exporter = VideoRunExporter(bundle: Bundle(url: bundleURL)!, directory: folder)
    var paths: [String: String] = [:]
    for (name, run) in runs {
      let files = try await exporter.export(run, sourceURL: source)
      paths[name] = files.directory.path
    }
    try JSONSerialization.data(withJSONObject: paths).write(
      to: folder.appendingPathComponent("fixture_index.json"))
  }
}
