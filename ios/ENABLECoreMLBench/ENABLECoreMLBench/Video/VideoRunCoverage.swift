import CoreMedia
import CryptoKit
import Foundation

/// Binds an ordered source sequence to the exact video bytes in a bundled manifest.
nonisolated struct VideoCoverageReference: Sendable {
  let sourceID: String
  let pts: [Int64]
  let numerator: Int64
  let denominator: Int32

  /// Rejects absent, changed or malformed references instead of trusting frame counts.
  init?(input: [String: Any]) {
    guard let sourceID = input["source_id"] as? String,
      let actual = input["sha256"] as? String,
      let manifest = input["manifest"] as? [String: Any],
      let entry = manifest["entry"] as? [String: Any],
      entry["id"] as? String == sourceID, entry["sha256"] as? String == actual,
      let reference = entry["frame_reference"] as? [String: Any],
      let base = reference["time_base"] as? String,
      let pts = reference["pts"] as? [Int64], !pts.isEmpty,
      let video = entry["video"] as? [String: Any],
      video["frame_count"] as? Int == pts.count,
      let expectedDigest = video["pts_sha256"] as? String
    else { return nil }
    let parts = base.split(separator: "/")
    guard parts.count == 2, let numerator = Int64(parts[0]), numerator > 0,
      let denominator = Int32(parts[1]), denominator > 0,
      zip(pts, pts.dropFirst()).allSatisfy({ $0 < $1 }),
      pts.allSatisfy({ !$0.multipliedReportingOverflow(by: numerator).overflow })
    else { return nil }
    // Match stage_test_videos.py's existing ordered, compact digest convention.
    let canonical =
      "{\"time_base\":\"\(base)\",\"pts\":["
      + pts.map(String.init).joined(separator: ",") + "]}"
    let digest = SHA256.hash(data: Data(canonical.utf8))
      .map { String(format: "%02x", $0) }.joined()
    guard digest == expectedDigest else { return nil }
    self.sourceID = sourceID
    self.pts = pts
    self.numerator = numerator
    self.denominator = denominator
  }

  /// Compares exact rational media instants across equivalent decoder time bases.
  func matches(_ time: CMTime, index: Int) -> Bool {
    guard pts.indices.contains(index), time.isNumeric, time.epoch == 0 else { return false }
    return CMTimeCompare(
      time,
      CMTime(
        value: pts[index] * numerator,
        timescale: denominator)) == 0
  }
}

/// Reconciles admitted inputs and source identities without conflating skips with loss.
nonisolated struct VideoRunCoverage: Sendable {
  let inputFrames: Int
  let completed: Int
  let failed: Int
  let skipped: Int
  let unfinished: Int
  let observed: Int
  let expected: Int?
  let missing: [String]
  let duplicates: [String]
  let unexpected: [String]
  let timestampMismatches: [String]
  let invalid: [String]
  let decodedNotDelivered: Int
  let decodeLossEvents: Int
  let accountingBalanced: Bool
  let reasons: [String]
  var status: String { reasons.isEmpty ? "complete" : "incomplete" }

  /// Uses frozen references when available and retains a separate input balance.
  init(run: VideoRunExport, reference: VideoCoverageReference?) {
    var rows: [(loop: Int, index: Int, metadata: VideoFrameMetadata?)] = []
    var completed = 0
    var failed = 0
    var skipped = 0
    var unfinished = 0
    var input = 0
    var decodedNotDelivered = 0
    var decodeLossEvents = 0
    var countersValid = true
    var nativeOutcomesValid = true
    var firstLoop = 0
    var loops: Int? = 1
    var selected: [Int]?
    var explicitTrack = false
    switch run {
    case .offline(let r):
      firstLoop = r.loopIndex
      selected = r.selectedFrameIndices
      explicitTrack = r.requestedTrackID != nil
      let decodedSelected =
        selected.map { $0.filter { $0 < r.decodedFrames }.count }
        ?? r.decodedFrames
      input =
        decodedSelected
        + r.outcomes.filter {
          $0.metadata == nil && $0.issue?.stage == .decoding
        }.count
      for outcome in r.outcomes {
        rows.append((r.loopIndex, outcome.sourceFrameIndex, outcome.metadata))
        switch outcome.status {
        case .completed: completed += 1
        case .failed: failed += 1
        case .unfinished: unfinished += 1
        }
      }
      let c = r.coverage
      countersValid = c.countsAreConsistent
      nativeOutcomesValid =
        c.invalidOutcomeIndices.isEmpty
        && c.outOfOrderFrameIndices.isEmpty
      decodeLossEvents =
        r.decodeFailure != nil
          || (r.issue?.stage == .decoding && r.status == .failed) ? 1 : 0
    case .camera(let r):
      input = r.counts.inputFrames
      completed = r.counts.completedFrames
      failed = r.counts.failedFrames
      skipped = r.counts.skippedFrames
      unfinished = r.counts.unfinishedFrames
      for outcome in r.queue?.outcomes ?? [] {
        rows.append((0, outcome.metadata.sourceFrameIndex, outcome.metadata))
      }
      loops = nil // Camera coverage uses observed callbacks, never a frozen video extent.
    case .paced(let r):
      firstLoop = r.firstLoopIndex
      loops = r.requestedLoopCount
      explicitTrack = r.requestedTrackID != nil
      let c = r.counts ?? PacedVideoRunCounts()
      input = c.inputFrames
      completed = c.completedFrames
      failed = c.failedFrames
      skipped = c.skippedFrames
      unfinished = c.unfinishedFrames
      decodedNotDelivered = r.decodedButNotDeliveredFrames ?? 0
      decodeLossEvents = r.decodeLosses.count
      countersValid =
        (r.source?.deliveredFrames ?? 0) == input
        && (r.source?.decodedFrames ?? 0) >= input
        && (r.queue?.inputDeliveryMilliseconds.count ?? 0) == input
      for outcome in r.queue?.outcomes ?? [] {
        rows.append(
          (
            outcome.metadata.loopIndex, outcome.metadata.sourceFrameIndex,
            outcome.metadata
          ))
        if let result = outcome.result {
          nativeOutcomesValid =
            nativeOutcomesValid
            && result.metadata == outcome.metadata
            && result.status.rawValue == outcome.status.rawValue
        } else if outcome.status == .completed {
          nativeOutcomesValid = false
        }
      }
    }
    self.inputFrames = input
    self.completed = completed
    self.failed = failed
    self.skipped = skipped
    self.unfinished = unfinished
    observed = rows.count
    self.decodedNotDelivered = decodedNotDelivered
    self.decodeLossEvents = decodeLossEvents
    accountingBalanced =
      countersValid && nativeOutcomesValid
      && input == completed + failed + skipped + unfinished
      && input == rows.count

    let usable = reference.flatMap {
      $0.sourceID == run.sourceID && !explicitTrack ? $0 : nil
    }
    var required: Set<String>?
    if let usable, let loops, loops > 0, loops <= 10_000,
      firstLoop >= 0, firstLoop <= Int.max - loops
    {
      let indices = selected ?? Array(usable.pts.indices)
      required = Set(
        (firstLoop..<(firstLoop + loops)).flatMap { loop in
          indices.map { "\(loop):\($0)" }
        })
    }
    expected = required?.count
    var occurrences: [String: Int] = [:]
    var mismatches = Set<String>()
    var invalid = Set<String>()
    for row in rows {
      let key = "\(row.loop):\(row.index)"
      occurrences[key, default: 0] += 1
      if row.loop < 0 || row.index < 0 { invalid.insert(key) }
      if let metadata = row.metadata {
        if metadata.sourceID != run.sourceID || metadata.loopIndex != row.loop
          || metadata.sourceFrameIndex != row.index
        {
          invalid.insert(key)
        }
        if let usable, !usable.matches(metadata.presentationTime, index: row.index) {
          mismatches.insert(key)
        }
      } else {
        mismatches.insert(key)
      }
    }
    let observedKeys = Set(occurrences.keys)
    missing = required?.subtracting(observedKeys).sorted() ?? []
    unexpected = required.map { observedKeys.subtracting($0).sorted() } ?? []
    duplicates = occurrences.filter { $0.value > 1 }.keys.sorted()
    timestampMismatches = mismatches.sorted()
    self.invalid = invalid.sorted()
    var reasons: [String] = []
    if required == nil {
      reasons.append("No verified finite source reference for this selection/track")
    }
    if !accountingBalanced { reasons.append("Input counts or outcome records do not balance") }
    if !missing.isEmpty { reasons.append("Expected source identities are missing") }
    if !unexpected.isEmpty || !duplicates.isEmpty || !invalid.isEmpty {
      reasons.append("Unexpected, duplicate or invalid source identities")
    }
    if !mismatches.isEmpty {
      reasons.append("Source timestamps are missing or differ from the reference")
    }
    if decodeLossEvents > 0 { reasons.append("Decoder losses cannot be fully reconstructed") }
    if decodedNotDelivered > 0 {
      reasons.append("Decoded inputs were not delivered to the inference queue")
    }
    if unfinished > 0 { reasons.append("Input work remains unfinished") }
    if run.status == "running" { reasons.append("Run has not finalized") }
    self.reasons = reasons
  }

  /// Exports source completeness separately from numerical success and benchmark parity.
  var json: [String: Any] {
    [
      "status": status, "basis": expected == nil ? "unverified_source" : "frozen_manifest",
      "input_frames": inputFrames, "completed": completed, "failed": failed,
      "skipped": skipped, "unfinished": unfinished, "observed_outcomes": observed,
      "expected_frames": VideoExportRecord.nullable(expected),
      "missing_identities": missing, "duplicate_identities": duplicates,
      "unexpected_identities": unexpected, "timestamp_mismatches": timestampMismatches,
      "invalid_identities": invalid, "decoded_not_delivered": decodedNotDelivered,
      "decode_loss_events": decodeLossEvents, "decode_lost_frames": NSNull(),
      "decode_lost_frames_unavailable_reason": "Decoder errors do not identify every lost sample",
      "accounting_balanced": accountingBalanced, "reasons": reasons,
    ]
  }
}
