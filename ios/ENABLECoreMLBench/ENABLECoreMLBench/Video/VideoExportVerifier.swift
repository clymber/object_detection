import CryptoKit
import Foundation

/// Reports a disk round trip independently of source coverage and detector parity.
nonisolated struct VideoExportVerification: Sendable {
  let runID: String
  let records: Int
  let fingerprints: [String: String]
}

/// Reads returned files with independent JSON/CSV parsers on the calling worker.
nonisolated enum VideoExportVerifier {
  enum Invalid: LocalizedError {
    case files(String)
    var errorDescription: String? {
      switch self {
      case .files(let reason): "Export verification failed: \(reason)"
      }
    }
  }

  /// Checks integrity, every raw identity/timing field and an optional pre-share baseline.
  static func verify(_ directory: URL, expected: [String: String] = [:]) throws
    -> VideoExportVerification
  {
    let names = ["frames.jsonl", "timings.csv", "mobile_run.json"]
    var data: [String: Data] = [:]
    var hashes: [String: String] = [:]
    for name in names {
      try Task.checkCancellation()
      let bytes = try Data(contentsOf: directory.appendingPathComponent(name))
      let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
      if let baseline = expected[name], baseline != hash {
        throw Invalid.files("Saved \(name) differs from the prepared file")
      }
      data[name] = bytes
      hashes[name] = hash
    }
    let run = try object(data["mobile_run.json"]!)
    guard run["schema_version"] as? Int == 1,
      let runID = run["run_id"] as? String,
      let input = run["input"] as? [String: Any],
      let sourceID = input["source_id"] as? String,
      let files = run["files"] as? [String: [String: Any]],
      let count = run["frame_records"] as? Int,
      files["frames.jsonl"]?["sha256"] as? String == hashes["frames.jsonl"],
      files["timings.csv"]?["sha256"] as? String == hashes["timings.csv"]
    else { throw Invalid.files("Run metadata or file hashes do not match") }
    var lines = data["frames.jsonl"]!.split(separator: 0x0A).makeIterator()
    guard let csv = String(data: data["timings.csv"]!, encoding: .utf8) else {
      throw Invalid.files("CSV is not UTF-8")
    }
    var header = false
    var records = 0
    try parseCSV(csv) { columns in
      try Task.checkCancellation()
      if !header {
        guard columns == VideoExportRecord.csvColumns else {
          throw Invalid.files("CSV columns differ from schema v1")
        }
        header = true
        return
      }
      guard let line = lines.next() else { throw Invalid.files("Extra CSV records") }
      let row = try object(Data(line))
      guard row["run_id"] as? String == runID,
        row["source_id"] as? String == sourceID,
        columns.count == VideoExportRecord.csvColumns.count,
        let timing = row["timing_ms"] as? [String: Any]
      else { throw Invalid.files("Frame identity or column count differs") }
      let pts = row["pts"] as? [String: Any] ?? [:]
      let values: [Any?] =
        [
          row["run_id"], row["source_id"], row["loop_index"],
          row["source_frame_index"], pts["value"], pts["timescale"], pts["flags"],
          pts["epoch"], row["status"], row["publication_status"],
        ]
        + VideoExportRecord.timingNames.map { timing[$0] }
      for (index, value) in values.enumerated() {
        let cell = columns[index]
        if value == nil || value is NSNull {
          guard cell.isEmpty else { throw Invalid.files("Missing value became nonempty") }
        } else if let text = value as? String {
          guard cell == text else { throw Invalid.files("CSV identity/status differs") }
        } else if let number = value as? NSNumber {
          let equal =
            index >= 10
            ? Double(cell) == number.doubleValue
            : Int64(cell) == number.int64Value
          guard equal else { throw Invalid.files("CSV timing/PTS differs") }
        } else {
          throw Invalid.files("Unexpected CSV value type")
        }
      }
      records += 1
    }
    guard header, lines.next() == nil, records == count,
      files["frames.jsonl"]?["records"] as? Int == records,
      files["timings.csv"]?["records"] as? Int == records
    else { throw Invalid.files("File record counts differ") }
    return VideoExportVerification(runID: runID, records: records, fingerprints: hashes)
  }

  /// Requires a JSON object while Foundation rejects nonfinite numeric output.
  private static func object(_ data: Data) throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { throw Invalid.files("Expected a JSON object") }
    return object
  }

  /// Parses quoted fields, escaped quotes and CRLF, including newlines within IDs.
  private static func parseCSV(_ text: String, consume: ([String]) throws -> Void) throws {
    var fields: [String] = []
    var field = ""
    var quoted = false
    var afterQuote = false
    var skipLF = false
    for char in text.unicodeScalars {
      if skipLF {
        skipLF = false
        if char == "\n" { continue }
      }
      if quoted {
        if char == "\"" {
          quoted = false
          afterQuote = true
        } else {
          field.unicodeScalars.append(char)
        }
      } else if afterQuote && char == "\"" {
        field += "\""
        quoted = true
        afterQuote = false
      } else if char == "," || char == "\r" || char == "\n" {
        fields.append(field)
        field = ""
        afterQuote = false
        if char != "," {
          try consume(fields)
          fields = []
          skipLF = char == "\r"
        }
      } else if char == "\"" && field.isEmpty && !afterQuote {
        quoted = true
      } else if afterQuote || char == "\"" {
        throw Invalid.files("Malformed quoted CSV field")
      } else {
        field.unicodeScalars.append(char)
      }
    }
    guard !quoted else { throw Invalid.files("Unterminated quoted CSV field") }
    if !fields.isEmpty || !field.isEmpty || afterQuote {
      fields.append(field)
      try consume(fields)
    }
  }
}
