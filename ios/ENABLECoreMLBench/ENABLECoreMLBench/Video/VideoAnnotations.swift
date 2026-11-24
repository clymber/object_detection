import CryptoKit
import Foundation

/// Reports unusable benchmark inputs without silently repairing their geometry.
nonisolated struct VideoBenchmarkError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
    /// Presents a concrete input or scoring failure in the import/result UI.
    init(_ message: String) { self.message = message }
}

/// Reads the generator's evaluation JSON; generation manifests are not required.
nonisolated struct VideoAnnotations: Decodable, Sendable {
    struct Info: Decodable, Sendable {
        let bbox_format: String
        let coordinate_origin: String
        let frame_index_base: Int
    }
    struct Video: Decodable, Sendable {
        let id: Int
        let file_name: String
        let sha256: String
        let width: Int
        let height: Int
        let fps: Double
        let frame_count: Int
        let duration_seconds: Double
    }
    struct Frame: Decodable, Sendable {
        let id: Int
        let video_id: Int
        let frame_index: Int
        let timestamp_seconds: Double
        let width: Int
        let height: Int
    }
    struct Box: Decodable, Sendable {
        let id: Int
        let image_id: Int
        let category_id: Int
        let bbox: [Double]
        let iscrowd: Int?
    }
    struct Category: Decodable, Sendable {
        let id: Int
        let name: String
    }

    let schema_version: Int
    let type: String
    let info: Info
    let videos: [Video]
    let images: [Frame]
    let annotations: [Box]
    let categories: [Category]

    var video: Video { videos[0] }

    /// Validates complete, unambiguous frame/box identities before scoring any model.
    static func decode(_ data: Data) throws -> VideoAnnotations {
        let value = try JSONDecoder().decode(Self.self, from: data)
        guard [1, 2].contains(value.schema_version), value.type == "coco_video_boxes",
            value.info.bbox_format == "xywh",
            value.info.coordinate_origin == "top_left", value.info.frame_index_base == 0,
            value.videos.count == 1 else {
            throw VideoBenchmarkError("Choose a video annotation JSON from the generator.")
        }
        let video = value.video
        guard video.width > 0, video.height > 0, video.frame_count > 0,
            video.fps.isFinite, video.fps > 0,
            video.duration_seconds.isFinite,
            abs(video.duration_seconds - Double(video.frame_count) / video.fps) < 0.00001,
            video.sha256.count == 64,
            video.sha256.allSatisfy({ "0123456789abcdef".contains($0) }),
            value.images.count == video.frame_count,
            Set(value.images.map(\.id)).count == value.images.count,
            Set(value.images.map(\.frame_index)) == Set(0..<video.frame_count),
            !value.categories.isEmpty,
            Set(value.categories.map(\.id)).count == value.categories.count else {
            throw VideoBenchmarkError("Annotations have invalid video or frame metadata.")
        }
        for frame in value.images {
            guard frame.video_id == video.id, frame.width == video.width,
                frame.height == video.height, frame.timestamp_seconds.isFinite,
                abs(frame.timestamp_seconds - Double(frame.frame_index) / video.fps)
                    < 0.00001 else {
                throw VideoBenchmarkError("Annotation frame geometry or timestamps differ.")
            }
        }
        let imageIDs = Set(value.images.map(\.id))
        let categoryIDs = Set(value.categories.map(\.id))
        guard Set(value.annotations.map(\.id)).count == value.annotations.count else {
            throw VideoBenchmarkError("Annotations contain duplicate box IDs.")
        }
        for box in value.annotations {
            guard imageIDs.contains(box.image_id), categoryIDs.contains(box.category_id),
                box.iscrowd ?? 0 == 0, box.bbox.count == 4,
                box.bbox.allSatisfy(\.isFinite), box.bbox[0] >= 0, box.bbox[1] >= 0,
                box.bbox[2] > 0, box.bbox[3] > 0,
                box.bbox[0] + box.bbox[2] <= Double(video.width) + 0.000001,
                box.bbox[1] + box.bbox[3] <= Double(video.height) + 0.000001 else {
                throw VideoBenchmarkError("Annotations contain invalid or unsupported boxes.")
            }
        }
        return value
    }

    /// Streams file hashing outside the timed inference run, checking cancellation.
    static func fileHash(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var digest = SHA256()
        while let chunk = try file.read(upToCount: 1024 * 1024), !chunk.isEmpty {
            try Task.checkCancellation()
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Fingerprints the exact annotation bytes for the exported accuracy result.
    static func dataHash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

/// Binds parsed ground truth to the exact user-selected video, independent of its name.
nonisolated struct ImportedVideoAnnotations: Sendable {
    let annotations: VideoAnnotations
    let fileName: String
    let sha256: String

    /// Rejects the boxed preview, transcoded copies and other mismatched videos.
    static func load(_ url: URL, videoURL: URL) throws -> Self {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        let data = try Data(contentsOf: url)
        try Task.checkCancellation()
        let annotations = try VideoAnnotations.decode(data)
        guard try VideoAnnotations.fileHash(videoURL) == annotations.video.sha256 else {
            throw VideoBenchmarkError(
                "These annotations do not match the selected video. "
                    + "Choose the original clean MP4 from Files, not the boxed preview.")
        }
        return Self(annotations: annotations, fileName: url.lastPathComponent,
                    sha256: VideoAnnotations.dataHash(data))
    }
}
