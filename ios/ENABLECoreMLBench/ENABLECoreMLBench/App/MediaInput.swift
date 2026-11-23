import AVFoundation
import CoreMedia
import CoreTransferable
import Foundation
import Observation
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// Separates frozen images, bundled development clips and personal media.
nonisolated enum MediaInputSource: String, CaseIterable, Identifiable {
    case testImages = "Test images"
    case testVideos = "Test videos"
    case photo = "Photo"
    case video = "Video"
    case camera = "Live camera"

    var id: String { rawValue }
    var isVideo: Bool { self == .testVideos || self == .video }
}

/// Identifies the selected media type without relying on a filename extension.
nonisolated enum ImportedMediaKind: Sendable {
    case photo
    case video
}

/// Identifies an app-owned cache copy or a read-only bundled resource.
nonisolated struct ImportedMediaFile: Transferable, Sendable, Equatable {
    let url: URL
    var displayName: String { url.lastPathComponent }

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .image) { received in
            try await MediaImportWorker.shared.copy(received.file)
        }
        FileRepresentation(importedContentType: .movie) { received in
            try await MediaImportWorker.shared.copy(received.file)
        }
    }
}

/// Carries only validated playback metadata across actor boundaries.
nonisolated struct ImportedVideo: Sendable, Equatable {
    let file: ImportedMediaFile
    let durationSeconds: Double
    let sourceID: String?

    /// Keeps a stable catalog identity for bundled clips and URL identity for imports.
    init(file: ImportedMediaFile, durationSeconds: Double, sourceID: String? = nil) {
        self.file = file
        self.durationSeconds = durationSeconds
        self.sourceID = sourceID
    }
}

/// Copies security-scoped/provider files and reads metadata off the main actor.
actor MediaImportWorker {
    static let shared = MediaImportWorker()
    private let directory: URL

    /// Permits a private fixture cache for media import checks.
    init(directory: URL = URL.cachesDirectory.appendingPathComponent("ImportedMedia")) {
        self.directory = directory.standardizedFileURL
    }

    /// Copies a selected file before releasing its provider or security scope.
    func copy(_ source: URL) throws -> ImportedMediaFile {
        try Task.checkCancellation()
        let accessing = source.startAccessingSecurityScopedResource()
        defer { if accessing { source.stopAccessingSecurityScopedResource() } }
        let folder = directory.appendingPathComponent(UUID().uuidString)
        do {
            let properties = try source.resourceValues(forKeys: [.isDirectoryKey])
            guard properties.isDirectory != true else {
                throw DetectorError.invalidImage("Choose a media file, not a folder.")
            }
            try FileManager.default.createDirectory(
                at: folder, withIntermediateDirectories: true)
            let target = folder.appendingPathComponent(source.lastPathComponent)
            try FileManager.default.copyItem(at: source, to: target)
            try Task.checkCancellation()
            return ImportedMediaFile(url: target)
        } catch {
            try? FileManager.default.removeItem(at: folder)
            throw error
        }
    }

    /// Rejects unsupported or audio-only inputs before offering video playback.
    func video(
        _ file: ImportedMediaFile, sourceID: String? = nil
    ) async throws -> ImportedVideo {
        try Task.checkCancellation()
        let asset = AVURLAsset(url: file.url)
        let playable = try await asset.load(.isPlayable)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let duration = try await asset.load(.duration).seconds
        guard playable, !tracks.isEmpty, duration.isFinite, duration > 0 else {
            throw DetectorError.invalidImage("The selected video cannot be played.")
        }
        try Task.checkCancellation()
        return ImportedVideo(file: file, durationSeconds: duration, sourceID: sourceID)
    }

    /// Removes only a UUID folder inside this worker's private import cache.
    func remove(_ file: ImportedMediaFile) {
        let folder = file.url.deletingLastPathComponent().standardizedFileURL
        guard folder.deletingLastPathComponent().standardizedFileURL.path == directory.path,
            UUID(uuidString: folder.lastPathComponent) != nil
        else { return }
        try? FileManager.default.removeItem(at: folder)
    }
}

/// Publishes import progress while workers own file copying and video metadata.
@MainActor
@Observable
final class MediaImportSession {
    private(set) var isBusy = false
    private(set) var photo: ImportedMediaFile?
    private(set) var video: ImportedVideo?
    private(set) var errorMessage: String?
    @ObservationIgnored private let worker: MediaImportWorker
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var generation: UInt64 = 0

    /// Accepts a private worker/cache for import lifecycle checks.
    init(worker: MediaImportWorker = .shared) { self.worker = worker }

    /// Imports a Photos picker selection without requesting library-wide access.
    func select(_ item: PhotosPickerItem, as kind: ImportedMediaKind) {
        begin(kind: kind) {
            guard
                let file = try await item.loadTransferable(type: ImportedMediaFile.self)
            else {
                throw DetectorError.invalidImage(
                    "The selected media could not be imported.")
            }
            return file
        }
    }

    /// Copies a Files picker selection while its security scope is active.
    func importFile(_ url: URL, as kind: ImportedMediaKind) {
        let worker = worker
        begin(kind: kind) { try await worker.copy(url) }
    }

    /// Validates a bundled clip without copying it or acquiring cache ownership.
    func selectTestVideo(_ sample: BundledVideo, bundle: Bundle = .main) {
        begin(kind: .video, sourceID: sample.sourceID) {
            ImportedMediaFile(url: try sample.resourceURL(in: bundle))
        }
    }

    /// Presents picker errors while retaining an already selected input.
    func report(_ error: Error) { errorMessage = error.localizedDescription }

    /// Clears input state and releases only app-owned copies of old selections.
    func reset() {
        generation &+= 1
        task?.cancel()
        task = nil
        isBusy = false
        let files = [photo, video?.file].compactMap { $0 }
        photo = nil
        video = nil
        errorMessage = nil
        let worker = worker
        Task { for file in files { await worker.remove(file) } }
    }

    /// Stops an in-flight import without discarding a completed selection.
    func cancel() { if isBusy { reset() } }

    /// Rejects late transfer/metadata results and cleans up cancelled copies.
    private func begin(
        kind: ImportedMediaKind, sourceID: String? = nil,
        operation: @escaping @Sendable () async throws -> ImportedMediaFile
    ) {
        reset()
        let request = generation
        isBusy = true
        let worker = worker
        task = Task { [weak self] in
            var imported: ImportedMediaFile?
            do {
                let file = try await operation()
                imported = file
                try Task.checkCancellation()
                let video = kind == .video
                    ? try await worker.video(file, sourceID: sourceID) : nil
                try Task.checkCancellation()
                guard let self, self.generation == request else {
                    await worker.remove(file)
                    return
                }
                self.photo = kind == .photo ? file : nil
                self.video = video
                self.isBusy = false
                self.task = nil
            } catch {
                if let imported { await worker.remove(imported) }
                guard let self, self.generation == request else { return }
                self.errorMessage = error.localizedDescription
                self.isBusy = false
                self.task = nil
            }
        }
    }
}
