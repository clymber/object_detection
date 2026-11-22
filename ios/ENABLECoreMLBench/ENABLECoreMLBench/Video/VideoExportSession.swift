import Foundation
import Observation

/// Keeps saved-file feedback visible until another check or explicit input change.
nonisolated enum VideoExportVerificationStatus: String, Sendable {
    case idle = "Not checked"
    case checking = "Checking saved files…"
    case verified = "Saved files verified"
    case failed = "Verification failed"
    case cancelled = "Verification cancelled"
}

/// Owns export progress separately from the detector's measured run status.
@MainActor
@Observable
final class VideoExportSession {
    typealias Writer = @Sendable (VideoRunExport, URL) async throws -> VideoExportFiles
    typealias Reader = @Sendable (URL, [String: String]) async throws -> VideoExportVerification
    private let write: Writer
    private let read: Reader
    private var task: Task<Void, Never>?
    private var generation = UUID()
    private(set) var isBusy = false
    private(set) var files: VideoExportFiles?
    private(set) var message: String?
    private(set) var verification: VideoExportVerification?
    private(set) var verificationStatus = VideoExportVerificationStatus.idle
    private(set) var verificationMessage: String?

    /// Accepts isolated disk workers for lifecycle and stale-result checks.
    init(read: Reader? = nil, write: Writer? = nil) {
        let exporter = VideoRunExporter()
        self.write = write ?? { try await exporter.export($0, sourceURL: $1) }
        self.read = read ?? { directory, expected in
            let worker = Task.detached {
                let accessed = directory.startAccessingSecurityScopedResource()
                defer { if accessed { directory.stopAccessingSecurityScopedResource() } }
                return try VideoExportVerifier.verify(directory, expected: expected)
            }
            return try await withTaskCancellationHandler {
                try await worker.value
            } onCancel: { worker.cancel() }
        }
    }

    /// Freezes a finished report while disk work runs outside the UI actor.
    func start(_ run: VideoRunExport, sourceURL: URL) {
        guard !isBusy else { return }
        reset()
        let request = generation
        let write = self.write
        isBusy = true
        message = "Preparing export files…"
        task = Task { [weak self] in
            do {
                let files = try await write(run, sourceURL)
                guard !Task.isCancelled, let self, self.generation == request else { return }
                self.files = files
                self.message = "Three files ready to save or share."
                self.isBusy = false
                self.task = nil
            } catch {
                guard let self, self.generation == request else { return }
                self.message = error is CancellationError
                    ? "Export cancelled." : "Export failed: \(error.localizedDescription)"
                self.isBusy = false
                self.task = nil
            }
        }
    }

    /// Invalidates old receipts without deleting files an active share sheet may use.
    func reset() {
        generation = UUID()
        task?.cancel()
        task = nil
        isBusy = false
        files = nil
        message = nil
        verification = nil
        verificationStatus = .idle
        verificationMessage = nil
    }

    /// Cancels explicit Stop while retaining prepared files and visible feedback.
    func cancel() {
        guard isBusy else { return }
        generation = UUID()
        task?.cancel()
        task = nil
        isBusy = false
        if verificationStatus == .checking {
            verificationStatus = .cancelled
            verificationMessage = "The prepared files are still available. Choose a saved folder to try again."
        } else { message = "Export cancelled." }
    }

    /// Reopens a user-selected saved folder and compares all files to pre-share hashes.
    func verifySaved(_ directory: URL) {
        guard !isBusy, let files else { return }
        let request = generation
        let expected = files.fingerprints
        let read = self.read
        isBusy = true
        verification = nil
        verificationStatus = .checking
        verificationMessage = "Folder: \(directory.lastPathComponent)"
        task = Task { [weak self] in
            do {
                let report = try await read(directory, expected)
                guard !Task.isCancelled, let self, self.generation == request else { return }
                self.verification = report
                self.verificationStatus = .verified
                self.verificationMessage = "\(report.records) frame records match. Folder: \(directory.lastPathComponent)"
                self.isBusy = false
                self.task = nil
            } catch {
                guard let self, self.generation == request else { return }
                self.verificationStatus = error is CancellationError ? .cancelled : .failed
                self.verificationMessage = error.localizedDescription
                self.isBusy = false
                self.task = nil
            }
        }
    }

    /// Shows file-provider failures while allowing a cancelled folder picker to close.
    func reportVerificationFailure(_ error: any Error) {
        guard (error as NSError).code != NSUserCancelledError else { return }
        verification = nil
        verificationStatus = .failed
        verificationMessage = "Could not open saved results: \(error.localizedDescription)"
    }
}
