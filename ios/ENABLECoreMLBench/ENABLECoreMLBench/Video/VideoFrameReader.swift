import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation

/// Reports reader failures separately from clean end-of-file and cancellation.
nonisolated enum VideoReaderError: Error, Equatable, LocalizedError {
    case invalidInput(String)
    case noVideoTrack
    case trackNotFound(CMPersistentTrackID)
    case cannotOpen(String)
    case cannotStart(String)
    case notReading
    case decodeFailed(frameIndex: Int, reason: String)
    case invalidFrame(frameIndex: Int, reason: String)

    var errorDescription: String? {
        switch self {
        case .invalidInput(let reason): return "Invalid video input: \(reason)"
        case .noVideoTrack: return "The selected file has no video track."
        case .trackNotFound(let id): return "Video track \(id) was not found."
        case .cannotOpen(let reason): return "Cannot open video: \(reason)"
        case .cannotStart(let reason): return "Cannot start video decoding: \(reason)"
        case .notReading: return "Open a video before requesting frames."
        case .decodeFailed(let index, let reason):
            return "Video decoding failed at frame \(index): \(reason)"
        case .invalidFrame(let index, let reason):
            return "Invalid decoded frame \(index): \(reason)"
        }
    }
}

/// Decodes one local video track on demand, keeping mutable pixels on this actor.
actor VideoFrameReader {
    /// Distinguishes clean exhaustion, explicit/task cancellation and failures.
    enum State: Equatable, Sendable {
        case idle, opening, reading, completed, cancelled, failed
    }

    private(set) var state = State.idle
    private(set) var failure: VideoReaderError?
    private(set) var selectedTrackID: CMPersistentTrackID?
    private(set) var framesRead = 0

    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var sourceID = ""
    private var loopIndex = 0
    private var preferredTransform = CGAffineTransform.identity
    private var lastPresentationTime: CMTime?
    private var generation: UInt64 = 0

    /// Opens an app-owned local file, selecting the first video track by default.
    /// Reopening cancels the previous reader and starts source indexing at zero.
    func open(
        _ url: URL, sourceID: String, loopIndex: Int = 0,
        trackID: CMPersistentTrackID? = nil
    ) async throws {
        generation &+= 1
        let request = generation
        releaseReader()
        state = .opening
        failure = nil
        selectedTrackID = nil
        framesRead = 0
        lastPresentationTime = nil
        do {
            try Task.checkCancellation()
            guard url.isFileURL else {
                throw VideoReaderError.invalidInput("Choose a local video file.")
            }
            guard !sourceID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                loopIndex >= 0
            else {
                throw VideoReaderError.invalidInput(
                    "Source ID must be nonempty and loop index nonnegative.")
            }
            let properties = try url.resourceValues(forKeys: [.isRegularFileKey])
            guard properties.isRegularFile == true else {
                throw VideoReaderError.invalidInput("Choose a file, not a folder.")
            }
            let asset = AVURLAsset(url: url)
            let tracks = try await asset.loadTracks(withMediaType: .video)
            try checkRequest(request)
            guard !tracks.isEmpty else { throw VideoReaderError.noVideoTrack }
            let track: AVAssetTrack
            if let trackID {
                guard let selected = tracks.first(where: { $0.trackID == trackID })
                else { throw VideoReaderError.trackNotFound(trackID) }
                track = selected
            } else {
                track = tracks[0]
            }
            let transform = try await track.load(.preferredTransform)
            try checkRequest(request)
            let decoding = try AVAssetReader(asset: asset)
            let trackOutput = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [
                    kCVPixelBufferPixelFormatTypeKey as String:
                        kCVPixelFormatType_32BGRA
                ])
            trackOutput.alwaysCopiesSampleData = false
            guard decoding.canAdd(trackOutput) else {
                throw VideoReaderError.cannotStart("The video track is unsupported.")
            }
            decoding.add(trackOutput)
            // Store before starting so every failure/cancellation releases resources.
            reader = decoding
            output = trackOutput
            guard decoding.startReading() else {
                let reason = decoding.error?.localizedDescription
                    ?? "AVAssetReader did not start."
                throw VideoReaderError.cannotStart(reason)
            }
            try checkRequest(request)
            self.sourceID = sourceID
            self.loopIndex = loopIndex
            preferredTransform = transform
            selectedTrackID = track.trackID
            state = .reading
        } catch {
            // A superseded open must not overwrite or cancel the newer reader.
            guard request == generation else { throw CancellationError() }
            if error is CancellationError || Task.isCancelled {
                cancel()
                throw CancellationError()
            }
            let reported = error as? VideoReaderError
                ?? .cannotOpen(error.localizedDescription)
            throw fail(reported)
        }
    }

    /// Processes one frame synchronously on this actor and returns only Sendable data.
    /// Returns nil only at clean EOF. Consumer errors propagate after consuming the
    /// frame; decoding can continue. A synchronous decode finishes before a pending
    /// task cancellation is observed, and its result is then discarded.
    func nextFrame<Result: Sendable>(
        _ process: @Sendable (VideoFrame) throws -> Result
    ) throws -> Result? {
        do {
            try Task.checkCancellation()
            return try autoreleasepool {
                guard let frame = try readFrame() else { return nil }
                try Task.checkCancellation()
                let result = try process(frame)
                try Task.checkCancellation()
                return result
            }
        } catch is CancellationError {
            cancel()
            throw CancellationError()
        }
    }

    /// Stops reading and invalidates pending opens; completed/failed states persist.
    func cancel() {
        generation &+= 1
        guard state != .completed, state != .failed else { return }
        releaseReader()
        state = .cancelled
    }

    /// Reads decoded samples in presentation order and preserves their actual PTS.
    private func readFrame() throws -> VideoFrame? {
        switch state {
        case .completed: return nil
        case .cancelled: throw CancellationError()
        case .failed: throw failure ?? VideoReaderError.notReading
        case .idle, .opening: throw VideoReaderError.notReading
        case .reading: break
        }
        guard let reader, let output else { throw VideoReaderError.notReading }
        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            // Marker-only samples carry no frame and must not advance its index.
            if CMSampleBufferGetNumSamples(sample) == 0 { continue }
            guard CMSampleBufferGetNumSamples(sample) == 1,
                CMSampleBufferDataIsReady(sample),
                let pixels = CMSampleBufferGetImageBuffer(sample)
            else {
                throw fail(.invalidFrame(
                    frameIndex: framesRead, reason: "Missing decoded frame pixels."))
            }
            let frame: VideoFrame
            do {
                frame = try VideoFrame(
                    sourceID: sourceID, sourceFrameIndex: framesRead,
                    presentationTime: CMSampleBufferGetPresentationTimeStamp(sample),
                    preferredTransform: preferredTransform, loopIndex: loopIndex,
                    pixelBuffer: pixels)
            } catch {
                throw fail(.invalidFrame(
                    frameIndex: framesRead, reason: String(describing: error)))
            }
            if let previous = lastPresentationTime,
                CMTimeCompare(frame.metadata.presentationTime, previous) < 0
            {
                throw fail(.invalidFrame(
                    frameIndex: framesRead,
                    reason: "Presentation timestamps regressed."))
            }
            lastPresentationTime = frame.metadata.presentationTime
            framesRead += 1
            return frame
        }
        try Task.checkCancellation()
        switch reader.status {
        case .completed:
            releaseReader()
            state = .completed
            return nil
        case .cancelled:
            cancel()
            throw CancellationError()
        case .failed, .unknown, .reading:
            throw fail(.decodeFailed(
                frameIndex: framesRead,
                reason: reader.error?.localizedDescription
                    ?? "AVAssetReader stopped without reaching end-of-file."))
        @unknown default:
            throw fail(.decodeFailed(
                frameIndex: framesRead, reason: "Unknown AVAssetReader state."))
        }
    }

    /// Rejects cancelled or superseded metadata loads before committing a reader.
    private func checkRequest(_ request: UInt64) throws {
        try Task.checkCancellation()
        guard request == generation else { throw CancellationError() }
    }

    /// Retains the explicit error while releasing decoder resources.
    private func fail(_ error: VideoReaderError) -> VideoReaderError {
        releaseReader()
        failure = error
        state = .failed
        return error
    }

    /// Cancels any outstanding decode and releases the track output and asset reader.
    private func releaseReader() {
        reader?.cancelReading()
        output = nil
        reader = nil
    }
}
