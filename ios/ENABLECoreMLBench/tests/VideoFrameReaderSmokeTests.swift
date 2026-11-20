import AVFAudio
import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation

/// Exercises real movie decoding, EOF, cancellation, recovery and damaged media.
@main
@MainActor
struct VideoFrameReaderSmokeTests {
    private static let timestamps: [Int64] = [0, 1001, 4004, 6006]
    private static let transform = CGAffineTransform(
        a: 0, b: 1, c: -1, d: 0, tx: 32, ty: 0)

    /// Creates private fixtures and runs the reader without models or app UI.
    static func main() async throws {
        let directory = URL(
            fileURLWithPath: "outputs/ios_builds/video_pipeline/fixtures")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let video = directory.appendingPathComponent("irregular_rotated.mov")
        try await writeVideo(to: video)
        try await testFramesAndEOF(video)
        try await testTrackSelection(video)
        try await testCancellationAndRestart(video)
        try await testTaskCancellation(video)
        try await testInvalidInputs(video, directory: directory)
        try await testCorruptDecode(video, directory: directory)
        try await testConsumerFailure(video)
        print("Video frame reader: all 7 checks passed.")
    }

    /// Checks irregular PTS, real pixels, presentation orientation and repeated EOF.
    private static func testFramesAndEOF(_ url: URL) async throws {
        let reader = VideoFrameReader()
        try await reader.open(url, sourceID: "fixture-track")
        for index in timestamps.indices {
            let snapshot = try await reader.nextFrame { try inspectFrame($0) }
            let frame = snapshot!.metadata
            precondition(frame.sourceID == "fixture-track" && frame.loopIndex == 0)
            precondition(frame.sourceFrameIndex == index)
            precondition(frame.presentationTime.value == timestamps[index])
            precondition(frame.presentationTime.timescale == 30000)
            precondition(frame.width == 64 && frame.height == 32)
            precondition(frame.preferredTransform == transform)
            precondition(abs(Int(snapshot!.blue) - (16 + 64 * index)) <= 8)
        }
        let eof = try await reader.nextFrame { $0.metadata }
        let repeatedEOF = try await reader.nextFrame { $0.metadata }
        let state = await reader.state
        let count = await reader.framesRead
        let failure = await reader.failure
        precondition(eof == nil && repeatedEOF == nil && state == .completed)
        precondition(count == timestamps.count && failure == nil)
    }

    /// Reopens a specific track and reports absent track IDs without falling back.
    private static func testTrackSelection(_ url: URL) async throws {
        let reader = VideoFrameReader()
        try await reader.open(url, sourceID: "fixture-track")
        let trackID = await reader.selectedTrackID
        precondition(trackID != nil)
        try await reader.open(url, sourceID: "fixture-track", trackID: trackID)
        let first = try await reader.nextFrame { $0.metadata }
        precondition(first?.sourceFrameIndex == 0)
        let error = try await expectFailure(reader) {
            try await reader.open(url, sourceID: "fixture-track", trackID: .max)
        }
        precondition(error == .trackNotFound(.max))
        await reader.cancel()
        let state = await reader.state
        precondition(state == .failed, "Cancellation erased a recorded failure")
    }

    /// Stops midstream, rejects further reads and restarts with a fresh loop identity.
    private static func testCancellationAndRestart(_ url: URL) async throws {
        let reader = VideoFrameReader()
        try await reader.open(url, sourceID: "fixture-track")
        _ = try await reader.nextFrame { $0.metadata }
        await reader.cancel()
        let state = await reader.state
        let count = await reader.framesRead
        precondition(state == .cancelled && count == 1)
        do {
            _ = try await reader.nextFrame { $0.metadata }
            preconditionFailure("Cancelled reader returned a frame")
        } catch is CancellationError {}
        try await reader.open(url, sourceID: "fixture-track", loopIndex: 1)
        let restarted = try await reader.nextFrame { $0.metadata }
        precondition(restarted?.sourceFrameIndex == 0 && restarted?.loopIndex == 1)
        precondition(restarted?.presentationTime == .zero)
        await reader.cancel()
    }

    /// Observes task cancellation before opening or decoding and releases the reader.
    private static func testTaskCancellation(_ url: URL) async throws {
        let reader = VideoFrameReader()
        let opening = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            try await reader.open(url, sourceID: "fixture-track")
        }
        do {
            try await opening.value
            preconditionFailure("Cancelled open succeeded")
        } catch is CancellationError {}
        let openingState = await reader.state
        precondition(openingState == .cancelled)
        try await reader.open(url, sourceID: "fixture-track")
        let reading = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await reader.nextFrame { $0.metadata }
        }
        do {
            _ = try await reading.value
            preconditionFailure("Cancelled task returned a frame")
        } catch is CancellationError {}
        let state = await reader.state
        let count = await reader.framesRead
        precondition(state == .cancelled && count == 0)
    }

    /// Rejects unopened reads, missing/invalid files, audio-only inputs and bad IDs.
    private static func testInvalidInputs(_ video: URL, directory: URL) async throws {
        let reader = VideoFrameReader()
        do {
            _ = try await reader.nextFrame { $0.metadata }
            preconditionFailure("Unopened reader returned EOF")
        } catch let error as VideoReaderError {
            precondition(error == .notReading)
        }
        let invalid = directory.appendingPathComponent("invalid.mov")
        try Data("not a movie".utf8).write(to: invalid)
        let audio = directory.appendingPathComponent("audio_only.caf")
        try writeAudio(to: audio)
        let audioError = try await expectFailure(reader) {
            try await reader.open(audio, sourceID: "audio")
        }
        precondition(audioError == .noVideoTrack)
        for url in [invalid, directory.appendingPathComponent("missing.mov"), directory,
                    URL(string: "https://example.invalid/video.mp4")!] {
            _ = try await expectFailure(reader) {
                try await reader.open(url, sourceID: "fixture-track")
            }
        }
        for (source, loop) in [(" \n", 0), ("fixture-track", -1)] {
            _ = try await expectFailure(reader) {
                try await reader.open(video, sourceID: source, loopIndex: loop)
            }
        }
        try await reader.open(video, sourceID: "recovered-track")
        let frame = try await reader.nextFrame { $0.metadata }
        precondition(frame?.sourceID == "recovered-track")
        await reader.cancel()
    }

    /// Forces decode failure by damaging pixels while retaining movie metadata.
    private static func testCorruptDecode(_ video: URL, directory: URL) async throws {
        let damaged = directory.appendingPathComponent("damaged.mov")
        try corruptMedia(video, to: damaged)
        let reader = VideoFrameReader()
        let error = try await expectFailure(reader) {
            try await reader.open(damaged, sourceID: "damaged-track")
            while let _ = try await reader.nextFrame({ $0.metadata }) {}
        }
        switch error {
        case .cannotStart, .decodeFailed: break
        default: preconditionFailure("Expected a decode failure, received \(error)")
        }
        // A failed read must remain a failure, including on subsequent requests.
        do {
            _ = try await reader.nextFrame { $0.metadata }
            preconditionFailure("Failed reader returned EOF")
        } catch let repeated as VideoReaderError { precondition(repeated == error) }
    }

    /// Keeps consumer failures distinct from decoder errors and advances only once.
    private static func testConsumerFailure(_ url: URL) async throws {
        let reader = VideoFrameReader()
        try await reader.open(url, sourceID: "fixture-track")
        do {
            _ = try await reader.nextFrame { _ -> VideoFrameMetadata in
                throw ConsumerFailure.fixture
            }
            preconditionFailure("Consumer failure was lost")
        } catch ConsumerFailure.fixture {}
        let state = await reader.state
        let failure = await reader.failure
        precondition(state == .reading && failure == nil)
        let next = try await reader.nextFrame { $0.metadata }
        precondition(next?.sourceFrameIndex == 1)
        await reader.cancel()
    }

    /// Requires a retained reader error, distinguishing it from EOF and cancellation.
    private static func expectFailure(
        _ reader: VideoFrameReader, operation: () async throws -> Void
    ) async throws -> VideoReaderError {
        do {
            try await operation()
            preconditionFailure("Invalid video operation succeeded")
        } catch let error as VideoReaderError {
            let state = await reader.state
            let retained = await reader.failure
            precondition(state == .failed && retained == error)
            return error
        }
    }

    /// Writes a rotated movie with distinguishable frames and irregular sample times.
    private static func writeVideo(to url: URL) async throws {
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 64, AVVideoHeightKey: 32,
        ])
        input.mediaTimeScale = 30000
        input.transform = transform
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: 64,
                kCVPixelBufferHeightKey as String: 32,
            ])
        writer.add(input)
        guard writer.startWriting() else {
            throw FixtureFailure.encoding(String(describing: writer.error))
        }
        writer.startSession(atSourceTime: .zero)
        for index in timestamps.indices {
            let deadline = ContinuousClock.now.advanced(by: .seconds(10))
            while !input.isReadyForMoreMediaData && ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(5))
            }
            precondition(
                input.isReadyForMoreMediaData, "Video fixture encoder timed out")
            var buffer: CVPixelBuffer?
            precondition(CVPixelBufferCreate(
                kCFAllocatorDefault, 64, 32, kCVPixelFormatType_32BGRA,
                nil, &buffer) == kCVReturnSuccess)
            let pixels = buffer!
            precondition(CVPixelBufferLockBaseAddress(pixels, []) == kCVReturnSuccess)
            let base = CVPixelBufferGetBaseAddress(pixels)!
                .assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(pixels)
            for row in 0..<32 {
                for column in 0..<64 {
                    let offset = row * stride + column * 4
                    for channel in 0..<3 {
                        base[offset + channel] = UInt8(16 + 64 * index)
                    }
                    base[offset + 3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(pixels, [])
            precondition(adaptor.append(
                pixels, withPresentationTime: CMTime(value: timestamps[index],
                                                    timescale: 30000)))
        }
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else {
            throw FixtureFailure.encoding(String(describing: writer.error))
        }
    }

    /// Produces a valid audio-only asset to distinguish it from malformed files.
    private static func writeAudio(to url: URL) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: 8000, channels: 1)!
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 160)!
        buffer.frameLength = 160
        buffer.floatChannelData![0].initialize(repeating: 0, count: 160)
        try file.write(from: buffer)
    }

    /// Damages only mdat payloads, preserving track metadata and sample tables.
    private static func corruptMedia(_ source: URL, to target: URL) throws {
        var bytes = try Data(contentsOf: source)
        var offset = 0
        var corrupted = false
        while offset + 8 <= bytes.count {
            let shortSize = bytes[offset..<offset + 4].reduce(UInt64(0)) {
                ($0 << 8) | UInt64($1)
            }
            let type = String(decoding: bytes[offset + 4..<offset + 8], as: UTF8.self)
            let header = shortSize == 1 ? 16 : 8
            precondition(offset + header <= bytes.count)
            let size: UInt64
            if shortSize == 1 {
                size = bytes[offset + 8..<offset + 16].reduce(UInt64(0)) {
                    ($0 << 8) | UInt64($1)
                }
            } else {
                size = shortSize == 0 ? UInt64(bytes.count - offset) : shortSize
            }
            precondition(size >= UInt64(header) && size <= UInt64(bytes.count - offset))
            let end = offset + Int(size)
            if type == "mdat" {
                let damage = repeatElement(UInt8(255), count: Int(size) - header)
                bytes.replaceSubrange(offset + header..<end, with: damage)
                corrupted = true
            }
            offset = end
        }
        precondition(corrupted, "Movie fixture has no media-data atom")
        try bytes.write(to: target)
    }
}

/// Returns only value metadata and a pixel sample from the reader actor.
nonisolated private struct FrameSnapshot: Sendable {
    let metadata: VideoFrameMetadata
    let blue: UInt8
}

/// Represents a downstream processing failure unrelated to video decoding.
nonisolated private enum ConsumerFailure: Error { case fixture }

/// Preserves the encoder's diagnostic when generating test media fails.
nonisolated private enum FixtureFailure: Error { case encoding(String) }

/// Inspects real decoded pixels inside the reader worker, away from the UI thread.
nonisolated private func inspectFrame(_ frame: VideoFrame) throws -> FrameSnapshot {
    precondition(!Thread.isMainThread)
    precondition(CVPixelBufferGetPixelFormatType(frame.pixelBuffer)
                 == kCVPixelFormatType_32BGRA)
    precondition(CVPixelBufferLockBaseAddress(frame.pixelBuffer, .readOnly)
                 == kCVReturnSuccess)
    defer { CVPixelBufferUnlockBaseAddress(frame.pixelBuffer, .readOnly) }
    let blue = CVPixelBufferGetBaseAddress(frame.pixelBuffer)!
        .assumingMemoryBound(to: UInt8.self)[0]
    return FrameSnapshot(metadata: frame.metadata, blue: blue)
}
