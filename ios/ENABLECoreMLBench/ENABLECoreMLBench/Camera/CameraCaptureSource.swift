import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

/// Shares only a session reference with the MainActor preview layer.
/// Configuration/start/stop stay on the source's serial session queue.
nonisolated final class CameraPreviewHandle: @unchecked Sendable {
    let session = AVCaptureSession()
}

#if os(iOS)
    /// Confines AVFoundation mutations to sessionQueue and sample pixels to frameQueue.
    /// The lock protects numerical snapshots and the synchronous cancellation boundary.
    nonisolated final class CameraCaptureSource: NSObject, CameraCapturing,
        @unchecked Sendable
    {
        let preview: CameraPreviewHandle? = CameraPreviewHandle()
        private let sessionQueue = DispatchQueue(label: "bench.camera.session")
        private let frameQueue = DispatchQueue(
            label: "bench.camera.frames", qos: .userInitiated)
        private let lock = NSLock()
        private var report: CameraCaptureReport?
        private var token = UUID()
        private var active = false
        private var closing = false
        private var accepting = false
        // Accessed only on sessionQueue after initialization.
        private var continuation: CheckedContinuation<Void, Error>?
        private var delegate: CameraSampleDelegate?
        private var observers: [NSObjectProtocol] = []
        private var device: AVCaptureDevice?

        var snapshot: CameraCaptureReport? { lock.withLock { report } }
        var authorization: CameraAuthorization {
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized: .authorized
            case .notDetermined: .unknown
            case .restricted: .restricted
            default: .denied
            }
        }

        /// Requests only video access; no audio input or microphone permission is used.
        func authorize() async -> Bool {
            if authorization == .authorized { return true }
            guard authorization == .unknown else { return false }
            return await AVCaptureDevice.requestAccess(for: .video)
        }

        /// Runs until explicit Stop, task cancellation, interruption or capture failure.
        func run(
            configuration: CameraConfiguration, sourceID: String,
            started: @escaping @Sendable (Double) -> Void,
            deliver: @escaping @Sendable (PacedVideoFrame) throws -> Void
        ) async throws {
            try configuration.validate()
            let request = UUID()
            try lock.withLock {
                guard !active else { throw CameraError.alreadyRunning }
                token = request
                active = true
                closing = false
                accepting = false
                report = CameraCaptureReport(
                    sourceID: sourceID, configuration: configuration)
            }
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { pending in
                    sessionQueue.async { [self] in
                        continuation = pending
                        if lock.withLock({ closing }) {
                            finish()
                            return
                        }
                        do {
                            guard authorization == .authorized else {
                                throw CameraError.denied
                            }
                            try configure(
                                configuration, request: request, deliver: deliver)
                            guard !lock.withLock({ closing }) else {
                                finish()
                                return
                            }
                            let before = InferenceClock.nowMilliseconds()
                            let host = CMClockGetTime(CMClockGetHostTimeClock())
                            let mapping = try CameraClockMapping(
                                host: host, before: before,
                                after: InferenceClock.nowMilliseconds())
                            let startTime = InferenceClock.nowMilliseconds()
                            lock.withLock {
                                report?.mapping = mapping
                                report?.startedMilliseconds = startTime
                                report?.isRunning = true
                                accepting = true
                            }
                            started(startTime)
                            preview!.session.startRunning()
                            if !preview!.session.isRunning {
                                throw CameraError.interrupted("Capture did not start")
                            }
                        } catch { fail(error, request: request) }
                    }
                }
            } onCancel: {
                self.stop()
            }
        }

        /// Closes admission immediately, before the blocking hardware stop is enqueued.
        func stop() {
            let changed = lock.withLock { () -> Bool in
                guard active, !closing else { return false }
                closing = true
                accepting = false
                report?.endedMilliseconds = InferenceClock.nowMilliseconds()
                return true
            }
            if changed { sessionQueue.async { [self] in finish() } }
        }

        /// Selects an exact format/cadence and physically orients/mirrors BGRA buffers.
        private func configure(
            _ config: CameraConfiguration, request: UUID,
            deliver: @escaping @Sendable (PacedVideoFrame) throws -> Void
        ) throws {
            let session = preview!.session
            session.beginConfiguration()
            defer { session.commitConfiguration() }
            session.inputs.forEach { session.removeInput($0) }
            session.outputs.forEach { session.removeOutput($0) }
            session.sessionPreset = .inputPriority
            let position: AVCaptureDevice.Position =
                config.lens == .rear ? .back : .front
            let selected =
                AVCaptureDevice.default(
                    .builtInWideAngleCamera, for: .video, position: position)
                ?? (position == .front
                    ? AVCaptureDevice.default(
                        .builtInTrueDepthCamera, for: .video, position: .front) : nil)
            guard let camera = selected,
                let format = camera.formats.first(where: { format in
                    let size = CMVideoFormatDescriptionGetDimensions(
                        format.formatDescription)
                    return size.width == config.resolution.width
                        && size.height == config.resolution.height
                        && format.videoSupportedFrameRateRanges.contains {
                            $0.minFrameRate <= Double(config.framesPerSecond)
                                && $0.maxFrameRate >= Double(config.framesPerSecond)
                        }
                })
            else {
                throw CameraError.configuration(
                    "Requested camera resolution/FPS is unavailable.")
            }
            let input = try AVCaptureDeviceInput(device: camera)
            guard session.canAddInput(input) else { throw CameraError.unavailable }
            session.addInput(input)
            try camera.lockForConfiguration()
            camera.activeFormat = format
            let duration = CMTime(value: 1, timescale: Int32(config.framesPerSecond))
            camera.activeVideoMinFrameDuration = duration
            camera.activeVideoMaxFrameDuration = duration
            if camera.isFocusModeSupported(.continuousAutoFocus) {
                camera.focusMode = .continuousAutoFocus
            }
            if camera.isExposureModeSupported(.continuousAutoExposure) {
                camera.exposureMode = .continuousAutoExposure
            }
            if camera.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                camera.whiteBalanceMode = .continuousAutoWhiteBalance
            }
            camera.unlockForConfiguration()
            device = camera
            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            guard session.canAddOutput(output) else { throw CameraError.unavailable }
            session.addOutput(output)
            guard let connection = output.connection(with: .video),
                connection.isVideoRotationAngleSupported(
                    CGFloat(config.orientation.rawValue))
            else {
                throw CameraError.configuration(
                    "Requested camera orientation is unsupported.")
            }
            connection.videoRotationAngle = CGFloat(config.orientation.rawValue)
            connection.automaticallyAdjustsVideoMirroring = false
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = config.mirrored
            } else if config.mirrored {
                throw CameraError.configuration("Camera mirroring is unsupported.")
            }
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
            }
            delegate = CameraSampleDelegate(
                owner: self, request: request, deliver: deliver)
            output.setSampleBufferDelegate(delegate, queue: frameQueue)
            lock.withLock {
                report?.actual = [
                    "device_id": camera.uniqueID, "device_name": camera.localizedName,
                    "device_type": camera.deviceType.rawValue,
                    "position": config.lens.rawValue,
                    "sensor_width": String(config.resolution.width),
                    "sensor_height": String(config.resolution.height),
                    "pixel_format": "BGRA32",
                    "rotation_degrees": String(config.orientation.rawValue),
                    "mirrored": String(connection.isVideoMirrored),
                    "min_frame_duration":
                        "\(camera.activeVideoMinFrameDuration.value)/\(camera.activeVideoMinFrameDuration.timescale)",
                    "max_frame_duration":
                        "\(camera.activeVideoMaxFrameDuration.value)/\(camera.activeVideoMaxFrameDuration.timescale)",
                    "focus_mode": String(camera.focusMode.rawValue),
                    "exposure_mode": String(camera.exposureMode.rawValue),
                    "white_balance_mode": String(camera.whiteBalanceMode.rawValue),
                    "stabilization_mode": String(
                        connection.activeVideoStabilizationMode.rawValue),
                ]
            }
            for name in [
                AVCaptureSession.runtimeErrorNotification,
                AVCaptureSession.wasInterruptedNotification,
            ] {
                observers.append(
                    NotificationCenter.default.addObserver(
                        forName: name, object: session, queue: nil
                    ) {
                        [weak self] note in
                        let reason =
                            (note.userInfo?[AVCaptureSessionErrorKey] as? NSError)?
                            .localizedDescription
                            ?? "Capture became unavailable (\(note.userInfo?[AVCaptureSessionInterruptionReasonKey] ?? "unknown"))"
                        self?.fail(CameraError.interrupted(reason), request: request)
                    })
            }
        }

        /// Converts sample clock time to host time and admits without awaiting inference.
        fileprivate func receive(
            _ sample: CMSampleBuffer, dropped: Bool, request: UUID,
            deliver: @Sendable (PacedVideoFrame) throws -> Void
        ) {
            let received = InferenceClock.nowMilliseconds()
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            let context = lock.withLock { () -> (Int, CameraClockMapping, String)? in
                guard token == request, accepting, let mapping = report?.mapping else {
                    return nil
                }
                let index = report!.events.count
                report?.events.append(
                    CameraCaptureEvent(
                        index: index, kind: dropped ? .dropped : .delivered,
                        presentationTime: pts, hostPresentationTime: nil,
                        capturedMilliseconds: nil,
                        receivedMilliseconds: received))
                if let format = CMSampleBufferGetFormatDescription(sample) {
                    let dimensions = CMVideoFormatDescriptionGetDimensions(format)
                    report?.actual["buffer_width"] = String(dimensions.width)
                    report?.actual["buffer_height"] = String(dimensions.height)
                }
                return (index, mapping, report!.sourceID)
            }
            guard let (index, mapping, sourceID) = context else { return }
            var event = CameraCaptureEvent(
                index: index, kind: dropped ? .dropped : .delivered,
                presentationTime: pts, hostPresentationTime: nil,
                capturedMilliseconds: nil,
                receivedMilliseconds: received)
            do {
                guard let clock = preview!.session.synchronizationClock else {
                    throw CameraError.invalidTimestamp
                }
                let host = CMSyncConvertTime(
                    pts, from: clock, to: CMClockGetHostTimeClock())
                let due = try mapping.map(host)
                event = CameraCaptureEvent(
                    index: index, kind: event.kind,
                    presentationTime: pts, hostPresentationTime: host,
                    capturedMilliseconds: due,
                    receivedMilliseconds: received)
                if let device {
                    event.observation = [
                        "observed_ms": received, "iso": Double(device.iso),
                        "exposure_seconds": device.exposureDuration.seconds,
                        "lens_position": Double(device.lensPosition),
                        "adjusting_focus": device.isAdjustingFocus ? 1 : 0,
                        "adjusting_exposure": device.isAdjustingExposure ? 1 : 0,
                    ]
                    .filter { $0.value.isFinite }
                }
                if dropped {
                    let reason =
                        CMGetAttachment(
                            sample,
                            key: kCMSampleBufferAttachmentKey_DroppedFrameReason,
                            attachmentModeOut: nil) as? String ?? "unknown"
                    event.reason = reason
                    event.knownLostFrames =
                        reason
                            == (kCMSampleBufferDroppedFrameReason_FrameWasLate as String)
                            || reason
                                == (kCMSampleBufferDroppedFrameReason_OutOfBuffers
                                    as String)
                        ? 1 : nil
                } else {
                    guard let pixels = CMSampleBufferGetImageBuffer(sample) else {
                        throw CameraError.configuration("Camera sample has no pixels.")
                    }
                    let frame = try VideoFrameConverter.convert(
                        VideoFrame(
                            sourceID: sourceID, sourceFrameIndex: index,
                            presentationTime: pts,
                            pixelBuffer: pixels, readyMilliseconds: received))
                    // Stop wins even if CPU conversion was already active.
                    guard lock.withLock({ token == request && accepting }) else {
                        throw CancellationError()
                    }
                    let delivered = InferenceClock.nowMilliseconds()
                    try deliver(
                        PacedVideoFrame(
                            frame: frame, dueMilliseconds: due,
                            deliveredMilliseconds: delivered))
                    event.admitted = true
                    event.deliveredMilliseconds = delivered
                }
            } catch {
                event.cancelled =
                    error is CancellationError
                    || !lock.withLock { accepting && token == request }
                event.failure = VideoInferenceFailure(
                    event.cancelled ? CancellationError() : error)
                if !event.cancelled { fail(error, request: request) }
            }
            lock.withLock {
                if token == request, report!.events.indices.contains(index) {
                    report!.events[index] = event
                }
            }
        }

        /// Preserves genuine hardware/clock errors; expected stop notifications are ignored.
        private func fail(_ error: any Error, request: UUID) {
            let shouldStop = lock.withLock { () -> Bool in
                guard token == request, active, !closing else { return false }
                closing = true
                accepting = false
                report?.issue = VideoInferenceFailure(error)
                report?.endedMilliseconds = InferenceClock.nowMilliseconds()
                return true
            }
            if shouldStop { sessionQueue.async { [self] in finish() } }
        }

        /// Drains the sample queue before publishing the final numerical snapshot.
        private func finish() {
            guard let continuation else { return }
            self.continuation = nil
            let session = preview!.session
            session.stopRunning()
            for output in session.outputs {
                (output as? AVCaptureVideoDataOutput)?.setSampleBufferDelegate(
                    nil, queue: nil)
            }
            frameQueue.sync {}
            observers.forEach { NotificationCenter.default.removeObserver($0) }
            observers = []
            delegate = nil
            let failure = lock.withLock { () -> VideoInferenceFailure? in
                active = false
                accepting = false
                report?.isRunning = false
                if report?.endedMilliseconds == nil {
                    report?.endedMilliseconds = InferenceClock.nowMilliseconds()
                }
                return report?.issue
            }
            if let failure {
                continuation.resume(throwing: CameraError.interrupted(failure.message))
            } else {
                continuation.resume()
            }
        }
    }

    /// Carries a generation token so delayed callbacks cannot join a newer capture run.
    private nonisolated final class CameraSampleDelegate: NSObject,
        AVCaptureVideoDataOutputSampleBufferDelegate
    {
        weak var owner: CameraCaptureSource?
        let request: UUID
        let deliver: @Sendable (PacedVideoFrame) throws -> Void

        /// Binds this delegate to one serial callback generation.
        init(
            owner: CameraCaptureSource, request: UUID,
            deliver: @escaping @Sendable (PacedVideoFrame) throws -> Void
        ) {
            self.owner = owner
            self.request = request
            self.deliver = deliver
        }
        /// Offers only a converted immutable image to the bounded numerical scheduler.
        func captureOutput(
            _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
            from connection: AVCaptureConnection
        ) {
            owner?.receive(
                sampleBuffer, dropped: false, request: request, deliver: deliver)
        }
        /// Records camera-system loss independently of detector queue replacement.
        func captureOutput(
            _ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer,
            from connection: AVCaptureConnection
        ) {
            owner?.receive(
                sampleBuffer, dropped: true, request: request, deliver: deliver)
        }
    }
#else
    /// Allows deterministic host tests while real capture remains an iPhone feature.
    nonisolated final class CameraCaptureSource: CameraCapturing, Sendable {
        var authorization: CameraAuthorization { .restricted }
        var preview: CameraPreviewHandle? { nil }
        var snapshot: CameraCaptureReport? { nil }
        /// Reports unsupported host capture without prompting for a camera.
        func authorize() async -> Bool { false }
        /// Retains hardware-independent test builds without pretending to capture.
        func run(
            configuration: CameraConfiguration, sourceID: String,
            started: @escaping @Sendable (Double) -> Void,
            deliver: @escaping @Sendable (PacedVideoFrame) throws -> Void
        ) async throws {
            throw CameraError.unavailable
        }
        /// Leaves an unsupported host capture already stopped.
        func stop() {}
    }
#endif
