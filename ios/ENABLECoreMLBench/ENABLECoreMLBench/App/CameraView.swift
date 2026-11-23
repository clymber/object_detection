import SwiftUI

#if os(iOS)
    import AVFoundation
    import UIKit
#endif

/// Presents frozen camera controls, independent live preview and exact detection frames.
struct CameraView: View {
    @Bindable var session: CameraSession
    @Binding var configuration: CameraConfiguration
    let model: DetectorModel
    let otherBusy: Bool
    let verifySaved: () -> Void

    var body: some View {
        Group {
            Section("Camera settings") {
                Picker("Lens", selection: $configuration.lens) {
                    ForEach(CameraConfiguration.Lens.allCases) {
                        Text($0.rawValue).tag($0)
                    }
                }
                Picker("Resolution", selection: $configuration.resolution) {
                    ForEach(CameraConfiguration.Resolution.allCases) {
                        Text($0.rawValue).tag($0)
                    }
                }
                Picker("Target FPS", selection: $configuration.framesPerSecond) {
                    Text("15").tag(15)
                    Text("30").tag(30)
                }
                Picker("Orientation", selection: $configuration.orientation) {
                    ForEach(CameraConfiguration.Orientation.allCases) {
                        Text($0.title).tag($0)
                    }
                }
                Toggle("Mirror capture and preview", isOn: $configuration.mirrored)
                Toggle("Show live camera preview", isOn: $configuration.livePreview)
                Toggle(
                    "Show processed frames with boxes",
                    isOn: $configuration.detectionPreview)
                TextField(
                    "Lighting, distance and basketball activity",
                    text: $configuration.sceneNotes,
                    axis: .vertical)
            }
            .disabled(session.isBusy || otherBusy)
            Section {
                if configuration.livePreview {
                    #if os(iOS)
                        if let preview = session.preview,
                            session.status == .running || session.status == .preparing
                        {
                            CameraLivePreview(
                                handle: preview,
                                configuration: session.configurationForRun
                                    ?? configuration
                            )
                            .frame(height: 280).accessibilityIdentifier(
                                "liveCameraPreview")
                        } else {
                            ContentUnavailableView(
                                "Camera stopped", systemImage: "camera")
                        }
                    #else
                        ContentUnavailableView(
                            "Use a physical iPhone", systemImage: "camera")
                    #endif
                } else {
                    Text("Live preview is off.").foregroundStyle(.secondary)
                }
            } header: {
                Text("Live camera")
            } footer: {
                Text(
                    "Capture orientation is fixed for each run. Leave both previews off for baseline timing."
                )
            }
            if let frame = session.previewFrame,
                session.configurationForRun?.detectionPreview == true
            {
                Section("Processed camera frame") {
                    detectionFrame(frame)
                    Text("Capture callback \(frame.result.metadata.sourceFrameIndex)")
                        .font(.caption).monospacedDigit()
                    if frame.result.detections.isEmpty {
                        Text("No detections in this frame.").font(.caption)
                    }
                }
            } else if session.isCapturing
                && session.configurationForRun?.detectionPreview == true
            {
                Section("Processed camera frame") {
                    Text("Waiting for a fresh detection frame (≤ 200 ms).")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            Section {
                LabeledContent("Status", value: session.status.rawValue)
                    .accessibilityIdentifier("cameraStatus")
                Text(session.message).font(.footnote)
                    .foregroundStyle(session.status == .error ? Color.red : .secondary)
                    .accessibilityIdentifier("cameraMessage")
                HStack {
                    Button("Start", systemImage: "play.fill") {
                        session.start(configuration: configuration, model: model)
                    }
                    .buttonStyle(.borderedProminent).disabled(
                        session.isBusy || otherBusy
                    )
                    .accessibilityIdentifier("startCameraDetection")
                    Spacer()
                    Button("Stop", systemImage: "stop.fill") { session.stop() }
                        .buttonStyle(.bordered).disabled(
                            !session.isCapturing || session.status == .stopping
                        )
                        .accessibilityIdentifier("stopCameraDetection")
                }
                if session.authorization == .denied
                    || session.authorization == .restricted
                {
                    #if os(iOS)
                        Link(
                            "Open camera settings",
                            destination: URL(
                                string: UIApplication.openSettingsURLString)!)
                    #endif
                }
            } header: {
                Text("Camera detection")
            } footer: {
                Text(
                    "One active inference and one pending frame. Only results no older than 200 ms publish. "
                        + "Boxes belong to the exact processed frame shown separately from the live preview."
                )
            }
            Section("Camera counts and rates") {
                LabeledContent(
                    "Delivered capture callbacks",
                    value: String(session.deliveredFrames))
                LabeledContent(
                    "Capture drop callbacks", value: String(session.captureDrops))
                LabeledContent(
                    "Known lost camera frames",
                    value: session.knownLostFrames.map(String.init) ?? "Unknown")
                LabeledContent(
                    "Capture/conversion failures", value: String(session.sourceFailures)
                )
                LabeledContent(
                    "Queue inputs", value: String(session.counts.inputFrames))
                LabeledContent(
                    "Processed", value: String(session.counts.processedFrames))
                LabeledContent(
                    "Queue skips", value: String(session.counts.skippedFrames))
                LabeledContent(
                    "Unfinished", value: String(session.counts.unfinishedFrames))
                LabeledContent(
                    "Fresh receipts (total)",
                    value: String(session.counts.publishedResults))
                LabeledContent(
                    "Late results", value: String(session.counts.lateResults))
                LabeledContent(
                    "Capture FPS (delivered)", value: rate(session.captureFPS))
                LabeledContent(
                    "Input FPS (queue)",
                    value: rate(session.metrics?.inputFramesPerSecond))
                LabeledContent(
                    "Completed FPS",
                    value: rate(session.metrics?.completedFramesPerSecond))
                LabeledContent(
                    "Fresh-result FPS",
                    value: rate(session.metrics?.freshResultsPerSecond))
                LabeledContent("Detection preview FPS", value: rate(session.displayFPS))
                LabeledContent(
                    "Model mean",
                    value: number(
                        session.metrics?.modelLatency.meanMilliseconds, unit: "ms"))
                LabeledContent(
                    "Frame p50 / p95", value: percentiles(session.metrics?.frameLatency)
                )
                LabeledContent(
                    "Published age p50 / p95",
                    value: percentiles(session.metrics?.publishedAge))
            }
            .monospacedDigit()
            Section("Camera export") {
                Button("Prepare export", systemImage: "doc.badge.gearshape") {
                    session.exportResults()
                }
                .disabled(session.isBusy || session.latestReport == nil)
                if let message = session.export.message {
                    Text(message).font(.footnote)
                }
                if let files = session.export.files {
                    ShareLink("Save or share results", items: files.urls) {
                        SharePreview($0.lastPathComponent)
                    }
                    Button(
                        "Verify saved results", systemImage: "checkmark.shield",
                        action: verifySaved
                    )
                    .disabled(session.isBusy)
                }
                if session.export.verificationStatus != .idle {
                    Text(session.export.verificationStatus.rawValue).font(.headline)
                        .foregroundStyle(
                            session.export.verificationStatus == .verified
                                ? Color.green
                                : session.export.verificationStatus == .failed
                                    ? .red : .primary)
                    if let detail = session.export.verificationMessage {
                        Text(detail).font(.footnote).textSelection(.enabled)
                    }
                }
            }
        }
    }

    /// Uses the same source-pixel box geometry and distinct-draw receipts as video.
    private func detectionFrame(_ frame: VideoFramePreview) -> some View {
        let generation = session.previewGeneration
        let token = frame.drawToken
        return FramePreview(
            image: frame.image, detections: frame.result.detections,
            imageTitle: "Processed camera frame",
            didDraw: { [weak session] time in
                session?.recordDraw(token, generation: generation, at: time)
            })
    }

    /// Leaves unavailable rates visibly distinct from measured zero.
    private func rate(_ value: Double?) -> String { number(value, unit: "FPS") }
    /// Formats finite scalar measurements without changing retained export precision.
    private func number(_ value: Double?, unit: String) -> String {
        guard let value, value.isFinite else { return "—" }
        return value.formatted(.number.precision(.fractionLength(2))) + " " + unit
    }
    /// Shows latency percentiles only after worker-side finalization.
    private func percentiles(_ stats: VideoLatencyStatistics?) -> String {
        guard let p50 = stats?.p50Milliseconds, let p95 = stats?.p95Milliseconds else {
            return "—"
        }
        return number(p50, unit: "") + "/ " + number(p95, unit: "ms")
    }
}

#if os(iOS)
    /// Shows AVFoundation's live preview independently of inference or SwiftUI image copies.
    private struct CameraLivePreview: UIViewRepresentable {
        let handle: CameraPreviewHandle
        let configuration: CameraConfiguration

        /// Attaches only the preview layer on MainActor; hardware setup stays on its worker.
        func makeUIView(context: Context) -> CameraPreviewUIView {
            let view = CameraPreviewUIView()
            view.previewLayer.session = handle.session
            view.previewLayer.videoGravity = .resizeAspect
            return view
        }
        /// Applies the same fixed presentation rotation and mirroring as the capture buffers.
        func updateUIView(_ view: CameraPreviewUIView, context: Context) {
            guard let connection = view.previewLayer.connection else { return }
            let angle = CGFloat(configuration.orientation.rawValue)
            if connection.isVideoRotationAngleSupported(angle),
                connection.videoRotationAngle != angle
            {
                connection.videoRotationAngle = angle
            }
            connection.automaticallyAdjustsVideoMirroring = false
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = configuration.mirrored
            }
        }
    }

    /// Uses a layer-backed view so layout changes never run capture configuration.
    private final class CameraPreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }
#endif
