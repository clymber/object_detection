import SwiftUI

#if os(iOS)
    import AVFoundation
    import UIKit
#endif

/// Presents frozen camera controls, independent live preview and exact detection frames.
struct CameraView: View {
    @Bindable var session: CameraSession
    @Binding var configuration: CameraConfiguration
    let otherBusy: Bool
    let verifySaved: () -> Void
    let showFullScreen: () -> Void

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
                Toggle(
                    "Record annotated video",
                    isOn: $configuration.recordAnnotatedVideo)
                    .accessibilityIdentifier("recordAnnotatedCameraVideo")
                durationControls
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
                Section {
                    ProcessedCameraFrame(
                        session: session, frame: frame, presentation: .inline)
                    Text("Capture callback \(frame.result.metadata.sourceFrameIndex)")
                        .font(.caption).monospacedDigit()
                    if frame.result.detections.isEmpty {
                        Text("No detections in this frame.").font(.caption)
                    }
                } header: {
                    HStack {
                        Text("Processed camera frame")
                        Spacer()
                        Button(action: showFullScreen) {
                            Image(systemName: "arrow.up.left.and.arrow.down.right")
                                .frame(width: 44, height: 44)
                        }
                        .buttonStyle(.borderless)
                        .accessibilityLabel("Show processed camera frame full screen")
                        .accessibilityIdentifier("expandCameraPreview")
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
                LabeledContent(
                    "Elapsed capture time",
                    value: number(session.elapsedSeconds, unit: "s"))
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
            if session.configurationForRun?.recordAnnotatedVideo == true {
                Section {
                    if let recording = session.latestReport?.recording {
                        LabeledContent(
                            "Recorded frames", value: String(recording.encodedFrames))
                        LabeledContent(
                            "Skipped recording frames",
                            value: String(recording.skippedFrames))
                        if let url = recording.url {
                            ShareLink(
                                "Save or share MP4", item: url,
                                preview: SharePreview("Annotated camera detection"))
                            .accessibilityIdentifier("shareCameraRecording")
                        } else {
                            Text(recording.issue?.message
                                 ?? "No processed frames were recorded.")
                                .font(.footnote)
                        }
                    } else {
                        Text(session.status == .stopping ? "Finishing MP4…"
                             : session.isCapturing ? "Recording processed frames…"
                             : "No processed frames were recorded.")
                            .font(.footnote)
                    }
                } header: {
                    Text("Annotated video")
                } footer: {
                    Text(
                        "Records boxes, classes and confidence without audio. "
                        + "Recording adds processing load. "
                        + "Save the MP4 before starting another run.")
                }
            }
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

    /// Applies the same configurable duration to recorded and unrecorded camera runs.
    @ViewBuilder
    private var durationControls: some View {
        Toggle("Limit run duration", isOn: $configuration.durationLimitEnabled)
            .accessibilityIdentifier("limitCameraRunDuration")
        if configuration.durationLimitEnabled {
            LabeledContent("Duration (seconds)") {
                TextField("Seconds", value: $configuration.durationLimitSeconds,
                          format: .number)
                    .multilineTextAlignment(.trailing)
                    .accessibilityIdentifier("cameraRunDurationSeconds")
                    #if os(iOS)
                        .keyboardType(.decimalPad)
                    #endif
            }
            Text("Starts when camera capture begins. Applies with recording on or off.")
                .font(.footnote).foregroundStyle(.secondary)
        }
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

/// Renders the same frame/result pair and draw identity in either presentation.
private struct ProcessedCameraFrame: View {
    @Bindable var session: CameraSession
    let frame: VideoFramePreview
    let presentation: FramePreview.Presentation

    var body: some View {
        let generation = session.previewGeneration
        let token = frame.drawToken
        return FramePreview(
            image: frame.image, detections: frame.result.detections,
            imageTitle: "Processed camera frame",
            didDraw: { [weak session] time in
                session?.recordDraw(token, generation: generation, at: time)
            }, presentation: presentation)
    }
}

#if os(iOS)
    /// Expands processed frames without changing the capture run or its frozen settings.
    struct CameraFullScreenView: View {
        @Bindable var session: CameraSession
        let configuration: CameraConfiguration
        let model: DetectorModel
        let otherBusy: Bool
        @Environment(\.dismiss) private var dismiss
        @Environment(\.scenePhase) private var scenePhase

        var body: some View {
            Group {
                if let frame = session.previewFrame {
                    ProcessedCameraFrame(
                        session: session, frame: frame, presentation: .fullScreen)
                } else {
                    ContentUnavailableView(
                        "Waiting for a processed frame", systemImage: "camera")
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.black.ignoresSafeArea())
            .safeAreaInset(edge: .top, spacing: 0) {
                HStack {
                    VStack(alignment: .leading) {
                        Text("Processed camera frame").font(.headline)
                        Text(session.status.rawValue)
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    .lineLimit(1)
                    Spacer()
                    Button("Close", systemImage: "xmark") { dismiss() }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .accessibilityIdentifier("closeCameraFullScreen")
                }
                .padding().background(.bar)
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                CameraControlBar(
                    session: session, configuration: configuration,
                    model: model, otherBusy: otherBusy)
            }
            .preferredColorScheme(.dark)
            .onChange(of: scenePhase) { _, phase in
                session.setActivity(
                    active: phase == .active, background: phase == .background)
            }
        }
    }
#endif

/// Keeps one run action reachable while the detection list changes height.
struct CameraControlBar: View {
    @Bindable var session: CameraSession
    let configuration: CameraConfiguration
    let model: DetectorModel
    let otherBusy: Bool

    var body: some View {
        HStack {
            Button {
                if session.isCapturing {
                    session.stop()
                } else {
                    session.start(configuration: configuration, model: model)
                }
            } label: {
                Label(
                    session.status == .stopping ? "Stopping…"
                        : session.isCapturing ? "Stop" : "Start",
                    systemImage: session.isCapturing ? "stop.fill" : "play.fill"
                )
                .frame(maxWidth: .infinity)
            }
            .tint(session.isCapturing ? .red : .accentColor)
            .disabled(
                session.isCapturing ? session.status == .stopping
                    : session.isBusy || otherBusy)
            .accessibilityIdentifier(
                session.isCapturing ? "stopCameraDetection" : "startCameraDetection")
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("cameraControlBar")
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
