import AVFoundation
import AVKit
import CoreMedia
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers
#if os(iOS)
import UIKit
#endif

/// Selects frozen inputs, personal photos or videos without mixing their results.
struct ContentView: View {
    /// Freezes file routing before presentation, including after the picker dismisses.
    private enum FileImportPurpose: Equatable {
        case photo, video, videoAnnotations, savedVideoResults

        var contentTypes: [UTType] {
            switch self {
            case .photo: [.image]
            case .video: [.movie]
            case .videoAnnotations: [.json]
            case .savedVideoResults: [.folder]
            }
        }
    }

    @Environment(\.scenePhase) private var scenePhase
    @State private var session = StillImageSession()
    @State private var batch = StillImageBatchSession()
    @State private var media = MediaImportSession()
    @State private var videoSession: VideoSession
    @State private var cameraSession: CameraSession
    @State private var cameraConfiguration = CameraConfiguration()
    @State private var showingCameraFullScreen = false
    @State private var inputSource = MediaInputSource.testImages
    @State private var selectedModel = DetectorModel.rfdetrSmall
    @State private var selectedSample = BundledSample.testPositive6988
    @State private var selectedTestVideo = BundledVideo.nba
    @State private var photoSelection: PhotosPickerItem?
    @State private var videoSelection: PhotosPickerItem?
    @State private var showingFiles = false
    @State private var fileImportPurpose = FileImportPurpose.photo
    @State private var player: AVPlayer?

    private var isBusy: Bool {
        session.isBusy || batch.isBusy || media.isBusy || videoSession.isBusy || cameraSession.isBusy
    }

    /// Shares one loaded detector between mutually exclusive video and camera runs.
    init() {
        let engine = DetectorEngine()
        _videoSession = State(initialValue: VideoSession(engine: engine))
        _cameraSession = State(initialValue: CameraSession(worker: CameraRunner(engine: engine)))
    }
    private var previewTitle: String {
        inputSource == .photo
            ? (media.photo?.displayName ?? "Imported photo") : selectedSample.title
    }

    var body: some View {
        NavigationStack {
            Form {
                inputSection
                modelSection
                if inputSource == .camera {
                    CameraView(session: cameraSession, configuration: $cameraConfiguration,
                        otherBusy: session.isBusy || batch.isBusy || media.isBusy || videoSession.isBusy,
                        verifySaved: {
                            fileImportPurpose = .savedVideoResults
                            showingFiles = true
                        }, showFullScreen: { showingCameraFullScreen = true })
                } else if inputSource.isVideo {
                    videoSection
                    videoDetectionSection
                    VideoAccuracyView(
                        session: videoSession.accuracy,
                        disabled: isBusy || media.video == nil,
                        chooseAnnotations: {
                            fileImportPurpose = .videoAnnotations
                            showingFiles = true
                        }, removeAnnotations: { videoSession.removeAnnotations() })
                    if videoSession.mode == .offline {
                        VideoMetricsView(
                            metrics: videoSession.offline.metrics,
                            completedFrames: videoSession.offline.completedFrames,
                            failedFrames: videoSession.offline.failedFrames,
                            unfinishedFrames: videoSession.offline.unfinishedFrames,
                            previewEnabled: videoSession.offline.previewEnabledForRun,
                            previewFrames: videoSession.offline.previewStatistics.drawnFrames,
                            previewFPS: videoSession.offline.previewStatistics.framesPerSecond)
                    } else {
                        PacedVideoMetricsView(session: videoSession.paced,
                            accuracy: videoSession.accuracy.result?.report)
                    }
                } else {
                    Section {
                        FramePreview(
                            image: session.image, detections: session.detections,
                            imageTitle: previewTitle)
                    } header: {
                        Text("Preview")
                    } footer: {
                        Text("Boxes show detections at confidence ≥ 25%.")
                    }
                    detectionSection
                    if inputSource == .testImages { batchSection }
                    BenchmarkMetricsView(
                        latencyMilliseconds: session.latencyMilliseconds,
                        processedImages: session.processedImages,
                        detectionCount: session.detections.count)
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                if inputSource == .camera {
                    CameraControlBar(
                        session: cameraSession, configuration: cameraConfiguration,
                        model: selectedModel,
                        otherBusy: session.isBusy || batch.isBusy || media.isBusy
                            || videoSession.isBusy)
                }
            }
            .navigationTitle("ENABLE Core ML Bench")
            .toolbarTitleDisplayMode(.inline)
        }
        .fullScreenCover(isPresented: $showingCameraFullScreen) {
            CameraFullScreenView(
                session: cameraSession,
                configuration: cameraSession.configurationForRun ?? cameraConfiguration,
                model: selectedModel,
                otherBusy: session.isBusy || batch.isBusy || media.isBusy
                    || videoSession.isBusy)
        }
        .onAppear {
            cameraSession.setActivity(active: scenePhase == .active, background: scenePhase == .background)
            updateVideoIdleTimer()
        }
        .task { loadSelectedImage() }
        .fileImporter(
            isPresented: $showingFiles,
            allowedContentTypes: fileImportPurpose.contentTypes
        ) { [purpose = fileImportPurpose] result in
            handleFileImport(result, purpose: purpose)
        }
        .onChange(of: inputSource) { _, _ in
            stopPlayback()
            videoSession.select(nil)
            cameraSession.reset()
            session.clear()
            batch.reset()
            media.reset()
            photoSelection = nil
            videoSelection = nil
            loadSelectedImage()
            if inputSource == .testVideos { media.selectTestVideo(selectedTestVideo) }
        }
        .onChange(of: selectedSample) { _, sample in
            if inputSource == .testImages { session.load(sample) }
        }
        .onChange(of: selectedTestVideo) { _, sample in
            guard inputSource == .testVideos else { return }
            prepareForImport()
            media.selectTestVideo(sample)
        }
        .onChange(of: selectedModel) { _, _ in
            loadSelectedImage()
            batch.reset()
            videoSession.select(
                inputSource.isVideo ? media.video?.file.url : nil,
                sourceID: media.video?.sourceID)
            cameraSession.reset()
        }
        .onChange(of: videoSession.isPreviewEnabled) { _, enabled in
            if enabled { player?.pause() }
        }
        .onChange(of: videoSession.isBusy) { _, _ in updateVideoIdleTimer() }
        .onChange(of: photoSelection) { _, item in
            guard let item else { return }
            photoSelection = nil
            prepareForImport()
            media.select(item, as: .photo)
        }
        .onChange(of: videoSelection) { _, item in
            guard let item else { return }
            videoSelection = nil
            prepareForImport()
            media.select(item, as: .video)
        }
        .onChange(of: media.photo) { _, file in
            if inputSource == .photo, let file { session.loadPhoto(file.url) }
        }
        .onChange(of: media.video?.file.url) { _, url in
            stopPlayback()
            videoSession.select(
                inputSource.isVideo ? url : nil, sourceID: media.video?.sourceID)
            if inputSource.isVideo, let url { player = AVPlayer(url: url) }
        }
        .onDisappear {
            #if os(iOS)
            UIApplication.shared.isIdleTimerDisabled = false
            #endif
            session.stop()
            batch.stop()
            videoSession.suspendDetection()
            // A full-screen preview covers this view without ending capture.
            if !showingCameraFullScreen {
                cameraSession.setActivity(
                    active: false, background: scenePhase == .background)
            }
            media.cancel()
            player?.pause()
        }
        .onChange(of: scenePhase) { _, phase in
            cameraSession.setActivity(active: phase == .active, background: phase == .background)
            if phase != .active {
                session.stop()
                batch.stop()
                videoSession.suspendDetection()
                media.cancel()
                player?.pause()
            } else if !inputSource.isVideo && session.image == nil && !isBusy {
                loadSelectedImage()
            }
            updateVideoIdleTimer()
        }
    }

    /// Keeps long foreground video runs awake without enabling background processing.
    private func updateVideoIdleTimer() {
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = scenePhase == .active
            && (videoSession.isDetecting || videoSession.isPreparingBenchmark)
        #endif
    }

    private var inputSection: some View {
        Section {
            Picker("Source", selection: $inputSource) {
                ForEach(MediaInputSource.allCases) { source in
                    Text(source.rawValue).tag(source)
                }
            }
            .disabled(isBusy)
            .accessibilityIdentifier("inputSourcePicker")
            switch inputSource {
            case .testImages:
                Picker("Test image", selection: $selectedSample) {
                    ForEach(BundledSample.allCases) { sample in
                        Text(sample.title).tag(sample)
                    }
                }
                .pickerStyle(.menu)
                .disabled(isBusy)
                .accessibilityIdentifier("samplePicker")
                if session.image == nil && !isBusy {
                    Button("Load image") { session.load(selectedSample) }
                }
            case .testVideos:
                Picker("Test video", selection: $selectedTestVideo) {
                    ForEach(BundledVideo.allCases) { sample in
                        Text(sample.title).tag(sample)
                    }
                }
                .pickerStyle(.menu)
                .disabled(isBusy)
                .accessibilityIdentifier("testVideoPicker")
                if media.video == nil && !isBusy {
                    Button("Load video") { media.selectTestVideo(selectedTestVideo) }
                }
            case .photo:
                PhotosPicker(
                    selection: $photoSelection, matching: .images,
                    preferredItemEncoding: .current
                ) {
                    Label("Choose from Photos", systemImage: "photo.on.rectangle")
                }
                .disabled(isBusy)
                .accessibilityIdentifier("choosePhoto")
                filesButton
                if let file = media.photo {
                    Text(file.displayName).foregroundStyle(.secondary)
                }
            case .video:
                PhotosPicker(
                    selection: $videoSelection, matching: .videos,
                    preferredItemEncoding: .current
                ) {
                    Label("Choose from Photos", systemImage: "video")
                }
                .disabled(isBusy)
                .accessibilityIdentifier("chooseVideo")
                filesButton
            case .camera:
                Text("Live camera settings are frozen when Start is tapped.").font(.footnote)
            }
            if media.isBusy {
                ProgressView("Importing media…")
                Button("Cancel import", role: .cancel) { media.cancel() }
            }
            if let error = media.errorMessage {
                Text(error).font(.footnote).foregroundStyle(.red)
                    .accessibilityIdentifier("mediaImportError")
            }
        } header: {
            Text("Input")
        } footer: {
            switch inputSource {
            case .testImages:
                Text(
                    "\(BundledSample.allCases.count) frozen test images for parity comparison."
                )
            case .testVideos:
                Text("Bundled development clips. Choose a detector, then tap Start.")
            case .photo:
                Text("Detect basketballs in a photo from Photos or Files.")
            case .video:
                Text("Choose a video from Photos or Files, then tap Start.")
            case .camera:
                Text("Detect basketballs using the camera. No audio is captured.")
            }
        }
    }

    private var filesButton: some View {
        Button("Choose from Files", systemImage: "folder") {
            fileImportPurpose = inputSource.isVideo ? .video : .photo
            showingFiles = true
        }
            .disabled(isBusy)
            .accessibilityIdentifier("chooseMediaFile")
    }

    private var modelSection: some View {
        Section("Model") {
            Picker("Detector", selection: $selectedModel) {
                ForEach(DetectorModel.allCases) { model in
                    Text(model.displayName).tag(model)
                }
            }
            .pickerStyle(.menu)
            .disabled(isBusy)
            .accessibilityIdentifier("modelPicker")
        }
    }

    private var videoSection: some View {
        Section(
            videoSession.isPreviewEnabled ? "Detection preview" : "Video playback"
        ) {
            if let video = media.video {
                if videoSession.isPreviewEnabled {
                    if let frame = videoSession.previewFrame {
                        processedVideoPreview(frame, title: video.file.displayName)
                    } else {
                        ContentUnavailableView(
                            "Start detection to preview frames",
                            systemImage: "viewfinder")
                            .frame(height: 280)
                    }
                } else if let player {
                    VideoPlayer(player: player).frame(height: 280)
                        .disabled(videoSession.isBusy)
                        .accessibilityIdentifier("videoPreview")
                }
                Text(video.file.displayName)
                LabeledContent(
                    "Duration",
                    value:
                        "\(video.durationSeconds.formatted(.number.precision(.fractionLength(1)))) s"
                )
            } else {
                ContentUnavailableView("Choose a video", systemImage: "video")
                    .frame(height: 280)
            }
        }
    }

    /// Captures only a lightweight token in deferred draw callbacks, not frame pixels.
    private func processedVideoPreview(
        _ frame: VideoFramePreview, title: String
    ) -> some View {
        let token = frame.drawToken
        let generation = videoSession.previewGeneration
        let metadata = frame.result.metadata
        return VStack(alignment: .leading, spacing: 8) {
            FramePreview(
                image: frame.image, detections: frame.result.detections,
                imageTitle: "\(title), frame \(metadata.sourceFrameIndex)",
                didDraw: { [weak receiver = videoSession] milliseconds in
                    receiver?.recordPreviewDraw(
                        token, generation: generation, at: milliseconds)
                })
            Text(
                "Frame \(metadata.sourceFrameIndex) · PTS "
                    + metadata.presentationTime.seconds.formatted(
                        .number.precision(.fractionLength(3))) + " s")
                .font(.caption).monospacedDigit()
                .accessibilityIdentifier("processedVideoFrameIdentity")
            if let failure = frame.result.failure {
                Text("Frame inference failed: \(failure.message)")
                    .font(.caption).foregroundStyle(.red)
            } else if frame.result.detections.isEmpty {
                Text("No detections in this frame.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .accessibilityIdentifier("processedVideoPreview")
    }

    private var videoDetectionSection: some View {
        Section {
            Picker("Processing", selection: Binding(
                get: { videoSession.mode }, set: { videoSession.setMode($0) })) {
                ForEach(VideoProcessingMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .disabled(isBusy)
            .accessibilityIdentifier("videoProcessingModePicker")
            Toggle("Show detection preview", isOn: Binding(
                get: { videoSession.isPreviewEnabled },
                set: { videoSession.setPreviewEnabled($0) }))
                .disabled(videoSession.isBusy)
                .accessibilityIdentifier("showVideoDetectionPreview")
            LabeledContent("Status", value: videoSession.status.rawValue)
                .foregroundStyle(videoSession.status == .error ? Color.red : .primary)
                .accessibilityIdentifier("videoDetectionStatus")
            Text(videoSession.message).font(.footnote).foregroundStyle(.secondary)
                .accessibilityIdentifier("videoDetectionMessage")
            if videoSession.isDetecting || videoSession.isPreparingBenchmark {
                ProgressView {
                    Text(videoSession.isPreparingBenchmark
                        ? "Preparing benchmark…" : videoSession.progressText)
                }
                    .accessibilityIdentifier("videoDetectionProgress")
            } else if videoSession.hasReport {
                Text(videoSession.progressText)
                    .accessibilityIdentifier("videoDetectionProgress")
            }
            HStack {
                Button("Start", systemImage: "play.fill") {
                    player?.pause()
                    videoSession.start(model: selectedModel)
                }
                .buttonStyle(.borderedProminent)
                .disabled(
                    !videoSession.canStart || session.isBusy
                        || batch.isBusy || media.isBusy
                )
                .accessibilityIdentifier("startVideoDetection")
                Spacer()
                Button("Stop", systemImage: "stop.fill") { videoSession.stop() }
                    .buttonStyle(.bordered)
                    .disabled(!videoSession.isBusy || videoSession.status == .stopping)
                    .accessibilityIdentifier("stopVideoDetection")
            }
            if videoSession.hasReport {
                Button("Prepare export", systemImage: "doc.badge.gearshape") {
                    videoSession.exportResults()
                }
                .disabled(isBusy)
                .accessibilityIdentifier("prepareVideoExport")
            }
            if let message = videoSession.export.message {
                Text(message).font(.footnote).foregroundStyle(.secondary)
            }
            if let files = videoSession.export.files {
                ShareLink("Save or share results", items: files.urls) { url in
                    SharePreview(url.lastPathComponent)
                }
                .accessibilityIdentifier("shareVideoResults")
                Button("Verify saved results", systemImage: "checkmark.shield") {
                    fileImportPurpose = .savedVideoResults
                    showingFiles = true
                }
                .disabled(isBusy)
                .accessibilityIdentifier("verifySavedVideoResults")
                if videoSession.export.verificationStatus != .idle {
                    VStack(alignment: .leading, spacing: 6) {
                        if videoSession.export.verificationStatus == .checking {
                            ProgressView("Checking saved files…")
                        } else {
                            Label(videoSession.export.verificationStatus.rawValue,
                                  systemImage: verificationSymbol)
                                .font(.headline)
                                .foregroundStyle(verificationColor)
                        }
                        if let detail = videoSession.export.verificationMessage {
                            Text(detail).font(.footnote).textSelection(.enabled)
                        }
                    }
                    .accessibilityIdentifier("savedVideoVerificationStatus")
                }
            }
        } header: {
            Text("Video detection")
        } footer: {
            Text(videoSession.mode == .offline
                ? "Processes every frame from the beginning at the detector’s speed. "
                    + "Set preview before Start; leave it off for baseline benchmarks."
                : "Runs once from the beginning at source cadence, skipping older waiting "
                    + "frames. Results publish only within 200 ms of their source time. "
                    + "Preview shows the exact accepted frame; leave it off for baseline runs.")
        }
    }

    private var verificationSymbol: String {
        switch videoSession.export.verificationStatus {
        case .verified: "checkmark.circle.fill"
        case .failed: "exclamationmark.triangle.fill"
        default: "info.circle"
        }
    }

    private var verificationColor: Color {
        switch videoSession.export.verificationStatus {
        case .verified: .green
        case .failed: .red
        default: .secondary
        }
    }

    private var detectionSection: some View {
        Section("Detection") {
            LabeledContent("Status") {
                HStack {
                    if session.isBusy { ProgressView().controlSize(.small) }
                    Text(session.status.rawValue)
                        .foregroundStyle(
                            session.status == .error ? Color.red : .primary
                        )
                        .accessibilityIdentifier("detectionStatus")
                }
            }
            Text(session.message).font(.footnote).foregroundStyle(.secondary)
                .accessibilityIdentifier("detectionMessage")
            HStack {
                Button("Start", systemImage: "play.fill") {
                    session.start(model: selectedModel)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!session.canStart || batch.isBusy || media.isBusy)
                .accessibilityIdentifier("startDetection")
                Spacer()
                Button("Stop", systemImage: "stop.fill") { session.stop() }
                    .buttonStyle(.bordered).disabled(!session.isBusy)
                    .accessibilityIdentifier("stopDetection")
            }
        }
    }

    private var batchSection: some View {
        Section {
            Button("Run all test images", systemImage: "play.rectangle.on.rectangle") {
                batch.start(model: selectedModel)
            }
            .disabled(isBusy)
            .accessibilityIdentifier("runAllTestImages")
            if batch.isBusy {
                ProgressView(value: Double(batch.completed), total: Double(batch.total))
                {
                    Text("\(batch.completed) of \(batch.total) images")
                }
                Button("Stop batch", role: .cancel) { batch.stop() }
            }
            Text(batch.message).font(.footnote).foregroundStyle(.secondary)
            if let url = batch.exportURL {
                ShareLink("Export results", item: url)
                    .accessibilityIdentifier("exportTestResults")
            }
        } header: {
            Text("Test-set comparison")
        } footer: {
            Text("Export exact boxes and scores to compare on your Mac.")
        }
    }

    /// Reloads only the chosen image source after source/model/lifecycle changes.
    private func loadSelectedImage() {
        switch inputSource {
        case .testImages: session.load(selectedSample)
        case .photo:
            if let file = media.photo {
                session.loadPhoto(file.url)
            } else {
                session.clear(message: "Choose a photo, then tap Start.")
            }
        case .testVideos, .video, .camera: session.clear()
        }
    }

    /// Routes the captured purpose without resetting a video run for saved-file checks.
    private func handleFileImport(_ result: Result<URL, Error>, purpose: FileImportPurpose) {
        if purpose == .videoAnnotations {
            switch result {
            case .success(let url): videoSession.importAnnotations(url)
            case .failure(let error):
                if (error as NSError).code != NSUserCancelledError { media.report(error) }
            }
            return
        }
        if purpose == .savedVideoResults {
            let export = inputSource == .camera ? cameraSession.export : videoSession.export
            switch result {
            case .success(let directory): export.verifySaved(directory)
            case .failure(let error): export.reportVerificationFailure(error)
            }
            return
        }
        switch result {
        case .success(let url):
            prepareForImport()
            media.importFile(url, as: purpose == .video ? .video : .photo)
        case .failure(let error):
            if (error as NSError).code != NSUserCancelledError { media.report(error) }
        }
    }

    /// Removes old detections/playback before a new transfer can publish its input.
    private func prepareForImport() {
        session.clear(message: "Choose a photo, then tap Start.")
        batch.reset()
        videoSession.select(nil)
        stopPlayback()
    }

    /// Releases playback before removing or replacing its app-owned file.
    private func stopPlayback() {
        player?.pause()
        player?.replaceCurrentItem(with: nil)
        player = nil
    }
}

#Preview {
    ContentView()
}
