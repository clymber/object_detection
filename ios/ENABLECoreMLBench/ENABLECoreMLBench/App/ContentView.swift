import AVFoundation
import AVKit
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// Selects frozen inputs, personal photos or videos without mixing their results.
struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var session = StillImageSession()
    @State private var batch = StillImageBatchSession()
    @State private var media = MediaImportSession()
    @State private var videoSession = OfflineVideoSession()
    @State private var inputSource = MediaInputSource.testImages
    @State private var selectedModel = DetectorModel.rfdetrSmall
    @State private var selectedSample = BundledSample.testPositive6988
    @State private var photoSelection: PhotosPickerItem?
    @State private var videoSelection: PhotosPickerItem?
    @State private var showingFiles = false
    @State private var player: AVPlayer?

    private var isBusy: Bool {
        session.isBusy || batch.isBusy || media.isBusy || videoSession.isBusy
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
                if inputSource == .video {
                    videoSection
                    videoDetectionSection
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
            .navigationTitle("ENABLE Core ML Bench")
            .toolbarTitleDisplayMode(.inline)
        }
        .task { loadSelectedImage() }
        .fileImporter(
            isPresented: $showingFiles,
            allowedContentTypes: inputSource == .video ? [.movie] : [.image]
        ) {
            result in
            switch result {
            case .success(let url):
                prepareForImport()
                media.importFile(url, as: inputSource == .video ? .video : .photo)
            case .failure(let error):
                if (error as NSError).code != NSUserCancelledError {
                    media.report(error)
                }
            }
        }
        .onChange(of: inputSource) { _, _ in
            stopPlayback()
            videoSession.select(nil)
            session.clear()
            batch.reset()
            media.reset()
            photoSelection = nil
            videoSelection = nil
            loadSelectedImage()
        }
        .onChange(of: selectedSample) { _, sample in
            if inputSource == .testImages { session.load(sample) }
        }
        .onChange(of: selectedModel) { _, _ in
            loadSelectedImage()
            batch.reset()
            videoSession.select(inputSource == .video ? media.video?.file.url : nil)
        }
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
            videoSession.select(inputSource == .video ? url : nil)
            if inputSource == .video, let url { player = AVPlayer(url: url) }
        }
        .onDisappear {
            session.stop()
            batch.stop()
            videoSession.stop()
            media.cancel()
            player?.pause()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active {
                session.stop()
                batch.stop()
                videoSession.stop()
                media.cancel()
                player?.pause()
            } else if inputSource != .video && session.image == nil && !isBusy {
                loadSelectedImage()
            }
        }
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
            case .photo:
                Text("Detect basketballs in a photo from Photos or Files.")
            case .video:
                Text("Choose a video from Photos or Files, then tap Start.")
            }
        }
    }

    private var filesButton: some View {
        Button("Choose from Files", systemImage: "folder") { showingFiles = true }
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
        Section("Video preview") {
            if let player, let video = media.video {
                VideoPlayer(player: player).frame(height: 280)
                    .disabled(videoSession.isBusy)
                    .accessibilityIdentifier("videoPreview")
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

    private var videoDetectionSection: some View {
        Section {
            LabeledContent("Status", value: videoSession.status.rawValue)
                .foregroundStyle(videoSession.status == .error ? Color.red : .primary)
                .accessibilityIdentifier("videoDetectionStatus")
            Text(videoSession.message).font(.footnote).foregroundStyle(.secondary)
                .accessibilityIdentifier("videoDetectionMessage")
            if videoSession.isBusy {
                ProgressView { Text(videoSession.progressText) }
                    .accessibilityIdentifier("videoDetectionProgress")
            } else if videoSession.latestReport != nil {
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
        } header: {
            Text("Video detection")
        } footer: {
            Text("Processes every frame from the beginning at the detector’s speed.")
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
        case .video: session.clear()
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
