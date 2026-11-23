import SwiftUI

/// Displays source pixels and boxes using the same aspect-fit transform.
struct FramePreview: View {
    enum Presentation { case inline, fullScreen }

    typealias DrawObserver = @MainActor @Sendable (Double) -> Void
    let image: CGImage?
    let detections: [Detection]
    let imageTitle: String
    let didDraw: DrawObserver?
    let presentation: Presentation

    /// Enables draw acknowledgements for video without changing the still-image path.
    init(
        image: CGImage?, detections: [Detection], imageTitle: String,
        didDraw: DrawObserver? = nil, presentation: Presentation = .inline
    ) {
        self.image = image
        self.detections = detections
        self.imageTitle = imageTitle
        self.didDraw = didDraw
        self.presentation = presentation
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            imagePreview
            if presentation == .inline {
                ForEach(detections.indices, id: \.self) { index in
                    let detection = detections[index]
                    HStack {
                        Rectangle().stroke(.red, lineWidth: 2)
                            .frame(width: 12, height: 10)
                        Text(
                            detection.classID == 1
                                ? "Basketball" : "Class \(detection.classID)")
                        Spacer()
                        Text(
                            detection.confidence,
                            format: .percent.precision(.fractionLength(1))
                        )
                        .monospacedDigit()
                    }
                    .font(.caption)
                    .accessibilityElement(children: .combine)
                }
            }
        }
    }

    /// Fits the image and adds box labels only in the full-screen presentation.
    private var imagePreview: some View {
        GeometryReader { geometry in
            if let image {
                let layout = PreviewGeometry(
                    sourceSize: CGSize(width: image.width, height: image.height),
                    viewportSize: geometry.size
                )
                Group {
                    if didDraw != nil || presentation == .fullScreen {
                        Canvas(rendersAsynchronously: false) { context, _ in
                            context.draw(
                                Image(decorative: image, scale: 1, orientation: .up),
                                in: layout.imageRect)
                            for detection in detections {
                                if let box = layout.displayedBox(
                                    detection.boundingBox) {
                                    context.stroke(
                                        Path(box), with: .color(.red), lineWidth: 2)
                                }
                            }
                            if presentation == .fullScreen {
                                drawLabels(in: &context, layout: layout)
                            }
                            if let didDraw {
                                let drawn = InferenceClock.nowMilliseconds()
                                // Update state after this drawing pass completes.
                                Task { @MainActor in didDraw(drawn) }
                            }
                        }
                    } else {
                        stillImage(image, layout: layout)
                    }
                }
                .frame(width: geometry.size.width, height: geometry.size.height)
                .clipped()
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("\(imageTitle), \(detections.count) detections")
                .accessibilityValue(
                    presentation == .fullScreen
                        ? detections.map(detectionLabel).joined(separator: ", ") : "")
            } else {
                ContentUnavailableView("No image loaded", systemImage: "photo")
                    .frame(width: geometry.size.width, height: geometry.size.height)
            }
        }
        .frame(height: presentation == .inline ? 280 : nil)
        .background {
            if presentation == .fullScreen {
                Color.black
            } else {
                RoundedRectangle(cornerRadius: 8).fill(.quaternary)
            }
        }
        .accessibilityIdentifier("detectionPreview")
    }

    /// Names classes and formats confidence consistently for labels and VoiceOver.
    private func detectionLabel(_ detection: Detection) -> String {
        let name = detection.classID == 1 ? "Basketball" : "Class \(detection.classID)"
        return name + " " + detection.confidence.formatted(
            .percent.precision(.fractionLength(1)))
    }

    /// Draws readable labels after all outlines so boxes cannot overpaint the text.
    private func drawLabels(in context: inout GraphicsContext, layout: PreviewGeometry) {
        guard layout.imageRect.width > 12, layout.imageRect.height > 8 else { return }
        for detection in detections {
            guard let box = layout.displayedBox(detection.boundingBox) else { continue }
            let text = context.resolve(
                Text(detectionLabel(detection))
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white))
            let size = text.measure(in: CGSize(
                width: layout.imageRect.width - 12, height: .infinity))
            let rect = layout.labelRect(
                above: box, size: CGSize(width: size.width + 12, height: size.height + 8))
            context.fill(
                Path(roundedRect: rect, cornerRadius: 4), with: .color(.red))
            var labelContext = context
            labelContext.clip(to: Path(rect))
            labelContext.draw(text, in: rect.insetBy(dx: 6, dy: 4))
        }
    }

    /// Preserves the established SwiftUI image/box rendering for still images.
    private func stillImage(_ image: CGImage, layout: PreviewGeometry) -> some View {
        ZStack(alignment: .topLeading) {
            Image(decorative: image, scale: 1, orientation: .up)
                .resizable()
                .frame(width: layout.imageRect.width, height: layout.imageRect.height)
                .position(x: layout.imageRect.midX, y: layout.imageRect.midY)
            ForEach(detections.indices, id: \.self) { index in
                let detection = detections[index]
                if let box = layout.displayedBox(detection.boundingBox) {
                    Rectangle()
                        .stroke(.red, lineWidth: 2)
                        .frame(width: box.width, height: box.height)
                        .clipped()
                        .position(x: box.midX, y: box.midY)
                }
            }
        }
    }

}
