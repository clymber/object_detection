import SwiftUI

/// Displays source pixels and boxes using the same aspect-fit transform.
struct FramePreview: View {
    let image: CGImage?
    let detections: [Detection]
    let imageTitle: String

    var body: some View {
        GeometryReader { geometry in
            if let image {
                let layout = PreviewGeometry(
                    sourceSize: CGSize(width: image.width, height: image.height),
                    viewportSize: geometry.size
                )
                ZStack(alignment: .topLeading) {
                    Image(decorative: image, scale: 1, orientation: .up)
                        .resizable()
                        .frame(
                            width: layout.imageRect.width,
                            height: layout.imageRect.height
                        )
                        .position(x: layout.imageRect.midX, y: layout.imageRect.midY)
                    ForEach(detections.indices, id: \.self) { index in
                        let detection = detections[index]
                        if let box = layout.displayedBox(detection.boundingBox) {
                            Rectangle()
                                .stroke(.red, lineWidth: 2)
                                .overlay(alignment: .topLeading) {
                                    Text(label(for: detection))
                                        .font(.caption2.weight(.semibold))
                                        .foregroundStyle(.white)
                                        .padding(3)
                                        .background(.red)
                                }
                                .frame(width: box.width, height: box.height)
                                .clipped()
                                .position(x: box.midX, y: box.midY)
                        }
                    }
                }
                .frame(width: geometry.size.width, height: geometry.size.height)
                .clipped()
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("\(imageTitle), \(detections.count) detections")
            } else {
                ContentUnavailableView("No image loaded", systemImage: "photo")
                    .frame(width: geometry.size.width, height: geometry.size.height)
            }
        }
        .frame(height: 280)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
        .accessibilityIdentifier("detectionPreview")
    }

    /// Labels a category and confidence without rounding stored results.
    private func label(for detection: Detection) -> String {
        let category =
            detection.classID == 1 ? "Basketball" : "Class \(detection.classID)"
        return
            "\(category) \(detection.confidence.formatted(.percent.precision(.fractionLength(1))))"
    }
}
