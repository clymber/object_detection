import CoreGraphics

/// Maps top-left source pixels into a centered aspect-fit image preview.
nonisolated struct PreviewGeometry {
    let imageRect: CGRect
    private let sourceBounds: CGRect
    private let scale: CGFloat

    /// Computes the displayed image rectangle, including letterbox offsets.
    init(sourceSize: CGSize, viewportSize: CGSize) {
        sourceBounds = CGRect(origin: .zero, size: sourceSize)
        guard sourceSize.width > 0, sourceSize.height > 0,
            viewportSize.width > 0, viewportSize.height > 0
        else {
            scale = 0
            imageRect = .zero
            return
        }
        scale = min(
            viewportSize.width / sourceSize.width,
            viewportSize.height / sourceSize.height)
        let size = CGSize(
            width: sourceSize.width * scale, height: sourceSize.height * scale)
        imageRect = CGRect(
            x: (viewportSize.width - size.width) / 2,
            y: (viewportSize.height - size.height) / 2,
            width: size.width, height: size.height
        )
    }

    /// Clips only the drawn box; the underlying detection remains unchanged.
    func displayedBox(_ sourceBox: CGRect) -> CGRect? {
        guard scale > 0 else { return nil }
        let visible = sourceBox.intersection(sourceBounds)
        guard !visible.isNull, !visible.isEmpty else { return nil }
        return CGRect(
            x: imageRect.minX + visible.minX * scale,
            y: imageRect.minY + visible.minY * scale,
            width: visible.width * scale, height: visible.height * scale
        )
    }
}
