import CoreGraphics

/// Checks image/overlay alignment independently of the model and SwiftUI renderer.
@main
struct PreviewGeometrySmokeTests {
    /// Covers landscape/portrait letterboxing, display clipping and empty layouts.
    static func main() {
        let landscape = PreviewGeometry(
            sourceSize: CGSize(width: 1600, height: 900),
            viewportSize: CGSize(width: 400, height: 400)
        )
        precondition(
            landscape.imageRect == CGRect(x: 0, y: 87.5, width: 400, height: 225))
        let box = CGRect(x: 160, y: 90, width: 320, height: 180)
        precondition(
            landscape.displayedBox(box) == CGRect(x: 40, y: 110, width: 80, height: 45))
        let outside = CGRect(x: -100, y: 20, width: 300, height: 40)
        precondition(
            landscape.displayedBox(outside)
                == CGRect(x: 0, y: 92.5, width: 50, height: 10))
        precondition(
            outside.origin.x == -100, "Display clipping mutated source geometry")
        precondition(
            landscape.displayedBox(CGRect(x: 1700, y: 0, width: 10, height: 10)) == nil)
        let portrait = PreviewGeometry(
            sourceSize: CGSize(width: 900, height: 1600),
            viewportSize: CGSize(width: 400, height: 400)
        )
        precondition(
            portrait.imageRect == CGRect(x: 87.5, y: 0, width: 225, height: 400))
        let empty = PreviewGeometry(sourceSize: .zero, viewportSize: .zero)
        precondition(empty.displayedBox(box) == nil)
        precondition(
            landscape.labelRect(
                above: CGRect(x: 40, y: 180, width: 80, height: 45),
                size: CGSize(width: 120, height: 24))
                == CGRect(x: 40, y: 154, width: 120, height: 24))
        precondition(
            landscape.labelRect(
                above: CGRect(x: 390, y: 87.5, width: 10, height: 20),
                size: CGSize(width: 120, height: 24))
                == CGRect(x: 280, y: 87.5, width: 120, height: 24),
            "Top/right edge labels must stay inside the letterboxed image")
        precondition(
            portrait.labelRect(
                above: CGRect(x: 300, y: 100, width: 10, height: 20),
                size: CGSize(width: 500, height: 24))
                == CGRect(x: 87.5, y: 74, width: 225, height: 24),
            "Wide labels must fit the portrait image rather than the viewport")
        print(
            "Preview geometry: aspect fit, display clipping and edge label placement passed."
        )
    }
}
