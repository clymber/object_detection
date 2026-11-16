import SwiftUI

/// Reserves space for future frames and detection overlays without showing results.
struct FramePreview: View {
    let inputName: String

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "viewfinder")
                .font(.largeTitle)
                .accessibilityHidden(true)

            Text("No frame loaded")
                .font(.headline)

            Text("Selected input: \(inputName)")
                .font(.subheadline)
        }
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity, minHeight: 180)
        .padding()
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [6]))
                .foregroundStyle(.tertiary)
        }
        .accessibilityElement(children: .combine)
    }
}

#Preview {
    FramePreview(inputName: "Static image")
        .padding()
}
