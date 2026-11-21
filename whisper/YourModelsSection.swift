import SwiftUI

/// "Your models" in a model picker: the GGUFs the user put in the Your
/// Models folder, converted and ready, or still on their way, and how to
/// add one.
struct YourModelsSection: View {
    enum Tab { case image, music }
    let tab: Tab
    /// The built-in model it runs with, for an imported image or music
    /// model the user picks.
    let choose: (ImportedModel) -> Void
    let isChosen: (ImportedModel) -> Bool

    @State private var deleting: ImportedModel?
    private var store: YourModels { YourModels.shared }

    private var models: [ImportedModel] {
        store.models.filter { tab == .image ? ($0.isImage || $0.isRewriter) : $0.isMusic }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("YOUR MODELS")
                .font(Studio.mono(10, weight: .semibold))
                .foregroundColor(Studio.mute)
                .padding(.top, 14)
                .padding(.horizontal, 4)
            ForEach(models) { model in
                Button {
                    if model.isRewriter {
                        store.rewriterSelection = store.rewriterSelection == model.id ? nil : model.id
                    } else {
                        choose(model)
                    }
                } label: { card(model) }
                .buttonStyle(PressableButtonStyle())
                .accessibilityAddTraits(chosen(model) ? .isSelected : [])
                .contextMenu {
                    Button(role: .destructive) { deleting = model } label: {
                        Label("Delete (\(ImageModel.formatBytes(model.bytes)))", systemImage: "trash")
                    }
                }
            }
            ForEach(store.pending.keys.sorted(), id: \.self) { name in
                if let status = store.pending[name] { pendingCard(name, status) }
            }
            Text(help)
                .font(Studio.text(12))
                .foregroundColor(Studio.mute)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 4)
        }
        .task { store.scan() }
        .confirmationDialog(deleting.map { "Delete \($0.name)?" } ?? "", isPresented: Binding(
            get: { deleting != nil }, set: { if !$0 { deleting = nil } }), titleVisibility: .visible) {
            if let model = deleting {
                Button("Delete \(ImageModel.formatBytes(model.bytes))", role: .destructive) {
                    store.delete(model)
                    deleting = nil
                }
            }
        } message: {
            Text("The converted model leaves this device. If its .gguf is still in Your Models, it's converted again.")
        }
    }

    private var help: String {
        let models = tab == .image
            ? "Z-Image-Turbo, FLUX.2 [klein] 4B or Qwen-Image 2.1, or Qwen3.5 9B for Enhance prompt"
            : "ACE-Step 1.5 XL"
        return "Use your own: put a .gguf of \(models) (a fine-tune, say) in the Files app under On My \(UIDevice.current.model) › Whisperkit › Your Models. It's converted once, then listed here, and runs with the built-in model's other parts. Touch and hold one to delete it."
    }

    private func chosen(_ model: ImportedModel) -> Bool {
        model.isRewriter ? store.rewriterSelection == model.id : isChosen(model)
    }

    private func card(_ model: ImportedModel) -> some View {
        let selected = chosen(model)
        return VStack(alignment: .leading, spacing: 7) {
            Text(model.name)
                .font(Studio.text(16, weight: .semibold))
                .foregroundColor(Studio.ink)
                .lineLimit(2)
            Text(model.isRewriter
                 ? (selected ? "Used for Enhance prompt. Tap to go back to the built-in one." : "For Enhance prompt. Tap to use it there.")
                 : "Runs with \(model.baseName)'s prompt reader and decoder.")
                .font(Studio.text(12))
                .foregroundColor(Studio.mute)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 7) {
                chip("YOUR MODEL", tint: Studio.accent)
                chip(ImageModel.formatBytes(model.bytes))
                chip(model.baseName.uppercased())
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous)
            .fill(selected ? Studio.panel : Studio.sunk))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
            .strokeBorder(selected ? Studio.accent : Studio.rule, lineWidth: selected ? 1.5 : 0.5))
    }

    private func pendingCard(_ name: String, _ status: YourModels.Status) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(name)
                .font(Studio.text(15, weight: .semibold))
                .foregroundColor(Studio.ink)
                .lineLimit(2)
            switch status {
            case .converting(let fraction):
                ProgressView(value: fraction)
                    .tint(Studio.accent)
                Text("Converting · \(Int(fraction * 100))% · keep Whisperkit open")
                    .font(Studio.text(12))
                    .foregroundColor(Studio.mute)
            case .waiting(let message), .failed(let message):
                Text(message)
                    .font(Studio.text(12))
                    .foregroundColor(Studio.mute)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Studio.sunk))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(Studio.rule, lineWidth: 0.5))
    }

    private func chip(_ text: String, tint: Color = Studio.mute) -> some View {
        Text(text)
            .font(Studio.mono(9, weight: .semibold))
            .foregroundColor(tint)
            .lineLimit(1)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Capsule().fill(tint.opacity(0.12)))
    }
}
