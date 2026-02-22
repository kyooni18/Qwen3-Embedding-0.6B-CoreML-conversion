import SwiftUI

struct ContentView: View {
    @State var embedder: Qwen3Embedder?
    @State var loading: Bool = false
    @State var text: String = ""
    @State var vectors: [Double] = []
    var body: some View {
        VStack {
            ScrollView {
                ForEach(0..<vectors.count, id: \.self) { i in
                    Text("\(vectors[i])")
                        .contentTransition(.identity)
                        .animation(.spring, value: vectors)
                }
            }
            HStack() {
                TextField("Input some text", text: $text)
                    .padding(7)
                    .glassEffect()
                Button("OK") {
                    do {
                        vectors = try embedder?.embed(text) ?? []
                    } catch {}
                }
                .buttonStyle(.glass)
                .disabled(text.isEmpty)
                .disabled(embedder == nil)
            }
            .padding(.horizontal)
            
            Button(loading ? "Loading..." : "Load Model") {
                Task {
                    loading = true
                    embedder = try? await Qwen3Embedder()
                    loading = false
                }
            }
            .scaleEffect(embedder != nil ? 0.0000001 : 1)
            .opacity(embedder != nil ? 0 : 1)
            .animation(.bouncy, value: (embedder != nil))
            .animation(.bouncy, value: loading)
            .buttonStyle(.glass)
        }
        
    }
}

#Preview {
    ContentView()
}
