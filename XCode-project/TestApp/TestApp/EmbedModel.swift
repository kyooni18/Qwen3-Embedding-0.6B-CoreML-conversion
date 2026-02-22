import CoreML
import Tokenizers
import Foundation

final class Qwen3Embedder {
    private let model: Qwen3Embedding06B
    private let tokenizer: any Tokenizer
    private let maxLen: Int

    init(maxLen: Int = 256) async throws {
        self.maxLen = maxLen

        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try Qwen3Embedding06B(configuration: config)
        self.tokenizer = try await AutoTokenizer.from(pretrained: "Qwen/Qwen3-Embedding-0.6B")
    }

    func embed(_ text: String) throws -> [Double] {
        let enc = tokenizer.encode(text: text)
        let ids: [Int32] = padOrTruncate(enc.map(Int32.init), to: maxLen, pad: 0)
        let mask: [Int32] = padOrTruncate(Array(repeating: Int32(1), count: min(enc.count, maxLen)), to: maxLen, pad: 0)
        
        let inputIDs = try makeInt32MultiArray(ids, shape: [1, maxLen])
        let attentionMask = try makeInt32MultiArray(mask, shape: [1, maxLen])

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
        ])

        let out = try model.model.prediction(from: provider)

        let outputKeys = Array(model.model.modelDescription.outputDescriptionsByName.keys)
        guard let key = outputKeys.first,
              let arr = out.featureValue(for: key)?.multiArrayValue
        else {
            throw NSError(domain: "Qwen3Embedder", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "No MLMultiArray output found. Outputs: \(outputKeys)"
            ])
        }

        return multiArrayToDoubles(arr)
    }
}

// ---- helpers ----

private func padOrTruncate<T>(_ a: [T], to n: Int, pad: T) -> [T] {
    if a.count == n { return a }
    if a.count > n { return Array(a.prefix(n)) }
    return a + Array(repeating: pad, count: n - a.count)
}

private func makeInt32MultiArray(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32)
    for i in 0..<values.count { arr[i] = NSNumber(value: values[i]) }
    return arr
}

private func multiArrayToDoubles(_ arr: MLMultiArray) -> [Double] {
    let n = arr.count
    var out: [Double] = []
    out.reserveCapacity(n)

    switch arr.dataType {
    case .float32:
        let p = arr.dataPointer.bindMemory(to: Float32.self, capacity: n)
        for i in 0..<n { out.append(Double(p[i])) }
    case .double:
        let p = arr.dataPointer.bindMemory(to: Double.self, capacity: n)
        out.append(contentsOf: UnsafeBufferPointer(start: p, count: n))
    default:
        for i in 0..<n { out.append(arr[i].doubleValue) }
    }
    return out
}
