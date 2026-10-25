import CoreML
import CoreVideo
import Foundation

nonisolated struct SignPrediction: Sendable {
    var label: String
    var confidence: Float
    /// Trzy najbardziej prawdopodobne litery (po wygładzeniu).
    var top: [(label: String, confidence: Float)]
}

nonisolated enum SignClassifierError: Error {
    case unexpectedImage
    case missingOutput
}

/// Cechy dla Random Forest – liczone identycznie jak `features()` w `tools/build_coreml.py`:
/// jasność 0.299 R + 0.587 G + 0.114 B (0…1) płótna 400×400 uśredniona w blokach 8×8 → 50 × 50 = 2500 liczb,
/// wiersz po wierszu od góry.
nonisolated enum SkeletonFeatures {
    static let grid = 50
    static let count = grid * grid

    static func extract(_ buffer: CVPixelBuffer) throws -> MLMultiArray {
        let size = SkeletonImageRenderer.canvasSize
        let block = size / grid
        guard CVPixelBufferGetWidth(buffer) == size, CVPixelBufferGetHeight(buffer) == size,
              CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else {
            throw SignClassifierError.unexpectedImage
        }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw SignClassifierError.unexpectedImage }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)

        // Bufor BGRA (premultipliedFirst + byteOrder32Little): w pamięci B, G, R, A. Wiersz 0 = góra obrazka.
        var sums = [Double](repeating: 0, count: count)
        for y in 0..<size {
            let row = bytes + y * bytesPerRow
            let blockRow = (y / block) * grid
            for x in 0..<size {
                let pixel = row + x * 4
                sums[blockRow + x / block] += 0.299 * Double(pixel[2]) + 0.587 * Double(pixel[1]) + 0.114 * Double(pixel[0])
            }
        }

        let features = try MLMultiArray(shape: [NSNumber(value: count)], dataType: .double)
        let scale = 1 / (255 * Double(block * block))
        for (index, sum) in sums.enumerated() {
            features[index] = NSNumber(value: sum * scale)
        }
        return features
    }
}

/// Model Core ML: obraz szkieletu 400×400 → piksele 50×50 → Random Forest (A–Z).
/// Szuka w bundlu pierwszego skompilowanego modelu (`*.mlmodelc`) – wystarczy wrzucić `.mlmodel` do folderu `ASSN`.
nonisolated final class SignClassifier {
    let modelName: String
    private let model: MLModel
    private let inputName: String
    private let probabilitiesName: String

    init?() {
        guard let url = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil)?.first else { return nil }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        guard let model = try? MLModel(contentsOf: url, configuration: configuration),
              let inputName = model.modelDescription.inputDescriptionsByName.keys.first,
              let probabilitiesName = model.modelDescription.predictedProbabilitiesName else { return nil }

        modelName = url.deletingPathExtension().lastPathComponent
        self.model = model
        self.inputName = inputName
        self.probabilitiesName = probabilitiesName
    }

    /// Prawdopodobieństwa wszystkich klas (średnia głosów drzew).
    func classify(_ image: CVPixelBuffer) throws -> [String: Float] {
        let input = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(multiArray: SkeletonFeatures.extract(image))]
        )
        let output = try model.prediction(from: input)
        guard let probabilities = output.featureValue(for: probabilitiesName)?.dictionaryValue else {
            throw SignClassifierError.missingOutput
        }
        var result: [String: Float] = [:]
        for (label, probability) in probabilities {
            result["\(label)"] = probability.floatValue
        }
        return result
    }
}

/// Uśrednia prawdopodobieństwa z kilku ostatnich klatek, żeby litera nie migała.
nonisolated final class PredictionSmoother: @unchecked Sendable {
    var window = 6
    /// Poniżej tego progu nie pokazujemy litery.
    var minConfidence: Float = 0.5

    private var history: [[String: Float]] = []

    func reset() { history.removeAll() }

    func add(_ probabilities: [String: Float]) -> SignPrediction? {
        history.append(probabilities)
        if history.count > window { history.removeFirst(history.count - window) }

        var sums: [String: Float] = [:]
        for frame in history {
            for (label, probability) in frame { sums[label, default: 0] += probability }
        }
        let ranked = sums
            .map { (label: $0.key, confidence: $0.value / Float(history.count)) }
            .sorted { $0.confidence > $1.confidence }

        guard let best = ranked.first, best.confidence >= minConfidence else { return nil }
        return SignPrediction(label: best.label, confidence: best.confidence, top: Array(ranked.prefix(3)))
    }
}
