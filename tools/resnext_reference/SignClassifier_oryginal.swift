import CoreML
import CoreVideo
import Foundation
import Vision

nonisolated struct SignPrediction: Sendable {
    var label: String
    var confidence: Float
    /// Trzy najbardziej prawdopodobne litery (po wygładzeniu).
    var top: [(label: String, confidence: Float)]
}

/// Model Core ML: obraz szkieletu 224×224 → ResNeXt50 (embedding 2048) → regresja logistyczna (A–Z).
/// Szuka w bundlu pierwszego skompilowanego modelu (`*.mlmodelc`) – wystarczy wrzucić `.mlpackage` do folderu `ASSN`.
nonisolated final class SignClassifier {
    let modelName: String
    private let request: VNCoreMLRequest

    init?() {
        guard let url = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil)?.first else { return nil }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        guard let model = try? MLModel(contentsOf: url, configuration: configuration),
              let visionModel = try? VNCoreMLModel(for: model) else { return nil }

        modelName = url.deletingPathExtension().lastPathComponent
        request = VNCoreMLRequest(model: visionModel)
        // Płótno jest kwadratowe, więc to po prostu Resize((224, 224)) z notebooka.
        request.imageCropAndScaleOption = .scaleFill
    }

    /// Prawdopodobieństwa wszystkich klas.
    func classify(_ image: CVPixelBuffer) throws -> [String: Float] {
        let handler = VNImageRequestHandler(cvPixelBuffer: image, orientation: .up)
        try handler.perform([request])
        let results = request.results as? [VNClassificationObservation] ?? []
        return Dictionary(results.map { ($0.identifier, $0.confidence) }, uniquingKeysWith: max)
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
