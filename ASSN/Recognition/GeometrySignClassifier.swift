import CoreML
import Foundation

nonisolated enum GeometryClassifierError: LocalizedError {
    case missingOutput

    var errorDescription: String? { "Model nie zwrócił prawdopodobieństw klas" }
}

/// Klasyfikator NOWEJ architektury: 21 punktów Apple Vision → cechy geometryczne → Random Forest (Core ML).
/// Bez sieci neuronowej i bez obrazu – wejściem są wyłącznie liczby z `VisionFeatureExtractor`.
///
/// Model `ASLGeometryRF.mlmodel` (z `tools/geometry/export_geometry_coreml.py`) niesie w metadanych wersję cech,
/// wybrany zestaw (`indices`) i sposób ujednolicania strony dłoni; model o innej wersji cech jest odrzucany.
nonisolated final class GeometrySignClassifier {
    static let resourceName = "ASLGeometryRF"

    let modelName: String
    let featureSet: String
    let chirality: GeometryChirality
    private let indices: [Int]
    private let model: MLModel
    private let inputName: String
    private let probabilitiesName: String

    /// Dlaczego model się nie wczytał (do wyświetlenia w aplikacji).
    nonisolated(unsafe) static var lastLoadError: String?

    init?() {
        guard let url = compiledModelURL(named: Self.resourceName) else {
            Self.lastLoadError = "Brak \(Self.resourceName).mlmodel w folderze ASSN"
            return nil
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly          // drzewa decyzyjne liczą się na CPU
        guard let model = try? MLModel(contentsOf: url, configuration: configuration) else {
            Self.lastLoadError = "Nie można wczytać \(Self.resourceName)"
            return nil
        }
        let description = model.modelDescription
        let meta = description.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard meta["feature_version"] == VisionFeatureExtractor.version else {
            Self.lastLoadError = "Model ma cechy \(meta["feature_version"] ?? "?"), aplikacja \(VisionFeatureExtractor.version)"
            return nil
        }
        let indices = (meta["indices"] ?? "").split(separator: ",").compactMap { Int($0) }
        guard !indices.isEmpty, indices.allSatisfy({ (0..<VisionFeatureExtractor.featureCount).contains($0) }),
              let inputName = description.inputDescriptionsByName.keys.first,
              let probabilitiesName = description.predictedProbabilitiesName else {
            Self.lastLoadError = "Niepełne metadane modelu \(Self.resourceName)"
            return nil
        }
        self.model = model
        self.indices = indices
        self.inputName = inputName
        self.probabilitiesName = probabilitiesName
        self.featureSet = meta["feature_set"] ?? "?"
        self.chirality = GeometryChirality(rawValue: meta["chirality"] ?? "none") ?? .none
        self.modelName = "\(Self.resourceName) · \(indices.count) cech"
        Self.lastLoadError = nil
    }

    /// Cechy dla modelu z punktów `HandState`. `mirror` = widok lustrzany jak w datasecie (domyślnie tak).
    func features(for state: HandState, mirror: Bool) -> [Double] {
        let points = VisionFeatureExtractor.cameraPoints(normalized: state.points, imageSize: state.imageSize, mirror: mirror)
        // scikit-learn porównuje cechy jako float32 – to samo zaokrąglenie daje identyczne ścieżki w drzewach.
        return VisionFeatureExtractor.extract(points, chirality: chirality, indices: indices).map { Double(Float($0)) }
    }

    func classify(_ state: HandState, mirror: Bool) throws -> [String: Float] {
        let values = features(for: state, mirror: mirror)
        let array = try MLMultiArray(shape: [NSNumber(value: values.count)], dataType: .double)
        for (k, v) in values.enumerated() { array[k] = NSNumber(value: v) }
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: array)])
        let output = try model.prediction(from: input)
        guard let probabilities = output.featureValue(for: probabilitiesName)?.dictionaryValue else {
            throw GeometryClassifierError.missingOutput
        }
        var result: [String: Float] = [:]
        for (label, probability) in probabilities {
            result["\(label.base)"] = probability.floatValue
        }
        return result
    }
}
