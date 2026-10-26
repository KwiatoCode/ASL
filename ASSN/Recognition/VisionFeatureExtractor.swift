import Foundation

/// Opcja ujednolicania strony dłoni – musi być taka sama jak przy treningu (zapisana w metadanych modelu).
nonisolated enum GeometryChirality: String, Sendable {
    /// Bez zmian: widok lustrzany jak w datasecie.
    case none
    /// Odbicie osi X, gdy iloczyn wektorowy (nadgarstek→MCP wskazującego) × (nadgarstek→MCP małego) < 0.
    case canonical
}

/// Ekstraktor cech geometrycznych dłoni – wierna kopia `tools/geometry/hand_features.py` (wersja `geom-v1`).
/// Te same wzory, ta sama kolejność cech; zgodność sprawdza `tools/geometry/tests/parity`.
///
/// Wejście: 21 punktów w kolejności `HandJoint` (MediaPipe), w izotropowych pikselach (ta sama jednostka
/// na obu osiach, oś Y w dół), w widoku lustrzanym (selfie) – patrz `VisionFeatureExtractor.cameraPoints`.
/// Wyjście: 110 cech; model może używać podzbioru (`indices` w metadanych modelu).
///
/// Plik zależy tylko od Foundation, żeby test zgodności dało się skompilować samym `swiftc`.
nonisolated enum VisionFeatureExtractor {
    static let version = "geom-v1"
    static let featureCount = 110

    static let fingers: [[Int]] = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
    ]
    static let tips = [4, 8, 12, 16, 20]
    static let tipPairs: [(Int, Int)] = {
        var pairs: [(Int, Int)] = []
        for i in 0..<5 { for j in (i + 1)..<5 { pairs.append((tips[i], tips[j])) } }
        return pairs
    }()

    typealias Point = SIMD2<Double>

    private static func norm(_ v: Point) -> Double { (v.x * v.x + v.y * v.y).squareRoot() }
    private static func cross(_ u: Point, _ v: Point) -> Double { u.x * v.y - u.y * v.x }
    private static func dot(_ u: Point, _ v: Point) -> Double { u.x * v.x + u.y * v.y }
    /// Kąt między kolejnymi odcinkami: 0 = prosto, π = zawrócenie.
    private static func bend(_ u: Point, _ v: Point) -> Double { atan2(abs(cross(u, v)), dot(u, v)) }

    /// Wszystkie 110 cech. `points` musi mieć 21 elementów.
    static func extract(_ points: [Point], chirality: GeometryChirality) -> [Double] {
        precondition(points.count == 21, "potrzeba 21 punktów")
        var rel = points.map { $0 - points[0] }
        if chirality == .canonical, cross(rel[5], rel[17]) < 0 {
            rel = rel.map { Point(-$0.x, $0.y) }
        }
        let scale = (norm(rel[5]) + norm(rel[9]) + norm(rel[13]) + norm(rel[17])) / 4
        let q = rel.map { $0 / max(scale, 1e-9) }

        var f: [Double] = []
        f.reserveCapacity(featureCount)

        // A: współrzędne względne stawów 1…20
        for j in 1...20 { f.append(q[j].x); f.append(q[j].y) }
        // B1: odległości od nadgarstka
        for j in 1...20 { f.append(norm(q[j])) }
        // B2: odległości między czubkami
        for (a, b) in tipPairs { f.append(norm(q[a] - q[b])) }
        // B3: czubki – środek dłoni
        let center = (q[0] + q[5] + q[9] + q[13] + q[17]) / 5
        for t in tips { f.append(norm(q[t] - center)) }
        // B4: prostoliniowość palców
        for chain in fingers {
            var segments = 0.0
            for k in 1...3 { segments += norm(q[chain[k + 1]] - q[chain[k]]) }
            f.append(norm(q[chain[4]] - q[chain[1]]) / max(segments, 1e-9))
        }
        // C1: zgięcia w stawach
        for chain in fingers {
            for k in 1...3 {
                f.append(bend(q[chain[k]] - q[chain[k - 1]], q[chain[k + 1]] - q[chain[k]]))
            }
        }
        // C2: rozstaw palców
        let proximal = (1...4).map { q[fingers[$0][2]] - q[fingers[$0][1]] }
        f.append(bend(proximal[0], proximal[1]))
        f.append(bend(proximal[1], proximal[2]))
        f.append(bend(proximal[2], proximal[3]))
        f.append(bend(q[4] - q[1], q[8] - q[5]))
        // C3: kciuk
        let axis = q[9]
        let axisLength = max(norm(axis), 1e-9)
        f.append(norm(q[4] - q[5]))
        f.append(norm(q[4] - q[6]))
        f.append(norm(q[4] - q[9]))
        f.append(dot(q[4], axis) / axisLength)
        f.append(cross(axis, q[4]) / axisLength)
        // D: orientacja
        let line = q[17] - q[5]
        let lineLength = max(norm(line), 1e-9)
        f.append(axis.x / axisLength)
        f.append(axis.y / axisLength)
        f.append(line.x / lineLength)
        f.append(line.y / lineLength)
        f.append(cross(q[5], q[17]) / max(norm(q[5]) * norm(q[17]), 1e-9))
        f.append(lineLength / axisLength)

        assert(f.count == featureCount)
        return f
    }

    /// Cechy wybrane przez model (kolejność z metadanych `indices`).
    static func extract(_ points: [Point], chirality: GeometryChirality, indices: [Int]) -> [Double] {
        let all = extract(points, chirality: chirality)
        return indices.map { all[$0] }
    }
}
