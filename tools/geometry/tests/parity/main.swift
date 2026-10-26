// Test zgodności cech Swift ↔ Python.
// Kompilacja i uruchomienie (macOS z Xcode 26+ / Swift 6.2+), z katalogu głównego projektu:
//   swiftc -O ASSN/Recognition/VisionFeatureExtractor.swift tools/geometry/tests/parity/main.swift -o /tmp/assn_parity
//   /tmp/assn_parity tools/geometry/tests/parity/feature_fixtures.json
import Foundation

let path = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "tools/geometry/tests/parity/feature_fixtures.json"
guard let data = FileManager.default.contents(atPath: path),
      let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let cases = root["cases"] as? [[String: Any]] else {
    print("Nie można wczytać \(path)")
    exit(2)
}
let tolerance = root["tolerance"] as? Double ?? 1e-9
let version = root["version"] as? String ?? "?"
guard version == VisionFeatureExtractor.version else {
    print("Inna wersja cech: fixtures \(version), Swift \(VisionFeatureExtractor.version)")
    exit(1)
}

var worst = 0.0
var failures = 0
for (n, item) in cases.enumerated() {
    let raw = item["points"] as! [[Double]]
    let expected = item["features"] as! [Double]
    let chirality = GeometryChirality(rawValue: item["chirality"] as! String)!
    let got = VisionFeatureExtractor.extract(raw.map { SIMD2($0[0], $0[1]) }, chirality: chirality)
    guard got.count == expected.count else {
        print("przykład \(n): liczba cech \(got.count) ≠ \(expected.count)")
        failures += 1
        continue
    }
    for k in 0..<got.count {
        let diff = abs(got[k] - expected[k])
        worst = max(worst, diff)
        if diff > tolerance {
            print("przykład \(n), cecha \(k): Swift \(got[k]) ≠ Python \(expected[k])")
            failures += 1
        }
    }
}
print("Przykładów: \(cases.count), największa różnica: \(worst), błędów: \(failures)")
exit(failures == 0 ? 0 : 1)
