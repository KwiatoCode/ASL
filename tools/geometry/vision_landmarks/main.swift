// Punkty dłoni z PRAWDZIWYCH ZDJĘĆ przez natywne Apple Vision (macOS) – do testu zgodności Vision ↔ MediaPipe.
//
// Kompilacja i uruchomienie (macOS, Xcode 26+), z katalogu głównego projektu:
//   swiftc -O tools/geometry/vision_landmarks/main.swift -o /tmp/vision_landmarks
//   /tmp/vision_landmarks <folder ze zdjęciami, podfoldery A…Z> landmarks_vision.csv [--mirror]
//
// Wynik: CSV w tym samym formacie co extract_landmarks_mediapipe.py (coord_space = camera_px, source = apple_vision),
// kolejność stawów MediaPipe (= HandJoint w aplikacji). --mirror: odbicie poziome (widok lustrzany jak w datasecie).
import Foundation
import ImageIO
import Vision

let order: [VNHumanHandPoseObservation.JointName] = [
    .wrist, .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
    .indexMCP, .indexPIP, .indexDIP, .indexTip,
    .middleMCP, .middlePIP, .middleDIP, .middleTip,
    .ringMCP, .ringPIP, .ringDIP, .ringTip,
    .littleMCP, .littlePIP, .littleDIP, .littleTip,
]

let args = CommandLine.arguments
guard args.count >= 3 else {
    print("użycie: vision_landmarks <folder> <wynik.csv> [--mirror]")
    exit(2)
}
let root = URL(fileURLWithPath: args[1])
let mirror = args.contains("--mirror")
let fm = FileManager.default
var header = ["source", "letter", "file", "index", "group", "status", "reason", "coord_space", "image_w", "image_h",
              "handedness", "hand_score"]
for j in 0..<21 { header += ["x\(j)", "y\(j)"] }
header += (0..<21).map { "conf\($0)" }
var lines = [header.joined(separator: ",")]

let folders = ((try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? [])
    .filter { $0.lastPathComponent.count == 1 && $0.lastPathComponent.first!.isLetter }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }
var detected = 0, total = 0
for folder in folders {
    let letter = folder.lastPathComponent.uppercased()
    let files = ((try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [])
        .filter { ["jpg", "jpeg", "png", "bmp", "heic"].contains($0.pathExtension.lowercased()) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    var found = 0
    for (n, url) in files.enumerated() {
        total += 1
        let base = ["apple_vision", letter, "\(folder.lastPathComponent)/\(url.lastPathComponent)", "\(n)",
                    "apple_vision_\(letter)_\(n / 100)"]
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            lines.append((base + ["odrzucone", "blad_odczytu"]).joined(separator: ","))
            continue
        }
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 1
        try? VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
        guard let hand = request.results?.max(by: { $0.confidence < $1.confidence }),
              let points = try? hand.recognizedPoints(.all) else {
            lines.append((base + ["odrzucone", "brak_dloni"]).joined(separator: ","))
            continue
        }
        let w = Double(image.width), h = Double(image.height)
        var coords: [String] = [], confs: [String] = []
        var missing = 0
        for joint in order {
            if let p = points[joint], p.confidence > 0 {
                // Vision: 0…1, początek w lewym DOLNYM rogu → piksele z początkiem w lewym górnym rogu
                let x = mirror ? 1 - Double(p.location.x) : Double(p.location.x)
                coords += [String(format: "%.3f", x * w), String(format: "%.3f", (1 - Double(p.location.y)) * h)]
                confs.append(String(format: "%.3f", p.confidence))
            } else {
                coords += ["", ""]
                confs.append("0")
                missing += 1
            }
        }
        let chirality = hand.chirality == .left ? "left" : (hand.chirality == .right ? "right" : "")
        let row = base + [missing == 0 ? "ok" : "odrzucone", missing == 0 ? "" : "brak_\(missing)_punktow",
                          "camera_px", "\(Int(w))", "\(Int(h))", chirality, String(format: "%.4f", hand.confidence)]
        lines.append((row + coords + confs).joined(separator: ","))
        if missing == 0 {
            detected += 1
            found += 1
        }
    }
    print("\(letter): \(found)/\(files.count)")
}
try lines.joined(separator: "\n").write(toFile: args[2], atomically: true, encoding: .utf8)
print("Pełne dłonie (21 punktów): \(detected)/\(total) → \(args[2])")
