import CoreGraphics
import Foundation

nonisolated extension VisionFeatureExtractor {
    /// Punkty z `HandState` (0…1, początek w lewym górnym rogu, kadr NIEODBITY – tak jak analizował go Vision)
    /// → izotropowe piksele w widoku lustrzanym, czyli układ, w którym powstał dataset (nagrania selfie).
    ///
    /// Proporcje osi zachowuje przejście na piksele klatki; rozmiar klatki nie ma znaczenia dla cech
    /// (przesunięcie do nadgarstka, skala = śródręcze).
    static func cameraPoints(normalized: [CGPoint], imageSize: CGSize, mirror: Bool = true) -> [Point] {
        normalized.map { p in
            Point((mirror ? 1 - Double(p.x) : Double(p.x)) * Double(imageSize.width), Double(p.y) * Double(imageSize.height))
        }
    }
}
