import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import os

/// Sposób rozpoznawania litery.
nonisolated enum RecognitionMode: String, CaseIterable, Identifiable, Sendable {
    /// NOWA architektura: punkty Apple Vision → cechy geometryczne → Random Forest.
    case geometry
    /// Punkt odniesienia: obraz szkieletu → ResNeXt50 → regresja logistyczna.
    case image

    var id: String { rawValue }
    var title: String {
        switch self {
        case .geometry: "Geometria (RF)"
        case .image: "Obraz (ResNeXt)"
        }
    }
}

/// Diagnostyka jakości punktów i rozpoznawania – pokazywana w aplikacji.
nonisolated struct RecognitionDiagnostics: Sendable {
    /// Punkty uzupełnione przez `LandmarkProcessor` (Vision ich nie zwrócił lub miały za niską pewność).
    var filledPoints = 0
    /// Punkty zwrócone przez Vision z pewnością < 0.5.
    var lowConfidencePoints = 0
    var handConfidence: Float = 0
    /// Powód pominięcia klasyfikacji (nil = sklasyfikowano).
    var skipped: String?
    var inferenceMs: Double = 0
}

/// Łączy etapy: Frame Manager → Hand Detector → Landmark Processor → Hand Filter → Hand State → rozpoznawanie.
/// Śledzenie dłoni chodzi na kolejce wideo kamery z pełnym FPS; rozpoznawanie na osobnej kolejce
/// i pomija klatki, gdy model jeszcze liczy poprzednią – dzięki temu nie spowalnia podglądu.
///
/// Tryb geometryczny: HandState → VisionFeatureExtractor → GeometrySignClassifier (Random Forest) → PredictionSmoother.
/// Szkielet jest wtedy rysowany wyłącznie jako podgląd – nie jest wejściem klasyfikatora.
nonisolated final class HandTrackingPipeline: @unchecked Sendable {
    private let frameManager: FrameManager
    private let detector = HandDetector()
    private let processor = LandmarkProcessor()
    private let filter = HandFilter()

    private let renderer = SkeletonImageRenderer()
    private let geometryClassifier = GeometrySignClassifier()
    private let imageClassifier = ImageSignClassifier()
    private let smoother = PredictionSmoother()
    /// Stan zmieniany tylko na kolejce rozpoznawania.
    private var mode: RecognitionMode = .geometry
    private var mirror = true
    /// Więcej uzupełnionych punktów = nie klasyfikujemy (cechy byłyby w dużej części zgadywane).
    private let maxFilledPoints = 3
    private let recognitionQueue = DispatchQueue(label: "asl.recognition", qos: .userInitiated)
    private let isRecognizing = OSAllocatedUnfairLock(initialState: false)
    private let log = Logger(subsystem: "MyApp", category: "Pipeline")

    /// Nazwy dostępnych modeli (nil = brak pliku modelu).
    var geometryModelName: String? { geometryClassifier?.modelName }
    var imageModelName: String? { imageClassifier?.modelName }
    var geometryLoadError: String? { GeometrySignClassifier.lastLoadError }

    /// Wywoływane na kolejce wideo po każdej klatce.
    var onHandState: ((HandState?, FrameStats) -> Void)?
    /// Wywoływane na kolejce rozpoznawania. Obraz = podgląd szkieletu (w trybie obrazowym to wejście ResNeXt).
    var onPrediction: ((SignPrediction?, CGImage?, RecognitionDiagnostics) -> Void)?

    init(camera: CameraManager) {
        frameManager = FrameManager(clock: camera.session.synchronizationClock ?? CMClockGetHostTimeClock())
        camera.onFrame = { [weak self] sampleBuffer in self?.process(sampleBuffer) }
        camera.onDrop = { [weak self] in self?.frameManager.recordSystemDrop() }
    }

    func setFraming(_ framing: SkeletonFraming) {
        recognitionQueue.async { [renderer, smoother] in
            renderer.framing = framing
            smoother.reset()
        }
    }

    func setMode(_ mode: RecognitionMode) {
        recognitionQueue.async { [weak self] in
            self?.mode = mode
            self?.smoother.reset()
        }
    }

    /// Lustro = widok selfie jak w datasecie. Wyłącz, gdy pokazujesz znaki drugą dłonią niż w danych treningowych.
    func setMirror(_ mirror: Bool) {
        recognitionQueue.async { [weak self] in
            self?.mirror = mirror
            self?.renderer.mirror = mirror
            self?.smoother.reset()
        }
    }

    private func process(_ sampleBuffer: CMSampleBuffer) {
        guard let frame = frameManager.accept(sampleBuffer) else { return }
        defer { frameManager.finish(frame) }

        let state: HandState?
        do {
            state = try track(frame)
        } catch {
            log.error("Vision: \(error.localizedDescription)")
            state = nil
        }

        if state == nil {
            processor.reset()
        }
        onHandState?(state, frameManager.stats)
        recognize(state)
    }

    private func track(_ frame: Frame) throws -> HandState? {
        guard let raw = try detector.detect(in: frame.pixelBuffer),
              let hand = processor.process(raw, timestamp: frame.timestamp) else { return nil }

        let smoothed = filter.filter(hand, imageSize: frame.size, timestamp: frame.timestamp)
        return HandState(
            points: smoothed,
            pointConfidences: hand.confidences,
            confidence: hand.confidence,
            handedness: hand.handedness,
            timestamp: frame.timestamp,
            imageSize: frame.size
        )
    }

    /// Bez modelu i tak renderujemy płótno, żeby dało się porównać kadrowanie z datasetem.
    private func recognize(_ state: HandState?) {
        guard let state else {
            recognitionQueue.async { [smoother, onPrediction] in
                smoother.reset()
                onPrediction?(nil, nil, RecognitionDiagnostics(skipped: "Brak dłoni"))
            }
            return
        }
        // Model jeszcze liczy – pomijamy tę klatkę zamiast budować kolejkę.
        let busy = isRecognizing.withLock { busy in
            defer { busy = true }
            return busy
        }
        guard !busy else { return }

        recognitionQueue.async { [weak self] in
            guard let self else { return }
            defer { self.isRecognizing.withLock { $0 = false } }

            var diagnostics = RecognitionDiagnostics(
                filledPoints: state.pointConfidences.filter { $0 == 0 }.count,
                lowConfidencePoints: state.pointConfidences.filter { $0 > 0 && $0 < 0.5 }.count,
                handConfidence: state.confidence
            )
            let preview = self.renderer.render(state)
            let previewImage = preview.flatMap { Self.makeCGImage($0) }

            if diagnostics.filledPoints > self.maxFilledPoints {
                diagnostics.skipped = "Za dużo brakujących punktów (\(diagnostics.filledPoints))"
                self.smoother.reset()
                self.onPrediction?(nil, previewImage, diagnostics)
                return
            }

            let started = CFAbsoluteTimeGetCurrent()
            do {
                let probabilities: [String: Float]
                switch self.mode {
                case .geometry:
                    guard let classifier = self.geometryClassifier else {
                        diagnostics.skipped = GeometrySignClassifier.lastLoadError ?? "Brak modelu geometrycznego"
                        self.onPrediction?(nil, previewImage, diagnostics)
                        return
                    }
                    probabilities = try classifier.classify(state, mirror: self.mirror)
                case .image:
                    guard let classifier = self.imageClassifier, let preview else {
                        diagnostics.skipped = "Brak modelu ResNeXt (ASLClassifier.mlpackage)"
                        self.onPrediction?(nil, previewImage, diagnostics)
                        return
                    }
                    probabilities = try classifier.classify(preview)
                }
                diagnostics.inferenceMs = (CFAbsoluteTimeGetCurrent() - started) * 1000
                let prediction = self.smoother.add(probabilities)
                self.onPrediction?(prediction, previewImage, diagnostics)
            } catch {
                self.log.error("Core ML: \(error.localizedDescription)")
                diagnostics.skipped = "Błąd modelu: \(error.localizedDescription)"
                self.onPrediction?(nil, previewImage, diagnostics)
            }
        }
    }

    private static func makeCGImage(_ buffer: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        return CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer), bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )?.makeImage()
    }
}
