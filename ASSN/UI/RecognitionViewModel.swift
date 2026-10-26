import CoreGraphics
import Foundation
import Observation

@Observable
final class RecognitionViewModel {
    private(set) var handState: HandState?
    private(set) var stats = FrameStats()
    private(set) var prediction: SignPrediction?
    /// Płótno wysyłane do modelu – do porównania z obrazkami z datasetu.
    private(set) var modelInput: CGImage?
    private(set) var errorMessage: String?
    /// Czy podgląd kamery jest lustrzany – overlay odbija punkty tylko wtedy.
    private(set) var isPreviewMirrored = false
    private(set) var diagnostics = RecognitionDiagnostics()
    var framing: SkeletonFraming = .normalized {
        didSet { pipeline.setFraming(framing) }
    }
    var mode: RecognitionMode = .geometry {
        didSet { pipeline.setMode(mode) }
    }
    /// Widok lustrzany jak w datasecie; wyłącz przy pokazywaniu znaków drugą dłonią.
    var mirror = true {
        didSet { pipeline.setMirror(mirror) }
    }

    let camera = CameraManager()
    private let pipeline: HandTrackingPipeline
    private var isRunning = false

    /// Model aktualnego trybu (nil = brak pliku modelu).
    var modelName: String? {
        switch mode {
        case .geometry: pipeline.geometryModelName
        case .image: pipeline.imageModelName
        }
    }
    var missingModelMessage: String {
        switch mode {
        case .geometry: pipeline.geometryLoadError ?? "Brak ASLGeometryRF.mlmodel w folderze ASSN"
        case .image: "Brak ASLClassifier.mlpackage (ResNeXt) w folderze ASSN"
        }
    }

    init() {
        pipeline = HandTrackingPipeline(camera: camera)
        camera.onPreviewMirroringChanged = { [weak self] mirrored in
            self?.isPreviewMirrored = mirrored
        }
        pipeline.onHandState = { [weak self] state, stats in
            Task { @MainActor in
                self?.handState = state
                self?.stats = stats
            }
        }
        pipeline.onPrediction = { [weak self] prediction, image, diagnostics in
            Task { @MainActor in
                self?.prediction = prediction
                self?.modelInput = image
                self?.diagnostics = diagnostics
            }
        }
    }

    func start() async {
        guard !isRunning else { return }
        do {
            try await camera.start()
            isRunning = true
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func stop() {
        camera.stop()
        isRunning = false
    }
}
