# ASSN — rozpoznawanie alfabetu ASL w czasie rzeczywistym

Notebook bazowy (Kaggle, wersja z ResNeXt): https://www.kaggle.com/code/mikedelong/f1-0-99-with-logistic-regression-and-resnext

Link dataset: https://www.kaggle.com/datasets/usifelshafie/american-sign-language-asl-hand-skeleton-dataset

> Kamera → szkielet dłoni → obraz 400×400 → **piksele 50×50** → **Random Forest** → litera A–Z

Aplikacja na iOS i macOS (SwiftUI + Apple Vision + Core ML).
Projekt: `~/Desktop/ASSN/ASSN.xcodeproj`

---

## 1. Przepływ danych — od fotonu do litery

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ① KAMERA  · AVFoundation                                                      │
│     CVPixelBuffer BGRA  640×480 (iPhone w pionie: 480×640)  · do 60 FPS        │
│     klatka obrócona fizycznie do pionu, NIE odbita lustrzanie                  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ② FRAME MANAGER                                                               │
│     odrzuca klatki starsze niż 100 ms i klatki spóźnione (alwaysDiscardsLate)  │
│     mierzy FPS (EMA) i opóźnienie kamera → wynik                               │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ③ HAND DETECTOR  · Apple Vision  VNDetectHumanHandPoseRequest                 │
│     → 21 × (x, y, pewność)  w układzie Vision (0…1, początek: lewy DÓŁ)        │
│     → strona dłoni (left / right)                                              │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ④ LANDMARK PROCESSOR                                                          │
│     • odrzuca punkty z pewnością < 0.3 i całą dłoń z pewnością < 0.5           │
│     • wymaga ≥ 14 / 21 dobrych punktów                                         │
│     • y → 1 − y  (układ MediaPipe: początek w lewym GÓRNYM rogu)               │
│     • kolejność punktów Vision → kolejność MediaPipe (0 = nadgarstek…20)       │
│     • braki: poprzednia klatka (≤ 250 ms) → interpolacja w palcu → ekstrapol.  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ⑤ HAND FILTER  · One Euro Filter  (42 filtry: x i y każdego punktu)           │
│     filtrowanie w pikselach · minCutoff 1.0 Hz · β 0.02 · reset po 300 ms      │
│     wolny ruch → mocne wygładzanie,  szybki ruch → małe opóźnienie             │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ⑥ HAND STATE   21 × CGPoint · pewności · strona dłoni · timestamp · rozmiar   │
│                 (Codable → JSON)                                               │
└──────────────┬───────────────────────────────────────────────┬───────────────┘
               ▼                                               ▼
┌─────────────────────────────┐   ┌────────────────────────────────────────────┐
│  ⑦ RENDERER (podgląd)        │   │  ⑧ SKELETON IMAGE RENDERER  (wejście sieci) │
│  kamera + overlay            │   │  płótno 400×400, białe tło, styl datasetu  │
│  zielone punkty, czerwone    │   │  odbicie lustrzane + kadrowanie            │
│  linie, aspect-fill, lustro  │   └──────────────────────┬─────────────────────┘
└─────────────────────────────┘                          ▼
                                  ┌────────────────────────────────────────────┐
                                  │  ⑨ MODEL  · Core ML  (rozdział 3)          │
                                  │  400×400 → jasność w blokach 8×8 → 2500    │
                                  │  → Random Forest (100 drzew) → 26 klas     │
                                  └──────────────────────┬─────────────────────┘
                                                         ▼
                                  ┌────────────────────────────────────────────┐
                                  │  ⑩ PREDICTION SMOOTHER                     │
                                  │  średnia z 6 ostatnich wyników · próg 0.5  │
                                  │  → litera + top-3                          │
                                  └────────────────────────────────────────────┘
```

### Wątki

| Kolejka | Etapy | Tempo |
|---|---|---|
| `asl.camera.video` | ① – ⑥ | każda klatka, do 60 FPS |
| `asl.recognition` | ⑧ – ⑩ | tyle, ile zdąży model; gdy jest zajęty, klatka jest **pomijana**, a nie kolejkowana |
| main | ⑦, UI | odświeżanie SwiftUI |

Dzięki temu podgląd ze szkieletem chodzi płynnie, nawet gdy model liczy wolniej niż kamera.

---

## 2. Wejście modelu — obraz szkieletu (krok ⑧)

Model uczył się na datasecie **[ASL Hand Skeleton](https://www.kaggle.com/datasets/usifelshafie/american-sign-language-asl-hand-skeleton-dataset)**.
To 26 000 **obrazków** szkieletu narysowanych przez `mediapipe.drawing_utils`, a nie współrzędne.
Dlatego aplikacja odtwarza ten sam styl rysowania. Parametry zmierzyłem na obrazkach z datasetu:

```
        przykład z datasetu (litera L)          parametry rysowania
     ┌──────────────────────────────┐
     │            ●                 │      płótno   400 × 400, białe tło
     │            │                 │      kości    21 połączeń MediaPipe
     │            ●                 │               czerwone (255,0,0), grubość 2
     │            │                 │      stawy    zielone (0,255,0), r = 2,
     │            ●   ●  ●  ●       │               biała obwódka r = 3
     │  ●         │  ╱│ ╱│ ╱│       │      widok    lustrzany (selfie)
     │   ╲        ● ● ● ● ●  ●      │
     │    ●       │╲ ╲│╱  ╱ ╱       │      zgodność z cv2 po skalowaniu
     │     ╲      │ ╲ │ ╱ ╱         │      do 224×224: średnio 0.3/255
     │      ●     │  ╲│╱╱           │      na piksel
     │       ╲    │   ●             │
     │        ●───┴──╱              │
     │                              │
     └──────────────────────────────┘
```

**Kadrowanie** (przełącznik w aplikacji):

| Tryb | Przekształcenie punktów |
|---|---|
| **Kamera** | `(x, y) × 400`, dokładnie jak w datasecie (kamera 640×480 rysowana na kwadracie) |
| **Normalizacja** (domyślnie) | dłoń przeskalowana tak, by odległość nadgarstek → MCP środkowego palca wynosiła **130 px**; oś X ściśnięta o **480/640** (jak w datasecie); środek dłoni przesunięty do **(188, 209)** — średnia z datasetu |

---

## 3. Architektura modelu

Jeden plik Core ML `ASLClassifier.mlmodel` (tree ensemble). Zawiera tylko las losowy;
cechy liczy aplikacja (`SkeletonFeatures` w `SignClassifier.swift`) dokładnie tak samo jak Python (`features()` w `tools/build_coreml.py`).

```
 obraz szkieletu 400×400 RGB  (ten sam co wcześniej, krok ⑧)
          │
          ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ JASNOŚĆ     0.299 R + 0.587 G + 0.114 B,  / 255  →  [0, 1]          │
 ├───────────────────────────────────────────────────────────────────┤
 │ ŚREDNIA     w blokach 8×8 pikseli     400×400  →  50×50            │
 ├───────────────────────────────────────────────────────────────────┤
 │ WEKTOR      wiersz po wierszu, od góry                →  2500      │
 ╞═══════════════════════════════════════════════════════════════════╡
 │                 RANDOM FOREST (sklearn → Core ML)                  │
 ├───────────────────────────────────────────────────────────────────┤
 │ 100 drzew decyzyjnych, pełna głębokość, √2500 = 50 cech na podział │
 │ każde drzewo głosuje na literę                                     │
 │ prawdopodobieństwo = średnia z drzew  (predict_proba)              │
 ├───────────────────────────────────────────────────────────────────┤
 │ WYJŚCIE  classLabel  (A…Z)  ·  classProbability  {A: p, …, Z: p}   │
 └───────────────────────────────────────────────────────────────────┘
```

### Dlaczego piksele 50×50

Random Forest nie ma warstw konwolucyjnych, więc dostaje obraz bezpośrednio jako liczby.
Pełne 400×400×3 = 480 000 cech byłoby za dużo, a szkielet jest prosty (czerwone linie i zielone punkty na białym tle),
więc uśrednienie bloków 8×8 zachowuje kształt dłoni. Uśrednianie bloków (a nie skalowanie obrazka)
daje w Swift i w Pythonie identyczny wynik — różnica w teście: ~1e-15.

### Parametry

| Część | Parametry | Trenowane? |
|---|---|---|
| Cechy pikselowe | brak (stała formuła) | nie |
| Random Forest | 100 drzew (`n_estimators=100`, `random_state=2025`) | tak (`random_forest.pkl`) |

### Trening (`random_forest_asl.ipynb`)

```
 dataset 26 × 1000 obrazków
        │  500 / literę
        ▼
 piksele 50×50  →  wektor 2500
        │
        ▼
 train/test split 75 / 25 (stratify, seed 2025)
        │
        ▼
 RandomForestClassifier(n_estimators=100, random_state=2025)   →   accuracy, F1, classification report
```

Reszta notebooka bez zmian: t-SNE na cechach, wykres bokeh, histogram prawdopodobieństw.

### Strojenie hiperparametrów (`tools/tune_random_forest.py`)

GridSearchCV na części treningowej (5-fold, metryka F1 ważone), potem ocena najlepszego zestawu na części testowej
i porównanie z lasem o domyślnych parametrach.

| parametr | wartości |
|---|---|
| `n_estimators` | 50, 100, 200, 300 |
| `max_depth` | None, 10, 20, 30 |
| `max_features` | sqrt, log2 |
| `min_samples_leaf` | 1, 2, 4 |

96 zestawów × 5 podziałów = 480 treningów. Wyniki (tabele i wykresy do sprawozdania) trafiają do `wyniki_strojenia/`:
`podsumowanie.md`, `wszystkie_wyniki.csv`, `raport_klasyfikacji.txt`, wykresy wpływu parametrów i macierz pomyłek,
oraz najlepszy las `random_forest.pkl`.

```bash
python tools/tune_random_forest.py --train <dataset>            # strojenie
python tools/tune_random_forest.py --train <dataset> --export   # + ASLClassifier.mlmodel z najlepszego lasu
```

### Konwersja do Core ML (`tools/build_coreml.py`)

```
 random_forest.pkl ──► coremltools.converters.sklearn.convert
                         wejście  features: MultiArray(2500)
                         wyjście  classLabel + classProbability
                                           ▼
                             ASSN/ASLClassifier.mlmodel
```

---

## 4. Mapa kodu

| Etap | Plik |
|---|---|
| ① kamera | `ASSN/Camera/CameraManager.swift` |
| ② frame manager | `ASSN/Camera/FrameManager.swift` |
| ③ detektor | `ASSN/Tracking/HandDetector.swift` · `HandJoint.swift` |
| ④ przetwarzanie | `ASSN/Tracking/LandmarkProcessor.swift` |
| ⑤ filtr | `ASSN/Tracking/OneEuroFilter.swift` |
| ⑥ stan | `ASSN/Tracking/HandState.swift` |
| ①–⑩ orkiestracja | `ASSN/Tracking/HandTrackingPipeline.swift` |
| ⑦ podgląd | `ASSN/UI/CameraPreviewView.swift` · `HandOverlayView.swift` |
| ⑧ płótno | `ASSN/Recognition/SkeletonImageRenderer.swift` |
| ⑨ ⑩ cechy + Random Forest + wygładzanie | `ASSN/Recognition/SignClassifier.swift` |
| konwersja / trening | `tools/build_coreml.py` · `random_forest_asl.ipynb` |

---

## 5. Ograniczenia i kierunki rozwoju

**Ograniczenia:**
- **MediaPipe kontra Apple Vision.** Model widział szkielety z MediaPipe, a aplikacja daje szkielety z Vision.
  Różnice są największe przy zaciśniętej dłoni (A, E, M, N, S, T).
- **Brak augmentacji.** Model uczył się na 500 obrazkach na literę, bez przesunięć, obrotów i skali,
  więc jest wrażliwy na kadrowanie.
- **J i Z** to litery z ruchem, a w datasecie są pojedynczymi klatkami.
- **Random Forest na pikselach** jest wrażliwszy na przesunięcie dłoni niż sieć konwolucyjna — dlatego ważne jest kadrowanie *Normalizacja*.
- **Prawdopodobieństwa lasu** to udział głosujących drzew, więc są zwykle niższe niż z regresji logistycznej; próg wygładzania to nadal 0.5.

**Kierunki rozwoju:**
- **A.** Dotrenowanie lasu na obrazkach z augmentacją (przesunięcie, skala, obrót, szum punktów).
- **B.** Model na współrzędnych: 42 liczby z Vision → MLP → 26 klas, uczony na danych z iPhone'a.
  Działa na tych samych danych co na żywo.

---

## 6. Uruchomienie (skrót)

```bash
# przebudowanie modelu
python tools/build_coreml.py --model random_forest.pkl     # z lasu z notebooka
python tools/build_coreml.py --train <dataset> --count 500 # trening jak w notebooku
```

Przed uruchomieniem usuń stary `ASSN/ASLClassifier.mlpackage` (ResNeXt) — aplikacja bierze pierwszy model z bundla.

- **Mac:** wybierz *My Mac* i naciśnij ⌘R.
- **iPhone:** *Signing & Capabilities* → Team + unikalny Bundle ID → Tryb dewelopera na telefonie → ⌘R (telefon odblokowany).
- Pytanie o hasło do pęku kluczy: wpisz hasło do Maca i kliknij *Always Allow*.

**Wymagania:** Xcode 27 · iOS 27 / macOS 26.6+ · iPhone, iPad, Mac.
