# ASSN — rozpoznawanie alfabetu ASL: Apple Vision + geometria dłoni + Random Forest

> Kamera iPhone'a → **Apple Vision** (21 punktów dłoni) → normalizacja → **110 cech geometrycznych** → **Random Forest** → litera A–Z

Bez sieci neuronowej i bez obrazu jako wejścia klasyfikatora. Szkielet jest rysowany już tylko jako podgląd.
Poprzednia wersja (obraz szkieletu → ResNeXt-50 → regresja logistyczna) jest zachowana jako punkt odniesienia
i można ją wybrać przełącznikiem w aplikacji.

Instrukcja krok po kroku (Windows → Mac → iPhone): **[INSTRUKCJA.md](INSTRUKCJA.md)**.

---

## 0. Stan prac – co jest sprawdzone, a co nie

| Element | Stan |
|---|---|
| Ekstraktor cech (Python) | gotowy, testy jednostkowe przechodzą (niezmienniczość, znane wartości) |
| Ekstraktor cech (Swift) | napisany wzór w wzór; **test zgodności Swift ↔ Python czeka na uruchomienie na Macu** (`swiftc`) |
| Rekonstrukcja punktów z obrazów datasetu | gotowa i zmierzona: przyjmuje ~52 % obrazów próbki, przyjęte w ~98,6 % poprawne (test syntetyczny); słabe litery C, G, J, O, P, Q, T, Z (rozdz. 3) |
| Alternatywne dane (zdjęcia + MediaPipe) | skrypt gotowy; **nie uruchomiony** (zbiór trzeba pobrać na Windowsie) – potrzebny do słabych liter |
| Trening, strojenie, raport | gotowe, przetestowane na danych syntetycznych |
| Eksport do Core ML + weryfikacja | gotowe; matematyka drzew sprawdzona testem (zgodność z `predict_proba` do 1e-12); samo `coremltools` uruchamia się u Ciebie |
| Aplikacja SwiftUI | zmodyfikowana; **nieskompilowana** (brak Xcode w moim środowisku) |
| Wytrenowany model `ASLGeometryRF.mlmodel` | **brak** – powstaje dopiero po uruchomieniu treningu na prawdziwych danych; w repozytorium nie ma atrapy modelu |
| Test na iPhonie | **niewykonany** |

---

## 1. Przepływ danych

```
 ① KAMERA (AVFoundation)  ─►  ② FRAME MANAGER  ─►  ③ HAND DETECTOR · Apple Vision VNDetectHumanHandPoseRequest
                                                         │  21 × (x, y, pewność), strona dłoni
                                                         ▼
                                                   ④ LANDMARK PROCESSOR  (walidacja, układ MediaPipe, uzupełnianie)
                                                         ▼
                                                   ⑤ ONE EURO FILTER  ─►  ⑥ HAND STATE ──► podgląd (zielone punkty)
                                                         ▼
                              ┌──────────── tryb „Geometria (RF)” (domyślny) ────────────┐
                              │ ⑦ VisionFeatureExtractor  – 110 cech geom-v1              │
                              │ ⑧ GeometrySignClassifier  – Random Forest (Core ML)      │
                              └──────────────────────────────────────────────────────────┘
                              ┌──────────── tryb „Obraz (ResNeXt)” – punkt odniesienia ───┐
                              │ SkeletonImageRenderer 400×400 → ResNeXt-50 → regresja log. │
                              └──────────────────────────────────────────────────────────┘
                                                         ▼
                                                   ⑨ PREDICTION SMOOTHER  (6 klatek, próg 0.4, top-3)
                                                         ▼
                                                   ⑩ RecognitionViewModel → interfejs
```

Kolejki: `asl.camera.video` (①–⑥, pełny FPS), `asl.recognition` (⑦–⑨, klatka pomijana, gdy model liczy),
główna – tylko UI. Diagnostyka w aplikacji: punkty uzupełnione / niepewne, pewność dłoni, czas modelu;
przy > 3 uzupełnionych punktach klasyfikacja jest pomijana (komunikat zamiast zgadywania).

---

## 2. Zgodność Apple Vision ↔ dane treningowe

| Kwestia | Dataset szkieletów | Aplikacja (Vision) | Ujednolicenie |
|---|---|---|---|
| Kolejność stawów | MediaPipe 0…20 | nazwy Vision | `HandJoint` = kolejność MediaPipe |
| Początek układu / oś Y | lewy górny róg, Y w dół | lewy **dolny**, Y w górę | `LandmarkProcessor`: y → 1 − y |
| Proporcje osi | płótno 400×400 z kadru 640×480 (X ściśnięte) | 0…1 względem klatki | oba na izotropowe piksele: X·640/400, Y·480/400 / X·W, Y·H |
| Odbicie lustrzane | nagrania selfie (lustro) | klatka nieodbita | aplikacja odbija X (przełącznik „Lustro”) |
| Lewa / prawa dłoń | nieznana | `chirality` | opcja `canonical` w cechach (wybierana eksperymentem) lub przełącznik „Lustro” |
| Orientacja kamery | – | klatka obrócona do pionu przed Vision | bez zmian |
| Brakujące punkty | brak | uzupełniane, oznaczane | pominięcie klasyfikacji przy > 3 uzupełnionych |
| Położenie i wielkość dłoni | – | – | cechy względem nadgarstka, skala = średnia nadgarstek → MCP |

Normalizacja nie usuwa różnic perspektywy ani różnic między detektorami (MediaPipe ≠ Vision). Dlatego są:
test odporności na szum/obrót/skalę (`train_geometry_rf.py`), test przenoszenia między zbiorami oraz
`compare_detectors.py` + `vision_landmarks` (Vision i MediaPipe na tych samych zdjęciach – uruchamiane na Macu).

---

## 3. Dane treningowe

### 3.1 Dataset szkieletów z Kaggle nie ma współrzędnych

[ASL Hand Skeleton Dataset](https://www.kaggle.com/datasets/usifelshafie/american-sign-language-asl-hand-skeleton-dataset)
to 26 000 plików JPG (26 × 1000), bez CSV ze współrzędnymi. Obrazy narysowano `mediapipe draw_landmarks`:
21 czerwonych kości, potem 21 punktów (biała obwódka r = 3 + zielone koło r = 2), na płótnie 400×400.

### 3.2 Rekonstrukcja punktów z rysunków (`tools/geometry/`)

1. Maski kolorów (odporne na JPEG), zielone plamy → kandydaci; zlane plamy dzielone (punkty nałożone).
2. Pokrycie każdego odcinka czerwoną kością (z pominięciem obwódek punktów).
3. **Kolejność rysowania**: MediaPipe rysuje punkty 0…20 po kolei, a obwódka późniejszego nadgryza wcześniejszy.
   Dla punktów bliższych niż 6 px kolejność odczytuje się trafnie w 97–100 % (pomiar na syntetycznych rysunkach).
4. Przeszukiwanie wiązkowe przypisań staw → kandydat zgodnych z 21 kośćmi MediaPipe (najpierw śródręcze).
5. Ocena globalna: zgodność narysowanego rozwiązania z czerwonymi pikselami (recall × precision) + anatomia.
   **Strona dłoni**: podstawa kciuka (CMC) leży po tej samej stronie osi dłoni co MCP wskazującego – rozstrzyga
   odbicie wskazujący ↔ mały nawet przy kciuku schowanym w pięści (100 % trafnych na danych syntetycznych).
6. **Etap 2 – na poziomie litery (EM)**: rysunek bywa niejednoznaczny (pięść, punkty na jednej prostej, odbicie
   wskazujący ↔ mały). Wszystkie obrazy litery budują model kształtu; każdy obraz wybiera wariant zgodny z rysunkiem
   i z kształtem litery; potem szablony liter kierują ponownym przeszukiwaniem (2 przebiegi).
7. Obraz jest **odrzucany**, gdy zwycięzca nie wygrywa wyraźnie, rysunek jest słaby albo kształt nietypowy –
   zamiast zgadywać.

**Pomiar wiarygodności** (szczegóły: [tools/geometry/RAPORT_WALIDACJI.md](tools/geometry/RAPORT_WALIDACJI.md)):

| Test | Wynik |
|---|---|
| Syntetyczne „litery” ze znaną prawdą (20 liter × 25, trudniejsze niż dataset) | najlepszy wariant poprawny w 90,8 %; przyjęte 14 %, z nich **1,4 % z błędną numeracją** |
| Prawdziwa próbka datasetu (1040 obrazów, co 25. plik) | przyjęte **52 %**; kontrola wzrokowa wszystkich liter: wiarygodne poza Q (2 obrazy, prawdopodobnie odbite); **P odrzucone automatycznie** (dłoń bokiem); słabe pokrycie C, G, J, O, Q, T, Z |

**Wniosek:** rekonstrukcja daje wiarygodne punkty dla około połowy obrazów i większości liter. Dla liter z dłonią
bokiem lub w dół (P, Q, G) i zakrzywionych (C, O) brakuje danych albo jest ryzyko odbicia – te litery trzeba
uzupełnić innym źródłem. Po przetworzeniu pełnego datasetu obejrzyj `qa/` i tabelę kontroli strony dłoni. Dlatego:

### 3.3 Alternatywne dane z oryginalnymi punktami

`extract_landmarks_mediapipe.py` – punkty MediaPipe z prawdziwych zdjęć dłoni z podfolderami liter, np.
**ASL-HG** (Pranto i in., *Data in Brief* 2026, [Mendeley Data](https://data.mendeley.com/datasets/j4y5w2c8w9/1)):
zdjęcia A–Z i 0–9; według [opisu modelu na tym zbiorze](https://huggingface.co/nocontextdoruk/asl-landmark-mlp)
MediaPipe znajduje dłoń na ~31 tys. zdjęć. To nadal **punkty MediaPipe, nie Apple Vision** – różnicę mierzy
`compare_detectors.py` (te same zdjęcia przez Vision na Macu). Licencję i zgodność alfabetu sprawdź na stronie zbioru
przed użyciem; nie zweryfikowałem ich, bo nie mam dostępu do pobrania.

Trening może łączyć źródła (`--landmarks a.csv b.csv`); raport pokazuje wtedy też trening na jednym źródle i test na
drugim – najlepszą dostępną bez nagrywania miarę, czy model przenosi się na inne dłonie i inny detektor.

---

## 4. Cechy geometryczne `geom-v1`

Specyfikacja: `tools/geometry/hand_features.py` (referencja) = `ASSN/Recognition/VisionFeatureExtractor.swift`.

| Grupa | Opis | Liczba |
|---|---|---|
| A | x, y stawów 1…20 względem nadgarstka, w jednostkach śródręcza | 40 |
| B1 | odległości stawów od nadgarstka | 20 |
| B2 | odległości między czubkami palców | 10 |
| B3 | czubki palców – środek dłoni | 5 |
| B4 | prostoliniowość palca (czubek–podstawa / suma paliczków) | 5 |
| C1 | zgięcia MCP/PIP/DIP (kciuk CMC/MCP/IP), 0 = prosto | 15 |
| C2 | rozstaw sąsiednich palców, kciuk–wskazujący | 4 |
| C3 | kciuk względem wskazującego i środkowego, położenie wzdłuż/w poprzek osi dłoni | 5 |
| D | kierunek osi dłoni i linii MCP, strona dłoni (iloczyn wektorowy), proporcja śródręcza | 6 |
| **basic** | A + B1 + oś dłoni + iloczyn wektorowy | **63** |
| **extended** | wszystkie | **110** |

Wersja cech, zestaw i wybrane indeksy są zapisywane w metadanych modelu Core ML; aplikacja odrzuca model o innej
wersji. Skala: średnia odległość nadgarstek → MCP 5, 9, 13, 17. Bez obrotu – orientacja należy do znaku (K/P, H/U, G/Q).

---

## 5. Trening i ocena (`train_geometry_rf.py`)

- **Podział grupowy**: sąsiednie numery plików to najpewniej klatki tego samego nagrania. Grupa = litera + blok
  100 kolejnych numerów; test = ~20 % bloków każdej litery, nigdy nie używany do strojenia; walidacja krzyżowa po blokach.
  Metadanych sesji brak – bloki to ostrożne przybliżenie (dwa nagrania mogą dzielić blok, jedno nagranie może
  przekraczać granicę bloku).
- **Analiza przecieku**: ten sam model przy podziale losowym i grupowym + odległość do najbliższego sąsiada.
- **Eksperymenty**: basic / extended × strona dłoni none / canonical; potem siatka `n_estimators` 100/200/300/500,
  `max_depth` None/16/24/32, `min_samples_leaf` 1/2/4, `class_weight="balanced_subsample"`.
- **Augmentacja** (tylko część treningowa, po podziale na bloki): 3 kopie każdej próbki z szumem położenia stawów
  2,5 % śródręcza i obrotem dłoni ±10° – odporność na różnice MediaPipe ↔ Apple Vision. Obrót jest mały, bo K/P, H/U,
  G/Q różnią się orientacją; skalowania nie ma, bo cechy są normalizowane. Raport porównuje model z augmentacją
  i bez niej. Na danych syntetycznych (sprawdzenie kodu): F1 w walidacji bez zmian, a trafność przy szumie 10 %
  0,914 → 0,967 i przy obrocie ±20° 0,967 → 0,994. Opcje: `--augment N` (0 = wyłączona), `--aug-noise`, `--aug-rotate`.
- **Raport**: accuracy, precision, recall, F1 macro i weighted, wyniki per litera, macierz 26×26, mylone pary,
  ważność cech (permutacyjna na teście), odporność (szum 2/5/10 % śródręcza, obrót ±10/20°, skala ±30 %) –
  dla modelu z augmentacją i bez niej.

## 6. Eksport (`export_geometry_coreml.py`)

Zespół drzew Core ML budowany **wprost z drzew scikit-learn** (bez konwertera sklearn z coremltools, który obsługuje
tylko wybrane wersje scikit-learn): liść = rozkład klas / liczba drzew, tak jak `predict_proba`. Weryfikacja odczytuje
drzewa z zapisanej specyfikacji Core ML i przelicza zbiór testowy w Pythonie – zgodność liter i różnica
prawdopodobieństw (próg 1e-4). Aplikacja zaokrągla cechy do float32, tak jak scikit-learn wewnętrznie.

## 7. Litery J i Z

J i Z to ruch. Model statyczny widzi jedną klatkę (w datasecie też są pojedyncze klatki), więc rozpoznaje co najwyżej
charakterystyczną pozę początkową. Nie twierdzę, że rozpoznaje te znaki. Klasyfikator trajektorii wymagałby sekwencji
punktów z czasem – takich danych tu nie ma.

## 8. Mapa kodu

| Etap | Plik |
|---|---|
| kamera, klatki | `ASSN/Camera/CameraManager.swift`, `FrameManager.swift` |
| Vision, punkty, filtr | `ASSN/Tracking/HandDetector.swift`, `HandJoint.swift`, `LandmarkProcessor.swift`, `OneEuroFilter.swift` |
| potok, tryby, diagnostyka | `ASSN/Tracking/HandTrackingPipeline.swift` |
| cechy | `ASSN/Recognition/VisionFeatureExtractor.swift` (+ `+Vision.swift`) |
| Random Forest | `ASSN/Recognition/GeometrySignClassifier.swift` |
| ResNeXt (odniesienie), wygładzanie | `ASSN/Recognition/SignClassifier.swift`, `SkeletonImageRenderer.swift` |
| UI | `ASSN/ContentView.swift`, `ASSN/UI/*` |
| rekonstrukcja punktów | `tools/geometry/reconstruct_skeleton.py`, `resolve_letters.py`, `pipeline_landmarks.py` |
| alternatywne dane | `tools/geometry/extract_landmarks_mediapipe.py`, `vision_landmarks/main.swift`, `compare_detectors.py` |
| cechy, trening, eksport | `tools/geometry/hand_features.py`, `train_geometry_rf.py`, `export_geometry_coreml.py` |
| testy | `tools/geometry/tests/` (unittest, syntetyczna walidacja, zgodność Swift ↔ Python) |
| poprzednie wersje | `tools/resnext_reference/` (ResNeXt + regresja), `tools/pixel_rf_experiment/` (RF na pikselach – działał źle na żywo) |

## 9. Ograniczenia i dalsze kroki

- Brak danych z Apple Vision w treningu – wszystkie dane to MediaPipe (odtworzone lub z MediaPipe na zdjęciach).
- Rekonstrukcja z rysunków: nierówne pokrycie liter (P, Q, G, C, O, J, Z słabo); odrzucanie ogranicza błędy kosztem danych.
- Wyniki testowe dotyczą tych samych zbiorów; rzeczywistą skuteczność trzeba zmierzyć w aplikacji (pokazując litery).
- Dalej: (1) punkty Apple Vision z publicznych zdjęć jako dodatkowe dane treningowe (narzędzie `vision_landmarks` już
  je zapisuje), (2) augmentacja szumem/obrotem przy treningu, (3) klasyfikator sekwencji dla J i Z.

**Wymagania:** Xcode 27 · iOS 27 / macOS 26.6+ (jak w projekcie) · Python 3.10–3.12 (Windows) · `tools/geometry/requirements.txt`.
