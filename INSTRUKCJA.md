# ASSN – instrukcja: dane i trening (Windows) → model → aplikacja (Mac) → iPhone

Wszystkie komendy Windows to **PowerShell**. Zakładam, że repozytorium jest sklonowane do `$HOME\Downloads\ASL`
na gałęzi `geometry-rf` (rozdz. 5), a dataset szkieletów leży w `$HOME\Downloads\asl_dataset\dataset`.

---

## 1. Windows – środowisko (raz)

```powershell
cd $HOME\Downloads\ASL
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r tools\geometry\requirements.txt
cd tools\geometry
..\..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Wszystkie testy powinny skończyć się `OK`. Dalsze komendy uruchamiaj z folderu `tools\geometry`.
Dla wygody: `Set-Alias py3 ..\..\.venv\Scripts\python.exe` – wtedy zamiast długiej ścieżki piszesz `py3`.

---

## 2. Windows – dane treningowe

### 2A. Punkty odtworzone z obrazów datasetu szkieletów (wolne, niepewne dla pięści)

```powershell
py3 pipeline_landmarks.py --dataset $HOME\Downloads\asl_dataset\dataset --out wyniki_rekonstrukcji --limit 300 --jobs 10
```

- `--limit 300` = 300 obrazów na literę (7 800 razem). Cały dataset (`--limit 0`) trwa wielokrotnie dłużej.
  Czas: około 1–2 s na obraz na rdzeń; przy 10 procesach i 7 800 obrazach rzędu 30–60 min na przebieg (są 3,
  dwa ostatnie tylko dla nieprzyjętych obrazów). Przerwany proces wznowi się od ostatniego zapisanego przebiegu.
- Wynik: `wyniki_rekonstrukcji\landmarks.csv`, `raport_rekonstrukcji.md` (przyjęte per litera),
  `qa\` – **obejrzyj podglądy**: kciuk fioletowy, wskazujący niebieski, środkowy zielony, serdeczny pomarańczowy,
  mały ciemnoczerwony. Jeśli dla jakiejś litery palce są pomylone, nie używaj tej litery z tego źródła.

### 2B. Punkty z prawdziwych zdjęć – zbiór ASL-HG + MediaPipe (zalecane)

1. Pobierz zbiór ze strony https://data.mendeley.com/datasets/j4y5w2c8w9/1 (przycisk *Download All*), rozpakuj
   np. do `$HOME\Downloads\ASL-HG`. Sprawdź na stronie licencję.
2. Pobierz model MediaPipe i zainstaluj bibliotekę:
   ```powershell
   Invoke-WebRequest https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -OutFile hand_landmarker.task
   py3 -m pip install mediapipe
   ```
3. Wyciągnij punkty (skrypt sam znajdzie podfoldery o nazwach A…Z; cyfry pomija):
   ```powershell
   mkdir wyniki_aslhg -Force
   py3 extract_landmarks_mediapipe.py --images $HOME\Downloads\ASL-HG --model hand_landmarker.task --out wyniki_aslhg\landmarks_aslhg.csv --source aslhg --mirror
   ```
   `--mirror` przelicza zdjęcia na widok lustrzany, w którym pracuje aplikacja i dataset szkieletów.

---

## 3. Windows – trening, raport, eksport

```powershell
py3 train_geometry_rf.py --landmarks wyniki_aslhg\landmarks_aslhg.csv wyniki_rekonstrukcji\landmarks.csv --out wyniki_rf --jobs 10
```

(Jedno źródło też działa – podaj tylko jeden plik.) Wynik w `wyniki_rf\`:
`raport.md` (wszystkie tabele), `macierz_pomylek.png`, `waznosc_cech.png`, `raport_klasyfikacji.txt`,
`cv_strojenie.csv`, `metrics.json`, `feature_spec.json`, modele `.joblib`.
Przy dwóch źródłach raport zawiera też **trening na jednym źródle → test na drugim** – to najlepsza miara,
czy model działa na innych dłoniach.

Eksport do aplikacji z weryfikacją:

```powershell
py3 export_geometry_coreml.py --model wyniki_rf\model_rf.joblib --check wyniki_rf\model_rf_ocena.joblib --output ..\..\ASSN\ASLGeometryRF.mlmodel
```

Musi się pojawić `zgodność liter 100.00%` i różnica prawdopodobieństwa poniżej 1e-4. Jeśli nie – skrypt
zatrzymuje się z ostrzeżeniem i takiego modelu nie należy używać.

---

## 4. Mac (kolega) – testy, aplikacja, iPhone

### 4.1 Pobranie

```bash
git clone -b geometry-rf https://github.com/KwiatoCode/ASL.git
cd ASL
```

W folderze `ASSN/` musi być `ASLGeometryRF.mlmodel`. Opcjonalnie (tryb porównawczy ResNeXt) – oryginalny
`ASLClassifier.mlpackage` z pierwszej wersji projektu. Jeśli w `ASSN/` leży `ASLClassifier.mlmodel` z eksperymentu
pikselowego – usuń go (aplikacja go nie użyje).

### 4.2 Test zgodności cech Swift ↔ Python (obowiązkowy, ~1 min)

```bash
swiftc -O ASSN/Recognition/VisionFeatureExtractor.swift tools/geometry/tests/parity/main.swift -o /tmp/assn_parity
/tmp/assn_parity tools/geometry/tests/parity/feature_fixtures.json
```

Oczekiwane: `błędów: 0`. Inny wynik = Swift liczy cechy inaczej niż trening → model w aplikacji byłby błędny.

### 4.3 (Zalecane) Apple Vision na tych samych zdjęciach co MediaPipe

```bash
swiftc -O tools/geometry/vision_landmarks/main.swift -o /tmp/vision_landmarks
/tmp/vision_landmarks ~/Downloads/ASL-HG landmarks_vision.csv --mirror
```

Plik `landmarks_vision.csv` odeślij – na Windowsie:

```powershell
py3 compare_detectors.py --mediapipe wyniki_aslhg\landmarks_aslhg.csv --vision landmarks_vision.csv --model wyniki_rf\model_rf.joblib --out wyniki_rf\porownanie_detektorow.md
```

To pokazuje, jak bardzo punkty Vision różnią się od MediaPipe i jaka jest trafność modelu na punktach Vision.

### 4.4 Uruchomienie aplikacji

1. Otwórz `ASSN.xcodeproj`, wybierz **My Mac**, ⌘R. Zezwól na kamerę.
2. Na górze po prawej: przełącznik **Geometria (RF) / Obraz (ResNeXt)** i **Lustro (jak w datasecie)**.
   Jeśli pokazujesz litery drugą dłonią niż ta w danych treningowych, wyłącz „Lustro”.
3. Po lewej diagnostyka: liczba punktów uzupełnionych i niepewnych, czas modelu. Przy > 3 uzupełnionych
   punktach aplikacja nie zgaduje litery i pokazuje powód.

Jeśli kompilacja się nie powiedzie – zrzut ekranu panelu błędów (⌘5) i odesłanie. Kod Swift był pisany bez Xcode.

### 4.5 iPhone

1. Podłącz iPhone kablem, odblokuj, zaufaj komputerowi.
2. Xcode → projekt **ASSN** → target → *Signing & Capabilities*: **Team** = swoje Apple ID,
   **Bundle Identifier** unikalny, np. `pl.imie.assn`.
3. Na iPhonie: *Ustawienia → Prywatność i ochrona → Tryb dewelopera* → włącz (restart telefonu).
4. W Xcode wybierz iPhone jako cel i ⌘R. Przy pierwszym uruchomieniu: *Ustawienia → Ogólne → VPN i urządzenia →
   zaufaj deweloperowi*.
5. Sprawdź każdą literę kilka razy i zapisz, które są mylone – to jedyny pomiar skuteczności na żywo.

---

## 5. Gałąź w repozytorium (Windows)

```powershell
cd $HOME\Downloads
git clone https://github.com/KwiatoCode/ASL.git ASL
cd ASL
git checkout -b geometry-rf
# skopiuj tu zawartość paczki (z zastąpieniem plików), potem:
git add -A
git commit -m "Apple Vision + cechy geometryczne + Random Forest (ResNeXt zachowany jako odniesienie)"
git push -u origin geometry-rf
```

Model `ASLGeometryRF.mlmodel` dodaj osobnym commitem po treningu (krok 3). Pliki wyników (`wyniki_*`, `.joblib`,
`.npz`) są w `.gitignore` – do sprawozdania wybierz z nich raporty i wykresy.
