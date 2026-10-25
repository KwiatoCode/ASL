# ASSN – uruchomienie na Macu (wersja z Random Forest)

Model ResNeXt + regresja logistyczna został zastąpiony lasem losowym (Random Forest).
Gotowy model `ASLClassifier.mlmodel` leży już w folderze `ASSN/` – nie trzeba nic trenować.

## Wymagania

- Mac z Xcode (wersja z README, sekcja 6)
- do testu na iPhonie: kabel, iPhone z włączonym Trybem dewelopera

## Uruchomienie

1. Rozpakuj paczkę, np. na Biurko.
2. Sprawdź, że w folderze `ASSN/` jest plik **`ASLClassifier.mlmodel`** i że **nie ma** starego `ASLClassifier.mlpackage`
   (aplikacja bierze pierwszy model, jaki znajdzie – stary model ResNeXt by ją zmylił).
3. Otwórz `ASSN.xcodeproj` (dwuklik).
4. Wybierz cel uruchomienia u góry okna Xcode:
   - **My Mac** – najprościej, kamera wbudowana w Maca,
   - **iPhone** – najpierw *Signing & Capabilities* → wybierz swoje **Team** (Apple ID) i wpisz unikalny **Bundle Identifier**,
     np. `pl.twojeimie.assn`; na telefonie: Ustawienia → Prywatność i ochrona → Tryb dewelopera → włącz.
5. Naciśnij **⌘R**. Przy pierwszym uruchomieniu zezwól na dostęp do kamery.
   Jeśli Mac zapyta o hasło do pęku kluczy: wpisz hasło do Maca i kliknij *Always Allow*.

Po starcie pokaż dłoń do kamery – na dole pojawi się litera i 3 najbardziej prawdopodobne litery z procentami.
Jeśli zamiast tego widać „Brak modelu – wrzuć plik .mlmodel do folderu ASSN”, wróć do kroku 2.

## Gdy coś nie działa

- **Błąd kompilacji** – zrób zrzut ekranu z panelu błędów (ikona ⚠️ / ❌ po lewej) i odeślij.
  Kod Swift (`ASSN/Recognition/SignClassifier.swift`) był zmieniany bez dostępu do Maca, więc drobna poprawka może być potrzebna.
- **Litera rzadko się pojawia** – las daje niższe pewności niż poprzedni model. W `SignClassifier.swift`
  zmień `minConfidence: Float = 0.5` na `0.3`.
- **Słabe rozpoznawanie** – przełącz kadrowanie w aplikacji na *Normalizacja* (domyślne) i trzymaj dłoń w całości w kadrze.

## Co się zmieniło względem poprzedniej wersji

| plik | zmiana |
|---|---|
| `ASSN/Recognition/SignClassifier.swift` | zamiast sieci: obraz szkieletu 400×400 → jasność w blokach 8×8 (2500 liczb) → Random Forest |
| `ASSN/ASLClassifier.mlmodel` | nowy model (Random Forest), zastępuje `ASLClassifier.mlpackage` |
| `ASSN/ContentView.swift` | komunikat o brakującym modelu |
| `tools/build_coreml.py` | trening lasu i eksport do Core ML |
| `tools/tune_random_forest.py` | strojenie hiperparametrów (GridSearchCV) + wykresy do sprawozdania |
| `tools/colab_strojenie.ipynb` | to samo strojenie w Google Colab |
| `random_forest_asl.ipynb` | notebook z Random Forest zamiast ResNeXt |

Kamera, wykrywanie szkieletu (Apple Vision), filtr i rysowanie szkieletu – bez zmian.

## Ponowne wytrenowanie modelu (opcjonalnie)

```bash
cd ASSN-folder-projektu
python3 -m venv .venv && source .venv/bin/activate
pip install "scikit-learn==1.5.1" coremltools pillow joblib numpy
python tools/build_coreml.py --train <folder_dataset_z_literami_A-Z> --count 500 --trees 100
```
