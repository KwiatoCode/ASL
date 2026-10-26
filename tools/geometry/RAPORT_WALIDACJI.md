# Rekonstrukcja punktów dłoni z obrazów datasetu – walidacja

Wszystkie liczby poniżej zmierzono w trakcie przygotowania kodu (Linux, Python 3.13). Skrypty do powtórzenia
pomiarów są w `tests/`. Pełny dataset (26 000 obrazów) nie był przetwarzany – tylko próbka 1040 obrazów
(co 25. plik każdej litery, dostarczona przez autora projektu).

## 1. Czy współrzędne istnieją w datasecie?

Nie. Archiwum z Kaggle zawiera dokładnie 26 000 plików JPG 400×400 (26 × 1000), bez CSV/JSON.
Strona zbioru nie opisuje współrzędnych. Obrazy narysowano funkcją `mediapipe draw_landmarks`
(czerwone kości grubości 2, zielone punkty r = 2 z białą obwódką r = 3), co potwierdza wygląd pikseli.

## 2. Syntetyczne rysunki ze znaną prawdą

`synthetic_hands.py`: kinematyczny model dłoni 3D (zgięcia MCP/PIP/DIP, kciuk, obroty do ±40°),
rzut na kadr 640×480, rysunek identyczny z MediaPipe, JPEG 75–95. Celowo trudniejszy niż dataset
(losowe obroty 3D, dużo pięści). Błąd > 10 % długości śródręcza na którymkolwiek stawie = błędna numeracja.

| Pomiar | Wynik |
|---|---|
| Kandydaci (zielone plamy) obejmują wszystkie prawdziwe stawy | 100 % |
| Odczyt kolejności rysowania nakładających się punktów (< 6 px) | 97–100 % trafnych |
| Reguła strony dłoni (podstawa kciuka po stronie MCP wskazującego) | 100 % trafnych |

Pełny proces (etap 1 + EM po literach + 2 przebiegi z szablonami), 20 syntetycznych liter × 25 obrazów
(`tests/eval_letters_synthetic.py --letters 20 --per-letter 25`):

| Wersja | Najlepszy wariant poprawny | Przyjęte | Błędnie przyjęte (z przyjętych) |
|---|---|---|---|
| bez reguły strony dłoni | 66,6 % | 13,4 % | 19,4 % |
| **z regułą strony dłoni (obecna)** | **90,8 %** | **14,2 %** | **1,4 %** |

Odrzucanie jest ostrożne (wymaga wyraźnej przewagi zwycięskiego wariantu), dlatego na tych trudnych danych
przyjmowana jest mała część obrazów – ale przyjęte są w ~98,6 % poprawne.

## 3. Prawdziwa próbka datasetu (1040 obrazów)

`pipeline_landmarks.py --dataset probka/dataset`, wynik: [raport_rekonstrukcji_probka.md](walidacja/raport_rekonstrukcji_probka.md).

- Przyjęte: **540 / 1040 (51,9 %)**.
- Dobre pokrycie (≥ 50 % obrazów litery): A, B, D, E, F, H, I, K, L, M, N, U, V, W, Y.
- Słabe (< 25 %): C, G, J, O, Q, T, Z – zakrzywione palce (C, O), dłoń bokiem / w dół (G, Q), ruch (J, Z).
- **P odrzucone w całości automatycznie**: strona dłoni nierozstrzygalna (dłoń w dół, bokiem); ocena wzrokowa
  potwierdziła odbitą numerację przed tą regułą.
- Kontrola wzrokowa po 3 losowe przyjęte obrazy z każdej litery: [walidacja/kontrola_wzrokowa_*.png](walidacja/)
  (kciuk fioletowy, wskazujący niebieski, środkowy zielony, serdeczny pomarańczowy, mały ciemnoczerwony).
  Wynik oceny: litery A, B, C, D, E, F, G, H, I, J, K, L, M, N, R, S, T, U, V, W, X, Y, Z – numeracja wiarygodna;
  **Q – 2 przyjęte obrazy z prawdopodobnie odbitą numeracją** (wskazujący ↔ mały przy dłoni skierowanej w dół).
- Historia: przed regułą strony dłoni całe litery W i P miały odbitą numerację (szablony liter utrwaliły błąd);
  dodana reguła naprawiła W, a P jest teraz odrzucane.

## 4. Wnioski

1. Automatyczna rekonstrukcja daje wiarygodne punkty dla około połowy obrazów i większości liter.
2. Dla liter z dłonią bokiem/w dół (P, Q, G) oraz zakrzywionych (C, O) – mało danych lub ryzyko odbicia.
   Te litery trzeba uzupełnić innym źródłem (zdjęcia + MediaPipe, `extract_landmarks_mediapipe.py`).
3. Po przetworzeniu pełnego datasetu obowiązkowo obejrzyj `qa/` i tabelę kontroli strony dłoni w raporcie.
4. Punkty z rekonstrukcji to punkty MediaPipe zaokrąglone do pikseli płótna 400×400 (błąd ~0,5–1 px,
   ~1 % śródręcza) – nie są pomiarami Apple Vision.
