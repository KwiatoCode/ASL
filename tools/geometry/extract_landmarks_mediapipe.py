"""
Punkty dłoni z PRAWDZIWYCH ZDJĘĆ (alternatywny dataset) – MediaPipe HandLandmarker, uruchamiane na Windowsie.

Do użycia z publicznym zbiorem zdjęć z podfolderami liter, np. ASL-HG (Pranto i in., Data in Brief 2026,
Mendeley Data https://data.mendeley.com/datasets/j4y5w2c8w9/1). Foldery inne niż pojedyncza litera A–Z
(cyfry, „space”, „del”…) są pomijane.

UWAGA: to są punkty MediaPipe, a NIE Apple Vision. Format jest ujednolicony (kolejność stawów MediaPipe = HandJoint),
ale różnice między detektorami pozostają – ocenia je test odporności i test na innym zbiorze w train_geometry_rf.py.

Przygotowanie (raz):
  pip install mediapipe
  Model: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Użycie:
  python extract_landmarks_mediapipe.py --images <folder z podfolderami A…Z> --model hand_landmarker.task \\
         --out landmarks_aslhg.csv --source aslhg [--mirror]

--mirror: odbij poziomo, jeśli zdjęcia NIE są lustrzane (dataset szkieletów i aplikacja pracują w widoku lustrzanym).
Wynik: CSV w formacie rozumianym przez train_geometry_rf.py (coord_space = camera_px – izotropowe piksele).
"""

import argparse
import csv
import os
import re
from glob import glob
from os.path import basename, join

FIELDS = (['source', 'letter', 'file', 'index', 'group', 'status', 'reason', 'coord_space', 'image_w', 'image_h',
           'handedness', 'hand_score'] + [f'{c}{j}' for j in range(21) for c in ('x', 'y')])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', required=True)
    parser.add_argument('--model', required=True, help='plik hand_landmarker.task')
    parser.add_argument('--out', default='landmarks_mediapipe.csv')
    parser.add_argument('--source', default='mediapipe_photos')
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--min-score', type=float, default=0.5)
    args = parser.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=args.min_score, min_hand_presence_confidence=args.min_score)
    landmarker = vision.HandLandmarker.create_from_options(options)

    folders = [f for f in sorted(glob(join(args.images, '**', '*'), recursive=True))
               if os.path.isdir(f) and re.fullmatch(r'[A-Za-z]', basename(f))]
    rows, ok = [], 0
    for folder in folders:
        letter = basename(folder).upper()
        files = sorted(f for f in glob(join(folder, '*')) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
        for n, path in enumerate(files):
            row = {'source': args.source, 'letter': letter, 'file': os.path.relpath(path, args.images),
                   'index': n, 'group': f'{args.source}_{letter}_{n // 100}', 'coord_space': 'camera_px'}
            try:
                image = mp.Image.create_from_file(path)
                result = landmarker.detect(image)
            except Exception as error:      # uszkodzony plik itp.
                row.update(status='odrzucone', reason=f'blad_odczytu: {error}')
                rows.append(row)
                continue
            if not result.hand_landmarks:
                row.update(status='odrzucone', reason='brak_dloni')
                rows.append(row)
                continue
            w, h = image.width, image.height
            hand = result.hand_landmarks[0]
            category = result.handedness[0][0] if result.handedness else None
            row.update(status='ok', reason='', image_w=w, image_h=h,
                       handedness=category.category_name if category else '',
                       hand_score=round(category.score, 4) if category else '')
            for j, lm in enumerate(hand):
                x = (1 - lm.x) if args.mirror else lm.x
                row[f'x{j}'] = round(x * w, 3)
                row[f'y{j}'] = round(lm.y * h, 3)
            rows.append(row)
            ok += 1
        print(f'{letter}: {sum(r["letter"] == letter and r["status"] == "ok" for r in rows)}/{len(files)}', flush=True)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wykryte dłonie: {ok}/{len(rows)} → {args.out}')


if __name__ == '__main__':
    main()
