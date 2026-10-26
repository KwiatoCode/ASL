"""
Buduje ASLClassifier.mlmodel dla aplikacji: obraz szkieletu 400×400 → piksele (siatka 50×50) → Random Forest → A–Z.
Odpowiada notebookowi `random_forest_asl.ipynb` (ResNeXt + regresja logistyczna zastąpione zwykłym Random Forest).

Dwa tryby:

  # 1) Masz już wytrenowany las (pickle/joblib sklearn RandomForestClassifier):
  python build_coreml.py --model random_forest.pkl

  # 2) Wytrenuj od zera na datasecie „American Sign Language (ASL) Hand Skeleton Dataset”:
  python build_coreml.py --train /ścieżka/do/dataset --count 500

W trybie 1 zapisz model w notebooku:  import joblib; joblib.dump(forest, 'random_forest.pkl')
(wersja scikit-learn przy zapisie i tutaj powinna być ta sama).

Wynik: ../ASSN/ASLClassifier.mlmodel (Xcode sam go dołączy i skompiluje).
Usuń stary ../ASSN/ASLClassifier.mlpackage – aplikacja bierze pierwszy model z bundla.

Zależności: pip install coremltools scikit-learn pillow joblib numpy
(konwerter sklearn w coremltools obsługuje tylko niektóre wersje scikit-learn – przy błędzie zainstaluj tę,
którą podaje komunikat coremltools).
"""

import argparse
from glob import glob
from os.path import basename, dirname, join, abspath

import joblib
import numpy as np
from PIL import Image

# Cechy MUSZĄ być liczone identycznie jak w aplikacji (SignClassifier.swift → SkeletonFeatures).
CANVAS = 400            # płótno szkieletu (jak w datasecie i SkeletonImageRenderer)
GRID = 50               # 400 / 50 = bloki 8×8 pikseli
BLOCK = CANVAS // GRID
N_FEATURES = GRID * GRID
LUMA = np.array([0.299, 0.587, 0.114])
INPUT_NAME = 'features'
DEFAULT_OUTPUT = 'ASLClassifier_pixelRF.mlmodel'   # eksperyment archiwalny – nie trafia do aplikacji


def features(image: Image.Image) -> np.ndarray:
    """Obraz → jasność (0…1) uśredniona w blokach 8×8 → wektor 2500 liczb (wiersz po wierszu, od góry)."""
    image = image.convert('RGB')
    if image.size != (CANVAS, CANVAS):
        image = image.resize((CANVAS, CANVAS), Image.BILINEAR)
    gray = np.asarray(image, dtype=np.uint8) @ LUMA / 255.0
    return gray.reshape(GRID, BLOCK, GRID, BLOCK).mean(axis=(1, 3)).ravel()


def train(dataset: str, count: int, trees: int):
    """Jak notebook: max `count` obrazków na literę, podział 75/25 (stratify, seed 2025), RandomForestClassifier."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    rows, labels = [], []
    for folder in sorted(glob(join(dataset, '*'))):
        files = sorted(glob(join(folder, '*.*')))[:count]
        for f in files:
            with Image.open(f) as image:
                rows.append(features(image))
        labels += [basename(folder)] * len(files)
        print(f'{basename(folder)}: {len(files)} obrazków')
    x = np.stack(rows)

    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, test_size=0.25, random_state=2025, stratify=labels)
    forest = RandomForestClassifier(n_estimators=trees, random_state=2025, n_jobs=-1).fit(x_train, y_train)
    predicted = forest.predict(x_test)
    print('accuracy (test): {:.4f}'.format(accuracy_score(y_test, predicted)))
    print('f1 (test): {:.4f}'.format(f1_score(y_test, predicted, average='weighted')))
    joblib.dump(forest, 'random_forest.pkl')
    return forest


def export(forest, output: str):
    import coremltools as ct

    if getattr(forest, 'n_features_in_', N_FEATURES) != N_FEATURES:
        raise SystemExit(f'Las ma {forest.n_features_in_} cech, a aplikacja liczy {N_FEATURES} (siatka {GRID}×{GRID}).')
    labels = [str(c) for c in forest.classes_]
    mlmodel = ct.converters.sklearn.convert(
        forest,
        input_features=[(INPUT_NAME, ct.models.datatypes.Array(N_FEATURES))],
        output_feature_names='classLabel',
    )
    mlmodel.short_description = f'ASL A–Z ze szkieletu dłoni (piksele {GRID}×{GRID} + Random Forest)'
    mlmodel.input_description[INPUT_NAME] = f'Jasność 0…1 płótna {CANVAS}×{CANVAS} uśredniona w blokach {BLOCK}×{BLOCK}'
    mlmodel.save(output)
    print(f'Zapisano {abspath(output)} ({len(labels)} klas: {" ".join(labels)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', help='plik .pkl/.joblib z sklearn RandomForestClassifier')
    source.add_argument('--train', help='folder dataset/ z podfolderami A…Z')
    parser.add_argument('--count', type=int, default=500, help='obrazków na literę (notebook: 500)')
    parser.add_argument('--trees', type=int, default=100, help='liczba drzew (notebook: 100)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    forest = joblib.load(args.model) if args.model else train(args.train, args.count, args.trees)
    export(forest, args.output)
