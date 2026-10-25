"""
Strojenie hiperparametrów Random Forest (GridSearchCV) + wyniki do sprawozdania.

  python tune_random_forest.py --train /ścieżka/do/dataset            # pełna siatka (96 zestawów × 5-fold CV)
  python tune_random_forest.py --train /ścieżka/do/dataset --quick    # mała siatka do szybkiego testu
  python tune_random_forest.py --train /ścieżka/do/dataset --export   # + od razu ASLClassifier.mlmodel z najlepszego lasu

Przebieg (te same dane i podział co notebook i build_coreml.py):
  500 obrazków / literę → cechy pikselowe 50×50 → podział 75/25 (stratify, seed 2025)
  → GridSearchCV na części treningowej (5-fold, metryka F1 ważone)
  → najlepszy zestaw trenowany na całej części treningowej → ocena na części testowej (niewidzianej przy strojeniu)
  → porównanie z lasem o domyślnych parametrach.

Wyniki w folderze `wyniki_strojenia/`:
  podsumowanie.md            najlepsze parametry, porównanie z domyślnym lasem, top 10 – gotowe do wklejenia
  wszystkie_wyniki.csv       każdy zestaw: parametry, F1/accuracy CV (średnia ± odch.), czas treningu
  raport_klasyfikacji.txt    precision / recall / F1 dla każdej litery (zbiór testowy)
  wykres_liczba_drzew.png    F1 (CV) w zależności od liczby drzew, osobno dla max_depth
  wykres_heatmapa.png        F1 (CV): max_depth × max_features
  wykres_min_samples_leaf.png
  macierz_pomylek.png        macierz pomyłek najlepszego lasu na zbiorze testowym
  random_forest.pkl          najlepszy las (do build_coreml.py --model)

Zależności: pip install scikit-learn pillow joblib numpy matplotlib
"""

import argparse
import csv
import os
import time
from glob import glob
from os.path import basename, join

import joblib
import numpy as np
from PIL import Image

from build_coreml import features, export, DEFAULT_OUTPUT, GRID

SEED = 2025

FULL_GRID = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4],
}
QUICK_GRID = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1],
}
PARAMS = list(FULL_GRID)


def load(dataset: str, count: int, cache: str):
    """Cechy wszystkich obrazków; zapisuje je w pliku .npz, żeby kolejne uruchomienia nie czytały obrazków od nowa."""
    if os.path.exists(cache):
        data = np.load(cache)
        if data['x'].shape[0] > 0 and int(data['count']) == count:
            print(f'Cechy wczytane z {cache}')
            return data['x'], data['y'].astype(str)
    folders = [(basename(folder), sorted(glob(join(folder, '*.*')))[:count])
               for folder in sorted(glob(join(dataset, '*')))]
    # Jedna tablica float32 od razu na wszystkie obrazki (las i tak liczy na float32) – mało pamięci.
    x = np.empty((sum(len(files) for _, files in folders), GRID * GRID), dtype=np.float32)
    labels, row = [], 0
    for letter, files in folders:
        for f in files:
            with Image.open(f) as image:
                x[row] = features(image)
            row += 1
        labels += [letter] * len(files)
        print(f'{letter}: {len(files)} obrazków')
    y = np.array(labels)
    np.savez_compressed(cache, x=x, y=y, count=count)
    return x, y


def show(value):
    return 'brak (None)' if value is None else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', required=True, help='folder dataset/ z podfolderami A…Z')
    parser.add_argument('--count', type=int, default=500, help='obrazków na literę (notebook: 500)')
    parser.add_argument('--folds', type=int, default=5, help='liczba podziałów w walidacji krzyżowej')
    parser.add_argument('--quick', action='store_true', help='mała siatka do szybkiego testu')
    parser.add_argument('--jobs', type=int, default=-1,
                        help='ile treningów naraz (-1 = wszystkie rdzenie; przy braku pamięci np. 2)')
    parser.add_argument('--out', default='wyniki_strojenia')
    parser.add_argument('--export', action='store_true', help='zapisz też ASLClassifier.mlmodel z najlepszego lasu')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='ścieżka pliku .mlmodel (z --export)')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

    os.makedirs(args.out, exist_ok=True)
    x, y = load(args.train, args.count, join(args.out, f'cechy_{GRID}x{GRID}_{args.count}.npz'))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)
    print(f'Zbiór treningowy: {len(y_train)}, testowy: {len(y_test)}, cech: {x.shape[1]}')

    grid = QUICK_GRID if args.quick else FULL_GRID
    combos = int(np.prod([len(v) for v in grid.values()]))
    print(f'Strojenie: {combos} zestawów × {args.folds} podziałów = {combos * args.folds} treningów')

    search = GridSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        grid,
        scoring={'f1': 'f1_weighted', 'accuracy': 'accuracy'},
        refit='f1',
        cv=StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED),
        n_jobs=args.jobs,
        pre_dispatch='n_jobs',
        verbose=1,
    )
    started = time.time()
    search.fit(x_train, y_train)
    tuning_minutes = (time.time() - started) / 60
    best = search.best_estimator_

    # Punkt odniesienia: las z domyślnymi parametrami (jak w notebooku).
    baseline = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=args.jobs).fit(x_train, y_train)

    def evaluate(model):
        predicted = model.predict(x_test)
        return accuracy_score(y_test, predicted), f1_score(y_test, predicted, average='weighted'), predicted

    best_acc, best_f1, best_pred = evaluate(best)
    base_acc, base_f1, _ = evaluate(baseline)

    # ---------- tabela wszystkich wyników ----------
    r = search.cv_results_
    order = np.argsort(r['rank_test_f1'])
    rows = []
    for i in order:
        p = r['params'][i]
        rows.append({
            'ranking': int(r['rank_test_f1'][i]),
            **{k: show(p[k]) for k in PARAMS},
            'f1_cv_srednia': round(float(r['mean_test_f1'][i]), 4),
            'f1_cv_odch_std': round(float(r['std_test_f1'][i]), 4),
            'accuracy_cv_srednia': round(float(r['mean_test_accuracy'][i]), 4),
            'czas_treningu_s': round(float(r['mean_fit_time'][i]), 2),
        })
    with open(join(args.out, 'wszystkie_wyniki.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=';')
        writer.writeheader()
        # Polski Excel: separator ';' i przecinek dziesiętny.
        writer.writerows({k: str(v).replace('.', ',') if isinstance(v, float) else v for k, v in row.items()}
                         for row in rows)

    with open(join(args.out, 'raport_klasyfikacji.txt'), 'w', encoding='utf-8') as f:
        f.write('Najlepszy Random Forest – zbiór testowy\n')
        f.write(f'Parametry: {search.best_params_}\n\n')
        f.write(classification_report(y_test, best_pred, digits=4))

    # ---------- wykresy ----------
    def mean_f1(**fixed):
        """Średnie F1 (CV) z zestawów spełniających warunki – uśrednione po pozostałych parametrach."""
        mask = np.ones(len(r['params']), bool)
        for key, value in fixed.items():
            mask &= np.array([p[key] == value for p in r['params']])
        return float(np.mean(r['mean_test_f1'][mask]))

    plt.figure(figsize=(7, 4.5))
    for depth in grid['max_depth']:
        plt.plot(grid['n_estimators'], [mean_f1(n_estimators=n, max_depth=depth) for n in grid['n_estimators']],
                 marker='o', label=f'max_depth = {show(depth)}')
    plt.xlabel('liczba drzew (n_estimators)')
    plt.ylabel('F1 ważone (walidacja krzyżowa)')
    plt.title('Wpływ liczby drzew i głębokości')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(args.out, 'wykres_liczba_drzew.png'), dpi=150)
    plt.close()

    heat = np.array([[mean_f1(max_depth=d, max_features=m) for m in grid['max_features']] for d in grid['max_depth']])
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(heat, cmap='viridis', aspect='auto')
    plt.colorbar(label='F1 ważone (CV)')
    plt.xticks(range(len(grid['max_features'])), grid['max_features'])
    plt.yticks(range(len(grid['max_depth'])), [show(d) for d in grid['max_depth']])
    plt.xlabel('max_features')
    plt.ylabel('max_depth')
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            plt.text(j, i, f'{heat[i, j]:.4f}', ha='center', va='center',
                     color='white' if heat[i, j] < heat.mean() else 'black')
    plt.title('max_depth × max_features')
    plt.tight_layout()
    plt.savefig(join(args.out, 'wykres_heatmapa.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(5.5, 4))
    leaves = grid['min_samples_leaf']
    values = [mean_f1(min_samples_leaf=v) for v in leaves]
    bars = plt.bar([str(v) for v in leaves], values, color='#4C72B0')
    plt.bar_label(bars, labels=[f'{v:.4f}' for v in values])
    plt.ylim(max(0, min(values) - 0.02), min(1, max(values) + 0.01))
    plt.xlabel('min_samples_leaf')
    plt.ylabel('F1 ważone (CV)')
    plt.title('Minimalna liczba próbek w liściu')
    plt.tight_layout()
    plt.savefig(join(args.out, 'wykres_min_samples_leaf.png'), dpi=150)
    plt.close()

    labels = list(best.classes_)
    matrix = confusion_matrix(y_test, best_pred, labels=labels)
    plt.figure(figsize=(9, 8))
    plt.imshow(matrix, cmap='Blues')
    plt.colorbar(label='liczba obrazków')
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('przewidziana litera')
    plt.ylabel('prawdziwa litera')
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j]:
                plt.text(j, i, matrix[i, j], ha='center', va='center', fontsize=6,
                         color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    plt.title(f'Macierz pomyłek – najlepszy las (test, accuracy {best_acc:.4f})')
    plt.tight_layout()
    plt.savefig(join(args.out, 'macierz_pomylek.png'), dpi=150)
    plt.close()

    # ---------- podsumowanie ----------
    errors = [(labels[i], labels[j], int(matrix[i, j]))
              for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i, j]]
    errors.sort(key=lambda e: -e[2])

    header = '| # | ' + ' | '.join(PARAMS) + ' | F1 CV | ± odch. | accuracy CV | czas treningu [s] |'
    lines = [
        '# Strojenie hiperparametrów Random Forest',
        '',
        f'- Dane: {len(y)} obrazków ({args.count} na literę), cechy: piksele {GRID}×{GRID} = {x.shape[1]}',
        f'- Podział: {len(y_train)} treningowych / {len(y_test)} testowych (75/25, stratify, seed {SEED})',
        f'- Metoda: GridSearchCV, {args.folds}-krotna walidacja krzyżowa na zbiorze treningowym, metryka: F1 ważone',
        f'- Przetestowano {combos} zestawów parametrów ({combos * args.folds} treningów), czas: {tuning_minutes:.1f} min',
        '',
        '## Siatka parametrów',
        '',
        '| parametr | wartości |',
        '|---|---|',
        *[f'| {k} | {", ".join(show(v) for v in values)} |' for k, values in grid.items()],
        '',
        '## Najlepszy zestaw',
        '',
        '| parametr | wartość |',
        '|---|---|',
        *[f'| {k} | {show(search.best_params_[k])} |' for k in PARAMS],
        '',
        '## Porównanie na zbiorze testowym',
        '',
        '| model | accuracy | F1 ważone |',
        '|---|---|---|',
        f'| Random Forest – domyślne parametry (100 drzew) | {base_acc:.4f} | {base_f1:.4f} |',
        f'| Random Forest – po strojeniu | {best_acc:.4f} | {best_f1:.4f} |',
        '',
        f'F1 najlepszego zestawu w walidacji krzyżowej: {search.best_score_:.4f}',
        '',
        '## 10 najlepszych zestawów (walidacja krzyżowa)',
        '',
        header,
        '|' + '---|' * (len(PARAMS) + 5),
        *[f"| {row['ranking']} | " + ' | '.join(row[k] for k in PARAMS)
          + f" | {row['f1_cv_srednia']:.4f} | {row['f1_cv_odch_std']:.4f} | {row['accuracy_cv_srednia']:.4f}"
          + f" | {row['czas_treningu_s']} |" for row in rows[:10]],
        '',
        '## Najczęstsze pomyłki (zbiór testowy)',
        '',
        '| prawdziwa | przewidziana | liczba |',
        '|---|---|---|',
        *([f'| {a} | {b} | {n} |' for a, b, n in errors[:10]] or ['| – | – | brak pomyłek |']),
        '',
        '## Wykresy',
        '',
        '![Liczba drzew](wykres_liczba_drzew.png)',
        '![max_depth × max_features](wykres_heatmapa.png)',
        '![min_samples_leaf](wykres_min_samples_leaf.png)',
        '![Macierz pomyłek](macierz_pomylek.png)',
        '',
    ]
    with open(join(args.out, 'podsumowanie.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    joblib.dump(best, join(args.out, 'random_forest.pkl'))

    print()
    print(f'Najlepsze parametry: {search.best_params_}')
    print(f'F1 (CV): {search.best_score_:.4f}')
    print(f'Test – domyślny las:   accuracy {base_acc:.4f}  F1 {base_f1:.4f}')
    print(f'Test – po strojeniu:   accuracy {best_acc:.4f}  F1 {best_f1:.4f}')
    print(f'Wyniki zapisane w folderze: {os.path.abspath(args.out)}')

    if args.export:
        export(best, args.output)


if __name__ == '__main__':
    main()
