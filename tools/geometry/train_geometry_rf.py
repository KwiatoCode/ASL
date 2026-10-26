"""
Trening i rzetelna ocena Random Forest na cechach geometrycznych dłoni.

  python train_geometry_rf.py --landmarks wyniki_rekonstrukcji/landmarks.csv --out wyniki_rf --jobs 6

Przebieg:
  1. Wczytanie przyjętych rekonstrukcji (status = ok), przeliczenie płótna → izotropowe piksele kamery.
  2. PODZIAŁ GRUPOWY. Pliki są numerowane 0…999 w obrębie litery, a sąsiednie numery pochodzą najpewniej
     z tego samego nagrania (prawie identyczne klatki). Grupa = litera + blok kolejnych numerów (--block, domyślnie 100).
     Zbiór testowy = 2 z 10 bloków każdej litery (≈ 20 %), nigdy nie używany do strojenia.
     Strojenie: 4-krotna walidacja krzyżowa po blokach (bloki każdej litery rozdzielone między podziały).
  3. Analiza przecieku: ten sam model oceniony przy podziale losowym i grupowym + odległość do najbliższego
     sąsiada (test → trening) w obu podziałach.
  4. Porównanie zestawów cech (basic 63 / extended 110) × strona dłoni (none / canonical) – walidacja krzyżowa.
  5. Strojenie hiperparametrów (siatka) – walidacja krzyżowa.
  6. Model końcowy na wszystkich blokach poza testowymi → ocena na teście: accuracy, precision, recall,
     F1 macro i weighted, wyniki per litera, macierz pomyłek 26×26, najczęściej mylone pary, ważność cech.
  7. Odporność: szum położenia punktów, obrót i skala na zbiorze testowym (symulacja innego detektora).

AUGMENTACJA (--augment N, domyślnie 3): każda próbka TRENINGOWA dostaje N kopii z szumem położenia stawów
(--aug-noise, domyślnie 2,5 % długości śródręcza) i obrotem całej dłoni wokół nadgarstka (--aug-rotate, ±10°).
Kopie powstają wyłącznie z części treningowej danego podziału – walidacja i test zawsze na oryginalnych punktach.
Cel: odporność na różnice MediaPipe ↔ Apple Vision i inne trzymanie dłoni. Obrót jest mały, bo część liter różni się
tylko orientacją (K/P, H/U, G/Q). Skalowanie pominięto – cechy i tak są normalizowane do wielkości dłoni.
Raport porównuje model z augmentacją i bez niej (walidacja, test, odporność). Augmentacja zostaje włączona,
chyba że obniża F1 w walidacji o więcej niż 0,01 (--augment 0 wyłącza ją całkowicie).

Wyniki w --out: model_rf.joblib, feature_spec.json, metrics.json, raport.md, wykresy PNG, cv_*.csv.
"""

import argparse
import csv
import json
import os
import time

import joblib
import numpy as np

from hand_features import FEATURE_NAMES, FEATURE_SETS, FEATURE_VERSION, dataset_to_camera, extract, select, spec

SEED = 2025
BASELINE = dict(n_estimators=300, max_depth=24, min_samples_leaf=2, class_weight='balanced_subsample')
GRID = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 16, 24, 32],
    'min_samples_leaf': [1, 2, 4],
}


def load(paths, block):
    """Jeden lub kilka CSV. Rekonstrukcje szkieletów: piksele płótna 400×400 → izotropowe piksele kamery.
    Zdjęcia przez MediaPipe (coord_space = camera_px): już izotropowe piksele. Zwraca punkty, litery, grupy, źródła."""
    points, letters, groups, sources = [], [], [], []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        for r in csv.DictReader(open(path, encoding='utf-8')):
            if r['status'] != 'ok':
                continue
            xy = np.array([[float(r[f'x{j}']), float(r[f'y{j}'])] for j in range(21)])
            if r.get('coord_space', 'canvas') != 'camera_px':
                xy = dataset_to_camera(xy)
            source = r.get('source') or name
            points.append(xy)
            letters.append(r['letter'].upper())
            groups.append(r.get('group') or f"{source}_{r['letter']}_{int(r['index']) // block}")
            sources.append(source)
    return np.array(points), np.array(letters), np.array(groups), np.array(sources)


def group_split(letters, groups, test_fraction=0.2, folds=4, seed=SEED):
    """Grupa = litera + blok kolejnych numerów plików (lub kolumna `group`). Test: ~20 % grup każdej litery.
    Walidacja: pozostałe grupy każdej litery rozdzielane po kolei między podziały – każda litera jest w treningu
    każdego podziału, a żadna grupa nie jest jednocześnie w treningu i walidacji/teście."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(letters), bool)
    fold = np.full(len(letters), -1)
    for L in np.unique(letters):
        bl = rng.permutation(np.unique(groups[letters == L]))
        n_test = max(1, int(round(test_fraction * len(bl)))) if len(bl) > 1 else 0
        test |= (letters == L) & np.isin(groups, bl[:n_test])
        for k, b in enumerate(bl[n_test:]):
            fold[(letters == L) & (groups == b)] = k % folds
    return test, fold


def make_rf(params, jobs):
    from sklearn.ensemble import RandomForestClassifier
    p = dict(BASELINE)
    p.update(params)
    return RandomForestClassifier(random_state=SEED, n_jobs=jobs, **p)


def training_set(mask, x, y, xa=None):
    """Cechy treningowe dla próbek `mask` + (opcjonalnie) ich augmentowane kopie xa[mask] (n, N, F)."""
    if xa is None:
        return x[mask], y[mask]
    copies = xa[mask]
    return (np.concatenate([x[mask], copies.reshape(-1, copies.shape[-1])]),
            np.concatenate([y[mask], np.repeat(y[mask], copies.shape[1])]))


def cv_score(x, y, fold, params, jobs, xa=None):
    from sklearn.metrics import f1_score
    scores = []
    for k in np.unique(fold[fold >= 0]):
        tr, va = (fold >= 0) & (fold != k), fold == k
        model = make_rf(params, jobs).fit(*training_set(tr, x, y, xa))
        scores.append(f1_score(y[va], model.predict(x[va]), average='macro'))
    return float(np.mean(scores)), float(np.std(scores))


def augment(points, rng, noise=0.0, rotate_deg=0.0, scale=0.0):
    """Zaburzenie punktów (izotropowe piksele): szum ∝ śródręcze, obrót wokół nadgarstka, skala."""
    p = points.copy()
    palm = np.mean(np.linalg.norm(p[:, [5, 9, 13, 17]] - p[:, :1], axis=-1), axis=1)
    if noise:
        p += rng.normal(0, 1, p.shape) * (noise * palm)[:, None, None]
    if rotate_deg:
        a = np.deg2rad(rng.uniform(-rotate_deg, rotate_deg, len(p)))
        c, s = np.cos(a), np.sin(a)
        rel = p - p[:, :1]
        p = p[:, :1] + np.stack([c[:, None] * rel[..., 0] - s[:, None] * rel[..., 1],
                                 s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]], -1)
    if scale:
        p = p * rng.uniform(1 - scale, 1 + scale, len(p))[:, None, None]
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--landmarks', required=True, nargs='+', help='jeden lub kilka plików CSV z punktami')
    parser.add_argument('--out', default='wyniki_rf')
    parser.add_argument('--block', type=int, default=100, help='rozmiar bloku kolejnych numerów plików (grupa)')
    parser.add_argument('--jobs', type=int, default=-1)
    parser.add_argument('--quick', action='store_true', help='mała siatka – szybki test')
    parser.add_argument('--min-per-letter', type=int, default=20, help='litery z mniejszą liczbą próbek są pomijane')
    parser.add_argument('--augment', type=int, default=3, help='kopii augmentowanych na próbkę treningową (0 = brak)')
    parser.add_argument('--aug-noise', type=float, default=0.025, help='szum położenia stawów (ułamek śródręcza)')
    parser.add_argument('--aug-rotate', type=float, default=10.0, help='maks. obrót dłoni w stopniach (±)')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    from sklearn.model_selection import train_test_split

    os.makedirs(args.out, exist_ok=True)
    started = time.time()
    points, y, groups, sources = load(args.landmarks, args.block)
    names, counts = np.unique(y, return_counts=True)
    too_few = [str(n) for n, c in zip(names, counts) if c < args.min_per_letter]
    if too_few:
        print(f'Pominięte litery (mniej niż {args.min_per_letter} próbek): {", ".join(too_few)}')
        keep = ~np.isin(y, too_few)
        points, y, groups, sources = points[keep], y[keep], groups[keep], sources[keep]
    labels = sorted(np.unique(y))
    print(f'Próbek: {len(y)}, liter: {len(labels)}, źródła: {dict(zip(*np.unique(sources, return_counts=True)))}')
    test, fold = group_split(y, groups)
    train = ~test
    print(f'Trening+walidacja: {train.sum()}, test: {test.sum()} (bloki po {args.block} numerów)')
    feats = {ch: extract(points, ch) for ch in ('none', 'canonical')}
    aug_feats = {}
    if args.augment > 0:
        aug_rng = np.random.default_rng(SEED + 1)
        aug_points = np.stack([augment(points, aug_rng, noise=args.aug_noise, rotate_deg=args.aug_rotate)
                               for _ in range(args.augment)], axis=1)                 # (n, N, 21, 2)
        flat = aug_points.reshape(-1, 21, 2)
        aug_feats = {ch: extract(flat, ch).reshape(len(points), args.augment, -1) for ch in ('none', 'canonical')}
        print(f'Augmentacja: {args.augment} kopie na próbkę treningową (szum {100 * args.aug_noise:.1f}% śródręcza, '
              f'obrót ±{args.aug_rotate:g}°)')
    report = {'samples': int(len(y)), 'letters': labels, 'excluded_letters': too_few,
              'feature_version': FEATURE_VERSION,
              'block': args.block, 'train': int(train.sum()), 'test': int(test.sum())}

    # ---------- 3. przeciek danych
    x = select(feats['none'], 'extended')
    xr_tr, xr_te, yr_tr, yr_te = train_test_split(x, y, test_size=float(test.mean()), random_state=SEED, stratify=y)
    m_rand = make_rf({}, args.jobs).fit(xr_tr, yr_tr)
    m_group = make_rf({}, args.jobs).fit(x[train], y[train])
    f1_rand = f1_score(yr_te, m_rand.predict(xr_te), average='macro')
    f1_group = f1_score(y[test], m_group.predict(x[test]), average='macro')

    def nn_dist(a, b):
        d = []
        for i in range(0, len(a), 256):
            dd = ((a[i:i + 256, None, :] - b[None]) ** 2).sum(-1)
            d.append(np.sqrt(dd.min(1)))
        return float(np.median(np.concatenate(d)))
    leak = {'f1_macro_random_split': float(f1_rand), 'f1_macro_group_split': float(f1_group),
            'nn_dist_random_split': nn_dist(xr_te, xr_tr), 'nn_dist_group_split': nn_dist(x[test], x[train])}
    report['leakage'] = leak
    print(f'Przeciek: F1 losowy {f1_rand:.4f} vs grupowy {f1_group:.4f}; '
          f'odl. NN {leak["nn_dist_random_split"]:.3f} vs {leak["nn_dist_group_split"]:.3f}')

    # ---------- 4. zestawy cech × strona dłoni
    configs = []
    for fs in ('basic', 'extended'):
        for ch in ('none', 'canonical'):
            mean, sd = cv_score(select(feats[ch], fs), y, fold, {}, args.jobs)
            configs.append({'feature_set': fs, 'chirality': ch, 'n_features': len(FEATURE_SETS[fs]),
                            'cv_f1_macro': mean, 'cv_sd': sd})
            print(f'  {fs:8s} {ch:9s}: F1 macro (CV) {mean:.4f} ± {sd:.4f}')
    best_cfg = max(configs, key=lambda c: c['cv_f1_macro'])
    report['feature_configs'] = configs
    fs, ch = best_cfg['feature_set'], best_cfg['chirality']
    x = select(feats[ch], fs)
    xa_all = select(aug_feats[ch], fs) if aug_feats else None

    # ---------- 4b. augmentacja: czy nie szkodzi w walidacji (zysk widać w teście odporności)
    report['augmentation'] = {'copies': args.augment, 'noise': args.aug_noise, 'rotate_deg': args.aug_rotate}
    use_aug = False
    if xa_all is not None:
        base_cv = cv_score(x, y, fold, {}, args.jobs)
        aug_cv = cv_score(x, y, fold, {}, args.jobs, xa_all)
        use_aug = aug_cv[0] >= base_cv[0] - 0.01
        report['augmentation'].update({'cv_without': base_cv, 'cv_with': aug_cv, 'used': use_aug})
        print(f'Augmentacja w walidacji: bez {base_cv[0]:.4f} ± {base_cv[1]:.4f}, z {aug_cv[0]:.4f} ± {aug_cv[1]:.4f}'
              f' → {"włączona" if use_aug else "wyłączona"}')
    xa = xa_all if use_aug else None
    # ---------- 5. strojenie
    grid = {'n_estimators': [100, 300], 'max_depth': [None, 24], 'min_samples_leaf': [1, 2]} if args.quick else GRID
    combos = [dict(n_estimators=n, max_depth=d, min_samples_leaf=l)
              for n in grid['n_estimators'] for d in grid['max_depth'] for l in grid['min_samples_leaf']]
    tuning = []
    for params in combos:
        mean, sd = cv_score(x, y, fold, params, args.jobs, xa)
        tuning.append({**params, 'cv_f1_macro': mean, 'cv_sd': sd})
        print(f'  {params}: {mean:.4f} ± {sd:.4f}')
    tuning.sort(key=lambda t: -t['cv_f1_macro'])
    best = {k: tuning[0][k] for k in ('n_estimators', 'max_depth', 'min_samples_leaf')}
    report['tuning'] = tuning
    report['best_params'] = {**BASELINE, **best}
    with open(os.path.join(args.out, 'cv_strojenie.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(tuning[0]))
        w.writeheader()
        w.writerows(tuning)

    # ---------- 6. model końcowy i test
    model = make_rf(best, args.jobs).fit(*training_set(train, x, y, xa))
    pred = model.predict(x[test])
    prec, rec, f1s, sup = precision_recall_fscore_support(y[test], pred, labels=labels, zero_division=0)
    metrics = {
        'accuracy': float(accuracy_score(y[test], pred)),
        'f1_macro': float(f1_score(y[test], pred, average='macro')),
        'f1_weighted': float(f1_score(y[test], pred, average='weighted')),
        'precision_macro': float(prec.mean()), 'recall_macro': float(rec.mean()),
        'per_letter': {L: {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
                       for L, p, r, f, s in zip(labels, prec, rec, f1s, sup)},
    }
    cm = confusion_matrix(y[test], pred, labels=labels)
    pairs = sorted([(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels))
                    if i != j and cm[i, j]], key=lambda t: -t[2])
    metrics['confused_pairs'] = pairs[:15]
    report['test'] = metrics
    print(f'TEST: accuracy {metrics["accuracy"]:.4f}, F1 macro {metrics["f1_macro"]:.4f}, '
          f'F1 weighted {metrics["f1_weighted"]:.4f}')
    open(os.path.join(args.out, 'raport_klasyfikacji.txt'), 'w', encoding='utf-8').write(
        classification_report(y[test], pred, labels=labels, digits=4, zero_division=0))

    plt.figure(figsize=(10, 9))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar(label='liczba próbek')
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('przewidziana litera')
    plt.ylabel('prawdziwa litera')
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=6,
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.title(f'Macierz pomyłek – zbiór testowy (accuracy {metrics["accuracy"]:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'macierz_pomylek.png'), dpi=150)
    plt.close()

    names = FEATURE_SETS[fs]
    imp = model.feature_importances_
    perm = permutation_importance(model, x[test], y[test], n_repeats=5, random_state=SEED, n_jobs=args.jobs,
                                  scoring='f1_macro')
    order = np.argsort(perm.importances_mean)[::-1][:25]
    report['feature_importance'] = [{'feature': names[i], 'permutation': float(perm.importances_mean[i]),
                                     'impurity': float(imp[i])} for i in order]
    plt.figure(figsize=(8, 7))
    plt.barh([names[i] for i in order][::-1], perm.importances_mean[order][::-1], color='#4C72B0')
    plt.xlabel('spadek F1 macro po permutacji cechy (zbiór testowy)')
    plt.title('25 najważniejszych cech')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'waznosc_cech.png'), dpi=150)
    plt.close()

    # ---------- 7. odporność
    perturbations = [('bez zaburzeń', {}), ('szum 2% śródręcza', {'noise': 0.02}), ('szum 5%', {'noise': 0.05}),
                     ('szum 10%', {'noise': 0.10}), ('obrót ±10°', {'rotate_deg': 10}),
                     ('obrót ±20°', {'rotate_deg': 20}), ('skala ±30%', {'scale': 0.3})]
    rng = np.random.default_rng(SEED)
    perturbed = [select(extract(augment(points[test], rng, **kw), ch), fs) for _, kw in perturbations]
    # porównawczy model w drugim wariancie (z augmentacją ↔ bez), te same parametry i ten sam test
    other = None
    if xa_all is not None:
        other = make_rf(best, args.jobs).fit(*training_set(train, x, y, None if use_aug else xa_all))
    robust = []
    for (name, _), xp in zip(perturbations, perturbed):
        row = {'perturbation': name, 'accuracy': float(accuracy_score(y[test], model.predict(xp)))}
        if other is not None:
            row['accuracy_other'] = float(accuracy_score(y[test], other.predict(xp)))
        robust.append(row)
    report['robustness'] = robust
    report['robustness_model'] = 'z augmentacją' if use_aug else 'bez augmentacji'
    report['robustness_other'] = None if other is None else ('bez augmentacji' if use_aug else 'z augmentacją')
    if other is not None:
        p_other = other.predict(x[test])
        report['augmentation']['test_other'] = {'accuracy': float(accuracy_score(y[test], p_other)),
                                                'f1_macro': float(f1_score(y[test], p_other, average='macro'))}
    print('Odporność:', ', '.join(f"{r['perturbation']} {r['accuracy']:.3f}" for r in robust))

    # ---------- 8. przenoszenie między zbiorami (gdy jest więcej niż jedno źródło)
    report['cross_source'] = []
    for held in np.unique(sources):
        if len(np.unique(sources)) < 2:
            break
        tr, te = sources != held, sources == held
        common = np.intersect1d(np.unique(y[tr]), np.unique(y[te]))
        te &= np.isin(y, common)
        m = make_rf(best, args.jobs).fit(*training_set(tr, x, y, xa))
        p_ = m.predict(x[te])
        res = {'train_on': ', '.join(np.unique(sources[tr])), 'test_on': str(held), 'letters': int(len(common)),
               'samples': int(te.sum()), 'accuracy': float(accuracy_score(y[te], p_)),
               'f1_macro': float(f1_score(y[te], p_, average='macro'))}
        report['cross_source'].append(res)
        print(f"Przenoszenie {res['train_on']} → {held}: accuracy {res['accuracy']:.4f}, F1 macro {res['f1_macro']:.4f}")

    # ---------- zapis
    # model do aplikacji: wszystkie przyjęte próbki (+ ich kopie, jeśli augmentacja jest włączona)
    final = make_rf(best, args.jobs).fit(*training_set(np.ones(len(y), bool), x, y, xa))
    joblib.dump({'model': final, 'feature_set': fs, 'chirality': ch, 'feature_version': FEATURE_VERSION,
                 'augmentation': report['augmentation']},
                os.path.join(args.out, 'model_rf.joblib'))
    joblib.dump({'model': model, 'feature_set': fs, 'chirality': ch, 'feature_version': FEATURE_VERSION,
                 'x_test': x[test], 'y_test': y[test]}, os.path.join(args.out, 'model_rf_ocena.joblib'))
    with open(os.path.join(args.out, 'feature_spec.json'), 'w', encoding='utf-8') as f:
        json.dump(spec(fs, ch), f, ensure_ascii=False, indent=1)
    report['minutes'] = (time.time() - started) / 60
    with open(os.path.join(args.out, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    write_markdown(report, os.path.join(args.out, 'raport.md'))
    print(f'Gotowe w {report["minutes"]:.1f} min → {args.out}')


def write_markdown(r, path):
    t = r['test']
    L = ['# Random Forest na cechach geometrycznych – raport', '',
         f"Próbek (przyjęte rekonstrukcje): {r['samples']}, liter: {len(r['letters'])}, cechy: {r['feature_version']}.",
         f"Podział grupowy: bloki po {r['block']} kolejnych numerów plików; trening {r['train']}, test {r['test']}.",
         (f"**Litery pominięte (za mało próbek, model ich NIE rozpoznaje): {', '.join(r['excluded_letters'])}**"
          if r.get('excluded_letters') else 'Wszystkie litery z danych mają dość próbek.'), '',
         '## Wynik na zbiorze testowym', '',
         '| accuracy | precision (macro) | recall (macro) | F1 macro | F1 weighted |', '|---|---|---|---|---|',
         f"| {t['accuracy']:.4f} | {t['precision_macro']:.4f} | {t['recall_macro']:.4f} | {t['f1_macro']:.4f} | {t['f1_weighted']:.4f} |",
         '', f"Parametry: `{r['best_params']}`", '',
         '## Przeciek danych', '',
         '| podział | F1 macro (test) | mediana odległości test → najbliższy trening |', '|---|---|---|',
         f"| losowy | {r['leakage']['f1_macro_random_split']:.4f} | {r['leakage']['nn_dist_random_split']:.3f} |",
         f"| grupowy (bloki) | {r['leakage']['f1_macro_group_split']:.4f} | {r['leakage']['nn_dist_group_split']:.3f} |",
         '', 'Wyraźnie wyższy wynik przy podziale losowym = sąsiednie klatki tego samego nagrania w treningu i teście.', '',
         '## Zestawy cech (walidacja krzyżowa po blokach)', '',
         '| zestaw | strona dłoni | cech | F1 macro (CV) |', '|---|---|---|---|']
    L += [f"| {c['feature_set']} | {c['chirality']} | {c['n_features']} | {c['cv_f1_macro']:.4f} ± {c['cv_sd']:.4f} |"
          for c in r['feature_configs']]
    a = r.get('augmentation', {})
    if a.get('copies'):
        L += ['', '## Augmentacja danych treningowych', '',
              f"{a['copies']} kopie każdej próbki treningowej: szum położenia stawów {100 * a['noise']:.1f} % śródręcza, "
              f"obrót dłoni ±{a['rotate_deg']:g}°. Walidacja i test zawsze na oryginalnych punktach.", '']
        if 'cv_with' in a:
            L += ['| wariant | F1 macro (CV) |', '|---|---|',
                  f"| bez augmentacji | {a['cv_without'][0]:.4f} ± {a['cv_without'][1]:.4f} |",
                  f"| z augmentacją | {a['cv_with'][0]:.4f} ± {a['cv_with'][1]:.4f} |", '',
                  f"Model końcowy: **{'z augmentacją' if a['used'] else 'bez augmentacji'}** "
                  '(augmentacja zostaje, jeśli nie obniża F1 w walidacji o więcej niż 0,01; '
                  'jej zysk widać w tabeli odporności).']
            if 'test_other' in a:
                L += ['', f"Na zbiorze testowym wariant {'bez augmentacji' if a['used'] else 'z augmentacją'}: "
                      f"accuracy {a['test_other']['accuracy']:.4f}, F1 macro {a['test_other']['f1_macro']:.4f} "
                      f"(model końcowy: {t['accuracy']:.4f} / {t['f1_macro']:.4f})."]
    L += ['', '## Strojenie (10 najlepszych, CV)', '', '| n_estimators | max_depth | min_samples_leaf | F1 macro (CV) |',
          '|---|---|---|---|']
    L += [f"| {t_['n_estimators']} | {t_['max_depth']} | {t_['min_samples_leaf']} | {t_['cv_f1_macro']:.4f} ± {t_['cv_sd']:.4f} |"
          for t_ in r['tuning'][:10]]
    L += ['', '## Wyniki per litera (test)', '', '| litera | precision | recall | F1 | próbek |', '|---|---|---|---|---|']
    L += [f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
          for k, v in t['per_letter'].items()]
    L += ['', '## Najczęściej mylone pary', '', '| prawdziwa | przewidziana | liczba |', '|---|---|---|']
    L += [f'| {a} | {b} | {n} |' for a, b, n in t['confused_pairs']]
    L += ['', '## Najważniejsze cechy (permutacja na teście)', '', '| cecha | spadek F1 | ważność (impurity) |', '|---|---|---|']
    L += [f"| {f['feature']} | {f['permutation']:.4f} | {f['impurity']:.4f} |" for f in r['feature_importance'][:15]]
    L += ['', '## Odporność na zaburzenia punktów (symulacja innego detektora)', '']
    if r.get('robustness_other'):
        L += [f"| zaburzenie | accuracy – {r['robustness_model']} (model końcowy) | accuracy – {r['robustness_other']} |",
              '|---|---|---|']
        L += [f"| {x['perturbation']} | {x['accuracy']:.4f} | {x['accuracy_other']:.4f} |" for x in r['robustness']]
    else:
        L += ['| zaburzenie | accuracy |', '|---|---|']
        L += [f"| {x['perturbation']} | {x['accuracy']:.4f} |" for x in r['robustness']]
    if r.get('cross_source'):
        L += ['', '## Przenoszenie między zbiorami (trening na jednym źródle, test na drugim)', '',
              '| trening | test | liter | próbek | accuracy | F1 macro |', '|---|---|---|---|---|---|']
        L += [f"| {c['train_on']} | {c['test_on']} | {c['letters']} | {c['samples']} | {c['accuracy']:.4f} | {c['f1_macro']:.4f} |"
              for c in r['cross_source']]
    L += ['', '## Ograniczenia', '',
          '- Dane treningowe to punkty MediaPipe odtworzone z rysunków, a nie pomiary Apple Vision.',
          '- J i Z to znaki ruchome; model widzi tylko pojedynczą klatkę.',
          '- Wynik na teście dotyczy tego samego zbioru (te same osoby/nagrania); skuteczność na żywo trzeba zmierzyć osobno.',
          '![Macierz pomyłek](macierz_pomylek.png)', '![Ważność cech](waznosc_cech.png)', '']
    open(path, 'w', encoding='utf-8').write('\n'.join(L))


if __name__ == '__main__':
    main()
