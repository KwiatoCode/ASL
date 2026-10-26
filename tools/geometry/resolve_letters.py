"""
Etap 2 rekonstrukcji: rozstrzyganie niejednoznacznych numeracji stawów na poziomie litery.

Pojedynczy rysunek szkieletu bywa niejednoznaczny (palce zgięte w pięść, punkty na jednej prostej, odbicie
wskazujący ↔ mały palec). Etap 1 (`reconstruct_skeleton.py`) zapisuje dla każdego obrazu kilka najlepszych
wariantów numeracji wraz z wynikiem dopasowania do rysunku. Tutaj – iteracja w stylu EM, osobno dla każdej litery:

  start:  każdy obraz bierze swój najlepszy wariant z etapu 1,
  krok:   model kształtu litery = wybrane warianty pozostałych obrazów tej litery
          (kształt = stawy względem nadgarstka, skala = długość śródręcza, bez obrotu – orientacja należy do znaku);
          każdy obraz wybiera wariant maksymalizujący:
              dopasowanie do rysunku  −  ½ · (odległość od 3 najbliższych kształtów litery / σ_litery)²
  koniec: obraz przyjęty, gdy jakość rysunku jest dobra, zwycięski wariant wyraźnie wygrywa (przewaga ≥ 3)
          i kształt jest typowy dla litery (≤ 3 σ). Pozostałe – odrzucone, nie zgadywane.

Walidacja na syntetycznych „literach” ze znaną prawdą: tests/eval_letters_synthetic.py.

  python resolve_letters.py --alternatives landmarks_alternatives.npz --out landmarks_resolved.csv
"""

import argparse
import csv

import numpy as np

ASPECT_X = 640 / 480
MIN_FINAL_MARGIN = 3.0
K_NEIGHBOURS = 3
MAX_SHAPE_SIGMAS = 3.0
ITERATIONS = 6
TWIN = [0, 1, 2, 3, 4, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8]


def shape(xy):
    p = np.asarray(xy, float).copy()
    p[..., 0] *= ASPECT_X
    rel = p - p[..., :1, :]
    scale = np.mean(np.linalg.norm(rel[..., [5, 9, 13, 17], :], axis=-1), axis=-1)
    return (rel / np.maximum(scale, 1e-6)[..., None, None]).reshape(*p.shape[:-2], 42)


def rms(a, b):
    """Średnia kwadratowa odległość stawów między kształtami (w jednostkach śródręcza)."""
    d = (a - b) ** 2
    return np.sqrt(d.reshape(*d.shape[:-1], 21, 2).sum(-1).mean(-1))


def quality_ok(alt, i, k):
    return alt['n_hidden'][i, k] <= 2 and alt['min_edge'][i, k] >= 0.6 and alt['explained_red'][i, k] >= 0.85


def distinct(xy_a, xy_b, tol=4.0):
    return np.linalg.norm(xy_a - xy_b, axis=-1).max() > tol


def resolve(alt, weight=1.0, verbose=True):
    letters = alt['letter']
    n, m = alt['score'].shape
    valid = np.isfinite(alt['score'])
    shapes = shape(np.nan_to_num(alt['xy']))
    chosen = np.where(valid[:, 0], 0, -1)

    for it in range(ITERATIONS):
        changed = 0
        totals_all = np.full((n, m), -np.inf)
        dist_all = np.full((n, m), np.inf)
        sigma_of = {}
        for letter in np.unique(letters):
            idx = np.nonzero((letters == letter) & (chosen >= 0))[0]
            idx = np.array([i for i in idx if quality_ok(alt, i, chosen[i])], int)
            if len(idx) < K_NEIGHBOURS + 2:
                continue
            ref = shapes[idx, chosen[idx]]
            d = rms(ref[:, None], ref[None])
            np.fill_diagonal(d, np.inf)
            sigma = float(np.median(np.sort(d, 1)[:, 0])) + 1e-3
            sigma_of[letter] = sigma
            for i in np.nonzero(letters == letter)[0]:
                keep = idx != i
                for k in range(m):
                    if not valid[i, k]:
                        continue
                    dk = np.sort(rms(shapes[i, k][None], ref[keep]))[:K_NEIGHBOURS].mean()
                    dist_all[i, k] = dk / sigma
                    totals_all[i, k] = alt['score'][i, k] - weight * 0.5 * (dk / sigma) ** 2
        new = np.where(np.isfinite(totals_all).any(1), np.argmax(totals_all, 1), chosen)
        changed = int((new != chosen).sum())
        chosen = new
        if verbose:
            print(f'  iteracja {it + 1}: zmienione wybory {changed}')
        if changed == 0:
            break

    status = np.array(['odrzucone'] * n, dtype=object)
    reason = np.array([''] * n, dtype=object)
    margin = np.full(n, np.nan)
    sdist = np.full(n, np.nan)
    for i in range(n):
        if letters[i] not in sigma_of:
            reason[i] = 'za_malo_obrazow_litery'
            continue
        if chosen[i] < 0:
            reason[i] = 'brak_rozwiazania'
            continue
        b = chosen[i]
        totals = totals_all[i]
        rival = max((totals[k] for k in range(m) if k != b and np.isfinite(totals[k])
                     and distinct(alt['xy'][i, k], alt['xy'][i, b])), default=-np.inf)
        margin[i] = totals[b] - rival
        sdist[i] = dist_all[i, b]
        if not quality_ok(alt, i, b):
            reason[i] = 'jakosc_rysunku'
        elif margin[i] < MIN_FINAL_MARGIN:
            reason[i] = 'niejednoznaczne'
        elif sdist[i] > MAX_SHAPE_SIGMAS:
            reason[i] = 'nietypowy_ksztalt_dla_litery'
        else:
            status[i] = 'ok'
    return chosen, status, reason, margin, sdist


def templates(alt, chosen, ok=None):
    """Szablon kształtu każdej litery: mediana i rozrzut (MAD) położenia każdego stawu w wybranych wariantach."""
    letters = alt['letter']
    shapes = shape(np.nan_to_num(alt['xy']))
    out = {}
    for letter in np.unique(letters):
        idx = np.nonzero((letters == letter) & (chosen >= 0))[0]
        if ok is not None:
            idx = idx[ok[idx]]
        if len(idx) < 5:
            continue
        pts = shapes[idx, chosen[idx]].reshape(-1, 21, 2)
        mean = np.median(pts, 0)
        dev = np.hypot(*(pts - mean).transpose(2, 0, 1))
        sd = np.maximum(1.4826 * np.median(dev, 0), 0.08)
        out[str(letter)] = {'mean': mean, 'sd': sd}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--alternatives', required=True)
    parser.add_argument('--out', default='landmarks_resolved.csv')
    parser.add_argument('--weight', type=float, default=1.0, help='waga modelu kształtu litery')
    args = parser.parse_args()

    alt = dict(np.load(args.alternatives, allow_pickle=False))
    letters, files = alt['letter'], alt['filename']
    n = len(letters)
    chosen, status, reason, margin, sdist = resolve(alt, args.weight)

    fields = (['letter', 'file', 'status', 'reason', 'final_margin', 'shape_dist_sigma', 'n_hidden']
              + [f'{c}{j}' for j in range(21) for c in ('x', 'y')] + [f'hidden{j}' for j in range(21)])
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(n):
            ok = status[i] == 'ok'
            k = chosen[i]
            row = {'letter': letters[i], 'file': files[i], 'status': status[i], 'reason': reason[i],
                   'final_margin': round(float(margin[i]), 3), 'shape_dist_sigma': round(float(sdist[i]), 3),
                   'n_hidden': int(alt['n_hidden'][i, k]) if ok else ''}
            for j in range(21):
                row[f'x{j}'] = round(float(alt['xy'][i, k, j, 0]), 3) if ok else ''
                row[f'y{j}'] = round(float(alt['xy'][i, k, j, 1]), 3) if ok else ''
                row[f'hidden{j}'] = int(alt['hidden'][i, k, j]) if ok else ''
            writer.writerow(row)
    ok = status == 'ok'
    print(f'Przyjęte: {int(ok.sum())}/{n} ({100 * ok.mean():.1f}%) → {args.out}')
    for letter in np.unique(letters):
        sel = letters == letter
        print(f'  {letter}: {int(ok[sel].sum())}/{int(sel.sum())}')
    vals, counts = np.unique(reason[~ok], return_counts=True)
    print('Powody odrzucenia:', dict(zip(vals, counts.tolist())))


if __name__ == '__main__':
    main()
