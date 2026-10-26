"""
Zgodność Apple Vision ↔ MediaPipe na TYCH SAMYCH zdjęciach.

  python compare_detectors.py --mediapipe landmarks_aslhg.csv --vision landmarks_vision.csv \\
         [--model wyniki_rf/model_rf.joblib] --out porownanie_detektorow.md

Liczy dla par (to samo zdjęcie): błąd położenia każdego stawu po normalizacji (nadgarstek = 0, skala = śródręcze),
różnice cech, a z --model: trafność Random Forest na punktach MediaPipe vs Apple Vision i zgodność ich predykcji.
To jest właściwy test, czy model wytrenowany na punktach MediaPipe nadaje się do punktów z aplikacji.
"""

import argparse
import csv

import numpy as np

from hand_features import FEATURE_NAMES, extract, select
from hand_topology import JOINT_NAMES


def load(path):
    out = {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if r['status'] != 'ok':
            continue
        key = r['file'].replace('\\', '/').split('/')[-2:]
        out['/'.join(key)] = (r['letter'], np.array([[float(r[f'x{j}']), float(r[f'y{j}'])] for j in range(21)]))
    return out


def normalize(p):
    rel = p - p[:1]
    return rel / np.mean(np.linalg.norm(rel[[5, 9, 13, 17]], axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mediapipe', required=True)
    parser.add_argument('--vision', required=True)
    parser.add_argument('--model')
    parser.add_argument('--out', default='porownanie_detektorow.md')
    args = parser.parse_args()

    mp_, vi = load(args.mediapipe), load(args.vision)
    keys = sorted(set(mp_) & set(vi))
    print(f'Wspólne zdjęcia z pełną dłonią w obu detektorach: {len(keys)} '
          f'(MediaPipe {len(mp_)}, Vision {len(vi)})')
    letters = np.array([mp_[k][0] for k in keys])
    a = np.stack([mp_[k][1] for k in keys])
    b = np.stack([vi[k][1] for k in keys])
    err = np.linalg.norm(np.stack([normalize(p) for p in a]) - np.stack([normalize(p) for p in b]), axis=-1)
    lines = ['# Apple Vision ↔ MediaPipe – te same zdjęcia', '', f'Par zdjęć: {len(keys)}', '',
             '## Różnica położenia stawów (jednostki śródręcza, po normalizacji)', '',
             '| staw | mediana | 90. percentyl |', '|---|---|---|']
    lines += [f'| {j} {JOINT_NAMES[j]} | {np.median(err[:, j]):.3f} | {np.percentile(err[:, j], 90):.3f} |'
              for j in range(21)]
    if args.model:
        import joblib
        bundle = joblib.load(args.model)
        fs, ch, model = bundle['feature_set'], bundle['chirality'], bundle['model']
        xa, xb = select(extract(a, ch), fs), select(extract(b, ch), fs)
        pa, pb = model.predict(xa), model.predict(xb)
        lines += ['', '## Random Forest na punktach z obu detektorów', '', '| wejście | trafność |', '|---|---|',
                  f'| MediaPipe | {np.mean(pa == letters):.4f} |', f'| Apple Vision | {np.mean(pb == letters):.4f} |',
                  '', f'Zgodność predykcji MediaPipe ↔ Vision: {np.mean(pa == pb):.4f}', '',
                  '| litera | trafność MediaPipe | trafność Vision | zdjęć |', '|---|---|---|---|']
        for L in sorted(set(letters)):
            s = letters == L
            lines.append(f'| {L} | {np.mean(pa[s] == L):.3f} | {np.mean(pb[s] == L):.3f} | {int(s.sum())} |')
        d = np.abs(xa - xb).mean(0)
        names = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES))] if fs == 'extended' else None
        if names:
            top = np.argsort(d)[::-1][:10]
            lines += ['', '## Cechy najbardziej różne między detektorami (średnia |różnica|)', '',
                      '| cecha | różnica |', '|---|---|'] + [f'| {names[i]} | {d[i]:.4f} |' for i in top]
    open(args.out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:8]))
    print(f'→ {args.out}')


if __name__ == '__main__':
    main()
