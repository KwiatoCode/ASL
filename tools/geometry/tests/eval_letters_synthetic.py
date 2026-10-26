"""
Walidacja CAŁEJ rekonstrukcji (etap 1 + etap 2) na syntetycznych „literach” ze znaną prawdą.

Każda syntetyczna litera ma stałą pozę i ujęcie, próbki różnią się nieznacznie – jak powtórzenia jednego znaku.
Mierzymy, ile obrazów jest przyjętych i ile z przyjętych ma błąd > 10 % długości śródręcza (błędna numeracja).

  python tests/eval_letters_synthetic.py --letters 26 --per-letter 30 --jobs 4
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline_landmarks import full_pipeline  # noqa: E402
from synthetic_hands import letter_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--letters', type=int, default=26)
    parser.add_argument('--per-letter', type=int, default=30)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    data = letter_dataset(args.letters, args.per_letter, args.seed)
    truth = np.stack([d[2] for d in data])
    alt, chosen, status, reason, margin, sdist = full_pipeline([d[1] for d in data], [d[0] for d in data],
                                                               args.jobs)
    ok = status == 'ok'
    palm = np.mean(np.linalg.norm(truth[:, [5, 9, 13, 17]] - truth[:, :1], axis=-1), axis=1)
    err = np.full(len(data), np.inf)
    for i in np.nonzero(ok)[0]:
        err[i] = np.linalg.norm(alt['xy'][i, chosen[i]] - truth[i], axis=-1).max() / palm[i]
    wrong = ok & (err > 0.1)
    print(f'obrazów {len(data)}: przyjęte {ok.sum()} ({100 * ok.mean():.1f}%), '
          f'błędnie przyjęte {wrong.sum()} ({100 * wrong.sum() / max(ok.sum(), 1):.2f}% przyjętych)')
    covered = sum(ok[alt['letter'] == L].sum() >= 5 for L in np.unique(alt['letter']))
    print(f'liter z ≥ 5 przyjętymi obrazami: {covered}/{args.letters}')
    vals, counts = np.unique(reason[~ok], return_counts=True)
    print('odrzucenia:', dict(zip(vals, counts.tolist())))
    # tylko etap 1 (bez modelu litery) – dla porównania
    e1 = np.array([np.linalg.norm(alt['xy'][i, 0] - truth[i], axis=-1).max() / palm[i]
                   if np.isfinite(alt['score'][i, 0]) else np.inf for i in range(len(data))])
    print(f'sam etap 1 (najlepszy wariant, bez progów): poprawne {np.mean(e1 <= 0.1) * 100:.1f}%')
    return 0 if wrong.sum() <= 0.02 * max(ok.sum(), 1) else 1


if __name__ == '__main__':
    sys.exit(main())
