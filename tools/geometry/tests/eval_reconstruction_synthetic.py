"""
Pomiar jakości rekonstrukcji na syntetycznych szkieletach ze znaną prawdą.

  python tests/eval_reconstruction_synthetic.py --n 500

Liczy: odsetek przyjętych próbek, odsetek BŁĘDNIE przyjętych (choć jeden staw dalej niż 4 px od prawdy,
a dla stawu ukrytego 7 px), średni i maksymalny błąd położenia, powody odrzucenia.
Kryterium sukcesu: błędnie przyjętych < 1 % przyjętych.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from reconstruct_skeleton import reconstruct  # noqa: E402
from synthetic_hands import sample  # noqa: E402


def evaluate(n, seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    accepted, wrong, errors, reasons = 0, 0, [], {}
    started = time.time()
    for _ in range(n):
        image, truth = sample(rng)
        r = reconstruct(image)
        if r['status'] != 'ok':
            reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
            continue
        accepted += 1
        err = np.hypot(*(r['xy'] - truth).T)
        limit = np.where(r['hidden'] == 1, 7.0, 4.0)
        if (err > limit).any():
            wrong += 1
        errors.append(err)
    errors = np.concatenate(errors) if errors else np.zeros(1)
    result = {
        'n': n, 'accepted': accepted, 'wrong': wrong,
        'accept_rate': accepted / n, 'wrong_rate': wrong / max(accepted, 1),
        'mean_err_px': float(errors.mean()), 'p99_err_px': float(np.percentile(errors, 99)),
        'reasons': reasons, 'sec_per_image': (time.time() - started) / n,
    }
    if verbose:
        print(f"próbek {n}: przyjęte {accepted} ({100 * result['accept_rate']:.1f}%), "
              f"błędnie przyjęte {wrong} ({100 * result['wrong_rate']:.2f}% przyjętych)")
        print(f"błąd położenia: średnio {result['mean_err_px']:.2f} px, 99. percentyl {result['p99_err_px']:.2f} px")
        print(f"odrzucenia: {reasons}")
        print(f"czas: {result['sec_per_image'] * 1000:.0f} ms / obraz")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    res = evaluate(args.n, args.seed)
    sys.exit(0 if res['wrong_rate'] < 0.01 else 1)
