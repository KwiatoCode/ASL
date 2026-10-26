"""
Wspólne przykłady do testu zgodności cech Python ↔ Swift.

  python tests/make_feature_fixtures.py         # zapisuje tests/parity/feature_fixtures.json

Przykłady: syntetyczne dłonie (różne pozy, obie strony, różne skale i położenia) + przypadki brzegowe
(dłoń odbita, punkty prawie współliniowe). Wartości referencyjne liczy hand_features.extract.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hand_features import FEATURE_VERSION, extract  # noqa: E402
from synthetic_hands import build_hand, project, sample_params, CAMERA_W, CAMERA_H  # noqa: E402


def main():
    rng = np.random.default_rng(2025)
    cases = []
    while len(cases) < 40:
        normalized = project(build_hand(sample_params(rng)), rng)
        if normalized is None:
            continue
        pts = normalized * [CAMERA_W, CAMERA_H]
        chirality = 'canonical' if len(cases) % 2 else 'none'
        cases.append({'points': pts.round(6).tolist(), 'chirality': chirality})
    mirrored = np.array(cases[0]['points'])
    mirrored[:, 0] = CAMERA_W - mirrored[:, 0]
    cases.append({'points': mirrored.round(6).tolist(), 'chirality': 'canonical'})
    line = np.stack([np.linspace(100, 300, 21), np.linspace(200, 202, 21)], 1)
    line[[5, 9, 13, 17]] += [[0, 5], [0, -5], [0, 8], [0, -8]]
    cases.append({'points': line.round(6).tolist(), 'chirality': 'none'})
    for c in cases:
        c['features'] = extract(np.array(c['points']), c['chirality']).tolist()
    out = os.path.join(os.path.dirname(__file__), 'parity', 'feature_fixtures.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'version': FEATURE_VERSION, 'tolerance': 1e-9, 'cases': cases}, f)
    print(f'{len(cases)} przykładów → {out}')


if __name__ == '__main__':
    main()
