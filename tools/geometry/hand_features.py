"""
Ekstraktor cech geometrycznych dłoni – SPECYFIKACJA REFERENCYJNA (wersja FEATURE_VERSION).

Swift (`ASSN/Recognition/VisionFeatureExtractor.swift`) implementuje dokładnie te same wzory w tej samej kolejności;
zgodność sprawdza `tests/parity/` na wspólnych przykładach z `feature_fixtures.json`.

WEJŚCIE: 21 punktów w kolejności MediaPipe (= `HandJoint` w Swift), w IZOTROPOWYCH pikselach kamery
(ta sama jednostka na osi X i Y, oś Y w dół), w widoku LUSTRZANYM (selfie) – jak w datasecie.
  • dataset (płótno 400×400 z kadru 640×480, już lustrzany):  X = x_płótna · 640/400,  Y = y_płótna · 480/400
  • aplikacja (Vision, kadr nieodbity, punkty 0…1 z początkiem w lewym górnym rogu):  X = (1 − x) · W,  Y = y · H
Cechy nie zależą od położenia dłoni w kadrze ani od jej wielkości (przesunięcie do nadgarstka, skala = śródręcze).

OPCJA chirality = 'canonical': jeśli dłoń jest „odwrócona” (znak iloczynu wektorowego nadgarstek→MCP wskazującego
× nadgarstek→MCP małego < 0), oś X jest odbijana – lewa i prawa dłoń dają te same cechy. Kosztem jest utrata
rozróżnienia wnętrze/grzbiet dłoni. Wybór opcji rozstrzyga eksperyment (train_geometry_rf.py) i zapis w modelu.

GRUPY CECH (nazwy w FEATURE_NAMES, kolejność wiążąca):
  A  coords      x, y stawów 1…20 względem nadgarstka, w jednostkach śródręcza                       40
  B1 wrist_dist  odległość stawów 1…20 od nadgarstka                                                 20
  B2 tip_dist    odległości między czubkami palców (10 par)                                          10
  B3 tip_palm    odległość czubków od środka dłoni (średnia 0, 5, 9, 13, 17)                           5
  B4 extension   prostoliniowość palca: |czubek − podstawa| / suma długości paliczków                 5
  C1 bend        zgięcie w stawach (MCP/PIP/DIP; kciuk CMC/MCP/IP): kąt 0 = prosto, π = zawrócenie   15
  C2 spread      kąty między sąsiednimi palcami (paliczki bliższe) + kciuk–wskazujący                   4
  C3 thumb       kciuk: odległość czubka od MCP i PIP wskazującego i MCP środkowego, położenie czubka
                 wzdłuż i w poprzek osi dłoni                                                          5
  D  orientation kierunek osi dłoni (cos, sin), kierunek linii MCP (cos, sin), iloczyn wektorowy
                 (strona dłoni), proporcja szerokość/długość śródręcza                                  6
  basic    = A + B1 + oś dłoni (2) + iloczyn wektorowy (1)                                            63
  extended = wszystkie                                                                               110
"""

import json
import math

import numpy as np

FEATURE_VERSION = 'geom-v1'
CAMERA_SCALE_X = 640 / 400
CAMERA_SCALE_Y = 480 / 400

FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'little']
TIPS = [4, 8, 12, 16, 20]
TIP_PAIRS = [(a, b) for i, a in enumerate(TIPS) for b in TIPS[i + 1:]]


def _names():
    names = []
    names += [f'A_{c}{j}' for j in range(1, 21) for c in ('x', 'y')]
    names += [f'B1_wrist_d{j}' for j in range(1, 21)]
    names += [f'B2_tip_d{a}_{b}' for a, b in TIP_PAIRS]
    names += [f'B3_tip_palm_{f}' for f in FINGER_NAMES]
    names += [f'B4_extension_{f}' for f in FINGER_NAMES]
    names += [f'C1_bend_{f}_{k}' for f in FINGER_NAMES for k in (1, 2, 3)]
    names += ['C2_spread_index_middle', 'C2_spread_middle_ring', 'C2_spread_ring_little', 'C2_spread_thumb_index']
    names += ['C3_thumb_index_mcp', 'C3_thumb_index_pip', 'C3_thumb_middle_mcp', 'C3_thumb_along', 'C3_thumb_across']
    names += ['D_palm_cos', 'D_palm_sin', 'D_mcp_line_cos', 'D_mcp_line_sin', 'D_cross', 'D_palm_aspect']
    return names


FEATURE_NAMES = _names()
BASIC = (FEATURE_NAMES[:60] + ['D_palm_cos', 'D_palm_sin', 'D_cross'])
FEATURE_SETS = {'basic': BASIC, 'extended': FEATURE_NAMES}


def dataset_to_camera(xy_canvas):
    """Piksele płótna datasetu (400×400, widok lustrzany) → izotropowe piksele kamery 640×480 (wciąż lustrzane)."""
    p = np.asarray(xy_canvas, float).copy()
    p[..., 0] *= CAMERA_SCALE_X
    p[..., 1] *= CAMERA_SCALE_Y
    return p


def _bend(u, v):
    """Kąt między kolejnymi odcinkami: 0 = prosto, π = zawrócenie (atan2(|u×v|, u·v))."""
    cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    dot = (u * v).sum(-1)
    return np.arctan2(np.abs(cross), dot)


def _norm(v):
    return np.sqrt((v * v).sum(-1))


def extract(points, chirality='none'):
    """points: (21, 2) albo (N, 21, 2) w izotropowych pikselach (widok lustrzany). Zwraca (110,) / (N, 110)."""
    p = np.asarray(points, float)
    single = p.ndim == 2
    if single:
        p = p[None]
    p = p.copy()
    rel = p - p[:, :1]
    if chirality == 'canonical':
        cross0 = rel[:, 5, 0] * rel[:, 17, 1] - rel[:, 5, 1] * rel[:, 17, 0]
        flip = cross0 < 0
        rel[flip, :, 0] *= -1
    elif chirality != 'none':
        raise ValueError(chirality)
    scale = (_norm(rel[:, 5]) + _norm(rel[:, 9]) + _norm(rel[:, 13]) + _norm(rel[:, 17])) / 4
    q = rel / np.maximum(scale, 1e-9)[:, None, None]
    n = len(q)
    out = []

    out.append(q[:, 1:].reshape(n, 40))                                               # A
    out.append(_norm(q[:, 1:]))                                                       # B1
    out.append(np.stack([_norm(q[:, a] - q[:, b]) for a, b in TIP_PAIRS], 1))         # B2
    center = q[:, [0, 5, 9, 13, 17]].mean(1)
    out.append(np.stack([_norm(q[:, t] - center) for t in TIPS], 1))                  # B3
    ext = []
    for chain in FINGERS:
        base = chain[1]
        segs = sum(_norm(q[:, chain[k + 1]] - q[:, chain[k]]) for k in range(1, 4))
        ext.append(_norm(q[:, chain[4]] - q[:, base]) / np.maximum(segs, 1e-9))
    out.append(np.stack(ext, 1))                                                      # B4
    bends = []
    for chain in FINGERS:
        for k in (1, 2, 3):
            bends.append(_bend(q[:, chain[k]] - q[:, chain[k - 1]], q[:, chain[k + 1]] - q[:, chain[k]]))
    out.append(np.stack(bends, 1))                                                    # C1
    prox = {f: q[:, FINGERS[f][2]] - q[:, FINGERS[f][1]] for f in range(1, 5)}
    spread = [_bend(prox[1], prox[2]), _bend(prox[2], prox[3]), _bend(prox[3], prox[4]),
              _bend(q[:, 4] - q[:, 1], q[:, 8] - q[:, 5])]
    out.append(np.stack(spread, 1))                                                   # C2
    axis = q[:, 9]
    axis_len = np.maximum(_norm(axis), 1e-9)
    along = (q[:, 4] * axis).sum(-1) / axis_len
    across = (axis[:, 0] * q[:, 4, 1] - axis[:, 1] * q[:, 4, 0]) / axis_len
    out.append(np.stack([_norm(q[:, 4] - q[:, 5]), _norm(q[:, 4] - q[:, 6]), _norm(q[:, 4] - q[:, 9]),
                         along, across], 1))                                          # C3
    line = q[:, 17] - q[:, 5]
    line_len = np.maximum(_norm(line), 1e-9)
    cross = (q[:, 5, 0] * q[:, 17, 1] - q[:, 5, 1] * q[:, 17, 0]) / np.maximum(_norm(q[:, 5]) * _norm(q[:, 17]), 1e-9)
    out.append(np.stack([axis[:, 0] / axis_len, axis[:, 1] / axis_len, line[:, 0] / line_len,
                         line[:, 1] / line_len, cross, line_len / axis_len], 1))      # D
    features = np.concatenate(out, 1)
    assert features.shape[1] == len(FEATURE_NAMES)
    return features[0] if single else features


def select(features, feature_set):
    idx = [FEATURE_NAMES.index(name) for name in FEATURE_SETS[feature_set]]
    return features[..., idx]


def spec(feature_set, chirality):
    """Opis zapisywany obok modelu – aplikacja sprawdza wersję i liczbę cech."""
    return {'version': FEATURE_VERSION, 'feature_set': feature_set, 'chirality': chirality,
            'n_features': len(FEATURE_SETS[feature_set]), 'names': FEATURE_SETS[feature_set],
            'indices': [FEATURE_NAMES.index(n) for n in FEATURE_SETS[feature_set]],
            'input': 'isotropic camera pixels, mirrored (selfie) view, y down, MediaPipe joint order',
            'scale': 'mean distance wrist -> MCP 5, 9, 13, 17'}


def save_spec(path, feature_set, chirality):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(spec(feature_set, chirality), f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    print(FEATURE_VERSION, len(FEATURE_NAMES), 'cech;', 'basic:', len(BASIC))
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 400, (21, 2))
    f = extract(pts)
    assert np.allclose(extract(pts + [37, -12]), f), 'przesunięcie'
    assert np.allclose(extract(pts * 2.5), f), 'skala'
    mirrored = pts.copy()
    mirrored[:, 0] = 400 - mirrored[:, 0]
    assert np.allclose(extract(mirrored, 'canonical'), extract(pts, 'canonical')), 'odbicie (canonical)'
    print('testy niezmienniczości: OK')
